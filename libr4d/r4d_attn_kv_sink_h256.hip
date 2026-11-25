/* r4d_attn_kv_sink_h256.hip -- attn_kv_sink at head_dim 256: Hy4-flash's non-absorbed prefill.
 *
 * Dense causal attention over separate key and value rows (kv_b's expansion of the latent, split
 * by mla_kv_split), a sliding window or none, and a learned sink logit a head (docs/OPS.md,
 * libref's ref_attn_kv_sink):
 *
 *   sequence s: m_s = cu_q[s+1] - cu_q[s] query rows, n_s = cu_k[s+1] - cu_k[s] >= m_s key rows
 *   query i sits at key index p = n_s - m_s + i and sees j <= p, and p - j < window (window 0: all)
 *   s_j = scale * q . k_j (f32);  out = sum_j exp(s_j) v_j / (sum_j exp(s_j) + exp(sink[h]))
 *
 * The kernel is r4d_attn_paged_window.hip's WMMA prefill kernel at gqa 1 with the paged cache
 * replaced by two flat [N, heads, 256] planes (see that kernel's header for the fragment layouts):
 *
 *   - Workgroup: one head, 128 consecutive query rows of one sequence, 16 waves (512 threads).
 *     Wave (row group rg, slice sl) owns 16 rows by the head's dims [128 sl, 128 sl + 128).
 *   - Grid: x = 128-row tiles numbered sequence by sequence (found from cu_q by a block scan), y =
 *     heads. Tiles past the batch's last return at once.
 *   - Keys in tiles of 16 starting at the workgroup's lowest visible key, max(0, p_first - window +
 *     1), up to its last row's own key; nothing outside that range is read. The next tile is
 *     fetched into registers while the current one is multiplied.
 *   - S^T = K Q^T in bf16 WMMA with f32 sums; the two slices meet in LDS and add in slice order,
 *     so both see the same scores. A tile wholly visible to every row of the workgroup skips the
 *     per-element mask.
 *   - Online softmax in the log2 domain, row max over the keys alone. O^T += V^T P^T in f16 WMMA:
 *     P is in [0, 1] relative to the running max (11 bits), V widened from bf16 (exact within f16's
 *     range; a |v| past 65504 saturates, below 2^-14 it keeps f16's subnormal steps).
 *   - The sink enters at the store: den = sum P + exp2(sink' - max). Seeding the running max with
 *     the sink, as the paged kernel does, would leave every P tiny when the sink dominates and push
 *     it into f16's subnormals; here the largest key weight is always 1. A sink past 2^128 times the
 *     largest key weight makes the denominator infinite and the row zero, which is its limit.
 *
 * Rows of a sequence whose cu ranges are malformed (n_s < m_s, or past k's rows) cannot be refused
 * from the device; they are written as zeros. A query always sees its own key, so no valid row is
 * empty.
 *
 * Nothing here allocates or synchronises. Wave32 (RDNA4, gfx12 WMMA).
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "rad_plugin.h"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "this attention is written for wave32 (RDNA): build without -mwavefrontsize64"
#endif

namespace r4d_akvs {

constexpr int HD = 256;
constexpr int kThreads = 512;
constexpr int kWaves = kThreads / 32;
constexpr int kDQ = 128;                 /* dims of a wave's slice of the head */
constexpr int NS = HD / kDQ;             /* slices sharing one 16-row group */
constexpr int RG = kWaves / NS;          /* row groups a workgroup */
constexpr int BQ = RG * 16;              /* query rows a workgroup: 128 */
constexpr int KS = HD * 2 + 16;          /* bytes a key row of the K tile, 16 of pad */
constexpr int VS = 48;                   /* bytes a dim row of the V^T tile: 16 keys + pad */
constexpr int CW = 8;                    /* bf16 a 16-byte load carries */
constexpr int NCH = HD / CW;             /* such chunks a row */
constexpr int NKI = 16 * NCH;            /* K items a tile: (key, chunk) */
constexpr int NVI = 8 * NCH;             /* V items a tile: (key pair, chunk) */
static_assert(NKI == kThreads && NVI <= kThreads, "one K load and at most one V pair a thread");

typedef short v8s __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));

struct Args {
    const uint16_t* q;   long long q_rs, q_hs;     /* elements */
    const uint16_t* k;   long long k_rs, k_hs;
    const uint16_t* v;   long long v_rs, v_hs;
    uint16_t*       out; long long o_rs, o_hs;
    const int*      cuq;
    const int*      cuk;
    const float*    sinks;                         /* [heads] */
    float qscale;                                  /* softmax scale * log2(e) */
    int   n_seq, n_keys, window;
};

__device__ __forceinline__ float shfl_xor(float v, int m) {
    const int l = (int)(threadIdx.x & 31u);
    return __int_as_float(__builtin_amdgcn_ds_bpermute((l ^ m) << 2, __float_as_int(v)));
}
__device__ __forceinline__ int shfl_up(int v, int d) {
    const int l = (int)(threadIdx.x & 31u);
    return __builtin_amdgcn_ds_bpermute((l >= d ? l - d : l) << 2, v);
}
__device__ __forceinline__ float ex2(float x) {
#if __has_builtin(__builtin_amdgcn_exp2f)
    return __builtin_amdgcn_exp2f(x);   /* v_exp_f32: exp2(-inf) = 0 */
#else
    return exp2f(x);
#endif
}
__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }

__device__ __forceinline__ unsigned bf16_rtne(float f) {
    unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}
__device__ __forceinline__ unsigned pack2(float a, float b) { return bf16_rtne(a) | (bf16_rtne(b) << 16); }
/* f32 -> f16 bits, round to nearest even; a finite value past f16's range saturates, NaN stays */
__device__ __forceinline__ unsigned f16_bits(float f) {
    if (f == f) f = __builtin_fminf(__builtin_fmaxf(f, -65504.0f), 65504.0f);
    return (unsigned)__builtin_bit_cast(unsigned short, (_Float16)f);
}
__device__ __forceinline__ void bf16x8_f16(const uint4 w, unsigned (&h)[8]) {
    const unsigned in[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        h[2 * q] = f16_bits(__uint_as_float(in[q] << 16));
        h[2 * q + 1] = f16_bits(__uint_as_float(in[q] & 0xffff0000u));
    }
}
__device__ __forceinline__ v8f wmma_bf16(v8s a, v8s b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ v8f wmma_f16(v8h a, v8h b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

/* Tile `tile`'s sequence and its BQ-row slice: tiles are numbered sequence by sequence,
 * ceil(m_s / BQ) each, found by a block-wide prefix sum over cu_q (r4d_attn_paged_window.hip's
 * find_tile). All threads get the same answer from LDS. */
__device__ void find_tile(const int* cu, int n_seq, int tile, int* s_scan, int* s_res, int& s_out,
                          int& lt_out) {
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    if (tid == 0) { s_res[0] = -1; s_res[1] = 0; }
    __syncthreads();
    int base = 0;
    for (int c0 = 0; c0 < n_seq; c0 += kThreads) {
        const int s = c0 + tid;
        int nt = 0;
        if (s < n_seq) {
            const int ql = cu[s + 1] - cu[s];
            nt = ql > 0 ? (ql + BQ - 1) / BQ : 0;
        }
        int x = nt;
        for (int d = 1; d < 32; d <<= 1) {
            const int y = shfl_up(x, d);
            if (lane >= d) x += y;
        }
        if (lane == 31) s_scan[wave] = x;
        __syncthreads();
        int before = 0, total = 0;
        for (int w = 0; w < kWaves; ++w) {
            const int v = s_scan[w];
            before += (w < wave) ? v : 0;
            total += v;
        }
        const int incl = base + before + x;
        if (nt > 0 && tile >= incl - nt && tile < incl) { s_res[0] = s; s_res[1] = tile - (incl - nt); }
        base += total;
        __syncthreads();
        if (s_res[0] >= 0 || base > tile) break;
    }
    s_out = s_res[0];
    lt_out = s_res[1];
}

__global__ __launch_bounds__(kThreads) void attn_kv_sink_kernel(Args A) {
    __shared__ uint4 s_k[16 * KS / 16];
    __shared__ uint4 s_vt[HD * VS / 16];
    __shared__ float4 s_x[kWaves * 32 * 2];
    __shared__ int s_scan[kWaves];
    __shared__ int s_res[2];

    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int rg = wave / NS, sl = wave - rg * NS;
    const int tile = (int)blockIdx.x, head = (int)blockIdx.y;

    int s = -1, lt = 0;
    find_tile(A.cuq, A.n_seq, tile, s_scan, s_res, s, lt);
    if (s < 0) return;                               /* uniform across the block */
    const int qb = A.cuq[s], m = A.cuq[s + 1] - qb;
    const int kb = A.cuk[s], n = A.cuk[s + 1] - kb;
    const int pfirst = lt * BQ;
    const int npos = imax(0, imin(BQ, m - pfirst));
    if (npos <= 0) return;
    const bool bad = n < m || kb < 0 || A.cuk[s + 1] > A.n_keys;

    const int tok0 = qb + pfirst;
    const int qbase = n - m + pfirst;                /* key index of the tile's first row */
    const int plast = qbase + npos - 1;
    const int window = A.window;
    const int hi = bad ? 0 : plast + 1;              /* keys [lo, hi) */
    const int lo = window > 0 ? imax(0, qbase - window + 1) : 0;
    const int nkt = hi > lo ? (hi - lo + 15) / 16 : 0;

    /* ---- this lane's query row: column nn of every S^T and O^T tile */
    const int nn = lane & 15, h = lane >> 4;
    const int p = rg * 16 + nn;
    const bool rval = p < npos;
    const int pos = qbase + p;

    v8s qf[kDQ / 16];
    {
        const uint16_t* qp = A.q + (long long)(tok0 + (rval ? p : 0)) * A.q_rs +
                             (long long)head * A.q_hs + sl * kDQ + 8 * h;
#pragma unroll
        for (int kc = 0; kc < kDQ / 16; ++kc) {
            const uint4 w = rval ? *(const uint4*)(qp + kc * 16) : make_uint4(0u, 0u, 0u, 0u);
            qf[kc] = __builtin_bit_cast(v8s, w);
        }
    }
    v8f oacc[kDQ / 16];
#pragma unroll
    for (int mc = 0; mc < kDQ / 16; ++mc)
#pragma unroll
        for (int e = 0; e < 8; ++e) oacc[mc][e] = 0.0f;
    float mrun = -__builtin_inff(), l = 0.0f;

    /* ---- the key tiles, prefetched one ahead through registers. A key past hi is a zero row. */
    uint4 pk, pv0, pv1;
    const uint16_t* kbase = A.k + (long long)kb * A.k_rs + (long long)head * A.k_hs;
    const uint16_t* vbase = A.v + (long long)kb * A.v_rs + (long long)head * A.v_hs;
    auto fetch = [&](int kt) __attribute__((always_inline)) {
        const int k0 = lo + kt * 16;
        {
            const int key = tid / NCH, ch = tid - key * NCH;
            pk = make_uint4(0u, 0u, 0u, 0u);
            if (k0 + key < hi) pk = *(const uint4*)(kbase + (long long)(k0 + key) * A.k_rs + ch * CW);
        }
        pv0 = pv1 = make_uint4(0u, 0u, 0u, 0u);
        if (tid < NVI) {
            const int kp = tid & 7, ch = tid >> 3;
            const int ka = k0 + 2 * kp;
            if (ka < hi) pv0 = *(const uint4*)(vbase + (long long)ka * A.v_rs + ch * CW);
            if (ka + 1 < hi) pv1 = *(const uint4*)(vbase + (long long)(ka + 1) * A.v_rs + ch * CW);
        }
    };

    unsigned char* kl = (unsigned char*)s_k;
    unsigned char* vl = (unsigned char*)s_vt;
    if (nkt > 0) fetch(0);
    for (int kt = 0; kt < nkt; ++kt) {
        const int k0 = lo + kt * 16;
        __syncthreads();                             /* the previous tile's LDS is consumed */
        {
            const int key = tid / NCH, ch = tid - key * NCH;
            *(uint4*)(kl + key * KS + ch * 16) = pk;
        }
        /* V: [dim][key] as f16, two keys a dword; the row order rotated by 2 per 8-lane group, so
         * the four groups of a wave write four different rows mod 8 -- 32 banks */
        if (tid < NVI) {
            const int kp = tid & 7, ch = tid >> 3;
            unsigned ha[8], hb[8], d[CW];
            bf16x8_f16(pv0, ha);
            bf16x8_f16(pv1, hb);
#pragma unroll
            for (int e = 0; e < CW; ++e) d[e] = ha[e] | hb[e] << 16;
            const int rot = 2 * ((lane >> 3) & 3);
            unsigned* col = (unsigned*)(vl + (ch * CW) * VS) + kp;
#pragma unroll
            for (int t = 0; t < CW; ++t) {
                const unsigned v0 = d[t], v2 = d[(t + 2) % CW], v4 = d[(t + 4) % CW], v6 = d[(t + 6) % CW];
                const unsigned vv = rot == 0 ? v0 : rot == 2 ? v2 : rot == 4 ? v4 : v6;
                const int e = (t + rot) % CW;
                col[e * (VS / 4)] = vv;
            }
        }
        __syncthreads();
        if (kt + 1 < nkt) fetch(kt + 1);

        /* ---- S^T partial over this wave's slice: A = K rows (key nn), B = Q^T (row nn) */
        v8f sacc;
#pragma unroll
        for (int e = 0; e < 8; ++e) sacc[e] = 0.0f;
        {
            const unsigned char* ka = kl + nn * KS + (sl * kDQ + 8 * h) * 2;
#pragma unroll
            for (int kc = 0; kc < kDQ / 16; ++kc)
                sacc = wmma_bf16(__builtin_bit_cast(v8s, *(const uint4*)(ka + kc * 32)), qf[kc], sacc);
        }
        /* ---- the slices' partials meet: both sum them in slice order */
        s_x[(wave * 2 + 0) * 32 + lane] = make_float4(sacc[0], sacc[1], sacc[2], sacc[3]);
        s_x[(wave * 2 + 1) * 32 + lane] = make_float4(sacc[4], sacc[5], sacc[6], sacc[7]);
        __syncthreads();
        float tot[8];
#pragma unroll
        for (int j = 0; j < NS; ++j) {
            float vv[8];
            if (j == sl) {
#pragma unroll
                for (int e = 0; e < 8; ++e) vv[e] = sacc[e];
            } else {
                const float4 a = s_x[((rg * NS + j) * 2 + 0) * 32 + lane];
                const float4 b = s_x[((rg * NS + j) * 2 + 1) * 32 + lane];
                vv[0] = a.x; vv[1] = a.y; vv[2] = a.z; vv[3] = a.w;
                vv[4] = b.x; vv[5] = b.y; vv[6] = b.z; vv[7] = b.w;
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) tot[e] = j == 0 ? vv[e] : tot[e] + vv[e];
        }

        /* ---- online softmax, row nn: eight keys here, eight in lane ^ 16 */
        float sv[8], mx = -__builtin_inff();
        /* A tile every row of the workgroup sees whole: before the first row's own key, inside the
         * key range, and within the last row's window. */
        const bool interior = k0 + 16 <= hi && k0 + 15 <= qbase && (window <= 0 || plast - k0 < window);
        if (interior) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                sv[e] = rval ? tot[e] * A.qscale : -__builtin_inff();
                mx = __builtin_fmaxf(mx, sv[e]);
            }
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int kk = k0 + 8 * h + e;
                bool vis = rval && kk < hi && kk <= pos;
                if (window > 0) vis = vis && pos - kk < window;
                sv[e] = vis ? tot[e] * A.qscale : -__builtin_inff();
                mx = __builtin_fmaxf(mx, sv[e]);
            }
        }
        mx = __builtin_fmaxf(mx, shfl_xor(mx, 16));
        const float mn = __builtin_fmaxf(mrun, mx);
        const float corr = mn == -__builtin_inff() ? 1.0f : ex2(mrun - mn);
        /* P in f16, and the denominator summed from P as rounded: a row is a weighted mean of V */
        float ps = 0.0f;
        v8h pf;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float pe = sv[e] == -__builtin_inff() ? 0.0f : ex2(sv[e] - mn);
            pf[e] = (_Float16)pe;
            ps += (float)pf[e];
        }
        l = l * corr + ps;
        mrun = mn;

        /* ---- O^T += V^T * P^T in f16: A = V^T rows (dim nn of the chunk), keys 8h .. 8h + 7 */
        const unsigned char* va = vl + (sl * kDQ + nn) * VS + h * 16;
#pragma unroll
        for (int mc = 0; mc < kDQ / 16; ++mc) {
            oacc[mc] *= corr;
            oacc[mc] = wmma_f16(__builtin_bit_cast(v8h, *(const uint4*)(va + mc * 16 * VS)), pf, oacc[mc]);
        }
    }

    /* ---- out: the row's denominator is its two lanes' partial sums and the sink */
    const float lt2 = l + shfl_xor(l, 16);
    if (rval) {
        float inv = 0.0f;
        if (mrun != -__builtin_inff()) {
            const float den = lt2 + ex2(A.sinks[head] * 1.4426950408889634f - mrun);
            inv = den > 0.0f ? 1.0f / den : 0.0f;    /* den = inf: the sink swamps, zero */
        }
        uint16_t* op = A.out + (long long)(tok0 + p) * A.o_rs + (long long)head * A.o_hs + sl * kDQ + 8 * h;
#pragma unroll
        for (int mc = 0; mc < kDQ / 16; ++mc)
            *(uint4*)(op + mc * 16) =
                make_uint4(pack2(oacc[mc][0] * inv, oacc[mc][1] * inv), pack2(oacc[mc][2] * inv, oacc[mc][3] * inv),
                           pack2(oacc[mc][4] * inv, oacc[mc][5] * inv), pack2(oacc[mc][6] * inv, oacc[mc][7] * inv));
    }
}

/* [rows, heads, hd] with the last stride 1, or [rows, heads * hd] */
static bool head_view(const RadTensor* t, int64_t heads, int64_t hd, int64_t* rows, int64_t* rs,
                      int64_t* hs) {
    if (t->rank == 3) {
        if (t->shape[2] != hd || t->shape[1] != heads || t->stride[2] != 1) return false;
        *rows = t->shape[0]; *rs = t->stride[0]; *hs = t->stride[1];
        return true;
    }
    if (t->rank == 2) {
        if (t->shape[1] != heads * hd || t->stride[1] != 1) return false;
        *rows = t->shape[0]; *rs = t->stride[0]; *hs = hd;
        return true;
    }
    return false;
}

static int launch(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    const RadTensor* q = rad_arg_in(a, 0);
    const RadTensor* k = rad_arg_in(a, 1);
    const RadTensor* v = rad_arg_in(a, 2);
    const RadTensor* cq = rad_arg_in(a, 3);
    const RadTensor* ck = rad_arg_in(a, 4);
    const RadTensor* sk = rad_arg_in(a, 5);
    const RadTensor* o = rad_arg_in(a, 6);
    if (!q || !k || !v || !cq || !ck || !sk || !o || !o->data) return RAD_E_INVAL;
    const int64_t H = rad_args_geti_or(a, "heads", 0);
    const long long window = rad_args_geti_or(a, "window", 0);
    if (rad_args_geti_or(a, "head_dim", HD) != HD) return RAD_E_SHAPE;
    if (H < 1 || H > 65535 || window < 0 || window > (1ll << 30)) return RAD_E_SHAPE;
    const double scale = rad_args_getf_or(a, "scale", 1.0 / 16.0);
    if (!(scale > 0) || !(scale < 1e30)) return RAD_E_SHAPE;
    if (q->dtype != RAD_BF16 || k->dtype != RAD_BF16 || v->dtype != RAD_BF16 || o->dtype != RAD_BF16)
        return RAD_E_DTYPE;
    if (cq->dtype != RAD_I32 || ck->dtype != RAD_I32 || sk->dtype != RAD_F32) return RAD_E_DTYPE;

    int64_t rows = 0, q_rs = 0, q_hs = 0, nk = 0, k_rs = 0, k_hs = 0, nv = 0, v_rs = 0, v_hs = 0,
            orows = 0, o_rs = 0, o_hs = 0;
    if (!head_view(q, H, HD, &rows, &q_rs, &q_hs) || !head_view(k, H, HD, &nk, &k_rs, &k_hs) ||
        !head_view(v, H, HD, &nv, &v_rs, &v_hs) || !head_view(o, H, HD, &orows, &o_rs, &o_hs))
        return RAD_E_SHAPE;
    if (nv != nk || orows != rows || rows < 1 || nk < 1 || nk > (1ll << 30) || rows > (1ll << 30))
        return RAD_E_SHAPE;
    if (cq->rank != 1 || ck->rank != 1 || cq->shape[0] != ck->shape[0] || cq->shape[0] < 2)
        return RAD_E_SHAPE;
    if (rad_tensor_numel(sk) != H) return RAD_E_SHAPE;
    const int64_t n_seq = cq->shape[0] - 1;
    if (n_seq > (1 << 24)) return RAD_E_SHAPE;
    /* 16-byte loads and stores of every row and head */
    if (!rad_ptr_aligned(q->data, 16) || !rad_ptr_aligned(k->data, 16) || !rad_ptr_aligned(v->data, 16) ||
        !rad_ptr_aligned(o->data, 16) || (q_rs | q_hs | k_rs | k_hs | v_rs | v_hs | o_rs | o_hs) % 8)
        return RAD_E_ALIGN;

    /* an upper bound on the tiles of a ragged batch; the ones past the last exit at once */
    const int64_t nt = (rows + n_seq * (BQ - 1)) / BQ;
    if (nt <= 0) return RAD_OK;
    if (nt > 0x7fffffff) return RAD_E_SHAPE;
    Args A{};
    A.q = (const uint16_t*)q->data;  A.q_rs = q_rs; A.q_hs = q_hs;
    A.k = (const uint16_t*)k->data;  A.k_rs = k_rs; A.k_hs = k_hs;
    A.v = (const uint16_t*)v->data;  A.v_rs = v_rs; A.v_hs = v_hs;
    A.out = (uint16_t*)o->data;      A.o_rs = o_rs; A.o_hs = o_hs;
    A.cuq = (const int*)cq->data;
    A.cuk = (const int*)ck->data;
    A.sinks = (const float*)sk->data;
    A.qscale = (float)(scale * 1.4426950408889634);
    A.n_seq = (int)n_seq;
    A.n_keys = (int)nk;
    A.window = (int)window;
    R4D_KLAUNCH(attn_kv_sink_kernel, dim3((unsigned)nt, (unsigned)H, 1u), dim3(kThreads),
                (hipStream_t)stream, A);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* namespace r4d_akvs */

extern "C" int r4d_rad_attn_kv_sink_h256(const RadArgs* a, RadStream s) { return r4d_akvs::launch(a, s); }
