// The compressed indexer's group score (Hy4-flash, docs/OPS.md "idx_score"): how much one query
// wants each closed group of key history.
//
//   w_h      = gate[m][h] * (heads * n)^-0.5
//   score[m][b] = ( sum_{h < 8} w_h * relu(d_h) ) + ( sum_{h >= 8} w_h * relu(d_h) )
//   d_h      = q[m][h] . key(bt[seq(m)][b])
//
// heads 16 and n 128 are compiled in. One wave scores one query row against 16 keys: the 16 heads'
// query fragments are the A operand (loaded once a row), the 16 keys the B operand, read straight
// into registers in fragment order, and eight WMMA steps give D[head][key]. A lane then holds one
// key against eight heads (8 * half + e): relu, times w_h and the sum over e run in the lane, and
// one lane exchange adds the two halves, which is the order the reference states. A workgroup is
// four waves on 64 consecutive keys of one row: grid (M, ceil(columns / 64)). Rows do not interact,
// so a row's bytes are independent of M and of the other rows.
//
// Dot products run through WMMA, so a score is not bit-equal to libref's f32 sum; the order of the
// head sums is. A key past `nc[seq]`, without a page, or with a page outside the cache scores
// -inf, as does every column past the table. The op does no causal work (qsa_select does).
//
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr int kWave    = 32;
constexpr int kN       = 128;
constexpr int kHeads   = 16;
constexpr int kKS      = kN / 16;       /* WMMA K steps */
constexpr int kWaves   = 4;
constexpr int kKeysWg  = kWaves * 16;   /* keys a workgroup */

#define IDX_SCORE_PARAMS                                                                        \
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ gate,             \
    const unsigned short* __restrict__ key, const int32_t* __restrict__ bt,                     \
    const int32_t* __restrict__ cu, const int32_t* __restrict__ nc, float* __restrict__ out,    \
    int nseq, int cols, int out_cols, int pages, long long q_ld, long long gate_ld, long long key_ld,          \
    long long bt_ld, long long out_ld, float wscale
__device__ __forceinline__ void idx_score_body(IDX_SCORE_PARAMS) {

    const int lane = threadIdx.x & (kWave - 1), wave = threadIdx.x >> 5;
    const int half = lane >> 4, col = lane & 15;
    const int m  = (int)blockIdx.x;
    const int b0 = (int)blockIdx.y * kKeysWg + wave * 16;
    if (b0 >= out_cols) return;          /* wave-uniform; nothing below synchronises */

    /* The row's sequence: cu is nondecreasing, so a binary search for the last cu[i] <= m. */
    int lo = 0, hi = nseq;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (cu[mid] <= m) lo = mid; else hi = mid;
    }
    const int seq = (m >= cu[lo] && m < cu[lo + 1]) ? lo : -1;
    const int nlive = seq < 0 ? 0 : (nc[seq] < cols ? nc[seq] : cols);

    /* The key of this lane's column: its page, or page 0 (always present) when it has none. */
    const int b = b0 + col;
    const int page = (b < nlive) ? bt[(size_t)seq * bt_ld + b] : -1;
    const bool ok  = page >= 0 && page < pages;

    /* A operand: head `col`'s query row. B operand: the key. Same fragment layout for both. */
    const unsigned short* qp = q + (size_t)m * q_ld + (size_t)col * kN + 4 * half;
    const unsigned short* kp = key + (size_t)(ok ? page : 0) * key_ld + 4 * half;
    v8bf qf[kKS], kf[kKS];
#pragma unroll
    for (int ks = 0; ks < kKS; ++ks) qf[ks] = fragP(qp + ks * 16);
#pragma unroll
    for (int ks = 0; ks < kKS; ++ks) kf[ks] = fragP(kp + ks * 16);

    v8f acc;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < kKS; ++ks) acc = mma(qf[ks], kf[ks], acc);

    /* D[head 8 * half + e][key col]: relu, the head weight and the in-lane sum, e ascending. */
    const unsigned short* gp = gate + (size_t)m * gate_ld + 8 * half;
    float s = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float w = bf16_to_f32(gp[e]) * wscale;
        const float d = acc[e];
        s = s + w * (d < 0.0f ? 0.0f : d);
    }
    /* The first half of the heads plus the second. */
    const float other = swap16(s);
    const float total = half == 0 ? s + other : other + s;
    if (half == 0 && b < out_cols) out[(size_t)m * out_ld + b] = ok ? total : -INFINITY;
}

template <class CH>
__global__ __launch_bounds__(kWaves * kWave)
void idx_score_kernel(IDX_SCORE_PARAMS) {
    if (CH::on) r4d_chain_wait();
    idx_score_body(q, gate, key, bt, cu, nc, out, nseq, cols, out_cols, pages, q_ld, gate_ld,
                   key_ld, bt_ld, out_ld, wscale);
    if (CH::on) r4d_chain_done();
}
#undef IDX_SCORE_PARAMS

}  // namespace

// ============================================================== the plugin ABI entry point

enum { IS_Q = 0, IS_GATE = 1, IS_K = 2, IS_BT = 3, IS_CU = 4, IS_NC = 5, IS_SCORE = 6 };

extern "C" int r4d_rad_idx_score(const RadArgs* a, RadStream stream) {
    const RadTensor* q  = r4d_opt(a, IS_Q);
    const RadTensor* g  = r4d_opt(a, IS_GATE);
    const RadTensor* k  = r4d_opt(a, IS_K);
    const RadTensor* bt = r4d_opt(a, IS_BT);
    const RadTensor* cu = r4d_opt(a, IS_CU);
    const RadTensor* nc = r4d_opt(a, IS_NC);
    const RadTensor* o  = r4d_opt(a, IS_SCORE);
    if (!q || !g || !k || !bt || !cu || !nc || !o) return R4D_NO(RAD_E_INVAL);
    const long long n = r4d_geti_or(a, "n", 0), H = r4d_geti_or(a, "heads", 0);
    if (n != kN || H != kHeads) return R4D_NO(RAD_E_SHAPE);
    if (q->dtype != RAD_BF16 || g->dtype != RAD_BF16 || k->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (bt->dtype != RAD_I32 || cu->dtype != RAD_I32 || nc->dtype != RAD_I32 ||
        o->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    /* q: [M, heads, n] with heads and n dense, or [M, heads * n]. */
    long long M = 0;
    if (q->rank == 3) {
        if (q->shape[1] != H || q->shape[2] != n || q->stride[2] != 1 || q->stride[1] != n)
            return R4D_NO(RAD_E_SHAPE);
        M = q->shape[0];
    } else if (q->rank == 2) {
        if (q->shape[1] != H * n || q->stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
        M = q->shape[0];
    } else {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (M < 1 || M > (1ll << 30)) return R4D_NO(RAD_E_SHAPE);
    if (g->rank != 2 || g->shape[0] != M || g->shape[1] < H || g->stride[1] != 1)
        return R4D_NO(RAD_E_SHAPE);
    /* The keys: the cache [pages, 1, 4, n/4] (a page `stride[0]` apart) or flat [G, n]. */
    long long pages = 0, key_ld = 0;
    if (k->rank == 4) {
        if (k->shape[1] != 1 || k->shape[2] * k->shape[3] != n || k->stride[3] != 1 ||
            k->stride[2] != k->shape[3] || k->stride[0] < n)
            return R4D_NO(RAD_E_SHAPE);
        pages = k->shape[0]; key_ld = k->stride[0];
    } else if (k->rank == 2) {
        if (k->shape[1] != n || k->stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
        pages = k->shape[0]; key_ld = k->stride[0];
    } else {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (pages < 1) return R4D_NO(RAD_E_SHAPE);
    if (bt->rank != 2 || o->rank != 2 || cu->rank != 1 || cu->shape[0] < 2)
        return R4D_NO(RAD_E_SHAPE);
    if (bt->stride[1] != 1 || o->stride[1] != 1) return R4D_NO(RAD_E_STRIDE);
    const long long nseq = cu->shape[0] - 1;
    if (bt->shape[0] < nseq || r4d_numel(nc) < nseq || o->shape[0] != M || o->shape[1] < 1)
        return R4D_NO(RAD_E_SHAPE);
    const long long cols = bt->shape[1] < o->shape[1] ? bt->shape[1] : o->shape[1];
    const long long out_cols = o->shape[1];
    if (out_cols > (1ll << 20)) return R4D_NO(RAD_E_SHAPE);
    const float wscale = (float)(std::pow((double)H, -0.5) * std::pow((double)n, -0.5));
    const unsigned gy = (unsigned)((out_cols + kKeysWg - 1) / kKeysWg);
    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
#define R4D_ISC(CH_)                                                                            \
    R4D_EMIT(stream, (idx_score_kernel<CH_>), dim3((unsigned)M, gy), dim3(kWaves * kWave), 0,   \
             (const unsigned short*)r4d_p(q), (const unsigned short*)r4d_p(g),                  \
             (const unsigned short*)r4d_p(k), (const int32_t*)r4d_p(bt),                        \
             (const int32_t*)r4d_p(cu), (const int32_t*)r4d_p(nc), (float*)r4d_wp(o),           \
             (int)nseq, (int)cols, (int)out_cols, (int)pages, (long long)q->stride[0],               \
             (long long)g->stride[0], (long long)key_ld, (long long)bt->stride[0], (long long)o->stride[0], wscale)
    return chain ? R4D_ISC(r4dchain) : R4D_ISC(R4dUnchained);
#undef R4D_ISC
}
