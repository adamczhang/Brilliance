// r4d_mla_kv_bf16.hip -- Hy4-flash's kv_b read apart from the sink attention kernels:
// mla_project_kv (absorbed MLA's per-head projections with unequal K and V widths) and
// mla_kv_split (the kv_b output laid out as k and v heads for the non-absorbed prefill), and
// mla_expand_gate (the expansion with the attention gate's product folded into its store).
//
// mla_project_kv: head h's rows of the checkpoint's kv_b start at h * (k_dim + v_dim), its k_dim key
// rows first and then its v_dim value rows (kernels/mla_sink_attn.md section 5). Compiled for
// (k_dim, v_dim, latent) = (192, 256, 512), up to 64 heads.
//   absorb: y[m,h,j] = sum_{c<192} a[m,h,c] * b[h*448 + c][j]            [M,H,192] -> [M,H,512]
//   expand: y[m,h,c] = sum_{j<512} a[m,h,j] * b[h*448 + 192 + c][j]      [M,H,512] -> [M,H,256]
// mla_expand_gate is expand with gate_mul's epilogue: ag = bf16(f32(bf16(sum)) * sigmoid(f32(g))),
// byte-equal to the expand followed by gate_mul (r4d_ihc_bf16.hip's formula, the same expf).
//
// The law (kernels/perf.md section 3.3, WP16.3): gfx12's bf16 WMMA, 16x16x16, the output columns
// as the A operand's rows and the tokens as B's columns, f32 accumulators, one rounding at the
// store. absorb: the 192 in steps of 16 ascending, one chain. expand: the 512 in four quarters,
// each a chain of eight steps ascending from zero, the quarters added in order ((q0 + q1) + q2) + q3
// in f32. A token's 16 products of a step and the accumulator it adds them to are its own, so its
// bytes depend on its own input row only: not on M, on the tile it shares (a token column past M
// reads the last row and is never stored), on `mb` (token tiles a wave, a rad-tune axis), on the
// column tiles a wave takes or on the grid. A different sum order from the scalar kernel this
// replaced (an fma chain): within one bf16 ulp of libref's, as that one was.
//
// Fragments (gfx12 wave32): a lane holds row (A) / column (B) lane % 16 and the eight k
// 8 * (lane / 16) + e. Absorb's A is the key block read across its rows, Wk[c][col] at c along
// the fragment: global_load_tr_b128 gives it from eight 16-byte row pieces, one load a lane, which
// is what a latent-major copy of the key block (perf.md's phase B) would give, without a second
// stored form of kv_b. Expand's A rows are kv_b's value rows (contiguous). The tokens' fragments are
// 16-byte loads of the token's own row. Operands off the 16-byte grid (a pitched view, a slice
// starting off it) take every fragment element by element: the same fragments into the same WMMAs
// (perf.md's phase A), so the same bytes.
//
// A workgroup is four waves. absorb: a wave takes NC column tiles of 16 (two past 16 rows) and MB
// token tiles; expand: the workgroup takes one column tile, a wave a quarter of the contraction
// and MB token tiles, the partial sums meeting in LDS. Each A fragment is loaded once for its MB
// token tiles, each token fragment once for its NC column tiles, two chunks of steps in flight.
//
// Chained forms (r4d_chain.h): the aligned path at NC 1, chosen at decode heights (M <= 64).
// mla_kv_split is a copy: k[i,h,0:kd] = kvb[i, h*(kd+vd) : +kd], k[i,h,kd:kd+r] = k_rope[i],
// v[i,h,:] = kvb[i, h*(kd+vd)+kd : +vd]. One thread a 16 bytes of output.
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_fwht.h"
#include "r4d_q8.h"
#include <cstdint>

namespace {

constexpr int kKd = 192, kVd = 256, kLatent = 512, kRows = kKd + kVd;

struct KvProjectMeta {
    long long xr, xh, wp, yr, yh, gr;
    int rows;
};

__device__ __forceinline__ v8f kvp_wmma(v8s a, v8s b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
/* gate_mul's sigmoid (r4d_ihc_bf16.hip ihc_sigmoid), for the fused store */
__device__ __forceinline__ float kvp_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* Eight consecutive bf16: one 16-byte load (VEC) or eight. */
template <bool VEC>
__device__ __forceinline__ v8s kvp_ld8(const uint16_t* p) {
    if constexpr (VEC) {
        return *(const v8s*)p;
    } else {
        v8s v;
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = (short)p[e];
        return v;
    }
}

/* The weight's fragment for output columns [o0, o0 + 16) at contraction step k0. `wb` is the
 * head's first row of the block read (its key rows for absorb, its value rows for expand). */
template <bool ABSORB, bool VEC>
__device__ __forceinline__ v8s kvp_wfrag(const uint16_t* wb, long long wp, int o0, int k0, int lane) {
    if constexpr (ABSORB) {
        /* A[col][c] = Wk[c][col]: lane l names row c = k0 + 8 (l / 16) + l % 8 at columns
         * o0 + 8 ((l / 8) % 2), and receives column o0 + l % 16 across its group's eight rows */
        if constexpr (VEC) {
            return load_tr_b128(wb + (long long)(k0 + 8 * (lane >> 4) + (lane & 7)) * wp + o0 +
                                8 * ((lane >> 3) & 1));
        } else {
            const uint16_t* p = wb + (long long)(k0 + 8 * (lane >> 4)) * wp + o0 + (lane & 15);
            v8s v;
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] = (short)p[(long long)e * wp];
            return v;
        }
    } else {
        return kvp_ld8<VEC>(wb + (long long)(o0 + (lane & 15)) * wp + k0 + 8 * (lane >> 4));
    }
}

/* The contraction steps a wave loads at once, two such chunks in flight (one computed while the
 * next one loads): as many as keep the fragments and accumulators near 176 VGPRs, since the load
 * latency is what a wave with few tiles waits on (a decode launch; expand's long contraction).
 * Changes when a fragment is loaded, not what is summed. */
/* Expand's contraction is split in four (kSplit): wave q of a workgroup sums j in [128 q,
 * 128 q + 128) for the workgroup's one column tile, and the four partial sums are added in wave
 * order, ((p0 + p1) + p2) + p3, before the rounding. Part of the law at every M: a token's bytes
 * still depend on its own row only. Absorb's 192 are one wave's. */
template <bool ABSORB>
constexpr int kvp_split() { return ABSORB ? 1 : 4; }
constexpr int kvp_threads(bool) { return 128; }

template <bool ABSORB, int MB, int NC>
constexpr int kvp_chunk() {
    constexpr int KS = (ABSORB ? kKd : kLatent) / 16 / kvp_split<ABSORB>();
    int best = 1;
    for (int kc = 1; kc <= KS / 2; ++kc)
        if (KS % (2 * kc) == 0 && 8 * kc * (NC + MB) + 8 * NC * MB <= 176) best = kc;
    return best;
}

template <bool ABSORB, int MB, int NC, bool VEC, bool GATE, class CH>
__device__ __forceinline__ void kvp_body(const uint16_t* __restrict__ x,
                                         const uint16_t* __restrict__ w,
                                         const uint16_t* __restrict__ g, uint16_t* __restrict__ y,
                                         const KvProjectMeta& m) {
    constexpr int S = kvp_split<ABSORB>(), K = (ABSORB ? kKd : kLatent) / S, KS = K / 16,
                  KC = kvp_chunk<ABSORB, MB, NC>(), NCH = KS / KC;
    static_assert(KS % (2 * KC) == 0, "chunks in pairs");
    static_assert(S == 1 || NC == 1, "a split workgroup takes one column tile");
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, head = blockIdx.y;
    /* S 1: the four waves take four column tiles (of NC each); S 4: one, a quarter of k each */
    const int o0 = (S == 1 ? blockIdx.x * 4 + wave : blockIdx.x) * 16 * NC, t0 = blockIdx.z * 16 * MB;
    const int kq = S == 1 ? 0 : wave * K;
    const uint16_t* const wb = w + ((long long)head * kRows + (ABSORB ? 0 : kKd)) * m.wp + (ABSORB ? 0 : kq);
    /* A token past M reads the last row: its column of the tile is never stored, and no other
     * column reads it (a WMMA output is its own row of A times its own column of B). */
    const uint16_t* xp[MB];
#pragma unroll
    for (int r = 0; r < MB; ++r) {
        const int t = min(t0 + 16 * r + (lane & 15), m.rows - 1);
        xp[r] = x + (long long)t * m.xr + (long long)head * m.xh + kq + 8 * (lane >> 4);
    }
    v8f acc[NC][MB];
#pragma unroll
    for (int n = 0; n < NC; ++n)
#pragma unroll
        for (int r = 0; r < MB; ++r)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[n][r][e] = 0.0f;
    v8s a0[KC][NC], b0[KC][MB], a1[KC][NC], b1[KC][MB];
    auto load = [&](v8s (&a)[KC][NC], v8s (&b)[KC][MB], int c) {
#pragma unroll
        for (int q = 0; q < KC; ++q) {
            const int k0 = 16 * (c * KC + q);
#pragma unroll
            for (int n = 0; n < NC; ++n) a[q][n] = kvp_wfrag<ABSORB, VEC>(wb, m.wp, o0 + 16 * n, k0, lane);
#pragma unroll
            for (int r = 0; r < MB; ++r) b[q][r] = kvp_ld8<VEC>(xp[r] + k0);
        }
    };
    /* The quarter's weight fragments (the `load`'s a half) and token fragments (its b half) on
     * their own, so that a chained launch can ask for the weight before the acquire. */
    auto wload = [&](v8s (&a)[KC][NC], int c) {
#pragma unroll
        for (int q = 0; q < KC; ++q) {
            const int k0 = 16 * (c * KC + q);
#pragma unroll
            for (int n = 0; n < NC; ++n) a[q][n] = kvp_wfrag<ABSORB, VEC>(wb, m.wp, o0 + 16 * n, k0, lane);
        }
    };
    auto xload = [&](v8s (&b)[KC][MB], int c) {
#pragma unroll
        for (int q = 0; q < KC; ++q) {
            const int k0 = 16 * (c * KC + q);
#pragma unroll
            for (int r = 0; r < MB; ++r) b[q][r] = kvp_ld8<VEC>(xp[r] + k0);
        }
    };
    auto step = [&](const v8s (&a)[KC][NC], const v8s (&b)[KC][MB]) {
#pragma unroll
        for (int q = 0; q < KC; ++q)
#pragma unroll
            for (int n = 0; n < NC; ++n)
#pragma unroll
                for (int r = 0; r < MB; ++r) acc[n][r] = kvp_wmma(a[q][n], b[q][r], acc[n][r]);
    };
    /* Chained, the block waits in the poll while its predecessor runs (aql.cpp's chain plan), so
     * the first quarter of kv_b is issued before the acquire: a static weight's bytes are written
     * by no op of a pass (operand role RAD_OPD_WEIGHT, core/runtime/issue.cpp's haz_note orders no
     * read of one), so neither load can see a partial write, and the fragments stay in registers
     * across the acquire. The tokens, the attention output the predecessor wrote, stay behind the
     * wait, exactly where the plain form's single `load` has them. */
    if constexpr (CH::on) {
        wload(a0, 0);
        r4d_chain_wait();
        xload(b0, 0);
    } else {
        load(a0, b0, 0);
    }
#pragma unroll 1
    for (int c = 0; c < NCH; c += 2) {
        load(a1, b1, c + 1);
        step(a0, b0);
        if (c + 2 < NCH) load(a0, b0, c + 2);
        step(a1, b1);
    }
    if constexpr (S > 1) {
        /* the partial sums through LDS; wave r % 4 adds and stores token tile r */
        __shared__ float part[S][MB][32][8];
#pragma unroll
        for (int r = 0; r < MB; ++r)
#pragma unroll
            for (int e = 0; e < 8; ++e) part[wave][r][lane][e] = acc[0][r][e];
        __syncthreads();
#pragma unroll
        for (int r = 0; r < MB; ++r) {
            if (r % S != wave) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float v = part[0][r][lane][e];
#pragma unroll
                for (int q = 1; q < S; ++q) v += part[q][r][lane][e];
                acc[0][r][e] = v;
            }
        }
    }
    /* lane l holds token t0 + 16 r + l % 16 at the output columns o0 + 16 n + 8 (l / 16) + e */
#pragma unroll
    for (int r = 0; r < MB; ++r) {
        if (S > 1 && r % S != wave) continue;
        const long long t = t0 + 16 * r + (lane & 15);
        if (t >= m.rows) continue;
#pragma unroll
        for (int n = 0; n < NC; ++n) {
            const int c = o0 + 16 * n + 8 * (lane >> 4);
            uint16_t* const yp = y + t * m.yr + (long long)head * m.yh + c;
            uint16_t o[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) o[e] = f32_to_bf16(acc[n][r][e]);
            if constexpr (GATE) {
                const v8s gv = kvp_ld8<VEC>(g + t * m.gr + (long long)head * kVd + c);
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    o[e] = f32_to_bf16(bf16_to_f32(o[e]) * kvp_sigmoid(bf16_to_f32((uint16_t)gv[e])));
            }
            if constexpr (VEC) {
                uint4 v;
                v.x = (uint32_t)o[0] | (uint32_t)o[1] << 16;
                v.y = (uint32_t)o[2] | (uint32_t)o[3] << 16;
                v.z = (uint32_t)o[4] | (uint32_t)o[5] << 16;
                v.w = (uint32_t)o[6] | (uint32_t)o[7] << 16;
                *(uint4*)yp = v;
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) yp[e] = o[e];
            }
        }
    }
}

template <bool ABSORB, int MB, int NC, bool VEC, bool GATE, typename CH>
__global__ __launch_bounds__(kvp_threads(ABSORB)) void kv_project_wmma(const uint16_t* __restrict__ x,
                                                       const uint16_t* __restrict__ w,
                                                       const uint16_t* __restrict__ g,
                                                       uint16_t* __restrict__ y, KvProjectMeta m) {
    kvp_body<ABSORB, MB, NC, VEC, GATE, CH>(x, w, g, y, m);
    if constexpr (CH::on) r4d_chain_done();
}

// ---------------------------------------------------------------- mla_expand_gate_had_quant_i8g
// The expansion with the gate folded in (GATE, above) and, behind it, had_quant_act_i8g over the
// [M, heads * v_dim] gated product: the codes attn_output's rotated int8 GEMM reads, in the launch that
// computes them (G-3c). The quantiser's group is 128 columns of a head's 256, so a workgroup takes
// both the column tiles of one group, not one: eight column tiles x the four contraction quarters is
// 32 waves (1024 threads), one token tile of 16, one group. Every wave runs expand's body as it is
// (S 4, NC 1, MB 1): the same fragments, the same WMMA chain over its quarter, the quarters added in
// wave order ((p0 + p1) + p2) + p3 in f32, the same rounding to bf16 and the same gate product. Then
// the bf16 product of the group goes through LDS (16 tokens x 128) to one wave a token, which runs
// had_quant_act_i8g's pass on it (quant_act_fp8_body at ROT, I8: r4d_fwht_wave<128> over the four
// values a lane owns, the wave's absmax, scale amax / 127 and the rounded codes), so the codes are the
// chain's bytes and, like the expansion's, a token's depend on its own row only (not on M, on the
// token tile or on the grid). `y`, when given, receives the bf16 product (the plane the chain stores).
struct KvQuantMeta {
    long long xr, wp, gr, qr, sr, yr;
    int rows;
};
constexpr int kQG = 128;          // the quantiser's group: the rotation width and the scale's
constexpr int kQTiles = kQG / 16; // column tiles of a group
constexpr int kQThreads = 32 * kQTiles * 4;   // 8 tiles x 4 quarters of the contraction = 1024

template <class CH>
__device__ __forceinline__ void kvq_body(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                         const uint16_t* __restrict__ g, signed char* __restrict__ q,
                                         float* __restrict__ sc, uint16_t* __restrict__ y,
                                         const KvQuantMeta& m) {
    constexpr int S = 4, K = kLatent / S, KS = K / 16, KC = kvp_chunk<false, 1, 1>(), NCH = KS / KC;
    static_assert(KS % (2 * KC) == 0, "chunks in pairs");
    __shared__ float part[S][kQTiles][32][8];
    __shared__ uint16_t ag[16][kQG];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, head = blockIdx.y;
    const int ct = wave % kQTiles, quarter = wave / kQTiles;
    const int o0 = (int)blockIdx.x * kQG + ct * 16, t0 = (int)blockIdx.z * 16, kq = quarter * K;
    const uint16_t* const wb = w + ((long long)head * kRows + kKd) * m.wp + kq;
    const int tl = min(t0 + (lane & 15), m.rows - 1);
    const uint16_t* const xp = x + (long long)tl * m.xr + (long long)head * kLatent + kq + 8 * (lane >> 4);
    v8f acc;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
    v8s a0[KC], b0[KC], a1[KC], b1[KC];
    auto load = [&](v8s (&a)[KC], v8s (&b)[KC], int c) {
#pragma unroll
        for (int qq = 0; qq < KC; ++qq) {
            const int k0 = 16 * (c * KC + qq);
            a[qq] = kvp_wfrag<false, true>(wb, m.wp, o0, k0, lane);
            b[qq] = kvp_ld8<true>(xp + k0);
        }
    };
    /* kvp_body's wload / xload, over one column tile a wave: the weight's fragments of the first
     * chunk on their own, so a chained launch can ask for them before the acquire. */
    auto wload = [&](v8s (&a)[KC], int c) {
#pragma unroll
        for (int qq = 0; qq < KC; ++qq) {
            const int k0 = 16 * (c * KC + qq);
            a[qq] = kvp_wfrag<false, true>(wb, m.wp, o0, k0, lane);
        }
    };
    auto xload = [&](v8s (&b)[KC], int c) {
#pragma unroll
        for (int qq = 0; qq < KC; ++qq) {
            const int k0 = 16 * (c * KC + qq);
            b[qq] = kvp_ld8<true>(xp + k0);
        }
    };
    auto step = [&](const v8s (&a)[KC], const v8s (&b)[KC]) {
#pragma unroll
        for (int qq = 0; qq < KC; ++qq) acc = kvp_wmma(a[qq], b[qq], acc);
    };
    /* Chained, the weight's first chunk goes out before the acquire and the tokens' after it, as
     * kvp_body's do (a static weight's bytes; see the note there). */
    if constexpr (CH::on) {
        wload(a0, 0);
        r4d_chain_wait();
        xload(b0, 0);
    } else {
        load(a0, b0, 0);
    }
#pragma unroll 1
    for (int c = 0; c < NCH; c += 2) {
        load(a1, b1, c + 1);
        step(a0, b0);
        if (c + 2 < NCH) load(a0, b0, c + 2);
        step(a1, b1);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) part[quarter][ct][lane][e] = acc[e];
    __syncthreads();
    if (quarter == 0) {
        // lane l holds token t0 + l % 16 at the group's columns 16 ct + 8 (l / 16) + e
        const int tt = lane & 15;
        const long long t = t0 + tt;
        const int cg = ct * 16 + 8 * (lane >> 4);
        const v8s gv = kvp_ld8<true>(g + min(t, (long long)m.rows - 1) * m.gr + (long long)head * kVd +
                                     (long long)blockIdx.x * kQG + cg);
        uint16_t o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float v = part[0][ct][lane][e];
#pragma unroll
            for (int qq = 1; qq < S; ++qq) v += part[qq][ct][lane][e];
            o[e] = f32_to_bf16(v);
            o[e] = f32_to_bf16(bf16_to_f32(o[e]) * kvp_sigmoid(bf16_to_f32((uint16_t)gv[e])));
            ag[tt][cg + e] = o[e];
        }
        if (y && t < m.rows) {
            uint4 v;
            v.x = (uint32_t)o[0] | (uint32_t)o[1] << 16;
            v.y = (uint32_t)o[2] | (uint32_t)o[3] << 16;
            v.z = (uint32_t)o[4] | (uint32_t)o[5] << 16;
            v.w = (uint32_t)o[6] | (uint32_t)o[7] << 16;
            *(uint4*)(y + t * m.yr + (long long)head * kVd + (long long)blockIdx.x * kQG + cg) = v;
        }
    }
    __syncthreads();
    if (wave < 16) {
        const long long t = t0 + wave;
        if (t < m.rows) {
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = bf16_to_f32(ag[wave][lane * 4 + j]);
            r4d_fwht_wave<kQG>(v, lane);
            float mx = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
            mx = r4d_wave_max(mx);
            const float s = mx > 0.0f ? mx * (1.0f / q8_top<true>()) : 1.0f;
            if (lane == 0) sc[t * m.sr + head * 2 + blockIdx.x] = s;
            *(uint32_t*)(q + t * m.qr + (long long)head * kVd + (long long)blockIdx.x * kQG + lane * 4) =
                q8_pack4<true>(v[0], v[1], v[2], v[3], 1.0f / s);
        }
    }
}

template <typename CH>
__global__ __launch_bounds__(kQThreads) void kv_expand_gate_quant(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, const uint16_t* __restrict__ g,
    signed char* __restrict__ q, float* __restrict__ sc, uint16_t* __restrict__ y, KvQuantMeta m) {
    kvq_body<CH>(x, w, g, q, sc, y, m);
    if constexpr (CH::on) r4d_chain_done();
}

/* The launch: grid (column blocks, heads, token groups of 16 MB). */
template <bool ABSORB, int MB, int NC, bool VEC, bool GATE, typename CH>
int kvp_emit(RadStream s, unsigned heads, const uint16_t* x, const uint16_t* w, const uint16_t* g,
             uint16_t* y, const KvProjectMeta& m) {
    constexpr int OUT = ABSORB ? kLatent : kVd, PER = 16 * NC * (kvp_split<ABSORB>() == 1 ? 4 : 1);
    static_assert(OUT % PER == 0, "the workgroups' column tiles divide the output");
    const unsigned groups = (unsigned)((m.rows + 16 * MB - 1) / (16 * MB));
    return R4D_EMIT(s, (kv_project_wmma<ABSORB, MB, NC, VEC, GATE, CH>),
                    dim3(OUT / PER, heads, groups), dim3(kvp_threads(ABSORB)), 0, x, w, g, y, m);
}

template <bool ABSORB, bool GATE>
int kvp_pick(RadStream s, unsigned heads, int mb, bool vec, bool chain, const uint16_t* x,
             const uint16_t* w, const uint16_t* g, uint16_t* y, const KvProjectMeta& m) {
    if (!vec) return kvp_emit<ABSORB, 1, 1, false, GATE, R4dUnchained>(s, heads, x, w, g, y, m);
    if (chain) {
        switch (mb) {
            case 1: return kvp_emit<ABSORB, 1, 1, true, GATE, r4dchain>(s, heads, x, w, g, y, m);
            case 2: return kvp_emit<ABSORB, 2, 1, true, GATE, r4dchain>(s, heads, x, w, g, y, m);
            case 4: return kvp_emit<ABSORB, 4, 1, true, GATE, r4dchain>(s, heads, x, w, g, y, m);
            default: return kvp_emit<ABSORB, 8, 1, true, GATE, r4dchain>(s, heads, x, w, g, y, m);
        }
    }
    /* absorb: two column tiles a wave past decode heights, half the token fragments' traffic;
     * expand's split workgroup takes one */
    if (!ABSORB || m.rows <= 16) {
        switch (mb) {
            case 1: return kvp_emit<ABSORB, 1, 1, true, GATE, R4dUnchained>(s, heads, x, w, g, y, m);
            case 2: return kvp_emit<ABSORB, 2, 1, true, GATE, R4dUnchained>(s, heads, x, w, g, y, m);
            case 4: return kvp_emit<ABSORB, 4, 1, true, GATE, R4dUnchained>(s, heads, x, w, g, y, m);
            default: return kvp_emit<ABSORB, 8, 1, true, GATE, R4dUnchained>(s, heads, x, w, g, y, m);
        }
    }
    if constexpr (ABSORB) {
        switch (mb) {
            case 1: return kvp_emit<true, 1, 2, true, GATE, R4dUnchained>(s, heads, x, w, g, y, m);
            case 2: return kvp_emit<true, 2, 2, true, GATE, R4dUnchained>(s, heads, x, w, g, y, m);
            case 4: return kvp_emit<true, 4, 2, true, GATE, R4dUnchained>(s, heads, x, w, g, y, m);
            /* eight token tiles take one column tile: two would need 256 VGPRs and spill */
            default: return kvp_emit<true, 8, 1, true, GATE, R4dUnchained>(s, heads, x, w, g, y, m);
        }
    }
    return R4D_NO(RAD_E_INVAL);
}

// One thread a chunk of C elements (8 for 16-byte accesses, 1 otherwise) of one (row, head): the
// k row's chunks, kvb's then k_rope's, then the v row's.
template <int C>
__global__ __launch_bounds__(256) void kv_split_kernel(
    const uint16_t* __restrict__ kvb, const uint16_t* __restrict__ kr, uint16_t* __restrict__ k,
    uint16_t* __restrict__ v, uint32_t N, uint32_t heads, uint32_t kd, uint32_t vd, uint32_t rd,
    uint64_t bp, uint64_t rp, uint64_t kp, uint64_t vp) {
    const uint32_t kc = (kd + rd) / C, vc = vd / C, per = kc + vc;
    const uint64_t g = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    if (g >= (uint64_t)N * heads * per) return;
    const uint64_t rh = g / per;
    const uint32_t c = (uint32_t)(g % per), h = (uint32_t)(rh % heads);
    const uint64_t i = rh / heads;
    const uint16_t* src = kvb + i * bp + (uint64_t)h * (kd + vd);
    uint16_t* dst;
    const uint16_t* from;
    if (c < kc) {
        const uint32_t e = c * C;
        dst = k + i * kp + (uint64_t)h * (kd + rd) + e;
        from = e < kd ? src + e : kr + i * rp + (e - kd);
    } else {
        const uint32_t e = (c - kc) * C;
        dst = v + i * vp + (uint64_t)h * vd + e;
        from = src + kd + e;
    }
    if (C == 8) *(uint4*)dst = *(const uint4*)from;
    else        *dst = *from;
}

}  // namespace

/* The `mb` law (the row's tunable, 0 = this): one token tile a wave up to 16 rows, four past them,
 * so a prefill launch reads kv_b's block once for every 64 tokens (measured on the R9700 over 1, 2,
 * 4 and 8 at 8 to 64 heads and M 1 to 1024: four is the fastest or within a few percent of it
 * from 32 rows on, in both modes). Bit-neutral. */
static int kvp_mb(long long M, long long tuned) {
    if (tuned == 1 || tuned == 2 || tuned == 4 || tuned == 8) return (int)tuned;
    return M <= 16 ? 1 : 4;
}

/* mla_project_kv (gate null) and mla_expand_gate (gate the [M, heads * v_dim] logits). */
static int kvp_launch(const RadArgs* a, RadStream s, const RadTensor* x, const RadTensor* w,
                      const RadTensor* gt, const RadTensor* y, bool absorb) {
    if (x->dtype != RAD_BF16 || w->dtype != RAD_BF16 || y->dtype != RAD_BF16 ||
        (gt && gt->dtype != RAD_BF16))
        return R4D_NO(RAD_E_DTYPE);
    const long long h = r4d_geti_or(a, "heads", 0);
    if (h < 1 || h > 64 || r4d_geti_or(a, "k_dim", 0) != kKd || r4d_geti_or(a, "v_dim", 0) != kVd ||
        r4d_geti_or(a, "latent_dim", 0) != kLatent)
        return R4D_NO(RAD_E_SHAPE);
    const long long xw = absorb ? kKd : kLatent, yw = absorb ? kLatent : kVd;
    /* [M, heads * width] or [M, heads, width]: a token's heads are one row, a head a slice. */
    long long M = 0, My = 0, Mg = 0;
    const long long xr = r4d_rows_pitch(x, h * xw, &M), yr = r4d_rows_pitch(y, h * yw, &My);
    const long long gr = gt ? r4d_rows_pitch(gt, h * kVd, &Mg) : 0;
    if (xr < 0 || yr < 0 || gr < 0) return R4D_NO(RAD_E_STRIDE);
    if (M < 1 || M > 65535 || My != M || (gt && Mg != M)) return R4D_NO(RAD_E_SHAPE);
    if (w->rank != 2 || w->shape[0] != h * kRows || w->shape[1] != kLatent || w->stride[1] != 1 ||
        w->stride[0] < kLatent)
        return R4D_NO(RAD_E_SHAPE);
    const KvProjectMeta m{ (long long)xr, xw, (long long)w->stride[0], (long long)yr, yw, (long long)gr, (int)M };
    const auto* xp = (const uint16_t*)x->data;
    const auto* wp = (const uint16_t*)w->data;
    const auto* gp = gt ? (const uint16_t*)gt->data : nullptr;
    auto* yp = (uint16_t*)y->data;
    /* 16-byte fragments, load_tr's row pieces and 16-byte stores need every row on the grid */
    const bool vec = xr % 8 == 0 && yr % 8 == 0 && gr % 8 == 0 && m.wp % 8 == 0 &&
                     r4d_aligned(xp, 16) && r4d_aligned(wp, 16) && r4d_aligned(yp, 16) &&
                     (!gp || r4d_aligned(gp, 16));
    const int mb = kvp_mb(M, r4d_geti_or(a, "mb", 0));
    const bool chain = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
    const unsigned H = (unsigned)h;
    if (absorb) return kvp_pick<true, false>(s, H, mb, vec, chain, xp, wp, nullptr, yp, m);
    if (gp) return kvp_pick<false, true>(s, H, mb, vec, chain, xp, wp, gp, yp, m);
    return kvp_pick<false, false>(s, H, mb, vec, chain, xp, wp, nullptr, yp, m);
}

extern "C" int r4d_rad_mla_project_kv(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* w = r4d_opt(a, 1);
    const RadTensor* y = r4d_opt(a, 2);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    const char* mode = r4d_gets(a, "mode");
    const bool absorb = mode && !std::strcmp(mode, "absorb"),
               expand = mode && !std::strcmp(mode, "expand");
    if (!absorb && !expand) return R4D_NO(RAD_E_SHAPE);
    return kvp_launch(a, s, x, w, nullptr, y, absorb);
}

/* Operands: [0] x (the latent attention's output) [1] w (kv_b) [2] g (the gate's logits) [3] y */
extern "C" int r4d_rad_mla_expand_gate(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *w = r4d_opt(a, 1), *g = r4d_opt(a, 2), *y = r4d_opt(a, 3);
    if (!x || !w || !g || !y) return R4D_NO(RAD_E_INVAL);
    return kvp_launch(a, s, x, w, g, y, false);
}

/* Operands: [0] x (the latent attention's output) [1] w (kv_b) [2] g (the gate's logits) [3] q
 * (int8 [M, heads * v_dim]) [4] scale (f32 [M, heads * v_dim / 128]) [5] out_bf16? (the gated product,
 * [M, heads, v_dim] or [M, heads * v_dim]). */
extern "C" int r4d_rad_mla_expand_gate_had_quant_i8g(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *w = r4d_opt(a, 1), *g = r4d_opt(a, 2), *q = r4d_opt(a, 3),
                    *sc = r4d_opt(a, 4), *y = r4d_opt(a, 5);
    if (!x || !w || !g || !q || !sc) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || w->dtype != RAD_BF16 || g->dtype != RAD_BF16 || q->dtype != RAD_I8 ||
        sc->dtype != RAD_F32 || (y && y->dtype != RAD_BF16))
        return R4D_NO(RAD_E_DTYPE);
    const long long h = r4d_geti_or(a, "heads", 0);
    if (h < 1 || h > 64 || r4d_geti_or(a, "k_dim", 0) != kKd || r4d_geti_or(a, "v_dim", 0) != kVd ||
        r4d_geti_or(a, "latent_dim", 0) != kLatent || r4d_geti_or(a, "group", kQG) != kQG)
        return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mg = 0, Mq = 0, Ms = 0, My = 0;
    const long long xr = r4d_rows_pitch(x, h * kLatent, &M), gr = r4d_rows_pitch(g, h * kVd, &Mg),
                    qr = r4d_rows_pitch(q, h * kVd, &Mq), sr = r4d_rows_pitch(sc, h * kVd / kQG, &Ms),
                    yr = y ? r4d_rows_pitch(y, h * kVd, &My) : 0;
    if (xr < 0 || gr < 0 || qr < 0 || sr < 0 || yr < 0) return R4D_NO(RAD_E_STRIDE);
    if (M < 1 || M > 65535 || Mg != M || Mq != M || Ms != M || (y && My != M)) return R4D_NO(RAD_E_SHAPE);
    if (w->rank != 2 || w->shape[0] != h * kRows || w->shape[1] != kLatent || w->stride[1] != 1 ||
        w->stride[0] < kLatent)
        return R4D_NO(RAD_E_SHAPE);
    const auto* xp = (const uint16_t*)x->data;
    const auto* wp = (const uint16_t*)w->data;
    const auto* gp = (const uint16_t*)g->data;
    auto* yp = y ? (uint16_t*)y->data : nullptr;
    /* 16-byte fragments, load_tr's row pieces, 16-byte stores and the 4-byte code stores */
    if (xr % 8 || gr % 8 || yr % 8 || w->stride[0] % 8 || qr % 4 || !r4d_aligned(xp, 16) ||
        !r4d_aligned(wp, 16) || !r4d_aligned(gp, 16) || (yp && !r4d_aligned(yp, 16)) ||
        !r4d_aligned(q->data, 4))
        return R4D_NO(RAD_E_ALIGN);
    const KvQuantMeta m{ (long long)xr, (long long)w->stride[0], (long long)gr, (long long)qr, (long long)sr, (long long)yr, (int)M };
    const dim3 grid(kVd / kQG, (unsigned)h, (unsigned)((M + 15) / 16));
    if (r4d_geti_or(a, "chain", 0) != 0 && M <= 64)
        return R4D_EMIT(s, (kv_expand_gate_quant<r4dchain>), grid, dim3(kQThreads), 0, xp, wp, gp,
                        (signed char*)q->data, (float*)sc->data, yp, m);
    return R4D_EMIT(s, (kv_expand_gate_quant<R4dUnchained>), grid, dim3(kQThreads), 0, xp, wp, gp,
                    (signed char*)q->data, (float*)sc->data, yp, m);
}

extern "C" int r4d_rad_mla_kv_split(const RadArgs* a, RadStream s) {
    const RadTensor *kvb = r4d_opt(a, 0), *kr = r4d_opt(a, 1), *k = r4d_opt(a, 2),
                    *v = r4d_opt(a, 3);
    if (!kvb || !kr || !k || !v) return R4D_NO(RAD_E_INVAL);
    if (kvb->dtype != RAD_BF16 || kr->dtype != RAD_BF16 || k->dtype != RAD_BF16 ||
        v->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    const long long H = r4d_geti_or(a, "heads", 0), kd = r4d_geti_or(a, "k_dim", 0),
                    vd = r4d_geti_or(a, "v_dim", 0), rd = r4d_geti_or(a, "rope_dim", 0);
    if (H < 1 || kd < 1 || vd < 1 || rd < 1) return R4D_NO(RAD_E_SHAPE);
    long long N = 0, Nr = 0, Nk = 0, Nv = 0;
    const long long bp = r4d_rows_pitch(kvb, H * (kd + vd), &N), rp = r4d_rows_pitch(kr, rd, &Nr),
                    kp = r4d_rows_pitch(k, H * (kd + rd), &Nk),
                    vp = r4d_rows_pitch(v, H * vd, &Nv);
    if (bp < 0 || rp < 0 || kp < 0 || vp < 0) return R4D_NO(RAD_E_STRIDE);
    if (N < 1 || Nr != N || Nk != N || Nv != N) return R4D_NO(RAD_E_SHAPE);
    const bool vec = kd % 8 == 0 && vd % 8 == 0 && rd % 8 == 0 && bp % 8 == 0 && rp % 8 == 0 &&
                     kp % 8 == 0 && vp % 8 == 0 && r4d_aligned(kvb->data, 16) &&
                     r4d_aligned(kr->data, 16) && r4d_aligned(k->data, 16) &&
                     r4d_aligned(v->data, 16);
    const long long per = (kd + rd + vd) / (vec ? 8 : 1);
    const unsigned long long total = (unsigned long long)N * H * per;
    const unsigned long long blocks = (total + 255) / 256;
    if (blocks > 0x7fffffffull) return R4D_NO(RAD_E_SHAPE);
#define R4D_KVS(C)                                                                               \
    R4D_EMIT(s, (kv_split_kernel<C>), dim3((uint32_t)blocks), dim3(256), 0,                      \
             (const uint16_t*)kvb->data, (const uint16_t*)kr->data, (uint16_t*)k->data,         \
             (uint16_t*)v->data, (uint32_t)N, (uint32_t)H, (uint32_t)kd, (uint32_t)vd,           \
             (uint32_t)rd, (uint64_t)bp, (uint64_t)rp, (uint64_t)kp, (uint64_t)vp)
    return vec ? R4D_KVS(8) : R4D_KVS(1);
#undef R4D_KVS
}
