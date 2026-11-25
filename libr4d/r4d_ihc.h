// r4d_ihc.h -- the independent hyper-connection's device pieces, shared by ihc_pre
// (r4d_ihc_bf16.hip), the fused junction rmsnorm_ihc_junction (same unit) and its all-reduce twin
// ar_rmsnorm_ihc_junction (r4d_ar_ihc_junction.hip). One summation order for all of them, so the
// fused rows are ihc_pre's bytes.
//
// ihc_pre's order (WP16.2; the scalar kernel it replaced strided single columns by 256 threads):
// a workgroup of kIhcThreads = 512 a token row. The row's W = hc * n values are pieces of eight
// consecutive values (the last one shorter when W % 8 != 0); thread t owns pieces t, t + 512, ...
// and keeps rows + 1 f32 sums: for each projection j, fn[j][c] * f[c], and f[c]^2, each product
// rounded to f32 and added (no FMA), the pieces ascending and the eight values of a piece
// ascending. The 32 lanes of a wave then add by r4d_xor_sum<32> (offsets 16, 8, 4, 2, 1), and one
// thread adds the 16 waves' sums from 0.0f in wave order. A row's bytes depend on its own values
// only (not on M, the pitch or the alignment: the scalar-load form runs the same order).
#pragma once
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include "r4d_common.h"
#include "r4d_q8.h"   // q8_row_had_i8g: the next norm's rotated int8 codes (G-3b)

namespace r4d_ihc {

enum { kThreads = 512, kWaves = kThreads / 32 };

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Eight bf16 of a 16-byte word as floats.
__device__ __forceinline__ void unpack8(const uint4 v, float (&f)[8]) {
    const uint32_t w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        f[2 * k] = __uint_as_float(w[k] << 16);
        f[2 * k + 1] = __uint_as_float(w[k] & 0xffff0000u);
    }
}
__device__ __forceinline__ uint4 pack8(const uint16_t (&b)[8]) {
    return make_uint4((uint32_t)b[0] | ((uint32_t)b[1] << 16), (uint32_t)b[2] | ((uint32_t)b[3] << 16),
                      (uint32_t)b[4] | ((uint32_t)b[5] << 16), (uint32_t)b[6] | ((uint32_t)b[7] << 16));
}

// One piece's contribution: `f` the piece's `cnt` values (cnt <= 8) at flattened column `c0`.
// VF: fn's rows are 16-byte aligned with a pitch of whole float4 and the piece is whole.
template <int R, bool VF>
__device__ __forceinline__ void acc_piece(float (&acc)[R + 1], const float (&f)[8], uint32_t cnt,
                                          const float* __restrict__ fn, uint64_t fp, uint32_t c0) {
#pragma unroll
    for (int j = 0; j < R; ++j) {
        const float* __restrict__ fr = fn + (uint64_t)j * fp + c0;
        float w[8];
        if (VF) {
            const float4 a = *(const float4*)fr, b = *(const float4*)(fr + 4);
            w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w;
            w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[j] += w[e] * f[e];
        } else {
            for (uint32_t e = 0; e < cnt; ++e) acc[j] += fr[e] * f[e];
        }
    }
    if (VF) {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[R] += f[e] * f[e];
    } else {
        for (uint32_t e = 0; e < cnt; ++e) acc[R] += f[e] * f[e];
    }
}

// The lanes' sums of `acc` into `put(wave, k, sum)`: the waves [wlo, whi) this block owns (all of
// them with the row on one block, its run of them when the row is split over blocks), then the
// barrier that publishes them. Every thread calls; a wave's 32 lanes take the same branch, so the
// xor_sum sees a whole wave either way.
template <int HC, bool POST, class P>
__device__ __forceinline__ void finish_sums(const float (&acc)[(POST ? 2 * HC : HC) + 1], P put,
                                            int wlo, int whi) {
    constexpr int R = POST ? 2 * HC : HC;
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    if ((int)wave >= wlo && (int)wave < whi) {
#pragma unroll
        for (int k = 0; k <= R; ++k) {
            const float s = r4d_xor_sum<32>(acc[k]);
            if (lane == 0) put(wave, k, s);
        }
    }
    __syncthreads();
}

// The waves' sums of the row (through `pread(wave, k)`) into r, the gates (into `gate`) and the
// write weights (into `post_row` when POST); one thread forms them. Every thread calls; a barrier
// on entry to the reads and one behind the gates, so `gate` is readable on return.
template <int HC, bool POST, class P>
__device__ __forceinline__ void finish_gates(float* gate, const float* __restrict__ scale,
                                             const float* __restrict__ base,
                                             float* __restrict__ post_row, uint32_t width, float eps,
                                             float heps, float pmag, P pread) {
    constexpr int R = POST ? 2 * HC : HC;
    if (threadIdx.x == 0) {
        float s[R + 1];
#pragma unroll
        for (int k = 0; k <= R; ++k) {
            float v = 0.0f;
            for (int w = 0; w < kWaves; ++w) v += pread(w, k);
            s[k] = v;
        }
        const float inv = 1.0f / sqrtf(s[R] / (float)width + eps);
#pragma unroll
        for (int j = 0; j < HC; ++j) gate[j] = sigmoid(s[j] * inv * scale[0] + base[j]) + heps;
        if constexpr (POST) {
#pragma unroll
            for (int j = 0; j < HC; ++j)
                post_row[j] = pmag * sigmoid(s[HC + j] * inv * scale[1] + base[HC + j]) + heps;
        }
    }
    __syncthreads();
}

// The row on one block: the lanes, then the waves through the block's LDS `part`.
template <int HC, bool POST>
__device__ __forceinline__ void finish(float (&acc)[(POST ? 2 * HC : HC) + 1],
                                       float (*part)[(POST ? 2 * HC : HC) + 1], float* gate,
                                       const float* __restrict__ scale, const float* __restrict__ base,
                                       float* __restrict__ post_row, uint32_t width, float eps,
                                       float heps, float pmag) {
    finish_sums<HC, POST>(acc, [&](uint32_t w, int k, float v) { part[w][k] = v; }, 0, kWaves);
    finish_gates<HC, POST>(gate, scale, base, post_row, width, eps, heps, pmag,
                           [&](int w, int k) { return part[w][k]; });
}

// x[i] = bf16(sum_j gate[j] * f32(h[j][i])), j ascending, for the 8-column groups q = t, t + 512,
// ... of the row (n % 8 == 0, rows 16-byte aligned). `store(q, word)` takes each group's word.
template <int HC, class S>
__device__ __forceinline__ void collapse8(const uint16_t* __restrict__ hr, const float* gate,
                                          uint32_t n, S store) {
    for (uint32_t q = threadIdx.x; q < n / 8u; q += kThreads) {
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = 0.0f;
#pragma unroll
        for (int j = 0; j < HC; ++j) {
            float f[8];
            unpack8(*(const uint4*)(hr + (uint32_t)j * n + 8u * q), f);
            const float g = gate[j];
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] += g * f[e];
        }
        uint16_t b[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) b[e] = f32_to_bf16(v[e]);
        store(q, pack8(b));
    }
}

// ------------------------------------------------------------------ the fused junction's row
// `[add] -> [all_reduce] -> rmsnorm -> ihc_post -> [ihc_pre -> rmsnorm]` over one token row, at
// kThreads threads, the unfused rows' bytes:
//
//   * both norms are rmsnorm_bf16's (r4d_model_bf16.hip): the sum of squares in kR4dSumLanes'
//     partition (r4d_lanes_sumsq), the wave butterfly, the waves from 0.0f in order (norm_sum, a
//     copy of its block_sum: the waves past the eighth add 0.0f there and nothing here), rsqrtf,
//     then `x * s * (g + wadd) * mult` per element (wadd 0 and mult 1 passed at run time, as the
//     standalone norm gets them), rounded once. The norm reads its input as stored: the raw row
//     here is held in LDS as the bf16 the chain stores.
//   * ihc_post's `bf16(post[j] * y + h)` with the product rounded first (__fmul_rn / __fadd_rn);
//   * ihc_pre's order above, over the h just written (from the same registers);
//   * the collapse x rounded to bf16 into LDS and normed from there.
//
// The row's input (`a`, or `a + b`, or the reduced sum) is in `ybuf` (n bf16, 16-byte aligned) on
// entry, each thread having written its own 8-column groups; `ybuf` ends holding x (TAIL) or y.
template <int HC, bool POST>
struct JuncLds {
    float red1[kWaves], red2[kWaves];
    float part[kWaves][(POST ? 2 * HC : HC) + 1];
    float gate[HC], pv[HC];
};

// rmsnorm_bf16's block_sum: the wave butterfly, then the waves from 0.0f in order.
__device__ __forceinline__ float norm_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < (uint32_t)kWaves; ++w) s += lds[w];
    return s;
}

// The norm of the bf16 row in `buf` with the f32 gain `g`, word by word to `put(q, word)`.
template <class P>
__device__ __forceinline__ void norm_row(const uint16_t* buf, const float* __restrict__ g,
                                         uint32_t n, float eps, float wadd, float mult, float* red,
                                         P put) {
    float ss = r4d_lanes_sumsq(buf, n, [](uint16_t v) { return bf16_to_f32(v); });
    ss = norm_sum(ss, red);
    const float s = rsqrtf(ss / (float)n + eps);
    for (uint32_t q = threadIdx.x; q < n / 8u; q += kThreads) {
        float f[8];
        unpack8(*(const uint4*)(buf + 8u * q), f);
        const float4 g0 = *(const float4*)(g + 8u * q), g1 = *(const float4*)(g + 8u * q + 4u);
        const float gv[8] = { g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w };
        uint16_t b[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) b[e] = f32_to_bf16(f[e] * s * (gv[e] + wadd) * mult);
        put(q, pack8(b));
    }
}

// Passed by value as a kernel argument: no padding anywhere (the pointers first, the 32-bit
// fields filling 32 bytes), so the argument bytes are the fields' and a recorded pass's audit
// (core/runtime/ctx.cpp tapes_agree) compares the same bytes every step.
struct JuncRow {
    const float* w_post;
    float* post_row;          // [hc] f32: read (this connection), written (the next one's, POST && TAIL)
    uint16_t* hr;             // [hc * n] the streams, in place
    const float* fn;          // [rows][hc * n] at pitch fp
    uint64_t fp;
    const float *scale, *base, *w_next;
    uint16_t* xnr;            // [n] the next block's normed input (TAIL)
    uint16_t* yr;             // [n] the post norm's output, when asked
    uint16_t* xr;             // [n] the collapse, when asked
    signed char* cq;          // [n] the next block's input norm's rotated int8 codes (TAIL), when asked
    float* cs;                // [n / 128] their scales
    uint64_t cqp, csp;        // the row pitches of cq and cs
    uint32_t n;
    float eps, ieps, heps, pmag, wadd, mult;
    uint32_t pad_ = 0;
};
static_assert(sizeof(JuncRow) == 15 * 8 + 8 * 4, "JuncRow carries no padding");

// The row's first two stages over the waves [wlo, whi): 1. y = rmsnorm(input) back into ybuf
//    (and `yr` when asked), 2. h += post * y and the next connection's sums in `acc` (TAIL).
// A thread's words and pieces are its own index's whatever wlo/whi are, so a block that owns a run
// of the waves writes and sums exactly what the same thread index did with the row on one block:
// only the writes are guarded, and ybuf is filled whole either way (a piece reads the words of the
// waves it passes over). `owns` is constant true for a row on one block.
template <int HC, bool POST, bool TAIL>
__device__ __forceinline__ void junction_pieces(JuncLds<HC, POST>& L, uint16_t* ybuf,
                                                const JuncRow& r, int wlo, int whi,
                                                float (&acc)[(POST ? 2 * HC : HC) + 1]) {
    constexpr int R = POST ? 2 * HC : HC;
    const uint32_t t = threadIdx.x, n = r.n;
    const bool owns = (int)(t >> 5) >= wlo && (int)(t >> 5) < whi;
    if (t < (uint32_t)HC) L.pv[t] = r.post_row[t];
    __syncthreads();   // the input row in ybuf (the caller's copy) and pv
    norm_row(ybuf, r.w_post, n, r.eps, r.wadd, r.mult, L.red1, [&](uint32_t q, uint4 w) {
        *(uint4*)(ybuf + 8u * q) = w;
        if (r.yr && owns) *(uint4*)(r.yr + 8u * q) = w;
    });
    __syncthreads();
#pragma unroll
    for (int k = 0; k <= R; ++k) acc[k] = 0.0f;
    const uint32_t pieces = (uint32_t)HC * n / 8u;
    if (owns) {   // no early exit: the barriers of finish_sums and the tail are every thread's
        for (uint32_t p = t; p < pieces; p += kThreads) {
            const uint32_t c0 = 8u * p, j = c0 / n, i0 = c0 - j * n;
            float hf[8], yf[8], f[8];
            unpack8(*(const uint4*)(r.hr + c0), hf);
            unpack8(*(const uint4*)(ybuf + i0), yf);
            const float pj = L.pv[j];
            uint16_t b[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                b[e] = f32_to_bf16(__fadd_rn(__fmul_rn(pj, yf[e]), hf[e]));
                f[e] = bf16_to_f32(b[e]);
            }
            *(uint4*)(r.hr + c0) = pack8(b);
            if (TAIL) acc_piece<R, true>(acc, f, 8u, r.fn, r.fp, c0);
        }
    }
}

// The row's tail (TAIL): the collapse into ybuf, its norm, and the quantiser's codes from it.
template <int HC, bool POST>
__device__ __forceinline__ void junction_tail(JuncLds<HC, POST>& L, uint16_t* ybuf, const JuncRow& r) {
    const uint32_t n = r.n;
    collapse8<HC>(r.hr, L.gate, n, [&](uint32_t q, uint4 w) {
        *(uint4*)(ybuf + 8u * q) = w;
        if (r.xr) *(uint4*)(r.xr + 8u * q) = w;
    });
    __syncthreads();
    norm_row(ybuf, r.w_next, n, r.eps, r.wadd, r.mult, L.red2, [&](uint32_t q, uint4 w) {
        *(uint4*)(r.xnr + 8u * q) = w;
        if (r.cq) *(uint4*)(ybuf + 8u * q) = w;   // each thread its own words, read before
    });
    // G-3b: had_quant_act_i8g's codes over the bf16 row just stored (q8_row_had_i8g, a wave a
    // 128-column group), so the consumers of xn read them and no quantiser follows.
    if (r.cq) {
        __syncthreads();
        q8_row_had_i8g<kThreads>(ybuf, r.cq, r.cs, n);
    }
}

template <int HC, bool POST, bool TAIL>
__device__ __forceinline__ void junction_row(JuncLds<HC, POST>& L, uint16_t* ybuf, const JuncRow& r) {
    constexpr int R = POST ? 2 * HC : HC;
    float acc[R + 1];
    junction_pieces<HC, POST, TAIL>(L, ybuf, r, 0, kWaves, acc);
    if constexpr (TAIL) {
        // 3. gates and post (finish's barriers also publish h to the block), the collapse into
        //    ybuf, its norm.
        finish<HC, POST>(acc, L.part, L.gate, r.scale, r.base, r.post_row, (uint32_t)HC * r.n,
                         r.ieps, r.heps, r.pmag);
        junction_tail<HC, POST>(L, ybuf, r);
    }
}

// ------------------------------------------------- K-13: the row's waves over several blocks
// The piece sums are per wave (finish_sums: a wave's 32 lanes, then the waves ascending), so a row
// divides over blocks by whole waves without moving a bit: block `part` of `parts` owns the waves
// [part * kWaves / parts, (part + 1) * kWaves / parts) and, through junction_pieces, exactly the
// words and pieces the same thread index handled with the row on one block. The blocks of a row
// meet at two words a (launch parity, row): `folded`, which the block that pushed and folded the
// row sets so that the others may read the reduced row, and `arrive`, which every block adds to
// once its h and its waves' sums are out, whose last arriver finishes the row (the gates, the
// collapse, the norm and the quantiser's codes) and resets both. Both are zero at a launch's start:
// zeroed at allocation, reset by the row's last arriver, and two launches in flight use different
// parities (the wire's own rule).
enum { kSplitRows = 8, kSplitParts = 4, kSplitStride = 2 * 8 + 1 };   /* 17: the widest R + 1 */
struct JuncSplit {
    float part[2][kSplitRows][kWaves][kSplitStride];   // the waves' sums, R + 1 a wave
    unsigned int arrive[2][kSplitRows];
    unsigned int folded[2][kSplitRows];
};

// One row over `parts` blocks: the stages of junction_row, the last arriver taking the tail. `part`
// is this row's scratch (its 16 waves' sums), `arrive` and `folded` its two words.
template <int HC, bool POST, bool TAIL>
__device__ __forceinline__ void junction_row_split(JuncLds<HC, POST>& L, uint16_t* ybuf,
                                                   const JuncRow& r, int wlo, int whi, int parts,
                                                   float* part, unsigned int* arrive,
                                                   unsigned int* folded) {
    constexpr int R = POST ? 2 * HC : HC;
    const uint32_t t = threadIdx.x;
    float acc[R + 1];
    junction_pieces<HC, POST, TAIL>(L, ybuf, r, wlo, whi, acc);
    if constexpr (TAIL) {
        // 3. my waves' sums straight out to the launch's scratch.
        finish_sums<HC, POST>(acc, [&](uint32_t w, int k, float v) {
            part[(uint32_t)w * kSplitStride + (uint32_t)k] = v;
        }, wlo, whi);
    }
    // My h and my waves' sums device-wide before my arrival; the row's last block takes it on (the
    // recurrent update's paired workgroups do the same).
    __threadfence();
    __syncthreads();
    __shared__ unsigned int s_last;
    if (t == 0)
        s_last = __hip_atomic_fetch_add(arrive, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u ==
                 (unsigned int)parts;
    __syncthreads();
    if (!s_last) return;
    if (t == 0) {
        __hip_atomic_store(arrive, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(folded, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __threadfence();
    if constexpr (TAIL) {
        if (t < (uint32_t)(kWaves * (R + 1))) {   // the row's sums into LDS, one a thread
            const uint32_t w = t / (uint32_t)(R + 1), k = t - w * (uint32_t)(R + 1);
            L.part[w][k] = part[w * kSplitStride + k];
        }
        __syncthreads();
        finish_gates<HC, POST>(L.gate, r.scale, r.base, r.post_row, (uint32_t)HC * r.n, r.ieps,
                               r.heps, r.pmag, [&](int w, int k) { return L.part[w][k]; });
        junction_tail<HC, POST>(L, ybuf, r);
    }
}

// A junction launch's operands, checked (r4d_ihc_junction_args, r4d_ihc_bf16.hip): the plain row
// and the all-reduce twin read the same list.
struct R4dIhcJunction {
    uint16_t* a = nullptr;
    const uint16_t* b = nullptr;
    JuncRow r{};
    long long M = 0, ap = 0, bp = 0, pp = 0, hp = 0, xnp = 0, yp = 0, xp = 0;
    int hc = 0;
    bool post = false, tail = false;
};

}  // namespace r4d_ihc

struct RadArgs;
/* RAD_OK or the RAD_E_* the operands earn; `o` filled on RAD_OK. */
int r4d_ihc_junction_args(const RadArgs* a, r4d_ihc::R4dIhcJunction* o);
