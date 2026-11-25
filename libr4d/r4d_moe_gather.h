// r4d_moe_gather.h -- moe_gather's kernel for decode and short passes, beside its launcher in
// r4d_moe.hip; tests/r4d_emu runs it from here against the inverse the sort writes.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

#include "r4d_common.h"
#include "r4d_fwht.h"

// Internal linkage, as the unit's own definition had.
#include "r4d_chain.h"

namespace r4d_moegather {
namespace {

/* The slot table is hoisted: `inv[t * top_k + j]` and `ew[t * ew_ld + j]` do not depend on the
 * column, and read inside the column loop each `inv` load would serialise the `ye` load behind it
 * (110 dependent latencies per layer at decode). Read once into LDS, the `ye` loads pipeline.
 *
 * The column range is also split over blocks, since decode has one token. Neither change touches
 * the sum: each column's slots are still summed in ascending order within one block. */
constexpr uint32_t kGatherSlots = 64;
/* When small, the inverse permutation is built in this kernel and the two prepass kernels are not
 * launched. A block owns one token, so it scans `sorted` for entries in [t*top_k, (t+1)*top_k),
 * giving exactly the table `inv` would hold. Every block repeats the scan, so the launcher caps it
 * (r4d_rad_moe_gather); at decode T = 40 and the scan is one pass, against two dependent dispatches
 * (~2.1 us each) of single-block kernels.
 *
 * `sorted == nullptr` selects the prepass path, used at prefill. */
constexpr uint32_t kGatherSelfT = 512;
/* The sixteen-slot form's columns a lane: one sixteen-byte load a slot (a `uint4` is eight bf16),
 * against the eight two-byte loads the scalar form issues for the same eight columns. */
constexpr uint32_t kGatherCols = 8;

template <bool F32>
__device__ float gather_weight(const void* ew, size_t i) {
    if constexpr (F32) return ((const float*)ew)[i];
    else return bf16_to_f32(((const uint16_t*)ew)[i]);
}
/* One column's slot folded in, in the form the item keeps: `fmaf` for the f32 scale, the same
 * multiply-add for bf16. Both forms below read it, so a column's sum is the same either way. */
template <bool F32>
__device__ __forceinline__ float gather_mac(float wj, float x, float acc) {
    if constexpr (F32) return fmaf(wj, x, acc);
    else return acc + wj * x;
}
/* The quantising form's epilogue (moe_gather_inv_i8g, K-19): the eight columns a lane has just
 * summed, as the rotated int8 codes and the scale the linear reading them needs. It is
 * quant_act_fp8_body<ROT, I8> (r4d_quant_act_fp8.hip) over those columns, from the values instead
 * of from a re-read: the rotation stays 128 wide (r4d_fwht_part<8, 16> is r4d_fwht_wave<128> over
 * sixteen lanes, r4d_fwht.h), the amax covers the scale group (`g8`: 128 columns, or 256 for an
 * IQ8R trunk, whose two 128-wide blocks share a scale), the scale is
 * amax/127 and the codes are round-to-nearest into -127..127. Same rotation, same sum order and
 * same rounding, so the codes are byte for byte the standalone quantiser's.
 *
 * `c` is the lane's first column (a multiple of eight) and `lane` its hardware lane: the sixteen
 * lanes of a 128-wide block are a contiguous column range, starting on a block boundary, so the
 * wave's lane `c` holds its 8 columns of the sum row, `q` is already at the row's column `c` and
 * `s` at the row's first scale. */
__device__ __forceinline__ void gather_quant(const uint32_t o[4], uint32_t lane, uint32_t c,
                                            uint32_t g8, uint8_t* __restrict__ q,
                                            float* __restrict__ s) {
    float v[kGatherCols];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        v[2 * i]     = bf16_to_f32((uint16_t)(o[i] & 0xFFFFu));
        v[2 * i + 1] = bf16_to_f32((uint16_t)(o[i] >> 16));
    }
    r4d_fwht_part<8, 16>(v, (int)lane);
    float m = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kGatherCols; ++j) m = fmaxf(m, fabsf(v[j]));
    /* A wave holds two 128-wide blocks, so one lane a block writes its block's scale; a 256-wide
     * group is both blocks, whose amax and scale are the wave's. */
    const bool wide = g8 > 128;
    m = wide ? r4d_wave_max(m) : r4d_xor_max<16>(m);
    const float sc = m > 0.0f ? m * (1.0f / 127.0f) : 1.0f;
    if ((lane & (wide ? 31u : 15u)) == 0) s[c / g8] = sc;
    const float inv = 1.0f / sc;
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        uint32_t b = 0;
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const int code = (int)rintf(fminf(fmaxf(v[4 * i + j] * inv, -127.0f), 127.0f));
            b |= ((uint32_t)code & 0xFFu) << (8 * j);
        }
        *(uint32_t*)(q + 4 * i) = b;
    }
}
/* The body, over the block's two shared tables, which the kernel that calls it owns: a function
 * because a block past the last token returns at once, and a chained block must still add to its
 * queue's counter (r4d_chain.h)).
 *
 * `TOPK` fixes top_k at compile time (0: the argument's), and `COLS` is the columns a lane: 1 for
 * the scalar form, `kGatherCols` for the sixteen-slot one. The launcher passes TOPK = 16 only for
 * top_k = 16, so both read the same table; the sums are unchanged -- a column's slots still go in
 * ascending order, and the shared arm's two narrows are the scalar form's. */
#define R4D_MG_PARAMS                                                                            \
    const uint16_t* __restrict__ ye, const void* __restrict__ ew,                                \
    const int32_t* __restrict__ inv, uint16_t* __restrict__ y, uint32_t M, uint32_t n,           \
    uint32_t top_k, uint32_t ye_ld, uint32_t ew_ld, uint32_t y_ld,                               \
    const int32_t* __restrict__ sorted, uint32_t T, const uint16_t* __restrict__ sh,             \
    const uint16_t* __restrict__ sg, uint32_t sh_ld, uint32_t sg_ld, int sig
#define R4D_MG_ARGS                                                                              \
    ye, ew, inv, y, M, n, top_k, ye_ld, ew_ld, y_ld, sorted, T, sh, sg, sh_ld, sg_ld, sig
template <uint32_t NTHR, bool F32, uint32_t TOPK = 0, uint32_t COLS = 1, bool Q = false>
__device__ __forceinline__ void gather_body(R4D_MG_PARAMS, int32_t* s_i, float* s_w,
                                            uint8_t* __restrict__ q8 = nullptr,
                                            float* __restrict__ s8 = nullptr,
                                            uint32_t q_ld = 0, uint32_t s_ld = 0,
                                            uint32_t g8 = 128) {
    const uint32_t t = blockIdx.x;
    if (t >= M) return;
    const uint32_t k = TOPK ? TOPK : top_k;
    const bool cached = k <= kGatherSlots;
    /* The shared arm's scalar, read once per block as scale_rows reads it once per row. */
    float gate = 0.0f;
    if (sh) {
        const float v0 = bf16_to_f32(sg[(size_t)t * sg_ld]);
        gate = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    }
    if (sorted) {
        /* An unclaimed slot must read -1, so fill and scan are separated by a barrier (as the
         * prepass is two kernels). */
        for (uint32_t j = threadIdx.x; j < k; j += NTHR) s_i[j] = -1;
        __syncthreads();
        const int32_t lo = (int32_t)(t * k), hi = lo + (int32_t)k;
        for (uint32_t i = threadIdx.x; i < T; i += NTHR) {
            const int32_t f = sorted[i];
            if (f >= lo && f < hi) s_i[f - lo] = (int32_t)i;
        }
        for (uint32_t j = threadIdx.x; j < k; j += NTHR)
            s_w[j] = gather_weight<F32>(ew, (size_t)t * ew_ld + j);
        __syncthreads();
    } else if (cached) {
        for (uint32_t j = threadIdx.x; j < k; j += NTHR) {
            s_i[j] = inv[t * k + j];
            s_w[j] = gather_weight<F32>(ew, (size_t)t * ew_ld + j);
        }
        __syncthreads();
    }
    if constexpr (COLS == 1) {
        for (uint32_t c = blockIdx.y * NTHR + threadIdx.x; c < n; c += NTHR * gridDim.y) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < k; ++j) {
                const int32_t i = cached ? s_i[j] : inv[t * k + j];
                if (i < 0) continue;
                const float wj = cached ? s_w[j] : gather_weight<F32>(ew, (size_t)t * ew_ld + j);
                acc = gather_mac<F32>(wj, bf16_to_f32(ye[(size_t)i * ye_ld + c]), acc);
            }
            const uint16_t g = f32_to_bf16(acc);
            if (!sh) { y[(size_t)t * y_ld + c] = g; continue; }
            /* The shared arm, folded with the unfused pair's two narrows: `scale_rows` reads the
             * gather as bf16 and narrows its own product before adding. */
            const uint16_t p = f32_to_bf16(bf16_to_f32(sh[(size_t)t * sh_ld + c]) * gate);
            y[(size_t)t * y_ld + c] = f32_to_bf16(bf16_to_f32(p) + bf16_to_f32(g));
        }
    } else {
        /* Eight columns a lane and one sixteen-byte load a slot: every slot's group is issued
         * before any is used (as the prefill form does), and each column still sums its slots in
         * ascending order. `n` is a multiple of COLS here and the rows are sixteen-byte aligned,
         * both checked by the launcher. */
        static_assert(COLS == kGatherCols && COLS == 8, "the eight-column form is a uint4");
        const uint32_t ng = n / COLS;
        for (uint32_t g = blockIdx.y * NTHR + threadIdx.x; g < ng; g += NTHR * gridDim.y) {
            const uint32_t c = g * COLS;
            /* `k` slots, each one load, then the columns' sums in the slots' order. */
            static_assert(TOPK != 0, "the eight-column form has top_k fixed at compile time");
            bool  lv[TOPK];
            float wj[TOPK];
            uint4 yv[TOPK];
#pragma unroll
            for (uint32_t j = 0; j < TOPK; ++j) {
                const int32_t i = cached ? s_i[j] : inv[t * k + j];
                lv[j]    = i >= 0;
                wj[j]    = cached ? s_w[j] : gather_weight<F32>(ew, (size_t)t * ew_ld + j);
                yv[j]    = make_uint4(0, 0, 0, 0);
                if (lv[j]) yv[j] = *(const uint4*)(ye + (size_t)i * ye_ld + c);
            }
            const uint4 shv = sh ? *(const uint4*)(sh + (size_t)t * sh_ld + c) : make_uint4(0, 0, 0, 0);
            float acc[COLS];
#pragma unroll
            for (uint32_t q = 0; q < COLS; ++q) acc[q] = 0.0f;
#pragma unroll
            for (uint32_t j = 0; j < TOPK; ++j) {
                if (!lv[j]) continue;
                const uint32_t d[4] = { yv[j].x, yv[j].y, yv[j].z, yv[j].w };
#pragma unroll
                for (uint32_t q = 0; q < 4; ++q) {
                    acc[2 * q]     = gather_mac<F32>(wj[j], bf16_to_f32((uint16_t)(d[q] & 0xFFFFu)),
                                                     acc[2 * q]);
                    acc[2 * q + 1] = gather_mac<F32>(wj[j], bf16_to_f32((uint16_t)(d[q] >> 16)),
                                                     acc[2 * q + 1]);
                }
            }
            const uint32_t s[4] = { shv.x, shv.y, shv.z, shv.w };
            uint32_t o[4];
#pragma unroll
            for (uint32_t q = 0; q < 4; ++q) {
                uint16_t lo = f32_to_bf16(acc[2 * q]), hi = f32_to_bf16(acc[2 * q + 1]);
                if (sh) {
                    const uint16_t plo = f32_to_bf16(bf16_to_f32((uint16_t)(s[q] & 0xFFFFu)) * gate);
                    const uint16_t phi = f32_to_bf16(bf16_to_f32((uint16_t)(s[q] >> 16)) * gate);
                    lo = f32_to_bf16(bf16_to_f32(plo) + bf16_to_f32(lo));
                    hi = f32_to_bf16(bf16_to_f32(phi) + bf16_to_f32(hi));
                }
                o[q] = (uint32_t)lo | ((uint32_t)hi << 16);
            }
            *(uint4*)(y + (size_t)t * y_ld + c) = make_uint4(o[0], o[1], o[2], o[3]);
            /* The same sums, as the codes a following int8 linear reads (K-19). */
            if constexpr (Q)
                gather_quant(o, threadIdx.x & 31u, c, g8, q8 + (size_t)t * q_ld + c,
                             s8 + (size_t)t * s_ld);
        }
    }
}
template <uint32_t NTHR, bool F32 = false>
__global__ __launch_bounds__(NTHR) void gather_kernel(R4D_MG_PARAMS) {
    __shared__ int32_t s_i[kGatherSlots];
    __shared__ float   s_w[kGatherSlots];
    gather_body<NTHR, F32>(R4D_MG_ARGS, s_i, s_w);
}
/* Sixteen slots and eight columns a lane: the decode form where the launcher's shape check allows
 * it, and the same sums as `gather_kernel` (r4d_selftest's byte compare, tests/r4d_emu). */
template <uint32_t NTHR, bool F32 = false>
__global__ __launch_bounds__(NTHR) void gather_kernel16(R4D_MG_PARAMS) {
    __shared__ int32_t s_i[kGatherSlots];
    __shared__ float   s_w[kGatherSlots];
    gather_body<NTHR, F32, 16, kGatherCols>(R4D_MG_ARGS, s_i, s_w);
}
/* The quantising sixteen-slot form (moe_gather_inv_i8g): the same sums and the same `y` store, and
 * the rotated int8 codes and scales of the row it sums, so `lat_out`'s quantiser launch goes away.
 * The launcher takes it only where `n` is a whole number of 256-column waves (a wave is two
 * 128-wide rotation blocks) and top_k is sixteen. */
template <uint32_t NTHR, bool F32 = false>
__global__ __launch_bounds__(NTHR) void gather_kernel16_q8(R4D_MG_PARAMS,
                                                          uint8_t* __restrict__ q8,
                                                          float* __restrict__ s8,
                                                          uint32_t q_ld, uint32_t s_ld,
                                                          uint32_t g8) {
    __shared__ int32_t s_i[kGatherSlots];
    __shared__ float   s_w[kGatherSlots];
    gather_body<NTHR, F32, 16, kGatherCols, true>(R4D_MG_ARGS, s_i, s_w, q8, s8, q_ld, s_ld, g8);
}
/* The chained form, taken where the gather is the op's one launch (a decode step). */
template <uint32_t NTHR, bool F32 = false>
__global__ __launch_bounds__(NTHR) void r4dchain_moe_gather(R4D_MG_PARAMS) {
    __shared__ int32_t s_i[kGatherSlots];
    __shared__ float   s_w[kGatherSlots];
    r4d_chain_wait();
    gather_body<NTHR, F32>(R4D_MG_ARGS, s_i, s_w);
    r4d_chain_done();
}
template <uint32_t NTHR, bool F32 = false>
__global__ __launch_bounds__(NTHR) void r4dchain_moe_gather16(R4D_MG_PARAMS) {
    __shared__ int32_t s_i[kGatherSlots];
    __shared__ float   s_w[kGatherSlots];
    r4d_chain_wait();
    gather_body<NTHR, F32, 16, kGatherCols>(R4D_MG_ARGS, s_i, s_w);
    r4d_chain_done();
}
#undef R4D_MG_ARGS
#undef R4D_MG_PARAMS

}  // namespace
}  // namespace r4d_moegather
