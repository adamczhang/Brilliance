// R4D grouped GEMM over NVFP4 experts against an int8 activation: moe_gemm_q at dtype nvfp4a8i8.
//
// An NVFP4 expert is e2m1 codes, an E4M3 scale per (row, 16 of K) and one f32 level for the whole
// expert (amax / denominator). Its value is e2m1 * e4m3 * level. The e2m1 magnitudes are {0, 0.5,
// 1, 1.5, 2, 3, 4, 6}, so twice them are the integers {0, 1, 2, 3, 4, 6, 8, 12}: an int8 activation
// multiplies those exactly, and the 1/2 is folded into the level. Each 16-wide group of K is one
// int32 sum (at most 16 * 12 * 127 in magnitude), folded to f32 under its scale and the
// activation's scale for that group.
//
// Two forms serve the shape:
//   - Decode (T <= 64, or shapes outside prefill alignment): 8 lanes along K per output column
//     (8 columns / 64 threads per block), coalesced uint2 code reads + int4 activation reads
//     with lookahead prefetch and v_dot4_i32_iu8 (`__builtin_amdgcn_sudot4`) + 3-XOR reduction.
//   - Prefill (T > 64, each class's part width N_q / parts a multiple of the column tile, K0/K1
//     multiples of 128): under the run-aligned map at a_group 128, a Direct-B f16 WMMA kernel
//     (64x256 over 8 waves or 64x128 over 4 waves, transposed (ny, map) launch grid, mid-loop
//     prefetch, carried row scales) converts int8 activations into f16 in LDS and unpacks packed
//     e2m1*e4m3 weights into exact f16 fragments via LUT permutes (`v_perm_b32`) + `v_pk_mul_f16`,
//     accumulating all 16-K groups directly in `v_wmma_f32_16x16x16_f16`. Other shapes fall back
//     to the 32-row LDS-staged iu8 WMMA path. When scratch is provided
//     (`r4d_rad_moe_gemm_nvfp4_scratch`), a single-block parallel prefix scan builds the run-aligned
//     block map (`nvfp4_blockmap_kernel`); otherwise a uniform grid runs the fallback WMMA body.
//
// Operands and parameters are moe_gemm_q's (r4d_rows.cpp, `moe_gemm_nvfp4a8i8`): `a` int8 [M, K]
// with an f32 scale per (row, a_group), `w` the even experts' e2m1 codes [E, N, K] (or a pointer
// table, stride 0), `w_scale` their stored scale blobs (r4d_layout.cpp, nvfp4_layout), and the
// odd class's `w_odd` and `w_odd_scale` at N_odd and K_odd.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_fp8_wmma.h"
#include "r4d_plugin.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

namespace {

#define R4D_NV_CHK() do { hipError_t e_ = hipGetLastError();                       \
                          if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

constexpr int      kNvDecCols     = 16;
constexpr int      kNvDecLanes    = 8;
constexpr int      kNvDecBlock    = kNvDecCols * kNvDecLanes;   /* 128 threads */
constexpr int      kNvMdecRM      = 8;                           /* rows per grouped decode/verify tile */

constexpr uint32_t kNvPfM32       = 32;
constexpr uint32_t kNvPfM64       = 64;
constexpr uint32_t kNvPfN64       = 64;
constexpr uint32_t kNvPfN128      = 128;
constexpr uint32_t kNvPfN256      = 256;
constexpr uint32_t kNvPfK         = 128;
constexpr long long kNvPrefillMinT = 64;
constexpr long long kNvMaxExpert   = 512;
constexpr uint32_t  kNvMapTail     = 0xffffffffu;

static inline bool nv_pick_prefill(long long T, long long n_expert, long long parts, long long N0,
                                   long long N1, long long K0, long long K1, uint32_t* pfM,
                                   uint32_t* pfN) {
    (void)n_expert;
    /* A column tile lies inside one stacked part of every class (gate or up of a split gate_up), so
     * the part widths, not the whole widths, must be whole tiles. */
    if (T <= kNvPrefillMinT || parts <= 0 || N0 % parts || N1 % parts || (K0 % kNvPfK) != 0 ||
        (K1 % kNvPfK) != 0)
        return false;
    N0 /= parts;
    N1 /= parts;
    if ((N0 % kNvPfN256) == 0 && (N1 % kNvPfN256) == 0) {
        *pfM = kNvPfM32;
        *pfN = kNvPfN256;
        return true;
    }
    if ((N0 % kNvPfN128) == 0 && (N1 % kNvPfN128) == 0) {
        *pfM = kNvPfM32;
        *pfN = kNvPfN128;
        return true;
    }
    if ((N0 % kNvPfN64) == 0 && (N1 % kNvPfN64) == 0) {
        *pfM = kNvPfM32;
        *pfN = kNvPfN64;
        return true;
    }
    return false;
}

/* One operand's experts: `data` the stacked base when `s0` is non-zero (bytes an expert), else the
 * pointer table of the experts' planes. */
struct NvTab {
    const uint8_t* data;
    size_t         s0;
};
__device__ __forceinline__ const uint8_t* nv_entry(NvTab t, uint32_t e) {
    if (t.s0 == 0) return ((const uint8_t* const*)t.data)[e];
    return t.data + (size_t)e * t.s0;
}

#if defined(__HIP__) && __has_builtin(__builtin_amdgcn_perm)
/* Expand 4 low e2m1 nibbles in the low nibble of each byte of `nib` to doubled signed int8 via v_perm_b32. */
__device__ __forceinline__ uint32_t nv_perm4_i8(uint32_t nib) {
    const uint32_t m   = nib & 0x07070707u;
    const uint32_t lo  = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, m);
    const uint32_t hi  = __builtin_amdgcn_perm(0xF4F8FAFCu, 0xFDFEFF00u, m);
    const uint32_t sel = 0x03020100u + ((nib & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(hi, lo, sel);
}

/* Unpack 8 low-to-high e2m1 nibbles in `w` into two packed 4xint8 words `p0` (k0..3), `p1` (k4..7). */
__device__ __forceinline__ void nv_unpack8_i8(uint32_t w, int32_t& p0, int32_t& p1) {
    const uint32_t ev = nv_perm4_i8(w & 0x0F0F0F0Fu);
    const uint32_t od = nv_perm4_i8((w >> 4) & 0x0F0F0F0Fu);
    p0 = (int32_t)__builtin_amdgcn_perm(od, ev, 0x05010400u);
    p1 = (int32_t)__builtin_amdgcn_perm(od, ev, 0x07030602u);
}
#else
/* e2m1's magnitudes doubled as signed int8: {0, 1, 2, 3, 4, 6, 8, 12} with bit 3 as sign. */
__device__ __forceinline__ int8_t nv_nib4_to_i8(uint32_t c) {
    constexpr uint64_t kMag2 = 0x0C08060403020100ULL;
    const int32_t v = (int32_t)((kMag2 >> ((c & 7u) * 8u)) & 0xFFu);
    return (int8_t)((c & 8u) ? -v : v);
}

__device__ __forceinline__ uint32_t nv_nib4_to_u8(uint32_t c) {
    return (uint32_t)(uint8_t)nv_nib4_to_i8(c);
}

/* Unpack 8 low-to-high e2m1 nibbles in `w` into two packed 4xint8 words `p0` (k0..3), `p1` (k4..7). */
__device__ __forceinline__ void nv_unpack8_i8(uint32_t w, int32_t& p0, int32_t& p1) {
    const uint32_t u0 = nv_nib4_to_u8(w)
                      | (nv_nib4_to_u8(w >> 4)  << 8)
                      | (nv_nib4_to_u8(w >> 8)  << 16)
                      | (nv_nib4_to_u8(w >> 12) << 24);
    const uint32_t u1 = nv_nib4_to_u8(w >> 16)
                      | (nv_nib4_to_u8(w >> 20) << 8)
                      | (nv_nib4_to_u8(w >> 24) << 16)
                      | (nv_nib4_to_u8(w >> 28) << 24);
    p0 = (int32_t)u0;
    p1 = (int32_t)u1;
}
#endif

/* Signed 4xint8 dot product accumulated into `c`: `v_dot4_i32_iu8` on gfx12, scalar on host emu. */
__device__ __forceinline__ int32_t nv_sdot4(int32_t a, int32_t b, int32_t c) {
#if defined(__HIP__) && __has_builtin(__builtin_amdgcn_sudot4)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    int32_t acc = c;
    for (int i = 0; i < 4; ++i) {
        const int8_t ai = (int8_t)((uint32_t)a >> (8 * i));
        const int8_t bi = (int8_t)((uint32_t)b >> (8 * i));
        acc += (int32_t)ai * (int32_t)bi;
    }
    return acc;
#endif
}

__device__ __forceinline__ int4 nv_load_i4(const void* p) {
#ifdef __HIP__
    return *(const int4*)p;
#else
    int4 v;
    __builtin_memcpy(&v, p, sizeof v);
    return v;
#endif
}

#ifndef __HIP__
__device__ __forceinline__ uint2 nv_load_u2(const void* p) {
    uint2 v;
    __builtin_memcpy(&v, p, sizeof v);
    return v;
}

__device__ __forceinline__ uint4 nv_load_u4(const void* p) {
    uint4 v;
    __builtin_memcpy(&v, p, sizeof v);
    return v;
}

__device__ __forceinline__ r4d_u32x4 nv_load_v4(const void* p) {
    r4d_u32x4 v;
    __builtin_memcpy(&v, p, sizeof v);
    return v;
}

__device__ __forceinline__ uint64_t nv_load_u64(const void* p) {
    uint64_t v;
    __builtin_memcpy(&v, p, sizeof v);
    return v;
}
#endif

__device__ __forceinline__ uint16_t nv_load_u16(const void* p) {
    uint16_t v;
    __builtin_memcpy(&v, p, sizeof v);
    return v;
}

__device__ __forceinline__ int nv_readfirstlane(int x) {
#ifdef __HIP__
    return __builtin_amdgcn_readfirstlane(x);
#else
    return x;
#endif
}

#ifdef __HIP__
typedef unsigned r4d_nv_u4v __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) const r4d_nv_u4v* r4d_nv_gu4;
typedef unsigned r4d_nv_u2v __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(1))) const r4d_nv_u2v* r4d_nv_gu2;
typedef __attribute__((address_space(1))) const uint64_t*   r4d_nv_gu64;

template <typename P>
__device__ __forceinline__ const P* nv_uw_ptr(const P* p) {
    const uint64_t v  = (uint64_t)(uintptr_t)p;
    const uint32_t lo = (uint32_t)__builtin_amdgcn_readfirstlane((int)(uint32_t)v);
    const uint32_t hi = (uint32_t)__builtin_amdgcn_readfirstlane((int)(uint32_t)(v >> 32));
    return (const P*)(uintptr_t)(((uint64_t)hi << 32) | lo);
}
__device__ __forceinline__ r4d_u32x4 nv_gload_v4(const void* p) {
    const r4d_nv_u4v v = *(r4d_nv_gu4)p;
    return r4d_u32x4{v.x, v.y, v.z, v.w};
}
__device__ __forceinline__ uint4 nv_gload_u4(const void* p) {
    const r4d_nv_u4v v = *(r4d_nv_gu4)p;
    return make_uint4(v.x, v.y, v.z, v.w);
}
__device__ __forceinline__ uint2 nv_gload_u2(const void* p) {
    const r4d_nv_u2v v = *(r4d_nv_gu2)p;
    return make_uint2(v.x, v.y);
}
__device__ __forceinline__ uint64_t nv_gload_u64(const void* p) { return *(r4d_nv_gu64)p; }
#else
template <typename P>
__device__ __forceinline__ const P* nv_uw_ptr(const P* p) { return p; }
__device__ __forceinline__ r4d_u32x4 nv_gload_v4(const void* p) { return nv_load_v4(p); }
__device__ __forceinline__ uint4 nv_gload_u4(const void* p) { return nv_load_u4(p); }
__device__ __forceinline__ uint2 nv_gload_u2(const void* p) { return nv_load_u2(p); }
__device__ __forceinline__ uint64_t nv_gload_u64(const void* p) { return nv_load_u64(p); }
#endif

/* One output element: sorted row `i`, column `col`, in the class of expert `e`. */
struct NvArgs {
    const int8_t* a;  size_t a_ld;     /* activation codes, row pitch in elements */
    const float*  as; size_t as_ld;    /* activation scales, row pitch in elements */
    int64_t       AR;                  /* rows the activation addresses */
    int           a_sorted;            /* 1: row i of `a` is sorted row i; 0: its token */
    int           a_group;             /* K the activation scale covers */
    int           top_k;
    int           n_expert;
    int           N;                   /* output width, the wider class's */
    int           parts;
    int           N_c[2], K_c[2];      /* per class: weight rows and K */
    NvTab         w[2], ws[2];         /* per class: codes and scale blobs */
};

/* ---- Decode kernel: 8 K-lanes per column, 16 columns per 128-thread block --------------------- */

template <bool FAST256>
__device__ __forceinline__ void nvfp4_moe_dec_body(
    NvArgs p, const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ y, size_t y_ld) {
    const int i = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int col_local = tid / kNvDecLanes;
    const int lk = tid & (kNvDecLanes - 1);
    const int col = (int)blockIdx.y * kNvDecCols + col_local;

    /* Row i's expert is the last x with off[x] <= i, or 0 when there is none (off is
     * nondecreasing), as a binary search over the offsets finds it: the one x where x is 0 or
     * off[x] <= i, and x is the last expert or i < off[x + 1]. Every thread tests its experts at
     * once, one load deep, where the search was log2(n_expert) dependent loads before the first
     * weight load could start. `sorted` is read beside the offsets, not ahead of them. */
    __shared__ int s_e;
    const int32_t flat = sorted[i];
    for (int x = tid; x < p.n_expert; x += kNvDecBlock) {
        if ((x == 0 || off[x] <= i) && (x + 1 == p.n_expert || i < off[x + 1])) s_e = x;
    }
    lds_barrier();
    if (col >= p.N) return;
    int e = -1;
    int64_t tok = 0;
    if (flat >= 0) {
        tok = p.a_sorted ? (int64_t)i : (int64_t)flat / p.top_k;
        if (tok < p.AR) e = s_e;
    }
    if (e < 0) {
        if (lk == 0) y[(size_t)i * y_ld + col] = 0;
        return;
    }

    const int q = e & 1;
    const uint32_t ix = (uint32_t)(e >> 1);
    const int64_t n_q = p.N_c[q], k_q = p.K_c[q];
    const int64_t fy = p.N / p.parts, fc = n_q / p.parts;
    const int64_t pp = col / fy, j = col - pp * fy;
    if (j >= fc) {
        if (lk == 0) y[(size_t)i * y_ld + col] = 0;
        return;
    }
    const int64_t wr = pp * fc + j;

    const uint8_t* crow  = nv_uw_ptr(nv_entry(p.w[q], ix)) + (size_t)wr * (size_t)(k_q / 2);
    const uint8_t* sblob = nv_uw_ptr(nv_entry(p.ws[q], ix));
    const uint8_t* srow  = sblob + (size_t)wr * (size_t)(k_q / 16);
    float level;
    __builtin_memcpy(&level, sblob + (size_t)n_q * (size_t)(k_q / 16), sizeof level);
    const float lv = 0.5f * level;

    const int8_t* arow = p.a + (size_t)tok * p.a_ld;
    const float* sarow = p.as + (size_t)tok * p.as_ld;

    float acc = 0.0f;
    if constexpr (FAST256) {
        /* Fast path: K % 128 == 0, a_group == 128, 16-byte aligned rows. Each of the 8 lanes reads
         * 16 weight bytes (32 weights = 2 groups of 16) + 32 int8 activations per 256-K step,
         * sharing one activation scale across lanes 0..3 and one across lanes 4..7. */
        const int64_t n256 = k_q >> 8;
        const int64_t as_off = (lk >> 2);
        auto eval16 = [&](uint32_t wx, uint32_t wy, const int4& a, uint8_t ws_b) -> float {
            int32_t w0, w1, w2, w3;
            nv_unpack8_i8(wx, w0, w1);
            nv_unpack8_i8(wy, w2, w3);
            int32_t sub = nv_sdot4(w0, a.x, 0);
            sub = nv_sdot4(w1, a.y, sub);
            sub = nv_sdot4(w2, a.z, sub);
            sub = nv_sdot4(w3, a.w, sub);
            return (float)sub * r4d_fp8::e4m3((uint32_t)ws_b);
        };
        if (n256 > 0) {
            int64_t bi = 0;
            if (n256 >= 2) {
                uint4    cw0  = nv_gload_u4(crow + (size_t)lk * 16);
                uint4    cw1  = nv_gload_u4(crow + 128 + (size_t)lk * 16);
                int4     ca00 = nv_load_i4(arow + (size_t)lk * 32);
                int4     ca01 = nv_load_i4(arow + (size_t)lk * 32 + 16);
                int4     ca10 = nv_load_i4(arow + 256 + (size_t)lk * 32);
                int4     ca11 = nv_load_i4(arow + 256 + (size_t)lk * 32 + 16);
                uint16_t cs0  = nv_load_u16(srow + lk * 2);
                uint16_t cs1  = nv_load_u16(srow + 16 + lk * 2);
                float    cas0 = sarow[as_off];
                float    cas1 = sarow[2 + as_off];
                for (; bi + 1 < n256; bi += 2) {
                    const int64_t bn = bi + 2;
                    uint4    nw0{}, nw1{};
                    int4     na00{}, na01{}, na10{}, na11{};
                    uint16_t ncs0 = 0, ncs1 = 0;
                    float    nas0 = 0.0f, nas1 = 0.0f;
                    if (bn + 1 < n256) {
                        nw0  = nv_gload_u4(crow + (size_t)bn * 128 + (size_t)lk * 16);
                        nw1  = nv_gload_u4(crow + (size_t)(bn + 1) * 128 + (size_t)lk * 16);
                        na00 = nv_load_i4(arow + (size_t)bn * 256 + (size_t)lk * 32);
                        na01 = nv_load_i4(arow + (size_t)bn * 256 + (size_t)lk * 32 + 16);
                        na10 = nv_load_i4(arow + (size_t)(bn + 1) * 256 + (size_t)lk * 32);
                        na11 = nv_load_i4(arow + (size_t)(bn + 1) * 256 + (size_t)lk * 32 + 16);
                        ncs0 = nv_load_u16(srow + bn * 16 + lk * 2);
                        ncs1 = nv_load_u16(srow + (bn + 1) * 16 + lk * 2);
                        nas0 = sarow[bn * 2 + as_off];
                        nas1 = sarow[(bn + 1) * 2 + as_off];
                    }
                    acc += (eval16(cw0.x, cw0.y, ca00, (uint8_t)(cs0 & 0xffu)) +
                            eval16(cw0.z, cw0.w, ca01, (uint8_t)(cs0 >> 8))) * cas0;
                    acc += (eval16(cw1.x, cw1.y, ca10, (uint8_t)(cs1 & 0xffu)) +
                            eval16(cw1.z, cw1.w, ca11, (uint8_t)(cs1 >> 8))) * cas1;
                    cw0 = nw0; cw1 = nw1;
                    ca00 = na00; ca01 = na01; ca10 = na10; ca11 = na11;
                    cs0 = ncs0; cs1 = ncs1; cas0 = nas0; cas1 = nas1;
                }
            }
            if (bi < n256) {
                const uint4    cw  = nv_gload_u4(crow + (size_t)bi * 128 + (size_t)lk * 16);
                const int4     ca0 = nv_load_i4(arow + (size_t)bi * 256 + (size_t)lk * 32);
                const int4     ca1 = nv_load_i4(arow + (size_t)bi * 256 + (size_t)lk * 32 + 16);
                const uint16_t cs  = nv_load_u16(srow + bi * 16 + lk * 2);
                const float    cas = sarow[bi * 2 + as_off];
                acc += (eval16(cw.x, cw.y, ca0, (uint8_t)(cs & 0xffu)) +
                        eval16(cw.z, cw.w, ca1, (uint8_t)(cs >> 8))) * cas;
            }
        }
        if ((k_q & 128) != 0) {
            const uint2   tw = nv_gload_u2(crow + (size_t)n256 * 128 + (size_t)lk * 8);
            const int4    ta = nv_load_i4(arow + (size_t)n256 * 256 + (size_t)lk * 16);
            const uint8_t ts = srow[n256 * 16 + lk];
            const float   sa = sarow[n256 * 2];
            acc += eval16(tw.x, tw.y, ta, ts) * sa;
        }
        acc *= lv;
    } else {
        const int64_t ng = k_q / 16;
        if ((int64_t)lk < ng) {
            uint2   cur_w  = nv_gload_u2(crow + (size_t)lk * 8);
            int4    cur_a  = nv_load_i4(arow + (size_t)lk * 16);
            uint8_t cur_ws = srow[lk];
            float   cur_as = sarow[(lk * 16) / p.a_group];

            for (int64_t g = lk; g < ng; g += kNvDecLanes) {
                const int64_t gn = g + kNvDecLanes;
                uint2   nxt_w{};
                int4    nxt_a{};
                uint8_t nxt_ws = 0;
                float   nxt_as = 0.0f;
                if (gn < ng) {
                    nxt_w  = nv_gload_u2(crow + (size_t)gn * 8);
                    nxt_a  = nv_load_i4(arow + (size_t)gn * 16);
                    nxt_ws = srow[gn];
                    nxt_as = sarow[(gn * 16) / p.a_group];
                }
                int32_t w0, w1, w2, w3;
                nv_unpack8_i8(cur_w.x, w0, w1);
                nv_unpack8_i8(cur_w.y, w2, w3);
                int32_t sub = nv_sdot4(w0, cur_a.x, 0);
                sub = nv_sdot4(w1, cur_a.y, sub);
                sub = nv_sdot4(w2, cur_a.z, sub);
                sub = nv_sdot4(w3, cur_a.w, sub);
                acc += (float)sub * cur_as * (r4d_fp8::e4m3((uint32_t)cur_ws) * lv);
                cur_w = nxt_w; cur_a = nxt_a; cur_ws = nxt_ws; cur_as = nxt_as;
            }
        }
    }
    acc = r4d_xor_sum<kNvDecLanes>(acc);
    if (lk == 0) y[(size_t)i * y_ld + col] = f32_to_bf16(acc);
}

template <bool FAST256, class CH>
__global__ __launch_bounds__(kNvDecBlock) void nvfp4_moe_dec_kernel(
    NvArgs p, const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ y, size_t y_ld) {
    if (CH::on) r4d_chain_wait();
    nvfp4_moe_dec_body<FAST256>(p, sorted, off, y, y_ld);
    if (CH::on) r4d_chain_done();
}

/* ---- Multi-row grouped decode/verify kernel: RM rows per expert tile, 16 cols/block -------- */

template <class CH>
__global__ __launch_bounds__(kNvDecBlock) void nvfp4_moe_mdec_kernel(
    NvArgs p, const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ y, size_t y_ld, uint32_t T) {
    if (CH::on) r4d_chain_wait();
    constexpr uint32_t NPT = (uint32_t)kNvMaxExpert / (uint32_t)kNvDecBlock;
    static_assert((uint32_t)kNvMaxExpert % (uint32_t)kNvDecBlock == 0u,
                  "experts divide over block threads");

    __shared__ uint32_t s_rc[kNvDecBlock / 32];
    __shared__ uint32_t s_sel[3];
    __shared__ int32_t  s_tok[kNvMdecRM];

    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t lane = tid & 31u, wave = tid >> 5;
    const uint32_t col_local = tid / (uint32_t)kNvDecLanes;
    const uint32_t lk = tid & (uint32_t)(kNvDecLanes - 1);
    const uint32_t col = blockIdx.y * (uint32_t)kNvDecCols + col_local;

    const uint32_t live_raw = (uint32_t)off[p.n_expert];
    const uint32_t live = live_raw < T ? live_raw : T;
    if (blockIdx.x == gridDim.x - 1u) {
        if (col < (uint32_t)p.N && lk == 0u) {
            for (uint32_t r = live; r < T; ++r) y[(size_t)r * y_ld + col] = 0;
        }
        if (CH::on) r4d_chain_done();
        return;
    }

    uint32_t ov[NPT + 1];
#pragma unroll
    for (uint32_t k = 0; k <= NPT; ++k) {
        const uint32_t x = tid * NPT + k;
        const uint32_t raw = (uint32_t)off[x <= (uint32_t)p.n_expert ? x : (uint32_t)p.n_expert];
        const uint32_t o = x <= (uint32_t)p.n_expert ? raw : 0u;
        ov[k] = o < live ? o : live;
    }
    uint32_t nt[NPT], cnt = 0;
#pragma unroll
    for (uint32_t k = 0; k < NPT; ++k) {
        const uint32_t run =
            (tid * NPT + k < (uint32_t)p.n_expert && ov[k + 1] > ov[k]) ? (ov[k + 1] - ov[k]) : 0u;
        nt[k] = (run + (uint32_t)kNvMdecRM - 1u) / (uint32_t)kNvMdecRM;
        cnt += nt[k];
    }
    const uint32_t inc = r4d_wave_prefix_u32(cnt, (int)lane);
    if (lane == 31u) s_rc[wave] = inc;
    lds_barrier();
    uint32_t before = inc - cnt, ntile = 0;
#pragma unroll
    for (uint32_t w = 0; w < (uint32_t)(kNvDecBlock / 32); ++w) {
        const uint32_t c = s_rc[w];
        before += w < wave ? c : 0u;
        ntile += c;
    }
    const uint32_t j = (uint32_t)blockIdx.x;
    if (j >= ntile) {
        if (CH::on) r4d_chain_done();
        return;
    }
    if (j >= before && j < before + cnt) {
        uint32_t m = j - before;
        bool hit = false;
#pragma unroll
        for (uint32_t k = 0; k < NPT; ++k) {
            if (!hit && m < nt[k]) {
                s_sel[0] = tid * NPT + k;
                s_sel[1] = ov[k] + m * (uint32_t)kNvMdecRM;
                s_sel[2] = ov[k + 1];
                hit = true;
            }
            if (!hit) m -= nt[k];
        }
    }
    lds_barrier();

    const uint32_t e  = (uint32_t)nv_readfirstlane((int)s_sel[0]);
    const uint32_t rs = (uint32_t)nv_readfirstlane((int)s_sel[1]);
    const uint32_t r1 = (uint32_t)nv_readfirstlane((int)s_sel[2]);
    const uint32_t re = rs + (uint32_t)kNvMdecRM < r1 ? rs + (uint32_t)kNvMdecRM : r1;
    const int nr = (int)(re - rs);

    if ((int)tid < nr) {
        const uint32_t row = rs + tid;
        const int32_t flat = sorted[row];
        int32_t tok = -1;
        if (flat >= 0) {
            const int64_t t = p.a_sorted ? (int64_t)row : (int64_t)flat / p.top_k;
            if (t < p.AR) tok = (int32_t)t;
        }
        s_tok[tid] = tok;
    }
    lds_barrier();

    if (col < (uint32_t)p.N) {
        const int q = (int)(e & 1u);
        const uint32_t ix = e >> 1;
        const int64_t n_q = p.N_c[q], k_q = p.K_c[q];
        const int64_t fy = p.N / p.parts, fc = n_q / p.parts;
        const int64_t pp = (int64_t)col / fy, jc = (int64_t)col - pp * fy;
        if (jc >= fc) {
            if (lk == 0u) {
                for (int r = 0; r < nr; ++r) y[(size_t)(rs + (uint32_t)r) * y_ld + col] = 0;
            }
        } else {
            const int64_t wr = pp * fc + jc;
            const uint8_t* crow  = nv_uw_ptr(nv_entry(p.w[q], ix)) + (size_t)wr * (size_t)(k_q / 2);
            const uint8_t* sblob = nv_uw_ptr(nv_entry(p.ws[q], ix));
            const uint8_t* srow  = sblob + (size_t)wr * (size_t)(k_q / 16);
            float level;
            __builtin_memcpy(&level, sblob + (size_t)n_q * (size_t)(k_q / 16), sizeof level);
            const float lv = 0.5f * level;

            float acc[kNvMdecRM];
#pragma unroll
            for (int r = 0; r < kNvMdecRM; ++r) acc[r] = 0.0f;

            const int64_t n256 = k_q >> 8;
            const int64_t as_off = (int64_t)(lk >> 2);
            if (n256 > 0) {
                uint4    cw = nv_gload_u4(crow + (size_t)lk * 16);
                uint16_t cs = nv_load_u16(srow + lk * 2u);
                for (int64_t bi = 0; bi < n256; ++bi) {
                    const int64_t bn = bi + 1;
                    uint4    nw{};
                    uint16_t ncs = 0;
                    if (bn < n256) {
                        nw  = nv_gload_u4(crow + (size_t)bn * 128 + (size_t)lk * 16);
                        ncs = nv_load_u16(srow + bn * 16 + lk * 2u);
                    }
                    int32_t w0, w1, w2, w3, w4, w5, w6, w7;
                    nv_unpack8_i8(cw.x, w0, w1);
                    nv_unpack8_i8(cw.y, w2, w3);
                    nv_unpack8_i8(cw.z, w4, w5);
                    nv_unpack8_i8(cw.w, w6, w7);
                    const float ws0 = r4d_fp8::e4m3((uint32_t)(cs & 0xffu));
                    const float ws1 = r4d_fp8::e4m3((uint32_t)(cs >> 8));
                    const size_t a_off0 = (size_t)bi * 256 + (size_t)lk * 32;
                    const size_t s_off  = (size_t)bi * 2 + (size_t)as_off;
                    for (int r = 0; r < nr; ++r) {
                        const int32_t tok = s_tok[r];
                        if (tok < 0) continue;
                        const int8_t* arow = p.a + (size_t)tok * p.a_ld + a_off0;
                        const int4 ca0 = nv_load_i4(arow);
                        const int4 ca1 = nv_load_i4(arow + 16);
                        const float cas = p.as[(size_t)tok * p.as_ld + s_off];
                        int32_t s0 = nv_sdot4(w0, ca0.x, 0);
                        s0 = nv_sdot4(w1, ca0.y, s0);
                        s0 = nv_sdot4(w2, ca0.z, s0);
                        s0 = nv_sdot4(w3, ca0.w, s0);
                        int32_t s1 = nv_sdot4(w4, ca1.x, 0);
                        s1 = nv_sdot4(w5, ca1.y, s1);
                        s1 = nv_sdot4(w6, ca1.z, s1);
                        s1 = nv_sdot4(w7, ca1.w, s1);
                        acc[r] += ((float)s0 * ws0 + (float)s1 * ws1) * cas;
                    }
                    cw = nw;
                    cs = ncs;
                }
            }
            if ((k_q & 128) != 0) {
                const uint2   tw  = nv_gload_u2(crow + (size_t)n256 * 128 + (size_t)lk * 8);
                const float   ws  = r4d_fp8::e4m3((uint32_t)srow[n256 * 16 + lk]);
                int32_t w0, w1, w2, w3;
                nv_unpack8_i8(tw.x, w0, w1);
                nv_unpack8_i8(tw.y, w2, w3);
                const size_t a_off = (size_t)n256 * 256 + (size_t)lk * 16;
                const size_t s_off = (size_t)n256 * 2;
                for (int r = 0; r < nr; ++r) {
                    const int32_t tok = s_tok[r];
                    if (tok < 0) continue;
                    const int4 ta = nv_load_i4(p.a + (size_t)tok * p.a_ld + a_off);
                    const float sa = p.as[(size_t)tok * p.as_ld + s_off];
                    int32_t s0 = nv_sdot4(w0, ta.x, 0);
                    s0 = nv_sdot4(w1, ta.y, s0);
                    s0 = nv_sdot4(w2, ta.z, s0);
                    s0 = nv_sdot4(w3, ta.w, s0);
                    acc[r] += (float)s0 * ws * sa;
                }
            }
            for (int r = 0; r < nr; ++r) {
                const float sum = r4d_xor_sum<kNvDecLanes>(acc[r]) * lv;
                if (lk == 0u) y[(size_t)(rs + (uint32_t)r) * y_ld + col] = f32_to_bf16(sum);
            }
        }
    }
    if (CH::on) r4d_chain_done();
}

/* ---- Block map for run-aligned prefill tiles (self-contained twin of moe_blockmap_kernel) ---- */

static inline long long nv_blockmap_cap(long long T, long long n_expert, long long FM) {
    return n_expert + (T + FM - 1) / FM + 1;
}

static inline long long nv_blockmap_bytes(long long cap) {
    return (1 + 2 * cap) * (long long)sizeof(uint32_t);
}

template <class CH>
__global__ __launch_bounds__(256) void nvfp4_blockmap_kernel(
    const int32_t* __restrict__ off, uint32_t n_expert, uint32_t T, uint32_t FM, uint32_t cap,
    uint32_t* __restrict__ map) {
    constexpr uint32_t NT = 256, NW = NT / 32, kTiles = (uint32_t)((kNvMaxExpert + NT) / NT);
    if (CH::on) r4d_chain_wait();
    __shared__ uint32_t s_cnt[(size_t)kNvMaxExpert + 1];
    __shared__ uint32_t s_wt[kTiles][NW];
    const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;

    int32_t lo[kTiles], hi[kTiles];
#pragma unroll
    for (uint32_t i = 0; i < kTiles; ++i) {
        const uint32_t e = tid + i * NT;
        lo[i] = off[e < n_expert ? e : n_expert];
        hi[i] = off[e + 1 < n_expert ? e + 1 : n_expert];
    }
    const uint32_t live_raw = (uint32_t)off[n_expert];
    const uint32_t live = live_raw < T ? live_raw : T;
#pragma unroll
    for (uint32_t i = 0; i < kTiles; ++i) {
        const uint32_t e = tid + i * NT;
        if (e < n_expert) {
            const uint32_t a = (uint32_t)lo[i] < live ? (uint32_t)lo[i] : live;
            const uint32_t b = (uint32_t)hi[i] < live ? (uint32_t)hi[i] : live;
            const uint32_t run = b > a ? (b - a) : 0u;
            s_cnt[e] = (run + FM - 1u) / FM;
        }
    }
    if (tid == 0) s_cnt[n_expert] = T > live ? (T - live + FM - 1u) / FM : 0u;
    __syncthreads();

    uint32_t carry = 0;
    for (uint32_t b = 0, tl = 0; b <= n_expert; b += NT, ++tl) {
        const uint32_t e = b + tid;
        const uint32_t v = e <= n_expert ? s_cnt[e] : 0u;
        const uint32_t x = r4d_wave_prefix_u32(v, (int)lane);
        if (lane == 31u) s_wt[tl][wave] = x;
        lds_barrier();
        uint32_t below = 0, tot = 0;
#pragma unroll
        for (uint32_t k = 0; k < NW; ++k) {
            const uint32_t c = s_wt[tl][k];
            below += (k < wave) ? c : 0u;
            tot   += c;
        }
        if (e <= n_expert) s_cnt[e] = carry + (below + x - v);
        carry += tot;
    }
    if (tid == 0) map[0] = carry < cap ? carry : cap;
    __syncthreads();

#pragma unroll
    for (uint32_t i = 0; i < kTiles; ++i) {
        const uint32_t e = tid + i * NT;
        if (e >= n_expert) continue;
        const uint32_t a = (uint32_t)lo[i] < live ? (uint32_t)lo[i] : live;
        const uint32_t b = (uint32_t)hi[i] < live ? (uint32_t)hi[i] : live;
        const uint32_t run = b > a ? (b - a) : 0u;
        uint32_t o = s_cnt[e];
        for (uint32_t r = 0; r < run; r += FM, ++o) {
            if (o >= cap) break;
            map[1 + 2 * o + 0] = e;
            map[1 + 2 * o + 1] = a + r;
        }
    }
    if (tid == 0) {
        uint32_t o = s_cnt[n_expert];
        for (uint32_t r = live; r < T; r += FM, ++o) {
            if (o >= cap) break;
            map[1 + 2 * o + 0] = kNvMapTail;
            map[1 + 2 * o + 1] = r;
        }
    }
    if (CH::on) r4d_chain_done();
}

/* ---- Prefill WMMA kernel: 32x256, 32x128 or 32x64 tiles, FK=128 slabs ------------------------- */

/* The weight rows behind the column tile at output column n0 for a class of n_q rows: the output is
 * `parts` stacked parts of N / parts columns and the class's weight `parts` of n_q / parts rows, so
 * the tile reads its part's rows at its offset in the part. A tile past the class's part width is
 * dead (zeros); the prefill pick keeps every tile inside one part and whole or dead. */
__device__ __forceinline__ void nv_part_rows(const NvArgs& p, uint32_t n_q, uint32_t n0,
                                             uint32_t* wn0, bool* live) {
    const uint32_t fy = (uint32_t)p.N / (uint32_t)p.parts, fc = n_q / (uint32_t)p.parts;
    const uint32_t pp = n0 / fy, j0 = n0 - pp * fy;
    *live = j0 < fc;
    *wn0 = pp * fc + j0;
}

template <uint32_t FM, uint32_t FN, uint32_t WGM, uint32_t WGN, bool AL, bool AG128>
__device__ __forceinline__ void nvfp4_moe_prefill_body(
    NvArgs p, const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ y, size_t y_ld, const uint32_t* __restrict__ map, uint32_t T) {
    constexpr uint32_t FK   = kNvPfK;
    constexpr uint32_t NTHR = WGM * WGN * 32u;
    constexpr uint32_t MT   = FM / (WGM * 16u);
    constexpr uint32_t NT   = FN / (WGN * 16u);
    static_assert(FM == WGM * MT * 16u && FN == WGN * NT * 16u, "tile must factor across WGMxWGN waves");
    static_assert(FK == 128u && (NTHR == 256u || NTHR == 512u), "staging assumes 256 or 512 threads over FK=128");

    __shared__ uint8_t  s_a[FM * r4d_fp8::lds_pitch<FK>()];
    __shared__ uint8_t  s_b[FN * r4d_fp8::lds_pitch<FK>()];
    __shared__ int32_t  s_tok[FM];
    __shared__ float    s_as[AG128 ? FM : 8u * FM];
    __shared__ float    s_ws[8u * FN];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t wgm  = wave / WGN;
    const uint32_t wgn  = wave % WGN;
    const uint32_t n0   = blockIdx.y * FN;

    const uint32_t live_raw = (uint32_t)off[p.n_expert];
    const uint32_t live = live_raw < T ? live_raw : T;

    uint32_t r0 = 0, rN = 0, e = 0, al_end = 0;
    if constexpr (AL) {
        if (blockIdx.x >= map[0]) return;
        const uint32_t* ent = map + 1 + 2 * blockIdx.x;
        e  = ent[0];
        r0 = ent[1];
        if (e == kNvMapTail) {
            const uint32_t rT = (r0 + FM < T) ? (r0 + FM) : T;
            for (uint32_t r = (r0 > live ? r0 : live); r < rT; ++r)
                for (uint32_t c = tid; c < FN && n0 + c < (uint32_t)p.N; c += NTHR)
                    y[(size_t)r * y_ld + n0 + c] = 0;
            return;
        }
        const uint32_t oe = (uint32_t)off[e + 1];
        al_end = oe < live ? oe : live;
        rN = (r0 + FM < al_end) ? (r0 + FM) : al_end;
        if (rN <= r0) return;
    } else {
        r0 = blockIdx.x * FM;
        if (r0 >= T) return;
        rN = (r0 + FM < T) ? (r0 + FM) : T;
        if (r0 < live) {
            uint32_t lo = 0, hi = (uint32_t)p.n_expert;
            while (lo + 1u < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if ((uint32_t)off[mid] <= r0) lo = mid; else hi = mid;
            }
            e = lo;
        }
    }

    constexpr uint32_t kSeg = 32u;
    constexpr uint32_t kTPR = FK / kSeg;       /* 4 threads across FK=128 */
    constexpr uint32_t kRPP = NTHR / kTPR;     /* 64 or 128 rows per pass */
    constexpr uint32_t kBP  = FN / kRPP;       /* 1 or 2 passes */
    constexpr uint32_t kASP = AG128 ? 1u : (8u * FM + NTHR - 1u) / NTHR;
    const uint32_t sn = tid >> 2;              /* 0..kRPP-1 row inside pass */
    const uint32_t sk = (tid & 3u) * kSeg;     /* 0, 32, 64, 96 */

    /* Early c0=0 weight prefetch under AL: issue weight & scale loads while tokens/s_tok settle. */
    [[maybe_unused]] r4d_u32x4      ew_bpre[AL ? kBP : 1]{};
    [[maybe_unused]] uint64_t       ew_pwsu = 0;
    [[maybe_unused]] const uint8_t* ew_wp   = nullptr;
    [[maybe_unused]] const uint8_t* ew_sbase = nullptr;
    [[maybe_unused]] uint32_t       ew_nq = 0, ew_kq = 0, ew_wn0 = 0;
    [[maybe_unused]] bool           ew_live = false;
    if constexpr (AL) {
        const uint32_t q0 = e & 1u, ix0 = e >> 1;
        ew_nq = (uint32_t)p.N_c[q0];
        ew_kq = (uint32_t)p.K_c[q0];
        nv_part_rows(p, ew_nq, n0, &ew_wn0, &ew_live);
        if (ew_live) {
            ew_wp    = nv_uw_ptr(nv_entry(p.w[q0], ix0));
            ew_sbase = nv_uw_ptr(nv_entry(p.ws[q0], ix0));
#pragma unroll
            for (uint32_t q_row = 0; q_row < kBP; ++q_row) {
                const uint32_t r = ew_wn0 + sn + q_row * kRPP;
                ew_bpre[q_row] = r < ew_nq
                    ? nv_gload_v4(ew_wp + (size_t)r * (size_t)(ew_kq >> 1) + (sk >> 1))
                    : r4d_u32x4{0u, 0u, 0u, 0u};
            }
            if (tid < FN && ew_wn0 + tid < ew_nq)
                ew_pwsu = nv_gload_u64(ew_sbase + (size_t)(ew_wn0 + tid) * (size_t)(ew_kq >> 4));
        }
    }

    auto zero_tail = [&](uint32_t z0, uint32_t z1) {
        for (uint32_t r = z0; r < z1; ++r)
            for (uint32_t c = tid; c < FN && n0 + c < (uint32_t)p.N; c += NTHR)
                y[(size_t)r * y_ld + n0 + c] = 0;
    };

    uint32_t rs = r0;
    while (rs < rN && rs < live) {
        if constexpr (!AL) {
            while (e < (uint32_t)p.n_expert && (uint32_t)off[e + 1] <= rs) ++e;
            if (e >= (uint32_t)p.n_expert) break;
        }
        const uint32_t ehi = AL ? al_end : (uint32_t)off[e + 1];
        const uint32_t rlim = rN < live ? rN : live;
        const uint32_t re = ehi < rlim ? ehi : rlim;
        if (re <= rs) break;

        const uint32_t q   = e & 1u;
        const uint32_t ix  = e >> 1;
        const uint32_t n_q = AL ? ew_nq : (uint32_t)p.N_c[q];
        const uint32_t k_q = AL ? ew_kq : (uint32_t)p.K_c[q];
        /* the tile's first weight row: its part's, at the tile's offset in the part */
        uint32_t wn0 = 0;
        bool tlive = false;
        if constexpr (AL) {
            wn0 = ew_wn0;
            tlive = ew_live;
        } else {
            nv_part_rows(p, n_q, n0, &wn0, &tlive);
        }

        if (!tlive) {
            zero_tail(rs, re);
            rs = re;
            if constexpr (AL) break;
            continue;
        }

        const uint8_t* Wp    = AL ? ew_wp    : nv_uw_ptr(nv_entry(p.w[q], ix));
        const uint8_t* Sbase = AL ? ew_sbase : nv_uw_ptr(nv_entry(p.ws[q], ix));
        float level;
        __builtin_memcpy(&level, Sbase + (size_t)n_q * (size_t)(k_q >> 4), sizeof level);
        const float lv = 0.5f * level;

        if (tid < FM) {
            const uint32_t r = r0 + tid;
            int32_t tok = -1;
            if (r >= rs && r < re) {
                const int32_t flat = sorted[r];
                if (flat >= 0) {
                    const int64_t t = p.a_sorted ? (int64_t)r : (int64_t)flat / p.top_k;
                    if (t < p.AR) tok = (int32_t)t;
                }
            }
            s_tok[tid] = tok;
        }
        __syncthreads();

        const bool wlive = (r0 + wgm * (MT * 16u) < re);
        r4d_fp8::f8v out[MT][NT];
#pragma unroll
        for (uint32_t tm = 0; tm < MT; ++tm)
#pragma unroll
            for (uint32_t tn = 0; tn < NT; ++tn)
                out[tm][tn] = r4d_fp8::acc_zero();

        r4d_fp8::Stage8 apre{};
        r4d_u32x4       bpre[kBP]{};
        uint64_t        pwsu = 0;
        float           pas[kASP]{};

        auto issue = [&](uint32_t c0, auto first) {
            if (sn < FM) {
                const int32_t tok = s_tok[sn];
                const bool ok = tok >= 0;
                const uint8_t* src = (const uint8_t*)p.a + (size_t)(ok ? tok : 0) * p.a_ld + c0 + sk;
                apre = r4d_fp8::stage8_load(src, ok);
            }
            if constexpr (AL && decltype(first)::value) {
#pragma unroll
                for (uint32_t q_row = 0; q_row < kBP; ++q_row) bpre[q_row] = ew_bpre[q_row];
                pwsu = ew_pwsu;
            } else {
#pragma unroll
                for (uint32_t q_row = 0; q_row < kBP; ++q_row) {
                    const uint32_t r = wn0 + sn + q_row * kRPP;
                    bpre[q_row] = r < n_q
                        ? nv_gload_v4(Wp + (size_t)r * (size_t)(k_q >> 1) + ((c0 + sk) >> 1))
                        : r4d_u32x4{0u, 0u, 0u, 0u};
                }
                if (tid < FN) {
                    pwsu = (wn0 + tid < n_q)
                        ? nv_gload_u64(Sbase + (size_t)(wn0 + tid) * (size_t)(k_q >> 4) + (c0 >> 4))
                        : 0ull;
                }
            }
            if constexpr (AG128) {
                if (tid < FM) {
                    const int32_t tok = s_tok[tid];
                    pas[0] = (tok >= 0) ? p.as[(size_t)tok * p.as_ld + (c0 >> 7)] : 0.0f;
                }
            } else {
#pragma unroll
                for (uint32_t pass = 0; pass < kASP; ++pass) {
                    const uint32_t idx = tid + pass * NTHR;
                    if (idx < 8u * FM) {
                        const uint32_t as_r  = idx % FM;
                        const uint32_t as_gk = idx / FM;
                        const int32_t  tok   = s_tok[as_r];
                        pas[pass] = (tok >= 0)
                                        ? p.as[(size_t)tok * p.as_ld +
                                               (c0 + as_gk * 16u) / (uint32_t)p.a_group]
                                        : 0.0f;
                    }
                }
            }
        };

        issue(0u, std::true_type{});
        for (uint32_t c0 = 0; c0 < k_q; c0 += FK) {
            lds_barrier();
#ifdef __HIP__
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
#endif
            if (sn < FM) r4d_fp8::stage8_store<FK>(s_a, sn, sk, apre);
#pragma unroll
            for (uint32_t q_row = 0; q_row < kBP; ++q_row) {
                int32_t p0, p1, p2, p3, p4, p5, p6, p7;
                nv_unpack8_i8(bpre[q_row][0], p0, p1);
                nv_unpack8_i8(bpre[q_row][1], p2, p3);
                nv_unpack8_i8(bpre[q_row][2], p4, p5);
                nv_unpack8_i8(bpre[q_row][3], p6, p7);
                const r4d_fp8::Stage8 b8 = {
                    make_uint4((uint32_t)p0, (uint32_t)p1, (uint32_t)p2, (uint32_t)p3),
                    make_uint4((uint32_t)p4, (uint32_t)p5, (uint32_t)p6, (uint32_t)p7),
                };
                r4d_fp8::stage8_store<FK>(s_b, sn + q_row * kRPP, sk, b8);
            }
            if constexpr (AG128) {
                if (tid < FM) s_as[tid] = pas[0];
            } else {
#pragma unroll
                for (uint32_t pass = 0; pass < kASP; ++pass) {
                    const uint32_t idx = tid + pass * NTHR;
                    if (idx < 8u * FM) s_as[idx] = pas[pass];
                }
            }
            if (tid < FN) {
#pragma unroll
                for (uint32_t gk = 0; gk < 8u; ++gk)
                    s_ws[gk * FN + tid] = r4d_fp8::e4m3((uint32_t)((pwsu >> (gk * 8u)) & 0xFFu)) * lv;
            }
            if (c0 + FK < k_q) issue(c0 + FK, std::false_type{});
            lds_barrier();

            if (!wlive) continue;

            if constexpr (AG128) {
                r4d_fp8::f8v slab[MT][NT];
#pragma unroll 1
                for (uint32_t gk = 0; gk < 8u; ++gk) {
                    r4d_fp8::i2v af[MT], bf[NT];
#pragma unroll
                    for (uint32_t tm = 0; tm < MT; ++tm)
                        af[tm] = r4d_fp8::frag<FK>(s_a, wgm * (MT * 16u) + tm * 16u, gk * 16u, lane);
#pragma unroll
                    for (uint32_t tn = 0; tn < NT; ++tn)
                        bf[tn] = r4d_fp8::frag<FK>(s_b, wgn * (NT * 16u) + tn * 16u, gk * 16u, lane);
#pragma unroll
                    for (uint32_t tm = 0; tm < MT; ++tm) {
#pragma unroll
                        for (uint32_t tn = 0; tn < NT; ++tn) {
                            const r4d_fp8::f8v raw =
                                r4d_fp8::mma8<true>(af[tm], bf[tn], r4d_fp8::acc_zero());
                            const float ws =
                                s_ws[gk * FN + wgn * (NT * 16u) + tn * 16u + r4d_fp8::frag_c(lane)];
#pragma unroll
                            for (uint32_t t = 0; t < 8u; ++t) {
                                const float prod = r4d_fp8::acc_val<true>(raw[t]) * ws;
                                slab[tm][tn][t] = (gk == 0u) ? prod : (slab[tm][tn][t] + prod);
                            }
                        }
                    }
                }
#pragma unroll
                for (uint32_t tm = 0; tm < MT; ++tm) {
#pragma unroll
                    for (uint32_t t = 0; t < 8u; ++t) {
                        const float sa =
                            s_as[wgm * (MT * 16u) + tm * 16u + r4d_fp8::acc_row(lane, t)];
#pragma unroll
                        for (uint32_t tn = 0; tn < NT; ++tn)
                            out[tm][tn][t] = fmaf(slab[tm][tn][t], sa, out[tm][tn][t]);
                    }
                }
            } else {
#pragma unroll 1
                for (uint32_t gk = 0; gk < 8u; ++gk) {
                    r4d_fp8::i2v af[MT], bf[NT];
#pragma unroll
                    for (uint32_t tm = 0; tm < MT; ++tm)
                        af[tm] = r4d_fp8::frag<FK>(s_a, wgm * (MT * 16u) + tm * 16u, gk * 16u, lane);
#pragma unroll
                    for (uint32_t tn = 0; tn < NT; ++tn)
                        bf[tn] = r4d_fp8::frag<FK>(s_b, wgn * (NT * 16u) + tn * 16u, gk * 16u, lane);
#pragma unroll
                    for (uint32_t tm = 0; tm < MT; ++tm) {
#pragma unroll
                        for (uint32_t tn = 0; tn < NT; ++tn) {
                            const r4d_fp8::f8v raw =
                                r4d_fp8::mma8<true>(af[tm], bf[tn], r4d_fp8::acc_zero());
                            const float ws =
                                s_ws[gk * FN + wgn * (NT * 16u) + tn * 16u + r4d_fp8::frag_c(lane)];
#pragma unroll
                            for (uint32_t t = 0; t < 8u; ++t) {
                                const float sa = s_as[gk * FM + wgm * (MT * 16u) + tm * 16u +
                                                      r4d_fp8::acc_row(lane, t)];
                                out[tm][tn][t] += r4d_fp8::acc_val<true>(raw[t]) * ws * sa;
                            }
                        }
                    }
                }
            }
        }

        if (wlive) {
#pragma unroll
            for (uint32_t tn = 0; tn < NT; ++tn) {
                const uint32_t col = n0 + wgn * (NT * 16u) + tn * 16u + r4d_fp8::frag_c(lane);
                if (wn0 + (col - n0) < n_q && col < (uint32_t)p.N) {
#pragma unroll
                    for (uint32_t tm = 0; tm < MT; ++tm) {
#pragma unroll
                        for (uint32_t t = 0; t < 8u; ++t) {
                            const uint32_t r = wgm * (MT * 16u) + tm * 16u + r4d_fp8::acc_row(lane, t);
                            const uint32_t row = r0 + r;
                            if (r < FM && row >= rs && row < re)
                                y[(size_t)row * y_ld + col] = f32_to_bf16(out[tm][tn][t]);
                        }
                    }
                }
            }
        }

        rs = re;
        if constexpr (AL) break;
    }

    if constexpr (!AL) {
        if (rN > live) zero_tail(r0 > live ? r0 : live, rN);
    }
}

/* ---- Direct-B f16 WMMA prefill: 64x256 (8w) / 64x128 (4w), run-aligned map, a_group 128 -----
 *
 * `e2m1` magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6} (1 mantissa bit) multiplied by any finite OCP-E4M3
 * scale (3 mantissa bits, [2^-9, 448]) have at most 5 mantissa bits and lie in {0} U [2^-10, 2688],
 * strictly inside IEEE f16's exact normal range [2^-14, 65504]; int8 activations [-128..127] are
 * also 100% exact in f16. Staged A rows are widened once into f16 in LDS (`s_a`, pitch 2*FK+16),
 * while each wave reads its column weights directly from global memory, exchanges half-wave B
 * chunks via `v_permlanex16`, unpacks & scales e2m1*e4m3 into f16 via `v_perm_b32` + `v_pk_mul_f16`,
 * and accumulates all 8 16-K groups directly in `v_wmma_f32_16x16x16_f16` with carried row scales. */

typedef _Float16 nv_v8h __attribute__((ext_vector_type(8)));
typedef _Float16 nv_h2  __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float nv_rcp_f32(float x) {
#ifdef __HIP__
    return __frcp_rn(x);
#else
    return 1.0f / x;
#endif
}

__device__ __forceinline__ uint32_t nv_dup_f16(uint32_t pks, bool hi) {
#if defined(__HIP__) && __has_builtin(__builtin_amdgcn_perm)
    return __builtin_amdgcn_perm(pks, pks, hi ? 0x03020302u : 0x01000100u);
#else
    const uint32_t s = hi ? (pks >> 16) : (pks & 0xFFFFu);
    return s | (s << 16);
#endif
}

__device__ __forceinline__ uint32_t nv_e4m3x2_to_f16x2(uint32_t b2) {
#if defined(__HIP__) && __has_builtin(__builtin_amdgcn_cvt_pkrtz)
    const v2f vf = r4d_fp8::e4m3_pk<false>(b2);
    const auto h2 = __builtin_amdgcn_cvt_pkrtz(vf[0], vf[1]);
    uint32_t u;
    __builtin_memcpy(&u, &h2, 4);
    return u;
#else
    const _Float16 h0 = (_Float16)r4d_fp8::e4m3(b2 & 0xFFu);
    const _Float16 h1 = (_Float16)r4d_fp8::e4m3((b2 >> 8) & 0xFFu);
    uint16_t u0, u1;
    __builtin_memcpy(&u0, &h0, 2);
    __builtin_memcpy(&u1, &h1, 2);
    return (uint32_t)u0 | ((uint32_t)u1 << 16);
#endif
}

__device__ __forceinline__ r4d_u32x4 nv_i8x8_to_f16x4_masked(uint32_t lo_raw, uint32_t hi_raw,
                                                            uint32_t mask) {
#if defined(__HIP__) && __has_builtin(__builtin_amdgcn_perm)
    const uint32_t lo = lo_raw ^ 0x80808080u;
    const uint32_t hi = hi_raw ^ 0x80808080u;
    const nv_h2 bias = { (_Float16)(-1152.0f), (_Float16)(-1152.0f) };
    const uint32_t p0 = __builtin_amdgcn_perm(0x64646464u, lo, 0x04010400u);
    const uint32_t p1 = __builtin_amdgcn_perm(0x64646464u, lo, 0x04030402u);
    const uint32_t p2 = __builtin_amdgcn_perm(0x64646464u, hi, 0x04010400u);
    const uint32_t p3 = __builtin_amdgcn_perm(0x64646464u, hi, 0x04030402u);
    const uint32_t q0 = __builtin_bit_cast(uint32_t, __builtin_bit_cast(nv_h2, p0) + bias) & mask;
    const uint32_t q1 = __builtin_bit_cast(uint32_t, __builtin_bit_cast(nv_h2, p1) + bias) & mask;
    const uint32_t q2 = __builtin_bit_cast(uint32_t, __builtin_bit_cast(nv_h2, p2) + bias) & mask;
    const uint32_t q3 = __builtin_bit_cast(uint32_t, __builtin_bit_cast(nv_h2, p3) + bias) & mask;
    return r4d_u32x4{ q0, q1, q2, q3 };
#else
    r4d_u32x4 out{};
    if (!mask) return out;
    for (int q = 0; q < 4; ++q) {
        const uint32_t src = (q < 2) ? (lo_raw >> (q * 16)) : (hi_raw >> ((q - 2) * 16));
        const _Float16 h0 = (_Float16)(float)(int8_t)(src & 0xFFu);
        const _Float16 h1 = (_Float16)(float)(int8_t)((src >> 8) & 0xFFu);
        uint16_t u0, u1;
        __builtin_memcpy(&u0, &h0, 2);
        __builtin_memcpy(&u1, &h1, 2);
        out[q] = ((uint32_t)u0 | ((uint32_t)u1 << 16)) & mask;
    }
    return out;
#endif
}

__device__ __forceinline__ nv_v8h nv_unpack8_f16_scaled(uint32_t w_bits, uint32_t sc32) {
#if defined(__HIP__) && __has_builtin(__builtin_amdgcn_perm)
    const nv_h2 s2 = __builtin_bit_cast(nv_h2, sc32);
    const uint32_t me = w_bits & 0x07070707u;
    const uint32_t mo = (w_bits >> 4) & 0x07070707u;
    const uint32_t he = __builtin_amdgcn_perm(0x46444240u, 0x3E3C3800u, me) | ((w_bits << 4) & 0x80808080u);
    const uint32_t ho = __builtin_amdgcn_perm(0x46444240u, 0x3E3C3800u, mo) | (w_bits & 0x80808080u);
    const nv_h2 h0 = __builtin_bit_cast(nv_h2, __builtin_amdgcn_perm(ho, he, 0x040C000Cu)) * s2;
    const nv_h2 h1 = __builtin_bit_cast(nv_h2, __builtin_amdgcn_perm(ho, he, 0x050C010Cu)) * s2;
    const nv_h2 h2 = __builtin_bit_cast(nv_h2, __builtin_amdgcn_perm(ho, he, 0x060C020Cu)) * s2;
    const nv_h2 h3 = __builtin_bit_cast(nv_h2, __builtin_amdgcn_perm(ho, he, 0x070C030Cu)) * s2;
    nv_v8h out;
    out[0] = h0[0]; out[1] = h0[1];
    out[2] = h1[0]; out[3] = h1[1];
    out[4] = h2[0]; out[5] = h2[1];
    out[6] = h3[0]; out[7] = h3[1];
    return out;
#else
    constexpr float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const uint16_t su = (uint16_t)(sc32 & 0xFFFFu);
    _Float16 sf;
    __builtin_memcpy(&sf, &su, 2);
    nv_v8h out;
    for (int i = 0; i < 8; ++i) {
        const uint32_t c = (w_bits >> (4 * i)) & 15u;
        const float v = (c & 8u) ? -kMag[c & 7u] : kMag[c & 7u];
        out[i] = (_Float16)v * sf;
    }
    return out;
#endif
}

__device__ __forceinline__ nv_v8h nv_load_v8h(const void* p) {
#ifdef __HIP__
    return *(const nv_v8h*)p;
#else
    nv_v8h r;
    __builtin_memcpy(&r, p, sizeof r);
    return r;
#endif
}

__device__ __forceinline__ r4d_fp8::f8v nv_wmma_f16(nv_v8h a, nv_v8h b, r4d_fp8::f8v c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

template <uint32_t FM, uint32_t FN, uint32_t WGM, uint32_t WGN, uint32_t FK, bool FULL_MT>
__device__ __forceinline__ void nvfp4_f16_kloop(
    const NvArgs& p, uint32_t n0, uint32_t r0, uint32_t rN, uint32_t n_q, uint32_t k_q,
    uint32_t wn0, uint32_t live_mt, const uint8_t* Wp, const uint8_t* Sbase, float level,
    uint16_t* __restrict__ y, size_t y_ld, uint8_t* __restrict__ s_a,
    const int32_t* __restrict__ s_tok, float* __restrict__ s_as_buf, uint8_t (&s_az)[2][FM]) {

    constexpr uint32_t NTHR   = WGM * WGN * 32u;
    constexpr uint32_t MT     = FM / (WGM * 16u);
    constexpr uint32_t NT     = FN / (WGN * 16u);
    constexpr uint32_t NGK    = FK / 16u;
    constexpr uint32_t APITCH = FK * 2u + 16u;
    constexpr uint32_t kTPR   = 4u;
    constexpr uint32_t kSeg   = FK / kTPR;
    constexpr uint32_t kRPP   = NTHR / kTPR;
    constexpr uint32_t kAP    = FM / kRPP;
    static_assert(FM == 64u && WGM == 1u && MT == 4u && FK == 128u && kRPP <= FM,
                  "64-row f16 prefill tile parameters");

    const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
    const uint32_t wgn = wave % WGN, h = lane >> 4, frag_col = lane & 15u;
    const uint32_t sn0 = tid >> 2, sk = (tid & 3u) * kSeg;

    const uint32_t fc_part = n_q / (uint32_t)p.parts;
    const uint32_t fy_part = (uint32_t)p.N / (uint32_t)p.parts;
    const uint32_t j0_part = n0 - (n0 / fy_part) * fy_part;

    uint32_t wcol[NT], bcol[NT];
    bool     bok[NT];
    const uint8_t* brow[NT];
    const uint8_t* srow[NT];
#pragma unroll
    for (uint32_t tn = 0; tn < NT; ++tn) {
        wcol[tn] = wn0 + wgn * (NT * 16u) + tn * 16u + frag_col;
        bcol[tn] = n0  + wgn * (NT * 16u) + tn * 16u + frag_col;
        bok[tn]  = (j0_part + wgn * (NT * 16u) + tn * 16u + frag_col) < fc_part;
        brow[tn] = Wp + (size_t)(bok[tn] ? wcol[tn] : 0u) * (size_t)(k_q >> 1) + h * (FK >> 2);
        srow[tn] = Sbase + (size_t)(bok[tn] ? wcol[tn] : 0u) * (size_t)(k_q >> 4);
    }

    r4d_fp8::f8v out[MT][NT];
#pragma unroll
    for (uint32_t tm = 0; tm < MT; ++tm)
#pragma unroll
        for (uint32_t tn = 0; tn < NT; ++tn) out[tm][tn] = r4d_fp8::acc_zero();

    r4d_u32x4 apre[kAP][2]{};
    r4d_u32x4 bpre[NT][2]{};
    uint64_t  pwsu[NT]{};
    float     pas = 0.0f;

    auto issue = [&](uint32_t c0) {
#pragma unroll
        for (uint32_t ap = 0; ap < kAP; ++ap) {
            const uint32_t sr  = sn0 + ap * kRPP;
            const int32_t  tok = s_tok[sr];
            const bool     ok  = tok >= 0;
            const uint8_t* src = (const uint8_t*)p.a + (size_t)(ok ? tok : 0) * p.a_ld + c0 + sk;
            apre[ap][0] = nv_gload_v4(src);
            apre[ap][1] = nv_gload_v4(src + 16);
        }
#pragma unroll
        for (uint32_t tn = 0; tn < NT; ++tn) {
            if (bok[tn]) {
                bpre[tn][0] = nv_gload_v4(brow[tn] + (c0 >> 1));
                bpre[tn][1] = nv_gload_v4(brow[tn] + (c0 >> 1) + 16u);
                pwsu[tn]    = nv_gload_u64(srow[tn] + (c0 >> 4));
            } else {
                bpre[tn][0] = r4d_u32x4{ 0u, 0u, 0u, 0u };
                bpre[tn][1] = r4d_u32x4{ 0u, 0u, 0u, 0u };
                pwsu[tn]    = 0ull;
            }
        }
        if (tid < FM) {
            const int32_t tok = s_tok[tid];
            pas = (tok >= 0) ? p.as[(size_t)tok * p.as_ld + (c0 >> 7)] : 0.0f;
            s_az[(c0 >> 7) & 1u][tid] = (pas == 0.0f) ? 1u : 0u;
        }
    };

    issue(0u);
    for (uint32_t c0 = 0; c0 < k_q; c0 += FK) {
        const uint32_t kb_par = (c0 >> 7) & 1u;
        lds_barrier();
#ifdef __HIP__
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
#endif

        if (tid < FM) {
            const float cur = s_as_buf[tid];
            if (pas == 0.0f) {
                s_as_buf[FM + tid] = 1.0f;
            } else {
                s_as_buf[FM + tid] = (cur == 0.0f) ? 0.0f : (cur * nv_rcp_f32(pas));
                s_as_buf[tid] = pas;
            }
        }

#pragma unroll
        for (uint32_t ap = 0; ap < kAP; ++ap) {
            const uint32_t sr   = sn0 + ap * kRPP;
            const bool     ok   = (s_tok[sr] >= 0) && (s_az[kb_par][sr] == 0u);
            const uint32_t mask = ok ? 0xFFFFFFFFu : 0u;
            *(r4d_u32x4*)(s_a + (size_t)sr * APITCH + (sk + 0u) * 2u) =
                nv_i8x8_to_f16x4_masked(apre[ap][0][0], apre[ap][0][1], mask);
            *(r4d_u32x4*)(s_a + (size_t)sr * APITCH + (sk + 8u) * 2u) =
                nv_i8x8_to_f16x4_masked(apre[ap][0][2], apre[ap][0][3], mask);
            *(r4d_u32x4*)(s_a + (size_t)sr * APITCH + (sk + 16u) * 2u) =
                nv_i8x8_to_f16x4_masked(apre[ap][1][0], apre[ap][1][1], mask);
            *(r4d_u32x4*)(s_a + (size_t)sr * APITCH + (sk + 24u) * 2u) =
                nv_i8x8_to_f16x4_masked(apre[ap][1][2], apre[ap][1][3], mask);
        }

        uint32_t wk[NT][NGK];
        uint32_t ws_pk[NT][NGK / 2u];
#pragma unroll
        for (uint32_t tn = 0; tn < NT; ++tn) {
            const uint32_t o[8] = {
                bpre[tn][0][0], bpre[tn][0][1], bpre[tn][0][2], bpre[tn][0][3],
                bpre[tn][1][0], bpre[tn][1][1], bpre[tn][1][2], bpre[tn][1][3]
            };
#pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t mine = h ? o[2u * j + 1u] : o[2u * j];
                const uint32_t send = h ? o[2u * j]      : o[2u * j + 1u];
                const uint32_t recv = r4d_xor_lane_u32<16>(send);
                wk[tn][j]      = h ? recv : mine;
                wk[tn][4u + j] = h ? mine : recv;
            }
            const uint32_t slo = (uint32_t)(pwsu[tn] & 0xFFFFFFFFu);
            const uint32_t shi = (uint32_t)(pwsu[tn] >> 32);
            ws_pk[tn][0] = nv_e4m3x2_to_f16x2(slo);
            ws_pk[tn][1] = nv_e4m3x2_to_f16x2(slo >> 16);
            ws_pk[tn][2] = nv_e4m3x2_to_f16x2(shi);
            ws_pk[tn][3] = nv_e4m3x2_to_f16x2(shi >> 16);
        }
        lds_barrier();

        if (c0 > 0u) {
#pragma unroll
            for (uint32_t tm = 0; tm < MT; ++tm) {
                if (!FULL_MT && tm >= live_mt) break;
                float r[8];
#pragma unroll
                for (uint32_t t = 0; t < 8u; ++t) r[t] = s_as_buf[FM + tm * 16u + h * 8u + t];
#pragma unroll
                for (uint32_t t = 0; t < 8u; ++t)
#pragma unroll
                    for (uint32_t tn = 0; tn < NT; ++tn) out[tm][tn][t] *= r[t];
                sched_barrier();
            }
        }

        const uint32_t a_lane_off = frag_col * APITCH + h * 16u;
#pragma unroll
        for (uint32_t gk = 0; gk < NGK; ++gk) {
            const uint8_t* a_gk = s_a + a_lane_off + gk * 32u;
            nv_v8h bf[NT];
#pragma unroll
            for (uint32_t tn = 0; tn < NT; ++tn)
                bf[tn] = nv_unpack8_f16_scaled(wk[tn][gk], nv_dup_f16(ws_pk[tn][gk >> 1], (gk & 1u) != 0u));

            if constexpr (FULL_MT) {
                nv_v8h af0 = nv_load_v8h(a_gk);
                nv_v8h af1 = nv_load_v8h(a_gk + 16u * APITCH);
                sched_barrier();
#pragma unroll
                for (uint32_t tn = 0; tn < NT; ++tn) out[0][tn] = nv_wmma_f16(af0, bf[tn], out[0][tn]);
                af0 = nv_load_v8h(a_gk + 32u * APITCH);
#pragma unroll
                for (uint32_t tn = 0; tn < NT; ++tn) out[1][tn] = nv_wmma_f16(af1, bf[tn], out[1][tn]);
                af1 = nv_load_v8h(a_gk + 48u * APITCH);
#pragma unroll
                for (uint32_t tn = 0; tn < NT; ++tn) out[2][tn] = nv_wmma_f16(af0, bf[tn], out[2][tn]);
#pragma unroll
                for (uint32_t tn = 0; tn < NT; ++tn) out[3][tn] = nv_wmma_f16(af1, bf[tn], out[3][tn]);
            } else {
#pragma unroll
                for (uint32_t tm = 0; tm < MT; ++tm) {
                    if (tm >= live_mt) break;
                    const nv_v8h af = nv_load_v8h(a_gk + (size_t)(tm * 16u) * APITCH);
#pragma unroll
                    for (uint32_t tn = 0; tn < NT; ++tn) out[tm][tn] = nv_wmma_f16(af, bf[tn], out[tm][tn]);
                }
            }
            if (gk == (NGK / 2u - 1u) && (c0 + FK < k_q)) issue(c0 + FK);
            sched_barrier();
        }
    }

#pragma unroll
    for (uint32_t tn = 0; tn < NT; ++tn) {
        if (bcol[tn] < (uint32_t)p.N && bok[tn]) {
#pragma unroll
            for (uint32_t tm = 0; tm < MT; ++tm) {
                if (!FULL_MT && tm >= live_mt) break;
#pragma unroll
                for (uint32_t t = 0; t < 8u; ++t) {
                    const uint32_t row = r0 + tm * 16u + h * 8u + t;
                    if (row < rN) {
                        const float val = out[tm][tn][t] * level * s_as_buf[tm * 16u + h * 8u + t];
                        y[(size_t)row * y_ld + bcol[tn]] = f32_to_bf16(val);
                    }
                }
            }
        }
    }
}

template <uint32_t FM, uint32_t FN, uint32_t WGM, uint32_t WGN, class CH>
__global__ __launch_bounds__(WGM * WGN * 32u) void nvfp4_moe_prefill_f16_kernel(
    NvArgs p, const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ y, size_t y_ld, const uint32_t* __restrict__ map, uint32_t T) {
    if (CH::on) r4d_chain_wait();
    constexpr uint32_t FK     = kNvPfK;
    constexpr uint32_t NTHR   = WGM * WGN * 32u;
    constexpr uint32_t MT     = FM / (WGM * 16u);
    constexpr uint32_t APITCH = FK * 2u + 16u;
    __shared__ uint8_t s_a[FM * APITCH];
    __shared__ int32_t s_tok[FM];
    __shared__ float   s_as_buf[2u * FM];
    __shared__ uint8_t s_az[2][FM];

    const uint32_t tid   = threadIdx.x;
    const uint32_t b_row = blockIdx.y;
    const uint32_t n0    = blockIdx.x * FN;
    const uint32_t live_raw = (uint32_t)off[p.n_expert];
    const uint32_t live = live_raw < T ? live_raw : T;

    bool run = b_row < map[0];
    uint32_t e = 0, r0 = 0, rN = 0;
    if (run) {
        const uint32_t* ent = map + 1 + 2 * b_row;
        e  = ent[0];
        r0 = ent[1];
        if (e == kNvMapTail) {
            const uint32_t rT = (r0 + FM < T) ? (r0 + FM) : T;
            for (uint32_t r = (r0 > live ? r0 : live); r < rT; ++r)
                for (uint32_t c = tid; c < FN && n0 + c < (uint32_t)p.N; c += NTHR)
                    y[(size_t)r * y_ld + n0 + c] = 0;
            run = false;
        } else {
            const uint32_t oe = (uint32_t)off[e + 1];
            const uint32_t al_end = oe < live ? oe : live;
            rN = (r0 + FM < al_end) ? (r0 + FM) : al_end;
            if (rN <= r0) run = false;
        }
    }
    if (run) {
        const uint32_t q = e & 1u, ix = e >> 1;
        const uint32_t n_q = (uint32_t)p.N_c[q], k_q = (uint32_t)p.K_c[q];
        uint32_t wn0 = 0;
        bool live_part = false;
        nv_part_rows(p, n_q, n0, &wn0, &live_part);
        if (!live_part) {
            for (uint32_t r = r0; r < rN; ++r)
                for (uint32_t c = tid; c < FN && n0 + c < (uint32_t)p.N; c += NTHR)
                    y[(size_t)r * y_ld + n0 + c] = 0;
        } else {
            const uint8_t* Wp    = nv_uw_ptr(nv_entry(p.w[q], ix));
            const uint8_t* Sbase = nv_uw_ptr(nv_entry(p.ws[q], ix));
            float level;
            __builtin_memcpy(&level, Sbase + (size_t)n_q * (size_t)(k_q >> 4), sizeof level);
            if (tid < FM) {
                const uint32_t r = r0 + tid;
                int32_t tok = -1;
                if (r < rN) {
                    const int32_t flat = sorted[r];
                    if (flat >= 0) {
                        const int64_t t = p.a_sorted ? (int64_t)r : (int64_t)flat / p.top_k;
                        if (t < p.AR) tok = (int32_t)t;
                    }
                }
                s_tok[tid] = tok;
                s_as_buf[tid] = 0.0f;
                s_as_buf[FM + tid] = 1.0f;
            }
            __syncthreads();

            const uint32_t live_mt = (uint32_t)nv_readfirstlane((int)((rN - r0 + 15u) >> 4));
            if (live_mt >= MT) {
                nvfp4_f16_kloop<FM, FN, WGM, WGN, FK, true>(
                    p, n0, r0, rN, n_q, k_q, wn0, live_mt, Wp, Sbase, level, y, y_ld,
                    s_a, s_tok, s_as_buf, s_az);
            } else {
                nvfp4_f16_kloop<FM, FN, WGM, WGN, FK, false>(
                    p, n0, r0, rN, n_q, k_q, wn0, live_mt, Wp, Sbase, level, y, y_ld,
                    s_a, s_tok, s_as_buf, s_az);
            }
        }
    }
    if (CH::on) r4d_chain_done();
}

template <uint32_t FM, uint32_t FN, uint32_t WGM, uint32_t WGN, bool AL, bool AG128, class CH>
__global__ __launch_bounds__(WGM * WGN * 32u) void nvfp4_moe_prefill_kernel(
    NvArgs p, const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ y, size_t y_ld, const uint32_t* __restrict__ map, uint32_t T) {
    if (CH::on) r4d_chain_wait();
    nvfp4_moe_prefill_body<FM, FN, WGM, WGN, AL, AG128>(p, sorted, off, y, y_ld, map, T);
    if (CH::on) r4d_chain_done();
}

/* A class's operand: a rank-3 stacked tensor [entries, rows, cols], or a pointer table with stride
 * 0 on its first axis. `bits` is the element width (4 for the e2m1 codes). */
bool nv_table(const RadTensor* t, uint32_t dtype, int64_t entries, int64_t rows, int64_t cols,
              int bits, NvTab* out) {
    if (!t || !t->data || t->rank != 3 || t->dtype != dtype) return false;
    if (t->shape[0] < entries || t->shape[1] < rows || t->shape[2] < cols) return false;
    if (t->stride[2] != 1 || t->stride[1] != cols) return false;
    out->data = (const uint8_t*)t->data;
    out->s0 = (size_t)((t->stride[0] * (int64_t)bits) / 8);
    /* a stacked entry is a whole number of bytes */
    if (t->stride[0] != 0 && (t->stride[0] * (int64_t)bits) % 8) return false;
    return true;
}

/* A class's scale blobs (r4d.nvfp4.s): `rows` rows of `cols` bytes, then the level's row. The pitch
 * is derived, not read, as moe_gemm_launch derives the codebook forms': the core hands the view at
 * the extent the plugin declared, the logical [N, K] with row stride K, while a stacked plane may
 * carry the blob's own [N + 1, K / 16]. The layout tag checked at load is what makes the bytes the
 * blob; here the entries need only be distinct. */
bool nv_scale_table(const RadTensor* t, int64_t entries, int64_t rows, int64_t cols, NvTab* out) {
    if (!t || !t->data || t->rank != 3 || t->dtype != RAD_U8) return false;
    if (t->shape[0] < entries || t->shape[1] < rows || t->stride[2] != 1) return false;
    if (t->stride[0] != 0 && t->stride[0] < (rows + 1) * cols) return false;
    out->data = (const uint8_t*)t->data;
    out->s0 = (size_t)t->stride[0];
    return true;
}

}  // namespace

extern "C" int64_t r4d_rad_moe_gemm_nvfp4_scratch(const RadArgs* a) {
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k    = r4d_geti_or(a, "top_k", 0);
    const long long M        = r4d_geti_or(a, "M", 0);
    if (n_expert <= 0 || n_expert > kNvMaxExpert || top_k <= 0 || M <= 0) return 0;
    const long long N0    = r4d_geti_or(a, "N", 0), K0 = r4d_geti_or(a, "K", 0);
    const long long N1    = r4d_geti_or(a, "N_odd", N0), K1 = r4d_geti_or(a, "K_odd", K0);
    const long long parts = r4d_geti_or(a, "parts", 1);
    const long long T     = M * top_k;
    uint32_t pfM = 0, pfN = 0;
    if (!nv_pick_prefill(T, n_expert, parts, N0, N1, K0, K1, &pfM, &pfN)) return 0;
    return nv_blockmap_bytes(nv_blockmap_cap(T, n_expert, kNvPfM32));
}

extern "C" int r4d_rad_moe_gemm_nvfp4a8i8(const RadArgs* a, RadStream stream) {
    const RadTensor* Aq = r4d_opt(a, 0);
    const RadTensor* As = r4d_opt(a, 1);
    const RadTensor* Wc[2] = { r4d_opt(a, 2), r4d_opt(a, 6) };
    const RadTensor* Ws[2] = { r4d_opt(a, 3), r4d_opt(a, 7) };
    const RadTensor* sorted = r4d_opt(a, 4);
    const RadTensor* off = r4d_opt(a, 5);
    const RadTensor* Y = r4d_opt(a, 8);
    if (!Aq || !As || !Wc[0] || !Ws[0] || !Wc[1] || !Ws[1] || !sorted || !off || !Y)
        return R4D_NO(RAD_E_INVAL);
    if (!Aq->data || !As->data || !sorted->data || !off->data || !Y->data)
        return R4D_NO(RAD_E_INVAL);

    const long long N0 = r4d_geti_or(a, "N", 0), K0 = r4d_geti_or(a, "K", 0);
    const long long N1 = r4d_geti_or(a, "N_odd", N0), K1 = r4d_geti_or(a, "K_odd", K0);
    const long long parts = r4d_geti_or(a, "parts", 1);
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0);
    const long long group = r4d_geti_or(a, "group", 128);
    const long long a_group = r4d_geti_or(a, "a_group", group);
    if (parts <= 0 || n_expert <= 0 || top_k <= 0 || N0 <= 0 || K0 <= 0 || N1 <= 0 || K1 <= 0)
        return R4D_NO(RAD_E_SHAPE);
    /* The layout is whole 16-wide groups, four to a word of scale bytes, and the activation's
     * scale covers whole groups. */
    if (K0 % 64 || K1 % 64 || N0 % parts || N1 % parts || a_group % 16 || group % 16 ||
        a_group <= 0)
        return R4D_NO(RAD_E_SHAPE);
    const char* ord = r4d_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return R4D_NO(RAD_E_UNSUPPORTED);

    const long long N = N0 > N1 ? N0 : N1, K = K0 > K1 ? K0 : K1;
    if (K % a_group) return R4D_NO(RAD_E_SHAPE);
    if (Aq->dtype != RAD_I8 || As->dtype != RAD_F32 || Y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (sorted->dtype != RAD_I32 || off->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    long long T = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &T);
    if (y_ld < 0 || T <= 0) return R4D_NO(RAD_E_STRIDE);
    long long M = 0;
    const long long a_ld = r4d_rows_pitch(Aq, K, &M);
    if (a_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    long long Ms = 0;
    const long long as_ld = r4d_rows_pitch(As, K / a_group, &Ms);
    if (as_ld < 0 || Ms < M) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(sorted) < T || r4d_numel(off) < n_expert + 1) return R4D_NO(RAD_E_SHAPE);
    const long long AR = a_sorted ? T : (T + top_k - 1) / top_k;
    if (M < AR) return R4D_NO(RAD_E_SHAPE);

    NvArgs p{};
    p.a = (const int8_t*)Aq->data;
    p.a_ld = (size_t)a_ld;
    p.as = (const float*)As->data;
    p.as_ld = (size_t)as_ld;
    p.AR = AR;
    p.a_sorted = a_sorted ? 1 : 0;
    p.a_group = (int)a_group;
    p.top_k = (int)top_k;
    p.n_expert = (int)n_expert;
    p.N = (int)N;
    p.parts = (int)parts;
    const long long Nc[2] = { N0, N1 }, Kc[2] = { K0, K1 };
    for (int q = 0; q < 2; ++q) {
        const long long entries = (n_expert + 1 - q) / 2;   /* ceil((n_expert - q) / 2) */
        p.N_c[q] = (int)Nc[q];
        p.K_c[q] = (int)Kc[q];
        /* codes: [entries, N_q, K_q] e2m1, two codes a byte; scales: [entries, N_q + 1, K_q / 16]
         * bytes, each expert's last row holding its f32 level */
        if (!nv_table(Wc[q], RAD_FP4E2M1, entries, Nc[q], Kc[q], 4, &p.w[q]))
            return R4D_NO(RAD_E_SHAPE);
        if (!nv_scale_table(Ws[q], entries, Nc[q], Kc[q] / 16, &p.ws[q]))
            return R4D_NO(RAD_E_SHAPE);
    }

    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    uint32_t pfM = 0, pfN = 0;
    const bool use_prefill = nv_pick_prefill(T, n_expert, parts, N0, N1, K0, K1, &pfM, &pfN);
    const bool ag128 = (a_group == 128);

    int emit_rc = RAD_OK;
    if (use_prefill) {
        long long cap = nv_blockmap_cap(T, n_expert, pfM);
        uint32_t* map = (n_expert <= kNvMaxExpert && a->scratch &&
                         a->scratch_bytes >= nv_blockmap_bytes(cap))
                            ? (uint32_t*)a->scratch
                            : nullptr;
        /* With the run-aligned map at a_group 128, a 64-row Direct-B f16 WMMA tile decodes each
         * weight byte for twice the rows of 32-row tiles (using 256-column 8-wave tiles when each
         * part width divides 256, else 128-column 4-wave tiles when it divides 128). */
        if (map && ag128 && ((N0 / parts) % kNvPfN256) == 0 && ((N1 / parts) % kNvPfN256) == 0) {
            pfM = kNvPfM64;
            pfN = kNvPfN256;
            cap = nv_blockmap_cap(T, n_expert, pfM);
        } else if (map && ag128 && ((N0 / parts) % kNvPfN128) == 0 && ((N1 / parts) % kNvPfN128) == 0) {
            pfM = kNvPfM64;
            pfN = kNvPfN128;
            cap = nv_blockmap_cap(T, n_expert, pfM);
        }
        const unsigned ny = (unsigned)(N / pfN);
        if (map) {
            emit_rc = R4D_EMIT(stream,
                               (chain ? nvfp4_blockmap_kernel<r4dchain>
                                      : nvfp4_blockmap_kernel<R4dUnchained>),
                               dim3(1), dim3(256), 0, (const int32_t*)off->data,
                               (uint32_t)n_expert, (uint32_t)T, pfM, (uint32_t)cap, map);
            if (emit_rc != RAD_OK) return emit_rc;
        }
        const unsigned gx = map ? (unsigned)cap : (unsigned)((T + pfM - 1) / pfM);
#define R4D_NV_PF(FM_, FN_, WGM_, WGN_, AL_, AG_, CH_)                                          \
    R4D_EMIT(stream, (nvfp4_moe_prefill_kernel<FM_, FN_, WGM_, WGN_, AL_, AG_, CH_>),           \
             dim3(gx, ny), dim3((WGM_) * (WGN_) * 32u), 0, p, (const int32_t*)sorted->data,     \
             (const int32_t*)off->data, (uint16_t*)Y->data, (size_t)y_ld,                       \
             (const uint32_t*)map, (uint32_t)T)
        if (pfM == kNvPfM64 && pfN == kNvPfN256) {
#define R4D_NV_F16_256(CH_)                                                                     \
    R4D_EMIT(stream, (nvfp4_moe_prefill_f16_kernel<64u, 256u, 1u, 8u, CH_>), dim3(ny, gx),      \
             dim3(256), 0, p, (const int32_t*)sorted->data, (const int32_t*)off->data,          \
             (uint16_t*)Y->data, (size_t)y_ld, (const uint32_t*)map, (uint32_t)T)
            emit_rc = chain ? R4D_NV_F16_256(r4dchain) : R4D_NV_F16_256(R4dUnchained);
#undef R4D_NV_F16_256
        } else if (pfM == kNvPfM64) {
#define R4D_NV_F16_128(CH_)                                                                     \
    R4D_EMIT(stream, (nvfp4_moe_prefill_f16_kernel<64u, 128u, 1u, 4u, CH_>), dim3(ny, gx),      \
             dim3(128), 0, p, (const int32_t*)sorted->data, (const int32_t*)off->data,          \
             (uint16_t*)Y->data, (size_t)y_ld, (const uint32_t*)map, (uint32_t)T)
            emit_rc = chain ? R4D_NV_F16_128(r4dchain) : R4D_NV_F16_128(R4dUnchained);
#undef R4D_NV_F16_128
        } else if (pfN == kNvPfN256) {
            if (map) {
                emit_rc = ag128 ? (chain ? R4D_NV_PF(32u, 256u, 2u, 8u, true,  true,  r4dchain)
                                         : R4D_NV_PF(32u, 256u, 2u, 8u, true,  true,  R4dUnchained))
                                : (chain ? R4D_NV_PF(32u, 256u, 2u, 8u, true,  false, r4dchain)
                                         : R4D_NV_PF(32u, 256u, 2u, 8u, true,  false, R4dUnchained));
            } else {
                emit_rc = ag128 ? (chain ? R4D_NV_PF(32u, 256u, 2u, 8u, false, true,  r4dchain)
                                         : R4D_NV_PF(32u, 256u, 2u, 8u, false, true,  R4dUnchained))
                                : (chain ? R4D_NV_PF(32u, 256u, 2u, 8u, false, false, r4dchain)
                                         : R4D_NV_PF(32u, 256u, 2u, 8u, false, false, R4dUnchained));
            }
        } else if (pfN == kNvPfN128) {
            if (map) {
                emit_rc = ag128 ? (chain ? R4D_NV_PF(32u, 128u, 2u, 4u, true,  true,  r4dchain)
                                         : R4D_NV_PF(32u, 128u, 2u, 4u, true,  true,  R4dUnchained))
                                : (chain ? R4D_NV_PF(32u, 128u, 2u, 4u, true,  false, r4dchain)
                                         : R4D_NV_PF(32u, 128u, 2u, 4u, true,  false, R4dUnchained));
            } else {
                emit_rc = ag128 ? (chain ? R4D_NV_PF(32u, 128u, 2u, 4u, false, true,  r4dchain)
                                         : R4D_NV_PF(32u, 128u, 2u, 4u, false, true,  R4dUnchained))
                                : (chain ? R4D_NV_PF(32u, 128u, 2u, 4u, false, false, r4dchain)
                                         : R4D_NV_PF(32u, 128u, 2u, 4u, false, false, R4dUnchained));
            }
        } else {
            if (map) {
                emit_rc = ag128 ? (chain ? R4D_NV_PF(32u, 64u,  2u, 4u, true,  true,  r4dchain)
                                         : R4D_NV_PF(32u, 64u,  2u, 4u, true,  true,  R4dUnchained))
                                : (chain ? R4D_NV_PF(32u, 64u,  2u, 4u, true,  false, r4dchain)
                                         : R4D_NV_PF(32u, 64u,  2u, 4u, true,  false, R4dUnchained));
            } else {
                emit_rc = ag128 ? (chain ? R4D_NV_PF(32u, 64u,  2u, 4u, false, true,  r4dchain)
                                         : R4D_NV_PF(32u, 64u,  2u, 4u, false, true,  R4dUnchained))
                                : (chain ? R4D_NV_PF(32u, 64u,  2u, 4u, false, false, r4dchain)
                                         : R4D_NV_PF(32u, 64u,  2u, 4u, false, false, R4dUnchained));
            }
        }
#undef R4D_NV_PF
    } else {
        const bool fast256 =
            ag128 && (K0 % 128) == 0 && (K1 % 128) == 0 && (a_ld % 16) == 0 &&
            rad_ptr_aligned(Aq->data, 16) && rad_ptr_aligned(Wc[0]->data, 16) &&
            rad_ptr_aligned(Wc[1]->data, 16) && (p.w[0].s0 % 16) == 0 && (p.w[1].s0 % 16) == 0;
        const unsigned groups = (unsigned)((N + kNvDecCols - 1) / kNvDecCols);
        if (fast256 && n_expert <= kNvMaxExpert && T > top_k) {
            const long long mt = (T < n_expert ? T : n_expert + T / kNvMdecRM) + 1;
            emit_rc = R4D_EMIT(stream,
                               (chain ? nvfp4_moe_mdec_kernel<r4dchain>
                                      : nvfp4_moe_mdec_kernel<R4dUnchained>),
                               dim3((unsigned)mt, groups), dim3(kNvDecBlock), 0, p,
                               (const int32_t*)sorted->data, (const int32_t*)off->data,
                               (uint16_t*)Y->data, (size_t)y_ld, (uint32_t)T);
        } else if (fast256) {
            emit_rc = R4D_EMIT(stream,
                               (chain ? nvfp4_moe_dec_kernel<true, r4dchain>
                                      : nvfp4_moe_dec_kernel<true, R4dUnchained>),
                               dim3((unsigned)T, groups), dim3(kNvDecBlock), 0, p,
                               (const int32_t*)sorted->data, (const int32_t*)off->data,
                               (uint16_t*)Y->data, (size_t)y_ld);
        } else {
            emit_rc = R4D_EMIT(stream,
                               (chain ? nvfp4_moe_dec_kernel<false, r4dchain>
                                      : nvfp4_moe_dec_kernel<false, R4dUnchained>),
                               dim3((unsigned)T, groups), dim3(kNvDecBlock), 0, p,
                               (const int32_t*)sorted->data, (const int32_t*)off->data,
                               (uint16_t*)Y->data, (size_t)y_ld);
        }
    }
    if (emit_rc != RAD_OK) return emit_rc;
    R4D_NV_CHK();
    return RAD_OK;
}
