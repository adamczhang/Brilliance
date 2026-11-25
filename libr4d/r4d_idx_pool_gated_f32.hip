// The compressed indexer's group key (Hy4-flash, docs/OPS.md "idx_pool_gated"): every `ratio` (4)
// consecutive raw token keys pool into one key, in f32, then are normed, rotated at the group's
// first position and rounded to bf16 as HF does.
//
//   val_s[c] = sum_j wv[c][j] * x_s[j]              gat_s[c] = sum_j wg[c][j] * x_s[j] + ape[s][c]
//   p_s[c]   = exp(gat_s[c] - max_s gat) / sum_s exp(..)         s ascending
//   y[c]     = sum_s val_s[c] * p_s[c]
//   z[c]     = bf16( gain[c] * (y[c] * rsqrt(sum_c y^2 / n + eps)) )      the sum in f64
//   then the last `rotary_dim` channels of z, adjacent pairs, rotated at pos[w] and rounded again.
//
// One thread a channel, 128 threads; a workgroup pools GPW groups so one read of a weight row serves
// GPW groups (the 128 KiB of weights are the cost, not the arithmetic). Each thread's dot products
// run j ascending with a separate multiply and add (the unit is built -ffp-contract=off), the exp,
// the trig and the sum of squares are f64 and the norm's sum runs in channel order on one thread per
// group, so the result is byte-equal to libref: the order of every sum is libref's.
//
// n 128 and ratio 4 are compiled in; `rotary_dim`, the rope mode (gptj, or imrope_gptj with
// sections), eps and theta are runtime values. The key is written to cache[page[w]] and/or out[w];
// a page outside the cache is skipped.
//
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_fwht.h"
#include "r4d_q8.h"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr int kN       = 128;
constexpr int kRatio   = 4;
constexpr int kThreads = kN;

/* Rotary component of pair j in imrope_gptj with sections (t, h, w): the h position at j % 3 == 1
 * below 3 * h, the w position at j % 3 == 2 below 3 * w, else t (libref's imrope_component). */
__device__ __forceinline__ int imrope_comp(int j, int sec_h, int sec_w) {
    if (j % 3 == 1 && j < 3 * sec_h) return 1;
    if (j % 3 == 2 && j < 3 * sec_w) return 2;
    return 0;
}

#define IDX_POOL_PARAMS                                                                         \
    const uint16_t* __restrict__ k, const float* __restrict__ wv, const float* __restrict__ wg, \
    const float* __restrict__ ape, const void* __restrict__ gain,                              \
    const int32_t* __restrict__ pos, const int32_t* __restrict__ page,                          \
    uint16_t* __restrict__ cache, uint16_t* __restrict__ out, int W, int k_ld, int out_ld,      \
    long long cache_pitch, int pages, int pos_ld, int rd, float eps, double theta, int sec_h,        \
    int sec_w, int gain_bf16, uint8_t* __restrict__ q8, float* __restrict__ qs, long long q_ld,      \
    long long qs_ld, float norm, int i8
/* FP8 (idx_pool_gated_fp8): nothing is rounded to bf16; the f32 key goes through the 128-wide
 * Hadamard (r4d_fwht_wave, a wave a group, four consecutive channels a lane), times `norm`, and into
 * E4M3 codes at a power-of-two scale: dsa_pool's packed page (`cache` as bytes, the scale at byte
 * 128) and/or q8/qs. */
template <int GPW, bool IMROPE, bool FP8>
__device__ __forceinline__ void idx_pool_gated_body(IDX_POOL_PARAMS) {

    __shared__ float xs[GPW][kRatio][kN];
    __shared__ float ys[GPW][kN];
    __shared__ float rs[GPW];

    const int c  = (int)threadIdx.x;
    const int g0 = (int)blockIdx.x * GPW;

    /* The raw keys, as f32: thread c holds channel c of every row. A group past W reads zeros. */
#pragma unroll
    for (int g = 0; g < GPW; ++g)
#pragma unroll
        for (int s = 0; s < kRatio; ++s) {
            const int w = g0 + g;
            xs[g][s][c] = w < W ? bf16_to_f32(k[(size_t)(w * kRatio + s) * k_ld + c]) : 0.0f;
        }
    __syncthreads();

    float val[GPW][kRatio], gat[GPW][kRatio];
#pragma unroll
    for (int g = 0; g < GPW; ++g)
#pragma unroll
        for (int s = 0; s < kRatio; ++s) { val[g][s] = 0.0f; gat[g][s] = 0.0f; }

    /* Row c of both weights, j ascending; every x is an LDS broadcast. */
    const float4* wvr = (const float4*)(wv + (size_t)c * kN);
    const float4* wgr = (const float4*)(wg + (size_t)c * kN);
    for (int j4 = 0; j4 < kN / 4; ++j4) {
        const float4 a = wvr[j4], b = wgr[j4];
#pragma unroll
        for (int g = 0; g < GPW; ++g)
#pragma unroll
            for (int s = 0; s < kRatio; ++s) {
                const float4 x = *(const float4*)&xs[g][s][4 * j4];
                float v = val[g][s], t = gat[g][s];
                v = v + a.x * x.x;  t = t + b.x * x.x;
                v = v + a.y * x.y;  t = t + b.y * x.y;
                v = v + a.z * x.z;  t = t + b.z * x.z;
                v = v + a.w * x.w;  t = t + b.w * x.w;
                val[g][s] = v; gat[g][s] = t;
            }
    }

    /* The softmax over the slots and the pooled value, per group. */
#pragma unroll
    for (int g = 0; g < GPW; ++g) {
        float gs[kRatio];
#pragma unroll
        for (int s = 0; s < kRatio; ++s) gs[s] = gat[g][s] + ape[s * kN + c];
        float mx = -INFINITY;
#pragma unroll
        for (int s = 0; s < kRatio; ++s) mx = mx < gs[s] ? gs[s] : mx;
        float e[kRatio], den = 0.0f;
#pragma unroll
        for (int s = 0; s < kRatio; ++s) {
            e[s] = (float)exp((double)(gs[s] - mx));
            den = den + e[s];
        }
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < kRatio; ++s) {
            const float p = __fdiv_rn(e[s], den);
            acc = acc + val[g][s] * p;
        }
        ys[g][c] = acc;
    }
    __syncthreads();

    /* The mean square, in channel order and in f64, one thread a group: libref's sum exactly. */
    if (c < GPW) {
        double sq = 0.0;
        for (int i = 0; i < kN; ++i) sq += (double)ys[c][i] * (double)ys[c][i];
        rs[c] = __fdiv_rn(1.0f, __fsqrt_rn((float)(sq / (double)kN) + eps));
    }
    __syncthreads();

    float z[GPW];
    const float gc = gain_bf16 ? bf16_to_f32(((const uint16_t*)gain)[c]) : ((const float*)gain)[c];
#pragma unroll
    for (int g = 0; g < GPW; ++g)
        z[g] = FP8 ? gc * (ys[g][c] * rs[g]) : bf16_to_f32(f32_to_bf16(gc * (ys[g][c] * rs[g])));
    __syncthreads();
#pragma unroll
    for (int g = 0; g < GPW; ++g) ys[g][c] = z[g];
    __syncthreads();

    /* The rotation: channel c of the last rd takes pair i = (c - (n - rd)) / 2 and computes its own
     * half of the pair from the two rounded values. */
    const int rc = c - (kN - rd);
#pragma unroll
    for (int g = 0; g < GPW; ++g) {
        const int w = g0 + g;
        float r = z[g];
        if (rc >= 0 && w < W) {
            const int i = rc >> 1;
            const int comp = IMROPE ? imrope_comp(i, sec_h, sec_w) : 0;
            const int pv = IMROPE ? pos[(size_t)w * pos_ld + comp] : pos[w];
            const float inv_freq = (float)(1.0 / pow(theta, (double)(2 * i) / (double)rd));
            const float ang = (float)pv * inv_freq;
            const float cs = (float)cos((double)ang), sn = (float)sin((double)ang);
            const float x0 = ys[g][c & ~1], x1 = ys[g][(c & ~1) | 1];
            if (FP8) r = (rc & 1) ? x1 * cs + x0 * sn : x0 * cs - x1 * sn;
            else
                r = (rc & 1) ? bf16_to_f32(f32_to_bf16(x1 * cs + x0 * sn))
                             : bf16_to_f32(f32_to_bf16(x0 * cs - x1 * sn));
        }
        if (FP8) {
            z[g] = r;
            continue;
        }
        if (w < W) {
            const uint16_t h = f32_to_bf16(r);
            if (out) out[(size_t)w * out_ld + c] = h;
            if (cache) {
                const int pg = page[w];
                if (pg >= 0 && pg < pages) cache[(size_t)pg * cache_pitch + c] = h;
            }
        }
    }
    if (!FP8) return;
    /* Every pair has read both of its rotated inputs: the f32 keys replace them in LDS. */
    __syncthreads();
#pragma unroll
    for (int g = 0; g < GPW; ++g) ys[g][c] = z[g];
    __syncthreads();
    const int lane = c & 31, wave = c >> 5;
    for (int g = wave; g < GPW; g += kThreads / 32) {
        const int w = g0 + g;
        if (w >= W) continue;
        float v[4];
#pragma unroll
        for (int d = 0; d < 4; ++d) v[d] = ys[g][lane * 4 + d];
        r4d_fwht_wave<kN>(v, lane);
        float am = 1e-4f;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            v[d] = v[d] * norm;
            am = fmaxf(am, fabsf(v[d]));
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) am = fmaxf(am, __shfl_xor(am, mask, 32));
        /* i8 (idx_pool_gated_i8): symmetric int8 codes at an f32 scale amax / 127, else E4M3 at a
         * power-of-two scale. */
        float sc;
        uint32_t packed;
        if (i8) {
            sc = am * (1.0f / 127.0f);
            packed = q8_pack4<true>(v[0], v[1], v[2], v[3], 1.0f / sc);
        } else {
            const uint32_t bits = __float_as_uint(am * (1.0f / 448.0f));
            sc = __uint_as_float((bits & 0x7f800000u) + ((bits & 0x7fffffu) ? 0x800000u : 0));
            packed = q8_pack4<false>(v[0], v[1], v[2], v[3], 1.0f / sc);
        }
        if (q8) {
            uint8_t* dst = q8 + (size_t)w * q_ld + lane * 4;
            if (((uintptr_t)dst & 3) == 0)
                *(uint32_t*)dst = packed;
            else
#pragma unroll
                for (int d = 0; d < 4; ++d) dst[d] = (uint8_t)(packed >> (8 * d));
            if (!lane) qs[(size_t)w * qs_ld] = sc;
        }
        if (cache) {
            const int pg = page[w];
            if (pg >= 0 && pg < pages) {
                uint8_t* dst = (uint8_t*)cache + (size_t)pg * cache_pitch;
                *(uint32_t*)(dst + lane * 4) = packed;
                if (!lane) *(float*)(dst + 128) = sc;
            }
        }
    }
}

template <int GPW, bool IMROPE, bool FP8, class CH>
__global__ __launch_bounds__(kThreads)
void idx_pool_gated_kernel(IDX_POOL_PARAMS) {
    if (CH::on) r4d_chain_wait();
    idx_pool_gated_body<GPW, IMROPE, FP8>(k, wv, wg, ape, gain, pos, page, cache, out, W, k_ld,
                                          out_ld, cache_pitch, pages, pos_ld, rd, eps, theta,
                                          sec_h, sec_w, gain_bf16, q8, qs, q_ld, qs_ld, norm, i8);
    if (CH::on) r4d_chain_done();
}
#undef IDX_POOL_PARAMS

constexpr int kWideGroups = 4;       /* groups a workgroup on the wide form */
constexpr int kWideMinW   = 16;      /* the wide form from this many groups */

}  // namespace

// ============================================================== the plugin ABI entry point

enum { IP_K = 0, IP_WV = 1, IP_WG = 2, IP_APE = 3, IP_GAIN = 4, IP_POS = 5, IP_PAGE = 6,
       IP_CACHE = 7, IP_OUT = 8, IP_SCALE = 9 };

/* `fp8`: idx_pool_gated_fp8, whose operand 8 is the codes `q` and 9 their `scale`, and whose cache
 * is dsa_pool's packed u8 page. */
static int pool_entry(const RadArgs* a, RadStream stream, bool fp8, bool i8 = false) {
    const RadTensor* k  = r4d_opt(a, IP_K);
    const RadTensor* wv = r4d_opt(a, IP_WV);
    const RadTensor* wg = r4d_opt(a, IP_WG);
    const RadTensor* ap = r4d_opt(a, IP_APE);
    const RadTensor* gn = r4d_opt(a, IP_GAIN);
    const RadTensor* po = r4d_opt(a, IP_POS);
    const RadTensor* pg = r4d_opt(a, IP_PAGE);
    const RadTensor* ca = r4d_opt(a, IP_CACHE);
    const RadTensor* ou = fp8 ? nullptr : r4d_opt(a, IP_OUT);
    const RadTensor* q8 = fp8 ? r4d_opt(a, IP_OUT) : nullptr;
    const RadTensor* qs = fp8 ? r4d_opt(a, IP_SCALE) : nullptr;
    if (!k || !wv || !wg || !ap || !gn || !po) return R4D_NO(RAD_E_INVAL);
    if (!pg != !ca || !q8 != !qs) return R4D_NO(RAD_E_INVAL);
    if (!ca && !ou && !q8) return R4D_NO(RAD_E_INVAL);
    const long long n = r4d_geti_or(a, "n", 0), ratio = r4d_geti_or(a, "ratio", 0);
    const long long rd = r4d_geti_or(a, "rotary_dim", 0);
    if (n != kN || ratio != kRatio || rd < 2 || rd > n || (rd & 1)) return R4D_NO(RAD_E_SHAPE);
    const double eps = r4d_getf(a, "eps", 0.0), theta = r4d_getf(a, "theta", 0.0);
    /* libref reads both as f32; so does this. */
    const float eps_f = (float)eps, theta_f = (float)theta;
    if (!(eps_f > 0.0f) || !std::isfinite(eps_f) || !(theta_f > 1.0f)) return R4D_NO(RAD_E_INVAL);
    const char* mode = r4d_gets(a, "mode");
    const bool imrope = mode && !std::strcmp(mode, "imrope_gptj");
    if (!mode || (!imrope && std::strcmp(mode, "gptj"))) return R4D_NO(RAD_E_INVAL);
    long long sec_h = 0, sec_w = 0;
    if (imrope) {
        const char* s = r4d_gets(a, "sections");
        long long t = 0, h = 0, w = 0;
        if (!s || std::sscanf(s, "%lld %lld %lld", &t, &h, &w) != 3 || t < 0 || h < 0 || w < 0 ||
            t + h + w != rd / 2)
            return R4D_NO(RAD_E_SHAPE);
        sec_h = h; sec_w = w;
    }
    if (k->dtype != RAD_BF16 || (ou && ou->dtype != RAD_BF16) ||
        (ca && ca->dtype != (fp8 ? RAD_U8 : RAD_BF16)) || (q8 && q8->dtype != (i8 ? RAD_I8 : RAD_F8E4M3)) ||
        (qs && qs->dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    if (wv->dtype != RAD_F32 || wg->dtype != RAD_F32 || ap->dtype != RAD_F32 ||
        (gn->dtype != RAD_F32 && gn->dtype != RAD_BF16) || po->dtype != RAD_I32 ||
        (pg && pg->dtype != RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    if (k->rank != 2 || k->shape[1] != n || k->shape[0] < ratio || k->shape[0] % ratio)
        return R4D_NO(RAD_E_SHAPE);
    const long long W = k->shape[0] / ratio;
    const long long kp = r4d_row_pitch(k);
    if (kp < 0) return R4D_NO(RAD_E_STRIDE);
    auto dense = [&](const RadTensor* t, long long rows) {
        return t->rank == 2 && t->shape[0] == rows && t->shape[1] == n && t->stride[1] == 1 &&
               t->stride[0] == n;
    };
    if (!dense(wv, n) || !dense(wg, n) || !dense(ap, ratio)) return R4D_NO(RAD_E_SHAPE);
    if (gn->rank != 1 || gn->shape[0] != n || gn->stride[0] != 1) return R4D_NO(RAD_E_SHAPE);
    long long pos_ld = 0;
    if (imrope) {
        /* [W, 3]: a group's three components side by side, qsa_work's `bpos` under M-RoPE. */
        if (po->rank != 2 || po->shape[1] != 3 || po->shape[0] < W || po->stride[1] != 1)
            return R4D_NO(RAD_E_SHAPE);
        pos_ld = po->stride[0];
    } else if (po->rank != 1 || po->shape[0] < W || po->stride[0] != 1) {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (pg && (pg->rank != 1 || pg->shape[0] < W || pg->stride[0] != 1)) return R4D_NO(RAD_E_SHAPE);
    long long cache_pitch = 0, pages = 0;
    if (ca && fp8) {
        if (ca->rank != 4 || ca->shape[0] < 1 || ca->shape[1] != 1 || ca->shape[2] != 4 ||
            ca->shape[3] != 34 || ca->stride[3] != 1 || ca->stride[2] != 34 ||
            ca->stride[0] < 136 || ca->stride[0] % 4 || ((uintptr_t)ca->data & 3))
            return R4D_NO(RAD_E_SHAPE);
        cache_pitch = ca->stride[0];
        pages = ca->shape[0];
    } else if (ca) {
        if (ca->rank != 4 || ca->shape[0] < 1 || ca->shape[1] != 1 ||
            ca->shape[2] * ca->shape[3] != n || ca->stride[3] != 1 || ca->stride[2] != ca->shape[3] ||
            ca->stride[0] < n)
            return R4D_NO(RAD_E_SHAPE);
        cache_pitch = ca->stride[0];
        pages = ca->shape[0];
    }
    long long op = 0;
    if (ou) {
        if (ou->rank != 2 || ou->shape[0] != W || ou->shape[1] != n) return R4D_NO(RAD_E_SHAPE);
        op = r4d_row_pitch(ou);
        if (op < 0) return R4D_NO(RAD_E_STRIDE);
    }
    long long q_ld = 0, qs_ld = 0;
    if (q8) {
        if (q8->rank != 2 || q8->shape[0] != W || q8->shape[1] != n || q8->stride[1] != 1 ||
            q8->stride[0] < n || qs->rank != 2 || qs->shape[0] != W || qs->shape[1] != 1 ||
            qs->stride[0] < 1)
            return R4D_NO(RAD_E_SHAPE);
        q_ld = q8->stride[0];
        qs_ld = qs->stride[0];
    }
    if (W > (1ll << 30)) return R4D_NO(RAD_E_SHAPE);
    /* libref's n^-0.5 in f32. */
    const float norm = 1.0f / std::sqrt((float)kN);

    const int gain_bf16 = gn->dtype == RAD_BF16;
    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    const bool wide = W >= kWideMinW;
    const int gpw = wide ? kWideGroups : 1;
    const unsigned grid = (unsigned)((W + gpw - 1) / gpw);
#define R4D_IPOOL(G_, IM_, CH_)                                                                 \
    (fp8 ? R4D_IPOOL1(G_, IM_, true, CH_) : R4D_IPOOL1(G_, IM_, false, CH_))
#define R4D_IPOOL1(G_, IM_, F8_, CH_)                                                           \
    R4D_EMIT(stream, (idx_pool_gated_kernel<G_, IM_, F8_, CH_>), dim3(grid), dim3(kThreads), 0, \
             (const uint16_t*)r4d_p(k), (const float*)r4d_p(wv), (const float*)r4d_p(wg),       \
             (const float*)r4d_p(ap), r4d_p(gn), (const int32_t*)r4d_p(po),                      \
             (const int32_t*)r4d_p(pg), (uint16_t*)r4d_wp(ca), (uint16_t*)r4d_wp(ou), (int)W,    \
             (int)kp, (int)op, (long long)cache_pitch, (int)pages, (int)pos_ld, (int)rd, eps_f,       \
             (double)theta_f, (int)sec_h, (int)sec_w, gain_bf16, (uint8_t*)r4d_wp(q8),          \
             (float*)r4d_wp(qs), (long long)q_ld, (long long)qs_ld, norm, i8 ? 1 : 0)
#define R4D_IPOOL_CH(G_, IM_) (chain ? R4D_IPOOL(G_, IM_, r4dchain) : R4D_IPOOL(G_, IM_, R4dUnchained))
    if (wide) return imrope ? R4D_IPOOL_CH(kWideGroups, true) : R4D_IPOOL_CH(kWideGroups, false);
    return imrope ? R4D_IPOOL_CH(1, true) : R4D_IPOOL_CH(1, false);
#undef R4D_IPOOL_CH
#undef R4D_IPOOL
#undef R4D_IPOOL1
}

extern "C" int r4d_rad_idx_pool_gated(const RadArgs* a, RadStream stream) {
    return pool_entry(a, stream, false);
}
extern "C" int r4d_rad_idx_pool_gated_fp8(const RadArgs* a, RadStream stream) {
    return pool_entry(a, stream, true);
}
extern "C" int r4d_rad_idx_pool_gated_i8(const RadArgs* a, RadStream stream) {
    return pool_entry(a, stream, true, true);
}
