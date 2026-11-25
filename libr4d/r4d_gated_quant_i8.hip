/* r4d_gated_quant_i8.hip -- gated_quant_i8: act(gate) * up, then per-group int8 codes against one f32
 * scale each, in one pass. libref/ref_quant.cpp's ref_gated_quant_i8 is the contract.
 *
 * Wave-per-(row, group): 32 lanes cooperate on one group, caching up to 256 elements in registers
 * (group <= 256) so act(gate) * up is evaluated once instead of twice, then reducing the absmax
 * via DPP wave max and quantizing in-register before storing. Larger groups fall back cleanly to
 * a wave-parallel two-pass loop. Supports r4dchain when requested.
 *
 * expf, not __expf: the oracle's sigmoid is exact, and an approximation moves a code that sits on a
 * rounding boundary.
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "r4d_chain.h"
#include "r4d_common.h"
#include "r4d_launch.h"
#include "r4d_plugin.h"
#include "rad_plugin.h"

namespace r4d_gq {

constexpr int kWaveThr = 32;

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }

struct Args {
    const uint16_t* g; const uint16_t* u;   /* gate and up rows, bf16 */
    long long gs, us;                        /* their row pitches, in elements */
    signed char* q; float* s;               /* codes [M, n] and scales [M, ngrp], dense */
    long long M;
    int n, group, ngrp;
    int act;                                 /* 0 silu, 1 sigmoid, 2 situ */
    float gcap, ucap;
};

/* One element of act(gate) * up, in the oracle's order: (gate * sigmoid(gate)) * up for silu, and
 * for SiTU (capped gate * sigmoid(raw gate)) * capped up. The sigmoid always reads the raw gate. */
__device__ __forceinline__ float gated(const Args& A, const uint16_t* g, const uint16_t* u, int i) {
    const float gv = bf(g[i]), uv = bf(u[i]);
    const float sg = 1.0f / (1.0f + expf(-gv));
    if (A.act == 2) {
        const float ga = A.gcap > 0.0f ? A.gcap * tanhf(gv / A.gcap) : gv;
        const float ua = A.ucap > 0.0f ? A.ucap * tanhf(uv / A.ucap) : uv;
        return ga * sg * ua;
    }
    return (A.act == 1 ? sg : gv * sg) * uv;
}

__device__ __forceinline__ void gq_wave_body(Args A) {
    const int gi = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    const int k0 = gi * A.group;
    const int k1 = k0 + A.group < A.n ? k0 + A.group : A.n;
    const int glen = k1 - k0;

    for (long long r = (long long)blockIdx.y; r < A.M; r += (long long)gridDim.y) {
        const uint16_t* g = A.g + r * A.gs;
        const uint16_t* u = A.u + r * A.us;
        signed char* q = A.q + r * A.n;
        float* s = A.s + r * A.ngrp;

        if (A.group <= 256) {
            float vals[8];
            float amax = 0.0f;
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                const int off = lane + p * 32;
                if (off < glen) {
                    const float v = gated(A, g, u, k0 + off);
                    vals[p] = v;
                    amax = fmaxf(amax, fabsf(v));
                }
            }
            amax = r4d_wave_max(amax);
            const float sc = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
            const float inv = 1.0f / sc;
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                const int off = lane + p * 32;
                if (off < glen) {
                    const float v = vals[p] * inv;
                    const int t = v != v ? 0 : (int)nearbyintf(fminf(fmaxf(v, -127.0f), 127.0f));
                    q[k0 + off] = (signed char)t;
                }
            }
            if (lane == 0) s[gi] = sc;
        } else {
            float amax = 0.0f;
            for (int i = k0 + lane; i < k1; i += 32)
                amax = fmaxf(amax, fabsf(gated(A, g, u, i)));
            amax = r4d_wave_max(amax);
            const float sc = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
            const float inv = 1.0f / sc;
            for (int i = k0 + lane; i < k1; i += 32) {
                const float v = gated(A, g, u, i) * inv;
                const int t = v != v ? 0 : (int)nearbyintf(fminf(fmaxf(v, -127.0f), 127.0f));
                q[i] = (signed char)t;
            }
            if (lane == 0) s[gi] = sc;
        }
    }
}

template <class CH>
__global__ __launch_bounds__(kWaveThr) void gq_wave_kernel(Args A) {
    if (CH::on) r4d_chain_wait();
    gq_wave_body(A);
    if (CH::on) r4d_chain_done();
}

/* act as libref spells it: the name, or libr4d's integer `mode` where the name is absent. */
int act_code(const RadArgs* a) {
    if (const char* s = rad_args_gets(a, "act")) {
        if (!std::strcmp(s, "silu")) return 0;
        if (!std::strcmp(s, "sigmoid")) return 1;
        if (!std::strcmp(s, "situ")) return 2;
        return -1;
    }
    const long long m = rad_args_geti_or(a, "mode", 0);
    return (m >= 0 && m <= 2) ? (int)m : -1;
}

}  // namespace r4d_gq

int r4d_rad_gated_quant_i8(const RadArgs* a, RadStream stream) {
    using namespace r4d_gq;
    if (!a) return RAD_E_INVAL;
    const RadTensor* ta = rad_arg_in(a, 0);
    const RadTensor* tb = rad_arg_in(a, 1);   /* absent = the packed gate_up form */
    const RadTensor* q  = rad_arg_in(a, 2);
    const RadTensor* sc = rad_arg_in(a, 3);
    if (!ta || !q || !sc) return RAD_E_INVAL;
    if (ta->dtype != RAD_BF16 || (tb && tb->dtype != RAD_BF16)) return RAD_E_DTYPE;
    if (q->dtype != RAD_I8 || sc->dtype != RAD_F32) return RAD_E_DTYPE;

    const long long n = rad_args_geti_or(a, "n", 0);
    if (n < 1) return RAD_E_SHAPE;
    long long group = rad_args_geti_or(a, "group", 128);
    if (group <= 0 || group > n) group = n;
    const long long ngrp = (n + group - 1) / group;

    if (q->rank != 2) return RAD_E_SHAPE;
    const long long m_cap = rad_args_geti_or(a, "M", 0);
    long long M = q->shape[0];
    if (m_cap > 0 && m_cap < M) M = m_cap;
    if (M < 1) return RAD_E_SHAPE;
    if (ta->rank != 2 || ta->stride[1] != 1 || ta->shape[0] < M) return RAD_E_SHAPE;
    const long long a_rs = ta->stride[0];
    long long b_rs = a_rs;
    if (tb) {
        if (tb->rank != 2 || tb->stride[1] != 1 || tb->shape[0] < M) return RAD_E_SHAPE;
        if (ta->shape[1] != n || tb->shape[1] != n) return RAD_E_SHAPE;
        b_rs = tb->stride[0];
    } else {
        if (ta->shape[1] != 2 * n) return RAD_E_SHAPE;
    }
    if (a_rs < 1 || b_rs < 1) return RAD_E_STRIDE;

    if (q->rank != 2 || q->shape[0] < M || q->shape[1] != n || q->stride[1] != 1 ||
        q->stride[0] != n) return RAD_E_SHAPE;
    if (sc->rank != 2 || sc->shape[0] < M || sc->shape[1] != ngrp || sc->stride[1] != 1 ||
        sc->stride[0] != ngrp) return RAD_E_SHAPE;

    const int act = act_code(a);
    if (act < 0) return RAD_E_INVAL;
    const double gcap = rad_args_getf_or(a, "gate_cap", 0.0), ucap = rad_args_getf_or(a, "up_cap", 0.0);
    if (!(gcap >= 0.0) || !(ucap >= 0.0) || !std::isfinite(gcap) || !std::isfinite(ucap))
        return RAD_E_INVAL;
    if (act != 2 && (gcap != 0.0 || ucap != 0.0)) return RAD_E_UNSUPPORTED;

    Args A{};
    A.g = (const uint16_t*)ta->data;
    A.u = tb ? (const uint16_t*)tb->data : (const uint16_t*)ta->data + n;
    A.gs = a_rs;
    A.us = tb ? b_rs : a_rs;
    A.q = (signed char*)q->data;
    A.s = (float*)sc->data;
    A.M = M;
    A.n = (int)n; A.group = (int)group; A.ngrp = (int)ngrp;
    A.act = act;
    A.gcap = (float)gcap; A.ucap = (float)ucap;

    const unsigned gy = (unsigned)(M < 65535LL ? M : 65535LL);
    const bool chain = rad_args_geti_or(a, "chain", 0) != 0;
    if (chain) R4D_KLAUNCH(R4D_KNAME(gq_wave_kernel<r4dchain>), dim3((unsigned)ngrp, gy), dim3(kWaveThr), (hipStream_t)stream, A);
    else       R4D_KLAUNCH(R4D_KNAME(gq_wave_kernel<R4dUnchained>), dim3((unsigned)ngrp, gy), dim3(kWaveThr), (hipStream_t)stream, A);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
