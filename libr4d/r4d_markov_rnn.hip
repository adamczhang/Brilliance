// Inkling 2's recurrent (`rnn_hidden`) DSpark head, walked greedily over the whole vocabulary
// (docs/OPS.md, markov_rnn_greedy; libref ref_quant.cpp is the oracle). A step is three launches:
// the gate (a rank-r state and latent from [state | w1[prev]] and the step's context, 32 channels a
// CTA), the vocabulary scan (64 ids a CTA, one winner each), and the reduction that names the
// step's token, which the next step's gate reads as its predecessor. The state alternates between
// two halves of the scratch row, so a step's CTAs read the whole previous state while they write
// their own channels of the next.
#include <hip/hip_runtime.h>
#include <climits>
#include "r4d_gdn_args.h"
#include "r4d_dflash_conv_body.h"  // r4d_bf2f, r4d_u16
#include "r4d_common.h"           // r4d_wave_sum

namespace {

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kWarps = kThreads / kWave;
constexpr int kTile = 64;                   // vocabulary ids a scan CTA, one winner each
constexpr int kPerWarp = kTile / kWarps;
constexpr int kChan = 32;                   // state channels a gate CTA
constexpr int kMaxRank = 512;

bool rows_fit(long long rows, long long pitch, long long width, int element_bytes) {
  const long long limit = LLONG_MAX / element_bytes;
  return rows > 0 && width > 0 && width <= limit && pitch >= width &&
         (rows == 1 || pitch <= (limit - (width - 1)) / (rows - 1));
}

// fp32 -> bf16 -> fp32, round to nearest even on the bits (the reference's every torch op).
__device__ __forceinline__ float rb(float x) {
  unsigned u = __float_as_uint(x);
  if ((u & 0x7fffffffu) > 0x7f800000u) return x;
  u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
  return __uint_as_float(u);
}

__device__ __forceinline__ uint32_t pack_bf16_pair(float a, float b) {
  const uint32_t lo = __float_as_uint(rb(a)) >> 16;
  const uint32_t hi = __float_as_uint(rb(b)) >> 16;
  return lo | (hi << 16);
}

union RnnV4 {
  r4d_u32x4 v;
  uint32_t u[4];
};

__device__ __forceinline__ r4d_u32x4 rnn_ld_v4(const void* p) {
#ifdef __HIP__
  return *(const r4d_u32x4*)p;
#else
  r4d_u32x4 v;
  __builtin_memcpy(&v, p, sizeof v);
  return v;
#endif
}

__device__ __forceinline__ float rnn_dot_v4(r4d_u32x4 w, const uint32_t* xp, float acc) {
  RnnV4 wv;
  wv.v = w;
#if defined(__HIP__) && __has_builtin(__builtin_amdgcn_fdot2_f32_bf16)
  typedef short r_v2s __attribute__((ext_vector_type(2)));
#pragma unroll
  for (int q = 0; q < 4; ++q)
    acc = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(r_v2s, wv.u[q]),
                                          __builtin_bit_cast(r_v2s, xp[q]), acc, false);
  return acc;
#else
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t a = wv.u[q], b = xp[q];
    acc += r4d_bf2f((r4d_u16)(a & 0xffffu)) * r4d_bf2f((r4d_u16)(b & 0xffffu));
    acc += r4d_bf2f((r4d_u16)(a >> 16))     * r4d_bf2f((r4d_u16)(b >> 16));
  }
  return acc;
#endif
}

__global__ __launch_bounds__(kThreads) void rnn_gate(
    const float* __restrict__ ctx, long long ctx_pitch, const float* __restrict__ cseq, long long cseq_pitch,
    const int* __restrict__ anchor, long long anchor_stride, const r4d_u16* __restrict__ w1,
    const r4d_u16* __restrict__ wsp, const r4d_u16* __restrict__ bias,
    float* __restrict__ scratch, long long scr_pitch, const int* __restrict__ out, long long out_pitch,
    int step, int steps, int rank, int vocab) {
  const int seq = blockIdx.y;
  const int c0 = blockIdx.x * kChan;
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int warp = tid / kWave;
  const int prev = step == 0 ? anchor[(long long)seq * anchor_stride]
                             : out[(long long)seq * out_pitch + step - 1];
  const bool ok = prev >= 0 && prev < vocab;
  float* row = scratch + (long long)seq * scr_pitch;

  __shared__ uint32_t s_in_p[kMaxRank];
  __shared__ float s_state[kChan];
  __shared__ float s_raw[3 * kChan];
  const int K = 2 * rank;
  const int half_r = rank >> 1;
  const int prev_off = ((step - 1) & 1) * rank;
  for (int i = tid; i < rank; i += kThreads) {
    if (i < half_r) {
      const float v0 = step == 0 ? 0.0f : row[prev_off + 2 * i];
      const float v1 = step == 0 ? 0.0f : row[prev_off + 2 * i + 1];
      s_in_p[i] = pack_bf16_pair(v0, v1);
    } else {
      const int k = 2 * i - rank;
      const long long idx = (long long)prev * rank + k;
      s_in_p[i] = ok ? ((uint32_t)w1[idx] | ((uint32_t)w1[idx + 1] << 16)) : 0u;
    }
  }
  if (tid < kChan) s_state[tid] = step == 0 ? 0.0f : row[prev_off + c0 + tid];
  __syncthreads();

  const long long crow = ((long long)seq * steps + step) * ctx_pitch;
  const int n4 = K >> 3;
  for (int q = warp; q < 3 * kChan; q += kWarps) {
    const int j = (q / kChan) * rank + c0 + (q % kChan);
    const r4d_u16* w = wsp + (long long)j * K;
    float dot = 0.0f;
    if ((((uintptr_t)w) & 15) == 0) {
      const r4d_u32x4* w4 = (const r4d_u32x4*)w;
      for (int g = lane; g < n4; g += kWave)
        dot = rnn_dot_v4(rnn_ld_v4(w4 + g), &s_in_p[g * 4], dot);
    } else {
      for (int k = lane; k < K; k += kWave) {
        const r4d_u16 xh = (r4d_u16)(s_in_p[k >> 1] >> ((k & 1) * 16));
        dot += r4d_bf2f(w[k]) * r4d_bf2f(xh);
      }
    }
    dot = r4d_wave_sum(dot);
    if (lane == 0)
      s_raw[q] = ctx[crow + j] + cseq[(long long)seq * cseq_pitch + j] + r4d_bf2f(bias[j]) + dot;
  }
  __syncthreads();
  if (tid < kChan) {
    const int c = c0 + tid;
    const float g = rb(1.0f / (1.0f + expf(-rb(s_raw[tid]))));
    const float cand = rb(tanhf(rb(s_raw[kChan + tid])));
    const float keep = rb(g * s_state[tid]);
    const float upd = rb(rb(1.0f - g) * cand);
    row[(step & 1) * rank + c] = rb(keep + upd);
    row[2 * rank + c] = rb(tanhf(rb(s_raw[2 * kChan + tid])));
  }
}

__global__ __launch_bounds__(kThreads) void rnn_score_tiles(
    const float* __restrict__ logits, long long logits_pitch, const r4d_u16* __restrict__ fold,
    const r4d_u16* __restrict__ fold_b, const float* __restrict__ scratch, long long scr_pitch,
    float* __restrict__ tile_scores, long long scores_pitch, int* __restrict__ tile_tokens,
    long long tokens_pitch, int step, int steps, int rank, int vocab, float cap) {
  const int seq = blockIdx.y;
  const int tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (kWave - 1);
  const int warp = tid / kWave;

  __shared__ uint32_t s_lat_p[kMaxRank / 2];
  __shared__ float s_best[kWarps];
  __shared__ int s_token[kWarps];
  const float* lat = scratch + (long long)seq * scr_pitch + 2 * rank;
  for (int i = tid; i < (rank >> 1); i += kThreads)
    s_lat_p[i] = pack_bf16_pair(lat[2 * i], lat[2 * i + 1]);
  __syncthreads();

  const float* lrow = logits + ((long long)seq * steps + step) * logits_pitch;
  const int n4 = rank >> 3;
  const bool fold_aligned = (((uintptr_t)fold) & 15) == 0;
  float wbest = -INFINITY;
  int wtok = tile * kTile + warp;
#pragma unroll
  for (int j = 0; j < kPerWarp; ++j) {
    const int token = tile * kTile + warp + j * kWarps;
    if (token < vocab) {
      float dot = 0.0f;
      const r4d_u16* fw = fold + (long long)token * rank;
      if (fold_aligned) {
        const r4d_u32x4* f4 = (const r4d_u32x4*)fw;
        for (int g = lane; g < n4; g += kWave)
          dot = rnn_dot_v4(rnn_ld_v4(f4 + g), &s_lat_p[g * 4], dot);
      } else {
        for (int r = lane; r < rank; r += kWave) {
          const r4d_u16 lh = (r4d_u16)(s_lat_p[r >> 1] >> ((r & 1) * 16));
          dot += r4d_bf2f(lh) * r4d_bf2f(fw[r]);
        }
      }
      dot = r4d_wave_sum(dot);
      if (lane == 0) {
        float x = rb(rb(lrow[token]) + rb(r4d_bf2f(fold_b[token]) + dot));
        if (cap > 0.0f) x = rb(cap * tanhf(x / cap));
        if (x > wbest || (x == wbest && token < wtok)) {
          wbest = x;
          wtok = token;
        }
      }
    }
  }
  if (lane == 0) {
    s_best[warp] = wbest;
    s_token[warp] = wtok;
  }
  __syncthreads();
  if (tid == 0) {
    float best = -INFINITY;
    int btok = tile * kTile;
#pragma unroll
    for (int w = 0; w < kWarps; ++w)
      if (s_best[w] > best || (s_best[w] == best && s_token[w] < btok)) {
        best = s_best[w];
        btok = s_token[w];
      }
    tile_scores[(long long)seq * scores_pitch + tile] = best;
    tile_tokens[(long long)seq * tokens_pitch + tile] = btok;
  }
}

__global__ __launch_bounds__(kThreads) void rnn_reduce_tiles(
    const float* __restrict__ tile_scores, long long scores_pitch, const int* __restrict__ tile_tokens,
    long long tokens_pitch, int* __restrict__ out, long long out_pitch, int step, int n_tiles) {
  const int seq = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float s_best[kThreads];
  __shared__ int s_token[kThreads];
  float best = -INFINITY;
  int btok = INT_MAX;
  for (int t = tid; t < n_tiles; t += kThreads) {
    const float s = tile_scores[(long long)seq * scores_pitch + t];
    const int k = tile_tokens[(long long)seq * tokens_pitch + t];
    if (s > best || (s == best && k < btok)) {
      best = s;
      btok = k;
    }
  }
  s_best[tid] = best;
  s_token[tid] = btok;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float s = s_best[tid + stride];
      const int k = s_token[tid + stride];
      if (s > s_best[tid] || (s == s_best[tid] && k < s_token[tid])) {
        s_best[tid] = s;
        s_token[tid] = k;
      }
    }
    __syncthreads();
  }
  if (tid == 0) out[(long long)seq * out_pitch + step] = s_token[0];
}

}  // namespace

extern "C" int r4d_rad_markov_rnn_greedy(const RadArgs* a, RadStream stream) {
  constexpr int LOGITS = 0, CTX = 1, CSEQ = 2, ANCHOR = 3, W1 = 4, WSP = 5, BIAS = 6, FOLD = 7,
                FOLDB = 8, SCR = 9, TSC = 10, TTK = 11, OUT = 12;
  const RadTensor* t[13];
  for (int i = 0; i < 13; ++i) {
    t[i] = r4d_opt(a, i);
    if (!t[i]) return R4D_NO(RAD_E_INVAL);
  }
  const long long steps = r4d_geti_or(a, "steps", 0);
  const long long rank = r4d_geti_or(a, "rank", 0);
  const long long vocab = r4d_geti_or(a, "n_vocab", 0);
  const long long astr = r4d_geti_or(a, "anchor_stride", 1);
  const float cap = (float)r4d_getf(a, "cap", 0.0);
  if (steps <= 0 || steps > INT_MAX || rank < kChan || rank > kMaxRank || rank % kChan ||
      vocab <= 0 || vocab > INT_MAX - kTile || astr <= 0)
    return R4D_NO(RAD_E_INVAL);

  const RadTensor* out = t[OUT];
  if (out->dtype != RAD_I32 || out->rank != 2 || out->shape[1] < steps) return R4D_NO(RAD_E_SHAPE);
  const long long M = out->shape[0];
  const long long R3 = 3 * rank;
  const RadTensor *lg = t[LOGITS], *cx = t[CTX], *cs = t[CSEQ], *sc = t[SCR];
  if (M <= 0 || M > 65535 || lg->dtype != RAD_F32 || lg->rank != 2 || cx->dtype != RAD_F32 ||
      cx->rank != 2 || cs->dtype != RAD_F32 || cs->rank != 2 || sc->dtype != RAD_F32 ||
      sc->rank != 2 || lg->shape[0] < M * steps || lg->shape[1] < vocab ||
      cx->shape[0] < M * steps || cx->shape[1] < R3 || cs->shape[0] < M || cs->shape[1] < R3 ||
      sc->shape[0] < M || sc->shape[1] < R3 ||
      !rows_fit(M * steps, r4d_row_pitch(lg), vocab, 4) ||
      !rows_fit(M * steps, r4d_row_pitch(cx), R3, 4) || !rows_fit(M, r4d_row_pitch(cs), R3, 4) ||
      !rows_fit(M, r4d_row_pitch(sc), R3, 4) || !rows_fit(M, r4d_row_pitch(out), steps, 4))
    return R4D_NO(RAD_E_SHAPE);
  const RadTensor* an = t[ANCHOR];
  if (an->dtype != RAD_I32 || !r4d_contig(an) || r4d_numel(an) < (M - 1) * astr + 1)
    return R4D_NO(RAD_E_SHAPE);
  for (int i : { W1, WSP, BIAS, FOLD, FOLDB })
    if (t[i]->dtype != RAD_BF16 || !r4d_contig(t[i])) return R4D_NO(RAD_E_SHAPE);
  if (r4d_numel(t[W1]) < vocab * rank || r4d_numel(t[WSP]) < R3 * 2 * rank ||
      r4d_numel(t[BIAS]) < R3 || r4d_numel(t[FOLD]) < vocab * rank || r4d_numel(t[FOLDB]) < vocab ||
      t[W1]->shape[t[W1]->rank - 1] != rank || t[FOLD]->shape[t[FOLD]->rank - 1] != rank)
    return R4D_NO(RAD_E_SHAPE);
  const long long n_tiles = (vocab + kTile - 1) / kTile;
  const RadTensor *ts = t[TSC], *tk = t[TTK];
  if (ts->dtype != RAD_F32 || tk->dtype != RAD_I32 || ts->rank != 2 || tk->rank != 2 ||
      ts->shape[0] < M || tk->shape[0] < M || ts->shape[1] < n_tiles || tk->shape[1] < n_tiles ||
      !rows_fit(M, r4d_row_pitch(ts), n_tiles, 4) || !rows_fit(M, r4d_row_pitch(tk), n_tiles, 4))
    return R4D_NO(RAD_E_SHAPE);

  hipStream_t s = (hipStream_t)stream;
  const long long lp = (long long)r4d_row_pitch(lg), cp = (long long)r4d_row_pitch(cx);
  const long long csp = (long long)r4d_row_pitch(cs), sp = (long long)r4d_row_pitch(sc);
  const long long tsp = (long long)r4d_row_pitch(ts), tkp = (long long)r4d_row_pitch(tk);
  const long long op = (long long)r4d_row_pitch(out);
  for (int step = 0; step < (int)steps; ++step) {
    hipLaunchKernelGGL(rnn_gate, dim3((unsigned)(rank / kChan), (unsigned)M), dim3(kThreads), 0, s,
                       (const float*)cx->data, cp, (const float*)cs->data, csp,
                       (const int*)an->data, (long long)astr, (const r4d_u16*)t[W1]->data,
                       (const r4d_u16*)t[WSP]->data, (const r4d_u16*)t[BIAS]->data,
                       (float*)sc->data, sp, (const int*)out->data, op, step, (int)steps,
                       (int)rank, (int)vocab);
    hipLaunchKernelGGL(rnn_score_tiles, dim3((unsigned)n_tiles, (unsigned)M), dim3(kThreads), 0, s,
                       (const float*)lg->data, lp, (const r4d_u16*)t[FOLD]->data,
                       (const r4d_u16*)t[FOLDB]->data, (const float*)sc->data, sp,
                       (float*)ts->data, tsp, (int*)tk->data, tkp, step, (int)steps, (int)rank,
                       (int)vocab, cap);
    hipLaunchKernelGGL(rnn_reduce_tiles, dim3((unsigned)M), dim3(kThreads), 0, s,
                       (const float*)ts->data, tsp, (const int*)tk->data, tkp, (int*)out->data, op,
                       step, (int)n_tiles);
  }
  return R4D_RAW((int)hipGetLastError());
}
