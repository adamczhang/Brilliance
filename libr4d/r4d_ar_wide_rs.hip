// r4d_ar_wide_rs.hip: Brilliance, the four-rank host-mailbox collectives with the resend (stages W2
// and W3 of the four-rank resend; the exchange is r4d_arw_exchange in
// r4d_ar_wide_x.h). r4d_ar_wide_host.hip's launchers launch these through the wrappers at the end
// when the resend is on (the default; BRILLIANCE_AR_RESEND=0 keeps W1's kernels there). A file of its
// own, so that W1's code object is byte for byte what it was.
#include "r4d.h"
#include "r4d_ar_wide_host.h"

// ================================================================ W2/W3: the resend
// The RS kernels: the W1 kernels above with the push check and the bounded handshake replaced by
// the exchange (r4d_arw_exchange, r4d_ar_wide_x.h), so a peer's wrong or missing block is asked for
// again and re-pushed from its unreduced source instead of stopping the engine. A peer that never
// sends a right block still ends in W1's record and the engine's stop. The kernels W1 has are kept
// byte for byte and run under BRILLIANCE_AR_RESEND=0. Where each resends from (design 3.3), held
// unreduced until the block's exchange ends:
//   ar_hc_write_4rank          y4[start, end): the in-place reduce waits for the exchange
//   ar_gather_hc_write_4rank   y4, the gathered words the push stored there (the slot table is not
//                              read again)
//   ar_oneshot_4rank_exact     in4 (out4 may be in4: the reduce waits)
//   ar_gather_4rank            in4 at its source pitch (the output is a separate buffer)
// Every other line is the W1 kernel's: the same push, check try, mirror and fold, so the same bytes.

// The resend's re-push of words [start, end) to the peer indices in `mask`, from `msg`, the
// injected words left out again when `stuck`.
template <bool INJ, typename FM>
__device__ __forceinline__ void arw_repush(int4* const (&ps)[ARW_NP], unsigned int mask, int start,
                                           int end, bool stuck, const R4dArwInjX& ij, const FM& msg) {
  const int tid = threadIdx.x, nt = blockDim.x;
  for (int k = start + tid; k < end; k += nt) {
    const int4 v = msg(k);
#pragma unroll
    for (int p = 0; p < ARW_NP; ++p) {
      if (!((mask >> p) & 1u)) continue;
      if (INJ && stuck && ((ij.pmask >> p) & 1u) && k >= ij.d.lo && k < ij.d.hi) continue;
      ps[p][k] = v;
    }
  }
}
// Injection late's withheld words, now; ctl-rq's overwrite of this rank's own slot from bpeer.
template <typename FM>
__device__ __forceinline__ void arw_inj_late(int4* const (&ps)[ARW_NP], const R4dArwInjX& ij,
                                             const FM& msg) {
  for (int k = ij.d.lo + (int)threadIdx.x; k < ij.d.hi; k += (int)blockDim.x) {
    const int4 v = msg(k);
#pragma unroll
    for (int p = 0; p < ARW_NP; ++p)
      if ((ij.pmask >> p) & 1u) ps[p][k] = v;
  }
}
__device__ __forceinline__ void arw_inj_bad(const ArwArgs& A, int parity, const R4dArwInjX& ij) {
  int4* const mw = const_cast<int4*>(arw_slot(A, parity, A.peers.rank[ij.bpeer < ARW_NP ? ij.bpeer : 0]));
  for (int k = ij.blo + (int)threadIdx.x; k < ij.bhi; k += (int)blockDim.x) {
    int4 v = mw[k];
    v.x ^= 0x5A5A5A5A; v.y ^= 0x5A5A5A5A; v.z ^= 0x5A5A5A5A; v.w ^= 0x5A5A5A5A;
    mw[k] = v;
  }
}
__device__ __forceinline__ int4* arw_pick(int4* const (&ps)[ARW_NP], int p) {
  return p == 0 ? ps[0] : p == 1 ? ps[1] : ps[2];
}

// ---------------------------------------------------------------- the one-shot all-reduce, RS (W3)
template <typename T, int LANES, bool INJ>
__global__ void arw_oneshot_rs_kernel(const int4* __restrict__ in4, int4* __restrict__ out4, int n16,
                                      ArwArgs A) {
  constexpr int WS = ARW_WS, NP = ARW_NP;
  constexpr unsigned int OP = R4D_AR_OP_W_ONESHOT;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int rank = A.rank;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(A.seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;
  const int w = (n16 + nb - 1) / nb;
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  int4* ps[NP];
#pragma unroll
  for (int p = 0; p < NP; ++p)
    ps[p] = A.peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)A.slot_stride16;
  const int4* sp[NP];
#pragma unroll
  for (int p = 0; p < NP; ++p) sp[p] = arw_slot(A, parity, A.peers.rank[p]);

  auto msg = [&](int k) { return in4[k]; };
  R4dArwInjX ij = r4d_arw_injx_none(start, end);
  if constexpr (INJ)
    ij = r4d_arw_injx_bcast<NP>(A.iw, b, 0u, start, end, w, n16, arw_ctl(A), msg,
                                [&](int p) { return arw_pick(ps, p); },
                                [&](int p) { return p == 0 ? sp[0] : p == 1 ? sp[1] : sp[2]; });
  auto put = [&](int p, int k, const int4& v) {
    if constexpr (INJ) r4d_arw_injx_put(ij, p, ps[p], k, v, msg);
    else ps[p][k] = v;
  };

  // Push, 4x unrolled; loaded words are reused across peers so the input is read once.
  unsigned int hw = 0u;
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      const int4 v0 = in4[k], v1 = in4[k + nt], v2 = in4[k + 2 * nt], v3 = in4[k + 3 * nt];
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        put(p, k, v0); put(p, k + nt, v1); put(p, k + 2 * nt, v2); put(p, k + 3 * nt, v3);
      }
      hw += r4d_archk_u4(v0, 4u * k) + r4d_archk_u4(v1, 4u * (k + nt)) +
            r4d_archk_u4(v2, 4u * (k + 2 * nt)) + r4d_archk_u4(v3, 4u * (k + 3 * nt));
    }
    for (; k < end; k += nt) {
      const int4 v = in4[k];
#pragma unroll
      for (int p = 0; p < NP; ++p) put(p, k, v);
      hw += r4d_archk_u4(v, 4u * k);
    }
  }
  if constexpr (INJ) r4d_arw_injx_pair(ij, arw_pick(ps, ij.cpeer), arw_ctl(A), msg);
  const unsigned int hws[NP] = { hw, hw, hw };
  unsigned int hck[NP];
  r4d_arw_put_t<WS, NP>(A.peers, b, rank, 0u, (unsigned)parity, hws, 2u * s, hck);
  r4d_arw_publish<WS, NP>(A.peers, b, rank, 2u * s, A.drain, A.pub, INJ ? ij.d.noflag : 0u);
  r4d_arw_exchange<WS, NP, INJ>(
      A.peers, A.my_flags, b, rank, 2u * s, 0u, (unsigned)parity, OP, hck, A.drain,
      [&](unsigned int mask, unsigned int (&h)[NP]) { arw_rehash_slots(sp, A.md, start, end, mask, h); },
      [&](unsigned int mask, bool stuck) { arw_repush<INJ>(ps, mask, start, end, stuck, ij, msg); },
      [&]() { arw_inj_late(ps, ij, msg); },
      [&]() { arw_inj_bad(A, parity, ij); }, ij);

  // Our own contribution comes from the input; the peers' from the mirror.
  const int4* src[WS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    src[r] = (r == rank) ? in4 : r4d_archk_moved(arw_slot(A, parity, r), A.md);
  {
    int k = start + tid;
    for (; k + nt < end; k += 2 * nt) {
      out4[k]      = arw_vsum<T, LANES>(src, k);
      out4[k + nt] = arw_vsum<T, LANES>(src, k + nt);
    }
    for (; k < end; k += nt) out4[k] = arw_vsum<T, LANES>(src, k);
  }
}

// ---------------------------------------------------------------- the fused hc writes, RS (W2)
template <bool GATHER, bool INV, bool INJ>
__global__ void arw_hc_write_rs_kernel(int4* __restrict__ y4, const unsigned short* __restrict__ inj,
                                       unsigned short* __restrict__ h, int n16, int n, int hc,
                                       long long irow, long long hrow, R4dArGather G,
                                       const int* __restrict__ inv, ArwArgs A) {
  constexpr int WS = ARW_WS, NP = ARW_NP;
  constexpr unsigned int OP = GATHER ? R4D_AR_OP_W_GHC : R4D_AR_OP_W_HC;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int rank = A.rank;
  __shared__ unsigned int s_seq, s_slot;
  __shared__ int   s_i[GATHER ? R4D_ARG_ROWS * R4D_ARG_SLOTS : 1];
  __shared__ float s_w[GATHER ? R4D_ARG_ROWS * R4D_ARG_SLOTS : 1];
  __shared__ float s_g[GATHER ? R4D_ARG_ROWS : 1];
  r4d_ar_seq_take(A.seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;
  const int w = (n16 + nb - 1) / nb;
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;
  const int wpr = n >> 3;
  const int row_lo = start / wpr;

  int4* ps[NP];
#pragma unroll
  for (int p = 0; p < NP; ++p)
    ps[p] = A.peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)A.slot_stride16;
  const int4* sp[NP];
#pragma unroll
  for (int p = 0; p < NP; ++p) sp[p] = arw_slot(A, parity, A.peers.rank[p]);

  if constexpr (GATHER) {
    /* Slot table, as the default kernel: a slot no sorted row claims reads -1, so clear and scan
     * are separated by a barrier. */
    const int nrow = end > start ? (end - 1) / wpr - row_lo + 1 : 0;
    const int K = G.top_k;
    if (!INV) {
      for (int x = tid; x < nrow * K; x += nt) s_i[x] = -1;
      __syncthreads();
      {
        const int lo = row_lo * K, hi = (row_lo + nrow) * K;
        for (int i = tid; i < G.T; i += nt) {
          const int f = G.sorted[i];
          if (f >= lo && f < hi) s_i[f - lo] = i;
        }
      }
      for (int x = tid; x < nrow * K; x += nt)
        s_w[x] = bf2f(G.ew[(long long)(row_lo + x / K) * G.ew_ld + x % K]);
      if (G.sh)
        for (int r = tid; r < nrow; r += nt) {
          const float v0 = bf2f(G.sg[(long long)(row_lo + r) * G.sg_ld]);
          s_g[r] = G.sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
        }
      __syncthreads();
    }
  }
  auto msg = [&](int k) -> int4 {
    if constexpr (GATHER) {
      const int row = k / wpr, i0 = (k - row * wpr) * 8;
      return r4d_ar_gather_word<INV>(G, inv, s_i, s_w, s_g, row, row - row_lo, i0);
    } else {
      return y4[k];
    }
  };
  /* The resend's source, and pair's: what the push stored in y (the gathered words when GATHER). */
  auto held = [&](int k) -> int4 { return y4[k]; };
  R4dArwInjX ij = r4d_arw_injx_none(start, end);
  if constexpr (INJ)
    ij = r4d_arw_injx_bcast<NP>(A.iw, b, 0u, start, end, w, n16, arw_ctl(A), msg,
                                [&](int p) { return arw_pick(ps, p); },
                                [&](int p) { return p == 0 ? sp[0] : p == 1 ? sp[1] : sp[2]; });
  auto put = [&](int p, int k, const int4& v) {
    if constexpr (INJ) r4d_arw_injx_put(ij, p, ps[p], k, v, msg);
    else ps[p][k] = v;
  };

  unsigned int hw = 0u;
  if constexpr (GATHER) {
    /* Gather one word per thread per pass; keep it and push it to every peer. */
    for (int k = start + tid; k < end; k += nt) {
      const int4 out = msg(k);
      y4[k] = out;
#pragma unroll
      for (int p = 0; p < NP; ++p) put(p, k, out);
      hw += r4d_archk_u4(out, 4u * k);
    }
  } else {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      const int4 v0 = y4[k], v1 = y4[k + nt], v2 = y4[k + 2 * nt], v3 = y4[k + 3 * nt];
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        put(p, k, v0); put(p, k + nt, v1); put(p, k + 2 * nt, v2); put(p, k + 3 * nt, v3);
      }
      hw += r4d_archk_u4(v0, 4u * k) + r4d_archk_u4(v1, 4u * (k + nt)) +
            r4d_archk_u4(v2, 4u * (k + 2 * nt)) + r4d_archk_u4(v3, 4u * (k + 3 * nt));
    }
    for (; k < end; k += nt) {
      const int4 v = y4[k];
#pragma unroll
      for (int p = 0; p < NP; ++p) put(p, k, v);
      hw += r4d_archk_u4(v, 4u * k);
    }
  }
  if constexpr (INJ) r4d_arw_injx_pair(ij, arw_pick(ps, ij.cpeer), arw_ctl(A), held);
  const unsigned int hws[NP] = { hw, hw, hw };
  unsigned int hck[NP];
  r4d_arw_put_t<WS, NP>(A.peers, b, rank, 0u, (unsigned)parity, hws, 2u * s, hck);
  r4d_arw_publish<WS, NP>(A.peers, b, rank, 2u * s, A.drain, A.pub, INJ ? ij.d.noflag : 0u);
  r4d_arw_exchange<WS, NP, INJ>(
      A.peers, A.my_flags, b, rank, 2u * s, 0u, (unsigned)parity, OP, hck, A.drain,
      [&](unsigned int mask, unsigned int (&hh)[NP]) { arw_rehash_slots(sp, A.md, start, end, mask, hh); },
      [&](unsigned int mask, bool stuck) { arw_repush<INJ>(ps, mask, start, end, stuck, ij, held); },
      [&]() { arw_inj_late(ps, ij, held); },
      [&]() { arw_inj_bad(A, parity, ij); }, ij);

  /* In place, held behind the exchange: a thread reads every source of word k before storing it,
   * every peer has acked this block's push (no resend can need y4 again), and no peer reads this
   * rank's `y`. */
  const int4* src[WS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    src[r] = (r == rank) ? (const int4*)y4 : r4d_archk_moved(arw_slot(A, parity, r), A.md);
  for (int k = start + tid; k < end; k += nt) {
    const int4 v = arw_vsum<__hip_bfloat16, 8>(src, k);
    y4[k] = v;
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    r4d_ar_hc_write_word(inj, h, row, i0, n, hc, irow, hrow,
                         reinterpret_cast<const unsigned short*>(&v));
  }
}

// ---------------------------------------------------------------- the all-gather, RS (W3)
template <bool INJ>
__global__ void arw_gather_rs_kernel(const int4* __restrict__ in4, int4* __restrict__ out4, int n16,
                                     int row16, int dp16, int sp16, ArwArgs A) {
  constexpr int WS = ARW_WS, NP = ARW_NP;
  constexpr unsigned int OP = R4D_AR_OP_W_GATHER;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int rank = A.rank;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(A.seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;
  const int w = (n16 + nb - 1) / nb;
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  int4* ps[NP];
#pragma unroll
  for (int p = 0; p < NP; ++p)
    ps[p] = A.peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)A.slot_stride16;
  const int4* sp[NP];
#pragma unroll
  for (int p = 0; p < NP; ++p) sp[p] = arw_slot(A, parity, A.peers.rank[p]);
  auto msg = [&](int k) { return in4[arw_g_src(k, row16, n16, sp16)]; };
  R4dArwInjX ij = r4d_arw_injx_none(start, end);
  if constexpr (INJ)
    ij = r4d_arw_injx_bcast<NP>(A.iw, b, 0u, start, end, w, n16, arw_ctl(A), msg,
                                [&](int p) { return arw_pick(ps, p); },
                                [&](int p) { return p == 0 ? sp[0] : p == 1 ? sp[1] : sp[2]; });
  auto put = [&](int p, int k, const int4& v) {
    if constexpr (INJ) r4d_arw_injx_put(ij, p, ps[p], k, v, msg);
    else ps[p][k] = v;
  };
  unsigned int hw = 0u;
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      const int4 v0 = msg(k), v1 = msg(k + nt), v2 = msg(k + 2 * nt), v3 = msg(k + 3 * nt);
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        put(p, k, v0); put(p, k + nt, v1); put(p, k + 2 * nt, v2); put(p, k + 3 * nt, v3);
      }
      hw += r4d_archk_u4(v0, 4u * k) + r4d_archk_u4(v1, 4u * (k + nt)) +
            r4d_archk_u4(v2, 4u * (k + 2 * nt)) + r4d_archk_u4(v3, 4u * (k + 3 * nt));
    }
    for (; k < end; k += nt) {
      const int4 v = msg(k);
#pragma unroll
      for (int p = 0; p < NP; ++p) put(p, k, v);
      hw += r4d_archk_u4(v, 4u * k);
    }
  }
  if constexpr (INJ) r4d_arw_injx_pair(ij, arw_pick(ps, ij.cpeer), arw_ctl(A), msg);
  const unsigned int hws[NP] = { hw, hw, hw };
  unsigned int hck[NP];
  r4d_arw_put_t<WS, NP>(A.peers, b, rank, 0u, (unsigned)parity, hws, 2u * s, hck);
  r4d_arw_publish<WS, NP>(A.peers, b, rank, 2u * s, A.drain, A.pub, INJ ? ij.d.noflag : 0u);

  // Each peer's slice, checked as it is placed (the output is the mirror: a retry rewrites it).
  r4d_arw_exchange<WS, NP, INJ>(
      A.peers, A.my_flags, b, rank, 2u * s, 0u, (unsigned)parity, OP, hck, A.drain,
      [&](unsigned int mask, unsigned int (&hh)[NP]) {
#pragma unroll
        for (int p = 0; p < NP; ++p) {
          if (!((mask >> p) & 1u)) continue;
          const int owner = A.peers.rank[p];
          for (int k = start + tid; k < end; k += nt) {
            const int4 v = sp[p][k];
            out4[arw_g_dst(k, row16, n16, owner, dp16)] = v;
            hh[p] += r4d_archk_u4(v, 4u * k);
          }
        }
      },
      [&](unsigned int mask, bool stuck) { arw_repush<INJ>(ps, mask, start, end, stuck, ij, msg); },
      [&]() { arw_inj_late(ps, ij, msg); },
      [&]() { arw_inj_bad(A, parity, ij); }, ij);
  // This rank's own slice, from the input at its own pitch.
  for (int k = start + tid; k < end; k += nt)
    out4[arw_g_dst(k, row16, n16, rank, dp16)] = msg(k);
}

// ================================================================ launches
void arw_rs_oneshot(int dtype, int nb, int nt, hipStream_t st, const int4* in4, int4* out4, int n16,
                    const ArwArgs& A) {
#define ARW_L(T, LN)                                                                              \
  do {                                                                                            \
    if (A.iw) arw_oneshot_rs_kernel<T, LN, true><<<nb, nt, 0, st>>>(in4, out4, n16, A);           \
    else arw_oneshot_rs_kernel<T, LN, false><<<nb, nt, 0, st>>>(in4, out4, n16, A);               \
  } while (0)
  if (dtype == 0) ARW_L(__hip_bfloat16, 8);
  else if (dtype == 1) ARW_L(__half, 8);
  else ARW_L(float, 4);
#undef ARW_L
}

void arw_rs_hc_write(bool gather, bool by_inv, int nb, int nt, hipStream_t st, int4* y4,
                     const unsigned short* inj, unsigned short* h, int n16, int n, int hc,
                     long long irow, long long hrow, const R4dArGather& G, const int* inv,
                     const ArwArgs& A) {
#define ARW_L(GA_, INV_, INJ_)                                                                    \
  arw_hc_write_rs_kernel<GA_, INV_, INJ_><<<nb, nt, 0, st>>>(y4, inj, h, n16, n, hc, irow, hrow,  \
                                                             G, inv, A)
  if (!gather) { if (A.iw) ARW_L(false, false, true); else ARW_L(false, false, false); }
  else if (by_inv) { if (A.iw) ARW_L(true, true, true); else ARW_L(true, true, false); }
  else { if (A.iw) ARW_L(true, false, true); else ARW_L(true, false, false); }
#undef ARW_L
}

void arw_rs_gather(int nb, int nt, hipStream_t st, const int4* in4, int4* out4, int n16, int row16,
                   int dp16, int sp16, const ArwArgs& A) {
  if (A.iw) arw_gather_rs_kernel<true><<<nb, nt, 0, st>>>(in4, out4, n16, row16, dp16, sp16, A);
  else arw_gather_rs_kernel<false><<<nb, nt, 0, st>>>(in4, out4, n16, row16, dp16, sp16, A);
}
