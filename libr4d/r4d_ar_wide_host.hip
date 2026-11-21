// r4d_ar_wide_host.hip: Brilliance, the four-rank host-mailbox collectives with the loud stop (stage W1
// of the four-rank resend; helpers and layout in r4d_ar_wide_x.h).
//
// r4d_ar_entry.hip launches these, not the default kernels, for every four-rank instance whose
// mailbox is in pinned host memory, on Windows (the instance's `wide_x`): the five kernels Flash-Next
// TP4 reaches on the host path (design section 1):
//   ar_oneshot_4rank_exact         arw_oneshot_kernel
//   ar_hc_write_4rank              arw_hc_write_kernel<false, ...>
//   ar_gather_hc_write_4rank       arw_hc_write_kernel<true, INV, ...>
//   ar_gather_4rank                arw_gather_kernel
//   ar_twoshot_4rank_exact (plain, and the one-shot row's ts_min band)   arw_twoshot_kernel
// Each is the default kernel's code step for step (the same sequence take, handshake values, slots,
// block partition, push loop and fp32 ascending-rank fold, so the same output bytes; they share an
// instance with the default kernels W1 does not cover: the coded and ti8 two-shots and the fused
// int8 norm), plus:
//   writer  every thread sums r4d_archk_mix over the words it pushes; thread 0 stores the block's sum,
//           salted by the handshake value, into each receiving peer's check word (phase, parity,
//           block, this rank) before the handshake's release (BRILLIANCE_AR4_CHECK's writer);
//   wait    the handshake's spin lanes are bounded (r4d_arw_handshake);
//   reader  the block checks each peer's words before it uses them, copying what it reads into the
//           instance's VRAM mirror (or, for a copy, straight into the output), and reduces from the
//           mirror (r4d_arw_check). A peer that never agrees, or a flag that never comes, ends in a
//           fault record and the engine's loud stop (status 70); the block finishes on what it read.
// The chained forms run plain here: the Windows backend never chains (core/device/dispatch.cpp turns
// chaining on only for the "aql" backend), and a plain kernel keeps the queue's barrier bit.
//
// This file was BRILLIANCE_AR4_CHECK's (a diagnostic that verified AFTER the reduce, reading the
// mailbox once): W1 retires the switch and keeps its check words and writer (design 2.3, item 5).
#include "r4d.h"
#include "r4d_log.h"   // Brilliance: W lines through the host's logger
#include <cstring>
#include <cstdio>
#include <mutex>
#include "r4d_ar_wide_x.h"
#include "r4d_ar_hc.h"
#include <stdexcept>
#include <string>

#include "r4d_ar_wide_host.h"

// ---------------------------------------------------------------- the one-shot all-reduce
template <typename T, int LANES, bool INJ>
__global__ void arw_oneshot_kernel(const int4* __restrict__ in4, int4* __restrict__ out4, int n16,
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

  auto msg = [&](int k) { return in4[k]; };
  R4dArwInj ij{ -1, 0, 0, 0, 0, 0, true, 0u };
  if constexpr (INJ)
    ij = r4d_arw_inj_bcast(A.iw, b, 0u, start, end, ps[arw_inj_peer(A.iw)], arw_ctl(A), msg);
  auto put = [&](int p, int k, const int4& v) {
    if constexpr (INJ) r4d_arw_inj_put(ij, p, ps[p], k, v, msg);
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
  const unsigned int hws[NP] = { hw, hw, hw };
  r4d_arw_put<WS, NP>(A.peers, b, rank, 0u, (unsigned)parity, hws, 2u * s);
  r4d_arw_handshake<WS, NP>(A.peers, A.my_flags, b, rank, 2u * s, A.drain, A.acq, A.pub,
                            INJ ? ij.noflag : 0u, OP, 0u, (unsigned)parity);

  const int4* sp[NP];
#pragma unroll
  for (int p = 0; p < NP; ++p) sp[p] = arw_slot(A, parity, A.peers.rank[p]);
  r4d_arw_check<WS, NP>(A.my_flags, A.peers, 0u, (unsigned)parity, b, 2u * s, OP,
                        [&](unsigned int mask, unsigned int (&h)[NP]) {
                          arw_rehash_slots(sp, A.md, start, end, mask, h);
                        });

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

// ---------------------------------------------------------------- the fused hc writes
// r4d_ar_hc_write_wide_kernel<4> (GATHER false) and r4d_ar_gather_hc_write_wide_kernel<4, INV>
// (GATHER true): up to the handshake the one-shot above with `y` as the message (gathered first when
// GATHER), then the reduce in place on `y` and the gated residual write.
template <bool GATHER, bool INV, bool INJ>
__global__ void arw_hc_write_kernel(int4* __restrict__ y4, const unsigned short* __restrict__ inj,
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
  R4dArwInj ij{ -1, 0, 0, 0, 0, 0, true, 0u };
  if constexpr (INJ)
    ij = r4d_arw_inj_bcast(A.iw, b, 0u, start, end, ps[arw_inj_peer(A.iw)], arw_ctl(A), msg);
  auto put = [&](int p, int k, const int4& v) {
    if constexpr (INJ) r4d_arw_inj_put(ij, p, ps[p], k, v, msg);
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
  const unsigned int hws[NP] = { hw, hw, hw };
  r4d_arw_put<WS, NP>(A.peers, b, rank, 0u, (unsigned)parity, hws, 2u * s);
  r4d_arw_handshake<WS, NP>(A.peers, A.my_flags, b, rank, 2u * s, A.drain, A.acq, A.pub,
                            INJ ? ij.noflag : 0u, OP, 0u, (unsigned)parity);

  const int4* sp[NP];
#pragma unroll
  for (int p = 0; p < NP; ++p) sp[p] = arw_slot(A, parity, A.peers.rank[p]);
  r4d_arw_check<WS, NP>(A.my_flags, A.peers, 0u, (unsigned)parity, b, 2u * s, OP,
                        [&](unsigned int mask, unsigned int (&hh)[NP]) {
                          arw_rehash_slots(sp, A.md, start, end, mask, hh);
                        });

  /* In place: a thread reads every source of word k before storing it, its block pushed the word
   * before the handshake, and no peer reads this rank's `y`. */
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

// ---------------------------------------------------------------- the exact two-shot (lockstep,
// one piece: the four-rank form). Hop 1: region A pushes, flag v0 = 2s - 1; hop 2: region B, v0 + 1.
struct ArwShard { int off[ARW_WS + 1]; int sh_max; };

template <typename T, int LANES, bool INJ>
__global__ void arw_twoshot_kernel(const int4* __restrict__ in4, int4* __restrict__ out4,
                                   ArwShard sh, ArwArgs A) {
  constexpr int WS = ARW_WS, NP = ARW_NP;
  constexpr unsigned int OP = R4D_AR_OP_W_TWOSHOT;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int rank = A.rank;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(A.seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot, 1u);
  const unsigned int v0 = 2u * s_seq - 1u;
  const int parity = (int)s_slot;
  const int sh_max = sh.sh_max;
  const int w = (sh_max + nb - 1) / nb;
  const int w0 = b * w;
  int w1 = w0 + w; if (w1 > sh_max) w1 = sh_max;
#define ARW_LEN(x) (sh.off[(x) + 1] - sh.off[x])
#define ARW_LIM(x) (w1 < ARW_LEN(x) ? w1 : ARW_LEN(x))

  int4* pa[NP];
#pragma unroll
  for (int p = 0; p < NP; ++p)
    pa[p] = A.peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)A.slot_stride16;

  // hop 1: my contribution to each peer-owned shard -> its owner's region A
  R4dArwInj ij1{ -1, 0, 0, 0, 0, 0, true, 0u };
  if constexpr (INJ) {
    const int ip = arw_inj_peer(A.iw), owner = A.peers.rank[ip];
    const int4* s1 = in4 + (size_t)sh.off[owner];
    ij1 = r4d_arw_inj_bcast(A.iw, b, 0u, w0, ARW_LIM(owner), pa[ip], arw_ctl(A),
                            [&](int k) { return s1[k]; });
  }
  unsigned int h1[NP] = { 0u, 0u, 0u };
  for (int i = 0; i < NP; ++i) {
    const int owner = A.peers.rank[i];
    const int4* s1 = in4 + (size_t)sh.off[owner];
    int4* dst = pa[i];
    const int lim = ARW_LIM(owner);
    unsigned int x = 0u;
    for (int k = w0 + tid; k < lim; k += nt) {
      const int4 v = s1[k];
      if constexpr (INJ) r4d_arw_inj_put(ij1, i, dst, k, v, [&](int kk) { return s1[kk]; });
      else dst[k] = v;
      x += r4d_archk_u4(v, 4u * k);
    }
    h1[i] = x;
  }
  r4d_arw_put<WS, NP>(A.peers, b, rank, 0u, (unsigned)parity, h1, v0);
  r4d_arw_handshake<WS, NP>(A.peers, A.my_flags, b, rank, v0, A.drain, A.acq, A.pub,
                            INJ ? ij1.noflag : 0u, OP, 0u, (unsigned)parity);

  // check hop 1: the peers' region A words of my shard, into the mirror
  const int my_lim = ARW_LIM(rank);
  const int4* sp[NP];
#pragma unroll
  for (int p = 0; p < NP; ++p) sp[p] = arw_slot(A, parity, A.peers.rank[p]);
  r4d_arw_check<WS, NP>(A.my_flags, A.peers, 0u, (unsigned)parity, b, v0, OP,
                        [&](unsigned int mask, unsigned int (&hh)[NP]) {
                          arw_rehash_slots(sp, A.md, w0, my_lim, mask, hh);
                        });

  // reduce my shard from the mirror (own contribution from the input) fused with the hop-2 push
  const int4* src[WS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    src[r] = (r == rank) ? (in4 + (size_t)sh.off[rank])
                         : r4d_archk_moved(arw_slot(A, parity, r), A.md);
  int4* my_out = out4 + (size_t)sh.off[rank];
  R4dArwInj ij2{ -1, 0, 0, 0, 0, 0, true, 0u };
  if constexpr (INJ)
    ij2 = r4d_arw_inj_bcast(A.iw, b, 1u, w0, my_lim, pa[arw_inj_peer(A.iw)] + sh_max, arw_ctl(A),
                            [&](int k) { return arw_vsum<T, LANES>(src, k); });
  unsigned int ho = 0u;
  for (int k = w0 + tid; k < my_lim; k += nt) {
    const int4 o = arw_vsum<T, LANES>(src, k);
    my_out[k] = o;
#pragma unroll
    for (int p = 0; p < NP; ++p) {
      if constexpr (INJ)
        r4d_arw_inj_put(ij2, p, pa[p] + sh_max, k, o, [&](int kk) { return arw_vsum<T, LANES>(src, kk); });
      else pa[p][sh_max + k] = o;
    }
    ho += r4d_archk_u4(o, 4u * k);
  }
  const unsigned int hos[NP] = { ho, ho, ho };
  r4d_arw_put<WS, NP>(A.peers, b, rank, 1u, (unsigned)parity, hos, v0 + 1u);
  r4d_arw_handshake<WS, NP>(A.peers, A.my_flags, b, rank, v0 + 1u, A.drain, A.acq, A.pub,
                            INJ ? ij2.noflag : 0u, OP, 1u, (unsigned)parity);

  // check hop 2 and assemble: each peer's region B copy of its shard, straight into the output (a
  // retry rewrites the same words; the output is this kernel's to write)
  r4d_arw_check<WS, NP>(A.my_flags, A.peers, 1u, (unsigned)parity, b, v0 + 1u, OP,
                        [&](unsigned int mask, unsigned int (&hh)[NP]) {
#pragma unroll
    for (int p = 0; p < NP; ++p) {
      if (!((mask >> p) & 1u)) continue;
      const int owner = A.peers.rank[p];
      const int4* srcB = arw_slot(A, parity, owner) + sh_max;
      int4* dst = out4 + (size_t)sh.off[owner];
      const int lim = ARW_LIM(owner);
      for (int k = w0 + tid; k < lim; k += nt) {
        const int4 v = srcB[k];
        dst[k] = v;
        hh[p] += r4d_archk_u4(v, 4u * k);
      }
    }
  });
#undef ARW_LIM
#undef ARW_LEN
}

// ---------------------------------------------------------------- the all-gather
template <bool INJ>
__global__ void arw_gather_kernel(const int4* __restrict__ in4, int4* __restrict__ out4, int n16,
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
  auto msg = [&](int k) { return in4[arw_g_src(k, row16, n16, sp16)]; };
  R4dArwInj ij{ -1, 0, 0, 0, 0, 0, true, 0u };
  if constexpr (INJ)
    ij = r4d_arw_inj_bcast(A.iw, b, 0u, start, end, ps[arw_inj_peer(A.iw)], arw_ctl(A), msg);
  auto put = [&](int p, int k, const int4& v) {
    if constexpr (INJ) r4d_arw_inj_put(ij, p, ps[p], k, v, msg);
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
  const unsigned int hws[NP] = { hw, hw, hw };
  r4d_arw_put<WS, NP>(A.peers, b, rank, 0u, (unsigned)parity, hws, 2u * s);
  r4d_arw_handshake<WS, NP>(A.peers, A.my_flags, b, rank, 2u * s, A.drain, A.acq, A.pub,
                            INJ ? ij.noflag : 0u, OP, 0u, (unsigned)parity);

  // Each peer's slice, checked as it is placed (the output is the mirror: a retry rewrites it).
  r4d_arw_check<WS, NP>(A.my_flags, A.peers, 0u, (unsigned)parity, b, 2u * s, OP,
                        [&](unsigned int mask, unsigned int (&hh)[NP]) {
#pragma unroll
    for (int p = 0; p < NP; ++p) {
      if (!((mask >> p) & 1u)) continue;
      const int owner = A.peers.rank[p];
      const int4* spp = arw_slot(A, parity, owner);
      for (int k = start + tid; k < end; k += nt) {
        const int4 v = spp[k];
        out4[arw_g_dst(k, row16, n16, owner, dp16)] = v;
        hh[p] += r4d_archk_u4(v, 4u * k);
      }
    }
  });
  // This rank's own slice, from the input at its own pitch.
  for (int k = start + tid; k < end; k += nt)
    out4[arw_g_dst(k, row16, n16, rank, dp16)] = msg(k);
}

// ================================================================ host side
namespace {

R4DArWidePeers<int4, ARW_NP> arw_peers(const long long* ps, const long long* pf, int rank,
                                       const char* what) {
  if (rank < 0 || rank >= ARW_WS)
    throw std::runtime_error(std::string(what) + ": rank " + std::to_string(rank) +
                             " out of range for four ranks");
  R4DArWidePeers<int4, ARW_NP> peers{};
  for (int i = 0; i < ARW_NP; ++i) {
    if (!ps[i] || !pf[i])
      throw std::runtime_error(std::string(what) + ": null peer scratch/flag pointer at index " +
                               std::to_string(i));
    peers.scratch[i] = (int4*)ps[i];
    peers.flags[i] = (unsigned int*)pf[i];
    peers.rank[i] = (i < rank) ? i : i + 1;
  }
  return peers;
}

int arw_nt(long long nthreads) {
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  return nt;
}

ArwArgs arw_args(const long long* ps, const long long* pf, long long my_scratch, long long my_flags,
                 long long seq_ctrs, long long slot_stride16, long long rank, long long drain,
                 long long acq, long long pub, long long stage, const char* what) {
  if (!stage) throw std::runtime_error(std::string(what) + ": no VRAM mirror (not a wide host instance)");
  ArwArgs A;
  std::memset(&A, 0, sizeof A);   // the struct is kernel-argument bytes
  A.peers = arw_peers(ps, pf, (int)rank, what);
  A.my_base = (const int4*)my_scratch;
  A.my_flags = (unsigned int*)my_flags;
  A.seq_ctrs = (unsigned int*)seq_ctrs;
  A.md = stage - my_scratch;
  A.slot_stride16 = (int)slot_stride16;
  A.rank = (int)rank;
  A.drain = (int)drain;
  /* acq 2 (no fence) is for fine-grained VRAM scratch only; a host mailbox is read past this card's
   * caches only after a system-scope acquire (design 3.1, the ar4-audit). */
  A.acq = acq == 2 ? 0 : (int)acq;
  A.pub = (int)pub;
  return A;
}

// ---- injection, the launcher's half: the entry (r4d_ar_entry.hip) counts the launches and passes a
// request: mode in bits 0-3, the target peer's RANK in 4-6, the phase (two-shot hop) in 7, the
// injection's ordinal from bit 8. This picks the block (stepping from the middle, ordinal * 5) and
// the middle line of the block, logs them, and keeps them for r4d_arw_inject_last.
std::mutex g_inj_mu;
long long g_inj_last[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
const char* const k_mode[] = { "?", "lose", "dup", "stuck", "noflag", "pair", "late", "ctl-ack",
                               "ctl-rq", "multi", "cross" };

const char* op_name(unsigned int op) {
  switch (op) {
  case R4D_AR_OP_W_ONESHOT: return "ar_oneshot_4rank_exact";
  case R4D_AR_OP_W_TWOSHOT: return "ar_twoshot_4rank_exact";
  case R4D_AR_OP_W_CODED:   return "ar_twoshot_4rank_exact (coded)";
  case R4D_AR_OP_W_HC:      return "ar_hc_write_4rank";
  case R4D_AR_OP_W_GHC:     return "ar_gather_hc_write_4rank";
  case R4D_AR_OP_W_GATHER:  return "ar_gather_4rank";
  default:                  return "?";
  }
}

// `rs`: the launch runs the exchange (W2/W3; the two-shot from W4). Without it
// (BRILLIANCE_AR_RESEND=0) the exchange's modes act on W1's kernels as lose (pair, late, multi) or as
// nothing (ctl-ack, ctl-rq), as the two-rank modes do on stage 1's. Cross reaches each of its two
// writers and damages that writer's push as lose.
unsigned int inj_pick(long long req, unsigned int op, int rank, int nb, int words_a_block, bool rs) {
  if (!req) return 0u;
  const unsigned int asked = (unsigned int)(req & 15);
  const int peer_rank = (int)((req >> 4) & 7);
  const unsigned int phase = (unsigned int)((req >> 7) & 1);
  const unsigned long long ord = (unsigned long long)req >> 8;
  if (!asked || asked > R4D_AR_INJ_CROSS || peer_rank == rank || peer_rank >= ARW_WS || nb < 1) return 0u;
  unsigned int mode = asked;
  if (asked == R4D_AR_INJ_CROSS) mode = R4D_AR_INJ_LOSE;
  if (!rs) {
    if (asked == R4D_AR_INJ_ACK || asked == R4D_AR_INJ_RQ) return 0u;
    if (asked == R4D_AR_INJ_PAIR || asked == R4D_AR_INJ_LATE || asked == R4D_AR_INJ_MULTI) mode = R4D_AR_INJ_LOSE;
  }
  const int pidx = peer_rank < rank ? peer_rank : peer_rank - 1;
  const int blk = (int)(((unsigned long long)(nb / 2) + ord * 5ull) % (unsigned long long)nb);
  const int lines = words_a_block / 4;
  int off = lines > 2 ? (lines - 1) / 2 : 0;
  if (off > 255) off = 255;
  {
    std::lock_guard<std::mutex> lk(g_inj_mu);
    ++g_inj_last[0];
    g_inj_last[1] = op; g_inj_last[2] = asked; g_inj_last[3] = blk; g_inj_last[4] = off;
    g_inj_last[5] = nb; g_inj_last[6] = (long long)ord; g_inj_last[7] = peer_rank;
    g_inj_last[8] = phase;
  }
  int dev = -1;
  if (hipGetDevice(&dev) != hipSuccess) (void)hipGetLastError();
  r4d_log(1, "r4d: fault injection (BRILLIANCE_AR_INJECT): %s%s%s into %s%s%s on device "
          "%d, rank %d's push to rank %d, block %d of %d (about %d words a block), line %d of the "
          "block; injection %llu", k_mode[asked], mode != asked ? " as " : "",
          mode != asked ? k_mode[mode] : "", op_name(op),
          op == R4D_AR_OP_W_TWOSHOT || op == R4D_AR_OP_W_CODED ? (phase ? " (hop 2)" : " (hop 1)") : "",
          rs ? " (with the resend)" : "", dev, rank, peer_rank, blk, nb, words_a_block, off,
          ord + 1ull);
  return r4d_arw_inj_word(mode, (unsigned int)blk, (unsigned int)pidx, phase, (unsigned int)off);
}

}  // namespace

// Test hook (r4d_selftest): the wide injections the launchers made, and the last one. out[0] how
// many, then the last's op (R4D_AR_OP_W_*), mode, block, line offset, block count, ordinal, target
// peer rank and phase. Returns the words written.
extern "C" int r4d_arw_inject_last(long long* out, int n) {
  if (!out || n <= 0) return 0;
  std::lock_guard<std::mutex> lk(g_inj_mu);
  const int m = n < 9 ? n : 9;
  for (int i = 0; i < m; ++i) out[i] = g_inj_last[i];
  return m;
}

extern "C" void r4d_arw_oneshot(const long long* ps, const long long* pf, long long my_scratch,
                                long long my_flags, long long seq_ctrs, long long slot_stride16,
                                long long inp, long long out, long long n_elem, long long dtype,
                                long long rank, long long stream, long long nblocks,
                                long long nthreads, long long drain, long long acq, long long pub,
                                long long stage, long long inj, long long rs) {
  const char* what = "ar_oneshot_4rank_exact (host)";
  const int esize = (dtype == 2) ? 4 : 2;
  const long long nbytes = n_elem * esize;
  if (nbytes % 16) throw std::runtime_error(std::string(what) + ": not 16B-aligned");
  const long long n16l = nbytes / 16;
  if (n16l > slot_stride16) throw std::runtime_error(std::string(what) + ": message exceeds the slot");
  const int n16 = (int)n16l;
  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  const int nt = arw_nt(nthreads);
  ArwArgs A = arw_args(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, rank, drain, acq, pub,
                       stage, what);
  A.iw = inj_pick(inj, R4D_AR_OP_W_ONESHOT, (int)rank, nb, (n16 + nb - 1) / nb, rs != 0);
  hipStream_t st = (hipStream_t)stream;
#define ARW_L(T, LN)                                                                              \
  do {                                                                                            \
    if (A.iw) arw_oneshot_kernel<T, LN, true><<<nb, nt, 0, st>>>((const int4*)inp, (int4*)out, n16, A); \
    else arw_oneshot_kernel<T, LN, false><<<nb, nt, 0, st>>>((const int4*)inp, (int4*)out, n16, A); \
  } while (0)
  if (rs) {
    if (dtype < 0 || dtype > 2) throw std::runtime_error(std::string(what) + ": bad dtype");
    arw_rs_oneshot((int)dtype, nb, nt, st, (const int4*)inp, (int4*)out, n16, A);
    return;
  }
  if (dtype == 0) ARW_L(__hip_bfloat16, 8);
  else if (dtype == 1) ARW_L(__half, 8);
  else if (dtype == 2) ARW_L(float, 4);
  else throw std::runtime_error(std::string(what) + ": bad dtype");
#undef ARW_L
}

extern "C" void r4d_arw_twoshot(const long long* ps, const long long* pf, long long my_scratch,
                                long long my_flags, long long seq_ctrs, long long slot_stride16,
                                long long inp, long long out, long long n_elem, long long dtype,
                                long long rank, long long stream, long long nblocks,
                                long long nthreads, long long drain, long long acq, long long pub,
                                long long stage, long long inj, long long rs) {
  const char* what = "ar_twoshot_4rank_exact (host)";
  const int esize = (dtype == 2) ? 4 : 2;
  const long long nbytes = n_elem * esize;
  if (nbytes % 16) throw std::runtime_error(std::string(what) + ": not 16B-aligned");
  if (n_elem > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS)
    throw std::runtime_error(std::string(what) + ": exceeds the two-shot band");
  const long long n16 = nbytes / 16;
  if (n16 < ARW_WS) throw std::runtime_error(std::string(what) + ": shorter than a word a rank");
  long long off[ARW_WS + 1];
  const long long sh_max = r4d_ar_shard_table(n16, ARW_WS, nullptr, off);
  ArwShard sh{};
  for (int s = 0; s <= ARW_WS; ++s) sh.off[s] = (int)off[s];
  sh.sh_max = (int)sh_max;
  if (2 * sh_max > slot_stride16) throw std::runtime_error(std::string(what) + ": shard exceeds the slot");
  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > sh_max) nb = (int)sh_max;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  const int nt = arw_nt(nthreads);
  ArwArgs A = arw_args(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, rank, drain, acq, pub,
                       stage, what);
  A.iw = inj_pick(inj, R4D_AR_OP_W_TWOSHOT, (int)rank, nb, (int)((sh_max + nb - 1) / nb), rs != 0);
  hipStream_t st = (hipStream_t)stream;
  if (rs) {   /* W4: the exchange (r4d_ar_wide_ts.hip) */
    if (dtype < 0 || dtype > 2) throw std::runtime_error(std::string(what) + ": bad dtype");
    arw_rs_twoshot((int)dtype, nb, nt, st, (const int4*)inp, (int4*)out, sh.off, sh.sh_max, A);
    return;
  }
#define ARW_L(T, LN)                                                                              \
  do {                                                                                            \
    if (A.iw) arw_twoshot_kernel<T, LN, true><<<nb, nt, 0, st>>>((const int4*)inp, (int4*)out, sh, A); \
    else arw_twoshot_kernel<T, LN, false><<<nb, nt, 0, st>>>((const int4*)inp, (int4*)out, sh, A); \
  } while (0)
  if (dtype == 0) ARW_L(__hip_bfloat16, 8);
  else if (dtype == 1) ARW_L(__half, 8);
  else if (dtype == 2) ARW_L(float, 4);
  else throw std::runtime_error(std::string(what) + ": bad dtype");
#undef ARW_L
}

// W4: the coded two-shot (overlap 1) with the exchange; only with the resend on (under
// BRILLIANCE_AR_RESEND=0 the entry keeps the default coded kernel, as W1 did). The default coded
// launcher's checks and grid (r4d_ar_twoshot_4rank_exact_coded).
extern "C" void r4d_arw_twoshot_coded(const long long* ps, const long long* pf, long long my_scratch,
                                      long long my_flags, long long seq_ctrs, long long slot_stride16,
                                      long long inp, long long out, long long n_elem, long long rank,
                                      long long stream, long long nblocks, long long nthreads,
                                      long long drain, long long acq, long long pub, long long stage,
                                      long long inj) {
  const char* what = "ar_twoshot_4rank_exact (coded, host)";
  if (n_elem <= 0 || n_elem % (ARW_WS * 256) != 0)
    throw std::runtime_error(std::string(what) + ": the message is not whole 256-value waves a rank");
  const long long n16 = n_elem / 8, SL = n16 / ARW_WS;
  if (2 * arw_coded_region_bytes(SL) > (size_t)slot_stride16 * 16)
    throw std::runtime_error(std::string(what) + ": a slot does not hold two coded regions");
  int nb = (int)nblocks;
  if (nb < 1) nb = 4;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  if (nb > SL / 32) nb = (int)(SL / 32);
  int nt = (int)nthreads;
  if (nt < 256) nt = 256;
  if (nt > 1024) nt = 1024;
  nt &= ~31;
  ArwArgs A = arw_args(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, rank, drain, acq, pub,
                       stage, what);
  const long long wb = (((SL + nb - 1) / nb) + 31) & ~31LL;
  A.iw = inj_pick(inj, R4D_AR_OP_W_CODED, (int)rank, nb, (int)wb, true);
  arw_rs_coded(nb, nt, (hipStream_t)stream, (const int4*)inp, (int4*)out, (int)n16, A);
}

extern "C" void r4d_arw_hc_write(const long long* ps, const long long* pf, long long my_scratch,
                                 long long my_flags, long long seq_ctrs, long long slot_stride16,
                                 long long y, long long inj, long long h, long long M, long long n,
                                 long long hc, long long irow, long long hrow, long long rank,
                                 long long stream, long long nblocks, long long nthreads,
                                 long long drain, long long acq, long long pub, long long stage,
                                 long long injreq, long long rs) {
  const char* what = "ar_hc_write_4rank (host)";
  if (M <= 0 || n <= 0 || n % 8) throw std::runtime_error(std::string(what) + ": n must be a positive multiple of 8");
  if (hc < 1 || hrow % 8) throw std::runtime_error(std::string(what) + ": the streams must be whole 16B words");
  const long long n16l = M * n / 8;
  if (n16l > slot_stride16) throw std::runtime_error(std::string(what) + ": message exceeds the slot");
  const int n16 = (int)n16l;
  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  const int nt = arw_nt(nthreads);
  ArwArgs A = arw_args(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, rank, drain, acq, pub,
                       stage, what);
  A.iw = inj_pick(injreq, R4D_AR_OP_W_HC, (int)rank, nb, (n16 + nb - 1) / nb, rs != 0);
  R4dArGather G;
  std::memset(&G, 0, sizeof G);
  hipStream_t st = (hipStream_t)stream;
  if (rs)
    arw_rs_hc_write(false, false, nb, nt, st, (int4*)y, (const unsigned short*)inj, (unsigned short*)h,
                    n16, (int)n, (int)hc, irow, hrow, G, nullptr, A);
  else if (A.iw)
    arw_hc_write_kernel<false, false, true><<<nb, nt, 0, st>>>(
        (int4*)y, (const unsigned short*)inj, (unsigned short*)h, n16, (int)n, (int)hc, irow, hrow, G,
        nullptr, A);
  else
    arw_hc_write_kernel<false, false, false><<<nb, nt, 0, st>>>(
        (int4*)y, (const unsigned short*)inj, (unsigned short*)h, n16, (int)n, (int)hc, irow, hrow, G,
        nullptr, A);
}

extern "C" void r4d_arw_gather_hc_write(
    const long long* ps, const long long* pf, long long my_scratch, long long my_flags,
    long long seq_ctrs, long long slot_stride16, long long y, long long inj, long long h, long long M,
    long long n, long long hc, long long irow, long long hrow, long long ye, long long ew,
    long long sorted, long long sh, long long sg, long long ye_ld, long long ew_ld, long long sh_ld,
    long long sg_ld, long long top_k, long long sig, long long rank, long long stream,
    long long nblocks, long long nthreads, long long drain, long long acq, long long pub,
    long long inv, long long stage, long long injreq, long long rs) {
  const char* what = "ar_gather_hc_write_4rank (host)";
  if (M <= 0 || n <= 0 || n % 8) throw std::runtime_error(std::string(what) + ": n must be a positive multiple of 8");
  if (hc < 1 || hrow % 8) throw std::runtime_error(std::string(what) + ": the streams must be whole 16B words");
  if (top_k < 1 || top_k > R4D_ARG_SLOTS)
    throw std::runtime_error(std::string(what) + ": top_k " + std::to_string(top_k) + " outside 1.." +
                             std::to_string(R4D_ARG_SLOTS));
  if (!ye || !ew || !sorted || ye_ld % 8 || (sh && sh_ld < n) || (sh != 0) != (sg != 0))
    throw std::runtime_error(std::string(what) + ": the gather's operands are not 16B rows");
  const long long n16l = M * n / 8;
  if (n16l > slot_stride16) throw std::runtime_error(std::string(what) + ": message exceeds the slot");
  const int n16 = (int)n16l;
  const int wpr = (int)(n / 8);
  // The default launcher's grid rule (r4d_ar_oneshot_Nrank_exact.hip).
  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  const int wmax = (R4D_ARG_ROWS - 2) * wpr + 1;
  const int nb_min = (n16 + wmax - 1) / wmax;
  const bool by_inv = inv != 0 && nb_min > nb;
  if (!by_inv && nb < nb_min) nb = nb_min;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS)
    throw std::runtime_error(std::string(what) + ": the slot table needs " + std::to_string(nb) +
                             " blocks, above the " + std::to_string(R4D_AR_WIDE_MAX_BLOCKS) +
                             " the flags hold, and no inverse table was given");
  const int nt = arw_nt(nthreads);
  ArwArgs A = arw_args(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, rank, drain, acq, pub,
                       stage, what);
  R4dArGather G;
  std::memset(&G, 0, sizeof G);
  G.ye = (const unsigned short*)ye;
  G.ew = (const unsigned short*)ew;
  G.sorted = (const int*)sorted;
  G.sh = (const unsigned short*)sh;
  G.sg = (const unsigned short*)sg;
  G.ye_ld = ye_ld; G.ew_ld = ew_ld; G.sh_ld = sh_ld; G.sg_ld = sg_ld;
  G.top_k = (int)top_k; G.T = (int)(M * top_k); G.sig = (int)sig;
  hipStream_t st = (hipStream_t)stream;
  if (by_inv) {
    const int irc = r4d_ar_gather_inv(inv, sorted, (long long)G.T, stream);
    if (irc != 0)
      throw std::runtime_error(std::string(what) + ": the inverse table's launch failed (" +
                               std::to_string(irc) + ")");
  }
  A.iw = inj_pick(injreq, R4D_AR_OP_W_GHC, (int)rank, nb, (n16 + nb - 1) / nb, rs != 0);
#define ARW_L(INV_, INJ_, inv_)                                                                   \
  arw_hc_write_kernel<true, INV_, INJ_><<<nb, nt, 0, st>>>(                                       \
      (int4*)y, (const unsigned short*)inj, (unsigned short*)h, n16, (int)n, (int)hc, irow, hrow,   \
      G, inv_, A)
  if (rs)
    arw_rs_hc_write(true, by_inv, nb, nt, st, (int4*)y, (const unsigned short*)inj, (unsigned short*)h,
                    n16, (int)n, (int)hc, irow, hrow, G, by_inv ? (const int*)inv : nullptr, A);
  else if (by_inv) { if (A.iw) ARW_L(true, true, (const int*)inv); else ARW_L(true, false, (const int*)inv); }
  else { if (A.iw) ARW_L(false, true, nullptr); else ARW_L(false, false, nullptr); }
#undef ARW_L
}

extern "C" void r4d_arw_gather(const long long* ps, const long long* pf, long long my_scratch,
                               long long my_flags, long long seq_ctrs, long long slot_stride16,
                               long long inp, long long out, long long n_elem, long long dtype,
                               long long rank, long long row_elems, long long stream,
                               long long nblocks, long long nthreads, long long drain,
                               long long acq, long long pub, long long dpitch, long long spitch,
                               long long stage, long long inj, long long rs) {
  const char* what = "ar_gather_4rank (host)";
  const int esize = (dtype == 2) ? 4 : 2;
  const long long nbytes = n_elem * esize;
  if (nbytes % 16) throw std::runtime_error(std::string(what) + ": not 16B-aligned");
  const long long n16l = nbytes / 16;
  if (n16l > slot_stride16) throw std::runtime_error(std::string(what) + ": message exceeds the slot");
  const int n16 = (int)n16l;
  const long long rowb = row_elems * esize;
  if (row_elems > 0 && (rowb % 16 || nbytes % rowb))
    throw std::runtime_error(std::string(what) + ": the row is not whole 16B words dividing the message");
  const int row16 = row_elems > 0 ? (int)(rowb / 16) : 0;
  const long long dpb = dpitch * esize;
  const int dp16 = dpitch > 0 ? (int)(dpb / 16) : ARW_WS * row16;
  const long long spb = spitch * esize;
  const int sp16 = spitch > 0 ? (int)(spb / 16) : row16;
  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  const int nt = arw_nt(nthreads);
  ArwArgs A = arw_args(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, rank, drain, acq, pub,
                       stage, what);
  A.iw = inj_pick(inj, R4D_AR_OP_W_GATHER, (int)rank, nb, (n16 + nb - 1) / nb, rs != 0);
  hipStream_t st = (hipStream_t)stream;
  if (rs)
    arw_rs_gather(nb, nt, st, (const int4*)inp, (int4*)out, n16, row16, dp16, sp16, A);
  else if (A.iw)
    arw_gather_kernel<true><<<nb, nt, 0, st>>>((const int4*)inp, (int4*)out, n16, row16, dp16, sp16, A);
  else
    arw_gather_kernel<false><<<nb, nt, 0, st>>>((const int4*)inp, (int4*)out, n16, row16, dp16, sp16, A);
}
