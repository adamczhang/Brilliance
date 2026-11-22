// r4d_ar_wide_ts.hip: Brilliance, the four-rank host-mailbox two-shot with the resend (stage W4 of
// the four-rank resend; the exchange is r4d_arw_exchange in
// r4d_ar_wide_x.h). r4d_ar_wide_host.hip's launchers launch these through the wrappers at the end
// when the resend is on (the default; BRILLIANCE_AR_RESEND=0 keeps W1's plain two-shot, and the
// coded form's default kernel, as before W4). A file of its own, so that W1's and W2/W3's code
// objects stay byte for byte what they were (one translation unit lays out its kernels' LDS
// together).
//
// The two-shot runs two exchanges a block, one a hop, each W2's loop with its own phase index (0, 1)
// and handshake value (v0 = 2s - 1, then v0 + 1), so the hops' check words, acks and requests never
// meet. Where each hop resends from (design 3.3), and why it survives:
//   hop 1  my contribution to each peer-owned shard, in4 at the owner's offset. Only the hop-2 check
//          (the assemble) writes those words when in place, and it runs after hop 1's exchange has
//          ended, i.e. after every peer acked (or was covered by its hop-2 flag).
//   hop 2  the reduced words of my own shard, read back from my out. Nothing writes them again: the
//          assemble writes the other shards only.
// The coded form re-encodes the same words for a resend: r4d_arc_enc is a pure function of a wave's
// 32 chunks, so the resent codes are the first push's bytes. Its check hashes the DECODED words
// (the codec gives back every value's 16 bits, r4d_ar_arc.h), so a damaged code that decodes
// differently fails, and the reduce reads the decoded words from the mirror.
#include "r4d.h"
#include "r4d_ar_wide_host.h"
#include "r4d_ar_arc.h"
#include "r4d_ar_wide_coded.h"   // tp4-link-weight: the coded helpers, shared with r4d_ar_wide_tw.hip

// ================================================================ the plain two-shot, RS (W4)
struct ArwTs { int off[ARW_WS + 1]; int sh_max; };

// The resend's re-push of words [w0, lim[p]) of s[p] to d[p], for the peer indices in `mask`; stuck
// leaves the injected words out again.
template <bool INJ>
__device__ __forceinline__ void arw_ts_repush(int4* const (&d)[ARW_NP], const int4* const (&s)[ARW_NP],
                                              const int (&lim)[ARW_NP], unsigned int mask, int w0,
                                              bool stuck, const R4dArwInjX& ij) {
  const int tid = threadIdx.x, nt = blockDim.x;
#pragma unroll
  for (int p = 0; p < ARW_NP; ++p) {
    if (!((mask >> p) & 1u)) continue;
    const bool skip = INJ && stuck && ((ij.pmask >> p) & 1u);
    for (int k = w0 + tid; k < lim[p]; k += nt) {
      if (skip && k >= ij.d.lo && k < ij.d.hi) continue;
      d[p][k] = s[p][k];
    }
  }
}
// Injection late's withheld words, now.
__device__ __forceinline__ void arw_ts_late(int4* const (&d)[ARW_NP], const int4* const (&s)[ARW_NP],
                                            const R4dArwInjX& ij) {
#pragma unroll
  for (int p = 0; p < ARW_NP; ++p) {
    if (!((ij.pmask >> p) & 1u)) continue;
    for (int k = ij.d.lo + (int)threadIdx.x; k < ij.d.hi; k += (int)blockDim.x) d[p][k] = s[p][k];
  }
}
// ctl-rq's overwrite of words [blo, bhi) of this rank's own slot region from bpeer (`mine`).
__device__ __forceinline__ void arw_ts_bad(int4* mine, const R4dArwInjX& ij) {
  for (int k = ij.blo + (int)threadIdx.x; k < ij.bhi; k += (int)blockDim.x) {
    int4 v = mine[k];
    v.x ^= 0x5A5A5A5A; v.y ^= 0x5A5A5A5A; v.z ^= 0x5A5A5A5A; v.w ^= 0x5A5A5A5A;
    mine[k] = v;
  }
}
// arw_ts_pick, arw_ts_pickc: r4d_ar_wide_coded.h (included above).

// The injection plan of one hop (thread 0's, to the block through LDS; two barriers). W2's plan
// (r4d_arw_injx_plan) with one change for hop 1, where each peer gets a different shard: multi's
// second peer is judged on its own words (`msgp(p, k)`), not the target's.
template <typename FM, typename FP, typename FS, typename FO>
__device__ __forceinline__ R4dArwInjX arw_ts_plan(unsigned int iw, int b, unsigned int phase, int start,
                                                  int end, int w, int n16, unsigned int* ctl,
                                                  const FM& msg, const FP& msgp, const FS& slotp,
                                                  const FO& mys, const int (&lim)[ARW_NP]) {
  __shared__ R4dArwInjX s_tsx;
  if (threadIdx.x == 0) {
    const bool multi1 = phase == 0u && (iw & 15u) == R4D_AR_INJ_MULTI;
    const unsigned int iwp = multi1 ? ((iw & ~15u) | R4D_AR_INJ_LOSE) : iw;
    R4dArwInjX r = r4d_arw_injx_plan<ARW_NP>(iwp, b, phase, start, end, w, n16, ctl, msg, slotp, mys);
    if (multi1 && r.d.peer >= 0) {
      const int p2 = (r.d.peer + 1) % ARW_NP;
      r.pmask |= 1u << p2;
      const int hi = r.d.hi < lim[p2] ? r.d.hi : lim[p2];
      if (r.d.lo < hi && r4d_arw_inj_changes(slotp(p2), r.d.lo, hi, [&](int k) { return msgp(p2, k); }))
        r4d_arw_inj_count(ctl);
    }
    s_tsx = r;
  }
  __syncthreads();
  const R4dArwInjX r = s_tsx;
  __syncthreads();
  return r;
}

// W1's arw_twoshot_kernel with the push check and bounded handshake of each hop replaced by the
// exchange. The same sequence take, slots, block partition, pushes and fold, so the same bytes.
template <typename T, int LANES, bool INJ>
__global__ void arw_twoshot_rs_kernel(const int4* __restrict__ in4, int4* __restrict__ out4, ArwTs sh,
                                      ArwArgs A) {
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

  int4* pa[NP];   // my slot in each peer's scratch, region A
  int4* pb[NP];   // and region B
  const int4* s1[NP];   // hop 1's words for peer index p: my input at its rank's shard
  const int4* sp[NP];   // peer p's slot in my scratch, region A
  int lim1[NP];
#pragma unroll
  for (int p = 0; p < NP; ++p) {
    const int owner = A.peers.rank[p];
    pa[p] = A.peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)A.slot_stride16;
    pb[p] = pa[p] + sh_max;
    s1[p] = in4 + (size_t)sh.off[owner];
    sp[p] = arw_slot(A, parity, owner);
    lim1[p] = ARW_LIM(owner);
  }
  const int my_lim = ARW_LIM(rank);
  int4* const my_out = out4 + (size_t)sh.off[rank];

  /* The injected words lie in what both the target peer's push and this rank's check cover. */
  [[maybe_unused]] int ip = 0, cn = 0;
  int cend = 0;
  if constexpr (INJ) {
    ip = arw_inj_peer(A.iw);
    const int ot = A.peers.rank[ip];
    cend = my_lim < ARW_LIM(ot) ? my_lim : ARW_LIM(ot);
    cn = ARW_LEN(rank) < ARW_LEN(ot) ? ARW_LEN(rank) : ARW_LEN(ot);
  }

  // ---- hop 1: my contribution to each peer-owned shard -> its owner's region A
  R4dArwInjX ij1 = r4d_arw_injx_none(w0, cend);
  if constexpr (INJ)
    ij1 = arw_ts_plan(A.iw, b, 0u, w0, cend, w, cn, arw_ctl(A),
                      [&](int k) { return arw_ts_pickc(s1, ip)[k]; },
                      [&](int p, int k) { return arw_ts_pickc(s1, p)[k]; },
                      [&](int p) { return arw_ts_pick(pa, p); },
                      [&](int p) { return arw_ts_pickc(sp, p); }, lim1);
  unsigned int h1[NP] = { 0u, 0u, 0u };
#pragma unroll
  for (int p = 0; p < NP; ++p) {
    const int4* const s = s1[p];
    int4* const dst = pa[p];
    unsigned int x = 0u;
    for (int k = w0 + tid; k < lim1[p]; k += nt) {
      const int4 v = s[k];
      if constexpr (INJ) r4d_arw_injx_put(ij1, p, dst, k, v, [&](int kk) { return arw_ts_pickc(s1, ip)[kk]; });
      else dst[k] = v;
      x += r4d_archk_u4(v, 4u * k);
    }
    h1[p] = x;
  }
  if constexpr (INJ)
    r4d_arw_injx_pair(ij1, arw_ts_pick(pa, ij1.cpeer), arw_ctl(A),
                      [&](int k) { return arw_ts_pickc(s1, ij1.cpeer)[k]; });
  unsigned int hck1[NP];
  r4d_arw_put_t<WS, NP>(A.peers, b, rank, 0u, (unsigned)parity, h1, v0, hck1);
  r4d_arw_publish<WS, NP>(A.peers, b, rank, v0, A.drain, A.pub, INJ ? ij1.d.noflag : 0u);
  r4d_arw_exchange<WS, NP, INJ>(
      A.peers, A.my_flags, b, rank, v0, 0u, (unsigned)parity, OP, hck1, A.drain,
      [&](unsigned int mask, unsigned int (&hh)[NP]) { arw_rehash_slots(sp, A.md, w0, my_lim, mask, hh); },
      [&](unsigned int mask, bool stuck) { arw_ts_repush<INJ>(pa, s1, lim1, mask, w0, stuck, ij1); },
      [&]() { arw_ts_late(pa, s1, ij1); },
      [&]() { arw_ts_bad(const_cast<int4*>(arw_ts_pickc(sp, ij1.bpeer < NP ? ij1.bpeer : 0)), ij1); },
      ij1);

  // ---- reduce my shard from the mirror (own contribution from the input), and hop 2: the reduced
  // words to my out and every peer's region B
  const int4* src[WS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    src[r] = (r == rank) ? (in4 + (size_t)sh.off[rank])
                         : r4d_archk_moved(arw_slot(A, parity, r), A.md);
  const int4* const myo = my_out;
  const int4* s2[NP] = { myo, myo, myo };
  int lim2[NP] = { my_lim, my_lim, my_lim };
  unsigned int ho = 0u;
  R4dArwInjX ij2 = r4d_arw_injx_none(w0, cend);
  if constexpr (!INJ) {
    /* W1's pass: the reduce fused with the push. */
    for (int k = w0 + tid; k < my_lim; k += nt) {
      const int4 o = arw_vsum<T, LANES>(src, k);
      my_out[k] = o;
#pragma unroll
      for (int p = 0; p < NP; ++p) pb[p][k] = o;
      ho += r4d_archk_u4(o, 4u * k);
    }
  } else {
    /* Injected: the reduce first, then the push from my out, so dup's and pair's other words are
     * the reduced ones even in place (the fused pass would read a neighbour's input word after its
     * thread stored the reduced one over it). */
    for (int k = w0 + tid; k < my_lim; k += nt) {
      const int4 o = arw_vsum<T, LANES>(src, k);
      my_out[k] = o;
      ho += r4d_archk_u4(o, 4u * k);
    }
    __syncthreads();
    const int4* sp2[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p) sp2[p] = sp[p] + sh_max;
    ij2 = arw_ts_plan(A.iw, b, 1u, w0, cend, w, cn, arw_ctl(A),
                      [&](int k) { return myo[k]; }, [&](int, int k) { return myo[k]; },
                      [&](int p) { return arw_ts_pick(pb, p); },
                      [&](int p) { return arw_ts_pickc(sp2, p); }, lim2);
    for (int k = w0 + tid; k < my_lim; k += nt) {
      const int4 o = myo[k];
#pragma unroll
      for (int p = 0; p < NP; ++p) r4d_arw_injx_put(ij2, p, pb[p], k, o, [&](int kk) { return myo[kk]; });
    }
    r4d_arw_injx_pair(ij2, arw_ts_pick(pb, ij2.cpeer), arw_ctl(A), [&](int k) { return myo[k]; });
  }
  const unsigned int hos[NP] = { ho, ho, ho };
  unsigned int hck2[NP];
  r4d_arw_put_t<WS, NP>(A.peers, b, rank, 1u, (unsigned)parity, hos, v0 + 1u, hck2);
  r4d_arw_publish<WS, NP>(A.peers, b, rank, v0 + 1u, A.drain, A.pub, INJ ? ij2.d.noflag : 0u);

  // ---- check hop 2 and assemble: each peer's region B copy of its shard, straight into the output
  // (a retry rewrites the same words; the output is this kernel's to write)
  r4d_arw_exchange<WS, NP, INJ>(
      A.peers, A.my_flags, b, rank, v0 + 1u, 1u, (unsigned)parity, OP, hck2, A.drain,
      [&](unsigned int mask, unsigned int (&hh)[NP]) {
#pragma unroll
        for (int p = 0; p < NP; ++p) {
          if (!((mask >> p) & 1u)) continue;
          const int4* srcB = sp[p] + sh_max;
          int4* dst = out4 + (size_t)sh.off[A.peers.rank[p]];
          for (int k = w0 + tid; k < lim1[p]; k += nt) {
            const int4 v = srcB[k];
            dst[k] = v;
            hh[p] += r4d_archk_u4(v, 4u * k);
          }
        }
      },
      [&](unsigned int mask, bool stuck) { arw_ts_repush<INJ>(pb, s2, lim2, mask, w0, stuck, ij2); },
      [&]() { arw_ts_late(pb, s2, ij2); },
      [&]() { arw_ts_bad(const_cast<int4*>(arw_ts_pickc(sp, ij2.bpeer < NP ? ij2.bpeer : 0)) + sh_max, ij2); },
      ij2);
#undef ARW_LIM
#undef ARW_LEN
}

// ================================================================ the coded two-shot, RS (W4)
// The coded form's layout, wave codec helpers, injection, push and check try: r4d_ar_wide_coded.h
// (included above; moved there verbatim for r4d_ar_wide_tw.hip).

// The coded two-shot (r4d_ar_twoshot_coded_body<4>) with the push check and the exchange: the same
// sequence take, slots, grid, codes and fold, so the same output bytes. bf16 only; the launcher
// checks that each shard is a whole number of waves' 32 chunks and that a slot holds two regions.
template <bool INJ>
__global__ void arw_coded_rs_kernel(const int4* __restrict__ in4, int4* __restrict__ out4, int n16,
                                    ArwArgs A) {
  constexpr int WS = ARW_WS, NP = ARW_NP;
  constexpr unsigned int OP = R4D_AR_OP_W_CODED;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int rank = A.rank;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(A.seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
  const unsigned int v0 = 2u * s_seq - 1u;
  const int parity = (int)s_slot;
  const int SL = n16 / WS;                       /* every shard: a multiple of 32 chunks */
  int w = (SL + nb - 1) / nb;
  w = (w + 31) & ~31;
  const int w0 = b * w;
  int w1 = w0 + w; if (w1 > SL) w1 = SL;
  const size_t RB = arwc_region_bytes(SL);
  const int lane = tid & 31;

  ArwcRegion dA[NP], dB[NP], sA[NP], sB[NP];
  const int4* s1[NP];   // hop 1's words for peer index p
  int4* mir[NP];        // peer p's decoded hop-1 words, in the mirror of its slot
  int4* asm_[NP];       // peer p's shard of the output (the hop-2 check writes it)
#pragma unroll
  for (int p = 0; p < NP; ++p) {
    const int owner = A.peers.rank[p];
    const int4* mine = A.peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)A.slot_stride16;
    dA[p] = arwc_region(mine, SL);
    dB[p] = arwc_region((const char*)mine + RB, SL);
    const int4* theirs = arw_slot(A, parity, owner);
    sA[p] = arwc_region(theirs, SL);
    sB[p] = arwc_region((const char*)theirs + RB, SL);
    s1[p] = in4 + (size_t)owner * SL;
    mir[p] = const_cast<int4*>(r4d_archk_moved(theirs, A.md));
    asm_[p] = out4 + (size_t)owner * SL;
  }
  int4* const my_out = out4 + (size_t)rank * SL;
  const int4* const myo = my_out;
  const int4* const s2[NP] = { myo, myo, myo };
  unsigned int* const ctl = arw_ctl(A);

  // ---- hop 1: my contribution to each peer-owned shard, coded, into the owner's region A for me
  ArwcInj cj1;
  cj1.x = r4d_arw_injx_none(w0, w1); cj1.k0 = -1; cj1.off = 0;
  if constexpr (INJ) cj1 = arwc_inj_bcast(A.iw, b, 0u, w0, w1, ctl);
  unsigned int h1[NP] = { 0u, 0u, 0u };
  arwc_push<INJ, true>(dA, s1, (1u << NP) - 1u, w0, w1, true, true, cj1, ctl, h1);
  unsigned int hck1[NP];
  r4d_arw_put_t<WS, NP>(A.peers, b, rank, 0u, (unsigned)parity, h1, v0, hck1);
  r4d_arw_publish<WS, NP>(A.peers, b, rank, v0, A.drain, A.pub, INJ ? cj1.x.d.noflag : 0u);
  unsigned int hz[NP] = { 0u, 0u, 0u };
  r4d_arw_exchange<WS, NP, INJ>(
      A.peers, A.my_flags, b, rank, v0, 0u, (unsigned)parity, OP, hck1, A.drain,
      [&](unsigned int mask, unsigned int (&hh)[NP]) { arwc_rehash(sA, mir, mask, w0, w1, hh); },
      [&](unsigned int mask, bool stuck) { arwc_push<INJ, false>(dA, s1, mask, w0, w1, stuck, false, cj1, ctl, hz); },
      [&]() { arwc_push<INJ, false>(dA, s1, cj1.x.pmask, w0, w1, false, false, cj1, ctl, hz); },
      [&]() { arwc_inj_bad(sA, cj1.x.bpeer, cj1.k0); },
      cj1.x);

  // ---- reduce my shard (fp32, ascending rank, my own contribution raw from the input; the peers'
  // decoded, from the mirror), code it, and push it to every peer's region B
  const int4* own = in4 + (size_t)rank * SL;
  unsigned int ho = 0u;
  ArwcInj cj2;
  cj2.x = r4d_arw_injx_none(w0, w1); cj2.k0 = -1; cj2.off = 0;
  auto fold = [&](int k, bool valid) -> int4 {
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
#pragma unroll
    for (int r = 0; r < WS; ++r) {
      int4 v = make_int4(0, 0, 0, 0);
      if (valid) {
        if (r == rank) v = own[k];
        else {
          const int p = r < rank ? r : r - 1;
          v = arw_ts_pick(mir, p)[k];
        }
      }
      const __hip_bfloat16* pv = reinterpret_cast<const __hip_bfloat16*>(&v);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += (float)pv[j];
    }
    int4 o;
    __hip_bfloat16* po = reinterpret_cast<__hip_bfloat16*>(&o);
#pragma unroll
    for (int j = 0; j < 8; ++j) po[j] = (__hip_bfloat16)acc[j];
    return o;
  };
  if constexpr (!INJ) {
    /* The default kernel's pass: the reduce, coded once, to all three regions. */
    for (int k = w0 + tid; k - lane < w1; k += nt) {
      const bool valid = k < w1;
      const int k0 = k - lane;
      const int4 o = fold(k, valid);
      if (valid) { my_out[k] = o; ho += r4d_archk_u4(o, 4u * k); }
      const R4dArcEnc e = r4d_arc_enc(o, valid);
#pragma unroll
      for (int p = 0; p < NP; ++p) arwc_put<false>(dB[p], k0, e, o, valid, -1);
    }
  } else {
    /* Injected: the reduce first, then the push from my out (in place, a word's own contribution
     * is overwritten by its reduced value; the push reads only the latter). */
    for (int k = w0 + tid; k - lane < w1; k += nt) {
      const bool valid = k < w1;
      const int4 o = fold(k, valid);
      if (valid) { my_out[k] = o; ho += r4d_archk_u4(o, 4u * k); }
    }
    __syncthreads();
    cj2 = arwc_inj_bcast(A.iw, b, 1u, w0, w1, ctl);
    arwc_push<INJ, false>(dB, s2, (1u << NP) - 1u, w0, w1, true, true, cj2, ctl, hz);
  }
  const unsigned int hos[NP] = { ho, ho, ho };
  unsigned int hck2[NP];
  r4d_arw_put_t<WS, NP>(A.peers, b, rank, 1u, (unsigned)parity, hos, v0 + 1u, hck2);
  r4d_arw_publish<WS, NP>(A.peers, b, rank, v0 + 1u, A.drain, A.pub, INJ ? cj2.x.d.noflag : 0u);

  // ---- check hop 2 and assemble: each peer's region B, decoded straight into its shard of the
  // output (a retry rewrites the same words)
  r4d_arw_exchange<WS, NP, INJ>(
      A.peers, A.my_flags, b, rank, v0 + 1u, 1u, (unsigned)parity, OP, hck2, A.drain,
      [&](unsigned int mask, unsigned int (&hh)[NP]) { arwc_rehash(sB, asm_, mask, w0, w1, hh); },
      [&](unsigned int mask, bool stuck) { arwc_push<INJ, false>(dB, s2, mask, w0, w1, stuck, false, cj2, ctl, hz); },
      [&]() { arwc_push<INJ, false>(dB, s2, cj2.x.pmask, w0, w1, false, false, cj2, ctl, hz); },
      [&]() { arwc_inj_bad(sB, cj2.x.bpeer, cj2.k0); },
      cj2.x);
}

// ================================================================ launches
void arw_rs_twoshot(int dtype, int nb, int nt, hipStream_t st, const int4* in4, int4* out4,
                    const int* off, int sh_max, const ArwArgs& A) {
  ArwTs sh{};
  for (int s = 0; s <= ARW_WS; ++s) sh.off[s] = off[s];
  sh.sh_max = sh_max;
#define ARW_L(T, LN)                                                                              \
  do {                                                                                            \
    if (A.iw) arw_twoshot_rs_kernel<T, LN, true><<<nb, nt, 0, st>>>(in4, out4, sh, A);            \
    else arw_twoshot_rs_kernel<T, LN, false><<<nb, nt, 0, st>>>(in4, out4, sh, A);                \
  } while (0)
  if (dtype == 0) ARW_L(__hip_bfloat16, 8);
  else if (dtype == 1) ARW_L(__half, 8);
  else ARW_L(float, 4);
#undef ARW_L
}

size_t arw_coded_region_bytes(long long SL) { return arwc_region_bytes(SL); }

void arw_rs_coded(int nb, int nt, hipStream_t st, const int4* in4, int4* out4, int n16,
                  const ArwArgs& A) {
  if (A.iw) arw_coded_rs_kernel<true><<<nb, nt, 0, st>>>(in4, out4, n16, A);
  else arw_coded_rs_kernel<false><<<nb, nt, 0, st>>>(in4, out4, n16, A);
}
