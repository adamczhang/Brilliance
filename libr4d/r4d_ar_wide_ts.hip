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
__device__ __forceinline__ int4* arw_ts_pick(int4* const (&ps)[ARW_NP], int p) {
  return p == 0 ? ps[0] : p == 1 ? ps[1] : ps[2];
}
__device__ __forceinline__ const int4* arw_ts_pickc(const int4* const (&ps)[ARW_NP], int p) {
  return p == 0 ? ps[0] : p == 1 ? ps[1] : ps[2];
}

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
// The coded form's layout and wave codec, as r4d_ar_twoshot_Nrank_exact.hip lays them out (that
// file keeps its own copies for the default kernels): a region is [3 dwords a chunk][a base byte a
// group][the shadow, 16 bytes a chunk]; A at the slot's start, B at RB.
struct ArwcRegion { unsigned* packed; unsigned* bases; int4* shadow; };
__host__ __device__ constexpr size_t arwc_shadow_off(long long SL) {
  return ((size_t)12 * SL + (size_t)SL / 8 + 15) & ~(size_t)15;
}
__host__ __device__ constexpr size_t arwc_region_bytes(long long SL) {
  return (arwc_shadow_off(SL) + (size_t)16 * SL + 255) & ~(size_t)255;
}
__device__ __forceinline__ ArwcRegion arwc_region(const void* p, int SL) {
  char* b = (char*)p;
  return ArwcRegion{(unsigned*)b, (unsigned*)(b + (size_t)12 * SL), (int4*)(b + arwc_shadow_off(SL))};
}
// ctl-rq's overwrite (injected launches only): the base bytes of the wave at k0 in peer index p's
// region of this rank's own slot, xored. Selects, not an index, so no scratch.
__device__ __forceinline__ void arwc_inj_bad(const ArwcRegion (&s)[ARW_NP], int p, int k0) {
  if (threadIdx.x != 0 || k0 < 0) return;
  unsigned* const bw = (p == 1 ? s[1].bases : p == 2 ? s[2].bases : s[0].bases) + k0 / 32;
  *bw ^= 0x5A5A5A5Au;
}

// A wave's coded chunks [k0, k0 + 32) into one region, every store contiguous (r4d_arc_put). `skip`
// (injected launches only, -1 for none): the one 64-byte line not stored: 0..5 the packed dwords
// [16 skip, +16) of the wave's 96, 6..13 the shadow chunks [4 (skip - 6), +4).
template <bool INJ>
__device__ __forceinline__ void arwc_put(const ArwcRegion& dst, int k0, const R4dArcEnc& e, int4 raw,
                                         bool valid, int skip) {
  const int lane = (int)(threadIdx.x & 31u);
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int p = 32 * j + lane, sl = p / 3, wi = p - sl * 3;
    const unsigned a0 = r4d_arc_shfl(e.w[0], sl), a1 = r4d_arc_shfl(e.w[1], sl);
    const unsigned a2 = r4d_arc_shfl(e.w[2], sl), vs = r4d_arc_shfl(valid ? 1u : 0u, sl);
    const unsigned val = wi == 0 ? a0 : wi == 1 ? a1 : a2;
    if (vs && (!INJ || p / 16 != skip)) dst.packed[(size_t)k0 * 3 + p] = val;
  }
  const unsigned b0 = r4d_arc_shfl(e.base, 0), b1 = r4d_arc_shfl(e.base, 8);
  const unsigned b2 = r4d_arc_shfl(e.base, 16), b3 = r4d_arc_shfl(e.base, 24);
  if (lane == 0 && valid) dst.bases[k0 / 32] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
  if (valid && e.base == 0xFFu && (!INJ || skip < 6 || lane / 4 != skip - 6)) dst.shadow[k0 + lane] = raw;
}
// A wave's chunks [k0, k0 + 32) back from a region; every lane calls (r4d_arc_get).
__device__ __forceinline__ int4 arwc_get(const ArwcRegion& src, int k0, bool valid) {
  const int lane = (int)(threadIdx.x & 31u);
  unsigned ld[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) ld[j] = valid ? src.packed[(size_t)k0 * 3 + 32 * j + lane] : 0u;
  unsigned w[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int q = 3 * lane + i, sl = q & 31, j = q >> 5;
    const unsigned a0 = r4d_arc_shfl(ld[0], sl), a1 = r4d_arc_shfl(ld[1], sl);
    const unsigned a2 = r4d_arc_shfl(ld[2], sl);
    w[i] = j == 0 ? a0 : j == 1 ? a1 : a2;
  }
  const unsigned base = ((valid ? src.bases[k0 / 32] : 0u) >> ((lane >> 3) * 8)) & 0xffu;
  if (base == 0xFFu) return valid ? src.shadow[k0 + lane] : make_int4(0, 0, 0, 0);
  return r4d_arc_dec(w, base);
}

// The injected wave's line (injected launches only; every lane of the wave calls): of the wave's 14
// lines, from `off` on, the first whose loss changes a decoded word: a packed line holding a dword of
// a coded group (base not 0xFF) that the slot does not already hold, or a shadow line holding a
// chunk of a raw group that differs. -1 for none (nothing to damage: the all-zero trap, design 5.2).
// Reads the destination over the link.
__device__ __forceinline__ int arwc_inj_pick(const ArwcRegion& dst, int k0, const R4dArcEnc& e,
                                             int4 raw, bool valid, int off) {
  const int lane = (int)(threadIdx.x & 31u);
  const unsigned g0 = r4d_arc_shfl(e.base, 0), g1 = r4d_arc_shfl(e.base, 8);
  const unsigned g2 = r4d_arc_shfl(e.base, 16), g3 = r4d_arc_shfl(e.base, 24);
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  unsigned mp[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int p = 32 * j + lane, sl = p / 3, wi = p - sl * 3;
    const unsigned a0 = r4d_arc_shfl(e.w[0], sl), a1 = r4d_arc_shfl(e.w[1], sl);
    const unsigned a2 = r4d_arc_shfl(e.w[2], sl), vs = r4d_arc_shfl(valid ? 1u : 0u, sl);
    const unsigned val = wi == 0 ? a0 : wi == 1 ? a1 : a2;
    const int g = sl >> 3;
    const unsigned gb = g == 0 ? g0 : g == 1 ? g1 : g == 2 ? g2 : g3;
    bool d = false;
    if (vs && gb != 0xFFu) d = r4d_ar_ld(dst.packed + (size_t)k0 * 3 + p) != val;
    mp[j] = __builtin_amdgcn_ballot_w32(d);
  }
  bool ds = false;
  if (valid && e.base == 0xFFu) {
    const unsigned int* q = reinterpret_cast<const unsigned int*>(dst.shadow + k0 + lane);
    ds = (int)r4d_ar_ld(q) != raw.x || (int)r4d_ar_ld(q + 1) != raw.y ||
         (int)r4d_ar_ld(q + 2) != raw.z || (int)r4d_ar_ld(q + 3) != raw.w;
  }
  const unsigned ms = __builtin_amdgcn_ballot_w32(ds);
  for (int c = 0; c < 14; ++c) {
    const int i = (off + c) % 14;
    const unsigned mj = (i >> 1) == 0 ? mp[0] : (i >> 1) == 1 ? mp[1] : mp[2];   /* selects: no scratch */
    const unsigned m = i < 6 ? (mj >> (16 * (i & 1))) & 0xFFFFu : (ms >> (4 * (i - 6))) & 0xFu;
    if (m) return i;
  }
  return -1;
}

// The coded form's injection (design 5.2, at the codec's granularity): the planner's block, one wave
// of it (`k0`, from the line offset) and one 64-byte line of that wave's codes, the first from `off`
// whose loss changes a decoded word (arwc_inj_pick). lose, dup, pair and stuck lose that line (dup
// and pair act as lose: a coded line has no neighbour of the same meaning to copy), multi in the
// pushes to two peers; late withholds it and re-pushes the block's codes at R4D_ARX_T_LATE; noflag,
// ctl-ack and ctl-rq are W2's (ctl-rq: one dword of this rank's own region from the peer, a wave's
// base bytes, xored). `x` is the exchange's view.
struct ArwcInj {
  R4dArwInjX x;
  int k0, off;
};
__device__ __forceinline__ ArwcInj arwc_inj_bcast(unsigned int iw, int b, unsigned int phase, int w0,
                                                  int w1, unsigned int* ctl) {
  __shared__ ArwcInj s_cj;
  if (threadIdx.x == 0) {
    ArwcInj r;
    r.x = r4d_arw_injx_none(w0, w1);
    r.k0 = -1; r.off = 0;
    const unsigned int mode = iw & 15u;
    const int bt = (int)((iw >> 4) & 63u), off = (int)((iw >> 14) & 255u);
    int peer = (int)((iw >> 10) & 7u);
    if (peer >= ARW_NP) peer = 0;
    const int ng = (w1 - w0) / 32;
    if (mode && ((iw >> 13) & 1u) == phase && bt == b && ng > 0) {
      const int k0 = w0 + 32 * (off % ng);
      switch (mode) {
      case R4D_AR_INJ_LOSE: case R4D_AR_INJ_DUP: case R4D_AR_INJ_PAIR: case R4D_AR_INJ_STUCK:
      case R4D_AR_INJ_LATE: case R4D_AR_INJ_MULTI:
        r.k0 = k0; r.off = off;
        r.x.pmask = 1u << peer;
        if (mode == R4D_AR_INJ_MULTI) r.x.pmask |= 1u << ((peer + 1) % ARW_NP);
        r.x.stuck = mode == R4D_AR_INJ_STUCK;
        r.x.late = mode == R4D_AR_INJ_LATE;
        break;
      case R4D_AR_INJ_NOFLAG:
        r.x.d.noflag = 1u << peer;
        r4d_arw_inj_count(ctl);
        break;
      case R4D_AR_INJ_ACK:
        r.x.skip_ack = 1u << peer;
        break;
      case R4D_AR_INJ_RQ:
        r.x.skip_rq = 1u << peer;
        r.x.bpeer = peer; r.x.blo = 0; r.x.bhi = 1;   /* due: the exchange calls bad() */
        r.k0 = k0;
        r4d_arw_inj_count(ctl);   /* the xor always changes the base bytes */
        break;
      default:
        break;
      }
    }
    s_cj = r;
  }
  __syncthreads();
  const ArwcInj r = s_cj;
  __syncthreads();
  return r;
}

// One hop's coded push of words [w0, w1) to the peer indices in `mask`: s[p] the words, d[p] the
// region. `dmg`: the injection's line is left out of the damaged pushes (the first push and stuck's
// resends); `count`: an effective injection is counted (the first push only). h[p] gains the hash of
// the words pushed to p (the first push only: `hash`).
template <bool INJ, bool HASH>
__device__ __forceinline__ void arwc_push(const ArwcRegion (&d)[ARW_NP], const int4* const (&s)[ARW_NP],
                                          unsigned int mask, int w0, int w1, bool dmg, bool count,
                                          const ArwcInj& cj, unsigned int* ctl,
                                          unsigned int (&h)[ARW_NP]) {
  const int tid = threadIdx.x, nt = blockDim.x, lane = tid & 31;
#pragma unroll
  for (int p = 0; p < ARW_NP; ++p) {
    if (!((mask >> p) & 1u)) continue;
    const int4* const sp = s[p];
    unsigned int x = 0u;
    for (int k = w0 + tid; k - lane < w1; k += nt) {
      const bool valid = k < w1;
      const int k0 = k - lane;
      const int4 v = valid ? sp[k] : make_int4(0, 0, 0, 0);
      const R4dArcEnc e = r4d_arc_enc(v, valid);
      int skip = -1;
      if (INJ && dmg && k0 == cj.k0 && ((cj.x.pmask >> p) & 1u)) {
        skip = arwc_inj_pick(d[p], k0, e, v, valid, cj.off);
        if (count && skip >= 0 && lane == 0) r4d_arw_inj_count(ctl);
      }
      arwc_put<INJ>(d[p], k0, e, v, valid, skip);
      if (HASH && valid) x += r4d_archk_u4(v, 4u * k);
    }
    if (HASH) h[p] += x;
  }
}
// The check try over the peer indices in `mask`: each peer's region decoded, every word kept at
// out[p][k] (the mirror, or the output for hop 2), and hashed.
__device__ __forceinline__ void arwc_rehash(const ArwcRegion (&s)[ARW_NP], int4* const (&out)[ARW_NP],
                                            unsigned int mask, int w0, int w1,
                                            unsigned int (&h)[ARW_NP]) {
  const int tid = threadIdx.x, nt = blockDim.x, lane = tid & 31;
#pragma unroll
  for (int p = 0; p < ARW_NP; ++p) {
    if (!((mask >> p) & 1u)) continue;
    for (int k = w0 + tid; k - lane < w1; k += nt) {
      const bool valid = k < w1;
      const int4 v = arwc_get(s[p], k - lane, valid);
      if (valid) {
        out[p][k] = v;
        h[p] += r4d_archk_u4(v, 4u * k);
      }
    }
  }
}

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
