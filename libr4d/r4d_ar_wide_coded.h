#pragma once
// r4d_ar_wide_coded.h: Brilliance, the four-rank coded two-shot's device helpers (the region layout,
// the wave codec's puts and gets, the injection at the codec's granularity, the push and the check
// try), moved verbatim out of r4d_ar_wide_ts.hip (tp4-link-weight) so that the link-weighted coded
// kernel (r4d_ar_wide_tw.hip) uses the same code as the even one. r4d_ar_wide_ts.hip includes this
// where the code stood, so its kernels compile from the same text.
#include "r4d_ar_wide_host.h"
#include "r4d_ar_arc.h"

__device__ __forceinline__ int4* arw_ts_pick(int4* const (&ps)[ARW_NP], int p) {
  return p == 0 ? ps[0] : p == 1 ? ps[1] : ps[2];
}
__device__ __forceinline__ const int4* arw_ts_pickc(const int4* const (&ps)[ARW_NP], int p) {
  return p == 0 ? ps[0] : p == 1 ? ps[1] : ps[2];
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
