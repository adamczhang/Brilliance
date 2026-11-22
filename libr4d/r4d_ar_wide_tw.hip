// r4d_ar_wide_tw.hip: Brilliance (tp4-link-weight), the four-rank coded two-shot with the resend
// over a link-weighted shard table (r4d_ar4_weights.h). r4d_ar_wide_host.hip's coded launcher picks
// it only when the instance's shards are weighted (BRILLIANCE_AR4_SHARD_WEIGHTS, or the start-time
// link probe) and the message is large enough; even shards keep r4d_ar_wide_ts.hip's kernel, whose
// code this unit leaves alone (a unit of its own, as W2-W4's are).
//
// It is arw_coded_rs_kernel (r4d_ar_wide_ts.hip) with three changes and nothing else:
//   - each shard s is the words [off[s], off[s + 1]) of the message (whole waves of 32), not
//     s * SL onwards; a shard may be empty (a rank weighted 0 reduces nothing and pushes nothing in
//     hop 2, but still sends its contribution in hop 1 and receives every shard in hop 2);
//   - the regions in a slot are sized for the longest shard (sh_max): region A at the slot's start,
//     region B at arwc_region_bytes(sh_max), and the blocks split sh_max in whole waves;
//   - every loop over a peer's words stops at that peer's own shard end within the block's span
//     (hop 1's push and hop 2's check: the owner's shard; hop 1's check, the reduce and hop 2's push:
//     this rank's), and the injection picks its wave within the range the damaged push covers.
// The fold (fp32, ascending rank, own contribution raw from the input, the peers' decoded from the
// mirror), the codec, the check words, the exchange and its phases, handshake values and counters
// (R4D_AR_OP_W_CODED) are the even kernel's. Which rank owns a word changes only the link it
// crosses, so every table gives the even kernel's output bits.
#include "r4d.h"
#include "r4d_ar_wide_host.h"
#include "r4d_ar_arc.h"
#include "r4d_ar_wide_coded.h"
#include "r4d_ar4_weights.h"

static_assert(arwc_region_bytes(1) == r4d_arw_coded_region_bytes(1) &&
              arwc_region_bytes(4096) == r4d_arw_coded_region_bytes(4096) &&
              arwc_region_bytes(655392) == r4d_arw_coded_region_bytes(655392),
              "r4d_ar4_weights.h restates the coded region's size");

// The table by value (kernel-argument bytes).
struct ArwcTab { int off[ARW_WS + 1]; int sh_max; };

// arwc_push with an end of its own for each peer: words [w0, lim[p]) of s[p] coded into d[p].
template <bool INJ, bool HASH>
__device__ __forceinline__ void arwcw_push(const ArwcRegion (&d)[ARW_NP], const int4* const (&s)[ARW_NP],
                                           const int (&lim)[ARW_NP], unsigned int mask, int w0, bool dmg,
                                           bool count, const ArwcInj& cj, unsigned int* ctl,
                                           unsigned int (&h)[ARW_NP]) {
  const int tid = threadIdx.x, nt = blockDim.x, lane = tid & 31;
#pragma unroll
  for (int p = 0; p < ARW_NP; ++p) {
    if (!((mask >> p) & 1u)) continue;
    const int4* const sp = s[p];
    const int w1 = lim[p];
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
// arwc_rehash with an end of its own for each peer.
__device__ __forceinline__ void arwcw_rehash(const ArwcRegion (&s)[ARW_NP], int4* const (&out)[ARW_NP],
                                             const int (&lim)[ARW_NP], unsigned int mask, int w0,
                                             unsigned int (&h)[ARW_NP]) {
  const int tid = threadIdx.x, nt = blockDim.x, lane = tid & 31;
#pragma unroll
  for (int p = 0; p < ARW_NP; ++p) {
    if (!((mask >> p) & 1u)) continue;
    const int w1 = lim[p];
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
__device__ __forceinline__ int arwcw_pickl(const int (&l)[ARW_NP], int p) {
  return p == 0 ? l[0] : p == 1 ? l[1] : l[2];
}

// arwc_inj_bcast with the wave picked inside the range the injection acts on: for the push modes
// the damaged push's (`push[peer]`, ending at that peer's limit), for ctl-rq the words this rank
// received from the peer (`recv[peer]`). A range of no whole wave leaves the injection with nothing
// to damage, as the even kernel's empty block does.
__device__ __forceinline__ ArwcInj arwcw_inj_bcast(unsigned int iw, int b, unsigned int phase, int w0,
                                                   const int (&push)[ARW_NP], const int (&recv)[ARW_NP],
                                                   unsigned int* ctl) {
  __shared__ ArwcInj s_cjw;
  if (threadIdx.x == 0) {
    ArwcInj r;
    r.x = r4d_arw_injx_none(w0, w0);
    r.k0 = -1; r.off = 0;
    const unsigned int mode = iw & 15u;
    const int bt = (int)((iw >> 4) & 63u), off = (int)((iw >> 14) & 255u);
    int peer = (int)((iw >> 10) & 7u);
    if (peer >= ARW_NP) peer = 0;
    const int w1 = mode == R4D_AR_INJ_RQ ? arwcw_pickl(recv, peer) : arwcw_pickl(push, peer);
    r.x = r4d_arw_injx_none(w0, w1 > w0 ? w1 : w0);
    const int ng = w1 > w0 ? (w1 - w0) / 32 : 0;
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
    s_cjw = r;
  }
  __syncthreads();
  const ArwcInj r = s_cjw;
  __syncthreads();
  return r;
}

template <bool INJ>
__global__ void arw_coded_w_kernel(const int4* __restrict__ in4, int4* __restrict__ out4, ArwcTab t,
                                   ArwArgs A) {
  constexpr int WS = ARW_WS, NP = ARW_NP;
  constexpr unsigned int OP = R4D_AR_OP_W_CODED;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int rank = A.rank;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(A.seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
  const unsigned int v0 = 2u * s_seq - 1u;
  const int parity = (int)s_slot;
  const int SL = t.sh_max;                      /* the regions' capacity; every shard whole waves */
  int w0, w1;
  r4d_arw_span(SL, nb, b, 32, &w0, &w1);
  const size_t RB = arwc_region_bytes(SL);
  const int lane = tid & 31;
  const int my_lim = r4d_arw_lim(t.off, w1, rank);

  ArwcRegion dA[NP], dB[NP], sA[NP], sB[NP];
  const int4* s1[NP];   // hop 1's words for peer index p: my input at its owner's shard
  int4* mir[NP];        // peer p's decoded hop-1 words, in the mirror of its slot
  int4* asm_[NP];       // peer p's shard of the output (the hop-2 check writes it)
  int lim_o[NP];        // peer p's shard end within this block's span
  int lim_m[NP];        // this rank's shard end, for every peer
#pragma unroll
  for (int p = 0; p < NP; ++p) {
    const int owner = A.peers.rank[p];
    const int4* mine = A.peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)A.slot_stride16;
    dA[p] = arwc_region(mine, SL);
    dB[p] = arwc_region((const char*)mine + RB, SL);
    const int4* theirs = arw_slot(A, parity, owner);
    sA[p] = arwc_region(theirs, SL);
    sB[p] = arwc_region((const char*)theirs + RB, SL);
    s1[p] = in4 + (size_t)t.off[owner];
    mir[p] = const_cast<int4*>(r4d_archk_moved(theirs, A.md));
    asm_[p] = out4 + (size_t)t.off[owner];
    lim_o[p] = r4d_arw_lim(t.off, w1, owner);
    lim_m[p] = my_lim;
  }
  int4* const my_out = out4 + (size_t)t.off[rank];
  const int4* const myo = my_out;
  const int4* const s2[NP] = { myo, myo, myo };
  unsigned int* const ctl = arw_ctl(A);

  // ---- hop 1: my contribution to each peer-owned shard, coded, into the owner's region A for me
  ArwcInj cj1;
  cj1.x = r4d_arw_injx_none(w0, w1); cj1.k0 = -1; cj1.off = 0;
  if constexpr (INJ) cj1 = arwcw_inj_bcast(A.iw, b, 0u, w0, lim_o, lim_m, ctl);
  unsigned int h1[NP] = { 0u, 0u, 0u };
  arwcw_push<INJ, true>(dA, s1, lim_o, (1u << NP) - 1u, w0, true, true, cj1, ctl, h1);
  unsigned int hck1[NP];
  r4d_arw_put_t<WS, NP>(A.peers, b, rank, 0u, (unsigned)parity, h1, v0, hck1);
  r4d_arw_publish<WS, NP>(A.peers, b, rank, v0, A.drain, A.pub, INJ ? cj1.x.d.noflag : 0u);
  unsigned int hz[NP] = { 0u, 0u, 0u };
  r4d_arw_exchange<WS, NP, INJ>(
      A.peers, A.my_flags, b, rank, v0, 0u, (unsigned)parity, OP, hck1, A.drain,
      [&](unsigned int mask, unsigned int (&hh)[NP]) { arwcw_rehash(sA, mir, lim_m, mask, w0, hh); },
      [&](unsigned int mask, bool stuck) { arwcw_push<INJ, false>(dA, s1, lim_o, mask, w0, stuck, false, cj1, ctl, hz); },
      [&]() { arwcw_push<INJ, false>(dA, s1, lim_o, cj1.x.pmask, w0, false, false, cj1, ctl, hz); },
      [&]() { arwc_inj_bad(sA, cj1.x.bpeer, cj1.k0); },
      cj1.x);

  // ---- reduce my shard (fp32, ascending rank, my own contribution raw from the input; the peers'
  // decoded, from the mirror), code it, and push it to every peer's region B
  const int4* own = in4 + (size_t)t.off[rank];
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
    /* The even kernel's pass: the reduce, coded once, to all three regions. */
    for (int k = w0 + tid; k - lane < my_lim; k += nt) {
      const bool valid = k < my_lim;
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
    for (int k = w0 + tid; k - lane < my_lim; k += nt) {
      const bool valid = k < my_lim;
      const int4 o = fold(k, valid);
      if (valid) { my_out[k] = o; ho += r4d_archk_u4(o, 4u * k); }
    }
    __syncthreads();
    cj2 = arwcw_inj_bcast(A.iw, b, 1u, w0, lim_m, lim_o, ctl);
    arwcw_push<INJ, false>(dB, s2, lim_m, (1u << NP) - 1u, w0, true, true, cj2, ctl, hz);
  }
  const unsigned int hos[NP] = { ho, ho, ho };
  unsigned int hck2[NP];
  r4d_arw_put_t<WS, NP>(A.peers, b, rank, 1u, (unsigned)parity, hos, v0 + 1u, hck2);
  r4d_arw_publish<WS, NP>(A.peers, b, rank, v0 + 1u, A.drain, A.pub, INJ ? cj2.x.d.noflag : 0u);

  // ---- check hop 2 and assemble: each peer's region B, decoded straight into its shard of the
  // output (a retry rewrites the same words)
  r4d_arw_exchange<WS, NP, INJ>(
      A.peers, A.my_flags, b, rank, v0 + 1u, 1u, (unsigned)parity, OP, hck2, A.drain,
      [&](unsigned int mask, unsigned int (&hh)[NP]) { arwcw_rehash(sB, asm_, lim_o, mask, w0, hh); },
      [&](unsigned int mask, bool stuck) { arwcw_push<INJ, false>(dB, s2, lim_m, mask, w0, stuck, false, cj2, ctl, hz); },
      [&]() { arwcw_push<INJ, false>(dB, s2, lim_m, cj2.x.pmask, w0, false, false, cj2, ctl, hz); },
      [&]() { arwc_inj_bad(sB, cj2.x.bpeer, cj2.k0); },
      cj2.x);
}

// ================================================================ launch
void arw_rs_coded_w(int nb, int nt, hipStream_t st, const int4* in4, int4* out4, const int* off,
                    int sh_max, const ArwArgs& A) {
  ArwcTab t{};
  for (int s = 0; s <= ARW_WS; ++s) t.off[s] = off[s];
  t.sh_max = sh_max;
  if (A.iw) arw_coded_w_kernel<true><<<nb, nt, 0, st>>>(in4, out4, t, A);
  else arw_coded_w_kernel<false><<<nb, nt, 0, st>>>(in4, out4, t, A);
}
