#pragma once
// r4d_ar_arc2.h: the two-rank coded push's slot layout over r4d_ar_arc.h's chunk codec: the fused
// writes' (r4d_ar_oneshot_2rank_exact.hip) and the fused all-reduce norm's (r4d_fused_quant_fp8.hip).
#include <hip/hip_runtime.h>
#include "r4d_ar_seq.h"   // the push check (r4d_archk_mix)
#include "r4d_ar_arc.h"   // the chunk codec

// ---- The coded form of the two fused writes: the push carries ~12.1 bits a bf16 value instead of
// 16, losslessly (the four-rank coded two-shot's chunk codec, r4d_ar_arc.h), for the messages whose
// time is the link's: a prefill chunk (2048 rows of 2560 are 10.5 MB each way a reduction) and a
// many-stream decode or verify step. The peer decodes the same bf16 words it would have been sent
// raw and folds them in the same arithmetic, so y and h are the plain kernels' byte for byte.
//
// A slot holds one coded message of n16 chunks (16-byte words), n16 a multiple of 32:
//
//   packed  u32[3 * n16]   at 0            three dwords a chunk
//   bases   u32[n16 / 32]  at 12 * n16     a base-exponent byte for each group of 8 chunks
//   spill   u32[n16]       behind, 16-byte aligned: the fourth dword of a raw group's chunks
//
// A wave codes 32 consecutive chunks and redistributes the packed dwords among its lanes, so that
// each of its three stores is 128 contiguous bytes and the P2P writes stay coalesced, as in the
// four-rank form. A group that does not code (a subnormal, an inf or NaN, an exponent range past
// 14: base byte 0xFF) travels raw in the same place: the first three dwords of each of its chunks
// in the chunk's packed cell and the fourth in the spill, so a slot is 16.125 bytes a chunk
// whatever the data (r4d_ar_entry.hip sizes a two-rank bf16 slot for it) and a message of raw
// groups only costs a spill store a chunk.
//
// The push check (r4d_ar_seq.h) runs over the dwords as they cross the link: the writer sums
// mix(dword, position) over what it stores, the reader over what it is about to decode, each
// region in its own index space. A stale base byte makes the reader sum a spill dword the writer
// did not store, or miss one it did, and the sums disagree as for any other stale dword.
//
// P2P instances only, like the register form: a host mailbox keeps the plain kernels and their
// mirror.
struct R4dArc2 { unsigned* packed; unsigned* bases; unsigned* spill; };
__host__ __device__ constexpr size_t r4d_arc2_spill_off(long long n16) {
  return ((size_t)12 * n16 + (size_t)n16 / 8 + 15) & ~(size_t)15;
}
// Bytes of a coded message of n16 chunks, n16 a multiple of 32.
__host__ __device__ constexpr size_t r4d_arc2_bytes(long long n16) {
  return r4d_arc2_spill_off(n16) + (size_t)4 * n16;
}
__device__ __forceinline__ R4dArc2 r4d_arc2_at(const void* p, int n16) {
  char* b = (char*)p;
  return R4dArc2{(unsigned*)b, (unsigned*)(b + (size_t)12 * n16),
                 (unsigned*)(b + r4d_arc2_spill_off(n16))};
}
#define R4D_ARC2_BASES 0x40000000u   // the push check's index spaces of the bases and the spill,
#define R4D_ARC2_SPILL 0x80000000u   // apart from the packed dwords' (under 2^30)

// Code this lane's chunk `v` of the wave's chunks [k0, k0 + 32) into the peer's slot; every lane of
// the wave calls. Returns the lane's share of the push check.
__device__ __forceinline__ unsigned r4d_arc2_put(const R4dArc2& d, int k0, int4 v) {
  const int lane = (int)(threadIdx.x & 31u);
  R4dArcEnc e = r4d_arc_enc(v, true);
  const bool raw = e.base == 0xFFu;      /* uniform over the lane's group of 8 */
  if (raw) { e.w[0] = (unsigned)v.x; e.w[1] = (unsigned)v.y; e.w[2] = (unsigned)v.z; }
  unsigned h = 0u;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int p = 32 * j + lane, sl = p / 3, wi = p - sl * 3;
    const unsigned a0 = r4d_arc_shfl(e.w[0], sl), a1 = r4d_arc_shfl(e.w[1], sl);
    const unsigned a2 = r4d_arc_shfl(e.w[2], sl);
    const unsigned val = wi == 0 ? a0 : wi == 1 ? a1 : a2;
    const unsigned at = 3u * (unsigned)k0 + (unsigned)p;
    d.packed[at] = val;
    h += r4d_archk_mix(val, at);
  }
  const unsigned b0 = r4d_arc_shfl(e.base, 0), b1 = r4d_arc_shfl(e.base, 8);
  const unsigned b2 = r4d_arc_shfl(e.base, 16), b3 = r4d_arc_shfl(e.base, 24);
  if (lane == 0) {
    const unsigned bw = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    d.bases[k0 >> 5] = bw;
    h += r4d_archk_mix(bw, R4D_ARC2_BASES + (unsigned)(k0 >> 5));
  }
  if (raw) {
    d.spill[k0 + lane] = (unsigned)v.w;
    h += r4d_archk_mix((unsigned)v.w, R4D_ARC2_SPILL + (unsigned)(k0 + lane));
  }
  return h;
}
// The reader's share of the push check over the wave's chunks [k0, k0 + 32): the dwords
// r4d_arc2_get is about to load, in the writer's index spaces.
__device__ __forceinline__ unsigned r4d_arc2_sum(const R4dArc2& s, int k0) {
  const int lane = (int)(threadIdx.x & 31u);
  const unsigned bw = s.bases[k0 >> 5];
  unsigned h = lane == 0 ? r4d_archk_mix(bw, R4D_ARC2_BASES + (unsigned)(k0 >> 5)) : 0u;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const unsigned at = 3u * (unsigned)k0 + (unsigned)(32 * j + lane);
    h += r4d_archk_mix(s.packed[at], at);
  }
  if (((bw >> ((lane >> 3) * 8)) & 0xffu) == 0xFFu)
    h += r4d_archk_mix(s.spill[k0 + lane], R4D_ARC2_SPILL + (unsigned)(k0 + lane));
  return h;
}
// This lane's chunk of the wave's chunks [k0, k0 + 32) back from a slot; every lane calls.
__device__ __forceinline__ int4 r4d_arc2_get(const R4dArc2& s, int k0) {
  const int lane = (int)(threadIdx.x & 31u);
  unsigned ld[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) ld[j] = s.packed[3u * (unsigned)k0 + (unsigned)(32 * j + lane)];
  const unsigned base = (s.bases[k0 >> 5] >> ((lane >> 3) * 8)) & 0xffu;
  unsigned w[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int q = 3 * lane + i, sl = q & 31, j = q >> 5;
    const unsigned a0 = r4d_arc_shfl(ld[0], sl), a1 = r4d_arc_shfl(ld[1], sl);
    const unsigned a2 = r4d_arc_shfl(ld[2], sl);
    w[i] = j == 0 ? a0 : j == 1 ? a1 : a2;
  }
  if (base == 0xFFu)
    return make_int4((int)w[0], (int)w[1], (int)w[2], (int)s.spill[k0 + lane]);
  return r4d_arc_dec(w, base);
}
