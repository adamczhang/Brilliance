#pragma once
// r4d_ar_wide_host.h: Brilliance, what the four-rank host-mailbox kernels share: W1's loud stop
// (r4d_ar_wide_host.hip) and W2/W3's resend (r4d_ar_wide_rs.hip). Moved here from
// r4d_ar_wide_host.hip unchanged, so W1's kernels compile to the same code; the resend's kernels live
// in their own file so that W1's code object stays byte for byte what it was (one translation unit
// lays out every kernel's LDS together).
#include "r4d_ar_wide_x.h"
#include "r4d_ar_hc.h"

#define ARW_WS 4
#define ARW_NP 3

// Everything a wide host kernel needs of its instance, by value.
struct ArwArgs {
  R4DArWidePeers<int4, ARW_NP> peers;
  const int4* my_base;      // this rank's slots (pinned host)
  unsigned int* my_flags;   // this rank's flags (pinned host, coherent)
  unsigned int* seq_ctrs;   // VRAM
  long long md;             // the VRAM mirror's offset from the slots, in bytes
  int slot_stride16, rank, drain, acq, pub;
  unsigned int iw;          // the injection word (INJ instantiations only)
};

__device__ __forceinline__ unsigned int* arw_ctl(const ArwArgs& A) {
  return A.my_flags + R4D_ARW_CTL_OFF(ARW_WS, R4D_AR_WIDE_MAX_BLOCKS);
}
__device__ __forceinline__ const int4* arw_slot(const ArwArgs& A, int parity, int src) {
  return A.my_base + ((size_t)parity * ARW_WS + src) * (size_t)A.slot_stride16;
}
__device__ __forceinline__ int arw_inj_peer(unsigned int iw) {
  const int p = (int)((iw >> 10) & 7u);
  return p < ARW_NP ? p : 0;
}

// The default one-shot's fold (ar_vsum): fp32, ascending rank, one rounding at the store.
template <typename T, int LANES>
__device__ __forceinline__ int4 arw_vsum(const int4* const (&src)[ARW_WS], int k) {
  float acc[LANES];
#pragma unroll
  for (int j = 0; j < LANES; ++j) acc[j] = 0.0f;
#pragma unroll
  for (int r = 0; r < ARW_WS; ++r) {
    const int4 v = src[r][k];
    const T* pv = reinterpret_cast<const T*>(&v);
#pragma unroll
    for (int j = 0; j < LANES; ++j) acc[j] += r4d_ar_to_f(pv[j]);
  }
  int4 o;
  T* po = reinterpret_cast<T*>(&o);
#pragma unroll
  for (int j = 0; j < LANES; ++j) r4d_ar_from_f(acc[j], po[j]);
  return o;
}

// The reader's try over a one-shot block: every peer in `mask`, words [start, end) of its slot,
// each kept in the mirror. The usual try (all three) loads the three peers' words together.
__device__ __forceinline__ void arw_rehash_slots(const int4* const (&sp)[ARW_NP], long long md,
                                                 int start, int end, unsigned int mask,
                                                 unsigned int (&h)[ARW_NP]) {
  const int tid = threadIdx.x, nt = blockDim.x;
  if (mask == 7u) {
    for (int k = start + tid; k < end; k += nt) {
      const int4 a0 = r4d_archk_keep(sp[0] + k, md);
      const int4 a1 = r4d_archk_keep(sp[1] + k, md);
      const int4 a2 = r4d_archk_keep(sp[2] + k, md);
      h[0] += r4d_archk_u4(a0, 4u * (unsigned)k);
      h[1] += r4d_archk_u4(a1, 4u * (unsigned)k);
      h[2] += r4d_archk_u4(a2, 4u * (unsigned)k);
    }
    return;
  }
#pragma unroll
  for (int p = 0; p < ARW_NP; ++p) {
    if (!((mask >> p) & 1u)) continue;
    for (int k = start + tid; k < end; k += nt)
      h[p] += r4d_archk_u4(r4d_archk_keep(sp[p] + k, md), 4u * (unsigned)k);
  }
}

__device__ __forceinline__ int arw_g_dst(int k, int row16, int n16, int who, int dp16) {
  if (row16 <= 0 || row16 >= n16) return who * n16 + k;
  const int m = k / row16, c = k - m * row16;
  return m * dp16 + who * row16 + c;
}
__device__ __forceinline__ int arw_g_src(int k, int row16, int n16, int sp16) {
  if (row16 <= 0 || row16 >= n16 || sp16 == row16) return k;
  const int m = k / row16, c = k - m * row16;
  return m * sp16 + c;
}

// ---- W2/W3's launches (r4d_ar_wide_rs.hip): the kernels with the exchange, picked by the launchers
// of r4d_ar_wide_host.hip when the resend is on. `A.iw` != 0 picks the injected instantiation.
// dtype 0 bf16, 1 fp16, 2 fp32.
void arw_rs_oneshot(int dtype, int nb, int nt, hipStream_t st, const int4* in4, int4* out4, int n16,
                    const ArwArgs& A);
void arw_rs_hc_write(bool gather, bool by_inv, int nb, int nt, hipStream_t st, int4* y4,
                     const unsigned short* inj, unsigned short* h, int n16, int n, int hc,
                     long long irow, long long hrow, const R4dArGather& G, const int* inv,
                     const ArwArgs& A);
void arw_rs_gather(int nb, int nt, hipStream_t st, const int4* in4, int4* out4, int n16, int row16,
                   int dp16, int sp16, const ArwArgs& A);
// ---- W4's (r4d_ar_wide_ts.hip): the two-shot, plain (`off` its WS + 1 shard offsets) and coded
// (bf16; `n16` the message's 16-byte words, every shard whole waves' 32 chunks).
void arw_rs_twoshot(int dtype, int nb, int nt, hipStream_t st, const int4* in4, int4* out4,
                    const int* off, int sh_max, const ArwArgs& A);
void arw_rs_coded(int nb, int nt, hipStream_t st, const int4* in4, int4* out4, int n16,
                  const ArwArgs& A);
size_t arw_coded_region_bytes(long long SL);   // one coded region of a shard of SL chunks
