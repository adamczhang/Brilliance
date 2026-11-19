#pragma once
// r4d_ar_seq.h: the first step of every all-reduce and all-gather kernel (its handshake sequence
// number and scratch slot) and the wrap-safe wait after the push. Shared by the 2-rank and wide
// families: kernels sharing an instance's buffers must take the step identically or the ranks
// disagree.
//
// Counter allocation: 2 * nbw words, zeroed once (nbw = block capacity the flag array was sized
// for), then the landed counter (r4d_ar_link_landed):
//
//   [0, nbw)       Handshake counters, one per block. Block b advances word b once per launch it
//                  runs in, publishes it to the peer and waits for the peer's. Per block so a
//                  block idle through narrow launches resumes one step past its peer's last flag;
//                  the wrap-safe compare needs that gap far under 2^31.
//
//   [nbw, 2*nbw)   Parity counters. Every launch advances all nbw of them once, whatever its block
//                  count (block b takes words b, b + nb, b + 2*nb, ...), so each equals the launch
//                  count and a block reads the slot parity off its own word.
//
//   [2*nbw]        Landed counter: blocks of the running launch that have seen their peer's push
//                  land; back at zero when the launch ends.
//
// The slot cannot come from the handshake counter. Consecutive launches on one instance must use
// opposite scratch slots, since launch k+1 may push while the peer still reduces launch k. Block
// counts and per-block byte ranges vary by launch, so handshake counters drift apart per block,
// and a block of launch k+1 could push into the slot a different block of launch k is still
// reading. The parity counters move in lockstep, so the slot alternates per launch for every block.
//
// Cost: one more returning atomic, issued from a second wave so its round trip overlaps the
// handshake's, plus nbw / nb fire-and-forget atomics over the other threads. Those only need to
// land before the next launch, which the kernel boundary guarantees. All state is device-resident,
// so a captured graph replays.
#include <hip/hip_runtime.h>

// Brilliance: what a collective's handshake does when its peer never answers -- a lockstep
// violation, since reducing a slot the peer never filled would hand stale bytes on as a sum.
// Linux traps, and the process dies with the fault named. On Windows a device trap HANGS the
// process instead (rocm-systems#12572), and a hung process cannot be stopped or restarted by
// anything short of a reboot. So there the wave keeps waiting, Windows' GPU timeout detection
// (TDR) resets the adapter, every later HIP call fails with a device error, and the engine stops
// loudly. Brought over from Brilliance 0.2.2; every trap site in libr4d's collectives takes it.
#ifndef R4D_AR_GIVE_UP
#if defined(_WIN32)
#define R4D_AR_GIVE_UP() ((void)0)
#else
#define R4D_AR_GIVE_UP() __builtin_trap()
#endif
#endif

// Sets `s_seq` to this block's handshake value and `s_slot` to the launch's scratch slot (0 or 1),
// then barriers so every thread can read both. `nbw` is the capacity the counters were allocated
// for, and gridDim.x must not exceed it. `steps` handshake values are taken at once (s_seq is the
// first): a kernel that handshakes more than twice per launch reserves the flag values it
// publishes, so every kernel on the instance stays in the same monotone flag space. All ranks must
// take the same number for the same block.
//
// The barrier fences LDS only, since it publishes just the two LDS words. A full __syncthreads
// would wait for the outstanding parity atomics and invalidate the vector caches, putting that
// traffic on the path to the first push.
__device__ __forceinline__ void r4d_ar_seq_take(unsigned int* __restrict__ ctrs, unsigned int nbw,
                                                unsigned int& s_seq, unsigned int& s_slot,
                                                unsigned int steps = 1u) {
  const unsigned int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  unsigned int* const par = ctrs + nbw;
  const unsigned int pt = nt > 32u ? 32u : 0u;   // first lane of the second wave, if there is one
  if (tid == 0) s_seq = atomicAdd(&ctrs[b], steps) + 1u;
  if (tid == pt) s_slot = (atomicAdd(&par[b], 1u) + 1u) & 1u;
  for (unsigned int j = b + (tid + 1u) * nb; j < nbw; j += nt * nb) atomicAdd(&par[j], 1u);
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// The engine's hold on the link, released by the launch (rad_abi.h, rad_kernel_collective
// returning 2). The word is odd exactly while the engine holds the link for this launch and
// nothing else writes it meanwhile, so each block's first thread reads it once at the start
// (r4d_ar_link_word), where the host-memory read overlaps the push instead of delaying the reduce.
//
// Every block calls r4d_ar_link_landed once its handshake has seen the peer's push land. The last
// block knows all of the peer's bytes have arrived, so the link is free for the rest of the
// kernel; it decrements the word if odd, atomically so it commutes with any barrier counting the
// same word down. `ctr` counts blocks and returns to zero at the end of the launch. The DMA engine
// polls the word past every cache, so it is accessed at system scope.
__device__ __forceinline__ long long r4d_ar_link_word(const long long* gate, unsigned int tid) {
  return gate && tid == 0 ? __hip_atomic_load(gate, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                          : 0;
}

__device__ __forceinline__ void r4d_ar_link_landed(long long* gate, long long word,
                                                   unsigned int* ctr, unsigned int nb,
                                                   unsigned int tid) {
  if (!gate || tid != 0) return;
  const unsigned int prev = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_RELAXED,
                                                   __HIP_MEMORY_SCOPE_AGENT);
  if (prev + 1u != nb) return;
  __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (word & 1) __hip_atomic_fetch_add(gate, -1LL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// True while `flag` has not reached `want`. The u32 counters wrap, so compare the signed
// difference: a peer at or past `want` (possibly a launch ahead) reads as arrived across the wrap,
// where a plain `flag < want` would stop waiting once `want` wraps to a small number.
__device__ __forceinline__ bool r4d_ar_seq_behind(unsigned int flag, unsigned int want) {
  return (int)(flag - want) < 0;
}

// ---- The push check. A peer's push over PCIe P2P can land after its release flag (measured on 2x
// R9700: the reader sometimes found the slot's previous contents right after the handshake, and
// the new bytes ~1 us later), so the flag alone does not say the bytes are there. The pushed bytes
// wait in the writer's L2 for the release's writeback, and that writeback and the flag store reach
// the peer by different paths. The writer sums mix(dword, index) over what it pushes and stores the
// block's sum, salted with its handshake value, into the peer's check word for (slot, block):
// fine-grained words after the flags (r4d_ar_entry.hip sizes the allocation). The reader sums the
// same over the bytes it is about to read and reads again until the two agree. A stale element and
// a stale check word (an older handshake value) both disagree, so only the landed push passes; the
// order the two land in does not matter. The sum is order-free, so writer and reader may split the
// bytes differently.
#define R4D_ARCHK_SALT 0x6C8E9CF5u
// wall_clock64 ticks (100 MHz): 2 s. A push that never lands is a lockstep fault, so the reader traps
// (Linux), or, in a two-rank host twin on Windows, gives up and records a fault (below).
#define R4D_ARCHK_DEADLINE 200000000ull

// ---- Brilliance: the bounded failure of a two-rank host twin (fix C1 stage 1,
// the first fix of the TP2 hang). On Windows a device trap hangs
// the process, so R4D_AR_GIVE_UP() does nothing there and a push that arrives wrong (one 64-byte line
// of a block lost or misdirected, SUPPORT.md issue 1) made the reader spin for ever. A host twin's
// waits now end: the push check after R4D_ARCHK_DEADLINE, the flag spin after R4D_AR_FLAG_DEADLINE.
// The waiting block writes a fault record into its own flags allocation (pinned host memory), lets the
// kernel finish (its output is wrong) and the engine, which reads the record after every step
// (r4d_ar_fault_poll, r4d_ar_entry.hip), fails the step before any of its tokens goes out and stops.
// Both ranks took their handshake value and parity when the launch began, so the counters stay in
// lockstep and every later collective on the instance completes; the give-up's slot is next used two
// launches on, re-pushed whole under a new salt.
//
// Which kernels: those passing a nonzero `op` (R4D_AR_OP_*), which the four two-rank kernels of the
// TP2 serve do exactly when they run as a host twin (`md != 0`: only build() makes a host instance's
// VRAM mirror, and it sizes that instance's flags for the control words below). Every other caller,
// and every caller on Linux, keeps R4D_AR_GIVE_UP. The healthy path is unchanged: the deadline is
// tested only in a try that failed, and the flag spin's clock only once the first load found the flag
// behind.
//
// A two-rank host twin's flags allocation, in words (nbw = r4d_ar_max_blocks()):
//
//   [0, nbw)              the peer's handshake flags                       (written by the peer)
//   [nbw, 3 nbw)          the push check words, two parity slots           (the peer)
//   [3 nbw, 5 nbw)        the reader's diagnostic words (r4d_archk_wait)   (this rank)
//   [5 nbw, 6 nbw)        stage 2: ack[b] = s, "your push of block b at s checked"   (the peer)
//   [6 nbw, 8 nbw)        stage 2: rq[b], 64-bit, (s << 32) | n: the peer's n-th request for a
//                         resend of block b at s                           (the peer)
//   [8 nbw, 12 nbw)       stage 2: counters, 4 a block (R4D_AR2_CNT_*): requests raised, resends
//                         served, control words re-stored, acks that never came  (this rank)
//   [12 nbw, +16)         the fault record, copy A (one 64-byte line)      (this rank)
//   [12 nbw + 16, +16)    the fault record, copy B (the next line)         (this rank)
//   [12 nbw + 32, +16)    R4D_AR2_CTL_*: park switch and release (the host), fault count (this rank),
//                         the request limit (the host)
//   [12 nbw + 48, +4 nbw) stage 3: ar_gather_hc_write's counters, as [8 nbw, 12 nbw) are
//                         ar_hc_write's, so the host can tell which collective a resend was in
//                         (this rank). At the end, so no word above moved.
//   [16 nbw + 48, +4 nbw) stage 4: the plain exact all-reduce's counters (this rank)
//   [20 nbw + 48, +4 nbw) stage 4: the two-rank all-gather's counters (this rank). Appended, as
//                         stage 3's were, so no word above moved.
//   [24 nbw + 48, +4 nbw) the rotated wire (tp-wire wht6): the wht6 all-reduce's counters
//   [28 nbw + 48, +4 nbw) the wht6 fused write's (r4d_ar_hc_write_wht6_kernel without the gather)
//   [32 nbw + 48, +4 nbw) the wht6 fused write's with the MoE gather. Appended, so no word above
//                         moved (this rank).
// Every word but the counters, the record and the CTL line is written by the PEER, over the link.
#define R4D_AR2_FLAG_WORDS(nbw) (36ull * (nbw) + 48ull)
#define R4D_AR2_ACK_OFF(nbw)    (5u * (nbw))
#define R4D_AR2_RQ_OFF(nbw)     (6u * (nbw))   // 64-bit words; 8-byte aligned, as nbw is even
#define R4D_AR2_CNT_OFF(nbw)    (8u * (nbw))
#define R4D_AR2_REC_OFF(nbw)    (12u * (nbw))
#define R4D_AR2_CTL_OFF(nbw)    (12u * (nbw) + 32u)
#define R4D_AR2_CNT_GHC_OFF(nbw) (12u * (nbw) + 48u)   // 64-byte aligned, as nbw is a multiple of 4
#define R4D_AR2_CNT_EXACT_OFF(nbw)  (16u * (nbw) + 48u)
#define R4D_AR2_CNT_GATHER_OFF(nbw) (20u * (nbw) + 48u)
#define R4D_AR2_CNT_WHT6_OFF(nbw)    (24u * (nbw) + 48u)
#define R4D_AR2_CNT_HC_WHT6_OFF(nbw) (28u * (nbw) + 48u)
#define R4D_AR2_CNT_GHC_WHT6_OFF(nbw) (32u * (nbw) + 48u)
// The exchange's counters of collective `op` (R4D_AR_OP_*): each collective its own 4 words a block,
// ar_hc_write's (and an unknown op's) at R4D_AR2_CNT_OFF.
#define R4D_AR2_CNT_OP_OFF(nbw, op)                                \
  ((op) == R4D_AR_OP_GATHER_HC ? R4D_AR2_CNT_GHC_OFF(nbw)         \
   : (op) == R4D_AR_OP_EXACT   ? R4D_AR2_CNT_EXACT_OFF(nbw)       \
   : (op) == R4D_AR_OP_GATHER  ? R4D_AR2_CNT_GATHER_OFF(nbw)      \
   : (op) == R4D_AR_OP_WHT6    ? R4D_AR2_CNT_WHT6_OFF(nbw)        \
   : (op) == R4D_AR_OP_HC_WHT6 ? R4D_AR2_CNT_HC_WHT6_OFF(nbw)     \
   : (op) == R4D_AR_OP_GHC_WHT6 ? R4D_AR2_CNT_GHC_WHT6_OFF(nbw)   \
                               : R4D_AR2_CNT_OFF(nbw))
#define R4D_AR2_CTL_PARK    0u    // host: nonzero parks a block that gave up (BRILLIANCE_STALL_CAPTURE)
#define R4D_AR2_CTL_RELEASE 1u    // host: bumped when the stall capture is done with the instance
#define R4D_AR2_CTL_FAULTS  2u    // device: give-ups on this instance (thread 0's plain read and store)
#define R4D_AR2_CTL_RQMAX   3u    // host: the request limit (BRILLIANCE_AR_RESEND_MAX); 0, the default
// The counters, words [CNT_OFF + 4 b + j): block b's thread 0 only, a plain load and store each.
#define R4D_AR2_CNT_RQ  0u   // requests raised, as the reader of the peer's push
#define R4D_AR2_CNT_RS  1u   // resends served, as the writer
#define R4D_AR2_CNT_RE  2u   // control words re-stored (the flag, the ack) while waiting on the peer
#define R4D_AR2_CNT_ACK 3u   // acks that never came: covered by the peer's next flag, or timed out
// The record's 16 words. MAGIC marks a written copy and SUM is over the first 15, so a copy torn by
// a lost line, or two blocks writing at once, is told from a whole one. The host treats ANY nonzero
// word of the record as a fault, whole or not: the region is zero until a give-up writes it.
#define R4D_AR_FAULT_MAGIC 0xFA17C1D1u
enum {
  R4D_ARF_MAGIC = 0, R4D_ARF_KIND, R4D_ARF_OP, R4D_ARF_BLOCK, R4D_ARF_SEQ, R4D_ARF_SLOT,
  R4D_ARF_TRIES, R4D_ARF_GOT, R4D_ARF_WANT, R4D_ARF_FLAG, R4D_ARF_NB, R4D_ARF_COUNT, R4D_ARF_US,
  R4D_ARF_R13, R4D_ARF_R14, R4D_ARF_SUM, R4D_ARF_WORDS
};
#define R4D_AR_FAULT_CHECK 1u   // kind: the push check never agreed with the check word
#define R4D_AR_FAULT_FLAG  2u   // kind: the peer's flag never reached the handshake value
#define R4D_AR_OP_EXACT       1u   // r4d_ar_oneshot_2rank_exact_kernel
#define R4D_AR_OP_HC_WRITE    2u   // r4d_ar_hc_write_kernel
#define R4D_AR_OP_GATHER_HC   3u   // r4d_ar_gather_hc_write_kernel
#define R4D_AR_OP_GATHER      4u   // r4d_ar_gather_2rank_kernel
// The rotated 6-bit wire (tp-wire wht6), its two-rank host twins with the exchange only: they have
// no stage 1 form (BRILLIANCE_AR_RESEND=0 runs their kernels as they were, with no bounded failure).
#define R4D_AR_OP_WHT6        5u   // r4d_ar_oneshot_2rank_wht6_kernel
#define R4D_AR_OP_HC_WHT6     6u   // r4d_ar_hc_write_wht6_kernel, without the gather
#define R4D_AR_OP_GHC_WHT6    7u   // r4d_ar_hc_write_wht6_kernel, with the MoE gather
#define R4D_AR_OP_LAST        7u
// wall_clock64 ticks: 30 s for the flag. Both rank threads issue a step behind one barrier, so a
// legitimate skew is far below it; past it, a flag store was lost (or the peer is not running).
#define R4D_AR_FLAG_DEADLINE 3000000000ull
// Once an instance has a fault on record, a later wait on it gives up after this (10 ms) instead, so
// a peer that is out of step does not cost a deadline per block per collective before the engine
// reads the record. A wait first looks after R4D_AR_FAULT_PEEK (1 ms) of failing.
#define R4D_AR_FAULT_FAST 1000000ull
#define R4D_AR_FAULT_PEEK 100000ull
// The park's limit (10 s): with the stall capture on, the block that recorded the fault waits for the
// capture to read its slot and streams before its reduce writes them.
#define R4D_AR_PARK_MAX 1000000000ull
// The record's sum, the same on the host (r4d_ar_entry.hip).
#define R4D_AR_FAULT_SUM_SEED 0x2545F491u
#define R4D_ARCHK_U16 0x80000000u    // index space of 16-bit fields (wire scales), apart from dwords

// MurmurHash3's finalizer over (value, index): every bit of the value reaches every bit of the term,
// so a sum of terms changes when any value does. A linear mix is not enough: (v ^ c_e) summed over
// a run of indices can lose v entirely, and a message of one repeated value (a test pattern, a
// zero-filled row) would then pass whatever the slot held.
__device__ __forceinline__ unsigned int r4d_archk_mix(unsigned int v, unsigned int e) {
  unsigned int x = v ^ (e * 0x9E3779B9u);
  x ^= x >> 16; x *= 0x85EBCA6Bu;
  x ^= x >> 13; x *= 0xC2B2AE35u;
  x ^= x >> 16;
  return x;
}
// A 16B word at dword index `e`.
__device__ __forceinline__ unsigned int r4d_archk_u4(int4 v, unsigned int e) {
  return r4d_archk_mix((unsigned int)v.x, e) + r4d_archk_mix((unsigned int)v.y, e + 1u) +
         r4d_archk_mix((unsigned int)v.z, e + 2u) + r4d_archk_mix((unsigned int)v.w, e + 3u);
}
__device__ __forceinline__ unsigned int r4d_archk_u4(uint4 v, unsigned int e) {
  return r4d_archk_mix(v.x, e) + r4d_archk_mix(v.y, e + 1u) + r4d_archk_mix(v.z, e + 2u) +
         r4d_archk_mix(v.w, e + 3u);
}
// The check word of (slot, block) in a flags allocation: two parity slots of nbw words after the
// nbw flags.
__device__ __forceinline__ unsigned int* r4d_archk_word(unsigned int* flags, unsigned int nbw,
                                                        unsigned int slot, unsigned int b) {
  return flags + nbw + slot * nbw + b;
}
__device__ __forceinline__ unsigned int r4d_archk_wave_sum(unsigned int h) {
  for (int o = 16; o > 0; o >>= 1) h += __shfl_xor(h, o, 32);
  return h;
}

// Writer: every thread of the block calls this after its pushes, before the handshake's barrier.
// Thread 0 stores the block's sum into the peer's check word, and returns it (unsalted; 0 on every
// other thread) for a resend's re-put (r4d_archk_exchange).
__device__ __forceinline__ unsigned int r4d_archk_put(unsigned int h, unsigned int* peer_flags,
                                              unsigned int nbw, unsigned int slot,
                                              unsigned int b, unsigned int seq,
                                              bool release = false) {
  __shared__ unsigned int s_ckp[32];
  const unsigned int tid = threadIdx.x, nt = blockDim.x;
  h = r4d_archk_wave_sum(h);
  if ((tid & 31u) == 0u) s_ckp[tid >> 5] = h;
  __syncthreads();
  unsigned int t = 0u;
  if (tid == 0) {
    for (unsigned int w = 0; w < (nt + 31u) / 32u; ++w) t += s_ckp[w];
    /* The pushed bytes sit in this card's L2 until a system-scope release writes them back. The
     * flag's release, right after this, does that, so the word itself can be relaxed (the reader
     * re-reads until the bytes agree with it). Where the flag went out before the push (the test
     * mode, drain 4) nothing else would write them back before the kernel ends: a release. */
    unsigned int* const cw = r4d_archk_word(peer_flags, nbw, slot, b);
    if (release)
      __hip_atomic_store(cw, t + seq * R4D_ARCHK_SALT, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    else
      __hip_atomic_store(cw, t + seq * R4D_ARCHK_SALT, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return t;
}

// ---- Brilliance: the bounded failure's device half (above). Thread 0 only, and only past a
// deadline, so none of this is on a healthy path.
__device__ __forceinline__ unsigned int r4d_ar_ld(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void r4d_ar_st(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
// Whether an instance already has a fault on record (its count, or either copy's magic).
__device__ __forceinline__ bool r4d_ar_faulted(unsigned int* flags, unsigned int nbw) {
  const unsigned int* rec = flags + R4D_AR2_REC_OFF(nbw);
  return r4d_ar_ld(flags + R4D_AR2_CTL_OFF(nbw) + R4D_AR2_CTL_FAULTS) != 0u ||
         r4d_ar_ld(rec + R4D_ARF_MAGIC) == R4D_AR_FAULT_MAGIC ||
         r4d_ar_ld(rec + R4D_ARF_WORDS + R4D_ARF_MAGIC) == R4D_AR_FAULT_MAGIC;
}
// Counts a give-up and, if the instance has none on record yet, writes the record (both copies). Then
// a system-scope fence, so the record is ordered before this block's next store to the peer: a fault
// whose wrong output reaches the other rank is on record before that rank can finish its step.
// Returns whether this give-up wrote the record (the first on the instance). Fault injection (stage
// 1b, BRILLIANCE_AR_INJECT) needs no hook here: it corrupts the writer's push, and this records the
// reader's give-up as for a real fault.
inline __device__ __noinline__ bool r4d_ar_fault_note(unsigned int* flags, unsigned int nbw,
                                               unsigned int kind, unsigned int op, unsigned int b,
                                               unsigned int seq, unsigned int slot,
                                               unsigned int tries, unsigned int got,
                                               unsigned int want, unsigned int flag,
                                               unsigned long long ticks) {
  unsigned int* const ctl = flags + R4D_AR2_CTL_OFF(nbw);
  unsigned int* const rec = flags + R4D_AR2_REC_OFF(nbw);
  const bool first = !r4d_ar_faulted(flags, nbw);
  const unsigned int count = r4d_ar_ld(ctl + R4D_AR2_CTL_FAULTS) + 1u;
  r4d_ar_st(ctl + R4D_AR2_CTL_FAULTS, count);
  if (first) {
    unsigned int w[R4D_ARF_WORDS];
    w[R4D_ARF_MAGIC] = R4D_AR_FAULT_MAGIC; w[R4D_ARF_KIND] = kind; w[R4D_ARF_OP] = op;
    w[R4D_ARF_BLOCK] = b; w[R4D_ARF_SEQ] = seq; w[R4D_ARF_SLOT] = slot; w[R4D_ARF_TRIES] = tries;
    w[R4D_ARF_GOT] = got; w[R4D_ARF_WANT] = want; w[R4D_ARF_FLAG] = flag; w[R4D_ARF_NB] = gridDim.x;
    w[R4D_ARF_COUNT] = count; w[R4D_ARF_US] = (unsigned int)(ticks / 100ull);
    w[R4D_ARF_R13] = 0u; w[R4D_ARF_R14] = 0u;
    unsigned int sum = R4D_AR_FAULT_SUM_SEED;
    for (unsigned int i = 0; i < R4D_ARF_SUM; ++i) sum += r4d_archk_mix(w[i], i);
    w[R4D_ARF_SUM] = sum;
    for (unsigned int c = 0; c < 2u; ++c)
      for (unsigned int i = 0; i < R4D_ARF_WORDS; ++i) r4d_ar_st(rec + c * R4D_ARF_WORDS + i, w[i]);
  }
  __threadfence_system();
  return first;
}
// The park (BRILLIANCE_STALL_CAPTURE): the block that recorded an instance's first fault waits, up to
// R4D_AR_PARK_MAX, until the host bumps the release word, so the stall capture reads the slot and the
// hc streams before the reduce writes them. The host arms it at build only with the capture on.
inline __device__ __noinline__ void r4d_ar_fault_park(unsigned int* flags, unsigned int nbw) {
  unsigned int* const ctl = flags + R4D_AR2_CTL_OFF(nbw);
  if (r4d_ar_ld(ctl + R4D_AR2_CTL_PARK) == 0u) return;
  const unsigned int r0 = r4d_ar_ld(ctl + R4D_AR2_CTL_RELEASE);
  const unsigned long long t0 = wall_clock64();
  while (r4d_ar_ld(ctl + R4D_AR2_CTL_RELEASE) == r0 && wall_clock64() - t0 < R4D_AR_PARK_MAX)
    __builtin_amdgcn_s_sleep(127);
}

// Thread 0's handshake spin, for the four two-rank kernels: wait until the peer's flag for block b
// reaches `s`. With op 0, or off Windows, as it always was: R4D_AR_GIVE_UP past `spin_max` loads.
// With op != 0 on Windows (a host twin), a wall-clock deadline instead, R4D_AR_FLAG_DEADLINE from the
// first time the clock is read: then the fault is recorded and it returns false. The clock is read
// every 64th failed load, so a flag that has arrived costs exactly the one acquire load it always did.
__device__ __forceinline__ bool r4d_ar_flag_wait(unsigned int* my_flags, unsigned int nbw,
                                                 unsigned int b, unsigned int s,
                                                 unsigned long long spin_max, unsigned int op,
                                                 unsigned int slot) {
  unsigned long long z = 0;
#if defined(_WIN32)
  unsigned long long t0 = 0, dl = R4D_AR_FLAG_DEADLINE;
  bool peeked = false;
#endif
  unsigned int f;
  while (r4d_ar_seq_behind(f = __hip_atomic_load(&my_flags[b], __ATOMIC_ACQUIRE,
                                                  __HIP_MEMORY_SCOPE_SYSTEM), s)) {
#if defined(_WIN32)
    if (op) {
      if ((++z & 63ull) != 0ull) continue;
      const unsigned long long now = wall_clock64();
      if (t0 == 0ull) { t0 = now; continue; }
      if (!peeked && now - t0 > R4D_AR_FAULT_PEEK) {
        peeked = true;
        if (r4d_ar_faulted(my_flags, nbw)) dl = R4D_AR_FAULT_FAST;
      }
      if (now - t0 > dl) {
        (void)r4d_ar_fault_note(my_flags, nbw, R4D_AR_FAULT_FLAG, op, b, s, slot, (unsigned int)z,
                                0u, s, f, now - t0);
        return false;
      }
      continue;
    }
#endif
    (void)op; (void)nbw; (void)slot;
    if (++z > spin_max) R4D_AR_GIVE_UP();
  }
  return true;
}

// ---- Brilliance: fault injection (fix C1 stage 1b; the design's section 4, Q2). Off by default:
// BRILLIANCE_AR_INJECT=<op>:<mode>:<every>[:<rank>] (r4d_ar_entry.hip parses it and counts each
// host instance's launches of that op), or r4d_ar_inject_pin for r4d_selftest. A launch chosen for
// it runs a separate instantiation of its kernel (template flag INJ), so the kernels the serve runs
// without the switch are the same code as before. The writer (rank 1 by default, the writer in all
// three captures) damages its push of ONE block of the launch:
//   lose   one 64-byte line of the block's push is not stored: the peer's slot keeps its old bytes
//          (cap-1 and cap-3 of the TP2 hang ledger);
//   dup    that line is stored with the bytes of another line of the same push: 449 lines earlier
//          (cap-2), or the block's first line when the block is shorter;
//   stuck  as lose, and in stage 2 on every resend too (a path that never heals); in stage 1,
//          which has no resend, the same as lose;
//   noflag the block's handshake flag is not stored (ctl-flag; the reader's 30 s flag deadline,
//          or in stage 2's exchange the writer's re-store).
// Stage 2 (the exchange, r4d_archk_exchange; ar_hc_write, from stage 3 ar_gather_hc_write, and from
// stage 4 the plain exact all-reduce and the all-gather: every two-rank host twin with the op):
//   pair   lose in block b; block b + 1 (b - 1 for the last block), after its own push, stores
//          b's lost line over its own line at the same offset: a misdirection into another block
//   late   the line is withheld from the push and stored from the loop R4D_ARX_T_LATE after
//   ctl-ack  the writer, as the reader of the peer's push, skips its first ack store
//   ctl-rq   the writer, as a reader, finds one line of its own slot wrong (it overwrites it, as
//            a lost line of the peer's push would leave it) and skips its first request store
// The check word still carries the sum of the intended bytes, so the reader's push check fails
// exactly as for the real fault and the bounded failure above records it.
//
// The word reaches the kernel in bits [R4D_AR_INJ_SHIFT, 32) of its `drain` argument (whose own
// values sit below bit 10; bit 10 is the launcher's resend switch): mode in 4 bits, block in 9
// (R4D_AR_MAX_BLOCKS is 512), and the line's offset from the block's first whole 64-byte line in
// 8. The launchers take it from bits 32 and up of their own 64-bit `drain` (mode, and the
// injection's ordinal, which picks the block).
#define R4D_AR_INJ_LOSE   1u
#define R4D_AR_INJ_DUP    2u
#define R4D_AR_INJ_STUCK  3u
#define R4D_AR_INJ_NOFLAG 4u
#define R4D_AR_INJ_PAIR   5u
#define R4D_AR_INJ_LATE   6u
#define R4D_AR_INJ_ACK    7u
#define R4D_AR_INJ_RQ     8u
#define R4D_AR_INJ_MODES  8u
#define R4D_AR_INJ_SHIFT  11
// Bit 10 of a two-rank launcher's 64-bit `drain`: the entry asks for the exchange (fix C1 stage 2,
// r4d_archk_exchange below). The launcher strips it; it never reaches a kernel.
#define R4D_AR_RESEND_BIT 0x400LL
#define R4D_AR_INJ_DUP_LINES 449   // cap-2: the wrong line was a copy of the line 0x7040 B earlier
__host__ __device__ __forceinline__ unsigned int r4d_ar_inj_word(unsigned int mode, unsigned int blk,
                                                                 unsigned int off) {
  return (mode & 15u) | ((blk & 511u) << 4) | ((off & 255u) << 13);
}
__host__ __device__ __forceinline__ int r4d_ar_inj_partner(int b, int nb) {
  return b + 1 < nb ? b + 1 : b - 1;   // -1 for a one-block launch: pair is lose there
}
// One launch's damage, for block b: words [lo, hi) of the slot are not pushed as they are; with
// `delta` != 0 (dup) they are pushed with the bytes of words [lo - delta, hi - delta) instead.
// Stage 2's fields (r4d_ar_inj_plan2): pair's copy, words [clo, chi) of this block get the bytes
// of [clo - cdelta, chi - cdelta) after the push; ctl-rq's own slot words [blo, bhi).
struct R4dArInj {
  int lo, hi, delta;
  bool noflag;
  int clo, chi, cdelta, blo, bhi;
  bool late, stuck, skip_ack, skip_rq;
};
// Words [lo, hi) of the whole line `off` lines into [start, end), lo == hi when there is none.
// `base` is the slot base: a line is a 64-byte line of the address space, so the slot's alignment
// (16 bytes) decides which four words share one.
__device__ __forceinline__ void r4d_ar_inj_line(int off, int start, int end, const void* base,
                                                int& lo, int& hi, int& lf, int& L) {
  const int ph = (int)(((uintptr_t)base >> 4) & 3u);   // the slot base's word within its line
  lf = (start + ph + 3) >> 2;
  const int ll = ((end + ph) >> 2) - 1;                 // whole lines in [start, end)
  lo = hi = 0; L = -1;
  if (ll < lf) return;                                  // a block shorter than one line: nothing
  if (off > ll - lf) off = ll - lf;
  L = lf + off;
  lo = 4 * L - ph;
  hi = lo + 4;
}
__device__ __forceinline__ R4dArInj r4d_ar_inj_plan(unsigned int iw, int b, int start, int end,
                                                    const void* ps) {
  R4dArInj r{ 0, 0, 0, false, 0, 0, 0, 0, 0, false, false, false, false };
  const unsigned int mode = iw & 15u;
  if (!mode || (int)((iw >> 4) & 511u) != b) return r;
  if (mode == R4D_AR_INJ_NOFLAG) { r.noflag = true; return r; }
  if (mode == R4D_AR_INJ_ACK || mode == R4D_AR_INJ_RQ) return r;   // stage 2's: r4d_ar_inj_plan2
  int lf = 0, L = -1;
  r4d_ar_inj_line((int)((iw >> 13) & 255u), start, end, ps, r.lo, r.hi, lf, L);
  if (L < 0) return r;
  if (mode == R4D_AR_INJ_DUP) {
    const int ph = (int)(((uintptr_t)ps >> 4) & 3u);
    const int ll = ((end + ph) >> 2) - 1;
    const int S = L - R4D_AR_INJ_DUP_LINES >= lf ? L - R4D_AR_INJ_DUP_LINES
                : L > lf ? lf : (ll > L ? ll : L);
    r.delta = 4 * (L - S);   // 0 (a one-line block): the line is lost instead
  }
  return r;
}
// Stage 2's plan, for a kernel with the exchange: block b of nb, each block `w` words of n16, its
// peer slot `ps` and its own slot `ms`. Modes of stage 1 as r4d_ar_inj_plan.
__device__ __forceinline__ R4dArInj r4d_ar_inj_plan2(unsigned int iw, int b, int nb, int w, int n16,
                                                     const void* ps, const void* ms) {
  const int start = b * w, end = start + w < n16 ? start + w : n16;
  R4dArInj r = r4d_ar_inj_plan(iw, b, start, end, ps);
  const unsigned int mode = iw & 15u;
  const int bt = (int)((iw >> 4) & 511u), off = (int)((iw >> 13) & 255u);
  int lf = 0, L = -1;
  if (bt == b) {
    r.stuck = mode == R4D_AR_INJ_STUCK;
    r.late = mode == R4D_AR_INJ_LATE;
    r.skip_ack = mode == R4D_AR_INJ_ACK;
    if (mode == R4D_AR_INJ_RQ) {
      r.skip_rq = true;
      r4d_ar_inj_line(off, start, end, ms, r.blo, r.bhi, lf, L);
    }
  } else if (mode == R4D_AR_INJ_PAIR && r4d_ar_inj_partner(bt, nb) == b) {
    const int ts = bt * w, te = ts + w < n16 ? ts + w : n16;
    int tlo = 0, thi = 0, tlf = 0, tL = -1;
    r4d_ar_inj_line(off, ts, te, ps, tlo, thi, tlf, tL);
    r4d_ar_inj_line(off, start, end, ps, r.clo, r.chi, lf, L);
    if (tL < 0 || L < 0) r.clo = r.chi = 0;
    else r.cdelta = r.clo - tlo;
  }
  return r;
}
// The launchers' half (r4d_ar_oneshot_2rank_exact.hip): from the 64-bit `drain` they were given,
// the kernel's word for a launch of nb blocks over n16 words, or 0; logs the injection and keeps it
// for r4d_ar_inject_last. `op` is R4D_AR_OP_*. The slot (and so the line's exact words) is the
// kernel's: its parity is taken on the card. `w`, when nonzero, is the kernel's own words a block
// (the rotated wire's whole chunks, where the last blocks may be empty); the block is then picked
// from those that hold words.
unsigned int r4d_ar_inj_pick(long long drain64, unsigned int op, int nb, int n16, int w = 0);

// Reader: every thread of the block calls this after the handshake; `rehash(h)` adds the thread's
// share of the bytes the block is about to read, with the same loads the kernel then makes.
//
// Brilliance: `op` (R4D_AR_OP_*, 0 for none) is the bounded failure's switch (above). With op 0, or
// off Windows, a check that never passes ends in R4D_AR_GIVE_UP as before. With op != 0 on Windows,
// past R4D_ARCHK_DEADLINE (or R4D_AR_FAULT_FAST once the instance has a fault on record) thread 0
// records a fault, parks if the host armed the park, and the block returns false; the kernel then
// runs its reduce as usual over the bytes the last try copied to the mirror. Returns true when the
// check passed.
template <typename F>
__device__ __forceinline__ bool r4d_archk_wait(unsigned int* my_flags, unsigned int nbw,
                                               unsigned int slot, unsigned int b,
                                               unsigned int seq, const F& rehash,
                                               unsigned int op = 0u) {
  __shared__ unsigned int s_ckw[32];
  __shared__ unsigned int s_ckok;
  const unsigned int tid = threadIdx.x, nt = blockDim.x;
  const unsigned long long t0 = tid == 0 ? wall_clock64() : 0ull;
  (void)op;
  unsigned int tries = 0u;   // Brilliance: failed tries, for the stall dump (below)
#if defined(_WIN32)
  unsigned long long dl = R4D_ARCHK_DEADLINE;   // Brilliance: the give-up's deadline (op != 0)
  bool peeked = false;
#endif
  for (;;) {
    /* The check word first, so its round trip overlaps the re-hash's loads. Read before the bytes
     * it is compared with, it can only be older than them, and an older word just fails this try. */
    const unsigned int want = tid == 0 ? __hip_atomic_load(r4d_archk_word(my_flags, nbw, slot, b),
                                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                                       : 0u;
    unsigned int h = 0u;
    rehash(h);
    h = r4d_archk_wave_sum(h);
    if ((tid & 31u) == 0u) s_ckw[tid >> 5] = h;
    __syncthreads();
    if (tid == 0) {
      unsigned int t = 0u;
      for (unsigned int w = 0; w < (nt + 31u) / 32u; ++w) t += s_ckw[w];
      const bool ok = t + seq * R4D_ARCHK_SALT == want;
      bool gone = false;   // this try gave up (a host twin on Windows): the block ends the wait
      if (!ok) {
        const unsigned long long el = wall_clock64() - t0;
#if defined(_WIN32)
        if (op) {
          if (!peeked && el > R4D_AR_FAULT_PEEK) {
            peeked = true;
            if (r4d_ar_faulted(my_flags, nbw)) dl = R4D_AR_FAULT_FAST;
          }
          if (el > dl) {
            gone = true;
            if (r4d_ar_fault_note(my_flags, nbw, R4D_AR_FAULT_CHECK, op, b, seq, slot, tries + 1u,
                                  t + seq * R4D_ARCHK_SALT, want, r4d_ar_ld(my_flags + b), el))
              r4d_ar_fault_park(my_flags, nbw);
          }
        } else
#endif
        if (el > R4D_ARCHK_DEADLINE) R4D_AR_GIVE_UP();
      }
      /* Brilliance, diagnostic: a failed try's count and the sum this card computed, in two words
       * after the check words ([3 nbw + 2 b], [3 nbw + 2 b + 1]; r4d_ar_entry.hip sizes the flags
       * for them). Stored on the 1st, 2nd, 4th, ... try and every 1024th, so a late push costs a
       * store or two and a block that never passes keeps the words moving; nothing reads them but
       * the stall dump (r4d_win_mailbox_dump), which tells a spin on the check from one on the
       * flag, and whether the card reads the bytes the CPU sees. The pass path is unchanged. */
      if (!ok) {
        ++tries;
        if ((tries & (tries - 1u)) == 0u || (tries & 1023u) == 0u) {
          unsigned int* const dw = my_flags + 3u * nbw + 2u * b;
          __hip_atomic_store(dw + 1, t, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          __hip_atomic_store(dw, tries, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
      s_ckok = ok ? 1u : (gone ? 2u : 0u);
    }
    __syncthreads();
    if (s_ckok) return s_ckok == 1u;
    __syncthreads();   // s_ckw and s_ckok are rewritten by the next try
    /* Drop the vector caches, so the next try reads the bytes from where the push lands. */
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    __builtin_amdgcn_s_sleep(2);
  }
}

// ---- Brilliance: the exchange, fix C1 stage 2 (the revised protocol of
// the TP2 hang's fix). One loop per block replaces a host twin's flag spin and
// r4d_archk_wait. Each rank is both the writer of its push of block b and the reader of the peer's,
// in the same launch, so the loop never waits in one role without serving the other:
//   - the reader: once the peer's flag has arrived, one check try per pass (r4d_archk_wait's, filling
//     the mirror). Pass: ack[b] = s into the peer's flags. Fail: after R4D_ARX_T_RQ from arrival and
//     at least two failed tries, a request rq[b] = (s << 32) | n into the peer's flags, then one
//     every R4D_ARX_T_RQ2; past the request limit, or R4D_ARCHK_DEADLINE after arrival, the
//     bounded failure of stage 1 (a fault record; the kernel ends, the engine stops);
//   - the writer: a request for (s, n) with n new re-pushes the block from the message, which the
//     kernel holds unreduced until this returns, and re-puts the check word with a release;
//   - both: the loop ends once this block's check passed (or gave up) AND the peer has acked this
//     rank's push: its ack equals s, or its flag for block b has passed s (it is in a later launch,
//     so its ack was lost: counted), or R4D_ARX_T_ACK after the check (counted). A block that gave
//     up leaves without waiting; it acks too, so the peer's writer leaves.
//   - re-stores: while waiting on the peer's ack, this rank's flag and ack again after R4D_ARX_T_RE,
//     doubling up to R4D_ARX_T_RE_MAX; before the peer's flag has arrived, the flag every
//     R4D_ARX_T_RE0. Each is counted.
// Acks and requests match s exactly (not >=), so a stale word, or a stray line, is ignored. A
// resend writes the same bytes and the same check word, so any mix of first and second landings is
// the push. The slot of launch s is not reused before launch s + 2, and neither rank leaves launch s
// before the peer has checked its push, so the double buffer's argument (above) is unchanged.
//
// Counters (R4D_AR2_CNT_*) are this launch's, added to block b's words by thread 0 at the end, only
// when nonzero: the healthy path stores nothing more than the ack. `op`'s own set of them
// (R4D_AR2_CNT_OP_OFF): one per collective (stages 2, 3 and 4).
//
// wall_clock64 ticks (100 MHz):
#define R4D_ARX_T_RE     100000ull     // 1 ms: the first re-store while waiting on the peer's ack
#define R4D_ARX_T_RE_MAX 25600000ull   // 256 ms: the re-store period's cap
#define R4D_ARX_T_RE0    10000000ull   // 100 ms: the flag's re-store before the peer's has arrived
#define R4D_ARX_T_RQ     20000ull      // 200 us from arrival to the first request
#define R4D_ARX_T_RQ2    100000ull     // 1 ms between requests
#define R4D_ARX_N_RQ     8u            // requests before the bounded failure (BRILLIANCE_AR_RESEND_MAX)
#define R4D_ARX_T_ACK    (R4D_ARCHK_DEADLINE + 100000000ull)   // 3 s: past the peer's own give-up
#define R4D_ARX_T_LATE   30000ull      // 300 us: injection mode late
// The pass's work, broadcast by thread 0 through LDS.
#define R4D_ARX_RESEND 1u
#define R4D_ARX_CHECK  2u
#define R4D_ARX_LEAVE  4u
#define R4D_ARX_LATE   8u
#define R4D_ARX_BAD    16u
#define R4D_ARX_PASSED 32u

// `hck` is thread 0's block sum (r4d_archk_put's return). `rehash(h)` as r4d_archk_wait's;
// `repush(stuck)` re-pushes every word of the block from the message (stuck: the injected line
// again lost); `late()` stores injection mode late's withheld line; `bad()` overwrites ctl-rq's line
// of this rank's own slot. With INJ false the last two are never called and `ij` is ignored.
// Returns true when this block's check passed, false when it gave up (the stage 1 bounded failure).
template <bool INJ, typename FH, typename FP, typename FL, typename FB>
__device__ __forceinline__ bool r4d_archk_exchange(unsigned int* my_flags, unsigned int* peer_flags,
                                                   unsigned int nbw, unsigned int slot,
                                                   unsigned int b, unsigned int s, unsigned int op,
                                                   unsigned int hck, int drain, const FH& rehash,
                                                   const FP& repush, const FL& late,
                                                   const FB& bad, const R4dArInj& ij) {
  /* Two command words by pass parity: thread 0 writes the next pass's before a slow thread has
   * necessarily read this one's, and the pass's barrier in between orders the reuse. */
  __shared__ unsigned int s_xcmd[2];
  __shared__ unsigned int s_xw[32];
  unsigned int par = 0u, cmd = 0u;
  const unsigned int tid = threadIdx.x, nt = blockDim.x;
  unsigned int* const ack_in = my_flags + R4D_AR2_ACK_OFF(nbw) + b;
  unsigned int* const ack_out = peer_flags + R4D_AR2_ACK_OFF(nbw) + b;
  unsigned long long* const rq_in =
      reinterpret_cast<unsigned long long*>(my_flags + R4D_AR2_RQ_OFF(nbw)) + b;
  unsigned long long* const rq_out =
      reinterpret_cast<unsigned long long*>(peer_flags + R4D_AR2_RQ_OFF(nbw)) + b;
  // Thread 0's state. The other threads keep only what they read from s_xcmd.
  bool arrived = false, checked = false, passed = false, acked = false, gone = false, fast = false;
  bool late_due = INJ && ij.late, bad_due = INJ && ij.blo < ij.bhi;
  bool skip_ack = INJ && ij.skip_ack, skip_rq = INJ && ij.skip_rq;
  unsigned int f = 0u, tries = 0u, n_srv = 0u, n_rq = 0u, rq_max = 0u, want = 0u;
  unsigned int c_rq = 0u, c_rs = 0u, c_re = 0u, c_ack = 0u;
  unsigned long long t0 = 0ull, t_arr = 0ull, t_chk = 0ull, t_re = 0ull, t_rq = 0ull, t_pk = 0ull;
  unsigned long long per = R4D_ARX_T_RE;
  if (tid == 0) { t0 = wall_clock64(); t_re = t0; t_pk = t0; }
  for (;; par ^= 1u) {
    if (tid == 0) {
      cmd = 0u;
      /* Three independent loads of pinned host memory, in flight together. */
      const unsigned int fl = r4d_ar_ld(my_flags + b);
      const unsigned long long rq = __hip_atomic_load(rq_in, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      const unsigned int ak = checked ? r4d_ar_ld(ack_in) : 0u;
      const unsigned long long now = wall_clock64();
#if defined(_WIN32)
      /* Once a wait has lasted R4D_AR_FAULT_PEEK, look every R4D_AR_FAULT_PEEK for a fault on record
       * on the instance: then every deadline below is R4D_AR_FAULT_FAST (stage 1's rule). */
      if (!fast && now - t_pk > R4D_AR_FAULT_PEEK) {
        t_pk = now;
        fast = r4d_ar_faulted(my_flags, nbw);
      }
#endif
      if (!arrived) {
        f = fl;
        if (!r4d_ar_seq_behind(fl, s)) {
          arrived = true;
          t_arr = now;
        } else if (now - t0 > (fast ? R4D_AR_FAULT_FAST : R4D_AR_FLAG_DEADLINE)) {
          /* Stage 1's flag give-up: recorded, and the check still runs (it passes if the bytes did
           * land; the record stands and the engine stops). The peer is out of step: fast. */
          (void)r4d_ar_fault_note(my_flags, nbw, R4D_AR_FAULT_FLAG, op, b, s, slot, 0u, 0u, s, fl,
                                  now - t0);
          arrived = true;
          fast = true;
          t_arr = now;
        } else if (now - t_re > R4D_ARX_T_RE0) {
          __hip_atomic_store(peer_flags + b, s, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
          ++c_re;
          t_re = now;
        }
      }
      /* The writer: a new request for this launch's push of block b. */
      if ((unsigned int)(rq >> 32) == s && (int)((unsigned int)rq - n_srv) > 0) {
        n_srv = (unsigned int)rq;
        cmd |= R4D_ARX_RESEND;
      }
      if (arrived && !checked) {
        cmd |= R4D_ARX_CHECK;
        if (INJ && bad_due) { bad_due = false; cmd |= R4D_ARX_BAD; }
      }
      if (checked && !acked) {
        if (ak == s || gone) {
          acked = true;
        } else if (!r4d_ar_seq_behind(fl, s + 1u)) {
          /* The peer is in a later launch on block b, so it left this one: its ack, stored before
           * that flag's release, was lost if it is not here now. Read again past the flag. */
          __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
          if (r4d_ar_ld(ack_in) != s) ++c_ack;
          acked = true;
        } else if (now - t_chk > (fast ? R4D_AR_FAULT_FAST : R4D_ARX_T_ACK)) {
          ++c_ack;
          acked = true;
        } else if (now - t_re > per) {
          /* The peer may wait on this rank's flag (it never arrived there) or its ack. */
          __hip_atomic_store(peer_flags + b, s, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
          r4d_ar_st(ack_out, s);
          ++c_re;
          t_re = now;
          per = per * 2ull < R4D_ARX_T_RE_MAX ? per * 2ull : R4D_ARX_T_RE_MAX;
        }
      }
      if (INJ && late_due && now - t0 > R4D_ARX_T_LATE) { late_due = false; cmd |= R4D_ARX_LATE; }
      if (checked && acked) cmd |= R4D_ARX_LEAVE | (passed ? R4D_ARX_PASSED : 0u);
      s_xcmd[par] = cmd;
    }
    __syncthreads();
    cmd = s_xcmd[par];
    if (cmd & (R4D_ARX_RESEND | R4D_ARX_LATE)) {
      if (cmd & R4D_ARX_RESEND) repush(INJ && ij.stuck);
      if (INJ && (cmd & R4D_ARX_LATE)) late();
      /* As the push: drained before thread 0's release, which writes the bytes back. */
      if (drain == 1) __threadfence_system();
      else asm volatile("s_wait_storecnt 0x0" ::: "memory");
      __syncthreads();
      if (tid == 0) {
        if (cmd & R4D_ARX_RESEND) {
          __hip_atomic_store(r4d_archk_word(peer_flags, nbw, slot, b), hck + s * R4D_ARCHK_SALT,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
          ++c_rs;
        } else {
          __threadfence_system();
        }
      }
    }
    if (cmd & R4D_ARX_CHECK) {
      if (INJ && (cmd & R4D_ARX_BAD)) {
        bad();
        __threadfence_system();
        __syncthreads();
      }
      /* Drop the vector caches, so this try reads the bytes from where the push lands; then the
       * check word first, so its round trip overlaps the re-hash's loads (r4d_archk_wait). */
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
      if (tid == 0) want = r4d_ar_ld(r4d_archk_word(my_flags, nbw, slot, b));
      unsigned int h = 0u;
      rehash(h);
      h = r4d_archk_wave_sum(h);
      if ((tid & 31u) == 0u) s_xw[tid >> 5] = h;
      __syncthreads();
      if (tid == 0) {
        unsigned int t = 0u;
        for (unsigned int w = 0; w < (nt + 31u) / 32u; ++w) t += s_xw[w];
        const unsigned long long now = wall_clock64();
        if (t + s * R4D_ARCHK_SALT == want) {
          checked = passed = true;
          t_chk = now;
          t_re = now;
          per = R4D_ARX_T_RE;
          if (INJ && skip_ack) skip_ack = false;
          else r4d_ar_st(ack_out, s);
        } else {
          ++tries;
          /* Stage 1's diagnostic words (r4d_archk_wait): the failed tries and this card's sum. */
          if ((tries & (tries - 1u)) == 0u || (tries & 1023u) == 0u) {
            unsigned int* const dw = my_flags + 3u * nbw + 2u * b;
            r4d_ar_st(dw + 1, t);
            r4d_ar_st(dw, tries);
          }
          const unsigned long long el = now - t_arr;
          bool give = el > (fast ? R4D_AR_FAULT_FAST : R4D_ARCHK_DEADLINE);
          if (!give && tries >= 2u && el > R4D_ARX_T_RQ && (n_rq == 0u || now - t_rq > R4D_ARX_T_RQ2)) {
            if (rq_max == 0u) {
              rq_max = r4d_ar_ld(my_flags + R4D_AR2_CTL_OFF(nbw) + R4D_AR2_CTL_RQMAX);
              if (rq_max == 0u) rq_max = R4D_ARX_N_RQ;
            }
            if (n_rq >= rq_max) {
              give = true;
            } else {
              ++n_rq;
              ++c_rq;
              t_rq = now;
              if (INJ && skip_rq) skip_rq = false;
              else __hip_atomic_store(rq_out, ((unsigned long long)s << 32) | n_rq, __ATOMIC_RELAXED,
                                      __HIP_MEMORY_SCOPE_SYSTEM);
            }
          }
          if (give) {
            /* The bounded failure (stage 1): recorded; the kernel reduces what the last try copied
             * to the mirror and ends. The ack lets the peer's writer leave now. */
            checked = gone = true;
            t_chk = now;
            const bool first = r4d_ar_fault_note(my_flags, nbw, R4D_AR_FAULT_CHECK, op, b, s, slot,
                                                 tries, t + s * R4D_ARCHK_SALT, want, f, el);
            r4d_ar_st(ack_out, s);
            if (first) r4d_ar_fault_park(my_flags, nbw);
          }
        }
      }
      __syncthreads();   // s_xw is rewritten by the next try
    }
    if (cmd & R4D_ARX_LEAVE) break;
    if (!(cmd & (R4D_ARX_CHECK | R4D_ARX_RESEND))) __builtin_amdgcn_s_sleep(1);
  }
  if (tid == 0 && (c_rq | c_rs | c_re | c_ack)) {
    unsigned int* const k = my_flags + R4D_AR2_CNT_OP_OFF(nbw, op) + 4u * b;
    if (c_rq) r4d_ar_st(k + R4D_AR2_CNT_RQ, r4d_ar_ld(k + R4D_AR2_CNT_RQ) + c_rq);
    if (c_rs) r4d_ar_st(k + R4D_AR2_CNT_RS, r4d_ar_ld(k + R4D_AR2_CNT_RS) + c_rs);
    if (c_re) r4d_ar_st(k + R4D_AR2_CNT_RE, r4d_ar_ld(k + R4D_AR2_CNT_RE) + c_re);
    if (c_ack) r4d_ar_st(k + R4D_AR2_CNT_ACK, r4d_ar_ld(k + R4D_AR2_CNT_ACK) + c_ack);
  }
  return (cmd & R4D_ARX_PASSED) != 0u;
}

// ---- The handshake of a kernel whose threads each own at most one word of the block's message
// (a decode or verify step: r4d_ar_hc_write_reg_kernel, r4d_ar_oneshot_2rank_exact.hip). Such a
// kernel is a chain of dependent round trips, and this is the plain kernels' protocol -- the same
// check word, the same flag value, a pass only behind the peer's flag and on an agreeing sum --
// with the chain shortened:
//
//  * Barriers order LDS only (r4d_lds_barrier). A __syncthreads also waits for every outstanding
//    load and store and invalidates the vector caches: 0.26 us against 0.03 on this part, five
//    times a launch in the plain kernels.
//  * The flag poll is a relaxed system-scope load. As an acquire load every poll is followed by a
//    system-scope cache invalidate (0.35 us, and whatever the card had cached for the kernels
//    behind goes with it), for as long as the peer is late. The slot is fine-grained memory, read
//    past the caches, and the push check re-reads behind a system-scope acquire whenever the
//    bytes do not agree, exactly as r4d_archk_wait does.
//  * `push` stores the thread's word and `rehash` loads the peer's into the caller's registers, so
//    the reduce reads nothing again.
//
// drain 4 is the plain kernels' test mode (r4d_ar_late_push_pin): the flag ahead of the push and
// the push ~20 us behind it.
#define R4D_ARW_SPIN_MAX 4000000000ULL

// A barrier that orders LDS only.
__device__ __forceinline__ void r4d_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// The wake word of a 2-rank instance: the word behind the flags and the two slots' check words
// (r4d_ar_entry.hip sizes the allocation). Nothing reads it.
//
// Brilliance: behind the push check's diagnostic words as well ([3 nbw, 5 nbw), r4d_archk_wait),
// so at 5 nbw rather than upstream's 3 nbw, which is block 0's diagnostic word here. Only the
// register form stores it, and that form runs on P2P instances only; a two-rank host mailbox's
// flags hold the acks at 5 nbw (R4D_AR2_ACK_OFF) and never see a wake.
//
// The path between two cards idles when nothing crosses it. On 2x R9700 behind an EPYC's two root
// complexes, with each card's own link busy all the while: a store into the peer's memory more
// than ~20 us after the last one becomes visible there ~4 us later than one behind recent traffic,
// and the path is back at full speed ~2-4 us after that first store. A step's all-reduces are
// ~100 us apart, and a handshake's last store is its flag, so the whole of that wait lands on the
// flag: both ranks sit in their polls for it (5.9 us a mixer all-reduce of Qwen3.8-Flash-Next
// where neither rank was late, against 2.4 behind a store 2-20 us earlier). A block's first lane
// stores into the wake word before it waits for its turn, which is as early as the kernel knows it
// is about to push: the path wakes while the block waits, takes its counters and pushes.
__device__ __forceinline__ void r4d_ar_link_wake(unsigned int* peer_flags, unsigned int nbw) {
  if (threadIdx.x == 0)
    __hip_atomic_store(&peer_flags[5u * nbw], blockIdx.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename P, typename F>
__device__ __forceinline__ void r4d_ar_word_handshake(unsigned int* peer_flags,
                                                      unsigned int* my_flags, unsigned int nbw,
                                                      unsigned int slot, unsigned int b,
                                                      unsigned int seq, int drain, const P& push,
                                                      const F& rehash) {
  __shared__ unsigned int s_wput[32];
  __shared__ unsigned int s_wchk[32];
  __shared__ unsigned int s_wok;
  const unsigned int tid = threadIdx.x, nt = blockDim.x, nw = (nt + 31u) / 32u;
  const bool late = drain == 4;
  if (late) {
    if (tid == 0)
      __hip_atomic_store(&peer_flags[b], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < 2000ull) __builtin_amdgcn_s_sleep(8);   // 100 MHz: 20 us
  }
  unsigned int hck = 0u;
  push(hck);
  hck = r4d_archk_wave_sum(hck);
  if ((tid & 31u) == 0u) s_wput[tid >> 5] = hck;
  /* Every wave's pushes are out before the one barrier in front of the flag (the drain modes of
   * the plain kernels; 2 is thread 0's fence below). */
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  r4d_lds_barrier();
  if (tid == 0) {
    unsigned int t = 0u;
    for (unsigned int w = 0; w < nw; ++w) t += s_wput[w];
    unsigned int* const cw = r4d_archk_word(peer_flags, nbw, slot, b);
    if (late) {
      /* Nothing else writes the pushes back before the kernel ends: a release (r4d_archk_put). */
      __hip_atomic_store(cw, t + seq * R4D_ARCHK_SALT, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    } else {
      /* The flag's release writes the pushes back, so the check word itself can be relaxed. */
      __hip_atomic_store(cw, t + seq * R4D_ARCHK_SALT, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      if (drain == 2) __threadfence_system();
      __hip_atomic_store(&peer_flags[b], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    unsigned long long z = 0;
    /* A handshake that cannot complete is a lockstep violation; trap rather than sum stale bytes. */
    while (r4d_ar_seq_behind(__hip_atomic_load(&my_flags[b], __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_SYSTEM), seq))
      if (++z > R4D_ARW_SPIN_MAX) R4D_AR_GIVE_UP();
  }
  r4d_lds_barrier();
  const unsigned long long t0 = tid == 0 ? wall_clock64() : 0ull;
  for (;;) {
    /* The check word first, so its round trip overlaps the re-hash's loads (r4d_archk_wait). */
    const unsigned int want = tid == 0 ? __hip_atomic_load(r4d_archk_word(my_flags, nbw, slot, b),
                                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                                       : 0u;
    unsigned int h = 0u;
    rehash(h);
    h = r4d_archk_wave_sum(h);
    if ((tid & 31u) == 0u) s_wchk[tid >> 5] = h;
    r4d_lds_barrier();
    if (tid == 0) {
      unsigned int t = 0u;
      for (unsigned int w = 0; w < nw; ++w) t += s_wchk[w];
      const bool ok = t + seq * R4D_ARCHK_SALT == want;
      if (!ok && wall_clock64() - t0 > R4D_ARCHK_DEADLINE) R4D_AR_GIVE_UP();
      s_wok = ok ? 1u : 0u;
    }
    r4d_lds_barrier();
    if (s_wok) return;
    r4d_lds_barrier();   // s_wchk and s_wok are rewritten by the next try
    /* Drop the vector caches, so the next try reads the bytes from where the push lands. */
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    __builtin_amdgcn_s_sleep(2);
  }
}

// ---- The host mailbox's one read. A host twin's slots are pinned host memory (r4d_ar_entry.hip),
// which the card reads past every cache over its own link, so reading the slot for the check and
// again for the reduce crosses the link twice: at --p2p off a prefill all-reduce's reads doubled
// (#323). Such an instance also has a mirror of its slots in this rank's VRAM, `md` bytes from
// them (`stage` - my_scratch; 0 over P2P, where the slots are VRAM already). The re-hash reads each
// word through r4d_archk_keep, which also stores it into the mirror, and once the check passes the
// kernel moves its slot pointers onto the mirror (r4d_archk_moved) for the reduce. Only this block
// writes and reads its words of the mirror, and the check's barriers order the two; a failed try's
// copy is overwritten by the next.
template <typename V>
__device__ __forceinline__ V r4d_archk_keep(const V* p, long long md) {
  const V v = *p;
  if (md) *reinterpret_cast<V*>(reinterpret_cast<uintptr_t>(p) + md) = v;
  return v;
}
template <typename V>
__device__ __forceinline__ const V* r4d_archk_moved(const V* p, long long md) {
  return reinterpret_cast<const V*>(reinterpret_cast<uintptr_t>(p) + md);
}
// The offset of an instance's mirror from its slots, from the launcher's `stage` (0: none).
__device__ __forceinline__ long long r4d_archk_md(long long stage, const void* my_scratch) {
  return stage ? stage - (long long)reinterpret_cast<uintptr_t>(my_scratch) : 0;
}
