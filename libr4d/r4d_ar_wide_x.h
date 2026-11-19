#pragma once
// r4d_ar_wide_x.h: Brilliance, the loud stop of the wide (four-rank) host mailboxes, stage W1 of
// the four-rank resend. The two-rank host
// twins stop loudly on a wrong or missing push (fix C1 stage 1, r4d_ar_seq.h); before this the wide
// family had no push check, so a wrong line was summed silently, and a lost flag store spun for ever
// (R4D_AR_GIVE_UP is a no-op on Windows). The wide host kernels (r4d_ar_wide_host.hip) now:
//   - put a check word per (phase, parity, block, source) beside every push: the block's sum of
//     r4d_archk_mix over the words it pushed, salted by the handshake value (the two-rank check's
//     hash; BRILLIANCE_AR4_CHECK's words, which this retires);
//   - bound every flag wait (each spin lane its own wall clock) and every push check;
//   - check each peer's block BEFORE the reduce, copying what the check read into the instance's VRAM
//     mirror, and reduce from the mirror, so a check that fails has not yet been summed;
//   - on a deadline, write a fault record (the two-rank record: R4D_ARF_*) into the instance's own
//     flags; the engine reads it after every step (r4d_ar_fault_poll) and stops, and the process
//     exits with status 70, as at two ranks.
// The helpers live here, not in r4d_ar_seq.h, so the wide work adds no conflict there with upstream.
// Only the wide host kernels include this; the default wide kernels (P2P, Linux) are unchanged.
#include <hip/hip_runtime.h>
#include "r4d_ar_seq.h"
#include "r4d_ar_wide.h"

// ---- A wide host instance's flags allocation, in words (ws ranks, nbw = R4D_AR_WIDE_MAX_BLOCKS):
//
//   [0, ws nbw)              handshake flags [b][src]                          (peer src)
//   [ws nbw, +4 nbw)         diagnostic words, kept from the old layout        (unused)
//   [CK,  +4 ws nbw)         check words [phase 2][parity 2][b][src]           (peer src)
//   [ACK, +2 ws nbw)         W2: ack[phase][b][src] (r4d_arw_exchange)         (peer src)
//   [RQ,  +4 ws nbw)         W2: rq[phase][b][src], 64-bit                     (peer src)
//   [SRC, +3 ws nbw)         per (b, src): [0 ..) give-ups, [ws nbw ..) late passes, [2 ws nbw ..)
//                            W2's requests                                     (this rank)
//   [CNT, +8 ops 8 nbw)      per op (R4D_AR_OP_W_*), per block, 8 words (R4D_ARW_CNT_*)  (this rank)
//   [REC, +32)               the fault record, copies A and B, one 64-byte line each (this rank)
//   [CTL, +16)               R4D_ARW_CTL_*: park (unused at width), release, fault count, W2's request
//                            limit, effective injections                       (this rank / host)
//
// Words [0, CK + 4 ws nbw) are BRILLIANCE_AR4_CHECK's layout, kept (design 3.2). W1 reserved the ACK
// and RQ words and the W2 counters; the exchange (W2, below) writes them. Every word a peer
// writes is a coherent 4-byte (8-byte) system-scope store, so lines need no padding. Counters and the
// record are written by this rank's block b's thread 0 only, with a plain load and store.
#define R4D_ARW_CK_OFF(ws, nbw)   ((unsigned)(ws) * (nbw) + 4u * (nbw))
#define R4D_ARW_ACK_OFF(ws, nbw)  (R4D_ARW_CK_OFF(ws, nbw) + 4u * (unsigned)(ws) * (nbw))
#define R4D_ARW_RQ_OFF(ws, nbw)   (R4D_ARW_ACK_OFF(ws, nbw) + 2u * (unsigned)(ws) * (nbw))
#define R4D_ARW_SRC_OFF(ws, nbw)  (R4D_ARW_RQ_OFF(ws, nbw) + 4u * (unsigned)(ws) * (nbw))
#define R4D_ARW_CNT_OFF(ws, nbw)  (R4D_ARW_SRC_OFF(ws, nbw) + 3u * (unsigned)(ws) * (nbw))
#define R4D_ARW_REC_OFF(ws, nbw)  ((R4D_ARW_CNT_OFF(ws, nbw) + 64u * (nbw) + 15u) & ~15u)
#define R4D_ARW_CTL_OFF(ws, nbw)  (R4D_ARW_REC_OFF(ws, nbw) + 2u * R4D_ARF_WORDS)
#define R4D_ARW_FLAG_WORDS(ws, nbw) (R4D_ARW_CTL_OFF(ws, nbw) + 16u)
// [SRC] sub-arrays, each [b][src].
#define R4D_ARW_SRC_GU   0u   // give-ups against source src (this rank, the reader)
#define R4D_ARW_SRC_LATE 1u   // checks that failed first and then passed: a late landing
#define R4D_ARW_SRC_RQ   2u   // W2: requests raised
// [CNT] words per (op, block).
#define R4D_ARW_CNT_GU   0u
#define R4D_ARW_CNT_LATE 1u
#define R4D_ARW_CNT_RQ   2u   // W2 (requests raised, resends served, re-stores, acks that never came)
#define R4D_ARW_CNT_RS   3u
#define R4D_ARW_CNT_RE   4u
#define R4D_ARW_CNT_ACK  5u
// [CTL] words.
#define R4D_ARW_CTL_PARK    0u
#define R4D_ARW_CTL_RELEASE 1u
#define R4D_ARW_CTL_FAULTS  2u
#define R4D_ARW_CTL_RQMAX   3u
#define R4D_ARW_CTL_INJEFF  4u   // injections whose damage changed bytes (BRILLIANCE_AR_INJECT)
#define R4D_ARW_CTL_INJEFF2 5u   // W2: the same, counted by pair's partner block (a second block of the
                                 // launch: one writer a word, so no increment is lost to the race)
// The wide collectives as the record and the counters name them (design 2.4). W1 builds the stop for
// ONESHOT, TWOSHOT (plain), HC, GHC and GATHER; the others are reserved for W4 and W6.
#define R4D_AR_OP_W_ONESHOT 16u   // ar_oneshot_4rank_exact
#define R4D_AR_OP_W_TWOSHOT 17u   // ar_twoshot_4rank_exact (plain), also the one-shot row's ts_min band
#define R4D_AR_OP_W_CODED   18u
#define R4D_AR_OP_W_TI8     19u
#define R4D_AR_OP_W_HC      20u   // ar_hc_write_4rank
#define R4D_AR_OP_W_GHC     21u   // ar_gather_hc_write_4rank
#define R4D_AR_OP_W_GATHER  22u   // ar_gather_4rank
#define R4D_AR_OP_W_NORM    23u
#define R4D_ARW_OPS 8u
// The entries of r4d_ar_mailbox_counters (r4d_ar_entry.hip): the two-rank host twins' 31, then W1's
// 22 (to 53), then the four-rank exchange's 28 (W2/W3, to 81), then the coded two-shot's 6 (W4, to 87).
#define R4D_AR_MB_COUNTERS 87
__host__ __device__ __forceinline__ unsigned int r4d_arw_op_index(unsigned int op) {
  return op >= R4D_AR_OP_W_ONESHOT && op < R4D_AR_OP_W_ONESHOT + R4D_ARW_OPS ? op - R4D_AR_OP_W_ONESHOT
                                                                             : 0u;
}

// The check word of (phase, parity, block b, source src) in a flags allocation.
__device__ __forceinline__ unsigned int* r4d_arw_ck(unsigned int* flags, unsigned int ws,
                                                    unsigned int nbw, unsigned int phase,
                                                    unsigned int parity, unsigned int b,
                                                    unsigned int src) {
  return flags + R4D_ARW_CK_OFF(ws, nbw) + ((phase * 2u + parity) * nbw + b) * ws + src;
}

// Whether the instance already has a fault on record (its count, or either copy's magic).
__device__ __forceinline__ bool r4d_arw_faulted(unsigned int* flags, unsigned int ws, unsigned int nbw) {
  const unsigned int* rec = flags + R4D_ARW_REC_OFF(ws, nbw);
  return r4d_ar_ld(flags + R4D_ARW_CTL_OFF(ws, nbw) + R4D_ARW_CTL_FAULTS) != 0u ||
         r4d_ar_ld(rec + R4D_ARF_MAGIC) == R4D_AR_FAULT_MAGIC ||
         r4d_ar_ld(rec + R4D_ARF_WORDS + R4D_ARF_MAGIC) == R4D_AR_FAULT_MAGIC;
}

// Thread 0 only, past a deadline: count a give-up against source `src` (instance, block-and-source,
// op) and, if the instance has none on record yet, write the record (both copies; R13 the source
// rank, R14 = (phase << 8) | ws). A system-scope fence then orders the record before this block's
// next store to a peer. Returns whether this give-up wrote the record.
inline __device__ __noinline__ bool r4d_arw_fault_note(unsigned int* flags, unsigned int ws,
                                                       unsigned int nbw, unsigned int kind,
                                                       unsigned int op, unsigned int b,
                                                       unsigned int seq, unsigned int slot,
                                                       unsigned int tries, unsigned int got,
                                                       unsigned int want, unsigned int flag,
                                                       unsigned long long ticks, unsigned int src,
                                                       unsigned int phase) {
  unsigned int* const ctl = flags + R4D_ARW_CTL_OFF(ws, nbw);
  unsigned int* const rec = flags + R4D_ARW_REC_OFF(ws, nbw);
  unsigned int* const gs = flags + R4D_ARW_SRC_OFF(ws, nbw) + R4D_ARW_SRC_GU * ws * nbw + b * ws + src;
  unsigned int* const go = flags + R4D_ARW_CNT_OFF(ws, nbw) + (r4d_arw_op_index(op) * nbw + b) * 8u +
                           R4D_ARW_CNT_GU;
  const bool first = !r4d_arw_faulted(flags, ws, nbw);
  const unsigned int count = r4d_ar_ld(ctl + R4D_ARW_CTL_FAULTS) + 1u;
  r4d_ar_st(ctl + R4D_ARW_CTL_FAULTS, count);
  r4d_ar_st(gs, r4d_ar_ld(gs) + 1u);
  r4d_ar_st(go, r4d_ar_ld(go) + 1u);
  if (first) {
    unsigned int w[R4D_ARF_WORDS];
    w[R4D_ARF_MAGIC] = R4D_AR_FAULT_MAGIC; w[R4D_ARF_KIND] = kind; w[R4D_ARF_OP] = op;
    w[R4D_ARF_BLOCK] = b; w[R4D_ARF_SEQ] = seq; w[R4D_ARF_SLOT] = slot; w[R4D_ARF_TRIES] = tries;
    w[R4D_ARF_GOT] = got; w[R4D_ARF_WANT] = want; w[R4D_ARF_FLAG] = flag; w[R4D_ARF_NB] = gridDim.x;
    w[R4D_ARF_COUNT] = count; w[R4D_ARF_US] = (unsigned int)(ticks / 100ull);
    w[R4D_ARF_R13] = src; w[R4D_ARF_R14] = (phase << 8) | ws;
    unsigned int sum = R4D_AR_FAULT_SUM_SEED;
    for (unsigned int i = 0; i < R4D_ARF_SUM; ++i) sum += r4d_archk_mix(w[i], i);
    w[R4D_ARF_SUM] = sum;
    for (unsigned int c = 0; c < 2u; ++c)
      for (unsigned int i = 0; i < R4D_ARF_WORDS; ++i) r4d_ar_st(rec + c * R4D_ARF_WORDS + i, w[i]);
  }
  __threadfence_system();
  return first;
}

// Thread 0 only, at the end of a block's check: `n` late passes against source `src`.
inline __device__ __noinline__ void r4d_arw_note_late(unsigned int* flags, unsigned int ws,
                                                      unsigned int nbw, unsigned int op,
                                                      unsigned int b, unsigned int src,
                                                      unsigned int n) {
  unsigned int* const ls = flags + R4D_ARW_SRC_OFF(ws, nbw) + R4D_ARW_SRC_LATE * ws * nbw + b * ws + src;
  unsigned int* const lo = flags + R4D_ARW_CNT_OFF(ws, nbw) + (r4d_arw_op_index(op) * nbw + b) * 8u +
                           R4D_ARW_CNT_LATE;
  r4d_ar_st(ls, r4d_ar_ld(ls) + n);
  r4d_ar_st(lo, r4d_ar_ld(lo) + n);
}

// The block sums of NP per-thread values, to thread 0 (`t`); the other threads' `t` is undefined.
// Two barriers, so it may be called again at once.
template <int NP>
__device__ __forceinline__ void r4d_arw_bsum(const unsigned int (&h)[NP], unsigned int (&t)[NP]) {
  __shared__ unsigned int s_wpart[NP][32];
  const unsigned int tid = threadIdx.x, nw = (blockDim.x + 31u) / 32u;
#pragma unroll
  for (int p = 0; p < NP; ++p) {
    const unsigned int x = r4d_archk_wave_sum(h[p]);
    if ((tid & 31u) == 0u) s_wpart[p][tid >> 5] = x;
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < NP; ++p) {
      unsigned int x = 0u;
      for (unsigned int w = 0; w < nw; ++w) x += s_wpart[p][w];
      t[p] = x;
    }
  }
  __syncthreads();
}

// Writer: every thread calls this after its pushes, with its share `h[p]` of the sum over the words
// it pushed to peer p. Thread 0 stores, for each peer, sum + hs * SALT into that peer's check word
// (phase, parity, b, rank). Relaxed: the handshake's release, right after, writes the pushed bytes
// and these words back; the reader re-reads until bytes and word agree.
template <int WS, int NP>
__device__ __forceinline__ void r4d_arw_put(const R4DArWidePeers<int4, NP>& peers, int b, int rank,
                                            unsigned int phase, unsigned int parity,
                                            const unsigned int (&h)[NP], unsigned int hs) {
  unsigned int t[NP];
  r4d_arw_bsum<NP>(h, t);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int p = 0; p < NP; ++p)
      __hip_atomic_store(r4d_arw_ck(peers.flags[p], WS, R4D_AR_WIDE_MAX_BLOCKS, phase, parity,
                                    (unsigned)b, (unsigned)rank),
                         t[p] + hs * R4D_ARCHK_SALT, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// The bounded handshake (design 2.2): r4d_ar_handshake's publish and per-source spin lanes, with each
// lane's wait ended by its own wall clock: R4D_AR_FLAG_DEADLINE (30 s), or R4D_AR_FAULT_FAST (10 ms)
// once the instance has a fault on record (looked at after R4D_AR_FAULT_PEEK). The clock is read
// every 64th failed load, so a flag that has arrived costs the one load it always did. A lane that
// expires sets its bit in LDS; after the barrier thread 0 records a FLAG fault naming that source
// (the record writer must be one thread). The block goes on: the push check decides about the bytes.
// `skip` (injection noflag only, else 0): a mask of peer indices whose flag this block does not store.
template <int WS, int NP>
__device__ __forceinline__ void r4d_arw_handshake(const R4DArWidePeers<int4, NP>& peers,
                                                  unsigned int* __restrict__ my_flags, int b,
                                                  int rank, unsigned int val, int drain, int acq,
                                                  int pub, unsigned int skip, unsigned int op,
                                                  unsigned int phase, unsigned int slot) {
  __shared__ unsigned int s_gone, s_fv[NP], s_us[NP];
  constexpr unsigned int nbw = R4D_AR_WIDE_MAX_BLOCKS;
  const int tid = threadIdx.x;
  if (tid == 0) s_gone = 0u;
  r4d_ar_drain(drain);
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    if (pub) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
#pragma unroll
      for (int p = 0; p < NP; ++p)
        if (!((skip >> p) & 1u)) r4d_ar_store_sys_rlx(&peers.flags[p][(size_t)b * WS + rank], val);
    } else {
#pragma unroll
      for (int p = 0; p < NP; ++p)
        if (!((skip >> p) & 1u)) r4d_ar_store_sys_rel(&peers.flags[p][(size_t)b * WS + rank], val);
    }
  }
  if (tid < NP) {
    const unsigned int* f = &my_flags[(size_t)b * WS + peers.rank[tid]];
    unsigned long long z = 0, t0 = 0, dl = R4D_AR_FLAG_DEADLINE;
    bool peeked = false;
    unsigned int fv;
    while ((int)((fv = r4d_ar_load_sys_rlx(f)) - val) < 0) {
      __builtin_amdgcn_s_sleep(2);
      if ((++z & 63ull) != 0ull) continue;
      const unsigned long long now = wall_clock64();
      if (t0 == 0ull) { t0 = now; continue; }
      if (!peeked && now - t0 > R4D_AR_FAULT_PEEK) {
        peeked = true;
        if (r4d_arw_faulted(my_flags, WS, nbw)) dl = R4D_AR_FAULT_FAST;
      }
      if (now - t0 > dl) {
        s_fv[tid] = fv;
        s_us[tid] = (unsigned int)((now - t0) / 100ull);
        atomicOr(&s_gone, 1u << tid);
        break;
      }
    }
  }
  __syncthreads();
  if (tid == 0 && s_gone) {
    for (int p = 0; p < NP; ++p)
      if ((s_gone >> p) & 1u)
        (void)r4d_arw_fault_note(my_flags, WS, nbw, R4D_AR_FAULT_FLAG, op, (unsigned)b, val, slot,
                                 0u, 0u, val, s_fv[p], (unsigned long long)s_us[p] * 100ull,
                                 (unsigned)peers.rank[p], phase);
  }
  r4d_ar_acquire(acq);
}

// The push check before use (design 2.3): every thread calls this after the handshake. `rehash(mask,
// h)` adds, for each peer index p in `mask`, this thread's share of the hash over the words it reads
// of peer p's block, storing each word where the kernel then reads it (the VRAM mirror, or the
// output for a copy). One try reads every peer still unchecked; a peer that passes is not read
// again. Thread 0 loads the check words before the hashes, so their round trips overlap. A peer that
// has not agreed R4D_ARCHK_DEADLINE (2 s) after the handshake, or R4D_AR_FAULT_FAST once the
// instance has a fault on record, is given up: thread 0 records a CHECK fault naming its source, and
// the kernel goes on with what the last try stored. A pass after a failed try is a late landing,
// counted per source. Each retry starts with a system-scope acquire, so it reads memory, not a stale
// line of this card's own cache.
template <int WS, int NP, typename F>
__device__ __forceinline__ void r4d_arw_check(unsigned int* my_flags, const R4DArWidePeers<int4, NP>& peers,
                                              unsigned int phase, unsigned int parity, int b,
                                              unsigned int hs, unsigned int op, const F& rehash) {
  __shared__ unsigned int s_cmask;
  constexpr unsigned int nbw = R4D_AR_WIDE_MAX_BLOCKS;
  constexpr unsigned int full = (1u << NP) - 1u;
  const unsigned int tid = threadIdx.x;
  unsigned int mask = full;
  unsigned int tries[NP];
#pragma unroll
  for (int p = 0; p < NP; ++p) tries[p] = 0u;
  const unsigned long long t0 = tid == 0 ? wall_clock64() : 0ull;
  unsigned long long dl = R4D_ARCHK_DEADLINE;
  bool peeked = false;
  for (;;) {
    unsigned int want[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p) want[p] = 0u;
    if (tid == 0) {
#pragma unroll
      for (int p = 0; p < NP; ++p)
        want[p] = ((mask >> p) & 1u)
                      ? r4d_ar_ld(r4d_arw_ck(my_flags, WS, nbw, phase, parity, (unsigned)b,
                                             (unsigned)peers.rank[p]))
                      : 0u;
    }
    unsigned int h[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p) h[p] = 0u;
    rehash(mask, h);
    unsigned int t[NP];
    r4d_arw_bsum<NP>(h, t);
    if (tid == 0) {
      const unsigned long long now = wall_clock64(), el = now - t0;
      if (!peeked && el > R4D_AR_FAULT_PEEK) {
        peeked = true;
        if (r4d_arw_faulted(my_flags, WS, nbw)) dl = R4D_AR_FAULT_FAST;
      }
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        if (!((mask >> p) & 1u)) continue;
        const unsigned int got = t[p] + hs * R4D_ARCHK_SALT;
        if (got == want[p]) {
          mask &= ~(1u << p);
          if (tries[p]) r4d_arw_note_late(my_flags, WS, nbw, op, (unsigned)b, (unsigned)peers.rank[p], 1u);
        } else {
          ++tries[p];
          if (el > dl) {
            mask &= ~(1u << p);
            (void)r4d_arw_fault_note(my_flags, WS, nbw, R4D_AR_FAULT_CHECK, op, (unsigned)b, hs,
                                     parity, tries[p], got, want[p],
                                     r4d_ar_ld(my_flags + (size_t)b * WS + peers.rank[p]), el,
                                     (unsigned)peers.rank[p], phase);
          }
        }
      }
      s_cmask = mask;
    }
    __syncthreads();
    mask = s_cmask;
    if (!mask) return;
    __syncthreads();   // s_cmask is rewritten by the next try
    /* Drop the vector caches (system scope), so the next try reads where the push lands. */
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    __builtin_amdgcn_s_sleep(2);
  }
}

// ---- Fault injection at width (design 5.2; BRILLIANCE_AR_INJECT=<op>:<mode>:<every>[:<rank>[:<peer>]],
// r4d_ar_entry.hip). The launcher passes the kernel a word: mode (R4D_AR_INJ_LOSE, _DUP, _STUCK,
// _NOFLAG) in bits 0-3, the block in 4-9, the target peer's INDEX (0..ws-2, in this rank's peer
// list) in 10-12, the phase (two-shot hop 1 = 0, hop 2 = 1) in 13, the line offset in 14-21. Only the
// INJ instantiations read it; the production kernels never see it.
//
// The all-zero trap (design 5.2): a lost line keeps the slot's bytes from two launches earlier, and
// damages nothing where those equal the new bytes. So before its push, thread 0 of the chosen block
// reads up to 8 candidate lines of the target peer's slot and picks the first whose damage changes
// bytes (lose: the old bytes differ from the new; dup: the copied line differs). It counts an
// effective injection in its own instance's control line. A line is a 64-byte line of the address
// space; a line only partly inside the block damages the block's words in it (sub-line blocks).
#define R4D_ARW_INJ_TRIES 8
__host__ __device__ __forceinline__ unsigned int r4d_arw_inj_word(unsigned int mode, unsigned int blk,
                                                                  unsigned int peer, unsigned int phase,
                                                                  unsigned int off) {
  return (mode & 15u) | ((blk & 63u) << 4) | ((peer & 7u) << 10) | ((phase & 1u) << 13) |
         ((off & 255u) << 14);
}
struct R4dArwInj {
  int peer;            // the damaged push's peer index, -1 for none
  int lo, hi, delta;   // words [lo, hi) of the block pushed wrong; dup: from [lo - delta, hi - delta)
  int start, end;      // the block's words: a dup word whose source falls outside is lost instead
  bool lose;           // lose (or stuck): not stored; else dup
  unsigned int noflag; // a mask of peer indices whose flag is not stored
};
// Thread 0 of every block calls it (the others get the result through LDS from the kernel). `msg(k)`
// is the word k this rank pushes to every peer in this phase; `slot` the target peer's slot base;
// [start, end) the block's words. Reads the candidate lines' old bytes over the link: injected
// launches only.
template <typename FM>
__device__ __forceinline__ R4dArwInj r4d_arw_inj_plan(unsigned int iw, int b, unsigned int phase,
                                                      int start, int end, const int4* slot,
                                                      unsigned int* my_ctl, const FM& msg) {
  R4dArwInj r{ -1, 0, 0, 0, start, end, true, 0u };
  const unsigned int mode = iw & 15u;
  if (!mode || (int)((iw >> 4) & 63u) != b || ((iw >> 13) & 1u) != phase || end <= start) return r;
  const int peer = (int)((iw >> 10) & 7u);
  if (mode == R4D_AR_INJ_NOFLAG) {
    r.noflag = 1u << peer;
    r4d_ar_st(my_ctl + R4D_ARW_CTL_INJEFF, r4d_ar_ld(my_ctl + R4D_ARW_CTL_INJEFF) + 1u);
    return r;
  }
  if (mode != R4D_AR_INJ_LOSE && mode != R4D_AR_INJ_DUP && mode != R4D_AR_INJ_STUCK) return r;
  const int ph = (int)(((uintptr_t)slot >> 4) & 3u);   // the slot's word 0 within its line
  const int L0 = (start + ph) >> 2, L1 = (end - 1 + ph) >> 2, nl = L1 - L0 + 1;
  const int off = (int)((iw >> 14) & 255u);
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  int pick = -1, plo = 0, phi = 0, pdelta = 0;
  for (int c = 0; c < R4D_ARW_INJ_TRIES && c < nl; ++c) {
    const int L = L0 + (off + c) % nl;
    const int lo = 4 * L - ph > start ? 4 * L - ph : start;
    const int hi = 4 * L - ph + 4 < end ? 4 * L - ph + 4 : end;
    int delta = 0;
    if (mode == R4D_AR_INJ_DUP) {
      const int S = L - R4D_AR_INJ_DUP_LINES >= L0 ? L - R4D_AR_INJ_DUP_LINES
                  : L > L0 ? L0 : (L1 > L ? L1 : L);
      delta = 4 * (L - S);
    }
    bool eff = false;
    for (int k = lo; k < hi; ++k) {
      const int4 want = msg(k);
      int4 got;
      if (mode == R4D_AR_INJ_DUP && delta != 0 && k - delta >= start && k - delta < end) {
        got = msg(k - delta);
      } else {
        const unsigned int* q = reinterpret_cast<const unsigned int*>(slot + k);
        got.x = (int)r4d_ar_ld(q); got.y = (int)r4d_ar_ld(q + 1);
        got.z = (int)r4d_ar_ld(q + 2); got.w = (int)r4d_ar_ld(q + 3);
      }
      eff = eff || got.x != want.x || got.y != want.y || got.z != want.z || got.w != want.w;
    }
    if (pick < 0 || eff) { pick = L; plo = lo; phi = hi; pdelta = delta; }
    if (eff) {
      r4d_ar_st(my_ctl + R4D_ARW_CTL_INJEFF, r4d_ar_ld(my_ctl + R4D_ARW_CTL_INJEFF) + 1u);
      break;
    }
  }
  if (pick < 0) return r;
  r.peer = peer; r.lo = plo; r.hi = phi; r.delta = pdelta;
  r.lose = mode != R4D_AR_INJ_DUP || pdelta == 0;
  return r;
}
// The plan, from thread 0 to the block: the kernel calls this where every thread can wait, before
// its push. Two barriers.
template <typename FM>
__device__ __forceinline__ R4dArwInj r4d_arw_inj_bcast(unsigned int iw, int b, unsigned int phase,
                                                       int start, int end, const int4* slot,
                                                       unsigned int* my_ctl, const FM& msg) {
  __shared__ R4dArwInj s_inj;
  if (threadIdx.x == 0) s_inj = r4d_arw_inj_plan(iw, b, phase, start, end, slot, my_ctl, msg);
  __syncthreads();
  const R4dArwInj r = s_inj;
  __syncthreads();
  return r;
}
// One store of an injected launch's push: word k of value v to peer index p's slot `dst`.
template <typename FM>
__device__ __forceinline__ void r4d_arw_inj_put(const R4dArwInj& ij, int p, int4* dst, int k,
                                                const int4& v, const FM& msg) {
  if (p != ij.peer || k < ij.lo || k >= ij.hi) { dst[k] = v; return; }
  if (ij.lose || k - ij.delta < ij.start || k - ij.delta >= ij.end) return;
  dst[k] = msg(k - ij.delta);
}

// ================================================================================================
// ---- W2/W3: the exchange at width, the resend (design section 3.1). One loop per block replaces
// W1's bounded handshake spin and push check (r4d_arw_handshake, r4d_arw_check) in the RS kernels of
// r4d_ar_wide_host.hip (ar_hc_write_4rank, ar_gather_hc_write_4rank: W2; ar_oneshot_4rank_exact,
// ar_gather_4rank: W3). Each rank is, for block b of a launch, the writer of three pushes (one to
// each peer) and the reader of three; the loop serves both roles in every pass, so no wait closes a
// cycle (every block of the launch is co-resident on every rank, as before):
//   - the reader, per peer q whose flag has arrived and whose push is not yet checked: one check try
//     (W1's, filling the mirror). Pass: ack[ph][b][me] = v into q's flags. Fail: after R4D_ARX_T_RQ
//     from arrival and two failed tries, a request rq[ph][b][me] = (v << 32) | n into q's flags, then
//     one every R4D_ARX_T_RQ2; past the request limit (BRILLIANCE_AR_RESEND_MAX, 8 by default), or
//     R4D_ARCHK_DEADLINE after arrival, W1's bounded failure: a CHECK fault naming q, an ack anyway (so
//     q's writer can leave), and the block leaves without waiting for its own acks;
//   - the writer, per peer q: a request from q for (v, n) with n new re-pushes this block's words for
//     q only, from the kernel's unreduced source (the kernel holds its reduce until this returns), and
//     re-puts q's check word with a system-scope release;
//   - both: the block leaves once it has checked all three peers' pushes AND every peer has acked its
//     own push: the ack equals v, or the peer's flag for block b has passed v (it is in a later
//     launch, so its ack was lost: counted), or R4D_ARX_T_ACK after this block's last check (counted);
//   - re-stores, each counted: before a peer's flag has arrived, this rank's flag to it every
//     R4D_ARX_T_RE0; while a peer's ack is missing, the flag and this rank's ack to it after
//     R4D_ARX_T_RE, doubling to R4D_ARX_T_RE_MAX.
// Every control word has one writer: a reader writes acks and requests into the writer's flags at
// its own rank's index (ACK and RQ, R4D_ARW_*), so no PCIe atomics are needed. Acks and requests match
// v exactly, so a stale word from an earlier launch, or a stray line, is ignored. A resend writes the
// same bytes and the same check word, so any mix of first and second landings is the push.
// Deadlines: the two-rank exchange's (R4D_ARX_*, r4d_ar_seq.h), and W1's for a flag (30 s) and a
// fault on record (R4D_AR_FAULT_FAST). Thread 0 keeps every peer's state and makes every control
// store and every counter and record write (the record writer must be one thread); the other threads
// follow a command word in LDS. Counters (R4D_ARW_CNT_RQ, _RS, _RE, _ACK per op and block, requests
// by source R4D_ARW_SRC_RQ, late passes as W1) are this launch's, added by thread 0 at the end, only
// when nonzero: the healthy path stores nothing more than three acks.
//
// Injection at width, the exchange's own modes (BRILLIANCE_AR_INJECT, r4d_ar_entry.hip), on top of
// W1's lose, dup, stuck (also on every resend: a path that never heals) and noflag (ctl-flag):
//   pair     lose in block b; block b + 1 (b - 1 for the last block), after its own push, stores b's
//            line over its own line at the same offset, to the same peer: a misdirection
//   late     the line is withheld from the push and stored from the loop R4D_ARX_T_LATE after
//   ctl-ack  the injected rank, as the reader of the target peer's push, skips its first ack store
//   ctl-rq   the injected rank, as a reader, finds one line of its own slot from the target peer
//            wrong (it overwrites it) and skips its first request store to that peer
//   multi    lose, in the pushes to two peers (the target and the next in the peer list)
//   cross    (the launcher's) two writers lose a line of the same block, to the same reader
#define R4D_AR_INJ_MULTI 9u
#define R4D_AR_INJ_CROSS 10u   // the launcher's mode: each writer's kernel word says lose
#define R4D_ARWX_LEAVE 0x10000u
#define R4D_ARWX_LATE  0x20000u
#define R4D_ARWX_BAD   0x40000u

// The ack of (phase, block b, source src) in a flags allocation, and the request (64-bit).
__device__ __forceinline__ unsigned int* r4d_arw_ack(unsigned int* flags, unsigned int ws,
                                                     unsigned int nbw, unsigned int phase,
                                                     unsigned int b, unsigned int src) {
  return flags + R4D_ARW_ACK_OFF(ws, nbw) + (phase * nbw + b) * ws + src;
}
__device__ __forceinline__ unsigned long long* r4d_arw_rq(unsigned int* flags, unsigned int ws,
                                                          unsigned int nbw, unsigned int phase,
                                                          unsigned int b, unsigned int src) {
  return reinterpret_cast<unsigned long long*>(flags + R4D_ARW_RQ_OFF(ws, nbw)) +
         (phase * nbw + b) * ws + src;
}

// Writer: r4d_arw_put, and thread 0 keeps the block sums `t` (the resend re-puts them).
template <int WS, int NP>
__device__ __forceinline__ void r4d_arw_put_t(const R4DArWidePeers<int4, NP>& peers, int b, int rank,
                                              unsigned int phase, unsigned int parity,
                                              const unsigned int (&h)[NP], unsigned int hs,
                                              unsigned int (&t)[NP]) {
  r4d_arw_bsum<NP>(h, t);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int p = 0; p < NP; ++p)
      __hip_atomic_store(r4d_arw_ck(peers.flags[p], WS, R4D_AR_WIDE_MAX_BLOCKS, phase, parity,
                                    (unsigned)b, (unsigned)rank),
                         t[p] + hs * R4D_ARCHK_SALT, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// The publish half of r4d_arw_handshake: once this block's stores have drained, thread 0 stores
// `val` into every peer's flag [b][rank], except the peer indices in `skip` (injection noflag).
template <int WS, int NP>
__device__ __forceinline__ void r4d_arw_publish(const R4DArWidePeers<int4, NP>& peers, int b, int rank,
                                                unsigned int val, int drain, int pub,
                                                unsigned int skip) {
  r4d_ar_drain(drain);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (drain == 2) __threadfence_system();
    if (pub) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
#pragma unroll
      for (int p = 0; p < NP; ++p)
        if (!((skip >> p) & 1u)) r4d_ar_store_sys_rlx(&peers.flags[p][(size_t)b * WS + rank], val);
    } else {
#pragma unroll
      for (int p = 0; p < NP; ++p)
        if (!((skip >> p) & 1u)) r4d_ar_store_sys_rel(&peers.flags[p][(size_t)b * WS + rank], val);
    }
  }
}

// An injected launch's plan with the exchange's modes. `d` is W1's data damage (its `peer` the first
// damaged peer index), `pmask` every peer index whose push of d's words is damaged (multi: two).
struct R4dArwInjX {
  R4dArwInj d;
  unsigned int pmask;
  int clo, chi, cdelta, cpeer;   // pair's partner block: words [clo, chi) to cpeer get pmsg(k - cdelta)
  bool late, stuck;
  unsigned int skip_ack, skip_rq;   // peer-index masks: the first ack / request not stored
  int bpeer, blo, bhi;              // ctl-rq: words [blo, bhi) of this rank's own slot from bpeer
};
__device__ __forceinline__ R4dArwInjX r4d_arw_injx_none(int start, int end) {
  R4dArwInjX r;
  r.d = R4dArwInj{ -1, 0, 0, 0, start, end, true, 0u };
  r.pmask = 0u; r.clo = r.chi = r.cdelta = r.cpeer = 0;
  r.late = r.stuck = false; r.skip_ack = r.skip_rq = 0u; r.bpeer = 0; r.blo = r.bhi = 0;
  return r;
}
// The one 64-byte line `off` (mod the block's lines) of [start, end) in a slot based at `base`:
// words [lo, hi) of it inside the block, and the line's first word `full` (may be below start).
__device__ __forceinline__ void r4d_arw_inj_line(int off, int start, int end, const void* base,
                                                 int& lo, int& hi, int& full) {
  const int ph = (int)(((uintptr_t)base >> 4) & 3u);
  const int L0 = (start + ph) >> 2, L1 = (end - 1 + ph) >> 2, nl = L1 - L0 + 1;
  full = 4 * (L0 + off % nl) - ph;
  lo = full > start ? full : start;
  hi = full + 4 < end ? full + 4 : end;
}
// Whether losing words [lo, hi) of a push to `slot` changes bytes (the slot does not already hold
// msg's words there). Reads the slot over the link: injected launches only.
template <typename FM>
__device__ __forceinline__ bool r4d_arw_inj_changes(const int4* slot, int lo, int hi, const FM& msg) {
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  bool eff = false;
  for (int k = lo; k < hi; ++k) {
    const int4 want = msg(k);
    const unsigned int* q = reinterpret_cast<const unsigned int*>(slot + k);
    eff = eff || (int)r4d_ar_ld(q) != want.x || (int)r4d_ar_ld(q + 1) != want.y ||
          (int)r4d_ar_ld(q + 2) != want.z || (int)r4d_ar_ld(q + 3) != want.w;
  }
  return eff;
}
__device__ __forceinline__ void r4d_arw_inj_count(unsigned int* my_ctl) {
  r4d_ar_st(my_ctl + R4D_ARW_CTL_INJEFF, r4d_ar_ld(my_ctl + R4D_ARW_CTL_INJEFF) + 1u);
}
// Thread 0's plan for block b of nb (each block `w` words of n16) in this phase. `slotp(p)`: the
// push's destination slot in peer index p; `mys(p)`: this rank's own slot from peer index p.
template <int NP, typename FM, typename FS, typename FO>
__device__ __forceinline__ R4dArwInjX r4d_arw_injx_plan(unsigned int iw, int b, unsigned int phase,
                                                        int start, int end, int w, int n16,
                                                        unsigned int* my_ctl, const FM& msg,
                                                        const FS& slotp, const FO& mys) {
  R4dArwInjX r = r4d_arw_injx_none(start, end);
  const unsigned int mode = iw & 15u;
  if (!mode || ((iw >> 13) & 1u) != phase) return r;
  const int bt = (int)((iw >> 4) & 63u), off = (int)((iw >> 14) & 255u);
  int peer = (int)((iw >> 10) & 7u);
  if (peer >= NP) peer = 0;
  if (mode == R4D_AR_INJ_PAIR && bt != b) {
    /* The partner block: the target block's line at `off`, over its own line at the same offset. */
    if (r4d_ar_inj_partner(bt, (int)gridDim.x) != b || end <= start) return r;
    const int ts = bt * w, te = ts + w < n16 ? ts + w : n16;
    if (te <= ts) return r;
    int tlo = 0, thi = 0, tf = 0, clo = 0, chi = 0, cf = 0;
    r4d_arw_inj_line(off, ts, te, slotp(peer), tlo, thi, tf);
    r4d_arw_inj_line(off, start, end, slotp(peer), clo, chi, cf);
    r.cdelta = cf - tf;
    r.clo = clo > tlo + r.cdelta ? clo : tlo + r.cdelta;
    r.chi = chi < thi + r.cdelta ? chi : thi + r.cdelta;
    r.cpeer = peer;
    return r;
  }
  if (bt != b || end <= start) return r;
  switch (mode) {
  case R4D_AR_INJ_LOSE: case R4D_AR_INJ_DUP: case R4D_AR_INJ_STUCK: case R4D_AR_INJ_NOFLAG:
  case R4D_AR_INJ_LATE: case R4D_AR_INJ_MULTI: {
    /* W1's plan (the effective-line search), as lose for late and multi. */
    const unsigned int m1 = (mode == R4D_AR_INJ_LATE || mode == R4D_AR_INJ_MULTI) ? R4D_AR_INJ_LOSE : mode;
    r.d = r4d_arw_inj_plan((iw & ~15u) | m1, b, phase, start, end, slotp(peer), my_ctl, msg);
    if (r.d.peer >= 0) r.pmask = 1u << r.d.peer;
    r.late = mode == R4D_AR_INJ_LATE;
    r.stuck = mode == R4D_AR_INJ_STUCK;
    if (mode == R4D_AR_INJ_MULTI && r.d.peer >= 0) {
      const int p2 = (peer + 1) % NP;
      r.pmask |= 1u << p2;
      if (r4d_arw_inj_changes(slotp(p2), r.d.lo, r.d.hi, msg)) r4d_arw_inj_count(my_ctl);
    }
    return r;
  }
  case R4D_AR_INJ_PAIR: {
    /* The target block: lose the fixed line `off` (the partner must find the same line). */
    int lo = 0, hi = 0, f = 0;
    r4d_arw_inj_line(off, start, end, slotp(peer), lo, hi, f);
    r.d.peer = peer; r.d.lo = lo; r.d.hi = hi; r.d.delta = 0; r.d.lose = true;
    r.pmask = 1u << peer;
    if (r4d_arw_inj_changes(slotp(peer), lo, hi, msg)) r4d_arw_inj_count(my_ctl);
    return r;
  }
  case R4D_AR_INJ_ACK:
    r.skip_ack = 1u << peer;
    return r;
  case R4D_AR_INJ_RQ: {
    int f = 0;
    r.skip_rq = 1u << peer;
    r.bpeer = peer;
    r4d_arw_inj_line(off, start, end, mys(peer), r.blo, r.bhi, f);
    r4d_arw_inj_count(my_ctl);   // the overwrite always changes bytes (an xor)
    return r;
  }
  default:
    return r;
  }
}
template <int NP, typename FM, typename FS, typename FO>
__device__ __forceinline__ R4dArwInjX r4d_arw_injx_bcast(unsigned int iw, int b, unsigned int phase,
                                                         int start, int end, int w, int n16,
                                                         unsigned int* my_ctl, const FM& msg,
                                                         const FS& slotp, const FO& mys) {
  __shared__ R4dArwInjX s_injx;
  if (threadIdx.x == 0)
    s_injx = r4d_arw_injx_plan<NP>(iw, b, phase, start, end, w, n16, my_ctl, msg, slotp, mys);
  __syncthreads();
  const R4dArwInjX r = s_injx;
  __syncthreads();
  return r;
}
// One store of an injected launch's push (the exchange's plan): word k of value v to peer index p.
template <typename FM>
__device__ __forceinline__ void r4d_arw_injx_put(const R4dArwInjX& ij, int p, int4* dst, int k,
                                                 const int4& v, const FM& msg) {
  if (!((ij.pmask >> p) & 1u) || k < ij.d.lo || k >= ij.d.hi) { dst[k] = v; return; }
  if (ij.d.lose || k - ij.d.delta < ij.d.start || k - ij.d.delta >= ij.d.end) return;
  dst[k] = msg(k - ij.d.delta);
}
// pair's partner, every thread after its push: thread 0 stores the target block's line (`pmsg`,
// the words as the kernel holds them; the target block's may not be there yet, which is wrong
// either way) over this block's own words to peer index cpeer, and counts the injection when that
// changed bytes.
template <typename FP>
__device__ __forceinline__ void r4d_arw_injx_pair(const R4dArwInjX& ij, int4* dst,
                                                  unsigned int* my_ctl, const FP& pmsg) {
  if (ij.clo >= ij.chi) return;
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    bool eff = false;
    for (int k = ij.clo; k < ij.chi; ++k) {
      const int4 v = pmsg(k - ij.cdelta), o = pmsg(k);
      eff = eff || v.x != o.x || v.y != o.y || v.z != o.z || v.w != o.w;
      dst[k] = v;
    }
    if (eff) r4d_ar_st(my_ctl + R4D_ARW_CTL_INJEFF2, r4d_ar_ld(my_ctl + R4D_ARW_CTL_INJEFF2) + 1u);
  }
}

// The exchange. Every thread calls it after r4d_arw_publish. `hck` is thread 0's block sums from
// r4d_arw_put_t; `v` the hop's handshake value, `phase` its index (one-shots 0). The kernel's:
//   rehash(mask, h)    W1's check try over the peer indices in `mask` (r4d_arw_check's);
//   repush(mask, stuck) re-pushes every word of the block to the peer indices in `mask`, from the
//                       unreduced source (stuck: the injected words left out again);
//   late()             stores injection late's withheld words;
//   bad()              overwrites ctl-rq's words of this rank's own slot.
// With INJ false the last two are never called and `ij` is ignored. A give-up is recorded (W1's
// record); the kernel then reduces what the last try stored, and the engine stops on the record.
template <int WS, int NP, bool INJ, typename FH, typename FP, typename FL, typename FB>
__device__ __forceinline__ void r4d_arw_exchange(const R4DArWidePeers<int4, NP>& peers,
                                                 unsigned int* my_flags, int b, int rank,
                                                 unsigned int v, unsigned int phase,
                                                 unsigned int parity, unsigned int op,
                                                 const unsigned int (&hck)[NP], int drain,
                                                 const FH& rehash, const FP& repush, const FL& late,
                                                 const FB& bad, const R4dArwInjX& ij) {
  /* Two command words by pass parity, as the two-rank exchange: thread 0 writes the next pass's
   * before a slow thread has necessarily read this one's; the pass's barrier orders the reuse. */
  __shared__ unsigned int s_xcmd[2];
  constexpr unsigned int nbw = R4D_AR_WIDE_MAX_BLOCKS;
  constexpr unsigned int full = (1u << NP) - 1u;
  const unsigned int tid = threadIdx.x, ub = (unsigned)b;
  unsigned int par = 0u, cmd = 0u;
  // Thread 0's state; masks of peer indices. The other threads keep only what they read of s_xcmd.
  unsigned int arrived = 0u, checked = 0u, acked = 0u;
  bool gone = false, fast = false, late_due = INJ && ij.late, bad_due = INJ && ij.blo < ij.bhi;
  unsigned int skip_ack = INJ ? ij.skip_ack : 0u, skip_rq = INJ ? ij.skip_rq : 0u;
  unsigned int f[NP], tries[NP], n_srv[NP], n_rq[NP], want[NP], c_src[NP], c_late[NP];
  unsigned long long t_arr[NP], t_re[NP], per[NP], t_rq[NP];
  unsigned int rq_max = 0u, c_rq = 0u, c_rs = 0u, c_re = 0u, c_ack = 0u;
  unsigned long long t0 = 0ull, t_chk = 0ull, t_pk = 0ull;
  if (tid == 0) { t0 = wall_clock64(); t_pk = t0; }
#pragma unroll
  for (int p = 0; p < NP; ++p) {
    f[p] = tries[p] = n_srv[p] = n_rq[p] = want[p] = c_src[p] = c_late[p] = 0u;
    t_arr[p] = t_rq[p] = 0ull; t_re[p] = t0; per[p] = R4D_ARX_T_RE;
  }
  for (;; par ^= 1u) {
    if (tid == 0) {
      cmd = 0u;
      /* Up to nine independent loads of pinned host memory, in flight together. */
      unsigned int fl[NP], ak[NP];
      unsigned long long rq[NP];
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        const unsigned int q = (unsigned)peers.rank[p];
        fl[p] = r4d_ar_ld(my_flags + ub * WS + q);
        rq[p] = __hip_atomic_load(r4d_arw_rq(my_flags, WS, nbw, phase, ub, q), __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
        ak[p] = (checked == full && !((acked >> p) & 1u))
                    ? r4d_ar_ld(r4d_arw_ack(my_flags, WS, nbw, phase, ub, q)) : 0u;
      }
      const unsigned long long now = wall_clock64();
      /* Once a wait has lasted R4D_AR_FAULT_PEEK, look every R4D_AR_FAULT_PEEK for a fault on record
       * on the instance: then every deadline below is R4D_AR_FAULT_FAST (W1's rule). */
      if (!fast && now - t_pk > R4D_AR_FAULT_PEEK) {
        t_pk = now;
        fast = r4d_arw_faulted(my_flags, WS, nbw);
      }
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        const unsigned int bit = 1u << p, q = (unsigned)peers.rank[p];
        if (!(arrived & bit)) {
          f[p] = fl[p];
          if (!r4d_ar_seq_behind(fl[p], v)) {
            arrived |= bit;
            t_arr[p] = now;
          } else if (now - t0 > (fast ? R4D_AR_FAULT_FAST : R4D_AR_FLAG_DEADLINE)) {
            /* W1's flag give-up: recorded; the check still runs (it passes if the bytes did land;
             * the record stands and the engine stops). The peer is out of step: fast. */
            (void)r4d_arw_fault_note(my_flags, WS, nbw, R4D_AR_FAULT_FLAG, op, ub, v, parity, 0u, 0u,
                                     v, fl[p], now - t0, q, phase);
            arrived |= bit;
            fast = true;
            t_arr[p] = now;
          } else if (now - t_re[p] > R4D_ARX_T_RE0) {
            r4d_ar_store_sys_rel(&peers.flags[p][(size_t)ub * WS + rank], v);
            ++c_re;
            t_re[p] = now;
          }
        }
        /* The writer: a new request from q for this launch's push of block b. */
        if ((unsigned int)(rq[p] >> 32) == v && (int)((unsigned int)rq[p] - n_srv[p]) > 0) {
          n_srv[p] = (unsigned int)rq[p];
          cmd |= bit;
        }
        if (checked == full && !(acked & bit)) {
          if (ak[p] == v || gone) {
            acked |= bit;
          } else if (!r4d_ar_seq_behind(fl[p], v + 1u)) {
            /* q is in a later hop or launch on block b, so it left this one: its ack, stored before
             * that flag's release, was lost if it is not here now. Read again past the flag. */
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
            if (r4d_ar_ld(r4d_arw_ack(my_flags, WS, nbw, phase, ub, q)) != v) ++c_ack;
            acked |= bit;
          } else if (now - t_chk > (fast ? R4D_AR_FAULT_FAST : R4D_ARX_T_ACK)) {
            ++c_ack;
            acked |= bit;
          } else if (now - t_re[p] > per[p]) {
            /* q may wait on this rank's flag (it never arrived there) or its ack. */
            r4d_ar_store_sys_rel(&peers.flags[p][(size_t)ub * WS + rank], v);
            r4d_ar_st(r4d_arw_ack(peers.flags[p], WS, nbw, phase, ub, (unsigned)rank), v);
            ++c_re;
            t_re[p] = now;
            per[p] = per[p] * 2ull < R4D_ARX_T_RE_MAX ? per[p] * 2ull : R4D_ARX_T_RE_MAX;
          }
        }
      }
      const unsigned int ckm = arrived & ~checked;
      if (ckm) {
        cmd |= ckm << 8;
        /* ctl-rq's overwrite only once that peer's push has landed (its flag is here). */
        if (INJ && bad_due && ((ckm >> ij.bpeer) & 1u)) { bad_due = false; cmd |= R4D_ARWX_BAD; }
      }
      if (INJ && late_due && now - t0 > R4D_ARX_T_LATE) { late_due = false; cmd |= R4D_ARWX_LATE; }
      if (checked == full && acked == full) cmd |= R4D_ARWX_LEAVE;
      s_xcmd[par] = cmd;
    }
    __syncthreads();
    cmd = s_xcmd[par];
    const unsigned int rsm = cmd & full, ckm = (cmd >> 8) & full;
    if (rsm || (INJ && (cmd & R4D_ARWX_LATE))) {
      if (rsm) repush(rsm, INJ && ij.stuck);
      if (INJ && (cmd & R4D_ARWX_LATE)) late();
      /* As the push: drained before thread 0's release, which writes the bytes back. */
      if (drain == 1) __threadfence_system();
      else asm volatile("s_wait_storecnt 0x0" ::: "memory");
      __syncthreads();
      if (tid == 0) {
        if (rsm) {
#pragma unroll
          for (int p = 0; p < NP; ++p)
            if ((rsm >> p) & 1u) {
              __hip_atomic_store(r4d_arw_ck(peers.flags[p], WS, nbw, phase, parity, ub, (unsigned)rank),
                                 hck[p] + v * R4D_ARCHK_SALT, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
              ++c_rs;
            }
        } else {
          __threadfence_system();
        }
      }
    }
    if (ckm) {
      if (INJ && (cmd & R4D_ARWX_BAD)) {
        bad();
        __threadfence_system();
        __syncthreads();
      }
      /* Drop the vector caches (system scope), so this try reads where the push lands; then the
       * check words first, so their round trips overlap the re-hash's loads (r4d_arw_check). */
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
      if (tid == 0) {
#pragma unroll
        for (int p = 0; p < NP; ++p)
          want[p] = ((ckm >> p) & 1u) ? r4d_ar_ld(r4d_arw_ck(my_flags, WS, nbw, phase, parity, ub,
                                                             (unsigned)peers.rank[p]))
                                      : 0u;
      }
      unsigned int h[NP];
#pragma unroll
      for (int p = 0; p < NP; ++p) h[p] = 0u;
      rehash(ckm, h);
      unsigned int t[NP];
      r4d_arw_bsum<NP>(h, t);   // two barriers: s_wpart is free again at once
      if (tid == 0) {
        const unsigned long long now = wall_clock64();
        const unsigned int was = checked;
#pragma unroll
        for (int p = 0; p < NP; ++p) {
          const unsigned int bit = 1u << p, q = (unsigned)peers.rank[p];
          if (!((ckm >> p) & 1u)) continue;
          const unsigned int got = t[p] + v * R4D_ARCHK_SALT;
          unsigned int* const ack_out = r4d_arw_ack(peers.flags[p], WS, nbw, phase, ub, (unsigned)rank);
          if (got == want[p]) {
            checked |= bit;
            if (tries[p] && !n_rq[p]) ++c_late[p];   // failed first, then landed with no request: late
            if (INJ && (skip_ack & bit)) skip_ack &= ~bit;
            else r4d_ar_st(ack_out, v);
          } else {
            ++tries[p];
            const unsigned long long el = now - t_arr[p];
            bool give = el > (fast ? R4D_AR_FAULT_FAST : R4D_ARCHK_DEADLINE);
            if (!give && tries[p] >= 2u && el > R4D_ARX_T_RQ &&
                (n_rq[p] == 0u || now - t_rq[p] > R4D_ARX_T_RQ2)) {
              if (rq_max == 0u) {
                rq_max = r4d_ar_ld(my_flags + R4D_ARW_CTL_OFF(WS, nbw) + R4D_ARW_CTL_RQMAX);
                if (rq_max == 0u) rq_max = R4D_ARX_N_RQ;
              }
              if (n_rq[p] >= rq_max) {
                give = true;
              } else {
                ++n_rq[p];
                ++c_rq;
                ++c_src[p];
                t_rq[p] = now;
                if (INJ && (skip_rq & bit)) skip_rq &= ~bit;
                else __hip_atomic_store(r4d_arw_rq(peers.flags[p], WS, nbw, phase, ub, (unsigned)rank),
                                        ((unsigned long long)v << 32) | n_rq[p], __ATOMIC_RELAXED,
                                        __HIP_MEMORY_SCOPE_SYSTEM);
              }
            }
            if (give) {
              /* W1's bounded failure: recorded; the kernel reduces what the last try stored and
               * ends, and the engine stops. The ack lets q's writer leave now. */
              checked |= bit;
              gone = true;
              (void)r4d_arw_fault_note(my_flags, WS, nbw, R4D_AR_FAULT_CHECK, op, ub, v, parity,
                                       tries[p], got, want[p], f[p], el, q, phase);
              r4d_ar_st(ack_out, v);
            }
          }
        }
        if (checked == full && was != full) {
          /* The last check: the ack waits and their re-stores start from here. */
          t_chk = now;
#pragma unroll
          for (int p = 0; p < NP; ++p) { t_re[p] = now; per[p] = R4D_ARX_T_RE; }
        }
      }
    }
    if (cmd & R4D_ARWX_LEAVE) break;
    if (!(cmd & (full | (full << 8)))) __builtin_amdgcn_s_sleep(1);
  }
  if (tid == 0) {
    if (c_rq | c_rs | c_re | c_ack) {
      unsigned int* const k = my_flags + R4D_ARW_CNT_OFF(WS, nbw) + (r4d_arw_op_index(op) * nbw + ub) * 8u;
      if (c_rq) r4d_ar_st(k + R4D_ARW_CNT_RQ, r4d_ar_ld(k + R4D_ARW_CNT_RQ) + c_rq);
      if (c_rs) r4d_ar_st(k + R4D_ARW_CNT_RS, r4d_ar_ld(k + R4D_ARW_CNT_RS) + c_rs);
      if (c_re) r4d_ar_st(k + R4D_ARW_CNT_RE, r4d_ar_ld(k + R4D_ARW_CNT_RE) + c_re);
      if (c_ack) r4d_ar_st(k + R4D_ARW_CNT_ACK, r4d_ar_ld(k + R4D_ARW_CNT_ACK) + c_ack);
    }
#pragma unroll
    for (int p = 0; p < NP; ++p) {
      const unsigned int q = (unsigned)peers.rank[p];
      if (c_src[p]) {
        unsigned int* const rs = my_flags + R4D_ARW_SRC_OFF(WS, nbw) + R4D_ARW_SRC_RQ * WS * nbw + ub * WS + q;
        r4d_ar_st(rs, r4d_ar_ld(rs) + c_src[p]);
      }
      if (c_late[p]) {
        /* W1's late passes (r4d_arw_note_late's words), written here: W1's kernels keep that
         * function to themselves, so their code is unchanged. */
        unsigned int* const ls = my_flags + R4D_ARW_SRC_OFF(WS, nbw) + R4D_ARW_SRC_LATE * WS * nbw + ub * WS + q;
        unsigned int* const lo = my_flags + R4D_ARW_CNT_OFF(WS, nbw) + (r4d_arw_op_index(op) * nbw + ub) * 8u +
                                 R4D_ARW_CNT_LATE;
        r4d_ar_st(ls, r4d_ar_ld(ls) + c_late[p]);
        r4d_ar_st(lo, r4d_ar_ld(lo) + c_late[p]);
      }
    }
  }
}
