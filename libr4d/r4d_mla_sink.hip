/* Absorbed MLA with a per-head sink logit (Hy4-flash): mla_attn_paged_sink (a paged latent cache,
 * one table a sequence, a sliding window counting the row's own token or none) and
 * mla_attn_sel_sink (a per-row table and length: the indexer's selection). Cache row of a token:
 * [latent 512 | rotated key 64] bf16. Design: kernels/mla_sink_attn.md of the Hy4-flash plan.
 *
 * The kernel is attend_rows (r4d_mla.hip) with three changes: the rotary columns join the score
 * (one more WMMA a wave a chunk), the token range has a window lower bound, and the sink joins the
 * softmax once, at the store of an unsplit launch or in the merge of a split one. The sink-in-reduce
 * rule follows vLLM's rocm_aiter_mla_sparse.py (@820fce0d, Apache-2.0): the splits' partials are
 * relative to their own maxima, the merge reads all maxima first, and the sink is applied after the
 * merge, never per split. The code is new.
 *
 * Workgroup: one query row and HG groups of sixteen heads, four waves a group. Scores are
 * S^T = K Q^T: tokens are the WMMA rows, heads the columns, so a head's running maximum and
 * denominator live in its lane. Wave `quarter` of a group takes latent columns
 * [128 quarter, 128 quarter + 128) and rotary columns [16 quarter, 16 quarter + 16); the four
 * partial dots meet in LDS and every wave adds them in the order 0, 1, 2, 3, so all four hold the
 * same scores. P goes to the value product as bf16 high part plus bf16 remainder. The next chunk's
 * keys are fetched into registers while the current one is multiplied; two barriers a chunk.
 *
 * The stable law (WP16 perf.md 3.4.1, anchored at the row's first visible key): a row's range
 * [lo, hi) is cut into slots of 64 tokens with a window and of 128 at window 0 and in the selection
 * form, slot j = [lo + SC j, lo + SC (j + 1)), each walked in 16-token chunks from its start. The
 * law has sink_law() slots (8 at a window of 512, kMaxSplits at window 0 and in the selection form),
 * the last one taking whatever lies past it. At window 0 and in the selection form lo is 0, so the
 * grid is absolute there. (perf.md's absolute grid for a window -- slots at multiples of 64 of the
 * sequence -- was built and measured: on the card under the engine a verify row then did not give
 * the decode row's bytes, although the emulator and r4d_selftest found the launch row-exact; the
 * lo-anchored grid does, and is the old split's at a full window. See notes/wp16s.md.)
 * The slots holding a visible key fold in ascending order
 * (the first copied, each later one acc * exp(acc_m - m') + part * exp(part_m - m')), then the sink
 * joins once. So a row's bytes are a function of its own operands, the cache it sees and the op's
 * parameters: not of M, the launch's other rows, HG, the form, the slots a launch needs (the
 * table's width bounds them) or the number of streams. A verify of k + 1 rows of any number of
 * streams gives the single-row decode at each position bit for bit, and the paged form at window 0
 * equals the selection form over a table listing the same pages in order (both anchor at 0).
 *
 * Two forms run the law. The split form (up to the form's row cap: kRowsWindow / kRowsFull for
 * `stable 0`, kRowsStableWindow / kRowsStableFull for `stable 1`): a workgroup a slot, the partial
 * (relative to its own maximum) to the scratch, merge_sink folds them. The fused form (more rows,
 * `stable 1`): one workgroup walks the row's chunks as the unsplit pass does and at
 * each super-chunk boundary folds the slot's state into an accumulator a row in the scratch with
 * the merge's arithmetic (sink_fold, explicit fmaf), so the bytes are the split form's without its
 * partials' traffic (8 or 20 of them a row and head). The order of the vLLM-style merge (every
 * maximum first) is not used: it cannot be computed in one pass.
 *
 * `stable 1` (the plugin's decode-class instance) always runs the law and refuses a launch whose
 * scratch does not hold the fused form's accumulator (RAD_E_SCRATCH); the scratch hook sizes the
 * split form's partials up to the cap and the accumulator for every row of the declared M.
 * `stable 0` (default; prefill rows) runs the split form up to the cap when the scratch holds it,
 * and unsplit (one pass over the row's chunks, one running maximum) past that, which differs from
 * the law in the last place. */
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_gdn_wmma.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>

namespace {
/* A partial: 512 values, max, den, padded to 16 bytes for the merge's vector loads. */
constexpr int D = 512, RD = 64, W = D + RD, PART = D + 4;
/* Row pitch of a staged token in LDS: 576 + 8, so consecutive rows do not share a bank pattern. */
constexpr int kPitch = W + 8;
constexpr int kPieces = 16 * W / 8; /* 16-byte pieces of a 16-token chunk */
/* Super-chunks: 64 tokens with a window, 128 at window 0 and in the selection form. A lone
 * workgroup takes about 1.8 us a 16-token chunk (gfx1201), so a decode row's time is the chunks of
 * one super-chunk; more of them cost the partials' write and merge. The merge holds kMaxSplits. */
constexpr int kShiftWindow = 6, kShiftFull = 7, kMaxSplits = 20;
/* The most rows a `stable 0` launch splits; past them the workgroups fill the card unsplit, which
 * is then also faster (measured: the window at 16 rows 79 us split against 68 unsplit, the
 * selection at 64 rows 362 against 269).
 *
 * `stable 1` (the plugin's decode-class instance) splits to its own, larger cap: its rows are the
 * decode-class ones, and a speculative step (c4-c8 DFly2) asks for 17 to 48 of them, where the
 * `stable 0` measurement above did not reach: one workgroup a row then walks the row's whole
 * range instead of the 4 chunks a slot. 48 rows is the widest such step today (max_seqs x
 * (max_spec + 1) = 8 x 6), and what it costs in scratch is in the scratch hook's note below. */
constexpr long long kRowsWindow = 16, kRowsFull = 32, kRowsStableWindow = 48, kRowsStableFull = 48;
/* Table entries a split reads through LDS: a selection split of 128 tokens at block size 1. */
constexpr int kSpanPages = 160;

struct SinkMeta {
    long long qr, qh, rr, rh, kp, kt, bp, bc, lp, orr, oh, ss;
    /* splits: the launch's slots (grid z, the partial pitch; 1 = unsplit), law: the law's slot
     * count, sc_shift: log2 of the super-chunk, fused: an unsplit launch that still runs the law
     * (one workgroup walks the row's slots in order and folds each into an accumulator in the
     * scratch as merge_sink folds the partials). row0: the absolute row the launch's row 0 stands
     * for -- 0 for a whole step, the piece's offset for the pipelined prefill's first piece (a
     * piece at rows [row0, row0 + rows) of a one-sequence step is asked for its queries' absolute
     * positions, which `used - qlen + i` cannot give: that is the last qlen rows' formula). */
    int rows, heads, seqs, blocks, bs, bs_shift, pages, splits, law, sc_shift, fused, window, row0;
    float scale;
    int pad[2] = { 0, 0 }; /* no tail padding: a recorded pass's audit compares the argument's bytes */
};
static_assert(sizeof(SinkMeta) == 12 * 8 + 13 * 4 + 4 + 8, "SinkMeta is a kernel argument by value "
              "and carries no padding");

/* r4d_mla_sink_pin_form: 0 the launcher's rule, 1 sixteen heads a workgroup, 2 thirty-two, 3 the
 * element-wise key load, 4 unsplit (not the law), 5 sixty-four heads a workgroup, 6 the fused
 * form (the law in one workgroup a row). A test hook (r4d_selftest, r4d_emu); nothing else sets it. */
int g_sink_form = 0;

__device__ __forceinline__ int sink_seq(const int* cu, int seqs, int r) {
    int s = 0;
    while (s + 1 < seqs && cu[s + 1] <= r) ++s;
    return s;
}

/* The visible token range of row r and the table row it reads. */
template <bool SEL>
__device__ __forceinline__ void sink_range(const int* used, const int* cu, const int* len,
                                           const SinkMeta& m, int r, int& lo, int& hi, int& trow) {
    const int cap = m.blocks * m.bs;
    if constexpr (SEL) {
        lo = 0;
        hi = min(max(len[(long long)r * m.lp], 0), cap);
        trow = r;
    } else {
        const int s = sink_seq(cu, m.seqs, r);
        const int qlen = cu[s + 1] - cu[s], i = r - cu[s];
        const long long p = (long long)used[s] - qlen + i + m.row0;
        const long long top = p + 1 < 0 ? 0 : p + 1 < cap ? p + 1 : cap,
                   bottom = m.window > 0 && p + 1 - m.window > 0 ? p + 1 - m.window : 0;
        hi = (int)top;
        lo = (int)(bottom < top ? bottom : top);
        trow = s;
    }
}

/* The law's merge, shared by merge_sink and the fused form so both give the same bytes: slots with
 * a visible key folded in ascending order into (acc_m, acc_d, acc), the first one copied, each later
 * one by acc = acc * exp(acc_m - m') + part * exp(part_m - m'), m' = max(acc_m, part_m); explicit
 * fmaf so no contraction choice of the compiler differs between the two kernels. */
struct SinkFold {
    float a, b, m;
};
__device__ __forceinline__ SinkFold sink_fold(float acc_m, float part_m) {
    const float mn = fmaxf(acc_m, part_m);
    return { expf(acc_m - mn), expf(part_m - mn), mn };
}
/* The sink once at the end: shift to max(m, sink) as libref does, so a sink far above every score
 * cannot overflow. No visible key gives zero, sink or not. */
__device__ __forceinline__ float sink_inv(float m, float d, float sink) {
    if (!(d > 0)) return 0;
    const float top = fmaxf(m, sink), c = expf(m - top);
    return c / fmaf(d, c, expf(sink - top));
}

/* FUSED: the fused form (m.fused), its own instance so its accumulator's registers do not touch
 * the other forms' schedule. */
template <int HG, bool WIDE, bool SEL, bool FUSED>
__device__ __forceinline__ void mla_sink_rows_body(
    const uint16_t* q, const uint16_t* qrot, const uint16_t* kv, const int* bt, const int* used,
    const int* cu, const int* len, const float* sinks, uint16_t* out, float* scratch, SinkMeta m) {
    constexpr int NT = HG * 128, LOADS = (kPieces + NT - 1) / NT;
    static_assert(LOADS <= 32, "staged mask bits");
    __shared__ __align__(16) uint16_t keys[2][16 * kPitch];
    __shared__ __align__(32) float partial[HG * 4][32 * 8];
    __shared__ int visible[2][16];
    __shared__ long long rows_at[2][16];
    __shared__ int span_pages[kSpanPages];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, quarter = wave & 3,
              group = wave >> 2, r = blockIdx.x, split = blockIdx.z;
    const int head = blockIdx.y * 16 * HG + group * 16 + (lane & 15);
    const bool mine = head < m.heads;
    int lo, hi, trow;
    sink_range<SEL>(used, cu, len, m, r, lo, hi, trow);
    int begin = lo, end = hi;
    if (m.splits > 1) {
        /* Slot `split` is the super-chunk [lo + SC split, lo + SC (split + 1)); the law's last
         * slot takes the rest of the range. */
        const long long sc = (long long)lo + ((long long)split << m.sc_shift),
                   sc_end = sc + (1LL << m.sc_shift);
        begin = sc < lo ? lo : sc < hi ? (int)sc : hi;
        if (split + 1 < m.law) end = sc_end < begin ? begin : sc_end < hi ? (int)sc_end : hi;
    }
    if (m.splits > 1 && begin >= end) {
        /* An empty slot: the merge skips it by its denominator and reads nothing else. */
        if (mine && !quarter && lane < 16) {
            float* dst = scratch + (((long long)r * m.heads + head) * m.splits + split) * PART;
            dst[D] = -INFINITY;
            dst[D + 1] = 0;
        }
        return;
    }
    /* The 16-token chunks from the slot's first key; a slot is a whole number of them, so the
     * fused form's walk from `lo` meets every slot boundary at a chunk boundary. */
    const int first = begin;
    auto slot_of = [&](int token) { return min((token - lo) >> m.sc_shift, m.law - 1); };
    /* Query fragments: eight latent k-steps of the wave's quarter, one rotary. */
    v8bf query[9];
    {
        const uint16_t* qp =
            q + (long long)r * m.qr + (long long)(mine ? head : 0) * m.qh + quarter * 128 + 4 * (lane >> 4);
        const uint16_t* rp =
            qrot + (long long)r * m.rr + (long long)(mine ? head : 0) * m.rh + quarter * 16 + 4 * (lane >> 4);
#pragma unroll
        for (int s = 0; s < 9; ++s) {
            const uint16_t* src = s < 8 ? qp + s * 16 : rp;
            union {
                uint16_t h[8];
                v8bf f;
            } u;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                u.h[e] = mine ? src[e] : (uint16_t)0;
                u.h[4 + e] = mine ? src[8 + e] : (uint16_t)0;
            }
            query[s] = u.f;
        }
    }
    const int* const table = bt + (long long)trow * m.bp;
    /* The cache offset of each token of a chunk, or -1 where nothing is visible, read from the
     * table two chunks ahead by sixteen lanes and passed through LDS, so a key load never waits
     * on a table load. */
    /* A split's whole table slice, where it fits, is read into LDS at once in the prologue: one
     * table round trip a workgroup instead of one every few chunks (a table read cold from
     * memory otherwise stalls the chunk that crosses into its next cache line). */
    const int first_block = begin >> m.bs_shift,
              span = begin < end ? ((end - 1) >> m.bs_shift) - first_block + 1 : 0;
    const bool spanned = span <= kSpanPages;
    auto token_at = [&](int token) -> long long {
        if (token < begin || token >= end) return -1;
        const int block = token >> m.bs_shift;
        const int page = spanned ? span_pages[block - first_block] : table[(long long)block * m.bc];
        return page >= 0 && page < m.pages
                   ? (long long)page * m.kp + (long long)(token & (m.bs - 1)) * m.kt
                   : -1;
    };
    uint4 staged[LOADS];
    unsigned staged_ok = 0;
    auto fetch = [&](int slot) {
#pragma unroll
        for (int j = 0; j < LOADS; ++j) {
            const int flat = tid + j * NT;
            if (LOADS * NT > kPieces && flat >= kPieces) continue;
            const int piece = flat % (W / 8);
            const long long row = rows_at[slot][flat / (W / 8)];
            const bool ok = row >= 0;
            const long long at = ok ? row + piece * 8 : 0;
            if constexpr (WIDE) {
                staged[j] = *(const uint4*)(kv + at);
                if (!ok) staged[j] = make_uint4(0, 0, 0, 0);
            } else {
                uint32_t w[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    w[i] = ok ? (uint32_t)kv[at + 2 * i] | (uint32_t)kv[at + 2 * i + 1] << 16 : 0u;
                staged[j] = make_uint4(w[0], w[1], w[2], w[3]);
            }
            staged_ok = ok ? staged_ok | 1u << j : staged_ok & ~(1u << j);
        }
    };
    auto stash = [&](int b) {
#pragma unroll
        for (int j = 0; j < LOADS; ++j) {
            const int flat = tid + j * NT;
            if (LOADS * NT > kPieces && flat >= kPieces) continue;
            const int token = flat / (W / 8), piece = flat % (W / 8);
            *(uint4*)(keys[b] + token * kPitch + piece * 8) = staged[j];
            if (!piece) visible[b][token] = (int)(staged_ok >> j & 1u);
        }
    };
    v8f value[8];
#pragma unroll
    for (int mt = 0; mt < 8; ++mt) value[mt] = (v8f){ 0 };
    float run_max = -INFINITY, run_den = 0;
    /* The fused form's accumulator: the row's (head's) maximum and denominator in registers, its
     * 512 values in the scratch, each lane reading back only what it wrote. */
    float acc_m = -INFINITY, acc_d = 0;
    float* const acc = scratch + ((long long)r * m.heads + (mine ? head : 0)) * PART;
    auto fold = [&]() {
        if (run_den > 0 && mine) {
            if (!(acc_d > 0)) {
                acc_m = run_max;
                acc_d = run_den;
#pragma unroll
                for (int mt = 0; mt < 8; ++mt) {
                    float* const to = acc + quarter * 128 + 8 * (lane >> 4) + mt * 16;
                    *(float4*)to = make_float4(value[mt][0], value[mt][1], value[mt][2], value[mt][3]);
                    *(float4*)(to + 4) =
                        make_float4(value[mt][4], value[mt][5], value[mt][6], value[mt][7]);
                }
            } else {
                /* Every load first (the chunk's staged registers are dead here), then the folds. */
                float4 old[16];
                const float* const at = acc + quarter * 128 + 8 * (lane >> 4);
#pragma unroll
                for (int mt = 0; mt < 8; ++mt) {
                    old[2 * mt] = *(const float4*)(at + mt * 16);
                    old[2 * mt + 1] = *(const float4*)(at + mt * 16 + 4);
                }
                const SinkFold f = sink_fold(acc_m, run_max);
                acc_d = fmaf(run_den, f.b, acc_d * f.a);
                acc_m = f.m;
#pragma unroll
                for (int mt = 0; mt < 8; ++mt) {
                    float4 lo4 = old[2 * mt], hi4 = old[2 * mt + 1];
                    lo4.x = fmaf(value[mt][0], f.b, lo4.x * f.a);
                    lo4.y = fmaf(value[mt][1], f.b, lo4.y * f.a);
                    lo4.z = fmaf(value[mt][2], f.b, lo4.z * f.a);
                    lo4.w = fmaf(value[mt][3], f.b, lo4.w * f.a);
                    hi4.x = fmaf(value[mt][4], f.b, hi4.x * f.a);
                    hi4.y = fmaf(value[mt][5], f.b, hi4.y * f.a);
                    hi4.z = fmaf(value[mt][6], f.b, hi4.z * f.a);
                    hi4.w = fmaf(value[mt][7], f.b, hi4.w * f.a);
                    *(float4*)(acc + quarter * 128 + 8 * (lane >> 4) + mt * 16) = lo4;
                    *(float4*)(acc + quarter * 128 + 8 * (lane >> 4) + mt * 16 + 4) = hi4;
                }
            }
        }
        run_max = -INFINITY;
        run_den = 0;
#pragma unroll
        for (int mt = 0; mt < 8; ++mt) value[mt] = (v8f){ 0 };
    };
    int slot = slot_of(first);
    if (begin < end) {
        if (spanned)
            for (int i = tid; i < span; i += NT) span_pages[i] = table[(long long)(first_block + i) * m.bc];
        __syncthreads();
        if (tid < 32) rows_at[tid >> 4][tid & 15] = token_at(first + tid);
        __syncthreads();
        fetch(0);
        stash(0);
    }
    __syncthreads();
    int buf = 0;
    for (int start = first; begin < end && start < end; start += 16) {
        if (FUSED && slot_of(start) != slot) {
            fold();
            slot = slot_of(start);
        }
        const bool more = start + 16 < end;
        if (more) fetch(buf ^ 1);
        long long ahead = -1;
        if (tid < 16 && start + 32 < end) ahead = token_at(start + 32 + tid);
        const uint16_t* tile = keys[buf];
        v8f dot = { 0 };
#pragma unroll
        for (int s = 0; s < 8; ++s)
            dot = mma(frag(tile, kPitch, 0, quarter * 128 + s * 16, lane), query[s], dot);
        dot = mma(frag(tile, kPitch, 0, D + quarter * 16, lane), query[8], dot);
        *(v8f*)&partial[wave][lane * 8] = dot;
        __syncthreads();
        v8f score = *(const v8f*)&partial[group * 4][lane * 8];
#pragma unroll
        for (int k = 1; k < 4; ++k) score += *(const v8f*)&partial[group * 4 + k][lane * 8];
        float chunk_max = -INFINITY;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            score[e] = visible[buf][8 * (lane >> 4) + e] ? score[e] * m.scale : -INFINITY;
            chunk_max = fmaxf(chunk_max, score[e]);
        }
        chunk_max = fmaxf(chunk_max, swap16(chunk_max));
        const float next_max = fmaxf(run_max, chunk_max);
        const float alpha = isfinite(next_max) ? expf(run_max - next_max) : 1;
        float p[8], sum = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            p[e] = isfinite(score[e]) ? expf(score[e] - next_max) : 0;
            sum += p[e];
        }
        /* Both halves add tokens 0-7 before 8-15, so the sixteen lanes of a head agree. */
        const float other = swap16(sum);
        run_den = run_den * alpha + (lane < 16 ? sum + other : other + sum);
        run_max = next_max;
        /* A B fragment's lane holds its head's tokens 0-3 and 8-11 (lanes 0-15) or 4-7 and 12-15
         * (16-31); a score lane holds 0-7 or 8-15. Each half sends the other the four it needs. */
        uint32_t phi[4], plo[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint16_t h0 = f32_to_bf16(p[2 * i]), h1 = f32_to_bf16(p[2 * i + 1]);
            phi[i] = (uint32_t)h0 | (uint32_t)h1 << 16;
            plo[i] = (uint32_t)f32_to_bf16(p[2 * i] - bf16_to_f32(h0)) |
                     (uint32_t)f32_to_bf16(p[2 * i + 1] - bf16_to_f32(h1)) << 16;
        }
        const bool low = lane < 16;
        const uint32_t hx0 = r4d_xor_lane_u32<16>(low ? phi[2] : phi[0]),
                       hx1 = r4d_xor_lane_u32<16>(low ? phi[3] : phi[1]),
                       lx0 = r4d_xor_lane_u32<16>(low ? plo[2] : plo[0]),
                       lx1 = r4d_xor_lane_u32<16>(low ? plo[3] : plo[1]);
        union {
            uint32_t w[4];
            v8bf f;
        } ph, pl;
        ph.w[0] = low ? phi[0] : hx0;
        ph.w[1] = low ? phi[1] : hx1;
        ph.w[2] = low ? hx0 : phi[2];
        ph.w[3] = low ? hx1 : phi[3];
        pl.w[0] = low ? plo[0] : lx0;
        pl.w[1] = low ? plo[1] : lx1;
        pl.w[2] = low ? lx0 : plo[2];
        pl.w[3] = low ? lx1 : plo[3];
#pragma unroll
        for (int mt = 0; mt < 8; ++mt) {
            const auto v = fragT(tile, kPitch, quarter * 128 + mt * 16, 0, lane);
            value[mt] *= alpha;
            value[mt] = mma(v, ph.f, value[mt]);
            value[mt] = mma(v, pl.f, value[mt]);
        }
        if (more) stash(buf ^ 1);
        if (tid < 16) rows_at[buf][tid] = ahead;
        __syncthreads();
        buf ^= 1;
    }
    if (!mine) return;
    if (FUSED) {
        fold();
        const float inv = sink_inv(acc_m, acc_d, sinks[(long long)head * m.ss]);
#pragma unroll
        for (int mt = 0; mt < 8; ++mt)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int col = quarter * 128 + mt * 16 + 8 * (lane >> 4) + e;
                out[(long long)r * m.orr + (long long)head * m.oh + col] =
                    f32_to_bf16(acc_d > 0 ? acc[col] * inv : 0.f);
            }
        return;
    }
    if (m.splits == 1) {
        /* The sink joins the softmax once (sink_inv). One slot: the law's bytes as well. */
        const float inv = sink_inv(run_max, run_den, sinks[(long long)head * m.ss]);
#pragma unroll
        for (int mt = 0; mt < 8; ++mt)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int col = quarter * 128 + mt * 16 + 8 * (lane >> 4) + e;
                out[(long long)r * m.orr + (long long)head * m.oh + col] = f32_to_bf16(value[mt][e] * inv);
            }
        return;
    }
    float* dst = scratch + (((long long)r * m.heads + head) * m.splits + split) * PART;
#pragma unroll
    for (int mt = 0; mt < 8; ++mt)
#pragma unroll
        for (int e = 0; e < 8; ++e) dst[quarter * 128 + mt * 16 + 8 * (lane >> 4) + e] = value[mt][e];
    if (!quarter && lane < 16) {
        dst[D] = run_max;
        dst[D + 1] = run_den;
    }
}

/* A workgroup a (row, head), thread t holding columns 4t..4t+3: every slot's maximum and partial
 * loaded at once (no serial dependency between the loads), then the slots with a visible key folded
 * in ascending order (sink_fold, the fused form's rule) and the sink once. An empty slot (den 0)
 * wrote only its maximum and denominator; its values are not read. */
__device__ __forceinline__ void merge_sink_body(const float* scratch, const float* sinks,
                                                uint16_t* out, SinkMeta m) {
    const int t = threadIdx.x, head = blockIdx.y, r = blockIdx.x;
    const float* src = scratch + ((long long)r * m.heads + head) * m.splits * PART;
    float part_max[kMaxSplits], part_den[kMaxSplits];
    float4 part[kMaxSplits];
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s)
        if (s < m.splits) {
            part_max[s] = src[s * PART + D];
            part_den[s] = src[s * PART + D + 1];
            part[s] = part_den[s] > 0 ? *(const float4*)(src + s * PART + 4 * t)
                                      : make_float4(0, 0, 0, 0);
        }
    float4 value = make_float4(0, 0, 0, 0);
    float acc_m = -INFINITY, den = 0;
#pragma unroll
    for (int s = 0; s < kMaxSplits; ++s) {
        if (!(s < m.splits && part_den[s] > 0)) continue;
        if (!(den > 0)) {
            acc_m = part_max[s];
            den = part_den[s];
            value = part[s];
            continue;
        }
        const SinkFold f = sink_fold(acc_m, part_max[s]);
        den = fmaf(part_den[s], f.b, den * f.a);
        acc_m = f.m;
        value.x = fmaf(part[s].x, f.b, value.x * f.a);
        value.y = fmaf(part[s].y, f.b, value.y * f.a);
        value.z = fmaf(part[s].z, f.b, value.z * f.a);
        value.w = fmaf(part[s].w, f.b, value.w * f.a);
    }
    const float inv = sink_inv(acc_m, den, sinks[(long long)head * m.ss]);
    uint16_t* const o = out + (long long)r * m.orr + (long long)head * m.oh + 4 * t;
    o[0] = f32_to_bf16(value.x * inv);
    o[1] = f32_to_bf16(value.y * inv);
    o[2] = f32_to_bf16(value.z * inv);
    o[3] = f32_to_bf16(value.w * inv);
}

/* CH `r4dchain`: the chained forms (r4d_chain.h) of both kernels of the launch. The bodies keep
 * their early returns (!mine lanes, empty slots); the wrappers bracket them so every block counts
 * once. The sinks, block table and cu_seqlens are scalar loads, which the wait's acquire covers. */
template <int HG, bool WIDE, bool SEL, bool FUSED, class CH>
__global__ __launch_bounds__(HG * 128) void mla_sink_rows(
    const uint16_t* q, const uint16_t* qrot, const uint16_t* kv, const int* bt, const int* used,
    const int* cu, const int* len, const float* sinks, uint16_t* out, float* scratch, SinkMeta m) {
    if (CH::on) r4d_chain_wait();
    mla_sink_rows_body<HG, WIDE, SEL, FUSED>(q, qrot, kv, bt, used, cu, len, sinks, out, scratch, m);
    if (CH::on) r4d_chain_done();
}

template <class CH>
__global__ __launch_bounds__(128) void merge_sink(const float* scratch, const float* sinks,
                                                  uint16_t* out, SinkMeta m) {
    if (CH::on) r4d_chain_wait();
    merge_sink_body(scratch, sinks, out, m);
    if (CH::on) r4d_chain_done();
}

int sink_shift(bool sel, long long window) { return !sel && window > 0 ? kShiftWindow : kShiftFull; }
/* The law's slots: a window's range in super-chunks, window / SC rounded up (8 at 512), at most
 * kMaxSplits; kMaxSplits at window 0 and in the selection form. A function of the op and the
 * window alone. */
int sink_law(bool sel, long long window) {
    if (sel || window <= 0) return kMaxSplits;
    const long long sc = 1LL << kShiftWindow;
    return (int)std::min<long long>(kMaxSplits, (window + sc - 1) / sc);
}
/* The slots a launch needs: no visible token lies at or past the table's capacity, so a slot past
 * it is empty in every row. Moves no bits (empty slots are skipped). */
int sink_slots(bool sel, long long window, long long cap) {
    const long long need = cap < 1 ? 1 : ((cap - 1) >> sink_shift(sel, window)) + 1;
    return (int)std::min<long long>(sink_law(sel, window), need);
}
/* The most rows the launch splits: the form's cap, `stable 1` past `stable 0`'s. Moves no bits (both
 * forms run the law). */
long long sink_rows_cap(bool sel, long long window, bool stable) {
    if (stable) return !sel && window > 0 ? kRowsStableWindow : kRowsStableFull;
    return !sel && window > 0 ? kRowsWindow : kRowsFull;
}

void heads_view(RadTensor& t, long long heads, long long width) {
    if (heads > 0 && t.rank == 2 && t.shape[1] == heads * width) {
        t.rank = 3;
        t.shape[1] = heads;
        t.shape[2] = width;
        t.stride[2] = t.stride[1];
        t.stride[1] *= width;
    }
}

bool row_tensor_ok(const RadTensor& t, long long rows, long long heads, long long width) {
    return t.rank == 3 && t.shape[0] == rows && t.shape[1] == heads && t.shape[2] == width &&
           t.stride[2] == 1 && t.stride[1] >= width && t.stride[0] >= heads * t.stride[1];
}

/* Operands, paged: q [M, heads, 512], q_rope [M, heads, 64], kv [pages, 1, bs, >= 576],
 * block_table [seqs, blocks], seqused [seqs], cu_seqlens [seqs + 1], sinks [>= heads] f32, out
 * [M, heads, 512]. Selection: q, q_rope, kv, bt [>= M, blocks], length [>= M], sinks, out. */
template <bool SEL>
int sink_body(const RadArgs* a, RadStream stream) {
    const int nt = SEL ? 7 : 8;
    if (!a || a->n_t < nt) return RAD_E_INVAL;
    for (int i = 0; i < nt; ++i)
        if (!a->t[i].data) return RAD_E_INVAL;
    RadTensor q = a->t[0], qr = a->t[1], o = a->t[nt - 1];
    const RadTensor &kv = a->t[2], &bt = a->t[3], &sk = a->t[nt - 2];
    const RadTensor* su = SEL ? nullptr : &a->t[4];
    const RadTensor* cu = SEL ? nullptr : &a->t[5];
    const RadTensor* len = SEL ? &a->t[4] : nullptr;
    const long long heads = r4d_geti_or(a, "heads", 0), bs = r4d_geti_or(a, "block_size", 0),
               window = SEL ? 0 : r4d_geti_or(a, "window", 0);
    const float scale = (float)r4d_getf(a, "scale", 0);
    const long long stable = r4d_geti_or(a, "stable", 0);
    heads_view(q, heads, D);
    heads_view(qr, heads, RD);
    heads_view(o, heads, D);
    if (q.dtype != RAD_BF16 || qr.dtype != RAD_BF16 || kv.dtype != RAD_BF16 ||
        o.dtype != RAD_BF16 || bt.dtype != RAD_I32 || sk.dtype != RAD_F32 ||
        (su && su->dtype != RAD_I32) || (cu && cu->dtype != RAD_I32) ||
        (len && len->dtype != RAD_I32))
        return RAD_E_DTYPE;
    const long long rows = q.shape[0];
    if (r4d_geti_or(a, "latent_dim", 0) != D || r4d_geti_or(a, "rope_dim", 0) != RD ||
        stable < 0 || stable > 1 || heads < 8 || heads > 64 || heads % 8 || bs < 1 || bs > 256 || (bs & (bs - 1)) || window < 0 ||
        window > INT_MAX || !std::isfinite(scale) || scale <= 0 || q.rank != 3 || rows < 1 ||
        rows > 65535 || !row_tensor_ok(q, rows, heads, D) || !row_tensor_ok(qr, rows, heads, RD) ||
        !row_tensor_ok(o, rows, heads, D) || kv.rank != 4 || kv.shape[0] < 1 ||
        kv.shape[0] > INT_MAX || kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] < W ||
        kv.stride[3] != 1 || kv.stride[2] < W || kv.stride[0] < bs * kv.stride[2] ||
        bt.rank != 2 || bt.shape[1] < 1 || bt.shape[1] > INT_MAX / bs || bt.stride[1] < 1 ||
        bt.stride[0] < bt.shape[1] * bt.stride[1] || sk.rank != 1 || sk.shape[0] < heads ||
        sk.stride[0] < 1)
        return RAD_E_SHAPE;
    if (SEL && (bt.shape[0] < rows || len->rank != 1 || len->shape[0] < rows ||
                len->stride[0] < 1))
        return RAD_E_SHAPE;
    if (!SEL && (su->rank != 1 || su->shape[0] < 1 || bt.shape[0] < su->shape[0] ||
                 !r4d_contig(su) || !r4d_contig(cu) || r4d_numel(cu) != su->shape[0] + 1))
        return RAD_E_SHAPE;
    const int form = g_sink_form;
    const int law = sink_law(SEL, window);
    /* The split form up to the row cap (a workgroup a slot, merge_sink); past it, or where the
     * scratch holds no partials, `stable 1` runs the fused form (the same bytes: a workgroup walks
     * the row's slots and folds them as the merge does, an accumulator a row in the scratch) and
     * refuses a launch without even that scratch rather than change a bit; `stable 0` runs
     * unsplit. */
    /* The launch's slots: the selection's by its table width (the op's `blocks`); the paged form
     * always the law's count, because its table is the batch's block table, whose pitch can differ
     * between a recorded pass and its replay (a launch sized at a narrower table would drop the
     * keys of a later, longer context). */
    int splits = SEL ? sink_slots(true, 0, bt.shape[1] * bs) : sink_law(false, window), fused = 0;
    if (splits > 1) {
        const int64_t have = a->scratch ? a->scratch_bytes : 0;
        const bool split_fits = have >= rows * heads * splits * PART * 4,
                   fused_fits = have >= rows * heads * PART * 4;
        if (form == 4) {
            splits = 1;
        } else if (form == 6 ||
                   (stable && (rows > sink_rows_cap(SEL, window, true) || !split_fits))) {
            if (!fused_fits) return RAD_E_SCRATCH; /* never silently unsplit */
            splits = 1;
            fused = 1;
        } else if (rows > sink_rows_cap(SEL, window, stable != 0) || !split_fits) {
            splits = 1;
        }
    }
    const bool wide = form != 3 && !((uintptr_t)kv.data & 15) && !(kv.stride[0] % 8) &&
                      !(kv.stride[2] % 8);
    /* More heads a workgroup load each key chunk once for more of them: the widest group (64,
     * 32 or 16 heads) that still leaves 64 workgroups, two to each of the card's 32 WGPs. The
     * choice moves no bits. */
    int hg = 1;
    for (int g : { 4, 2 })
        if (heads > 16 * (g / 2) && rows * splits * ((heads + 16 * g - 1) / (16 * g)) >= 64) {
            hg = g;
            break;
        }
    if (form == 1) hg = 1;
    if (form == 2 && heads > 16) hg = 2;
    if (form == 5 && heads > 32) hg = 4;
    const SinkMeta m{ q.stride[0],
                      q.stride[1],
                      qr.stride[0],
                      qr.stride[1],
                      kv.stride[0],
                      kv.stride[2],
                      bt.stride[0],
                      bt.stride[1],
                      len ? (long long)len->stride[0] : 0LL,
                      o.stride[0],
                      o.stride[1],
                      sk.stride[0],
                      (int)rows,
                      (int)heads,
                      su ? (int)su->shape[0] : 0,
                      (int)bt.shape[1],
                      (int)bs,
                      __builtin_ctzl((unsigned long long)bs),
                      (int)kv.shape[0],
                      splits,
                      law,
                      sink_shift(SEL, window),
                      fused,
                      (int)window,
                      SEL ? 0 : (int)r4d_geti_or(a, "row0", 0),
                      scale };
    const auto* qp = (const uint16_t*)q.data;
    const auto* rp = (const uint16_t*)qr.data;
    const auto* kp = (const uint16_t*)kv.data;
    const auto* bp = (const int*)bt.data;
    const int* up = su ? (const int*)su->data : nullptr;
    const int* cp = cu ? (const int*)cu->data : nullptr;
    const int* lp = len ? (const int*)len->data : nullptr;
    const auto* sp = (const float*)sk.data;
    auto* op = (uint16_t*)o.data;
    auto* scr = (float*)a->scratch;
    const bool chain = r4d_geti_or(a, "chain", 0) != 0 && rows <= 64;   /* decode heights */
#define SINK_EMIT3(HG, WD, FU, CH)                                                                 \
    R4D_EMIT(stream, (mla_sink_rows<HG, WD, SEL, FU, CH>),                                         \
             dim3((unsigned)rows, (unsigned)((heads + 16 * HG - 1) / (16 * HG)), (unsigned)splits), \
             dim3(HG * 128), 0, qp, rp, kp, bp, up, cp, lp, sp, op, scr, m)
#define SINK_EMIT2(HG, WD, FU)                                                                     \
    (chain ? SINK_EMIT3(HG, WD, FU, r4dchain) : SINK_EMIT3(HG, WD, FU, R4dUnchained))
#define SINK_EMIT(HG, WD) (fused ? SINK_EMIT2(HG, WD, true) : SINK_EMIT2(HG, WD, false))
    int rc;
    if (hg == 4)
        rc = wide ? SINK_EMIT(4, true) : SINK_EMIT(4, false);
    else if (hg == 2)
        rc = wide ? SINK_EMIT(2, true) : SINK_EMIT(2, false);
    else
        rc = wide ? SINK_EMIT(1, true) : SINK_EMIT(1, false);
#undef SINK_EMIT
#undef SINK_EMIT2
#undef SINK_EMIT3
    if (rc == RAD_OK && splits > 1)
        rc = chain ? R4D_EMIT(stream, (merge_sink<r4dchain>), dim3((unsigned)rows, (unsigned)heads),
                              dim3(128), 0, (const float*)scr, sp, op, m)
                   : R4D_EMIT(stream, (merge_sink<R4dUnchained>),
                              dim3((unsigned)rows, (unsigned)heads), dim3(128), 0,
                              (const float*)scr, sp, op, m);
    return rc;
}

/* `stable 1`: every row of the declared M; `stable 0`: up to the split row cap. The selection's
 * `blocks` bounds its slots; a paged table's width is not known here, so the law's count. */
int64_t sink_scratch(const RadArgs* a, bool sel) {
    const long long heads = r4d_geti_or(a, "heads", 0), rows = r4d_geti_or(a, "M", kRowsFull),
               window = sel ? 0 : r4d_geti_or(a, "window", 0), stable = r4d_geti_or(a, "stable", 0),
               blocks = sel ? r4d_geti_or(a, "blocks", 0) : 0, bs = r4d_geti_or(a, "block_size", 0);
    if (heads < 1 || heads > 64 || rows < 1 || rows > 65535 || window < 0 || stable < 0 ||
        stable > 1)
        return RAD_E_SHAPE;
    const int slots = sel && blocks > 0 && bs > 0 ? sink_slots(true, 0, blocks * bs)
                                                  : sink_law(sel, window);
    if (slots <= 1) return 0;
    const long long cap = std::min(rows, sink_rows_cap(sel, window, stable != 0));
    /* stable 1: the split form's partials up to the cap, the fused form's accumulator past it */
    return (stable ? std::max(cap * slots, rows) : cap * slots) * heads * PART * 4;
}
}  // namespace

extern "C" int r4d_rad_mla_attn_paged_sink(const RadArgs* a, RadStream stream) {
    return sink_body<false>(a, stream);
}
extern "C" int r4d_rad_mla_attn_sel_sink(const RadArgs* a, RadStream stream) {
    return sink_body<true>(a, stream);
}
/* Per row: 8 slots at a window of 512 and 64 heads 1.06 MiB (stable 0: 16 rows, 16.9 MiB; stable 1:
 * 48 rows at 32 heads, 24.2 MiB); 20 slots at window 0 and 32 heads 1.32 MiB (stable 0: 32 rows,
 * 42.3 MiB; stable 1: 48 rows, 60.4 MiB); the selection at blocks x block_size 2048 has 16 slots
 * (1.06 MiB a row at 32 heads: stable 1 48 rows, 48.4 MiB; 24.2 MiB at the 16 heads a rank has at
 * TP2). */
extern "C" int64_t r4d_rad_mla_attn_paged_sink_scratch(const RadArgs* a) {
    return sink_scratch(a, false);
}
extern "C" int64_t r4d_rad_mla_attn_sel_sink_scratch(const RadArgs* a) {
    return sink_scratch(a, true);
}
extern "C" void r4d_mla_sink_pin_form(int form) { g_sink_form = form; }
