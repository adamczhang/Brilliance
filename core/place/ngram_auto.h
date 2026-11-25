/* ngram_auto.h -- Brilliance: where --ngram-placement auto puts an n-gram table, as arithmetic.
 *
 * Auto takes the first tier that holds the table: the card, then a copy in RAM (pinned for a card's
 * gather, or locked for the host's), then the container read by the host gather. On 2026-11-22 a
 * one-card Flash-Next start on the test machine (125.5 GiB of RAM) chose the RAM tier for its 47.68 GiB table on
 * top of a 42 GiB pinned expert pool, because the room test kept only --host-headroom-mib (4 GiB)
 * to spare: it planned to leave about 4 GiB of the machine free. Two other servers started in the
 * same second, the table was pinned while about 12 GiB was available, and the next GPU call
 * (hipStreamCreateWithFlags) hung holding ~95 GiB until a reboot. Three rules follow:
 *
 *   - THE FLOOR. Auto's RAM tier keeps max(kHostRamFloor, the configured headroom) free after the
 *     table and the pools still to come, the floor the rest of the host admission keeps
 *     (place/host_admit.h). An explicit `ram` keeps the configured headroom, as before: the
 *     operator asked for it.
 *   - THE PIN LIMIT. A copy a card reads is pinned, and on Windows that counts against the
 *     process's limit (rad_dev_pin_limit); the plan's test is pinned + still + table <= cap.
 *     With two cards visible on the test machine the cap is 46.83 GiB, under the table alone.
 *   - THE FILE IS FAST ENOUGH (Windows). Since the disk tier's gather prefetches its rows through
 *     the mapping (libavx avx_ngram.cpp, 2026-11-22), a cold 12K prefill on one card runs at the
 *     warm speed (821 tok/s, against 379 without) and warm decode is the same. What a pinned copy
 *     still buys is the host gather's join, about 0.3 ms of a 13-16 ms TP2 step (the 11-22
 *     tp2-levers per-op tables), at the cost of RAM locked for the process's life and 12-51 s of
 *     load. So on Windows auto reads a table larger than kNgramPinShare of physical RAM from the
 *     file whenever a host gather can, even where a pin would fit. Smaller tables are still
 *     pinned when the floor allows. Linux's disk tier is a different reader (a row cache with
 *     direct reads, whose steps wait on the drive), so this rule is Windows-only.
 *
 * Pure: the caller measures the machine (meminfo_bytes, amdgpu_pin_cap, rad_dev_alloc_totals),
 * so a host test can hand it the test machine's 11-22 figures. */
#pragma once
#include "place/host_admit.h"

#include <algorithm>
#include <cstdint>

namespace rad {

/* The share of physical RAM above which Windows' auto reads the table from the file instead of
 * pinning it: a quarter. Flash-Next's 47.68 GiB table is 38% of the test machine's 125.5 GiB. */
constexpr int64_t kNgramPinShareDiv = 4;

/* The margin auto's RAM tier keeps free: the configured headroom, at least the floor. */
inline int64_t ngram_auto_margin(int64_t headroom) { return std::max(headroom, kHostRamFloor); }

/* True when the memory available now, less what is still to be allocated and the margin, cannot
 * hold `need`. Unreadable memory (`avail` < 0) is not short: the other limits decide. */
inline bool host_mem_short(int64_t avail, int64_t still, int64_t margin, int64_t need) {
    return avail >= 0 && avail - still - margin < need;
}

/* True when a pin of `need` would pass the process's pin limit `cap` with `pinned` already held
 * and `still` more to be pinned. `cap` <= 0 is no limit. */
inline bool pin_cap_short(int64_t cap, int64_t pinned, int64_t still, int64_t need) {
    return cap > 0 && pinned + still + need > cap;
}

/* True when the table is too large a share of physical RAM to pin when the file would serve
 * (Windows only; `total` <= 0, unreadable, never is). */
inline bool ngram_pin_too_large(int64_t need, int64_t total) {
    return total > 0 && need > total / kNgramPinShareDiv;
}

enum class NgramAutoTier { Vram, Ram, Disk, Nowhere };

struct NgramAutoIn {
    bool device = false;       /* a loaded kernel gathers the table on a card */
    bool host = false;         /* a loaded kernel gathers it on the host (the disk tier's reader) */
    bool card_holds = false;   /* the card holds it beside the other weights and a KV pool */
    bool ram_room = false;     /* the RAM copy passed the room test at ngram_auto_margin */
    bool file_preferred = false; /* this platform's file read is fast (Windows' prefetch) */
    int64_t need = 0;          /* the copy's bytes */
    int64_t total = -1;        /* physical RAM, -1 unreadable */
};

/* The tier auto takes, in order: the card, the file when the copy is too large a share of RAM and
 * the file is preferred, the RAM copy, the file. Nowhere when no tier is left. */
inline NgramAutoTier ngram_auto_choose(const NgramAutoIn& in) {
    if (in.device && in.card_holds) return NgramAutoTier::Vram;
    if (in.ram_room && !(in.host && in.file_preferred && ngram_pin_too_large(in.need, in.total)))
        return NgramAutoTier::Ram;
    if (in.host) return NgramAutoTier::Disk;
    return NgramAutoTier::Nowhere;
}

}  /* namespace rad */
