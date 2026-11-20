/* table_warm.h -- Brilliance: --ngram-warm, an n-gram table read once at load into the OS file cache.
 *
 * Under --ngram-placement disk the host gather reads the table's rows from the container. Off Linux
 * it reads them through the container's mapping (libavx avx_ngram.cpp: no row cache, no direct
 * reads), so a row the page cache does not hold is a page fault served by the drive, serially, with
 * the card waiting on the gather's join. On two R9700s with Flash-Next (a 47.68 GiB table) a cold
 * gather took 689 us against 16 us warm, one-stream decode lost about 9% and eight-stream about 22%,
 * and a cold 32K prompt took 22.4 s against 5.7 s (the 2026-11-20 TP2 census).
 *
 * --ngram-warm reads the table's bytes once, front to back, with large buffered reads, and keeps
 * nothing: the pages land in the file cache as any read file's do. On Windows the cache manager's
 * pages and a mapped view's are the same pages (one data section per file), so the gather's later
 * faults are soft. Nothing is locked or pinned and no pin limit or guard is touched: the cache stays
 * reclaimable, and memory pressure can give pages back to the drive, which then serves them as
 * without the warm.
 *
 * ngram_warm_plan is the decision alone, from facts the engine gathers, so it is tested without a
 * model or a card (tests/ngram_warm_test.cpp); table_warm_read is the read. */
#pragma once
#include "radfile.h"
#include "../rad_core.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rad {

/* Host memory --ngram-warm leaves available beside the table it reads: the warm is skipped unless
 * the table and this much more are available. It is the operator's host-RAM floor on the machine
 * the option was made for; the engine raises it to --host-headroom-mib where that is larger. */
constexpr int64_t kNgramWarmFloor = (int64_t)24 << 30;

/* What the decision reads. Gathered by Engine::warm_ngram_table after the weight load. */
struct NgramWarmFacts {
    bool           asked        = false;   /* --ngram-warm */
    int64_t        table_bytes  = 0;       /* the n-gram weights the container's mapping serves; 0 for none */
    TablePlacement resolved     = TablePlacement::Disk;   /* Config::ngram_resolved, after the load */
    bool           own_file     = false;   /* disk=DIR: the table is read from its own file */
    bool           direct_reads = false;   /* the gather reads the file with direct I/O (Linux's row cache) */
    int64_t        avail        = -1;      /* host memory available now (MemAvailable); -1 unreadable */
    int64_t        floor        = kNgramWarmFloor;   /* host memory to leave beside the table */
};

struct NgramWarmPlan {
    bool        warm = false;
    std::string why;    /* why it reads or skips, for the startup log and the decision report */
};

/* Whether --ngram-warm reads the table now, and why. It reads only when it was asked for, the
 * model has an n-gram table, the table resolved to the disk tier and is read from the container's
 * mapping through the page cache, and the host has the table's size plus `floor` available. Never
 * forced: every other case is a skip with its reason. */
NgramWarmPlan ngram_warm_plan(const NgramWarmFacts& f);

/* The byte ranges (file offset, length) of the container to read, from what the ranks publish:
 * every rank maps the same table, so a range is listed once; overlapping and touching ranges are
 * merged and empty ones dropped. Sorted by offset. */
std::vector<std::pair<int64_t, int64_t>> ngram_warm_ranges(std::vector<std::pair<int64_t, int64_t>> r);

/* Read the container's `ranges` once, front to back, through the file cache, keeping nothing: up to
 * eight readers, each one contiguous span in 16 MiB reads into a scratch buffer of its own. No
 * lock, no pin, no drop behind. RAD_OK with `*bytes` read and `*secs` taken; on a failure an error
 * with `why`, and `*bytes` what was read before it. */
int table_warm_read(const RadFile& file, const std::vector<std::pair<int64_t, int64_t>>& ranges,
                    int64_t* bytes, double* secs, std::string* why);

}  /* namespace rad */
