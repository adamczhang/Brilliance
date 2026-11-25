/* avx_ngram_pf.h -- Brilliance: the bookkeeping of the Windows n-gram prefetch, kept portable.
 *
 * On Windows the disk-tier n-gram gather reads the table as memory, through the container's mapping
 * (avx_ngram.cpp): a cold row is a page fault, served one at a time per thread. The gather therefore
 * hands each call's row pages to PrefetchVirtualMemory together first, and a read-ahead thread does
 * the same for the following chunks' ids (RowGather::ahead) and touches one byte per page, so the
 * pages are mapped for every thread before the gather reaches them.
 *
 * What is decided here, with no system call, so tests/ngram_prefetch_test.cpp checks it on any host:
 *   - span_pages / merge_pages: the pages a set of rows occupies, sorted and deduplicated, and
 *     those pages as ranges with adjacent pages merged (one PrefetchVirtualMemory entry each);
 *   - list_shift: whether a new ahead list continues the previous one, moved on by some ids;
 *   - AheadStream: which ids of a new list still need reading (pieces), how far the read-ahead has
 *     got (`done`), and whether a call's rows are already covered.
 *
 * Adapted from wmmunn's Windows n-gram prefetch (another Windows port of radiance, 2026-11-15). */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace avx::pf {

constexpr uintptr_t kPage = 4096;

/* A run of bytes a call is about to load: one row of one plane. */
struct Span {
    const void* p;
    size_t      bytes;
};

/* One prefetch entry: a run of whole pages. */
struct Range {
    uintptr_t addr;
    size_t    bytes;
};

/* Appends the page numbers (address / kPage) that [p, p + bytes) touches. Empty spans add none. */
inline void add_span_pages(const void* p, size_t bytes, std::vector<uintptr_t>& pages) {
    if (!p || bytes == 0) return;
    const uintptr_t lo = (uintptr_t)p;
    const uintptr_t hi = lo + (uintptr_t)bytes - 1;
    for (uintptr_t pg = lo / kPage; pg <= hi / kPage; ++pg) pages.push_back(pg);
}

/* Sorts `pages` and drops repeats. */
inline void unique_pages(std::vector<uintptr_t>& pages) {
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
}

/* The pages of n spans, sorted and unique, into `pages` (cleared first). */
inline void span_pages(const Span* s, size_t n, std::vector<uintptr_t>& pages) {
    pages.clear();
    for (size_t i = 0; i < n; ++i) add_span_pages(s[i].p, s[i].bytes, pages);
    unique_pages(pages);
}

/* The pages of the table rows ids[m] - voff (m < n where pick(m)), each `row_bytes` long, of a
 * table of n_vocab rows at `tab`, sorted and unique, into `pages`. Ids outside the table (padding,
 * another rank's vocabulary slice) have none. */
template <class Pick>
void row_pages(const uint8_t* tab, int64_t row_bytes, int64_t voff, int64_t n_vocab,
               const int32_t* ids, int64_t n, Pick pick, std::vector<uintptr_t>& pages) {
    pages.clear();
    for (int64_t m = 0; m < n; ++m) {
        if (!pick(m)) continue;
        const int64_t id = (int64_t)ids[m] - voff;
        if (id < 0 || id >= n_vocab) continue;
        add_span_pages(tab + id * row_bytes, (size_t)row_bytes, pages);
    }
    unique_pages(pages);
}

/* Sorted unique pages as ranges, adjacent pages merged, into `out` (cleared first). */
inline void merge_pages(const std::vector<uintptr_t>& pages, std::vector<Range>& out) {
    out.clear();
    for (size_t i = 0; i < pages.size();) {
        size_t j = i + 1;
        while (j < pages.size() && pages[j] == pages[j - 1] + 1) ++j;
        out.push_back({ pages[i] * kPage, (j - i) * (size_t)kPage });
        i = j;
    }
}

/* Whether the list w[0, na) continues the previous list p[0, np) moved on by some s ids, that is
 * w[0, k) == p[s, s + k) with k = min(np - s, na). The smallest such s is taken: it leaves the
 * fewest ids to hand over. Returns k (the ids of w already handed over) and sets *shift, or -1
 * when it does not continue it. An id repeats only where the text does, so few shifts pass the
 * first comparison; `tries` bounds a text that repeats a lot. This is the loop of Linux's
 * read_ahead (avx_ngram.cpp), which upstream owns: compare the two at each upstream merge. */
inline int64_t list_shift(const int32_t* p, int64_t np, const int32_t* w, int64_t na, int tries,
                          int64_t* shift) {
    if (na <= 0) return -1;
    for (int64_t s = 1, t = 0; s < np && t < tries; ++s) {
        if (p[s] != w[0]) continue;
        ++t;
        const int64_t k = std::min(np - s, na);
        if (std::equal(w, w + k, p + s)) {
            *shift = s;
            return k;
        }
    }
    return -1;
}

/* A piece of an ahead list for the read-ahead to read: at most a piece's ids, at stream positions
 * [start, end) (positions count ids from the stream's start), from list generation `gen`. */
struct Piece {
    uint64_t             gen   = 0;
    int64_t              start = 0, end = 0;
    std::vector<int32_t> ids;
};

/* One table's ahead list as last handed over, and how far the read-ahead has got through it. Every
 * id at a position before `done` has been prefetched and mapped. A new list that moves the last one
 * on keeps that progress; one that does not bumps `gen`, so its queued pieces are dropped.
 * Not thread-safe: the caller holds its lock. */
class AheadStream {
public:
    /* Hands over a new list w[0, na). Returns the pieces still to read, in list order (the nearest
     * chunk first), each at most `piece_rows` ids. *restarted is set when the list does not continue
     * the last one (the caller drops the last list's queued pieces). A repeat of the last list (the
     * other rank's call, or the same step again) hands over nothing. */
    std::vector<Piece> hand_over(const int32_t* w, int64_t na, size_t piece_rows, int tries,
                                 bool* restarted) {
        std::vector<Piece> out;
        *restarted = false;
        if (na < 0) na = 0;
        const int64_t np = (int64_t)ids_.size();
        if (np == na && std::equal(w, w + na, ids_.data())) return out;
        int64_t shift = 0;
        int64_t from = list_shift(ids_.data(), np, w, na, tries, &shift);
        if (from < 0) {
            *restarted = np > 0;
            ++gen_;
            base_ += np;
            done_ = base_;
            from = 0;
        } else {
            base_ += shift;
        }
        ids_.assign(w, w + na);
        if (piece_rows == 0) piece_rows = 1;
        for (int64_t m = from; m < na; m += (int64_t)piece_rows) {
            const int64_t e = std::min(na, m + (int64_t)piece_rows);
            Piece pc;
            pc.gen = gen_;
            pc.start = base_ + m;
            pc.end = base_ + e;
            pc.ids.assign(w + m, w + e);
            out.push_back(std::move(pc));
        }
        return out;
    }

    /* A piece the read-ahead has read and mapped. `done` moves only over contiguous pieces of the
     * current generation: one that starts past it (after a failed piece) leaves it, so a failed
     * piece's rows are never counted as mapped. */
    void complete(const Piece& pc) {
        if (pc.gen == gen_ && pc.start <= done_ && pc.end > done_) done_ = pc.end;
    }

    /* Whether the read-ahead has already read and mapped every row of a call with ids[0, M): its
     * ids are the head of the last list handed over, and `done` is past them. */
    bool covered(const int32_t* ids, int64_t M) const {
        return M > 0 && (int64_t)ids_.size() >= M && std::equal(ids, ids + M, ids_.data()) &&
               done_ >= base_ + M;
    }

    uint64_t gen() const { return gen_; }
    int64_t  base() const { return base_; }
    int64_t  done() const { return done_; }

private:
    std::vector<int32_t> ids_;
    int64_t              base_ = 0, done_ = 0;
    uint64_t             gen_  = 0;
};

}  /* namespace avx::pf */
