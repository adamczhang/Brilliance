/* coll_trace.cpp -- Brilliance: see coll_trace.h. */
#include "coll_trace.h"

#include <atomic>
#include <cstdio>
#include <string>

#include "rad_internal.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace rad {
namespace ctrace {
namespace {

constexpr int kRing  = 1024;   /* entries kept, all ranks together */
constexpr int kRanks = 16;
constexpr int kTail  = 24;     /* printed a rank */

struct Entry {
    std::atomic<uint64_t> seq{0};   /* 0: never written; the global issue number otherwise */
    int         rank = 0, lane = 0, rc = 0;
    bool        host = false;
    const char* op = nullptr;
    const char* row = nullptr;
    const void* inst = nullptr;
    const void* stream = nullptr;
    int64_t     n = 0;
    uint64_t    rank_coll = 0;      /* this rank's collective count, at this entry */
};

Entry                 g_ring[kRing];
std::atomic<uint64_t> g_next{0};
std::atomic<uint64_t> g_coll[kRanks];

}  // namespace

void note(int rank, int lane, const char* op, const char* row, const void* inst, int64_t n,
          const void* stream, bool host, int rc) {
    const uint64_t s = g_next.fetch_add(1, std::memory_order_relaxed) + 1;
    Entry& e = g_ring[s % kRing];
    const int r = rank >= 0 && rank < kRanks ? rank : kRanks - 1;
    e.rank = rank;
    e.lane = lane;
    e.rc = rc;
    e.host = host;
    e.op = op;
    e.row = row;
    e.inst = inst;
    e.stream = stream;
    e.n = n;
    e.rank_coll = host ? g_coll[r].load(std::memory_order_relaxed)
                       : g_coll[r].fetch_add(1, std::memory_order_relaxed) + 1;
    e.seq.store(s, std::memory_order_release);
}

void dump() {
    const uint64_t last = g_next.load(std::memory_order_acquire);
    if (last == 0) {
        RAD_WARN("device: collective trace: nothing issued yet");
        return;
    }
    for (int r = 0; r < kRanks; ++r) {
        /* This rank's newest kTail entries, found walking back through the ring. */
        const Entry* mine[kTail];
        int k = 0;
        for (uint64_t s = last; s > 0 && last - s < (uint64_t)kRing && k < kTail; --s) {
            const Entry& e = g_ring[s % kRing];
            if (e.seq.load(std::memory_order_acquire) != s || e.rank != r) continue;
            mine[k++] = &e;
        }
        if (k == 0) continue;
        std::string out;
        char b[320];
        for (int i = k - 1; i >= 0; --i) {
            const Entry& e = *mine[i];
            std::snprintf(b, sizeof b, "\n    #%llu %s lane %d  %s  [%s]  n=%lld  inst %p  stream %p%s",
                          (unsigned long long)e.rank_coll, e.host ? "host" : "coll", e.lane,
                          e.op ? e.op : "?", e.row ? e.row : "?", (long long)e.n, e.inst, e.stream,
                          e.rc < 0 ? "  (launch failed)" : "");
            out += b;
        }
        RAD_WARN("device: collective trace, rank %d: its last %d collective(s) and host op(s), "
                 "oldest first; %llu collective(s) issued in all:%s",
                 r, k, (unsigned long long)g_coll[r].load(std::memory_order_relaxed), out.c_str());
    }
    /* libr4d's host mailboxes, read where they live (pinned host memory): each rank's line holds
     * the handshake values its peer's blocks last published (r4d_win_mailbox_dump). */
    using DumpFn = int (*)(char*, int);
    HMODULE m = GetModuleHandleA("libr4d.dll");
    const DumpFn fn = m ? (DumpFn)(void*)GetProcAddress(m, "r4d_win_mailbox_dump") : nullptr;
    if (fn) {
        std::string buf(131072, '\0');   /* room for the host-side push check (r4d_ar_entry.hip) */
        const int n = fn(&buf[0], (int)buf.size());
        RAD_WARN("device: libr4d host mailboxes (%d two-rank instance(s)):%s", n, buf.c_str());
    }
}

}  // namespace ctrace
}  // namespace rad
