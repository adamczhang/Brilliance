/* coll_trace.h -- Brilliance: the last collectives and host joins each rank issued, for a stall.
 *
 * A step that hangs between ranks leaves nothing in the log: every thread waits, the cards poll a
 * mailbox (a polling kernel reads 0% on Windows' GPU-engine counters, so the cards look idle), and
 * the stall watch can name only streams. This keeps the last few hundred issues that matter across
 * ranks -- collectives and host-site ops (a host op is a join) -- with the rank, lane, op, kernel
 * row, libr4d instance, size and stream, and the stall watch prints each rank's tail
 * (win/device/hipnative.cpp, stall::kick). Two ranks whose tails differ show where they diverged.
 *
 * Cheap enough to stay on: one relaxed fetch_add and a few stores per collective, none for any
 * other op. Windows only (core/runtime/issue.cpp calls it under _WIN32). */
#ifndef BRL_COLL_TRACE_H
#define BRL_COLL_TRACE_H

#include <cstdint>

namespace rad {
namespace ctrace {

/* One issue. `host` marks a host-site op (its join waited for the device first); `rc` is the
 * launch's return. Strings must outlive the process's steps (op and row names do). */
void note(int rank, int lane, const char* op, const char* row, const void* inst, int64_t n,
          const void* stream, bool host, int rc);

/* Logs every rank's last issues, oldest first, with each rank's count of collectives. */
void dump();

}  // namespace ctrace
}  // namespace rad

#endif  /* BRL_COLL_TRACE_H */
