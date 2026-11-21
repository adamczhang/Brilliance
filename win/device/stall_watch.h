/* stall_watch.h -- Brilliance: the native HIP backend's stall watch, as a pure account.
 *
 * win/device/hipnative.cpp holds the one instance (namespace stall) and gives it the time; this
 * header holds the decisions, so a host test can drive them with a clock of its own
 * (win/tests/stall_watch_test.cpp).
 *
 * WHAT IS PROGRESS. HIP hides its queues, so the AQL backend's read indices have no twin here. The
 * host sees progress when
 *   - a blocking wait (stream or event synchronize) returns, or an event query finds its work done;
 *   - a BEACON moves: behind every `every`-th op the engine issues on a stream, the stream itself
 *     writes a fresh serial into a word of pinned host memory (hipStreamWriteValue64, a command
 *     processor write that makes HIP submit nothing). The watcher reads the words with a plain load,
 *     no HIP call. A word that changed since the last reading is work the card finished.
 *
 * Without the beacons a long step looked like a stall: a 32K-token prefill on one card is one
 * host wait of 8 to 25 s with nothing in it the host could see finish, so after kStallMs the watch
 * logged an E line and kicked healthy streams (2026-11-20, Gemma 4 12B and 31B, qualify-new-models
 * finding 1). With them, such a step moves its beacons every fraction of a second.
 *
 * WHAT IS A STALL. Some wait is blocked, and nothing above -- no wait returning, no event query,
 * no beacon -- has moved for stall_ms. A real stall still reads as one:
 *   - a hung kernel, or a wait that never ends: its stream's beacons stop with it;
 *   - work HIP never submitted: the beacons behind it are not submitted either;
 *   - a lost device: nothing moves.
 * Beacons only report work already queued finishing: once a hung step's queue drains, they stop,
 * and the stall is reported stall_ms later.
 *
 * Not thread-safe by itself; the caller holds a lock around every call. */
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace rad {

class StallWatch {
public:
    explicit StallWatch(int64_t stall_ms = 5000) : stall_ms_(stall_ms) {}

    int64_t stall_ms() const { return stall_ms_; }

    /* A blocking wait starts at `now`; its id ends it. */
    uint64_t wait_begin(int64_t now) {
        const uint64_t id = ++next_id_;
        waits_[id] = now;
        return id;
    }
    /* A wait returned: that is progress. Returns what progress() returns. */
    int64_t wait_end(uint64_t id, int64_t now) {
        waits_.erase(id);
        return progress(now);
    }

    /* Something finished at `now`. A stall reported before it is over; returns the time of the
     * kick that stall made (> 0), so the caller can say the device resumed after it, else 0. */
    int64_t progress(int64_t now) {
        if (now > last_progress_) last_progress_ = now;
        reported_ = false;
        const int64_t k = kicked_at_;
        kicked_at_ = 0;
        return k;
    }

    /* The beacon words as read at `now`, one per slot (a slot is a stream's word; 0 is a slot not
     * in use or not yet written). A slot whose value differs from the last reading is progress.
     * Returns what progress() returns when any moved, else 0; *moved says whether any did. */
    int64_t observe(const uint64_t* words, size_t n, int64_t now, bool* moved = nullptr) {
        if (seen_.size() < n) seen_.resize(n, 0);
        bool any = false;
        for (size_t i = 0; i < n; ++i) {
            if (words[i] != seen_[i]) {
                any = any || words[i] != 0;   /* a slot set back to 0 (reused) is not progress */
                seen_[i] = words[i];
            }
        }
        if (moved) *moved = any;
        if (!any) return 0;
        ++beacon_moves_;
        last_beacon_ = now;
        return progress(now);
    }

    struct Verdict {
        int64_t ms = 0;        /* how long nothing has moved while a wait is blocked; 0 below stall_ms */
        bool    report = false; /* the first verdict of this stall: log it and kick, once */
    };
    Verdict check(int64_t now) {
        Verdict v;
        int64_t oldest = -1;
        for (const auto& w : waits_)
            if (oldest < 0 || w.second < oldest) oldest = w.second;
        if (oldest < 0) return v;   /* nobody is waiting on the device */
        const int64_t since = oldest > last_progress_ ? oldest : last_progress_;
        const int64_t ms = now - since;
        if (ms < stall_ms_) return v;
        v.ms = ms;
        if (!reported_) {
            reported_ = true;
            v.report = true;
        }
        return v;
    }

    /* The kick the caller made for a reported stall, at `now`. */
    void kicked(int64_t now) { kicked_at_ = now; }

    /* For the stall's log line: how many readings found a beacon moved, and when the last did
     * (-1 never). */
    uint64_t beacon_moves() const { return beacon_moves_; }
    int64_t  last_beacon() const { return last_beacon_; }
    size_t   waiting() const { return waits_.size(); }

private:
    int64_t                     stall_ms_;
    std::map<uint64_t, int64_t> waits_;          /* wait id -> start */
    uint64_t                    next_id_ = 0;
    int64_t                     last_progress_ = 0;
    bool                        reported_ = false;
    int64_t                     kicked_at_ = 0;  /* the open kick's time, 0 when none */
    std::vector<uint64_t>       seen_;
    uint64_t                    beacon_moves_ = 0;
    int64_t                     last_beacon_ = -1;
};

/* When the issuing thread places a beacon: behind every `every`-th op it issues (0 is never). One
 * per issuing thread: a stream is issued from one thread. */
struct BeaconCadence {
    uint32_t every = 64;
    uint32_t n = 0;
    bool due() {
        if (every == 0) return false;
        if (++n < every) return false;
        n = 0;
        return true;
    }
};

}  /* namespace rad */
