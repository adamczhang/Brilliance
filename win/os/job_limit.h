/* job_limit.h -- Brilliance: a Windows job object's memory limit, and what it means for the engine.
 *
 * THE CASE (GitHub issue #1, 2026-11-24). A launcher put radiance.exe in a job object with
 * JOB_OBJECT_LIMIT_JOB_MEMORY at 56 GiB. A job's memory limit is a limit on committed memory, and
 * WDDM charges what HIP allocates on a card with hipMalloc to the process's commit about one for
 * one, as it does pinned and ordinary memory. Two Flash-Next ranks want about 80 GiB of the two
 * together, so rank 1's 25.5 GiB weights pool was refused with hipErrorOutOfMemory while its card had
 * 28 GiB free, and the refusal blamed another process on the card.
 *
 * MEASURED ON THE TEST MACHINE (2026-11-25): a two-card Flash-Next server (HIP virtual memory off,
 * every pool a hipMalloc) committed 87.30 GiB privately for about 59 GiB of VRAM and 30 GiB of pins.
 * A one-card server with virtual memory on held 8.61 GiB of VRAM at 0.87 GiB of private commit:
 * virtual-memory pools are not charged. So the cards' pools count only without virtual memory.
 *
 * WHAT IS DONE ABOUT IT:
 *   - os::job_memory_limit() reads the limits of the job this process runs in (os_win32.cpp);
 *   - startup logs them beside the pin limit (win/device/hipnative.cpp);
 *   - Engine::plan refuses before any pool is allocated when the pools alone pass the limit
 *     (job_limit_verdict), naming the limit and the figure that would clear it;
 *   - the pool refusal and the device out-of-memory message name the limit (job_limit_note).
 *
 * The text and the arithmetic are here, header-only, so a host test reads them without a card
 * (win/tests/job_limit_test.cpp). */
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace rad {

/* The memory limits of the job object this process runs in. A job nested in another is read at the
 * innermost level, which is the one Windows names when it is queried without a handle. */
struct JobMemoryLimit {
    bool    in_job  = false;
    int64_t job     = -1;   /* JOB_OBJECT_LIMIT_JOB_MEMORY: every process in the job together; -1 none */
    int64_t process = -1;   /* JOB_OBJECT_LIMIT_PROCESS_MEMORY: each process in the job; -1 none */

    /* The limit that binds this process: the smaller one set, or -1 when neither is. */
    int64_t binding() const {
        if (job < 0) return process;
        if (process < 0) return job;
        return job < process ? job : process;
    }
};

inline std::string job_limit_gib(int64_t b) {
    char s[48];
    std::snprintf(s, sizeof s, "%.2f GiB", (double)b / (double)((int64_t)1 << 30));
    return s;
}

/* Which limit binds, in words: "the job's limit for all of its processes together, 56.00 GiB". */
inline std::string job_limit_which(const JobMemoryLimit& j) {
    const int64_t b = j.binding();
    if (b < 0) return {};
    const bool whole_job = j.job >= 0 && (j.process < 0 || j.job <= j.process);
    return (whole_job ? "the job's limit for all of its processes together, "
                      : "the job's limit for each process in it, ") + job_limit_gib(b);
}

/* A sentence for a refusal or an out-of-memory message; empty when no limit is set. */
inline std::string job_limit_note(const JobMemoryLimit& j) {
    if (j.binding() < 0) return {};
    return "This process runs in a Windows job object that limits its committed memory (" +
           job_limit_which(j) +
           "). WDDM charges VRAM allocated with hipMalloc (every pool when more than one card is "
           "visible) to commit, as it does pinned and ordinary memory, so the job's limit covers "
           "the cards' pools too and a pool can be refused as out of memory while its card has "
           "room. Raise or remove the limit in whatever started radiance (a wrapper, sandbox or "
           "service manager)";
}

struct JobLimitVerdict {
    bool        limited = false;   /* a limit is set: `message` says how much of it the plan uses */
    bool        refuse  = false;
    std::string message;
};

/* The plan against the job's limit, before any pool is allocated. `committed_now` is this process's
 * private commit now; `vram` every rank's card pools, which count only when `vram_charged` (HIP
 * virtual memory off: hipMalloc pools); `pins` every rank's pinned host pools still to come. These
 * are a floor on what a start commits (the activation arena, the staging and the loader's buffers
 * come on top), so a refusal is certain and a pass is not a promise. */
inline JobLimitVerdict job_limit_verdict(const JobMemoryLimit& j, int64_t committed_now,
                                         int64_t vram, int64_t pins, bool vram_charged = true) {
    JobLimitVerdict v;
    const int64_t lim = j.binding();
    if (lim < 0) return v;
    v.limited = true;
    const int64_t now  = committed_now > 0 ? committed_now : 0;
    if (!vram_charged) vram = 0;
    const std::string cards = vram_charged
        ? "the cards' pools, " + job_limit_gib(vram) + " over every rank, which WDDM charges to "
          "commit as hipMalloc memory, and "
        : std::string("(the cards' pools are HIP virtual memory, which commit does not count) ");
    const int64_t need = now + vram + pins;
    if (need > lim) {
        v.refuse = true;
        v.message = "this process runs in a Windows job object whose memory limit (" +
                    job_limit_which(j) + ") is below what this start must commit: " + cards +
                    "the pinned host pools, " + job_limit_gib(pins) + ", beside the " +
                    job_limit_gib(now) + " it has committed, " + job_limit_gib(need) +
                    " at least, so a pool would be refused as out of memory with room for it. "
                    "Raise the limit above " + job_limit_gib(need) + " (more for the activation "
                    "arena and staging), or remove it, in whatever started radiance";
        return v;
    }
    v.message = "a Windows job object limits this process's committed memory (" + job_limit_which(j) +
                "); " + (vram_charged ? "WDDM charges the cards' hipMalloc pools to it as well. "
                                      : "the cards' pools are HIP virtual memory, not charged. ") +
                "The pools need " + job_limit_gib(vram + pins) +
                " of it beside the " + job_limit_gib(now) + " committed, leaving " +
                job_limit_gib(lim - need) + " for the activation arena, staging and the loader";
    return v;
}

}  // namespace rad
