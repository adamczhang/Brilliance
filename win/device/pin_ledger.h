/* pin_ledger.h -- Brilliance: what one process has pinned for its cards, as one account.
 *
 * Carried from Brilliance 0.2.2 (core/device/pin_ledger.h) unchanged in substance; the native HIP
 * backend (win/device/hipnative.cpp) holds the one instance, under its lock.
 *
 * Two parties pin host memory in a serving process. The engine does, through rad_dev_alloc, and
 * every such pin is taken from the process's limit before it is made. A kernel library does too
 * (libr4d's collectives' mailboxes, by hipHostMalloc or hipHostRegister of its own), and the limit
 * keeps a margin for those: past the true ceiling nothing fails, the next GPU operation hangs.
 *
 * The account, with no pin counted twice:
 *
 *   limit      what the process may pin, ALREADY less the margin (budget / devices - allowance)
 *   allowance  that margin: what the limit leaves for the library's pins and HIP's own
 *   engine     the engine's pins; each is refused when it would pass the limit
 *   lib        the library's pins, where a library reports them. The first `allowance` bytes cost
 *              the limit nothing; only the part past the allowance is charged, and a library pin
 *              is refused when that charge would pass the limit.
 *
 * A pure account with no locking of its own; the caller holds the lock. */
#pragma once
#include <algorithm>
#include <cstdint>

namespace rad {

struct PinLedger {
    int64_t engine = 0;
    int64_t lib = 0;
    int64_t allowance = 0;

    /* The library's pins past what the margin already covers. */
    int64_t lib_excess() const { return std::max<int64_t>(lib - allowance, 0); }
    /* What counts against the limit. */
    int64_t charged() const { return engine + lib_excess(); }

    /* `limit` below zero is "no limit known". */
    bool reserve_engine(int64_t bytes, int64_t limit) {
        if (bytes < 0) return false;
        if (limit >= 0 && charged() + bytes > limit) return false;
        engine += bytes;
        return true;
    }
    void release_engine(int64_t bytes) { engine = std::max<int64_t>(engine - bytes, 0); }

    bool reserve_lib(int64_t bytes, int64_t limit) {
        if (bytes < 0) return false;
        const int64_t excess = std::max<int64_t>(lib + bytes - allowance, 0);
        if (limit >= 0 && excess > 0 && engine + excess > limit) return false;
        lib += bytes;
        return true;
    }
    void release_lib(int64_t bytes) { lib = std::max<int64_t>(lib - bytes, 0); }
};

/* The process's pin limit from the shared-memory budget the OS states, the devices it sees and
 * BRILLIANCE_PIN_LIMIT_GIB (`env`, null when unset). Pure, so a host test can hand it any budget.
 *
 *   computed  budget / devices - margin, at least 0; -1 when the budget or the devices are unknown
 *   limit     the variable's value where it is at or below `computed` (it lowers the limit), else
 *             `computed`: past that budget a pin does not fail, the next GPU operation hangs, so
 *             the variable never raises the limit (2026-11-19, the multicard robustness run F3: 100
 *             on a 47.83 GiB limit stalled rank 1's pin with no log line). With no budget known
 *             the variable is all there is and is used as given.
 *   clamped   the variable asked for more than `computed`
 *   invalid   the variable is set but not a whole number of GiB above 0; it is ignored, as before
 *   requested the variable's value in bytes, 0 when unset or invalid */
struct PinLimitChoice {
    int64_t computed = -1;
    int64_t limit = -1;
    int64_t requested = 0;
    bool    from_env = false;
    bool    clamped = false;
    bool    invalid = false;
};

inline PinLimitChoice choose_pin_limit(int64_t budget, int devices, int64_t margin, const char* env) {
    PinLimitChoice c;
    if (budget > 0 && devices > 0) c.computed = std::max<int64_t>(budget / devices - margin, 0);
    c.limit = c.computed;
    if (!env || !*env) return c;
    int64_t gib = 0;
    const char* p = env;
    while (*p == ' ' || *p == '\t') ++p;
    const char* digits = p;
    constexpr int64_t kMaxGib = INT64_MAX >> 30;
    for (; *p >= '0' && *p <= '9'; ++p) gib = std::min<int64_t>(gib * 10 + (*p - '0'), kMaxGib);
    while (*p == ' ' || *p == '\t') ++p;
    if (p == digits || *p || gib <= 0) { c.invalid = true; return c; }
    c.requested = gib << 30;
    if (c.computed >= 0 && c.requested > c.computed) { c.clamped = true; return c; }
    c.limit = c.requested;
    c.from_env = true;
    return c;
}

}  /* namespace rad */
