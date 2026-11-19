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

}  /* namespace rad */
