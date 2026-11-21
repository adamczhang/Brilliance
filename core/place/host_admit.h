/* host_admit.h -- Brilliance: the host-memory admission for the routed-expert file and RAM tiers.
 *
 * A plan that keeps routed experts off the cards (--weights-file-relayout, --expert-ram-mib) asks
 * the host for three things: the ranks' pinned pools, which Windows charges against a per-process
 * pin limit; ordinary RAM for the experts the RAM store holds; and the transients of loading and
 * relaying them. Bring-up adds them up before any of it is allocated and refuses the plan when the
 * pins would pass the limit, or when the peak would leave less than kHostRamFloor of physical
 * memory available (24 GiB stays free through launch, serving and unload).
 *
 * The check is arithmetic on figures the caller measures (os::mem_info, rad_dev_pin_limit,
 * rad_dev_alloc_totals), so a host test can hand it any machine and see the refusal at the floor
 * without allocating anything. There is no option to lower the floor.
 */
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>

namespace rad {

/* Physical memory that stays available beside the engine's peak. */
constexpr int64_t kHostRamFloor = (int64_t)24 << 30;

/* Loader transients and system slack counted beside the pools and the store: the relayout pool's
 * staged inputs and outputs (Engine::kRefillHost, 4 GiB), the sweep's ring (1 GiB), and what the
 * engine and the driver allocate unpooled. The design's planning figure (ds41-research §3). */
constexpr int64_t kHostMiscAllowance = (int64_t)8 << 30;

struct HostAdmitIn {
    int64_t avail = -1;          /* physical memory available now; -1 when unreadable */
    int64_t commit_avail = -1;   /* commit room now (RAM plus page file); -1 when unreadable */
    int64_t pin_cap = -1;        /* what this process may pin for its cards; -1 or 0: no limit */
    int64_t pinned_now = 0;      /* what it holds pinned already (mailboxes, kernel tables) */
    int64_t pins = 0;            /* the pinned pools still to allocate, every rank's */
    int64_t ram_store = 0;       /* ordinary RAM the RAM store will hold, every rank's */
    int64_t relay_scratch = 0;   /* the relayout threads' input buffers, every rank's */
    int64_t misc = kHostMiscAllowance;
    int64_t floor = kHostRamFloor;
};

struct HostAdmitOut {
    bool        ok = false;
    int64_t     peak = 0;        /* new physical footprint: pins, store, scratch and misc */
    int64_t     left = -1;       /* available less the peak; -1 when availability is unknown */
    std::string why;             /* the refusal, or "" */
};

inline std::string host_admit_gib(int64_t b) {
    char s[48];
    std::snprintf(s, sizeof s, "%.2f GiB", (double)b / (double)((int64_t)1 << 30));
    return s;
}

/* Admits only what it can show fits: unreadable availability is a refusal, not a pass. */
inline HostAdmitOut host_admit(const HostAdmitIn& in) {
    HostAdmitOut o;
    o.peak = in.pins + in.ram_store + in.relay_scratch + in.misc;
    if (in.pin_cap > 0 && in.pinned_now + in.pins > in.pin_cap) {
        o.why = "the pinned pools need " + host_admit_gib(in.pins) + " beside the " +
                host_admit_gib(in.pinned_now) + " this process holds pinned, past the " +
                host_admit_gib(in.pin_cap) + " it may pin for its cards; lower --host-pool-mib";
        return o;
    }
    if (in.avail < 0) {
        o.why = "the available physical memory cannot be read, so the " +
                host_admit_gib(in.floor) + " floor cannot be shown to hold";
        return o;
    }
    o.left = in.avail - o.peak;
    if (o.left < in.floor) {
        o.why = "the peak (" + host_admit_gib(in.pins) + " pinned pools, " +
                host_admit_gib(in.ram_store) + " expert RAM, " + host_admit_gib(in.relay_scratch) +
                " relayout inputs, " + host_admit_gib(in.misc) + " loader and system) would leave " +
                host_admit_gib(o.left) + " of the " + host_admit_gib(in.avail) +
                " available, under the " + host_admit_gib(in.floor) +
                " floor; lower --expert-ram-mib or --host-pool-mib";
        return o;
    }
    if (in.commit_avail >= 0 && o.peak > in.commit_avail) {
        o.why = "the peak of " + host_admit_gib(o.peak) + " is more than the " +
                host_admit_gib(in.commit_avail) + " the system can still commit";
        return o;
    }
    o.ok = true;
    return o;
}

}  /* namespace rad */
