/* The all-reduce shims, and the init/fini they need.
 *
 * init/fini exist in the ABI for this (spec §2.1): libr4d's all-reduces handshake through peer
 * signal buffers and a scratch region that must be allocated on every rank, published to every
 * other rank and made peer-accessible before the first launch. That allocates, synchronises and
 * needs every rank, so it cannot run on the step path.
 *
 * No IPC handles: radiance runs one process with a thread per rank sharing an address space
 * (spec §1). Once hipDeviceEnablePeerAccess has run, a device pointer from rank 0's card is valid
 * on rank 1's card (unified virtual addressing), so the exchange is a pointer written to a shared
 * array plus a barrier (`r4d_ar_ipc_enable_peer`). This avoids handle marshalling and
 * hipIpcOpenMemHandle, which maps a second VA range for the same pages and must be closed in order
 * at shutdown.
 *
 * The barrier assumes every rank runs the same declare: rad_arch_declare runs once per rank with
 * the same plugin and model, so the Nth all-reduce instance is the Nth on every rank and the
 * rendezvous is keyed on arrival order alone. A rank declaring a different graph fails at startup
 * with a 60-second timeout and a named error.
 */

#include "r4d_args.h"
#include "r4d.h"
#include "r4d_log.h"       /* Brilliance: W lines through the host's logger */
#include "r4d_ar_shards.h"
#include "r4d_ar_links.h"
#include "r4d_ar4_weights.h"  /* tp4-link-weight: the four-rank two-shot's shard weights */
#include "r4d_ar_seq.h"     /* the two-rank flags layout (R4D_AR2_*) and the fault record */
#include "r4d_ar_wide_x.h"  /* the wide host flags layout (R4D_ARW_*), W1 */
#include "r4d_ihc.h"         // ar_rmsnorm_ihc_junction's operands

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <exception>
#include <map>
#include <set>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#if defined(_WIN32)
#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <windows.h>
#include <psapi.h>   /* QueryWorkingSetEx (K32QueryWorkingSetEx in kernel32): the stall dump's page check */
#endif

static_assert(R4D_AR_TWOSHOT_WIDE_MAX_ELEMS == 268435456,
              "r4d_rows.cpp restates this bound; libr4d changed it");
static_assert(R4D_MAX_RANKS == 8, "the 8-rank entry point names exactly seven peers");

/* ------------------------------------------------------------------ throw conversion
 *
 * The all-reduce entry points throw; the ABI returns error codes, so every call is wrapped. The
 * shim pre-checks everything the entry points throw about, so an escaping throw is a library or
 * device fault: it becomes RAD_E_DEVICE and is printed once. */
static int r4d_caught(const char* who, const char* what) {
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "[r4d] %s threw: %s\n", who, what ? what : "(null)");
    }
    return RAD_E_DEVICE;
}

#define R4D_TRY(who, ...)                                                       \
    do {                                                                        \
        try { __VA_ARGS__; }                                                    \
        catch (const std::exception& e_) { return r4d_caught(who, e_.what()); } \
        catch (...) { return r4d_caught(who, "unknown exception"); }            \
    } while (0)

#if defined(_WIN32)
static bool mbgeo_on();   /* Brilliance: the stall diagnostics' switch (r4d_win_mailbox_dump) */
static bool cap_on();     /* Brilliance: the stall capture's switch (BRILLIANCE_STALL_CAPTURE) */
/* Brilliance: the push shadow (r4d_ar_hc_write_kernel), with the stall diagnostics only:
 * BRILLIANCE_PUSH_SHADOW=full (every pushed word, bit 0x100), =hash (a 32-bit hash per word, bit
 * 0x200), unset or anything else off. Off by default: the full shadow's stores appear to suppress
 * the hang (why-hwq2-1, why-diag-4, 2026-11-18). */
static int push_shadow_mode() {
    static const int m = [] {
        if (!mbgeo_on()) return 0;
        const char* v = std::getenv("BRILLIANCE_PUSH_SHADOW");
        if (!v) return 0;
        if (!std::strcmp(v, "full")) return 1;
        if (!std::strcmp(v, "hash")) return 2;
        return 0;
    }();
    return m;
}
#endif

/* Brilliance: the four-rank host mailboxes' loud stop (W1; r4d_ar_wide_host.hip, r4d_ar_wide_x.h). A
 * four-rank instance whose mailbox is in pinned host memory, on Windows, is built with the wide
 * control words and a VRAM mirror of its slots (`wide_x`), and the five collectives Flash-Next TP4
 * reaches on the host path launch these kernels instead of the defaults. `stage` is the mirror;
 * `inj` a fault injection request (ar_inj_take_w below), 0 for none. `rs` (W2/W3): the one-shot, the
 * fused writes and the all-gather run the exchange, the resend (ar_resend_on below); 0 runs W1's
 * kernels. From W4 the two-shot takes `rs` too, and its coded form (overlap 1) has a launcher of its
 * own, r4d_arw_twoshot_coded, used only with the resend on (r4d_ar_wide_ts.hip). */
extern "C" void r4d_arw_oneshot(const long long* ps, const long long* pf, long long my_scratch,
                                long long my_flags, long long seq_ctrs, long long slot_stride16,
                                long long inp, long long out, long long n_elem, long long dtype,
                                long long rank, long long stream, long long nblocks,
                                long long nthreads, long long drain, long long acq, long long pub,
                                long long stage, long long inj, long long rs);
extern "C" void r4d_arw_twoshot(const long long* ps, const long long* pf, long long my_scratch,
                                long long my_flags, long long seq_ctrs, long long slot_stride16,
                                long long inp, long long out, long long n_elem, long long dtype,
                                long long rank, long long stream, long long nblocks,
                                long long nthreads, long long drain, long long acq, long long pub,
                                long long stage, long long inj, long long rs, const int* weights);
extern "C" void r4d_arw_twoshot_coded(const long long* ps, const long long* pf, long long my_scratch,
                                      long long my_flags, long long seq_ctrs, long long slot_stride16,
                                      long long inp, long long out, long long n_elem, long long rank,
                                      long long stream, long long nblocks, long long nthreads,
                                      long long drain, long long acq, long long pub, long long stage,
                                      long long inj, const int* weights);
/* tp4-link-weight: `weights` (both two-shots) is the instance's table (r4d_ar4_weights.h) or null for
 * even shards, which keeps the launches as they were; ar4_weights_for below passes it only when the
 * weighted table fits the slot. */
extern "C" void r4d_arw_hc_write(const long long* ps, const long long* pf, long long my_scratch,
                                 long long my_flags, long long seq_ctrs, long long slot_stride16,
                                 long long y, long long inj, long long h, long long M, long long n,
                                 long long hc, long long irow, long long hrow, long long rank,
                                 long long stream, long long nblocks, long long nthreads,
                                 long long drain, long long acq, long long pub, long long stage,
                                 long long injreq, long long rs);
extern "C" void r4d_arw_gather_hc_write(
    const long long* ps, const long long* pf, long long my_scratch, long long my_flags,
    long long seq_ctrs, long long slot_stride16, long long y, long long inj, long long h, long long M,
    long long n, long long hc, long long irow, long long hrow, long long ye, long long ew,
    long long sorted, long long sh, long long sg, long long ye_ld, long long ew_ld, long long sh_ld,
    long long sg_ld, long long top_k, long long sig, long long rank, long long stream,
    long long nblocks, long long nthreads, long long drain, long long acq, long long pub,
    long long inv, long long stage, long long injreq, long long rs);
extern "C" void r4d_arw_gather(const long long* ps, const long long* pf, long long my_scratch,
                               long long my_flags, long long seq_ctrs, long long slot_stride16,
                               long long inp, long long out, long long n_elem, long long dtype,
                               long long rank, long long row_elems, long long stream,
                               long long nblocks, long long nthreads, long long drain,
                               long long acq, long long pub, long long dpitch, long long spitch,
                               long long stage, long long inj, long long rs);

/* ------------------------------------------------------------------ the rendezvous */

namespace {

struct Rendezvous {
    std::mutex              m;
    std::condition_variable cv;
    int  arrived = 0;
    int  epoch   = 0;
    long long scratch[R4D_MAX_RANKS] = { 0 };
    long long flags  [R4D_MAX_RANKS] = { 0 };
    long long seq    [R4D_MAX_RANKS] = { 0 };
    int  dev    [R4D_MAX_RANKS] = { 0 };
    /* Each rank's card's link, which the eight-rank two-shot's weights are derived from. */
    R4DArLink ar8_link[R4D_MAX_RANKS];
    /* tp4-link-weight: each four-rank host rank's link rate (GB/s, the start-time probe; 0 unknown)
     * and its card's PCI bus (-1 unknown), which the shard weights are derived from (ar4_settle). */
    double ar4_gbs[R4D_MAX_RANKS] = { 0 };
    int  ar4_bus[R4D_MAX_RANKS] = { 0 };
    int  failed = 0;
};

/* One rendezvous per tensor-parallel group. Several engines may run in one process
 * (docs/MULTIMODEL.md), each with its own ranks 0..world-1, so the ranks that meet are the ranks of
 * one group: the core names it in init's `tp_group` parameter (absent, the one group 0). Entries
 * live for the process; a group's id is reused only by a later engine with the same ranks. */
std::mutex                                       g_rv_mu;
std::map<long long, std::unique_ptr<Rendezvous>> g_rvs;

Rendezvous& rv_for(long long group) {
    std::lock_guard<std::mutex> lk(g_rv_mu);
    std::unique_ptr<Rendezvous>& r = g_rvs[group];
    if (!r) r.reset(new Rendezvous());
    return *r;
}

/* Sense-reversing barrier on a monotone epoch, so a rank that wakes and re-enters for the next
 * instance cannot be counted twice. The timeout is not recovery; it turns a startup deadlock into
 * a named error instead of a silent hang. */
int rv_barrier(Rendezvous& rv, int world) {
    std::unique_lock<std::mutex> lk(rv.m);
    const int e = rv.epoch;
    if (++rv.arrived == world) {
        rv.arrived = 0;
        rv.epoch++;
        rv.cv.notify_all();
        return RAD_OK;
    }
    const bool ok = rv.cv.wait_for(lk, std::chrono::seconds(60),
                                   [&] { return rv.epoch != e; });
    /* A rank that gave up takes its arrival back: a peer that failed before arriving would
     * otherwise leave it counted, and the group's next engine (its id is reused) would be released
     * one rank early. */
    if (!ok) --rv.arrived;
    return ok ? RAD_OK : RAD_E_STATE;
}

/* One resolved instance's view of the world: its own buffers and its peers', the latter in
 * ascending global rank with this rank removed (the order libr4d's wide entry points take). */
struct ArInst {
    int  rank = 0, world = 0;
    long long group = 0;      /* the tensor-parallel group it rendezvoused in (rv_for) */
    long long my_scratch = 0, my_flags = 0, my_seq = 0, locpack = 0;
    /* A host-mailbox instance's VRAM mirror of its slots (r4d_ar_seq.h): every two-rank one, and a
     * four-rank one on Windows (`wide_x`); 0 otherwise. */
    long long my_stage = 0;
    /* Brilliance, W1: a four-rank host instance on Windows with the wide control words (R4D_ARW_*,
     * r4d_ar_wide_x.h) and the mirror, whose five reached collectives run the kernels with the loud
     * stop (r4d_ar_wide_host.hip). */
    int  wide_x = 0;
    long long peer_scratch[R4D_MAX_PEERS] = { 0 };
    long long peer_flags  [R4D_MAX_PEERS] = { 0 };
    long long slot_bytes = 0;      /* per-slot capacity, 16-byte aligned                  */
    long long slot16     = 0;      /* the same in 16-byte words                           */
    long long max_numel  = 0;      /* the declared bound the sizing was computed from     */
    int  esize      = 2;
    int  dtype      = 0;      /* libr4d code: 0 bf16, 1 fp16, 2 fp32                 */
    int  max_blocks = 0;      /* what the flag array was sized for; a hard bound      */
    /* Block count a launch may use: a tuning, unlike the max_blocks capacity. Each block drives its
     * own flag handshake, so more blocks help only until the link saturates, then add round trips
     * (at a 2048-token chunk the capacity allows 936 blocks; 24 is used). */
    int  nb_cap     = 0;
    /* Exchange buffers: 0 in each rank's VRAM, written by peers over P2P; 1 in pinned host memory
     * each card reaches over its own link (see build). */
    int  host       = 0;
    /* The block capacity the flags were allocated for (r4d_ar_max_blocks() at two ranks), which the
     * kernels address them by; max_blocks may be lowered by R4D_AR_MAX_BLOCKS. */
    int  flag_nbw   = 0;
    long long gate_ctr   = 0;      /* the blocks that have seen their peer's bytes land (r4d_ar_seq.h) */
    /* The split junction's wave sums and two words a (launch parity, row) (K-13, r4d_ihc.h's
     * JuncSplit): rank-local and coarse, since only this rank's blocks touch it. */
    long long jparts     = 0;
    /* The eight-rank two-shot's per-rank shard weights (r4d_ar_shards.h), fixed at build so the
     * scratch was sized for them and a recorded launch keeps its table. */
    int  shard_weighted = 0;
    int  shard_w[R4D_MAX_RANKS] = { 0 };
    /* Its piece count, 0 for the size rule (r4d_ar8_auto_pieces); a launch's `pieces` parameter
     * overrides it. */
    int  pieces = 0;
    /* Messages under these sizes (elements) run even and in one piece, whatever is configured. */
    long long weights_min = R4D_AR8_MIN_ELEMS_DEFAULT;
    long long pieces_min  = R4D_AR8_MIN_ELEMS_DEFAULT;
};

/* A parameter that may be declared as a range: init gets one band's frozen geometry, and buffers
 * are sized for the band's upper bound. */
long long pgeti_hi(const RadParam* p, int n_p, const char* k, long long dflt) {
    if (!p || !k) return dflt;
    for (int i = 0; i < n_p; ++i) {
        if (std::strcmp(p[i].key, k) != 0) continue;
        if (p[i].kind == RAD_P_INT)   return p[i].ival;
        if (p[i].kind == RAD_P_RANGE) return p[i].ihi;
    }
    return dflt;
}

long long align16(long long v) { return (v + 15) & ~15LL; }

/* The eight-rank two-shot's settings for the next build: the engine's defaults unless
 * r4d_ar8_test_settings set others. */
std::mutex     g_ar8_m;
R4DAr8Settings g_ar8;

R4DAr8Settings ar8_settings() {
    std::lock_guard<std::mutex> lk(g_ar8_m);
    return g_ar8;
}

/* This rank's card's slowest PCIe hop, from sysfs by the runtime's bus id. */
bool ar8_own_link(R4DArLink* l) {
    int dev = 0;
    char bus[64] = { 0 };
    if (hipGetDevice(&dev) != hipSuccess ||
        hipDeviceGetPCIBusId(bus, (int)sizeof bus, dev) != hipSuccess)
        return false;
    unsigned dom = 0, b = 0, d = 0, f = 0;
    if (std::sscanf(bus, "%x:%x:%x.%x", &dom, &b, &d, &f) != 4) return false;
    char bdf[32];
    std::snprintf(bdf, sizeof bdf, "%04x:%02x:%02x.%x", dom, b, d, f);
    return r4d_ar_link_bottleneck("", bdf, l);
}

/* The eight-rank two-shot's choice for one message, a function of its size and the instance
 * alone: a recorded launch keeps the table and piece count it was recorded with, and replays of
 * the same size agree with it. */
const int* ar8_weights_for(const ArInst* in, long long numel) {
    return in->shard_weighted && numel >= in->weights_min ? in->shard_w : nullptr;
}
long long ar8_pieces_for(const ArInst* in, long long numel, long long asked) {
    if (numel < in->pieces_min) return 1;
    return asked > 0 ? asked : r4d_ar8_auto_pieces(numel);
}

/* The eight-rank two-shot's settings, fixed before anything is sized: each rank publishes its own
 * card's link, and after one barrier every rank derives the same weights from the same eight
 * entries. */
int ar8_settle(ArInst* in, int rank, int world) {
    Rendezvous& rv = rv_for(in->group);
    static std::once_flag said_links, said_unknown, said_set;
    const R4DAr8Settings set = ar8_settings();
    R4DArLink own;
    if (set.weights == R4D_AR8_WEIGHTS_AUTO && !ar8_own_link(&own)) own = R4DArLink();
    {
        std::lock_guard<std::mutex> lk(rv.m);
        rv.ar8_link[rank] = own;
    }
    const int rc = rv_barrier(rv, world);
    if (rc != RAD_OK) return rc;

    const R4DArLink* all = rv.ar8_link;
    in->pieces = set.pieces;
    in->weights_min = set.weights_min;
    in->pieces_min = set.pieces_min;

    int w[R4D_MAX_RANKS] = { 0 };
    if (set.weights == R4D_AR8_WEIGHTS_SET) {
        std::memcpy(w, set.w, sizeof(set.w));
        std::call_once(said_set, [&w] {
            std::fprintf(stderr, "[r4d] eight-rank two-shot shards weighted "
                         "%d,%d,%d,%d,%d,%d,%d,%d (r4d_ar8_test_settings)\n", w[0], w[1], w[2],
                         w[3], w[4], w[5], w[6], w[7]);
        });
    } else if (set.weights == R4D_AR8_WEIGHTS_AUTO) {
        double gbs[R4D_MAX_RANKS] = { 0 };
        for (int r = 0; r < world; ++r) gbs[r] = r4d_ar_link_gbs(all[r]);
        if (!r4d_ar_weights_from_rates(gbs, world, w)) {
            std::call_once(said_unknown, [all, world] {
                std::string who;
                for (int r = 0; r < world; ++r)
                    if (r4d_ar_link_gbs(all[r]) <= 0.0) who += " " + std::to_string(r);
                std::fprintf(stderr, "[r4d] eight-rank two-shot: no PCIe link speed in sysfs for "
                             "rank%s; shards stay even\n", who.c_str());
            });
            return RAD_OK;
        }
        std::call_once(said_links, [all, world, &w] {
            std::string l;
            for (int r = 0; r < world; ++r) {
                char one[96];
                std::snprintf(one, sizeof one, "%s r%d %.1f GT/s x%d %.1f GB/s w%d",
                              r ? "," : "", r, all[r].gts, all[r].width,
                              r4d_ar_link_gbs(all[r]), w[r]);
                l += one;
            }
            std::fprintf(stderr, "[r4d] eight-rank two-shot shard weights from the slowest PCIe "
                         "hop of each card:%s\n", l.c_str());
        });
        bool same = true;
        for (int r = 1; r < world; ++r) same = same && w[r] == w[0];
        if (same) return RAD_OK;   /* equal links: the even split, on the unweighted path */
    } else {
        return RAD_OK;
    }
    in->shard_weighted = 1;
    std::memcpy(in->shard_w, w, sizeof(in->shard_w));
    return RAD_OK;
}

/* Brilliance, experiment (stage M, 2026-11-18): BRILLIANCE_MAILBOX_MEM picks the hipHostMalloc
 * memory type of the TWO-RANK host mailboxes only (the slots and the flag words; the four- and
 * eight-rank ones keep r4d_ar_host_alloc). The TP2 hang loses or misdirects one 64-byte line of a
 * push on its way to this memory, so the type of the destination is the variable under test.
 *   unset or empty  noncoherent-slots at two and four ranks, the default since 2026-11-19 (below)
 *   default         r4d_ar_host_alloc (Coherent | Mapped | Portable), then the readback below
 *   coherent        the same flags asked for here (the default is already coherent)
 *   wc              the default plus WriteCombined
 *   noncoherent     NonCoherent | Mapped | Portable for the data slots; the flags stay coherent
 *                   (C1 stage 1: the engine's CPU reads the fault record in them after every step,
 *                   so they must not be non-coherent in any mode). Until 2026-11-18 PM the flags
 *                   were non-coherent too (the why-mem-* runs); now this equals noncoherent-slots
 *                   but for the flags' print
 *   uncached        the default plus Uncached (0x10000000)
 *   noncoherent-slots  the data slots NonCoherent | Mapped | Portable; the flags, check words and
 *                   diagnostic words keep the default allocation (r4d_ar_host_alloc). The one value
 *                   that also reaches the FOUR-rank host mailboxes: their data slots, with the slot
 *                   stride padded to 256 bytes (build); their flags and check words stay coherent
 * Anything else is reported and treated as unset. Set or by default, the first allocation of each kind (slots,
 * flags) prints the flags asked for, hipHostGetFlags' readback and a CPU read rate of the buffer
 * (cached memory reads at GB/s, write-combined or uncached memory far slower), so a flag the
 * runtime ignores shows. A refused allocation is reported and falls back to the default. Host code
 * only: no kernel changes.
 *
 * Brilliance (2026-11-19, the owner's rule; it replaces stage 7's rule of 2026-11-18, which chose
 * it only when experts spilled to host memory): UNSET, every two-rank and four-rank host mailbox
 * takes noncoherent-slots, for any model and in every process (the engine, r4d_selftest, the
 * tools): its data slots non-coherent, its flags coherent. A set BRILLIANCE_MAILBOX_MEM always
 * wins; at four ranks any value but noncoherent-slots keeps the coherent slots. Eight ranks keep
 * r4d_ar_host_alloc. The engine names the choice in one line after placement
 * (r4d_win_mailbox_default). */
static const char* const kMbMemName[] = { "default", "coherent", "wc", "noncoherent", "uncached",
                                          "noncoherent-slots" };
static const int kMbNcSlots = 5;   /* kMbMemName's noncoherent-slots */
/* Brilliance: read once a process (r4d_ar_env_read reads it at rad_plugin_open), its warning
 * through r4d_log; r4d_ar_env_unread makes a test read it again. */
static std::atomic<int> g_mb_mem{ -2 };   /* -2 not read yet, -1 unset, else kMbMemName's index */
static int mailbox_mem_mode() {
    int m = g_mb_mem.load(std::memory_order_relaxed);
    if (m != -2) return m;
    const char* v = std::getenv("BRILLIANCE_MAILBOX_MEM");
    bool bad = false;
    m = -1;
    if (v && *v) {
        bad = true;
        for (int i = 0; i < 6; ++i)
            if (!std::strcmp(v, kMbMemName[i])) { m = i; bad = false; break; }
    }
    int want = -2;
    if (g_mb_mem.compare_exchange_strong(want, m) && bad)
        r4d_log(1, "r4d: BRILLIANCE_MAILBOX_MEM=%s is not one of default, coherent, wc, "
                "noncoherent, uncached, noncoherent-slots; ignored, the host mailboxes take the "
                "default (noncoherent-slots)", v);
    return g_mb_mem.load(std::memory_order_relaxed);
}

static unsigned mailbox_mem_flags(int mode, bool slots) {
    const unsigned base = hipHostMallocMapped | hipHostMallocPortable;
    switch (mode) {
        case 2:  return base | hipHostMallocCoherent | hipHostMallocWriteCombined;
        case 3:  return slots ? base | hipHostMallocNonCoherent : base | hipHostMallocCoherent;
        case 4:  return base | hipHostMallocCoherent | hipHostMallocUncached;
        case 5:  return slots ? base | hipHostMallocNonCoherent : base | hipHostMallocCoherent;
        default: return base | hipHostMallocCoherent;
    }
}

/* The memory type of a two-rank host mailbox: BRILLIANCE_MAILBOX_MEM when set, else
 * noncoherent-slots, the default (`*automatic` set). The same for every group. */
static int mailbox2_mode(long long /*group*/, bool* automatic) {
    const int env = mailbox_mem_mode();
    *automatic = env < 0;
    return env >= 0 ? env : kMbNcSlots;
}

/* Four ranks take only noncoherent-slots, only when BRILLIANCE_MAILBOX_MEM says so (the automatic
 * choice from placement stays two-rank), and only for their data slots: the flags, with the check
 * words and the fault record (W1, R4D_ARW_*), stay r4d_ar_host_alloc. Plan 8 (2026-11-18) audited the gfx1201
 * code of every four-rank host-mailbox kernel (r4d_ar_wide.h's handshake): each flag store follows
 * a system-scope writeback and its store-count wait, and each read of a slot follows a
 * system-scope invalidate, on every path the tunables reach (acq 0 or 1).
 *
 * Since 2026-11-19 (the owner's rule) an UNSET BRILLIANCE_MAILBOX_MEM gives four ranks
 * noncoherent-slots too, the same default as two ranks (mailbox2_mode); set, only noncoherent-slots
 * does, and any other value keeps the coherent slots. */
static bool mailbox4_nc_slots() {
    const int env = mailbox_mem_mode();
    return env < 0 || env == kMbNcSlots;
}

/* `slots`: the data slots (sc); false for the flags, check words and diagnostic words (fl).
 * `world`: 2, or 4 for the four-rank data slots under mailbox4_nc_slots (noncoherent-slots, by
 * default or set). */
static long long mailbox2_alloc(long long size, const char* what, bool slots, long long group,
                                int world = 2) {
    bool automatic = false;
    const int mode = world == 2 ? mailbox2_mode(group, &automatic) : kMbNcSlots;
    if (world != 2) automatic = mailbox_mem_mode() < 0;
    if (mode < 0) return r4d_ar_host_alloc(size);
    const unsigned want = mailbox_mem_flags(mode, slots);
    void* p = nullptr;
    bool refused = false;
    hipError_t ae = hipSuccess;
    if (mode == 0 || (mode == 5 && !slots)) {
        p = (void*)r4d_ar_host_alloc(size);
    } else {
        ae = hipHostMalloc(&p, (size_t)size, want);
        if (ae != hipSuccess || !p) {
            (void)hipGetLastError();
            refused = true;
            p = (void*)r4d_ar_host_alloc(size);
        } else {
            std::memset(p, 0, (size_t)size);
            void* d = nullptr;
            const hipError_t e = hipHostGetDevicePointer(&d, p, 0);
            if (e != hipSuccess || d != p) {
                (void)hipHostFree(p);
                throw std::runtime_error(e != hipSuccess
                    ? std::string("host mailbox (BRILLIANCE_MAILBOX_MEM): no card address: ") + hipGetErrorString(e)
                    : std::string("host mailbox (BRILLIANCE_MAILBOX_MEM): the card address differs from the CPU address"));
            }
        }
    }
    unsigned got = 0;
    const hipError_t ge = hipHostGetFlags(&got, p);
    if (ge != hipSuccess) (void)hipGetLastError();

    /* Printed for the first allocation of each kind, and again for any later one whose readback
     * differs from that kind's first. */
    static std::mutex mu;
    static bool said[2] = { false, false };
    static unsigned said_got[2] = { 0, 0 };
    const int k = slots ? 0 : 1;
    std::lock_guard<std::mutex> lk(mu);
    if (said[k] && !refused && ge == hipSuccess && got == said_got[k]) return (long long)p;
    /* The CPU read rate, over at most 16 MiB of the zeroed buffer: once a kind. */
    double gbs = -1.0;
    if (!said[k]) {
        const size_t n = (size_t)(size < (16LL << 20) ? size : (16LL << 20)) / 8;
        const volatile unsigned long long* q = (const volatile unsigned long long*)p;
        unsigned long long acc = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < n; ++i) acc += q[i];
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (s > 0 && acc == 0) gbs = (double)(n * 8) / s / 1e9;
    }
    std::fprintf(stderr, "I r4d: %s%s%s: %s-rank host mailbox (%s, %lld bytes) asked 0x%08x%s, "
                 "hipHostGetFlags %s 0x%08x%s",
                 automatic ? "" : "BRILLIANCE_MAILBOX_MEM=", kMbMemName[mode],
                 automatic ? " (default)" : "", world == 4 ? "four" : "two",
                 what, size, want,
                 refused ? " REFUSED, fell back to the default 0x40000003" : "",
                 ge == hipSuccess ? "reads" : "FAILED,", got,
                 (ge == hipSuccess && !refused && got != want) ? " (DIFFERS from the flags asked)" : "");
    if (refused) std::fprintf(stderr, " [hipHostMalloc: %s]", hipGetErrorString(ae));
    if (gbs >= 0) std::fprintf(stderr, "; CPU read %.2f GB/s", gbs);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
    said[k] = true;
    said_got[k] = got;
    return (long long)p;
}

/* ---- tp4-link-weight: the four-rank two-shot's shard weights (r4d_ar4_weights.h) ----------------
 *
 * On Windows every four-rank collective crosses host memory, each card over its own link, and the
 * two-shot puts (1 + 2 x) N on a link each way for a rank reducing a share x of a message of N
 * bytes. With unequal links (the test machine: Gen5 x8, x8, Gen5 x4 and a Gen4 x4 riser) even shards make every
 * large reduction wait on the slowest. BRILLIANCE_AR4_SHARD_WEIGHTS (r4d_ar4_weights.h) sets the
 * table; unset or "auto", each rank times a copy over its own link at its first instance and the
 * table comes from the four rates (r4d_ar4_weights_from_rates); "off" or "even" keeps the even
 * split, the kernels and launches of the build before this setting. Messages under
 * BRILLIANCE_AR4_MIN_ELEMS elements (default R4D_AR8_MIN_ELEMS_DEFAULT: every decode and verify
 * message) stay even. The weights reach the wide host kernels only (wide_x: Windows), both
 * two-shot forms: W1's and W4's plain kernels, which took a table already, and the coded form
 * through r4d_ar_wide_tw.hip's kernel. The coded form's default kernel (BRILLIANCE_AR_RESEND=0) and
 * the one-shot row's two-shot band stay even. A table never grows a slot: one that does not fit runs
 * even (ar4_weights_for). The fold is the same under every table, so the output bits are too. */

/* This rank's card's PCI bus number, -1 when the runtime does not say. */
int ar4_own_bus() {
    int dev = 0;
    char bus[64] = { 0 };
    if (hipGetDevice(&dev) != hipSuccess ||
        hipDeviceGetPCIBusId(bus, (int)sizeof bus, dev) != hipSuccess) {
        (void)hipGetLastError();
        return -1;
    }
    unsigned dom = 0, b = 0, d = 0, f = 0;
    if (std::sscanf(bus, "%x:%x:%x.%x", &dom, &b, &d, &f) != 4) return -1;
    return (int)b;
}

/* The current card's link as it carries data (GB/s, the slower direction), 0 when it cannot be
 * measured: a 32 MiB copy each way between pinned host memory and VRAM, one warm-up and three timed,
 * the fastest of each kept. Once a process per card, one card at a time (the copies of four cards at
 * once would share the host's memory and understate the fast links); about 20 ms a card on Gen5 x8,
 * 40 on Gen4 x4. The buffers are freed before it returns. 0.2.x's start probe timed a 128 MiB copy
 * each way per card the same way: a lane width says what a link was trained to, not what it carries
 * (a riser falls back a generation under load and still reports its width). */
double ar4_probe_gbs() {
    static std::mutex mu;
    static std::map<int, double> cache;
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess) { (void)hipGetLastError(); return 0.0; }
    std::lock_guard<std::mutex> lk(mu);
    const auto it = cache.find(dev);
    if (it != cache.end()) return it->second;
    const size_t bytes = (size_t)32 << 20;
    void* h = nullptr;
    void* d = nullptr;
    double rate = 0.0;
    if (hipHostMalloc(&h, bytes, hipHostMallocDefault) == hipSuccess && h &&
        hipMalloc(&d, bytes) == hipSuccess && d) {
        std::memset(h, 0, bytes);
        double best_up = 0.0, best_down = 0.0;   /* seconds, the fastest timed copy */
        bool ok = hipDeviceSynchronize() == hipSuccess;
        for (int i = 0; ok && i < 4; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            ok = hipMemcpy(d, h, bytes, hipMemcpyHostToDevice) == hipSuccess;
            const auto t1 = std::chrono::steady_clock::now();
            ok = ok && hipMemcpy(h, d, bytes, hipMemcpyDeviceToHost) == hipSuccess;
            const auto t2 = std::chrono::steady_clock::now();
            if (!ok || i == 0) continue;
            const double up = std::chrono::duration<double>(t1 - t0).count();
            const double down = std::chrono::duration<double>(t2 - t1).count();
            if (best_up <= 0.0 || up < best_up) best_up = up;
            if (best_down <= 0.0 || down < best_down) best_down = down;
        }
        if (ok && best_up > 0.0 && best_down > 0.0) {
            const double slow = best_up > best_down ? best_up : best_down;
            rate = (double)bytes / slow / 1e9;
        }
    }
    (void)hipGetLastError();
    if (d) (void)hipFree(d);
    if (h) (void)hipHostFree(h);
    (void)hipGetLastError();
    cache[dev] = rate;
    return rate;
}

/* The startup line, printed when it differs from the last one (a process builds several instances,
 * r4d_selftest changes the setting between its cases). */
void ar4_say(const std::string& line) {
    static std::mutex mu;
    static std::string last;
    std::lock_guard<std::mutex> lk(mu);
    if (line == last) return;
    last = line;
    std::fprintf(stderr, "I r4d: %s\n", line.c_str());
    std::fflush(stderr);
}

/* Every rank of a four-rank host instance, before anything is sized: each publishes its card's bus
 * and (for "auto") its link rate, and after one barrier every rank derives the same table from the
 * same four entries. A setting that does not parse, or a by-bus table that misses a rank, stops the
 * build by name. */
int ar4_settle(ArInst* in, int rank, int world) {
#if !defined(_WIN32)
    (void)in; (void)rank; (void)world;
    return RAD_OK;   /* the weights reach the wide host kernels only, which are Windows' */
#else
    const char* env = std::getenv("BRILLIANCE_AR4_SHARD_WEIGHTS");
    const R4DAr4Spec spec = r4d_ar4_spec_parse(env);
    long long min_elems = R4D_AR8_MIN_ELEMS_DEFAULT;
    bool min_bad = false;
    const char* menv = std::getenv("BRILLIANCE_AR4_MIN_ELEMS");
    if (menv && *menv) {
        char* end = nullptr;
        const long long v = std::strtoll(menv, &end, 10);
        if (end == menv || *end || v < 0) min_bad = true;
        else min_elems = v;
    }
    Rendezvous& rv = rv_for(in->group);
    const int bus = ar4_own_bus();
    const double gbs = spec.kind == R4D_AR4_SPEC_AUTO ? ar4_probe_gbs() : 0.0;
    {
        std::lock_guard<std::mutex> lk(rv.m);
        rv.ar4_gbs[rank] = gbs;
        rv.ar4_bus[rank] = bus;
    }
    const int rc = rv_barrier(rv, world);
    if (rc != RAD_OK) return rc;
    double all_gbs[R4D_AR4_WS];
    int all_bus[R4D_AR4_WS];
    {
        std::lock_guard<std::mutex> lk(rv.m);
        for (int r = 0; r < R4D_AR4_WS; ++r) { all_gbs[r] = rv.ar4_gbs[r]; all_bus[r] = rv.ar4_bus[r]; }
    }
    if (spec.kind == R4D_AR4_SPEC_BAD) {
        std::fprintf(stderr, "E r4d: BRILLIANCE_AR4_SHARD_WEIGHTS=%s is not auto, off, even, four weights "
                     "of 0 to %d in rank order (\"1,1,1,0\") or bus:weight pairs (\"3:1,9:1,6:1,32:0\"), "
                     "with a sum above zero\n", env, (int)R4D_AR4_WEIGHT_MAX);
        return RAD_E_INVAL;
    }
    if (min_bad) {
        std::fprintf(stderr, "E r4d: BRILLIANCE_AR4_MIN_ELEMS=%s is not a count of elements\n", menv);
        return RAD_E_INVAL;
    }
    std::string links;
    for (int r = 0; r < R4D_AR4_WS; ++r) {
        char one[96];
        if (spec.kind == R4D_AR4_SPEC_AUTO)
            std::snprintf(one, sizeof one, "%srank %d bus %d %.1f GB/s", r ? ", " : "", r, all_bus[r], all_gbs[r]);
        else
            std::snprintf(one, sizeof one, "%srank %d bus %d", r ? ", " : "", r, all_bus[r]);
        links += one;
    }
    if (spec.kind == R4D_AR4_SPEC_EVEN) {
        ar4_say("four-rank two-shot: even shards (BRILLIANCE_AR4_SHARD_WEIGHTS=" + std::string(env) +
                "; " + links + ")");
        return RAD_OK;
    }
    int w[R4D_AR4_WS] = { 0, 0, 0, 0 };
    std::string from;
    if (spec.kind == R4D_AR4_SPEC_RANKS) {
        std::memcpy(w, spec.w, sizeof w);
        from = "BRILLIANCE_AR4_SHARD_WEIGHTS=" + std::string(env) + ", by rank";
    } else if (spec.kind == R4D_AR4_SPEC_BUSES) {
        if (!r4d_ar4_spec_by_bus(spec, all_bus, w)) {
            std::fprintf(stderr, "E r4d: BRILLIANCE_AR4_SHARD_WEIGHTS=%s does not name every rank's bus with "
                         "weights summing above zero (%s)\n", env, links.c_str());
            return RAD_E_INVAL;
        }
        from = "BRILLIANCE_AR4_SHARD_WEIGHTS=" + std::string(env) + ", by bus";
    } else {
        if (!r4d_ar4_weights_from_rates(all_gbs, w)) {
            bool known = true;
            for (int r = 0; r < R4D_AR4_WS; ++r) known = known && all_gbs[r] > 0.0;
            ar4_say(std::string("four-rank two-shot: even shards, ") +
                    (known ? "the links are close enough to equal" : "a link's rate is unknown") +
                    " (the link probe: " + links + "; BRILLIANCE_AR4_SHARD_WEIGHTS=off keeps them even)");
            return RAD_OK;
        }
        from = "the link probe";
    }
    if (r4d_ar4_weights_equal(w)) {
        ar4_say("four-rank two-shot: even shards (" + from + ": equal weights; " + links + ")");
        return RAD_OK;
    }
    in->shard_weighted = 1;
    std::memset(in->shard_w, 0, sizeof(in->shard_w));
    std::memcpy(in->shard_w, w, sizeof w);
    in->weights_min = min_elems;
    char head[320];
    std::snprintf(head, sizeof head, "four-rank two-shot shards weighted %d,%d,%d,%d by rank (%s) from "
                  "%lld elements up; ", w[0], w[1], w[2], w[3], from.c_str(), min_elems);
    std::string line = head + links;
    if (spec.kind == R4D_AR4_SPEC_AUTO) {
        int slow = 0;
        for (int r = 1; r < R4D_AR4_WS; ++r)
            if (all_gbs[r] < all_gbs[slow]) slow = r;
        char t[224];
        std::snprintf(t, sizeof t, "; the slowest link (rank %d) carries %.2f of a message each way a "
                      "reduction, %.2f even (BRILLIANCE_AR4_SHARD_WEIGHTS=off keeps them even)",
                      slow, r4d_ar4_link_load(w, slow), r4d_ar4_link_load(nullptr, slow));
        line += t;
    }
    ar4_say(line);
    return RAD_OK;
#endif
}

/* The weights for one four-rank two-shot launch, or null for even: a weighted instance, a message of
 * at least weights_min elements, and a table that fits the slot (`coded`: two coded regions of the
 * longest shard, in whole waves; else two plain shards). One that does not fit runs even, said once. */
const int* ar4_weights_for(const ArInst* in, long long numel, bool coded) {
    const int* w = ar8_weights_for(in, numel);
    if (!w || in->world != 4 || !in->wide_x) return nullptr;
    const long long n16 = numel * in->esize / 16;
    if (coded && n16 % (32 * R4D_AR4_WS)) return nullptr;
    R4DArwTab t;
    r4d_arw_tab(n16, w, coded ? 32 : 1, &t);
    if (r4d_arw_tab_fits(t, coded, in->slot16)) return w;
    static std::once_flag said;
    std::call_once(said, [&] {
        std::fprintf(stderr, "I r4d: four-rank two-shot: a %s message of %lld elements does not fit its slot "
                     "under the weighted table (longest shard %d words, slot %lld), so it runs even; the "
                     "bits are the same\n", coded ? "coded" : "plain", numel, t.sh_max, in->slot16);
        std::fflush(stderr);
    });
    return nullptr;
}

/* Sequence counter allocation in words: 2 * nbw (r4d_ar_seq.h), then r4d_ar_link_landed's block
 * counter, padded to a cache line. */
long long kSeqWords(long long nbw) { return 2LL * nbw + 32; }

/* Behind a flags allocation's flags and check words: a 2-rank instance's wake word
 * (r4d_ar_link_wake, r4d_ar_seq.h), on a line of its own. Brilliance: behind the push check's
 * diagnostic words too, so it is added to the P2P flags only; a host instance never stores it. */
constexpr long long kWakeBytes = 64;

/* Build the world. `max_numel` is the largest message any launch on this instance carries.
 *
 * One sizing rule for the whole family: a slot holds a full message. The one-shots need exactly
 * that; the two-shot's two regions (2 * message / world_size) and the tiered-int8 wire's two
 * images (2 * ceil(shard/32) * 38 bytes) are smaller. RadInitFn is not told which row resolved,
 * so sizing for the maximum avoids reconstructing which kernel this instance serves. The one
 * exception is a weighted eight-rank two-shot whose largest weight is over half the sum: its two
 * regions of the longest shard outgrow a message, and the slot grows to hold them.
 *
 * Cost: at a 4096-token chunk on a 5120-wide residual, 40 MiB a slot and 80 MiB for the pair. One
 * set serves every op with the same (rank, world, dtype, bound); see the cache below. */
int build(ArInst* in, long long group, int rank, int world, long long max_numel, int dtype_code,
          int host) {
    if (world < 2 || world > R4D_MAX_RANKS) return RAD_E_SHAPE;
    if (rank < 0 || rank >= world) return RAD_E_INVAL;
    if (max_numel <= 0) return RAD_E_INVAL;

    in->group = group;
    in->rank = rank;
    in->world = world;
    in->host = host;
    in->dtype = dtype_code;
    in->esize = r4d_ar_esize(dtype_code);
    in->max_numel = (long long)max_numel;
    in->slot_bytes = align16((long long)max_numel * in->esize);
    /* A two-rank bf16 slot over P2P also holds the coded form of its largest message (the fused
     * writes' coded push, r4d_ar_oneshot_2rank_exact.hip): 16.125 bytes a 16-byte word, 1/128
     * more than the message. */
    if (world == 2 && dtype_code == 0 && !host) {
        const long long coded = align16(r4d_ar_coded2_bytes(in->slot_bytes / 16));
        if (coded > in->slot_bytes) in->slot_bytes = coded;
    }
    in->slot16 = in->slot_bytes / 16;
    if (world == 8) {
        const int rc8 = ar8_settle(in, rank, world);
        if (rc8 != RAD_OK) return rc8;
    }
    /* tp4-link-weight: the four-rank weights (ar4_settle) never grow the slot: a weighted table
     * that does not fit runs even (ar4_weights_for), so their memory is the even build's. */
    if (world == 4 && host) {
        const int rc4 = ar4_settle(in, rank, world);
        if (rc4 != RAD_OK) return rc4;
    }
    if (in->shard_weighted && world == 8) {
        const long long need16 =
            2 * r4d_ar_shard_max_bound((long long)max_numel * in->esize / 16, 8, in->shard_w);
        /* Sized only if some message this instance carries is weighted. */
        if (max_numel >= in->weights_min && need16 > in->slot16) {
            in->slot16 = need16;
            in->slot_bytes = need16 * 16;
        }
    }
    /* noncoherent-slots at four ranks (mailbox4_nc_slots: the default, or set): each writer's
     * card caches its pushes to a peer's slot in its own L2, so no two writers may share a cache
     * line. Neighbouring slots belong to different source ranks, so the slot stride is padded to
     * whole 256-byte lines; every slot then starts on one. A capacity only grows, and the kernels
     * take the stride as an argument. With coherent slots (set), nothing changes. */
    if (host && world == 4 && mailbox4_nc_slots()) {
        in->slot_bytes = (in->slot_bytes + 255) & ~255LL;
        in->slot16 = in->slot_bytes / 16;
    }

    int maxb = 0, nb_design = 0, max_peers = 0;
    if (world == 2) {
        maxb = r4d_ar_max_blocks();
    } else {
        r4d_ar_wide_dims(&maxb, &nb_design, &max_peers);
        if (world - 1 > max_peers) return RAD_E_SHAPE;
    }
    if (maxb <= 0) return RAD_E_DEVICE;
    in->max_blocks = maxb;
    /* R4D_AR_MAX_BLOCKS: a ceiling on every collective's block count, for two engines sharing one
     * set of cards. Each block of a collective spins until its twin on the peer card has pushed, and
     * a prefill-sized ar_rmsnorm_quant_fp8 launches up to 256 blocks of 16 waves, twice what an
     * R9700 holds. One engine survives that (both cards dispatch blocks in the same order, so the
     * lowest unfinished block is resident on both); two engines do not: one engine's spinning blocks
     * fill a card while their twins wait behind the other engine's kernel on the peer card, which
     * waits for a slot on the full card. Not capped: the hyper-connection gathers of the MoE paths
     * (r4d_ar_gather_hc_write without its inverse scratch, r4d_ar_hc_write_wht6), whose LDS table
     * needs a minimum block count, which may exceed the ceiling (dense models do not use them).
     * Two TP2 engines on two R9700s (Qwen3.8-27B and Gemma 3 27B) hung within a minute of
     * concurrent prefill; MES then failed to evict the queues and the failed reset froze the
     * host. Capped in both engines so that their spinners together cannot
     * fill a card, the cycle cannot close. The flag and counter arrays keep their full size (maxb);
     * only the launch counts shrink, and each element's sum is the same. Unset, nothing changes. */
    if (const char* e = std::getenv("R4D_AR_MAX_BLOCKS")) {
        const int cap = std::atoi(e);
        if (cap > 0 && cap < in->max_blocks) in->max_blocks = cap;
    }
    /* Launch ceilings: 24 at two ranks, the library's design count at width. The compressed
     * 2-rank wire uses up to 48, set at its own call site: it moves three eighths of the bytes, so
     * it needs more blocks to saturate the link. */
    in->nb_cap = (world == 2) ? 24 : (nb_design > 0 ? nb_design : maxb);
    if (in->nb_cap > in->max_blocks) in->nb_cap = in->max_blocks;

    /* Slots: one per (parity, source rank) at width; two at world 2, where there is one remote
     * writer. Flags: one word per (block, source rank) at width, one per block at world 2.
     * Sequence counters: a handshake and a parity word per block of capacity (r4d_ar_seq.h),
     * device-resident so a captured graph replays. */
    const int  nslots     = (world == 2) ? 2 : 2 * world;
    const long long flag_words = (long long)maxb * ((world == 2) ? 1 : world);

    char handle[R4D_AR_HANDLE_BYTES];
    int  hlen = 0;
    long long sc = 0, fl = 0, sq = 0, lp = 0;
    int  dev = 0;
    /* W1: a four-rank host instance on Windows gets the loud stop (r4d_ar_wide_host.hip). The kernels
     * address the control words by R4D_AR_WIDE_MAX_BLOCKS, which is maxb here. */
#if defined(_WIN32)
    const bool wide_x = host && world == 4 && maxb == R4D_AR_WIDE_MAX_BLOCKS;
#else
    const bool wide_x = false;
#endif

    /* Fine-grained: the peer writes the scratch and this rank reads it bypassing L2, so a
     * coarse-grained allocation could leave the peer's bytes in a cache this rank never
     * invalidates, and the launches run without an acquire fence. r4d_ar_ipc_alloc throws when it
     * cannot allocate fine-grained memory; there is no coarse fallback.
     *
     * Without peer access the scratch and flags move to pinned host memory and nothing else
     * changes: kernels touch other ranks only through those two, so no card addresses another's
     * VRAM. A message then crosses two links instead of one; only rows requiring device.p2p == 0
     * take this path (r4d_rows.cpp). Sequence counters stay in VRAM, since only their own rank
     * reads them, and coarse-grained: only this rank's kernels touch them, with device- and
     * agent-scope atomics, which then complete in L2 rather than past it -- the first returning
     * atomic of every launch (r4d_ar_seq_take) is on the push's path. */
    R4D_TRY("ar_ipc_alloc",
        if (hipGetDevice(&dev) != hipSuccess) return RAD_E_DEVICE;
        if (host) {
            /* Two ranks: noncoherent-slots by default, or BRILLIANCE_MAILBOX_MEM's type
             * (mailbox2_alloc). Four ranks: noncoherent-slots by default or set, else coherent,
             * and only for the data slots (mailbox4_nc_slots). */
            sc = (world == 2) ? mailbox2_alloc((long long)nslots * in->slot_bytes, "slots", true, group)
               : (world == 4 && mailbox4_nc_slots())
                     ? mailbox2_alloc((long long)nslots * in->slot_bytes, "slots", true, group, 4)
                     : r4d_ar_host_alloc((long long)nslots * in->slot_bytes);
            /* + 2 maxb: the push check's diagnostic words (r4d_archk_wait, r4d_ar_seq.h).
             * At two ranks, the control words after them (R4D_AR2_FLAG_WORDS, r4d_ar_seq.h): the
             * stage 2 words, the fault record, the park, stages 3's and 4's counters and the rotated
             * wire's (tp-wire wht6), about 72 KiB of pinned memory. At four ranks on Windows (W1), the
             * wide control words (R4D_ARW_FLAG_WORDS, r4d_ar_wide_x.h): the check words, the W2 words
             * reserved, the counters per source and per collective, the fault record, about 31 KiB.
             * The flags stay coherent in every BRILLIANCE_MAILBOX_MEM mode (mailbox_mem_flags): the
             * engine's CPU reads the fault record from them after every step. */
            const long long fl_bytes = world == 2
                ? (long long)R4D_AR2_FLAG_WORDS(maxb) * 4
                : wide_x ? (long long)R4D_ARW_FLAG_WORDS(world, (unsigned)maxb) * 4
                         : (flag_words + 4LL * maxb) * 4;
            fl = (world == 2) ? mailbox2_alloc(fl_bytes, "flags", false, group) : r4d_ar_host_alloc(fl_bytes);
        } else {
            sc = r4d_ar_ipc_alloc((long long)nslots * in->slot_bytes, 1, handle, &hlen);
            fl = r4d_ar_ipc_alloc((flag_words + 4LL * maxb) * 4 + kWakeBytes, 1, handle, &hlen);
        }
        sq = r4d_ar_ipc_alloc(kSeqWords(maxb) * 4, 0, handle, &hlen));

    /* A host mailbox is read over the link past every cache, so the 2-rank kernels' push check
     * copies what it reads into this VRAM mirror for the reduce, crossing the link once
     * (r4d_ar_seq.h). Coarse-grained: only this rank's kernels touch it, each block its own words.
     * Host-memory slots free the same VRAM over P2P, so the mirror keeps the footprint where the P2P
     * instance has it. Four ranks on Windows (W1): the same mirror, 2 x 4 slots, which the wide host
     * kernels check into before they reduce (r4d_ar_wide_host.hip); elsewhere the wide family has no
     * push check and reads its slots once. */
    long long stg = 0;
    if (wide_x) {
        R4D_TRY("ar_ipc_alloc(stage, four-rank)",
            stg = r4d_ar_ipc_alloc((long long)nslots * in->slot_bytes, 0, handle, &hlen));
        static std::once_flag said;
        std::call_once(said, [&] {
            std::fprintf(stderr, "I r4d: four-rank host mailbox: the loud stop is armed (W1: bounded "
                         "waits, a check of every push before the reduce, a fault record the engine "
                         "stops on); each instance keeps a VRAM mirror of its slots, %.2f MiB for "
                         "this one\n", (double)nslots * (double)in->slot_bytes / 1048576.0);
            std::fflush(stderr);
        });
    }
    if (host && world == 2) {
        /* + 4 slots with BRILLIANCE_SLOG_LAUNCHES: the fused hc write's diagnostic shadow of its
         * pushes, two generations a slot (r4d_ar_hc_write_kernel, mb_bad_async below). */
#if defined(_WIN32)
        const long long extra = push_shadow_mode() ? 4 : 0;
#else
        const long long extra = 0;
#endif
        R4D_TRY("ar_ipc_alloc(stage)",
            stg = r4d_ar_ipc_alloc((long long)(nslots + extra) * in->slot_bytes, 0, handle, &hlen));
    }

    /* K-13: the split junction's scratch, one JuncSplit a rank, zeroed by the allocation: its words
     * must read zero at the first launch. Coarse for the same reason as the counters above. */
    long long jsp = 0;
    if (world == 2) {
        R4D_TRY("ar_ipc_alloc(jparts)",
            jsp = r4d_ar_ipc_alloc((long long)sizeof(r4d_ihc::JuncSplit), 0, handle, &hlen));
    }

    /* This rank's own packed copy, for the rotated 6-bit wire only: the reduce must fold exactly
     * the bytes it sent. The 6-bit payload is 3/8 of the bf16 message and the f32 scales 1/32 of
     * it, laid out from half the slot; a row padded to whole 512-element chunks
     * (ar_rmsnorm_quant_fp8) adds up to 7 groups' scales a row, so the scales get a sixteenth
     * (enough for any row of 512 or more) plus a page. ar_rmsnorm_quant checks the fit. */
    if (in->dtype != 2) {
        R4D_TRY("ar_ipc_alloc(locpack)",
            lp = r4d_ar_ipc_alloc(in->slot_bytes / 2 + in->slot_bytes / 16 + 4096, 0,
                                  handle, &hlen));
    }

    in->my_scratch = sc;
    in->my_flags   = fl;
    in->my_seq     = sq;
    in->flag_nbw   = maxb;
#if defined(_WIN32)
    /* The park (r4d_ar_fault_park, r4d_ar_seq.h): with the stall capture on, a block that gives up
     * waits for the capture before its reduce. The words are zero from the allocation. */
    if (host && world == 2 && cap_on())
        ((volatile unsigned int*)fl)[R4D_AR2_CTL_OFF(maxb) + R4D_AR2_CTL_PARK] = 1u;
#endif
    /* The exchange's request limit (fix C1 stage 2, r4d_archk_exchange): BRILLIANCE_AR_RESEND_MAX
     * requests for one block of one launch before it gives up (1-64; the kernel's default, 8, when
     * unset). Read by the device only once a block has asked. */
    if (host && world == 2) {
        if (const char* e = std::getenv("BRILLIANCE_AR_RESEND_MAX")) {
            const int m = std::atoi(e);
            if (m >= 1 && m <= 64) ((volatile unsigned int*)fl)[R4D_AR2_CTL_OFF(maxb) + R4D_AR2_CTL_RQMAX] = (unsigned)m;
        }
    }
    /* The same limit for the four-rank exchange (W2/W3, r4d_arw_exchange). */
    if (wide_x) {
        if (const char* e = std::getenv("BRILLIANCE_AR_RESEND_MAX")) {
            const int m = std::atoi(e);
            if (m >= 1 && m <= 64)
                ((volatile unsigned int*)fl)[R4D_ARW_CTL_OFF(world, (unsigned)maxb) + R4D_ARW_CTL_RQMAX] = (unsigned)m;
        }
    }
    in->wide_x     = wide_x ? 1 : 0;
    in->locpack    = lp;
    in->my_stage   = stg;
    in->jparts     = jsp;
    in->gate_ctr   = sq + 2LL * maxb * 4;   /* the word after the sequence counters (r4d_ar_seq.h) */

    Rendezvous& rv = rv_for(group);
    {
        std::lock_guard<std::mutex> lk(rv.m);
        rv.scratch[rank] = sc;
        rv.flags[rank]   = fl;
        rv.seq[rank]     = sq;
        rv.dev[rank]     = dev;
    }
    int rc = rv_barrier(rv, world);
    if (rc != RAD_OK) return rc;

    /* Peer access is per context, so each rank enables its own, after the publish barrier where
     * peer device ordinals are known. A host can expose more HIP devices than ranks (e.g. an
     * integrated GPU), so the ordinal comes from the peer rather than the rank. A peer on this
     * rank's own card needs none (the runtime refuses to enable it): r4d_selftest runs eight ranks
     * on one card that way, each on its own stream. */
    int j = 0;
    for (int r = 0; r < world; ++r) {
        if (r == rank) continue;
        if (!rv.scratch[r] || !rv.flags[r]) return RAD_E_STATE;
        in->peer_scratch[j] = rv.scratch[r];
        in->peer_flags[j]   = rv.flags[r];
        ++j;
        if (!host && rv.dev[r] != dev)
            R4D_TRY("ar_ipc_enable_peer", r4d_ar_ipc_enable_peer((long long)rv.dev[r]));
    }

    /* Second barrier: no rank may launch until every rank has enabled peer access, or the first
     * message would be pushed into a mapping the receiver has not opened. */
    return rv_barrier(rv, world);
}

/* ------------------------------------------------------------------ one world, shared
 *
 * One instance per (rank, world, dtype, message bound), not per op. The core calls init per
 * resolved instance, and a 64-layer tensor-parallel model declares 128 identical all-reduces.
 * Scratch is sized by the largest message (at a 2048-token chunk on a 5120-wide residual, ~53 MiB
 * per instance), so per-op instances would cost 6.7 GiB a card. It comes from hipMalloc, outside
 * the VRAM budget, so over-allocation shows up as the KV cache or a prefill chunk failing to fit.
 *
 * Sharing is safe. Slots are double-buffered by the parity of the launch on this instance, and a
 * rank's all-reduces run on one stream in declared order, so all-reduce k+2 (the next user of k's
 * slot) cannot launch until k+1 completed, which needed the peer's k+1 flag, set only after the
 * peer's k kernel finished reading that slot. This needs the launch's parity, not a block's: ops
 * sharing an instance use different block counts (r4d_ar_seq.h).
 *
 * The rendezvous stays symmetric because every rank declares the same graph in the same order:
 * the Nth all-reduce misses the cache on every rank or on none. */
struct CacheKey {
    long long group;          /* the tensor-parallel group: another engine's ranks never share */
    int       rank, world, dtype;
    long long numel;
    int       host;
    bool operator==(const CacheKey& o) const {
        return group == o.group && rank == o.rank && world == o.world && dtype == o.dtype &&
               numel == o.numel && host == o.host;
    }
};

struct CacheSlot {
    CacheKey key;
    ArInst*  in   = nullptr;
    int      refs = 0;
};

std::mutex             g_cache_mu;
std::vector<CacheSlot> g_cache;

int acquire(const CacheKey& k, ArInst** out) {
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        for (CacheSlot& s : g_cache)
            if (s.key == k && s.in) { ++s.refs; *out = s.in; return RAD_OK; }
    }
    /* Built outside the lock: build() rendezvouses with the other ranks, and holding the mutex
     * across that barrier would deadlock with the peer's lookup. Keys include the rank and each
     * rank has one declare thread, so no two builders race for one key. */
    ArInst* in = new (std::nothrow) ArInst();
    if (!in) return RAD_E_NOMEM;
    int rc = build(in, k.group, k.rank, k.world, k.numel, k.dtype, k.host);
    if (rc != RAD_OK) { delete in; return rc; }
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        g_cache.push_back(CacheSlot{ k, in, 1 });
    }
    *out = in;
    return RAD_OK;
}

void release(ArInst* in) {
    if (!in) return;
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        for (size_t i = 0; i < g_cache.size(); ++i) {
            if (g_cache[i].in != in) continue;
            if (--g_cache[i].refs > 0) return;
            g_cache.erase(g_cache.begin() + (long long)i);
            break;
        }
    }
    /* Free only this rank's own allocations; peer pointers belong to the peer. */
    if (in->host) {
        r4d_ar_host_free(in->my_scratch);
        r4d_ar_host_free(in->my_flags);
    } else {
        r4d_ar_ipc_free(in->my_scratch);
        r4d_ar_ipc_free(in->my_flags);
    }
    r4d_ar_ipc_free(in->my_seq);
    r4d_ar_ipc_free(in->locpack);
    r4d_ar_ipc_free(in->my_stage);
    r4d_ar_ipc_free(in->jparts);
    delete in;
}

}  /* namespace */

/* ------------------------------------------------------------------ init / fini */

static int ar_init(const RadParam* p, int n_p, int rank, int world, int host, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;

    /* world_size is both an ABI argument and a declared parameter. The parameter is what the
     * constraint matched; the argument is what the engine runs. A mismatch means the graph and the
     * process describe different deployments, so refuse here instead of at the first handshake. */
    const long long ws_param = pgeti_hi(p, n_p, "world_size", world);
    if (ws_param != world) return RAD_E_SHAPE;

    const int dt = r4d_ar_dtype_code(r4d_pgets(p, n_p, "dtype"));
    if (dt < 0) return RAD_E_DTYPE;

    const long long numel = pgeti_hi(p, n_p, "numel", 0);
    if (numel <= 0) return RAD_E_INVAL;

    ArInst* in = nullptr;
    const long long group = pgeti_hi(p, n_p, "tp_group", 0);
    int rc = acquire(CacheKey{ group, rank, world, dt, numel, host }, &in);
    if (rc != RAD_OK) return rc;
    *out = in;
    return RAD_OK;
}

#if defined(_WIN32)
/* Brilliance: what every two-rank host-mailbox instance's flags say, read by the CPU while the
 * cards are stuck (the core's stall watch calls this by name, win/device/coll_trace.cpp). The flags
 * are pinned host memory, so they can be read at any time; the sequence counters are VRAM and are
 * not. An instance's flags are written by its PEER: word b is the handshake value the peer's block
 * b last published, so one rank's line shows how far the other rank's blocks got. The check words
 * (two parity slots) are the peer's push sums, salted with the same handshake value. Writes up to
 * `cap` bytes into `buf`, NUL-terminated; returns the instances described. */
/* Brilliance: THE HOST-SIDE PUSH CHECK, for a stall (BRILLIANCE_SLOG_LAUNCHES=1; nothing is
 * recorded otherwise). The two-rank fused writes' launchers note each host instance's recent
 * launch geometries (16-byte words, blocks), and the dump re-computes, from the CPU, the push
 * check the reader re-tries (r4d_archk_wait, r4d_ar_seq.h) over each slot's bytes as they sit in
 * pinned host memory, against the check word the peer published. A block whose bytes hash to the
 * check word holds a push that landed whole, so a reader still spinning on it reads something
 * else; a block that matches no recent handshake value holds a push that is not whole. Read-only:
 * nothing on the cards changes. */
/* A geometry, and the message buffers (`y`) the fused hc write was launched with at it, up to four
 * distinct, most recent first (0 for the gather, whose message is assembled in the kernel). With
 * the stall capture on, also the hc write's other operands at its last launch at that geometry:
 * the write gains `inj` and the wide stream `h` (addresses, row pitches in elements), the row
 * width `n` and the stream count `hc` (cap_run below reconstructs the push from them). */
struct MbGeo {
    long long w16; int nb; long long ys[4]; int nys;
    long long inj = 0, h = 0, irow = 0, hrow = 0;
    int n = 0, hc = 0;
};
static std::mutex g_mbgeo_mu;
static std::map<const void*, std::vector<MbGeo>> g_mbgeo;   /* most recent first, distinct, up to 6 */

/* BRILLIANCE_SLOG_LAUNCHES=1: the stall diagnostics of 2026-11-18 (the push check, the bad block's
 * comparisons, the drain print). */
static bool slog_on() {
    static const bool on = [] {
        const char* v = std::getenv("BRILLIANCE_SLOG_LAUNCHES");
        return v && *v && v[0] != '0';
    }();
    return on;
}

/* BRILLIANCE_STALL_CAPTURE=1: the stall capture (cap_run below). Off by default. On, the fused
 * writes' launchers note their geometry and operands on the host as with BRILLIANCE_SLOG_LAUNCHES
 * (a lock and a few stores per launch, on the issuing thread), and each card gets the diagnostic
 * stream and staging buffer at its first noted launch. Nothing is added to any kernel or to any
 * card's queue until the stall watch has seen the stall. */
static bool cap_on() {
    static const bool on = [] {
        const char* v = std::getenv("BRILLIANCE_STALL_CAPTURE");
        /* The backend's rule (env_on, win/device/hipnative.cpp), so both halves agree. */
        return v && *v && !(v[0] == '0' || _stricmp(v, "off") == 0 || _stricmp(v, "false") == 0 ||
                            _stricmp(v, "no") == 0);
    }();
    return on;
}

static bool mbgeo_on() {
    static const bool on = slog_on() || cap_on();
    return on;
}

/* Brilliance, diagnostic: per card, a stream and a pinned buffer for the stall dump's reads of card
 * memory, made at the first noted launch on that card. They cannot be made at the stall:
 * hipStreamCreateWithFlags blocks inside the runtime while the cards are stuck (why-diag-2,
 * 2026-11-18). `stuck` marks a card whose read did not complete; no further read is tried there. */
struct MbDev { bool made = false; bool stuck = false; hipStream_t st = nullptr; void* pin = nullptr; size_t cap = 0; };
static MbDev g_mbdev[16];
static std::map<const void*, int> g_mbinst_dev;   /* instance -> its card (under g_mbgeo_mu) */

static void mbdev_note(const ArInst* in) {
    if (g_mbinst_dev.count(in)) return;
    int d = -1;
    if (hipGetDevice(&d) != hipSuccess || d < 0 || d >= 16) { (void)hipGetLastError(); return; }
    g_mbinst_dev[in] = d;
    MbDev& m = g_mbdev[d];
    if (m.made) return;
    m.made = true;
    const size_t cap = 16u << 20;
    if (hipStreamCreateWithFlags(&m.st, hipStreamNonBlocking) != hipSuccess ||
        hipHostMalloc(&m.pin, cap, hipHostMallocDefault) != hipSuccess) {
        (void)hipGetLastError();
        m.st = nullptr;
        m.pin = nullptr;
        return;
    }
    m.cap = cap;
    if (cap_on()) {
        std::fprintf(stderr, "I r4d: stall capture armed (BRILLIANCE_STALL_CAPTURE) on card %d: its diagnostic "
                     "stream and a %zu MiB staging buffer are made; nothing more runs until a stall\n", d, cap >> 20);
        std::fflush(stderr);
    }
}

/* Brilliance, diagnostic: which drain a fused write launches with, once per (kernel, instance,
 * value), so a RADIANCE_TUNE pin can be seen to take on the host twins. */
static void mbcfg_say(const char* kernel, const ArInst* in, int drain) {
    if (!in || !slog_on()) return;
    static std::mutex mu;
    static std::set<std::string> said;
    char k[160];
    std::snprintf(k, sizeof k, "%s %p %d", kernel, (const void*)in, drain);
    std::lock_guard<std::mutex> lk(mu);
    if (!said.insert(k).second) return;
    std::fprintf(stderr, "I r4d: %s%s on rank %d's instance %p (numel %lld) launches with drain=%d\n",
                 kernel, in->host ? "_host" : "", in->rank, (const void*)in, in->max_numel, drain);
    std::fflush(stderr);
}

static void mbgeo_note(const ArInst* in, long long w16, int nb, long long y,
                       const MbGeo* hcargs = nullptr) {
    if (!in || !in->host || in->world != 2 || !mbgeo_on()) return;
    std::lock_guard<std::mutex> lk(g_mbgeo_mu);
    std::vector<MbGeo>& v = g_mbgeo[in];
    MbGeo g{ w16, nb, { 0, 0, 0, 0 }, 0 };
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i].w16 == w16 && v[i].nb == nb) {
            g = v[i];
            if (i) { v.erase(v.begin() + (std::ptrdiff_t)i); v.insert(v.begin(), g); }
            break;
        }
    if (v.empty() || v[0].w16 != w16 || v[0].nb != nb) {
        v.insert(v.begin(), g);
        if (v.size() > 6) v.pop_back();
    }
    if (y && v[0].ys[0] != y) {
        MbGeo& h = v[0];
        int j = 0;
        while (j < h.nys && h.ys[j] != y) ++j;
        if (j == h.nys && h.nys < 4) ++h.nys;
        if (j >= 4) j = 3;
        for (; j > 0; --j) h.ys[j] = h.ys[j - 1];
        h.ys[0] = y;
    }
    if (hcargs) {
        v[0].inj = hcargs->inj; v[0].h = hcargs->h;
        v[0].irow = hcargs->irow; v[0].hrow = hcargs->hrow;
        v[0].n = hcargs->n; v[0].hc = hcargs->hc;
    }
    mbdev_note(in);
}

/* Brilliance, diagnostic: `bytes` of card memory at `src` into `dst`, through a stream of its own,
 * for the stall dump. The card is stuck in a kernel on another queue, so this may never finish:
 * it gives up after `wait_ms` and leaks its stream and buffer rather than block the dump. */
static bool mb_vram_read(int dev, long long src, void* dst, size_t bytes, int wait_ms, std::string* why) {
    if (dev < 0 || dev >= 16 || !g_mbdev[dev].st || !g_mbdev[dev].pin) { *why = "no diagnostic stream on that card"; return false; }
    /* One read at a time: the card's staging buffer is shared by every reader thread. */
    static std::mutex read_mu;
    std::lock_guard<std::mutex> lk(read_mu);
    MbDev& m = g_mbdev[dev];
    if (m.stuck) { *why = "an earlier read on that card did not complete"; return false; }
    if (bytes > m.cap) { *why = "larger than the staging buffer"; return false; }
    if (hipSetDevice(dev) != hipSuccess) { (void)hipGetLastError(); *why = "set device"; return false; }
    bool ok = false;
    if (hipMemcpyAsync(m.pin, (const void*)src, bytes, hipMemcpyDeviceToHost, m.st) == hipSuccess) {
        for (int t = 0; t < wait_ms / 10 && !ok; ++t) {
            const hipError_t q = hipStreamQuery(m.st);
            if (q == hipSuccess) ok = true;
            else if (q != hipErrorNotReady) break;
            else std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (ok) std::memcpy(dst, m.pin, bytes);
        else { *why = "the copy did not complete in time"; m.stuck = true; }
    } else {
        *why = "could not start the copy";
    }
    (void)hipGetLastError();
    return ok;
}

static float mb_bf(unsigned short v) {
    const unsigned int u = (unsigned int)v << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/* Brilliance, diagnostic: what a block that matches no check word holds, word by word. `in` is the
 * lagging rank's instance (its mailbox holds the bad block, pushed by the peer); `peer` the other
 * rank's. Compared, per 16-byte word of the block, with
 *   - the peer's own message: its message buffer `y` holds the reduced sum (the peer finished this
 *     collective) and this rank's `y` still holds this rank's own message for the block (its reduce
 *     never ran), so the peer's message is their difference, to bf16 rounding of the sum. A good
 *     block, where both buffers hold the sum, checks that the two buffers are this launch's;
 *   - this rank's VRAM mirror of the slot (what the card itself last read);
 *   - the same range of the other slot, and of the peer's mailbox (a misdirected or foreign write).
 * Prints the wrong words' runs: count, lengths, and the alignment of their host addresses. */
/* What the comparison needs, copied at the stall so it runs on a thread of its own: reads of card
 * memory may never complete while the cards are stuck, and must not hold up the dump. */
static unsigned int mb_mix(unsigned int v, unsigned int e);   /* below: r4d_archk_mix on the host */

/* A range of host memory the stall capture searches, and what it is. */
struct CapRange { long long base = 0, bytes = 0; std::string label; };

struct MbJob {
    MbGeo g{}, gl{}, gp{};
    int slot = 0, blk = 0, dev_l = -1, dev_p = -1, rank = 0;
    long long start = 0, end = 0, stage = 0, slot16 = 0, peer_stage = 0;
    unsigned int flag = 0, want = 0;
    const unsigned char* Haddr = nullptr;
    std::vector<unsigned char> h, o, p;
    /* The stall capture only: the instance, and every two-rank host mailbox's slots and flags
     * (taken under g_cache_mu by mb_bad's caller). */
    const void* inst = nullptr;
    std::vector<CapRange> boxes;
};

/* The push shadow (r4d_ar_hc_write_kernel, bit 0x100 of drain): the writer's own copy of what it
 * pushed, in its VRAM beside its mirror, two generations a slot by its handshake value. The stuck
 * push is the one the good blocks matched as "prev" (the writer has published one launch more since,
 * its gather), so its handshake value is flag - 1; flag itself is tried too. The generation whose
 * bytes hash to the check word is the push; the other generation is the push before it into the same
 * slot, the bytes a lost write would have left. */
static std::string mb_shadow_cmp(const MbJob& j) {
    const int mode = push_shadow_mode();
    if (!mode) return " [shadow: off (BRILLIANCE_PUSH_SHADOW unset)]";
    if (!j.peer_stage || j.dev_p < 0) return " [shadow: none (no peer mirror or card)]";
    const long long n = j.end - j.start;
    /* full: the words themselves; hash: one 32-bit r4d_archk_u4 per word, at 4 bytes a word. */
    const size_t bytes = mode == 1 ? (size_t)n * 16 : (size_t)n * 4;
    std::vector<unsigned char> r(bytes), old(bytes);
    std::string why, out;
    char b[400];
    auto at = [&](int gen) {
        const long long region = j.peer_stage + (long long)(2 + 2 * j.slot + gen) * j.slot16 * 16;
        return region + j.start * (mode == 1 ? 16 : 4);
    };
    auto word_hash = [&](const unsigned char* w16, long long i) {
        const unsigned int* w = (const unsigned int*)w16;
        const unsigned int e = 4u * (unsigned int)(j.start + i);
        return mb_mix(w[0], e) + mb_mix(w[1], e + 1u) + mb_mix(w[2], e + 2u) + mb_mix(w[3], e + 3u);
    };
    /* Per word, the hash of what the writer pushed (or held before) and of what the host holds. */
    auto hashes = [&](const std::vector<unsigned char>& v, std::vector<unsigned int>& hv) {
        hv.resize((size_t)n);
        for (long long i = 0; i < n; ++i)
            hv[(size_t)i] = mode == 1 ? word_hash(&v[(size_t)i * 16], i) : ((const unsigned int*)v.data())[i];
    };
    std::vector<unsigned int> ph, oh, hh((size_t)n);
    for (long long i = 0; i < n; ++i) hh[(size_t)i] = word_hash(&j.h[(size_t)i * 16], i);
    int gen = -1;
    unsigned int s_used = 0;
    for (unsigned int s : { j.flag - 1u, j.flag }) {
        const int g = (int)((s >> 1) & 1u);
        if (!mb_vram_read(j.dev_p, at(g), r.data(), bytes, 3000, &why)) return " [shadow: " + why + "]";
        hashes(r, ph);
        unsigned int t = 0u;
        for (unsigned int x : ph) t += x;
        if (t + s * 0x6C8E9CF5u == j.want) { gen = g; s_used = s; break; }
    }
    if (gen < 0) {
        std::snprintf(b, sizeof b, " [shadow (%s): neither generation sums to the check word %08x (flag %u): "
                      "the check word itself does not belong to what the writer pushed]",
                      mode == 1 ? "full" : "hash", j.want, j.flag);
        return b;
    }
    const bool have_old = mb_vram_read(j.dev_p, at(gen ^ 1), old.data(), bytes, 3000, &why);
    if (have_old) hashes(old, oh);
    long long wrong = 0, eq_old = 0, zero = 0, part = 0;
    std::vector<long long> rs, rn;
    long long run = -1;
    static const unsigned char z[16] = {};
    for (long long i = 0; i < n; ++i) {
        const unsigned char* hw = &j.h[(size_t)i * 16];
        const bool bad = hh[(size_t)i] != ph[(size_t)i];
        if (bad) {
            ++wrong;
            if (have_old && hh[(size_t)i] == oh[(size_t)i]) ++eq_old;
            else if (have_old && mode == 1) {
                /* A word partly old: some of its four dwords are the old push's, the rest the new. */
                const unsigned int* a = (const unsigned int*)hw;
                const unsigned int* c = (const unsigned int*)&r[(size_t)i * 16];
                const unsigned int* d = (const unsigned int*)&old[(size_t)i * 16];
                bool mixed = true;
                for (int q = 0; q < 4; ++q) if (a[q] != c[q] && a[q] != d[q]) mixed = false;
                if (mixed) ++part;
            }
            if (!std::memcmp(hw, z, 16)) ++zero;
            if (run < 0) { run = i; rs.push_back(i); rn.push_back(0); }
            ++rn.back();
        } else {
            run = -1;
        }
    }
    std::snprintf(b, sizeof b, "\n            SHADOW (the writer's own copy of the push, handshake %u, generation %d; it hashes "
                  "to the check word): host words that differ from it: %lld of %lld, in %zu run(s); of them equal "
                  "to the PREVIOUS push into this slot %lld%s, partly old and partly new %lld, zero %lld",
                  s_used, gen, wrong, n, rs.size(), eq_old, have_old ? "" : " (previous push not read)", part, zero);
    out += b;
    long long a64 = 0, a128 = 0, a256 = 0, a4k = 0, l4 = 0, l8 = 0, l256 = 0, mn = -1, mx = 0;
    for (size_t i = 0; i < rs.size(); ++i) {
        const unsigned long long addr = (unsigned long long)(uintptr_t)j.Haddr + (unsigned long long)rs[i] * 16;
        if (addr % 64 == 0) ++a64;
        if (addr % 128 == 0) ++a128;
        if (addr % 256 == 0) ++a256;
        if (addr % 4096 == 0) ++a4k;
        if (rn[i] % 4 == 0) ++l4;
        if (rn[i] % 8 == 0) ++l8;
        if (rn[i] % 256 == 0) ++l256;
        if (mn < 0 || rn[i] < mn) mn = rn[i];
        if (rn[i] > mx) mx = rn[i];
    }
    std::snprintf(b, sizeof b, "\n            runs: shortest %lld, longest %lld words (16 B); starting 64 B-aligned %lld, "
                  "128 B %lld, 256 B %lld, 4 KiB %lld; lengths a multiple of 64 B %lld, 128 B %lld, 4 KiB %lld. "
                  "The first runs (word, host address, words):", mn, mx, a64, a128, a256, a4k, l4, l8, l256);
    out += b;
    for (size_t i = 0; i < rs.size() && i < 16; ++i) {
        std::snprintf(b, sizeof b, " [%lld %p %lld]", j.start + rs[i], (const void*)(j.Haddr + rs[i] * 16), rn[i]);
        out += b;
    }
    /* Per run, in bytes: its offset from the slot's start, its host address modulo 64, 128 and
     * 4096, and its length; and the page each end lies in, with whether Windows maps it with large
     * pages (QueryWorkingSetEx). */
    out += "\n            runs in bytes (offset from the slot's start; address mod 64/128/4096; length):";
    for (size_t i = 0; i < rs.size() && i < 16; ++i) {
        const unsigned long long a = (unsigned long long)(uintptr_t)(j.Haddr + rs[i] * 16);
        std::snprintf(b, sizeof b, " [+%lld; %llu/%llu/%llu; %lld B]", (j.start + rs[i]) * 16, a % 64, a % 128,
                      a % 4096, rn[i] * 16);
        out += b;
    }
    {
        PSAPI_WORKING_SET_EX_INFORMATION wi[2] = {};
        wi[0].VirtualAddress = (PVOID)j.Haddr;
        const size_t hb = (size_t)n * 16;   /* the block's bytes in the host mailbox */
        wi[1].VirtualAddress = (PVOID)(j.Haddr + hb - 1);
        if (QueryWorkingSetEx(GetCurrentProcess(), wi, sizeof wi)) {
            std::snprintf(b, sizeof b, "\n            the block's pages: first %s, %s; last %s, %s; the block spans %llu page boundar%s",
                          wi[0].VirtualAttributes.Valid ? "resident" : "not resident",
                          wi[0].VirtualAttributes.LargePage ? "LARGE page" : "4 KiB page",
                          wi[1].VirtualAttributes.Valid ? "resident" : "not resident",
                          wi[1].VirtualAttributes.LargePage ? "LARGE page" : "4 KiB page",
                          (unsigned long long)(((uintptr_t)j.Haddr + hb - 1) / 4096 - (uintptr_t)j.Haddr / 4096),
                          ((uintptr_t)j.Haddr + hb - 1) / 4096 - (uintptr_t)j.Haddr / 4096 == 1 ? "y" : "ies");
            out += b;
        }
    }
    if (wrong && mode == 2) {
        const size_t i = (size_t)rs[0];
        std::snprintf(b, sizeof b, "\n            the first wrong word: host hash %08x; pushed %08x; previous push %08x",
                      hh[i], ph[i], have_old ? oh[i] : 0u);
        out += b;
    }
    if (wrong && mode == 1) {
        const size_t k = (size_t)rs[0] * 16;
        const unsigned int* hw = (const unsigned int*)&j.h[k];
        const unsigned int* rw = (const unsigned int*)&r[k];
        const unsigned int* ow = (const unsigned int*)&old[k];
        std::snprintf(b, sizeof b, "\n            the first wrong word: host %08x %08x %08x %08x; pushed %08x %08x %08x %08x; "
                      "previous push %08x %08x %08x %08x", hw[0], hw[1], hw[2], hw[3], rw[0], rw[1], rw[2], rw[3],
                      ow[0], ow[1], ow[2], ow[3]);
        out += b;
    }
    return out;
}

static void mb_bad_async(MbJob j) {
    std::string out = mb_shadow_cmp(j);
    char b[300];
    const MbGeo& g = j.g;
    const int slot = j.slot, blk = j.blk;
    const long long start = j.start, end = j.end;
    const long long n = end - start;
    const size_t bytes = (size_t)n * 16;
    const unsigned char* H = j.Haddr;
    const unsigned char* O = j.o.data();
    const unsigned char* P = j.p.empty() ? nullptr : j.p.data();
    std::vector<unsigned char>& h = j.h;
    std::vector<unsigned char> m(bytes), ly(bytes), py(bytes), lg, pg;
    std::string why;
    const bool have_m = j.stage &&
        mb_vram_read(j.dev_l, j.stage + ((long long)slot * j.slot16 + start) * 16, m.data(), bytes, 3000, &why);
    if (!have_m) { out += " [mirror: " + (j.stage ? why : std::string("none")) + "]"; why.clear(); }
    const MbGeo& gl = j.gl;
    const MbGeo& gp = j.gp;
    long long y_l = 0, y_p = 0, same_good = -1, good_n = 0;
    {
        const long long w = (g.w16 + g.nb - 1) / g.nb;
        const long long gs = (blk == 0 ? 1 : 0) * w;
        long long ge = gs + w;
        if (ge > g.w16) ge = g.w16;
        good_n = ge - gs;
        lg.resize((size_t)good_n * 16);
        pg.resize((size_t)good_n * 16);
        for (int i = 0; i < gl.nys; ++i)
            for (int jj = 0; jj < gp.nys; ++jj) {
                if (!mb_vram_read(j.dev_l, gl.ys[i] + gs * 16, lg.data(), lg.size(), 3000, &why) ||
                    !mb_vram_read(j.dev_p, gp.ys[jj] + gs * 16, pg.data(), pg.size(), 3000, &why))
                    continue;
                long long same = 0;
                for (long long k = 0; k < good_n; ++k)
                    if (!std::memcmp(&lg[(size_t)k * 16], &pg[(size_t)k * 16], 16)) ++same;
                if (same > same_good) { same_good = same; y_l = gl.ys[i]; y_p = gp.ys[jj]; }
            }
    }
    const bool have_y = y_l && y_p && same_good > good_n / 2 &&
        mb_vram_read(j.dev_l, y_l + start * 16, ly.data(), bytes, 3000, &why) &&
        mb_vram_read(j.dev_p, y_p + start * 16, py.data(), bytes, 3000, &why);
    if (!have_y) out += " [y: " + (why.empty() ? std::string("no message buffer noted") : why) + "]";
    std::snprintf(b, sizeof b, "\n            bad block b%d, words %lld..%lld (%lld), host %p; y this rank %llx, peer %llx; "
                  "validation (a good block's y equal on both ranks): %lld of %lld words",
                  blk, start, end - 1, n, (const void*)H, (unsigned long long)y_l, (unsigned long long)y_p,
                  same_good, good_n);
    out += b;
    /* Per word: wrong (not the peer's message, to rounding), and what else it equals. */
    long long wrong = 0, eq_m = 0, eq_o = 0, eq_p = 0, eq_own = 0, zero = 0, nwrong_w_m = 0;
    std::vector<long long> runs_s, runs_n;
    long long run = -1;
    for (long long k = 0; k < n; ++k) {
        const unsigned char* hw = &h[(size_t)k * 16];
        bool bad = false;
        if (have_y) {
            for (int e = 0; e < 8; ++e) {
                unsigned short hv, lv, pv;
                std::memcpy(&hv, hw + 2 * e, 2);
                std::memcpy(&lv, &ly[(size_t)k * 16 + 2 * e], 2);
                std::memcpy(&pv, &py[(size_t)k * 16 + 2 * e], 2);
                const float sum = mb_bf(pv), own = mb_bf(lv), got = mb_bf(hv);
                const float est = sum - own;
                const float ulp = std::fabs(sum) * (1.0f / 128.0f) + 1e-30f;
                if (!(std::fabs(got - est) <= ulp)) { bad = true; break; }
            }
        }
        if (have_m && !std::memcmp(hw, &m[(size_t)k * 16], 16)) ++eq_m;
        if (bad) {
            ++wrong;
            if (!std::memcmp(hw, O + k * 16, 16)) ++eq_o;
            if (P && !std::memcmp(hw, P + k * 16, 16)) ++eq_p;
            if (!std::memcmp(hw, &ly[(size_t)k * 16], 16)) ++eq_own;
            static const unsigned char z[16] = {};
            if (!std::memcmp(hw, z, 16)) ++zero;
            if (have_m && std::memcmp(hw, &m[(size_t)k * 16], 16)) ++nwrong_w_m;
            if (run < 0) { run = k; runs_s.push_back(k); runs_n.push_back(0); }
            ++runs_n.back();
        } else {
            run = -1;
        }
    }
    std::snprintf(b, sizeof b, "\n            mirror (what the card last read) equals host: %lld of %lld words%s; "
                  "wrong words where the mirror differs from host: %lld",
                  eq_m, n, have_m ? "" : " (no mirror read)", nwrong_w_m);
    out += b;
    if (have_y) {
        std::snprintf(b, sizeof b, "\n            WRONG words (not the peer's message): %lld of %lld, in %zu run(s); of them equal to "
                      "the other slot %lld, the peer's mailbox %lld, this rank's own message %lld, zero %lld",
                      wrong, n, runs_s.size(), eq_o, eq_p, eq_own, zero);
        out += b;
        long long a64 = 0, a128 = 0, a256 = 0, a4k = 0, l4 = 0, l8 = 0, l256 = 0;
        for (size_t i = 0; i < runs_s.size(); ++i) {
            const unsigned long long addr = (unsigned long long)(uintptr_t)H + (unsigned long long)runs_s[i] * 16;
            if (addr % 64 == 0) ++a64;
            if (addr % 128 == 0) ++a128;
            if (addr % 256 == 0) ++a256;
            if (addr % 4096 == 0) ++a4k;
            if (runs_n[i] % 4 == 0) ++l4;
            if (runs_n[i] % 8 == 0) ++l8;
            if (runs_n[i] % 256 == 0) ++l256;
        }
        std::snprintf(b, sizeof b, "\n            runs starting 64 B-aligned %lld, 128 B %lld, 256 B %lld, 4 KiB %lld; "
                      "lengths a multiple of 64 B %lld, 128 B %lld, 4 KiB %lld; the first runs (word, address, words):",
                      a64, a128, a256, a4k, l4, l8, l256);
        out += b;
        for (size_t i = 0; i < runs_s.size() && i < 12; ++i) {
            std::snprintf(b, sizeof b, " [%lld %p %lld]", start + runs_s[i],
                          (const void*)(H + runs_s[i] * 16), runs_n[i]);
            out += b;
        }
        if (wrong) {
            const long long k = runs_s[0];
            const unsigned int* hw = (const unsigned int*)&h[(size_t)k * 16];
            std::snprintf(b, sizeof b, "\n            the first wrong word: host %08x %08x %08x %08x; own %08x %08x %08x %08x; "
                          "peer sum %08x %08x %08x %08x",
                          hw[0], hw[1], hw[2], hw[3],
                          ((unsigned int*)&ly[(size_t)k * 16])[0], ((unsigned int*)&ly[(size_t)k * 16])[1],
                          ((unsigned int*)&ly[(size_t)k * 16])[2], ((unsigned int*)&ly[(size_t)k * 16])[3],
                          ((unsigned int*)&py[(size_t)k * 16])[0], ((unsigned int*)&py[(size_t)k * 16])[1],
                          ((unsigned int*)&py[(size_t)k * 16])[2], ((unsigned int*)&py[(size_t)k * 16])[3]);
            out += b;
        }
    }
    r4d_log(1, "r4d: bad-block comparison (rank %d's mailbox, slot %d, block %d; read after the "
            "dump):%s", j.rank, slot, blk, out.c_str());
}

/* ------------------------------------------------------------------ the stall capture
 *
 * Brilliance, diagnostic (BRILLIANCE_STALL_CAPTURE=1, 2026-11-18): WHAT ARE THE WRONG BYTES? For a
 * block of a host-mailbox push that matches no check word (mb_check), after the stall watch has seen
 * the stall: the block's bytes, where they came from, and a word-level diff against what the writer
 * meant to push, classified as
 *   old       the block holds other bytes, found in no buffer: by elimination the slot's previous
 *             message, the writer's write lost (or, "old with its check word", bytes and check word
 *             both hash to an older handshake value);
 *   foreign   the wrong bytes are found in another buffer, named, at an offset;
 *   correct, check word wrong   every word is what the writer pushed;
 *   partial   some words are the push, the rest are not (runs, alignment);
 *   unknown   the intended data could not be read.
 *
 * NOTHING IS RECORDED PER STEP. The push's source, the writer's `y`, is gone by the stall: the
 * writer reduced into it, and its next collective (ar_gather_hc_write) wrote its next message
 * there. But the hyper-connection stream `h` is the same on both cards, and the stuck write is the
 * only thing between the two cards' copies of it now:
 *   - the reader (this rank) has not reduced the bad block, so its `y` there still holds its own
 *     message, and its `h` there the stream BEFORE the write; its `inj` is still this write's
 *     (hc_read writes x and inj, never h; the reader has run nothing since);
 *   - the writer passed this write in every block, so its `h` holds the stream AFTER it,
 *     h += inj * bf16(writer's word + reader's word) (r4d_ar_hc_write_word), and the writer's gather
 *     is still in its flag wait, before the reduce that would write h again.
 * So for each element of the block, the bf16 values the writer could have pushed are those x with
 * f2bf(h_before + inj_c * rne(own + x)) == h_after for every stream c, the kernels' exact arithmetic
 * (libr4d is built -ffp-contract=off: two roundings). The host's element passes or fails that
 * exactly; the set of passing x is also printed, as the intended value's interval. The good blocks
 * validate the reads: there the reader's h equals the writer's.
 *
 * The bytes' origin: every 8-byte window of the block at every even offset is looked up in every
 * pinned host buffer the engine owns (the backend's pinned allocations, handed over by
 * r4d_win_capture_ranges at the stall: the expert pool, the KV host tier, the prefix cache's host
 * storage, staging rings; and every collective's mailbox slots and flags) and in both cards'
 * messages; a window that matches is extended both ways, and matches of 64 bytes or more are
 * reported with the buffer and offset.
 *
 * Card memory is read through the stream and pinned buffer made at the first launch (MbDev); each
 * read reports whether it completed while the cards are stuck. Printed to the serve log, written
 * to "<serve log>.stall-capture.txt" beside it (the process's stderr file; the working directory if
 * stderr is not a file), the raw arrays to ".bin" files beside that. */

static std::mutex            g_cap_mu;
static std::vector<CapRange> g_cap_host;   /* the backend's pinned allocations (r4d_win_capture_ranges) */

/* The bf16 conversions the kernels use: __float2bfloat16 (round to nearest even, vadd) and f2bf
 * (r4d_gdn_wmma.h: round half away, r4d_ar_hc_write_word). */
static unsigned short cap_rne(float f) {
    unsigned int u;
    std::memcpy(&u, &f, 4);
    if (~u & 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);
    else if (u & 0xFFFFu) u |= 0x10000u;
    return (unsigned short)(u >> 16);
}
static unsigned short cap_away(float f) {
    unsigned int u;
    std::memcpy(&u, &f, 4);
    return (unsigned short)((u + 0x8000u) >> 16);
}
/* One stream's element of the hc write: f2bf(bf2f(hb) + g * bf2f(v)), multiply and add rounded
 * apart (volatile keeps the host compiler from fusing them, as -ffp-contract=off keeps the card's). */
static float cap_hstep(unsigned short hb, float g, unsigned short v) {
    volatile float p = g * mb_bf(v);
    const float s = mb_bf(hb) + p;
    return mb_bf(cap_away(s));
}
/* The finite bf16 values in increasing order: index 0 the most negative, kOrdN - 1 the largest. */
static const int kCapOrdN = 65280;
static unsigned short cap_bits(int i) {
    return i < 32640 ? (unsigned short)(0x8000 | (0x7F7F - i)) : (unsigned short)(i - 32640);
}
static int cap_first(int a, int b, const std::function<bool(int)>& pred) {
    while (a < b) {
        const int m = a + (b - a) / 2;
        if (pred(m)) b = m; else a = m + 1;
    }
    return a;
}
/* The sums v (ordered indices [*lo, *hi)) the writer's h_after allows, given h_before and inj. */
static void cap_v_range(const unsigned short* hb, const unsigned short* ha, const float* g, int hc,
                        int* lo, int* hi) {
    int L = 0, H = kCapOrdN;
    for (int c = 0; c < hc && L < H; ++c) {
        const float t = mb_bf(ha[c]);
        if (!(g[c] > 0.0f) && !(g[c] < 0.0f)) {
            if (!(mb_bf(hb[c]) == t)) L = H;   /* a zero gain leaves h as it was */
            continue;
        }
        const unsigned short hbc = hb[c];
        const float gc = g[c];
        auto y = [&](int i) { return cap_hstep(hbc, gc, cap_bits(i)); };
        int l, h;
        if (gc > 0.0f) {
            l = cap_first(0, kCapOrdN, [&](int i) { return y(i) >= t; });
            h = cap_first(0, kCapOrdN, [&](int i) { return y(i) > t; });
        } else {
            l = cap_first(0, kCapOrdN, [&](int i) { return y(i) <= t; });
            h = cap_first(0, kCapOrdN, [&](int i) { return y(i) < t; });
        }
        L = std::max(L, l);
        H = std::min(H, h);
    }
    *lo = L;
    *hi = H;
}
/* The pushed values x (ordered indices [*lo, *hi)) whose sum with the reader's own `own` falls in
 * the allowed sums [vlo, vhi). */
static void cap_x_range(unsigned short own, int vlo, int vhi, int* lo, int* hi) {
    if (vlo >= vhi) { *lo = *hi = 0; return; }
    const float a = mb_bf(cap_bits(vlo)), b = mb_bf(cap_bits(vhi - 1));
    auto s = [&](int i) { return mb_bf(cap_rne(mb_bf(own) + mb_bf(cap_bits(i)))); };
    *lo = cap_first(0, kCapOrdN, [&](int i) { return s(i) >= a; });
    *hi = cap_first(0, kCapOrdN, [&](int i) { return s(i) > b; });
}
/* Whether `x` is what the writer pushed, as far as h says: the kernels' arithmetic, exactly. */
static bool cap_x_ok(unsigned short x, unsigned short own, const unsigned short* hb,
                     const unsigned short* ha, const float* g, int hc) {
    const unsigned short v = cap_rne(mb_bf(own) + mb_bf(x));
    for (int c = 0; c < hc; ++c)
        if (!(cap_hstep(hb[c], g[c], v) == mb_bf(ha[c]))) return false;
    return true;
}

/* One source the origin search reads. */
struct CapSrc { const unsigned char* p = nullptr; long long bytes = 0; std::string label; };
/* A match: src[xoff + (t - hbeg)] == H[t] for t in [hbeg, hend). */
struct CapHit { int src = 0; long long xoff = 0, hbeg = 0, hend = 0; };

/* Every 8-byte window of H at an even offset, in a table; each source scanned at 16-byte steps, so a
 * copy of any stretch of H at any even alignment is found. Windows of four equal 16-bit lanes (zero
 * among them) are left out: they match everywhere. */
static std::vector<CapHit> cap_search(const std::vector<unsigned char>& H, const unsigned char* self,
                                      const std::vector<CapSrc>& srcs, std::string* notes) {
    std::vector<CapHit> out;
    const long long hn = (long long)H.size();
    if (hn < 64) return out;
    int bits = 4;
    while ((1LL << bits) < hn * 2) ++bits;
    const size_t tsz = (size_t)1 << bits;
    std::vector<uint64_t> tk(tsz, 0);
    std::vector<int> to(tsz, -1);
    const int bloom_bits = 23;
    std::vector<uint64_t> bloom(((size_t)1 << bloom_bits) / 64, 0);
    auto hsh = [](uint64_t k) { return k * 0x9E3779B97F4A7C15ull; };
    long long nkeys = 0;
    for (long long q = 0; q + 8 <= hn; q += 2) {
        uint64_t k;
        std::memcpy(&k, &H[(size_t)q], 8);
        const uint16_t l0 = (uint16_t)k, l1 = (uint16_t)(k >> 16), l2 = (uint16_t)(k >> 32), l3 = (uint16_t)(k >> 48);
        if (l0 == l1 && l1 == l2 && l2 == l3) continue;
        const uint64_t x = hsh(k);
        size_t s = (size_t)(x >> (64 - bits));
        while (to[s] >= 0) s = (s + 1) & (tsz - 1);
        tk[s] = k;
        to[s] = (int)q;
        const uint64_t bb = x & (((uint64_t)1 << bloom_bits) - 1);
        bloom[bb >> 6] |= (uint64_t)1 << (bb & 63);
        ++nkeys;
    }
    /* Work in chunks of 64 MiB over all sources, on up to 16 threads. */
    struct Chunk { int src; long long a, b; };
    std::vector<Chunk> chunks;
    long long total = 0;
    for (size_t i = 0; i < srcs.size(); ++i) {
        const CapSrc& s = srcs[i];
        if (!s.p || s.bytes < 8) continue;
        total += s.bytes;
        for (long long a = 0; a < s.bytes; a += (64LL << 20))
            chunks.push_back(Chunk{ (int)i, a, std::min(s.bytes, a + (64LL << 20)) });
    }
    std::atomic<size_t> next{ 0 };
    std::atomic<bool> flooded{ false };
    std::mutex omu;
    const auto t0 = std::chrono::steady_clock::now();
    auto work = [&] {
        std::vector<CapHit> mine;
        for (;;) {
            const size_t ci = next.fetch_add(1);
            if (ci >= chunks.size()) break;
            const Chunk& c = chunks[ci];
            const CapSrc& s = srcs[(size_t)c.src];
            const unsigned char* X = s.p;
            /* 16-byte steps from the first 16-aligned address in the chunk. */
            long long p = c.a + (long long)((16 - ((uintptr_t)(X + c.a) & 15)) & 15);
            std::unordered_map<long long, long long> covered;   /* delta -> end of the last match */
            for (; p + 8 <= c.b; p += 16) {
                uint64_t k;
                std::memcpy(&k, X + p, 8);
                const uint64_t x = hsh(k);
                const uint64_t bb = x & (((uint64_t)1 << bloom_bits) - 1);
                if (!(bloom[bb >> 6] & ((uint64_t)1 << (bb & 63)))) continue;
                for (size_t sl = (size_t)(x >> (64 - bits)); to[sl] >= 0; sl = (sl + 1) & (tsz - 1)) {
                    if (tk[sl] != k) continue;
                    const long long q = to[sl];
                    const long long delta = p - q;   /* H[t] sits at X[delta + t] */
                    if (X + delta == self) continue;   /* the block itself */
                    auto cv = covered.find(delta);
                    if (cv != covered.end() && q < cv->second) continue;
                    long long beg = q, end = q + 8;
                    while (beg > 0 && delta + beg - 1 >= 0 && H[(size_t)beg - 1] == X[delta + beg - 1]) --beg;
                    while (end < hn && delta + end < s.bytes && H[(size_t)end] == X[delta + end]) ++end;
                    covered[delta] = end;
                    if (end - beg >= 64) mine.push_back(CapHit{ c.src, delta + beg, beg, end });
                    if (mine.size() > 20000) { flooded = true; break; }
                }
                if (flooded) break;
            }
            if (flooded) break;
        }
        std::lock_guard<std::mutex> lk(omu);
        out.insert(out.end(), mine.begin(), mine.end());
    };
    unsigned nt = std::thread::hardware_concurrency();
    nt = nt > 2 ? nt / 2 : 1;
    if (nt > 16) nt = 16;
    std::vector<std::thread> th;
    for (unsigned i = 0; i < nt; ++i) th.emplace_back(work);
    for (std::thread& t : th) t.join();
    const long long ms = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - t0).count();
    /* One hit per (source, offset): a chunk boundary can split a match in two. */
    std::sort(out.begin(), out.end(), [](const CapHit& a, const CapHit& b) {
        if (a.src != b.src) return a.src < b.src;
        const long long da = a.xoff - a.hbeg, db = b.xoff - b.hbeg;
        if (da != db) return da < db;
        return a.hbeg < b.hbeg;
    });
    std::vector<CapHit> merged;
    for (const CapHit& h : out) {
        if (!merged.empty()) {
            CapHit& m = merged.back();
            if (m.src == h.src && m.xoff - m.hbeg == h.xoff - h.hbeg && h.hbeg <= m.hend) {
                m.hend = std::max(m.hend, h.hend);
                continue;
            }
        }
        merged.push_back(h);
    }
    char b[300];
    std::snprintf(b, sizeof b, "searched %zu source(s), %.2f GiB, for %lld windows of the block, on %u "
                  "thread(s) in %lld ms%s", srcs.size(), (double)total / (1 << 30), nkeys, nt, ms,
                  flooded ? "; STOPPED EARLY: over 20,000 matches (repetitive data)" : "");
    *notes = b;
    return merged;
}

/* Where the capture file goes: beside the serve log, which is this process's stderr. */
static std::string cap_path() {
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    if (h && h != INVALID_HANDLE_VALUE && GetFileType(h) == FILE_TYPE_DISK) {
        wchar_t w[2048];
        const DWORD n = GetFinalPathNameByHandleW(h, w, 2048, FILE_NAME_NORMALIZED);
        if (n > 0 && n < 2048) {
            char u[4096];
            const int m = WideCharToMultiByte(CP_UTF8, 0, w, (int)n, u, (int)sizeof u - 1, nullptr, nullptr);
            if (m > 0) {
                u[m] = '\0';
                std::string s = u;
                if (s.rfind("\\\\?\\UNC\\", 0) == 0) s = "\\\\" + s.substr(8);
                else if (s.rfind("\\\\?\\", 0) == 0) s = s.substr(4);
                return s + ".stall-capture";
            }
        }
    }
    char b[64];
    std::snprintf(b, sizeof b, "radiance-%u.stall-capture", (unsigned)GetCurrentProcessId());
    return b;
}

/* The capture, on a thread of its own (mb_bad starts it). */
static void cap_run(const MbJob& j) {
    const auto t_start = std::chrono::steady_clock::now();
    const long long n = j.end - j.start;
    const size_t bytes = (size_t)n * 16;
    const MbGeo& g = j.g;
    std::string rep;   /* the full report; the serve log gets it too */
    char b[600];
    auto add = [&](const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(b, sizeof b, fmt, ap);
        va_end(ap);
        rep += b;
    };
    const int peer_rank = 1 - j.rank;
    const long long w = (g.w16 + g.nb - 1) / g.nb;
    const unsigned int salt = 0x6C8E9CF5u;
    unsigned int inv = salt;   /* the salt's inverse modulo 2^32 (Newton) */
    for (int i = 0; i < 5; ++i) inv *= 2u - salt * inv;

    /* 1. The block, from host memory. */
    unsigned int hash = 0u;
    for (long long k = 0; k < n; ++k) {
        const unsigned int* p = (const unsigned int*)&j.h[(size_t)k * 16];
        const unsigned int e = 4u * (unsigned int)(j.start + k);
        hash += mb_mix(p[0], e) + mb_mix(p[1], e + 1u) + mb_mix(p[2], e + 2u) + mb_mix(p[3], e + 3u);
    }
    const unsigned int s_fit = (j.want - hash) * inv;   /* the salt that would make it pass */
    const unsigned int s_push = j.flag - 1u;            /* the stuck push's (its good blocks' "prev") */
    std::vector<unsigned char> h2(bytes);
    std::memcpy(h2.data(), j.Haddr, bytes);
    const bool settled = h2 == j.h;
    long long eq_o = 0, eq_p = 0, zero = 0;
    static const unsigned char z16[16] = {};
    for (long long k = 0; k < n; ++k) {
        if (!std::memcmp(&j.h[(size_t)k * 16], &j.o[(size_t)k * 16], 16)) ++eq_o;
        if (!j.p.empty() && !std::memcmp(&j.h[(size_t)k * 16], &j.p[(size_t)k * 16], 16)) ++eq_p;
        if (!std::memcmp(&j.h[(size_t)k * 16], z16, 16)) ++zero;
    }
    add("\n  GEOMETRY: rank %d's mailbox (instance %p; card %d reads it, card %d writes it), slot %d, "
        "block %d of %d; words %lld..%lld (%lld words, %zu B) of a %lld-word message, %lld words a block; "
        "slot bytes +%lld..+%lld; host %p..%p",
        j.rank, j.inst, j.dev_l, j.dev_p, j.slot, j.blk, g.nb, j.start, j.end - 1, n, bytes, g.w16, w,
        j.start * 16, j.end * 16, (const void*)j.Haddr, (const void*)(j.Haddr + bytes));
    add("\n  CHECK WORD %08x; the writer's flag for the block %u, so the stuck push's handshake %u; the "
        "CPU's hash of the host bytes %08x, which would pass with handshake %u (%+d from the push's)",
        j.want, j.flag, s_push, hash, s_fit, (int)(s_fit - s_push));
    add("\n  HOST BYTES: settled (a second read equal): %s; words equal to the other slot's %lld, to the "
        "peer's mailbox's %lld, zero %lld (of %lld)", settled ? "yes" : "NO, they changed", eq_o, eq_p,
        zero, n);

    /* 2. Card memory, through the pre-made streams. */
    struct Rd { const char* what; int dev; bool ok; long long ms; std::string why; };
    std::vector<Rd> rds;
    auto read = [&](const char* what, int dev, long long src, std::vector<unsigned char>& dst, size_t nb) {
        dst.assign(nb, 0);
        std::string why;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = src && nb && mb_vram_read(dev, src, dst.data(), nb, 3000, &why);
        if (!src) why = "no address noted";
        const long long ms = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - t0).count();
        rds.push_back(Rd{ what, dev, ok, ms, why });
        if (!ok) dst.clear();
        return ok;
    };
    const MbGeo& gl = j.gl;
    const MbGeo& gp = j.gp;
    std::vector<unsigned char> mir, ly, py, lh, ph, li, lhg, phg;
    read("the reader's VRAM mirror of the block", j.dev_l,
         j.stage ? j.stage + ((long long)j.slot * j.slot16 + j.start) * 16 : 0, mir, bytes);
    read("the reader's message y (whole)", j.dev_l, gl.nys ? gl.ys[0] : 0, ly, (size_t)g.w16 * 16);
    read("the writer's y, the push's source (whole)", j.dev_p, gp.nys ? gp.ys[0] : 0, py, (size_t)g.w16 * 16);
    const int hc = gl.hc, nn = gl.n;
    const bool hc_ok = hc >= 1 && hc <= 8 && nn >= 8 && nn % 8 == 0 && gl.h && gl.inj && gp.h &&
                       gl.hrow >= (long long)hc * nn && gp.hrow >= (long long)hc * nn && gl.irow >= hc &&
                       gp.n == nn && gp.hc == hc;
    const long long wpr = hc_ok ? nn / 8 : 1;
    const long long r0 = j.start / wpr, r1 = (j.end - 1) / wpr;
    /* A good block: the existing check's choice, 1 for block 0, else 0. */
    const long long gs = (j.blk == 0 ? 1 : 0) * w, ge = std::min(gs + w, g.w16);
    const long long gr0 = gs / wpr, gr1 = (ge - 1) / wpr;
    if (hc_ok) {
        const size_t hb = (size_t)((r1 - r0) * gl.hrow + (long long)hc * nn) * 2;
        const size_t hbp = (size_t)((r1 - r0) * gp.hrow + (long long)hc * nn) * 2;
        read("the reader's h, the block's rows (before the write)", j.dev_l, gl.h + r0 * gl.hrow * 2, lh, hb);
        read("the writer's h, the block's rows (after the write)", j.dev_p, gp.h + r0 * gp.hrow * 2, ph, hbp);
        read("the reader's inj, the block's rows", j.dev_l, gl.inj + r0 * gl.irow * 2, li,
             (size_t)((r1 - r0) * gl.irow + hc) * 2);
        read("the reader's h, a good block's rows", j.dev_l, gl.h + gr0 * gl.hrow * 2, lhg,
             (size_t)((gr1 - gr0) * gl.hrow + (long long)hc * nn) * 2);
        read("the writer's h, a good block's rows", j.dev_p, gp.h + gr0 * gp.hrow * 2, phg,
             (size_t)((gr1 - gr0) * gp.hrow + (long long)hc * nn) * 2);
    }
    add("\n  CARD READS (the pre-made per-card stream, 3 s each; a read that does not complete stops "
        "further reads on its card):");
    for (const Rd& r : rds)
        add("\n    %s, card %d: %s in %lld ms%s%s", r.what, r.dev, r.ok ? "COMPLETED" : "FAILED", r.ms,
            r.ok ? "" : ": ", r.ok ? "" : r.why.c_str());
    if (!hc_ok)
        add("\n    the h test's operands were not noted (hc %d, n %d, h %llx / %llx, inj %llx): no "
            "reconstruction", hc, nn, (unsigned long long)gl.h, (unsigned long long)gp.h,
            (unsigned long long)gl.inj);
    if (!mir.empty()) {
        long long eq = 0;
        for (long long k = 0; k < n; ++k) if (!std::memcmp(&mir[(size_t)k * 16], &j.h[(size_t)k * 16], 16)) ++eq;
        add("\n    the mirror equals the host bytes in %lld of %lld words (what the reader last read)", eq, n);
    }
    if (!py.empty()) {
        long long eq_ps = 0, eq_ph = 0;
        for (long long k = 0; k < n; ++k) {
            const unsigned char* pw = &py[(size_t)(j.start + k) * 16];
            if (!std::memcmp(pw, &j.o[(size_t)k * 16], 16)) ++eq_ps;
            if (!std::memcmp(pw, &j.h[(size_t)k * 16], 16)) ++eq_ph;
        }
        add("\n    the writer's y at the block now: equal to the other slot (its next push, the gather's) in "
            "%lld of %lld words, to the bad bytes in %lld: the push's source %s", eq_ps, n, eq_ph,
            eq_ps * 10 >= n * 9 ? "is overwritten by the next message, as expected; the intended data "
                                 "comes from h below" : "holds something else (see the counts)");
    }

    /* 3. The reconstruction: per element, does h say the host's value is what the writer pushed? */
    const bool have_h = hc_ok && !lh.empty() && !ph.empty() && !li.empty() && !ly.empty();
    long long g_eq = -1, g_n = 0;
    if (hc_ok && !lhg.empty() && !phg.empty()) {
        g_eq = 0;
        for (long long k = gs; k < ge; ++k) {
            const long long row = k / wpr, i0 = (k - row * wpr) * 8;
            bool same = true;
            for (int c = 0; c < hc && same; ++c) {
                const size_t ol = (size_t)(((row - gr0) * gl.hrow + (long long)c * nn + i0) * 2);
                const size_t op = (size_t)(((row - gr0) * gp.hrow + (long long)c * nn + i0) * 2);
                same = !std::memcmp(&lhg[ol], &phg[op], 16);
            }
            if (same) ++g_eq;
            ++g_n;
        }
    }
    std::vector<unsigned char> wrong((size_t)n, 0);   /* per word: 1 not the push, 0 the push */
    long long n_wrong = 0, n_mixed = 0, el_bad_h = 0, el_n = 0, h_same = 0;
    /* The control: the other slot's words at the same offsets (the writer's next message, an
     * unrelated one) through the same test. Their pass rate is how often a wrong word looks right
     * here by chance: where h is large against the write, rounding hides a small change. */
    long long ctl_w = 0, ctl_el = 0;
    long long wb[5] = { 0, 0, 0, 0, 0 };   /* interval widths: 1, <=4, <=16, <=256, more */
    double mag_h = 0.0, mag_i = 0.0;
    long long mag_n = 0;
    std::string firsts;
    bool h_valid = false;
    if (have_h) {
        for (long long k = 0; k < n; ++k) {
            const long long kk = j.start + k;
            const long long row = kk / wpr, i0 = (kk - row * wpr) * 8;
            const unsigned short* hw = (const unsigned short*)&j.h[(size_t)k * 16];
            const unsigned short* own = (const unsigned short*)&ly[(size_t)kk * 16];
            float gc[8];
            for (int c = 0; c < hc; ++c)
                gc[c] = mb_bf(((const unsigned short*)li.data())[(row - r0) * gl.irow + c]);
            int bad_el = 0, ctl_ok = 0;
            bool same = true;
            std::string detail;
            const unsigned short* ow = (const unsigned short*)&j.o[(size_t)k * 16];
            for (int e = 0; e < 8; ++e) {
                unsigned short hb[8], ha[8];
                for (int c = 0; c < hc; ++c) {
                    hb[c] = ((const unsigned short*)lh.data())[(row - r0) * gl.hrow + (long long)c * nn + i0 + e];
                    ha[c] = ((const unsigned short*)ph.data())[(row - r0) * gp.hrow + (long long)c * nn + i0 + e];
                    if (hb[c] != ha[c]) same = false;
                }
                int vlo, vhi, xlo, xhi;
                cap_v_range(hb, ha, gc, hc, &vlo, &vhi);
                ++el_n;
                if (vlo >= vhi) { ++el_bad_h; ++bad_el; continue; }
                cap_x_range(own[e], vlo, vhi, &xlo, &xhi);
                const int wd = xhi - xlo;
                ++wb[wd <= 1 ? 0 : wd <= 4 ? 1 : wd <= 16 ? 2 : wd <= 256 ? 3 : 4];
                const bool ok = cap_x_ok(hw[e], own[e], hb, ha, gc, hc);
                if (cap_x_ok(ow[e], own[e], hb, ha, gc, hc)) { ++ctl_ok; ++ctl_el; }
                if (!ok) {
                    ++bad_el;
                    if (xhi > xlo) {
                        mag_h += std::fabs(mb_bf(hw[e]));
                        mag_i += std::fabs(mb_bf(cap_bits(xlo + (xhi - xlo) / 2)));
                        ++mag_n;
                    }
                }
                char d[96];
                std::snprintf(d, sizeof d, " %s%.5g[%.5g..%.5g]", ok ? "" : "*", mb_bf(hw[e]),
                              xhi > xlo ? mb_bf(cap_bits(xlo)) : 0.0f, xhi > xlo ? mb_bf(cap_bits(xhi - 1)) : 0.0f);
                detail += d;
            }
            if (same) ++h_same;
            if (ctl_ok == 8) ++ctl_w;
            if (bad_el) {
                wrong[(size_t)k] = 1;
                ++n_wrong;
                if (bad_el < 8) ++n_mixed;
                if (n_wrong <= 4) {
                    std::snprintf(b, sizeof b, "\n      word %lld (host %p):", kk, (const void*)(j.Haddr + k * 16));
                    firsts += b;
                    firsts += detail;
                }
            }
        }
        /* Usable: the good block's h agrees on both cards, the h change has a sum for nearly every
         * element, and an unrelated message fails in at least 90% of words. */
        h_valid = g_eq >= 0 && g_n > 0 && g_eq * 100 >= g_n * 99 && el_bad_h * 100 <= el_n &&
                  ctl_w * 10 < n;
    }
    add("\n  RECONSTRUCTION from h (the intended push, per element, from the reader's h before the write, "
        "the writer's h after it, the reader's inj and own message):");
    if (!have_h) {
        add(" NOT AVAILABLE (the reads above)");
    } else {
        add("\n    validation: a good block's h equal on both cards in %lld of %lld words%s; elements whose h "
            "change no sum explains %lld of %lld; words where the write left h unchanged %lld",
            g_eq, g_n, g_eq < 0 ? " (not read)" : "", el_bad_h, el_n, h_same);
        add("\n    control: the other slot's words (an unrelated message) pass in %lld of %lld words, %lld of %lld "
            "elements: the chance a wrong word looks right here", ctl_w, n, ctl_el, el_n);
        add("\n    %s", h_valid ? "VALID: the reconstruction is used" :
                                  "NOT VALID: the h reads do not fit this write, or h is too coarse here (the "
                                  "control passes in 10%% of words or more); the classification does not use them");
        add("\n    how precisely h pins each element (bf16 values allowed): 1: %lld, 2-4: %lld, 5-16: %lld, "
            "17-256: %lld, more: %lld", wb[0], wb[1], wb[2], wb[3], wb[4]);
        add("\n    WORDS NOT THE PUSH: %lld of %lld (%.2f%%); of them with only some of the 8 elements wrong %lld",
            n_wrong, n, n ? 100.0 * (double)n_wrong / (double)n : 0.0, n_mixed);
        if (mag_n)
            add("\n    wrong elements: mean |host value| %.4g against the intended %.4g", mag_h / (double)mag_n,
                mag_i / (double)mag_n);
        if (!firsts.empty()) rep += "\n    the first wrong words, each element host[intended range], * not the push:" + firsts;
    }

    /* The wrong words' runs: offsets in the slot, host alignment, lengths. */
    std::vector<long long> rs, rn;
    if (h_valid) {
        for (long long k = 0; k < n; ++k) {
            if (!wrong[(size_t)k]) continue;
            if (k == 0 || !wrong[(size_t)k - 1]) { rs.push_back(k); rn.push_back(0); }
            ++rn.back();
        }
        long long a64 = 0, a128 = 0, a4k = 0, l64 = 0, l128 = 0, l4k = 0, mn = -1, mx = 0;
        for (size_t i = 0; i < rs.size(); ++i) {
            const uintptr_t a = (uintptr_t)(j.Haddr + rs[i] * 16), e = a + (uintptr_t)rn[i] * 16;
            if (a % 64 == 0 && e % 64 == 0) ++a64;
            if (a % 128 == 0 && e % 128 == 0) ++a128;
            if (a % 4096 == 0 && e % 4096 == 0) ++a4k;
            if (rn[i] * 16 % 64 == 0) ++l64;
            if (rn[i] * 16 % 128 == 0) ++l128;
            if (rn[i] * 16 % 4096 == 0) ++l4k;
            if (mn < 0 || rn[i] < mn) mn = rn[i];
            if (rn[i] > mx) mx = rn[i];
        }
        if (!rs.empty()) {
            add("\n  WRONG RUNS: %zu; %lld to %lld B long; both ends on 64 B lines %lld, 128 B %lld, 4 KiB pages %lld; "
                "lengths a multiple of 64 B %lld, 128 B %lld, 4 KiB %lld; the first (byte offset in the slot, host "
                "address mod 64/128/4096, bytes):", rs.size(), mn * 16, mx * 16, a64, a128, a4k, l64, l128, l4k);
            for (size_t i = 0; i < rs.size() && i < 24; ++i) {
                const uintptr_t a = (uintptr_t)(j.Haddr + rs[i] * 16);
                add(" [+%lld; %llu/%llu/%llu; %lld]", (j.start + rs[i]) * 16, (unsigned long long)(a % 64),
                    (unsigned long long)(a % 128), (unsigned long long)(a % 4096), rn[i] * 16);
            }
        }
    }

    /* 4. Where the bytes came from. */
    std::vector<CapSrc> srcs;
    {
        std::lock_guard<std::mutex> lk(g_cap_mu);
        for (const CapRange& r : g_cap_host)
            srcs.push_back(CapSrc{ (const unsigned char*)r.base, r.bytes, r.label });
    }
    const size_t n_backend = srcs.size();
    for (const CapRange& r : j.boxes) srcs.push_back(CapSrc{ (const unsigned char*)r.base, r.bytes, r.label });
    char lb[160];
    if (!ly.empty()) {
        std::snprintf(lb, sizeof lb, "card %d (rank %d, the reader): its message y, read now", j.dev_l, j.rank);
        srcs.push_back(CapSrc{ ly.data(), (long long)ly.size(), lb });
    }
    if (!py.empty()) {
        std::snprintf(lb, sizeof lb, "card %d (rank %d, the writer): its y, read now (its next message)", j.dev_p, peer_rank);
        srcs.push_back(CapSrc{ py.data(), (long long)py.size(), lb });
    }
    std::string snote;
    const std::vector<CapHit> hits = cap_search(j.h, j.Haddr, srcs, &snote);
    /* Per source: bytes of the block covered, and of its wrong words. */
    std::vector<long long> cov((size_t)srcs.size(), 0), covw((size_t)srcs.size(), 0);
    std::vector<unsigned char> any(bytes, 0);
    for (size_t s = 0; s < srcs.size(); ++s) {
        std::vector<unsigned char> m(bytes, 0);
        for (const CapHit& hh : hits) if (hh.src == (int)s) std::fill(m.begin() + hh.hbeg, m.begin() + hh.hend, 1);
        for (size_t t = 0; t < bytes; ++t) {
            if (!m[t]) continue;
            any[t] = 1;
            ++cov[s];
            if (h_valid && wrong[t / 16]) ++covw[s];
        }
    }
    long long any_n = 0, wrong_b = n_wrong * 16;
    for (unsigned char c : any) any_n += c;
    int best = -1;
    for (size_t s = 0; s < srcs.size(); ++s) {
        const long long key = h_valid ? covw[s] : cov[s];
        if (key > 0 && (best < 0 || key > (h_valid ? covw[(size_t)best] : cov[(size_t)best]))) best = (int)s;
    }
    add("\n  ORIGIN SEARCH: %s; %zu backend pinned allocation(s)%s, %zu mailbox range(s), both cards' messages",
        snote.c_str(), n_backend, n_backend ? "" : " (NOT handed over: the backend's ranges are missing)",
        j.boxes.size());
    add("\n    matches of 64 B or more: %zu; block bytes covered by any %lld of %zu", hits.size(), any_n, bytes);
    {
        std::vector<CapHit> top = hits;
        std::sort(top.begin(), top.end(), [](const CapHit& a, const CapHit& c) { return a.hend - a.hbeg > c.hend - c.hbeg; });
        for (size_t i = 0; i < top.size() && i < 12; ++i)
            add("\n    [%s +0x%llx: block bytes %lld..%lld, %lld B]", srcs[(size_t)top[i].src].label.c_str(),
                (unsigned long long)top[i].xoff, top[i].hbeg, top[i].hend - 1, top[i].hend - top[i].hbeg);
    }

    /* 5. The verdict. */
    std::string cls, why;
    const bool near_salt = (int)(s_fit - s_push) >= -64 && (int)(s_fit - s_push) <= 4 && s_fit != s_push;
    if (h_valid && n_wrong == 0) {
        cls = "CORRECT, CHECK WORD WRONG";
        std::snprintf(b, sizeof b, "every one of the %lld words is what the writer pushed (h reconstruction, valid); "
                      "the check word %08x is not their hash with the push's handshake %u%s", n, j.want, s_push,
                      near_salt ? " but with an older one: the check word is stale" : "");
        why = b;
    } else if (best >= 0 && (h_valid ? covw[(size_t)best] * 2 >= wrong_b : cov[(size_t)best] * 2 >= (long long)bytes)) {
        cls = "FOREIGN " + srcs[(size_t)best].label;
        std::snprintf(b, sizeof b, "%lld of %lld %s bytes are found there%s", h_valid ? covw[(size_t)best] : cov[(size_t)best],
                      h_valid ? wrong_b : (long long)bytes, h_valid ? "wrong" : "block",
                      h_valid && n_wrong < n ? "; the other words are the push (partial)" : "");
        why = b;
    } else if (near_salt) {
        cls = "OLD, WITH ITS CHECK WORD";
        std::snprintf(b, sizeof b, "the host bytes hash to the check word with handshake %u, %d from the push's: "
                      "bytes and check word are both an older push's", s_fit, (int)(s_fit - s_push));
        why = b;
    } else if (h_valid && n_wrong == n) {
        cls = "OLD";
        std::snprintf(b, sizeof b, "none of the %lld words is the push; they are in no buffer searched, not zero (%lld), "
                      "not the other slot (%lld), not the peer's mailbox (%lld): by elimination the slot's previous "
                      "message, the writer's block never landed", n, zero, eq_o, eq_p);
        why = b;
    } else if (h_valid) {
        cls = "PARTIAL";
        std::snprintf(b, sizeof b, "%lld of %lld words are not the push, in %zu run(s), the rest are; the wrong bytes "
                      "are in no buffer searched (%lld of %lld found): by elimination older bytes left where part of "
                      "the write did not land", n_wrong, n, rs.size(), best >= 0 ? covw[(size_t)best] : 0LL, wrong_b);
        why = b;
    } else {
        cls = "UNKNOWN";
        why = !have_h ? "the intended data could not be read (CARD READS above)"
                      : "the reads of h did not validate (RECONSTRUCTION above)";
        if (best >= 0) {
            std::snprintf(b, sizeof b, "; %lld of %zu block bytes match %s", cov[(size_t)best], bytes,
                          srcs[(size_t)best].label.c_str());
            why += b;
        }
    }
    const long long total_ms = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - t_start).count();

    /* 6. The files: the report, and the raw arrays for offline analysis. */
    const std::string base = cap_path();
    char tag[64];
    std::snprintf(tag, sizeof tag, "-r%d-s%d-b%d.bin", j.rank, j.slot, j.blk);
    const std::string binp = base + tag;
    bool bin_ok = false;
    if (FILE* f = std::fopen(binp.c_str(), "wb")) {
        /* "BRLCAP1\0", then records: a 32-byte NUL-padded name, a little-endian u64 size, the bytes. */
        std::fwrite("BRLCAP1\0", 1, 8, f);
        auto rec = [&](const char* name, const void* p, size_t nb) {
            char nm[32] = {};
            std::snprintf(nm, sizeof nm, "%s", name);
            const uint64_t sz = nb;
            std::fwrite(nm, 1, 32, f);
            std::fwrite(&sz, 8, 1, f);
            if (nb) std::fwrite(p, 1, nb, f);
        };
        char meta[512];
        std::snprintf(meta, sizeof meta, "rank=%d slot=%d blk=%d nb=%d w16=%lld start=%lld end=%lld flag=%u want=%08x "
                      "hash=%08x n=%d hc=%d wpr=%lld r0=%lld hrow_l=%lld hrow_p=%lld irow=%lld host=%p dev_l=%d dev_p=%d",
                      j.rank, j.slot, j.blk, g.nb, g.w16, j.start, j.end, j.flag, j.want, hash, nn, hc, wpr, r0,
                      gl.hrow, gp.hrow, gl.irow, (const void*)j.Haddr, j.dev_l, j.dev_p);
        rec("meta", meta, std::strlen(meta));
        rec("host_block", j.h.data(), j.h.size());
        rec("other_slot_block", j.o.data(), j.o.size());
        rec("peer_mailbox_block", j.p.data(), j.p.size());
        rec("reader_mirror_block", mir.data(), mir.size());
        rec("reader_y_message", ly.data(), ly.size());
        rec("writer_y_message", py.data(), py.size());
        rec("reader_h_rows", lh.data(), lh.size());
        rec("writer_h_rows", ph.data(), ph.size());
        rec("reader_inj_rows", li.data(), li.size());
        rec("wrong_words", wrong.data(), wrong.size());
        bin_ok = std::fclose(f) == 0;
    }
    std::string head;
    std::snprintf(b, sizeof b, "STALL CAPTURE, rank %d's mailbox, slot %d, block %d: %s -- ", j.rank, j.slot, j.blk,
                  cls.c_str());
    head = b;
    head += why;
    std::snprintf(b, sizeof b, " (%lld ms; report %s.txt, arrays %s%s)", total_ms, base.c_str(), binp.c_str(),
                  bin_ok ? "" : " NOT WRITTEN");
    head += b;
    if (FILE* f = std::fopen((base + ".txt").c_str(), "ab")) {
        SYSTEMTIME st;
        GetLocalTime(&st);
        std::fprintf(f, "==== %04d-%02d-%02d %02d:%02d:%02d.%03d, pid %u\n%s%s\n\n", st.wYear, st.wMonth, st.wDay,
                     st.wHour, st.wMinute, st.wSecond, st.wMilliseconds, (unsigned)GetCurrentProcessId(),
                     head.c_str(), rep.c_str());
        std::fclose(f);
    }
    r4d_log(1, "r4d: %s%s", head.c_str(), rep.c_str());
}

/* Brilliance, diagnostic: the backend's pinned host allocations, for the stall capture's origin
 * search (win/device/hipnative.cpp hands them over at a stall, before the dump). Replaces the list. */
extern "C" void r4d_win_capture_ranges(const long long* pairs, const char* const* labels, int n) {
    std::lock_guard<std::mutex> lk(g_cap_mu);
    g_cap_host.clear();
    for (int i = 0; pairs && i < n; ++i)
        if (pairs[2 * i] && pairs[2 * i + 1] > 0)
            g_cap_host.push_back(CapRange{ pairs[2 * i], pairs[2 * i + 1],
                                           labels && labels[i] ? labels[i] : "a pinned allocation" });
}

/* Brilliance, diagnostic: the bad block's comparison. The host bytes are copied now; the card-side
 * part runs on a detached thread and prints its own line (see mb_bad_async). */
/* Brilliance: ends every park (r4d_ar_fault_park, r4d_ar_seq.h) on the two-rank host twins: each
 * parked block waits for its instance's release word to move. Every instance, since a capture may
 * name a block of either rank's mailbox. */
static void ar_park_release_all() {
    std::lock_guard<std::mutex> lk(g_cache_mu);
    for (const CacheSlot& s : g_cache) {
        const ArInst* in = s.in;
        if (!in || !in->host || in->world != 2 || !in->my_flags || in->flag_nbw <= 0) continue;
        volatile unsigned int* const c = (volatile unsigned int*)in->my_flags +
                                         R4D_AR2_CTL_OFF((unsigned)in->flag_nbw) + R4D_AR2_CTL_RELEASE;
        *c = *c + 1u;
    }
}

static std::string mb_bad(const ArInst* in, const ArInst* peer, const MbGeo& g, int slot, int blk,
                          long long start, long long end) {
    MbJob j;
    j.g = g;
    j.slot = slot;
    j.blk = blk;
    j.rank = in->rank;
    j.start = start;
    j.end = end;
    j.stage = in->my_stage;
    j.slot16 = in->slot16;
    j.peer_stage = peer ? peer->my_stage : 0;
    {
        const volatile unsigned int* f = (const volatile unsigned int*)in->my_flags;
        const int nbw = in->max_blocks;
        j.flag = f[blk];
        j.want = f[nbw + slot * nbw + blk];
    }
    const size_t bytes = (size_t)(end - start) * 16;
    j.Haddr = (const unsigned char*)(in->my_scratch + ((long long)slot * in->slot16 + start) * 16);
    const unsigned char* O = (const unsigned char*)(in->my_scratch + ((long long)(slot ^ 1) * in->slot16 + start) * 16);
    j.h.assign(j.Haddr, j.Haddr + bytes);
    j.o.assign(O, O + bytes);
    if (peer) {
        const unsigned char* P = (const unsigned char*)(peer->my_scratch + ((long long)slot * peer->slot16 + start) * 16);
        j.p.assign(P, P + bytes);
    }
    {
        std::lock_guard<std::mutex> lk(g_mbgeo_mu);
        auto a = g_mbgeo.find(in);
        if (a != g_mbgeo.end()) for (const MbGeo& x : a->second) if (x.w16 == g.w16 && x.nys) { j.gl = x; break; }
        auto c = peer ? g_mbgeo.find(peer) : g_mbgeo.end();
        if (peer && c != g_mbgeo.end()) for (const MbGeo& x : c->second) if (x.w16 == g.w16 && x.nys) { j.gp = x; break; }
        auto d = g_mbinst_dev.find(in);
        if (d != g_mbinst_dev.end()) j.dev_l = d->second;
        auto e = peer ? g_mbinst_dev.find(peer) : g_mbinst_dev.end();
        if (peer && e != g_mbinst_dev.end()) j.dev_p = e->second;
    }
    /* Host-only, now: how the block compares with the other slot and the peer's mailbox. */
    long long eq_o = 0, eq_p = 0, zero = 0;
    static const unsigned char z[16] = {};
    for (size_t k = 0; k < bytes; k += 16) {
        if (!std::memcmp(&j.h[k], &j.o[k], 16)) ++eq_o;
        if (!j.p.empty() && !std::memcmp(&j.h[k], &j.p[k], 16)) ++eq_p;
        if (!std::memcmp(&j.h[k], z, 16)) ++zero;
    }
    char b[300];
    std::snprintf(b, sizeof b, " [host: of %lld words, %lld equal the other slot's, %lld the peer mailbox's, "
                  "%lld zero; cards %d (this) and %d (peer); the card-side comparison follows as a line "
                  "of its own]", (long long)(bytes / 16), eq_o, eq_p, zero, j.dev_l, j.dev_p);
    if (cap_on()) {
        /* The stall capture: every two-rank host mailbox's ranges, for its origin search. The
         * caller (r4d_win_mailbox_dump) holds g_cache_mu. Then, on one thread, the capture and,
         * with BRILLIANCE_SLOG_LAUNCHES too, the older comparison after it (they share the staging
         * buffers). */
        j.inst = in;
        for (const CacheSlot& s : g_cache) {
            const ArInst* x = s.in;
            if (!x || !x->host || x->world != 2) continue;
            char lb[160];
            for (int sl = 0; sl < 2; ++sl) {
                std::snprintf(lb, sizeof lb, "rank %d's mailbox (instance %p, numel %lld), slot %d", x->rank,
                              (const void*)x, x->max_numel, sl);
                if (x->my_scratch)
                    j.boxes.push_back(CapRange{ x->my_scratch + (long long)sl * x->slot16 * 16, x->slot16 * 16, lb });
            }
            std::snprintf(lb, sizeof lb, "rank %d's mailbox flags (instance %p)", x->rank, (const void*)x);
            if (x->my_flags) j.boxes.push_back(CapRange{ x->my_flags, (long long)R4D_AR2_FLAG_WORDS(x->flag_nbw) * 4, lb });
        }
        std::thread([](MbJob jj) {
            cap_run(jj);
            /* The capture has read what it needs: a block parked on its give-up may reduce now. */
            ar_park_release_all();
            if (slog_on()) mb_bad_async(std::move(jj));
        }, std::move(j)).detach();
        std::string r = b;
        return r + " [stall capture: running, its report follows as a line of its own]";
    }
    std::thread(mb_bad_async, std::move(j)).detach();
    return b;
}

/* r4d_archk_mix / r4d_archk_u4, on the host. */
static unsigned int mb_mix(unsigned int v, unsigned int e) {
    unsigned int x = v ^ (e * 0x9E3779B9u);
    x ^= x >> 16; x *= 0x85EBCA6Bu;
    x ^= x >> 13; x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x;
}

/* Per (geometry, slot): how many of the geometry's blocks hold bytes that hash to the slot's check
 * word salted with the handshake value the peer last published for that block ("cur"), with one
 * less ("prev": the launch before), or with neither ("none"), and the first few "none" blocks. */
static std::string mb_check(const ArInst* in, const ArInst* peer) {
    std::vector<MbGeo> geos;
    {
        std::lock_guard<std::mutex> lk(g_mbgeo_mu);
        auto it = g_mbgeo.find(in);
        if (it != g_mbgeo.end()) geos = it->second;
    }
    if (geos.empty() || !in->my_scratch || in->slot16 <= 0) return "";
    const volatile unsigned int* f = (const volatile unsigned int*)in->my_flags;
    const int nbw = in->max_blocks;
    std::string out = "\n        host-side push check (bytes in pinned host memory against the peer's check words):";
    char b[160];
    for (const MbGeo& g : geos) {
        if (g.nb <= 0 || g.nb > nbw || g.w16 <= 0 || g.w16 > in->slot16) continue;
        for (int slot = 0; slot < 2; ++slot) {
            const volatile unsigned int* base =
                (const volatile unsigned int*)(in->my_scratch + (long long)slot * in->slot16 * 16);
            const long long w = (g.w16 + g.nb - 1) / g.nb;
            int cur = 0, prev = 0, none = 0;
            std::string bad;
            auto hash = [&](long long start, long long end) {
                unsigned int h = 0u;
                for (long long k = start; k < end; ++k) {
                    const unsigned int e = 4u * (unsigned int)k;
                    const volatile unsigned int* p = base + 4 * k;
                    h += mb_mix(p[0], e) + mb_mix(p[1], e + 1u) + mb_mix(p[2], e + 2u) +
                         mb_mix(p[3], e + 3u);
                }
                return h;
            };
            for (int blk = 0; blk < g.nb; ++blk) {
                const long long start = (long long)blk * w;
                long long end = start + w;
                if (end > g.w16) end = g.w16;
                const unsigned int h = hash(start, end);
                const unsigned int want = f[nbw + slot * nbw + blk];
                const unsigned int s = f[blk];
                if (h + s * 0x6C8E9CF5u == want) ++cur;
                else if (h + (s - 1u) * 0x6C8E9CF5u == want) ++prev;
                else {
                    ++none;
                    /* For the geometry the slot's other blocks match (a mixed result), what the odd
                     * block holds: its zero 16-byte words, its first and last words, and whether a
                     * second read 200 ms later hashes the same (bytes still landing, or settled). */
                    if (none <= 4) {
                        std::snprintf(b, sizeof b, " b%d(hash %08x, word %08x, flag %u)", blk, h, want, s);
                        bad += b;
                    }
                    if (none <= 2 && (cur + prev) > 0) {
                        long long zeros = 0;
                        for (long long k = start; k < end; ++k) {
                            const volatile unsigned int* p = base + 4 * k;
                            if (!p[0] && !p[1] && !p[2] && !p[3]) ++zeros;
                        }
                        const volatile unsigned int* p0 = base + 4 * start;
                        const volatile unsigned int* p1 = base + 4 * (end - 1);
                        /* The reader's own words (r4d_archk_wait): failed tries and the sum it
                         * computed, read now and again after 200 ms. */
                        const volatile unsigned int* dw = f + 3 * nbw + 2 * blk;
                        const unsigned int tr1 = dw[0], cs1 = dw[1];
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        const unsigned int h2 = hash(start, end);
                        const unsigned int tr2 = dw[0], cs2 = dw[1];
                        char d[400];
                        std::snprintf(d, sizeof d, " [words %lld..%lld: %lld zero; first %08x %08x %08x %08x, "
                                      "last %08x %08x %08x %08x; 200 ms later hash %08x (%s); the card's "
                                      "failed tries %u then %u, its sum %08x then %08x (the CPU's %08x)]",
                                      start, end - 1, zeros, p0[0], p0[1], p0[2], p0[3], p1[0], p1[1],
                                      p1[2], p1[3], h2, h2 == h ? "same" : "CHANGED", tr1, tr2, cs1, cs2, h);
                        bad += d;
                        bad += mb_bad(in, peer, g, slot, blk, start, end);
                    }
                }
            }
            std::snprintf(b, sizeof b, "\n          words16 %lld, %d blocks, slot %d: cur %d, prev %d, none %d",
                          g.w16, g.nb, slot, cur, prev, none);
            out += b;
            out += bad;
        }
    }
    return out;
}

static bool arf_describe(const ArInst* in, std::string* out);   /* below: the fault record */
static const char* arf_op_name(unsigned int op);              /* below: a collective's name */

extern "C" int r4d_win_mailbox_dump(char* buf, int cap) {
    if (!buf || cap <= 0) return 0;
    buf[0] = '\0';
    std::string out;
    int n = 0;
    std::lock_guard<std::mutex> lk(g_cache_mu);
    for (const CacheSlot& s : g_cache) {
        const ArInst* in = s.in;
        if (in && in->wide_x && in->my_flags && in->flag_nbw > 0) {
            /* W1's four-rank host instances: each source's flag per block (the design count), the
             * fault record, and the give-ups and late passes by source rank. */
            const volatile unsigned int* f = (const volatile unsigned int*)in->my_flags;
            const unsigned int nbw = (unsigned int)in->flag_nbw, ws = (unsigned int)in->world;
            const int nb = in->nb_cap < (int)nbw ? in->nb_cap : (int)nbw;
            char b[160];
            std::snprintf(b, sizeof b, "\n    four-rank rank %d inst %p numel %lld: flags [block: from "
                          "rank 0..%u]", in->rank, (const void*)in, in->max_numel, ws - 1u);
            out += b;
            for (int i = 0; i < nb; ++i) {
                out += " [";
                for (unsigned int q = 0; q < ws; ++q) {
                    std::snprintf(b, sizeof b, q ? " %u" : "%u", f[(unsigned)i * ws + q]);
                    out += b;
                }
                out += "]";
            }
            std::string fr;
            if (arf_describe(in, &fr)) out += "\n        fault on record: " + fr;
            long long gu[8] = { 0 }, la[8] = { 0 };
            const volatile unsigned int* const gs = f + R4D_ARW_SRC_OFF(ws, nbw);
            for (unsigned int bb = 0; bb < nbw; ++bb)
                for (unsigned int q = 0; q < ws && q < 8u; ++q) {
                    gu[q] += gs[R4D_ARW_SRC_GU * ws * nbw + bb * ws + q];
                    la[q] += gs[R4D_ARW_SRC_LATE * ws * nbw + bb * ws + q];
                }
            out += "\n        give-ups by source rank:";
            for (unsigned int q = 0; q < ws && q < 8u; ++q) { std::snprintf(b, sizeof b, " %lld", gu[q]); out += b; }
            out += "; late passes by source rank:";
            for (unsigned int q = 0; q < ws && q < 8u; ++q) { std::snprintf(b, sizeof b, " %lld", la[q]); out += b; }
            /* The exchange (W2/W3): requests raised by source rank, and per collective when any moved. */
            long long rqs[8] = { 0 };
            for (unsigned int bb = 0; bb < nbw; ++bb)
                for (unsigned int q = 0; q < ws && q < 8u; ++q) rqs[q] += gs[R4D_ARW_SRC_RQ * ws * nbw + bb * ws + q];
            out += "; resend requests by source rank:";
            for (unsigned int q = 0; q < ws && q < 8u; ++q) { std::snprintf(b, sizeof b, " %lld", rqs[q]); out += b; }
            static const unsigned int k_wops[6] = { R4D_AR_OP_W_HC, R4D_AR_OP_W_GHC, R4D_AR_OP_W_ONESHOT,
                                                    R4D_AR_OP_W_GATHER, R4D_AR_OP_W_TWOSHOT,
                                                    R4D_AR_OP_W_CODED };
            for (int o = 0; o < 6; ++o) {
                long long c4[4] = { 0, 0, 0, 0 };
                const volatile unsigned int* const k = f + R4D_ARW_CNT_OFF(ws, nbw);
                for (unsigned int bb = 0; bb < nbw; ++bb)
                    for (int j = 0; j < 4; ++j) c4[j] += k[(r4d_arw_op_index(k_wops[o]) * nbw + bb) * 8u + R4D_ARW_CNT_RQ + (unsigned)j];
                if (c4[0] | c4[1] | c4[2] | c4[3]) {
                    char cb[240];
                    std::snprintf(cb, sizeof cb, "\n        exchange (%s): %lld request(s) raised, %lld "
                                  "resend(s) served, %lld control re-store(s), %lld ack(s) never came",
                                  arf_op_name(k_wops[o]), c4[0], c4[1], c4[2], c4[3]);
                    out += cb;
                }
            }
            ++n;
            continue;
        }
        if (!in || !in->host || in->world != 2 || !in->my_flags) continue;
        const volatile unsigned int* f = (const volatile unsigned int*)in->my_flags;
        const int nbw = in->max_blocks, nb = in->nb_cap < nbw ? in->nb_cap : nbw;
        char b[96];
        std::snprintf(b, sizeof b, "\n    rank %d inst %p numel %lld: peer's flags", in->rank,
                      (const void*)in, in->max_numel);
        out += b;
        for (int i = 0; i < nb; ++i) {
            std::snprintf(b, sizeof b, " %u", f[i]);
            out += b;
        }
        out += "; check words";
        for (int slot = 0; slot < 2; ++slot) {
            std::snprintf(b, sizeof b, " [%d]", slot);
            out += b;
            for (int i = 0; i < nb && i < 4; ++i) {
                std::snprintf(b, sizeof b, " %08x", f[nbw + slot * nbw + i]);
                out += b;
            }
        }
        const ArInst* peer = nullptr;
        for (const CacheSlot& t : g_cache)
            if (t.in && t.in != in && t.in->host && t.in->world == 2 && t.in->group == in->group &&
                t.in->max_numel == in->max_numel && t.in->dtype == in->dtype && t.in->rank != in->rank)
                peer = t.in;
        out += mb_check(in, peer);
        std::string fr;   /* the bounded failure's record, if a block of this instance gave up */
        if (arf_describe(in, &fr)) out += "\n        fault on record: " + fr;
        /* the exchange's counters (fix C1 stages 2 to 4), per collective, when any moved */
        static const unsigned int k_xops[7] = { R4D_AR_OP_HC_WRITE, R4D_AR_OP_GATHER_HC,
                                                R4D_AR_OP_EXACT, R4D_AR_OP_GATHER,
                                                R4D_AR_OP_WHT6, R4D_AR_OP_HC_WHT6,
                                                R4D_AR_OP_GHC_WHT6 };
        for (int o = 0; o < 7; ++o) {
            long long c[4] = { 0, 0, 0, 0 };
            const unsigned int nbw = (unsigned)in->flag_nbw;
            const volatile unsigned int* const k = f + R4D_AR2_CNT_OP_OFF(nbw, k_xops[o]);
            for (int i = 0; i < in->flag_nbw; ++i)
                for (int j = 0; j < 4; ++j) c[j] += k[4 * i + j];
            if (c[0] | c[1] | c[2] | c[3]) {
                char cb[240];
                std::snprintf(cb, sizeof cb, "\n        exchange (%s): %lld request(s) raised, %lld "
                              "resend(s) served, %lld control re-store(s), %lld ack(s) never came",
                              arf_op_name(k_xops[o]), c[0], c[1], c[2], c[3]);
                out += cb;
            }
        }
        ++n;
    }
    std::snprintf(buf, (size_t)cap, "%s", out.c_str());
    return n;
}

/* Brilliance, diagnostic: every two-rank host mailbox's ranges in pinned host memory, as
 * (base, bytes) pairs: its slots and its flags. The native backend's stall dump tests the copies it
 * logged against them (win/device/hipnative.cpp). Returns the pairs written, up to `max_pairs`. */
extern "C" int r4d_win_mailbox_ranges(long long* out, int max_pairs) {
    if (!out || max_pairs <= 0) return 0;
    int n = 0;
    std::lock_guard<std::mutex> lk(g_cache_mu);
    for (const CacheSlot& s : g_cache) {
        const ArInst* in = s.in;
        if (!in || !in->host || in->world != 2) continue;
        if (n < max_pairs && in->my_scratch) { out[2 * n] = in->my_scratch; out[2 * n + 1] = 2LL * in->slot_bytes; ++n; }
        if (n < max_pairs && in->my_flags) { out[2 * n] = in->my_flags; out[2 * n + 1] = (long long)R4D_AR2_FLAG_WORDS(in->flag_nbw) * 4; ++n; }
    }
    return n;
}

/* ------------------------------------------------------------------ the bounded failure, host half
 *
 * Brilliance, fix C1 stage 1 (the bounded failure of the TP2 hang's host mailbox):
 * a two-rank host twin's block that gives up writes a fault record into its own flags (pinned host
 * memory; r4d_ar_seq.h has the layout and the device half). The engine reads the records after
 * every step through its backend (win/device/hipnative.cpp), before any token of the step goes out,
 * and stops on one. */

static const char* arf_op_name(unsigned int op) {
    switch (op) {
    case R4D_AR_OP_EXACT:     return "ar_oneshot_2rank_exact";
    case R4D_AR_OP_HC_WRITE:  return "ar_hc_write";
    case R4D_AR_OP_GATHER_HC: return "ar_gather_hc_write";
    case R4D_AR_OP_GATHER:    return "ar_gather_2rank";
    case R4D_AR_OP_WHT6:      return "ar_oneshot_2rank_wht6";
    case R4D_AR_OP_HC_WHT6:   return "ar_hc_write (wht6)";
    case R4D_AR_OP_GHC_WHT6:  return "ar_gather_hc_write (wht6)";
    case R4D_AR_OP_W_ONESHOT: return "ar_oneshot_4rank_exact";
    case R4D_AR_OP_W_TWOSHOT: return "ar_twoshot_4rank_exact";
    case R4D_AR_OP_W_CODED:   return "ar_twoshot_4rank_exact (coded)";
    case R4D_AR_OP_W_HC:      return "ar_hc_write_4rank";
    case R4D_AR_OP_W_GHC:     return "ar_gather_hc_write_4rank";
    case R4D_AR_OP_W_GATHER:  return "ar_gather_4rank";
    default:                  return "an unknown collective";
    }
}

/* Where an instance's fault record and control line are: a two-rank host twin's (R4D_AR2_*), or a
 * four-rank host instance's with the loud stop (W1, R4D_ARW_*). False for any other instance. */
static bool arf_where(const ArInst* in, unsigned int* rec_off, unsigned int* ctl_off,
                      unsigned int* faults_word) {
    if (!in || !in->host || !in->my_flags || in->flag_nbw <= 0) return false;
    const unsigned int nbw = (unsigned int)in->flag_nbw;
    if (in->world == 2) {
        *rec_off = R4D_AR2_REC_OFF(nbw);
        *ctl_off = R4D_AR2_CTL_OFF(nbw);
        *faults_word = R4D_AR2_CTL_FAULTS;
        return true;
    }
    if (in->wide_x) {
        *rec_off = R4D_ARW_REC_OFF(in->world, nbw);
        *ctl_off = R4D_ARW_CTL_OFF(in->world, nbw);
        *faults_word = R4D_ARW_CTL_FAULTS;
        return true;
    }
    return false;
}

/* A copy of the record is whole when it has the magic and its sum (as r4d_ar_fault_note) agrees. */
static bool arf_whole(const unsigned int* w) {
    if (w[R4D_ARF_MAGIC] != R4D_AR_FAULT_MAGIC) return false;
    unsigned int sum = R4D_AR_FAULT_SUM_SEED;
    for (unsigned int i = 0; i < R4D_ARF_SUM; ++i) sum += mb_mix(w[i], i);
    return sum == w[R4D_ARF_SUM];
}

/* One instance's fault, if it has one: what its record says, in one line. False when its record
 * and count are zero, the healthy case, which costs five loads of pinned host memory. */
static bool arf_describe(const ArInst* in, std::string* out) {
    unsigned int rec_off = 0, ctl_off = 0, fw = 0;
    if (!arf_where(in, &rec_off, &ctl_off, &fw)) return false;
    const volatile unsigned int* const rec = (const volatile unsigned int*)in->my_flags + rec_off;
    const volatile unsigned int* const ctl = (const volatile unsigned int*)in->my_flags + ctl_off;
    /* A lost store takes a whole 64-byte line (SUPPORT.md, issue 1), and each copy and the count
     * are lines of their own, so a written record shows in one of these five words. */
    const unsigned int count = ctl[fw];
    if (!count && !rec[R4D_ARF_MAGIC] && !rec[R4D_ARF_SUM] && !rec[R4D_ARF_WORDS + R4D_ARF_MAGIC] &&
        !rec[R4D_ARF_WORDS + R4D_ARF_SUM])
        return false;
    if (!out) return true;
    unsigned int a[R4D_ARF_WORDS], bw[R4D_ARF_WORDS];
    for (int i = 0; i < R4D_ARF_WORDS; ++i) { a[i] = rec[i]; bw[i] = rec[R4D_ARF_WORDS + i]; }
    const bool wa = arf_whole(a), wb = arf_whole(bw);
    const unsigned int* w = wa ? a : (wb ? bw : nullptr);
    char b[640];
    std::snprintf(b, sizeof b, "rank %d, instance %p (numel %lld): ", in->rank, (const void*)in,
                  in->max_numel);
    std::string o = b;
    if (!w) {
        std::snprintf(b, sizeof b, "a fault record that is not whole in either copy (%u give-up(s) "
                      "counted; copy A begins %08x %08x %08x %08x, copy B %08x %08x %08x %08x): a "
                      "block gave up, or a stray write reached the record; taken as a fault",
                      count, a[0], a[1], a[2], a[3], bw[0], bw[1], bw[2], bw[3]);
        o += b;
    } else {
        const unsigned int nf = count > w[R4D_ARF_COUNT] ? count : w[R4D_ARF_COUNT];
        const char* copy = wa ? "" : " (record copy A not whole; copy B read)";
        if (in->world != 2) {
            /* The wide record (W1, r4d_arw_fault_note): R13 the source rank, R14 (phase << 8) |
             * world; the slot word is the launch's parity. */
            const unsigned int src = w[R4D_ARF_R13], phase = w[R4D_ARF_R14] >> 8;
            const bool ts = w[R4D_ARF_OP] == R4D_AR_OP_W_TWOSHOT || w[R4D_ARF_OP] == R4D_AR_OP_W_CODED;
            char ph[24] = "";
            if (ts) std::snprintf(ph, sizeof ph, " (hop %u)", phase + 1u);
            if (w[R4D_ARF_KIND] == R4D_AR_FAULT_FLAG)
                std::snprintf(b, sizeof b, "%s%s, block %u of %u, from rank %u, phase %u, handshake "
                              "%u, slot %u: rank %u's flag stood at %u and had not reached %u after "
                              "%u us (a flag store lost, or that rank out of step); %u give-up(s) on "
                              "this instance%s",
                              arf_op_name(w[R4D_ARF_OP]), ph, w[R4D_ARF_BLOCK], w[R4D_ARF_NB], src,
                              phase, w[R4D_ARF_SEQ], w[R4D_ARF_SLOT], src, w[R4D_ARF_FLAG],
                              w[R4D_ARF_SEQ], w[R4D_ARF_US], nf, copy);
            else
                std::snprintf(b, sizeof b, "%s%s, block %u of %u, from rank %u, phase %u, handshake "
                              "%u, slot %u: the push check failed for %u us (%u tries; the card's "
                              "salted sum %08x, rank %u's check word %08x, its flag %u), so this "
                              "block's bytes from rank %u are wrong; %u give-up(s) on this instance%s",
                              arf_op_name(w[R4D_ARF_OP]), ph, w[R4D_ARF_BLOCK], w[R4D_ARF_NB], src,
                              phase, w[R4D_ARF_SEQ], w[R4D_ARF_SLOT], w[R4D_ARF_US],
                              w[R4D_ARF_TRIES], w[R4D_ARF_GOT], src, w[R4D_ARF_WANT],
                              w[R4D_ARF_FLAG], src, nf, copy);
        } else if (w[R4D_ARF_KIND] == R4D_AR_FAULT_FLAG)
            std::snprintf(b, sizeof b, "%s, block %u of %u, handshake %u, slot %u: the peer's flag "
                          "stood at %u and had not reached %u after %u us (a flag store lost, or the "
                          "peer out of step); %u give-up(s) on this instance%s",
                          arf_op_name(w[R4D_ARF_OP]), w[R4D_ARF_BLOCK], w[R4D_ARF_NB],
                          w[R4D_ARF_SEQ], w[R4D_ARF_SLOT], w[R4D_ARF_FLAG], w[R4D_ARF_SEQ],
                          w[R4D_ARF_US], nf, copy);
        else
            std::snprintf(b, sizeof b, "%s, block %u of %u, handshake %u, slot %u: the push check "
                          "failed for %u us (%u tries; the card's salted sum %08x, the peer's check "
                          "word %08x, the peer's flag %u), so this block's bytes from the peer are "
                          "wrong; %u give-up(s) on this instance%s",
                          arf_op_name(w[R4D_ARF_OP]), w[R4D_ARF_BLOCK], w[R4D_ARF_NB],
                          w[R4D_ARF_SEQ], w[R4D_ARF_SLOT], w[R4D_ARF_US], w[R4D_ARF_TRIES],
                          w[R4D_ARF_GOT], w[R4D_ARF_WANT], w[R4D_ARF_FLAG], nf, copy);
        o += b;
    }
    *out = o;
    return true;
}

/* Every two-rank host twin's and four-rank host instance's (W1) fault, described into `buf` (up to `cap` bytes, NUL-terminated, one
 * "; "-separated entry per instance with a fault); returns how many instances have one, 0 when
 * none has. Read-only, and safe at any time: the records are pinned host memory. The engine calls
 * it after every step (win/device/hipnative.cpp, n_coll_fault_poll). */
extern "C" int r4d_ar_fault_poll(char* buf, int cap) {
    if (buf && cap > 0) buf[0] = '\0';
    int n = 0;
    std::string out;
    std::lock_guard<std::mutex> lk(g_cache_mu);
    for (const CacheSlot& s : g_cache) {
        if (!arf_describe(s.in, nullptr)) continue;
        std::string one;
        (void)arf_describe(s.in, &one);
        if (n++) out += "; ";
        out += one;
    }
    if (n && buf && cap > 0) std::snprintf(buf, (size_t)cap, "%s", out.c_str());
    return n;
}

static long long ar_inj_made();   /* below: injections made since the switch was last set */

/* The host mailboxes' counters, summed over every two-rank host twin: out[0] the instances, out[1]
 * the give-ups (faults), then the exchange's (fix C1 stage 2; r4d_archk_exchange): out[2] resend
 * requests raised, out[3] resends served, out[4] control words re-stored, out[5] acks that never
 * came (the peer's next flag covered one, or it timed out); out[6] the injections made
 * (BRILLIANCE_AR_INJECT), so a run under injection can compare resends with them; out[7] to
 * out[10], ar_gather_hc_write's share of out[2] to out[5] (fix C1 stage 3, its own counter words,
 * R4D_AR2_CNT_GHC_OFF); out[11] to out[14] the plain exact all-reduce's share and out[15] to
 * out[18] the two-rank all-gather's (fix C1 stage 4, R4D_AR2_CNT_EXACT_OFF and _GATHER_OFF); on the
 * rotated wire (tp-wire wht6), out[19] to out[22] the wht6 all-reduce's share, out[23] to out[26]
 * the wht6 fused write's and out[27] to out[30] the wht6 fused write's with the gather
 * (R4D_AR2_CNT_WHT6_OFF, _HC_WHT6_OFF, _GHC_WHT6_OFF); the rest is ar_hc_write's. The four-rank host
 * instances with the loud stop (W1, r4d_ar_wide_x.h), from out[31]: out[31] the instances, out[32]
 * their give-ups (faults), out[33] late passes (a check that failed first and then agreed: a push that
 * landed after its flag), out[34] the injections whose damage changed bytes; out[35..38] the give-ups
 * by source rank 0-3, out[39..42] the late passes by source rank; out[43..47] the give-ups and
 * out[48..52] the late passes of ar_hc_write_4rank, ar_gather_hc_write_4rank, ar_oneshot_4rank_exact,
 * ar_gather_4rank and ar_twoshot_4rank_exact. Their exchange (W2/W3, r4d_arw_exchange): out[53..56]
 * requests raised, resends served, control words re-stored, acks that never came; out[57..60] the
 * requests raised by source rank 0-3 (the rank whose pushes failed); out[61..80] the four of each
 * collective in the order above (hc4, ghc4, exact4, gather4, ts4: ts4's from W4). The coded
 * two-shot's (W4, overlap 1): out[81..84] its four, out[85] its give-ups, out[86] its late passes
 * (also in the totals out[32], out[33] and out[53..56]).
 * Returns the entries written (up to `n`, at most
 * R4D_AR_MB_COUNTERS). Reads every counter word of every instance, so it is for a periodic log or a
 * stats query, never for every step. */
extern "C" int r4d_ar_mailbox_counters(long long* out, int n) {
    if (!out || n <= 0) return 0;
    long long v[R4D_AR_MB_COUNTERS] = { 0 };
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        for (const CacheSlot& s : g_cache) {
            const ArInst* in = s.in;
            if (in && in->wide_x && in->my_flags && in->flag_nbw > 0) {
                /* W1's four-rank host instances (R4D_ARW_*): out[31] on, described above. */
                const unsigned int nbw = (unsigned int)in->flag_nbw, ws = (unsigned int)in->world;
                const volatile unsigned int* const f = (const volatile unsigned int*)in->my_flags;
                ++v[31];
                const unsigned int c = f[R4D_ARW_CTL_OFF(ws, nbw) + R4D_ARW_CTL_FAULTS];
                v[32] += c ? (long long)c : (arf_describe(in, nullptr) ? 1 : 0);
                v[34] += f[R4D_ARW_CTL_OFF(ws, nbw) + R4D_ARW_CTL_INJEFF] +
                         f[R4D_ARW_CTL_OFF(ws, nbw) + R4D_ARW_CTL_INJEFF2];   /* W2: pair's partner block */
                const volatile unsigned int* const gs = f + R4D_ARW_SRC_OFF(ws, nbw);
                const volatile unsigned int* const ls = gs + R4D_ARW_SRC_LATE * ws * nbw;
                for (unsigned int b = 0; b < nbw; ++b)
                    for (unsigned int q = 0; q < ws && q < 4u; ++q) {
                        v[35 + q] += gs[R4D_ARW_SRC_GU * ws * nbw + b * ws + q];
                        v[39 + q] += ls[b * ws + q];
                        v[33] += ls[b * ws + q];
                    }
                static const unsigned int k_ops[5] = { R4D_AR_OP_W_HC, R4D_AR_OP_W_GHC,
                                                       R4D_AR_OP_W_ONESHOT, R4D_AR_OP_W_GATHER,
                                                       R4D_AR_OP_W_TWOSHOT };
                const volatile unsigned int* const k = f + R4D_ARW_CNT_OFF(ws, nbw);
                for (int o = 0; o < 5; ++o)
                    for (unsigned int b = 0; b < nbw; ++b) {
                        const volatile unsigned int* const kb = k + (r4d_arw_op_index(k_ops[o]) * nbw + b) * 8u;
                        v[43 + o] += kb[R4D_ARW_CNT_GU];
                        v[48 + o] += kb[R4D_ARW_CNT_LATE];
                        for (int j = 0; j < 4; ++j) {
                            const long long x = kb[R4D_ARW_CNT_RQ + (unsigned)j];
                            v[53 + j] += x;
                            v[61 + 4 * o + j] += x;
                        }
                    }
                /* W4's coded two-shot, after W2/W3's entries. */
                for (unsigned int b = 0; b < nbw; ++b) {
                    const volatile unsigned int* const kb = k + (r4d_arw_op_index(R4D_AR_OP_W_CODED) * nbw + b) * 8u;
                    v[85] += kb[R4D_ARW_CNT_GU];
                    v[86] += kb[R4D_ARW_CNT_LATE];
                    for (int j = 0; j < 4; ++j) {
                        const long long x = kb[R4D_ARW_CNT_RQ + (unsigned)j];
                        v[53 + j] += x;
                        v[81 + j] += x;
                    }
                }
                const volatile unsigned int* const rs = gs + R4D_ARW_SRC_RQ * ws * nbw;
                for (unsigned int b = 0; b < nbw; ++b)
                    for (unsigned int q = 0; q < ws && q < 4u; ++q) v[57 + q] += rs[b * ws + q];
                continue;
            }
            if (!in || !in->host || in->world != 2 || !in->my_flags || in->flag_nbw <= 0) continue;
            const unsigned int nbw = (unsigned int)in->flag_nbw;
            const volatile unsigned int* const f = (const volatile unsigned int*)in->my_flags;
            ++v[0];
            const unsigned int c = f[R4D_AR2_CTL_OFF(nbw) + R4D_AR2_CTL_FAULTS];
            /* A record whose count store was lost still counts once. */
            v[1] += c ? (long long)c : (arf_describe(in, nullptr) ? 1 : 0);
            const volatile unsigned int* const k = f + R4D_AR2_CNT_OFF(nbw);
            const volatile unsigned int* const g = f + R4D_AR2_CNT_GHC_OFF(nbw);
            const volatile unsigned int* const x = f + R4D_AR2_CNT_EXACT_OFF(nbw);
            const volatile unsigned int* const a = f + R4D_AR2_CNT_GATHER_OFF(nbw);
            const volatile unsigned int* const w6 = f + R4D_AR2_CNT_WHT6_OFF(nbw);
            const volatile unsigned int* const h6 = f + R4D_AR2_CNT_HC_WHT6_OFF(nbw);
            const volatile unsigned int* const g6 = f + R4D_AR2_CNT_GHC_WHT6_OFF(nbw);
            for (unsigned int b = 0; b < nbw; ++b)
                for (unsigned int j = 0; j < 4u; ++j) {
                    const long long gj = g[4u * b + j], xj = x[4u * b + j], aj = a[4u * b + j];
                    const long long wj = w6[4u * b + j], hj = h6[4u * b + j], qj = g6[4u * b + j];
                    v[2 + j] += k[4u * b + j] + gj + xj + aj + wj + hj + qj;
                    v[7 + j] += gj;
                    v[11 + j] += xj;
                    v[15 + j] += aj;
                    v[19 + j] += wj;
                    v[23 + j] += hj;
                    v[27 + j] += qj;
                }
        }
    }
    v[6] = ar_inj_made();
    const int m = n < R4D_AR_MB_COUNTERS ? n : R4D_AR_MB_COUNTERS;
    for (int i = 0; i < m; ++i) out[i] = v[i];
    return m;
}

/* Brilliance (2026-11-19, the owner's rule): THE HOST MAILBOXES' MEMORY TYPE, NAMED. The engine
 * calls this once per tensor-parallel group, after placement and before any weight load or launch
 * (Engine::plan, core/engine_bringup.cpp): `spill` when any routed expert of the group sits off the
 * cards, with this rank's `n_vram` of `n_experts` routed units in VRAM, for the log only.
 *
 * Every two-rank and four-rank host mailbox already holds its type: noncoherent-slots by default,
 * from its build (mailbox2_mode, mailbox4_nc_slots), so nothing moves here. (Until 2026-11-19 the
 * default followed placement, and this moved the two-rank data slots when experts spilled.) One
 * line per rank count the group has host instances at names the choice and why, and every
 * instance's hipHostGetFlags readback of its slots and flags (Mapped | Portable with NonCoherent
 * 0x80000003 or Coherent 0x40000003), so a flag the runtime ignored shows. Returns the instances
 * named. */
extern "C" int r4d_win_mailbox_default(long long group, int spill, long long n_vram,
                                       long long n_experts) {
    const std::string why = spill
        ? "experts spill to host: " + std::to_string(n_vram) + " of " + std::to_string(n_experts) + " in VRAM"
        : n_experts > 0 ? "every routed expert in VRAM: " + std::to_string(n_vram) + " of " +
                          std::to_string(n_experts)
                        : std::string("no routed experts");
    const int env = mailbox_mem_mode();

    std::lock_guard<std::mutex> lk(g_cache_mu);
    int named = 0;
    for (int world : { 2, 4 }) {
        /* The distinct readbacks over this rank count's instances, slots and flags. */
        std::map<unsigned, int> got_sc, got_fl;
        int n = 0;
        for (const CacheSlot& s : g_cache) {
            if (!s.in || s.key.group != group || s.key.world != world || !s.key.host) continue;
            ++n;
            for (int k = 0; k < 2; ++k) {
                unsigned f = 0;
                const long long a = k == 0 ? s.in->my_scratch : s.in->my_flags;
                if (hipHostGetFlags(&f, (void*)a) != hipSuccess) {
                    (void)hipGetLastError();
                    f = 0xffffffffu;
                }
                ++(k == 0 ? got_sc : got_fl)[f];
            }
        }
        if (!n) continue;
        named += n;
        auto list = [](const std::map<unsigned, int>& m) {
            std::string t;
            for (const auto& kv : m) {
                char b[48];
                if (kv.first == 0xffffffffu) std::snprintf(b, sizeof b, "%sFAILED x%d", t.empty() ? "" : ", ", kv.second);
                else std::snprintf(b, sizeof b, "%s0x%08x x%d", t.empty() ? "" : ", ", kv.first, kv.second);
                t += b;
            }
            return t;
        };
        const bool nc = world == 2 ? (env < 0 || env == 3 || env == kMbNcSlots) : mailbox4_nc_slots();
        const char* mode = nc ? "noncoherent-slots" : world == 2 && env > 0 ? kMbMemName[env] : "coherent";
        std::string reason;
        if (env < 0) reason = "the default at two and four ranks";
        else reason = std::string("BRILLIANCE_MAILBOX_MEM=") + kMbMemName[env] +
                      " overrides the default noncoherent-slots";
        std::fprintf(stderr, "I mailbox: %s-rank host mailboxes %s (%s; %s)%s [%d instance(s): slots %s; "
                     "flags %s]\n", world == 2 ? "two" : "four", mode, reason.c_str(), why.c_str(),
                     env < 0 ? "; set BRILLIANCE_MAILBOX_MEM to override" : "", n,
                     list(got_sc).c_str(), list(got_fl).c_str());
    }
    std::fflush(stderr);
    return named;
}
#endif

extern "C" void r4d_ar8_test_settings(const R4DAr8Settings* s) {
    std::lock_guard<std::mutex> lk(g_ar8_m);
    g_ar8 = s ? *s : R4DAr8Settings();
}

extern "C" int r4d_rad_ar_init(const RadParam* p, int n_p, int rank, int world, void** out) {
    return ar_init(p, n_p, rank, world, 0, out);
}
extern "C" int r4d_rad_ar_host_init(const RadParam* p, int n_p, int rank, int world, void** out) {
    return ar_init(p, n_p, rank, world, 1, out);
}

/* The fused all-reduce + norm declares `M` (range) and `n` instead of `numel`; the bound is
 * M * n, over the same world and buffers. */
static int ar_mn_acquire(const RadParam* p, int n_p, int rank, int world, int host,
                         void** out) {
    const long long M = pgeti_hi(p, n_p, "M", 0);
    const long long K = pgeti_hi(p, n_p, "n", 0);
    if (M <= 0 || K <= 0) return RAD_E_INVAL;

    ArInst* in = nullptr;
    const long long group = pgeti_hi(p, n_p, "tp_group", 0);
    int rc = acquire(CacheKey{ group, rank, world, 0 /* bf16 */, M * K, host }, &in);
    if (rc != RAD_OK) return rc;
    *out = in;
    return RAD_OK;
}

static int ar_ln_init(const RadParam* p, int n_p, int rank, int world, int host, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;

    const long long ws_param = pgeti_hi(p, n_p, "world_size", world);
    if (ws_param != world) return RAD_E_SHAPE;
    if (world != 2) return RAD_E_SHAPE;   /* the handshake is one peer flag per block */

    return ar_mn_acquire(p, n_p, rank, world, host, out);
}

extern "C" int r4d_rad_ar_ln_init(const RadParam* p, int n_p, int rank, int world, void** out) {
    return ar_ln_init(p, n_p, rank, world, 0, out);
}
extern "C" int r4d_rad_ar_ln_host_init(const RadParam* p, int n_p, int rank, int world,
                                       void** out) {
    return ar_ln_init(p, n_p, rank, world, 1, out);
}

/* The four-rank ar_hc_write, ar_gather_hc_write and ar_rmsnorm_quant_i8 (the norm declares no
 * `wire`). The key is {rank, 4, bf16, M_hi * n}, shared with a standalone all_reduce band only where
 * the bounds agree. Either way is safe on one stream: every wide kernel advances the instance's seq
 * counters once per all-reduce and writes only its (parity, source) slot (r4d_ar_wide.h), so ops on
 * one instance interleave in any order all ranks share. Exact wire only; `wire` 6 is refused. */
static int ar_hc_wide_init(const RadParam* p, int n_p, int rank, int world, int host,
                           void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;

    const long long ws_param = pgeti_hi(p, n_p, "world_size", world);
    if (ws_param != world) return RAD_E_SHAPE;
    if (world != 4) return RAD_E_SHAPE;   /* the peer bundle is compiled for four */
    if (pgeti_hi(p, n_p, "wire", 0) != 0) return RAD_E_UNSUPPORTED;

    return ar_mn_acquire(p, n_p, rank, world, host, out);
}

extern "C" int r4d_rad_ar_hc_wide_init(const RadParam* p, int n_p, int rank, int world,
                                       void** out) {
    return ar_hc_wide_init(p, n_p, rank, world, 0, out);
}
extern "C" int r4d_rad_ar_hc_wide_host_init(const RadParam* p, int n_p, int rank, int world,
                                            void** out) {
    return ar_hc_wide_init(p, n_p, rank, world, 1, out);
}

extern "C" void r4d_rad_ar_fini(void* instance) {
    /* One fini per successful init; the instance is shared, so the last release frees it. */
    release((ArInst*)instance);
}

/* ------------------------------------------------------------------ launch helpers */

/* The engine's link word, if it passed one (rad_abi.h, `link_gate`). Over P2P the fused kernels
 * end the hold when the peer's push lands, on either wire, and return RAD_LINK_ENDED. Through the
 * host mailbox the reduce itself reads across the link, so those launches return RAD_OK and leave
 * the hold to the engine. */
static long long link_gate(const RadArgs* a) {
    long long g = 0;
    return rad_args_geti(a, "link_gate", &g) ? (long long)g : 0LL;
}

enum { AR_X = 0, AR_Y };

struct ArCall {
    const ArInst* in;
    long long inp, out;
    long long numel;
    int  nblocks, nthreads, drain, acq, pub;
};

/* Block count. PCIe saturates with very few blocks and more only help the local reduce, so the
 * count scales with the message and then stops: 1400 sixteen-byte words per block, floor 4,
 * ceiling max_nb. A tuned `nblocks` overrides it. */
static int nb_for(long long words16, int max_nb) {
    long long nb = words16 / 1400;
    if (nb < 4) nb = 4;
    if (nb > max_nb) nb = max_nb;
    if (nb > words16) nb = words16 > 0 ? words16 : 1;
    return (int)nb;
}

/* The four-rank exact two-shot at decode and verify heights (the one-shot row's band from ts_min,
 * the junction's tail form, the two-shot row up to kTsDecodeShard words a shard), over the words of
 * one shard: one block per 2688 words, floor 2, ceiling 8 (and max_nb). A two-shot takes two
 * handshakes, each a PCIe round trip a block, so a block more costs more than its share of the
 * bytes saves until the shards are large. On 4x R9700, one launch at a time on the 31B's payload,
 * 64 rows of 5376 (8 users x DFlash2's 8 rows) took 85.6 us at 16 blocks against 75.3 at 4, 32
 * rows 47.1 at 8 against 44.2 at 4, and 128 rows 143.5 at 16 against 137.9 at 8. The block count
 * moves no byte: every word is folded by one thread in ascending rank order whatever the grid. */
/* The chained two-shot (r4d_chain.h) where the row resolved with `chain`, for a decode-sized
 * message only (64 rows of an 8192-wide residual): DFlash2's verify-height reductions. */
static long long ar_chain(const RadArgs* a, long long numel) {
    return r4d_geti_or(a, "chain", 0) != 0 && numel <= 64LL * 8192LL ? 1LL : 0LL;
}

static constexpr long long kTsDecodeShard = 43008;   /* 256 rows of 5376 at tp 4 */
static int nb_for_shard(long long shard16, int max_nb) {
    long long nb = shard16 / 2688;
    if (nb < 2) nb = 2;
    if (nb > 8) nb = 8;
    if (nb > max_nb) nb = max_nb;
    if (nb > shard16) nb = shard16 > 0 ? shard16 : 1;
    return (int)nb;
}

/* Shared front half of every all-reduce launch: find the instance, check the message against
 * init's sizing, resolve the tuned point. Every kernel of the family is safe in place (`y` absent
 * or equal to `x`); each entry point says why. */
static int ar_setup(const RadArgs* a, ArCall* c, int shard_nb) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;                  /* init did not run: a core sequencing bug */
    const RadTensor* x = r4d_opt(a, AR_X);
    if (!x) return RAD_E_INVAL;
    if (!r4d_contig(x)) return RAD_E_STRIDE;

    const RadTensor* y = r4d_opt(a, AR_Y);
    if (y) {
        if (!r4d_contig(y)) return RAD_E_STRIDE;
        if (r4d_numel(y) != r4d_numel(x)) return RAD_E_SHAPE;
    }

    const long long n = (long long)r4d_numel(x);
    if (n <= 0) return RAD_E_SHAPE;
    if (n > in->max_numel) return RAD_E_SHAPE;    /* the peer scratch was sized for max_numel */

    /* The push uses 16-byte vector stores and libr4d throws on a partial word; check it here. */
    if (((long long)n * in->esize) % 16) return RAD_E_ALIGN;
    if (!r4d_aligned(x->data, 16)) return RAD_E_ALIGN;
    if (y && !r4d_aligned(y->data, 16)) return RAD_E_ALIGN;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);

    c->in    = in;
    c->inp   = r4d_addr(x);
    c->out   = y ? r4d_addr(y) : r4d_addr(x);
    c->numel = n;
    c->nthreads = cfgv.nthreads;
    c->drain    = cfgv.drain;
    c->acq      = cfgv.acq;
    c->pub      = cfgv.pub;

    const long long words16 = (long long)n * in->esize / 16;
    long long basis = shard_nb ? words16 / in->world : words16;
    if (shard_nb && ar8_weights_for(in, n)) {
        long long off[R4D_MAX_RANKS + 1];
        basis = r4d_ar_shard_table(words16, in->world, in->shard_w, off);
    }
    /* The two-shots move shards, so they use the smaller basis and a tighter ceiling; the
     * full-message rule over-launches across the mid band. */
    int cap = in->nb_cap;
    if (shard_nb && cap > 24) cap = 24;
    /* Every block handshakes with all seven peers at eight ranks; past eight blocks the extra
     * flag rounds cost more than the added reduce bandwidth (8 against 16 blocks measured up to 3 %
     * faster from 2097152 elements on). */
    if (shard_nb && in->world == 8 && cap > 8) cap = 8;
    /* a decode-sized shard takes the decode rule (nb_for_shard), a prompt-sized one nb_for's */
    c->nblocks = cfgv.nblocks > 0 ? cfgv.nblocks
               : (shard_nb && basis <= kTsDecodeShard) ? nb_for_shard(basis, cap) : nb_for(basis, cap);
    if (c->nblocks > in->max_blocks) c->nblocks = in->max_blocks;
    return RAD_OK;
}

/* ------------------------------------------------------------------ all-gather
 *
 * The instance, scratch and flags are the all-reduce's; a slot holds a full input message, so
 * r4d_rad_ar_init sizes it from the same `numel` parameter. The wide form uses the wide family's
 * peer arrays and r4d_ar_handshake (r4d_ar_wide.h), as the wide one-shots do. Its launchers live in
 * r4d_ar_gather_Nrank.hip and are declared here, not in r4d.h, since their only caller is this
 * plugin. */
extern "C" void r4d_ar_gather_4rank(long long peer_scratch0, long long peer_scratch1, long long peer_scratch2,
                                    long long peer_flags0, long long peer_flags1, long long peer_flags2,
                                    long long my_scratch, long long my_flags, long long seq_ctrs,
                                    long long slot_stride16, long long inp, long long out, long long n_elem,
                                    long long dtype, long long rank, long long row_elems, long long stream,
                                    long long nblocks, long long nthreads, long long drain, long long acq, long long pub,
                                    long long gate, long long gate_ctr, long long dpitch, long long spitch, long long chain);
extern "C" void r4d_ar_gather_8rank(long long peer_scratch0, long long peer_scratch1, long long peer_scratch2,
                                    long long peer_scratch3, long long peer_scratch4, long long peer_scratch5,
                                    long long peer_scratch6, long long peer_flags0, long long peer_flags1,
                                    long long peer_flags2, long long peer_flags3, long long peer_flags4,
                                    long long peer_flags5, long long peer_flags6, long long my_scratch,
                                    long long my_flags, long long seq_ctrs, long long slot_stride16, long long inp,
                                    long long out, long long n_elem, long long dtype, long long rank, long long row_elems,
                                    long long stream, long long nblocks, long long nthreads, long long drain, long long acq,
                                    long long pub, long long dpitch, long long spitch);

/* Not ar_setup: an all-gather's output is world_size times its input. With `row`, `y` may also be
 * the leading world_size * row columns of a wider buffer and `x` a row-wide window of one
 * (docs/OPS.md); `*dpitch_out` and `*spitch_out` are then their row pitches in elements, else 0. */
static int ar_gather_wide_setup(const RadArgs* a, int ws, const ArInst** out_in, ArCall* c,
                                long long* row_out, long long* dpitch_out,
                                long long* spitch_out) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;                  /* init did not run: a core sequencing bug */
    if (in->world != ws) return RAD_E_SHAPE;      /* the peer bundle is compiled for this width */

    const RadTensor* x = r4d_opt(a, AR_X);
    const RadTensor* y = r4d_opt(a, AR_Y);
    if (!x || !y) return RAD_E_INVAL;             /* the gather has no in-place form */
    if (y->data == x->data) return RAD_E_INVAL;

    const long long n = (long long)r4d_numel(x);
    if (n <= 0 || n > in->max_numel) return RAD_E_SHAPE;
    /* y is [world_size * numel], rank r's contribution at offset r * numel (docs/OPS.md). */
    if ((long long)r4d_numel(y) != (long long)n * in->world) return RAD_E_SHAPE;

    const long long row = r4d_pgeti_or(a->p, a->n_p, "row", 0);
    if (row < 0 || (row > 0 && (n % row || (row * in->esize) % 16))) return RAD_E_SHAPE;
    /* A row-interleaved gather's [rows][world * row] may sit in the leading columns of a wider
     * buffer, the pitch in whole 16-byte words: rows land at that pitch, the rest of each row is
     * left alone. */
    long long dpitch = 0;
    if (!r4d_contig(y)) {
        const long long p = r4d_row_pitch(y);
        if (row <= 0 || y->rank != 2 || p <= 0 || y->shape[1] != (int64_t)row * in->world ||
            (p * in->esize) % 16)
            return RAD_E_STRIDE;
        dpitch = p;
    }
    /* Likewise x may be [rows][row] read at a wider pitch: a rank's window of a row it computed
     * wider (a split linear's block-aligned superset, arch/common/rad_block_dflash2.h). */
    long long spitch = 0;
    if (!r4d_contig(x)) {
        const long long p = r4d_row_pitch(x);
        if (row <= 0 || x->rank != 2 || p <= 0 || x->shape[1] != (int64_t)row ||
            (p * in->esize) % 16)
            return RAD_E_STRIDE;
        spitch = p;
    }

    if (((long long)n * in->esize) % 16) return RAD_E_ALIGN;
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(y->data, 16)) return RAD_E_ALIGN;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long long words16 = (long long)n * in->esize / 16;
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;

    *out_in = in;
    c->in = in;
    c->inp = r4d_addr(x);
    c->out = r4d_addr(y);
    c->numel = n;
    c->nblocks = nblocks;
    c->nthreads = cfgv.nthreads;
    c->drain = cfgv.drain;
    c->acq = cfgv.acq;
    c->pub = cfgv.pub;
    *row_out = row;
    *dpitch_out = dpitch;
    *spitch_out = spitch;
    return RAD_OK;
}

/* ------------------------------------------------------------------ fault injection
 *
 * Brilliance, fix C1 stage 1b (r4d_ar_seq.h has the modes and the kernels' half): off by default.
 * BRILLIANCE_AR_INJECT=<op>:<mode>:<every>[:<rank>] damages one block's push of every <every>-th
 * launch of <op> on each two-rank host instance of rank <rank> (the writer; 1 by default), counted
 * on the host per instance. <op> is hc (ar_hc_write), ghc (ar_gather_hc_write), exact (the exact
 * all-reduce, also as the wht6 row's small-message fallback) or gather (the two-rank all-gather),
 * or on the rotated wire (tp-wire wht6) wht6 (the wht6 all-reduce), hc6 (ar_hc_write's wht6 form)
 * or ghc6 (ar_gather_hc_write's wht6 form), which have the exchange only: they are injected only
 * with it on (BRILLIANCE_AR_RESEND=0 runs them as they were, with no injection);
 * <mode> is lose, dup, stuck or noflag (ctl-flag), or one of the exchange's own, pair, late, ctl-ack
 * and ctl-rq, which every op takes from stage 4 on. Windows only: elsewhere a damaged push traps
 * (R4D_AR_GIVE_UP).
 * The four-rank host instances (W1, r4d_ar_wide_host.hip) take
 * <op>:<mode>:<every>[:<rank>[:<peer>]]: <op> hc4 (ar_hc_write_4rank), ghc4 (ar_gather_hc_write_4rank),
 * exact4 (ar_oneshot_4rank_exact), gather4 (ar_gather_4rank), ts4a or ts4b (the plain exact
 * two-shot's hop 1 or hop 2, also as the one-shot row's two-shot band), ts4ca or ts4cb (W4: the
 * coded two-shot's hop 1 or hop 2, overlap 1, with the resend on only); <mode> lose, dup, stuck,
 * noflag (ctl-flag), and from W2/W3 the exchange's pair, late, ctl-ack and ctl-rq and the two of
 * width, multi and cross (r4d_ar_wide_x.h; ar_inj_parse below); <rank> the writer, 0-3 (1 by
 * default); <peer> the rank whose push from it is damaged (the next rank by default). Launches are
 * counted per instance.
 * r4d_ar_inject_pin sets the same from a test. Unset, each launch site pays one relaxed load. */
static std::atomic<int> g_inj_state{ -1 };   /* -1 not read yet, 0 off, 1 on */
static std::mutex g_inj_cfg_mu;
struct ArInjCfg { unsigned op = 0, mode = 0; long long every = 0; int rank = 1; int peer = -1;
                  int phase = 0; bool wide = false; int rank2 = -1; };
static ArInjCfg g_inj_cfg;
static std::map<const void*, long long> g_inj_count;   /* launches of the op, per instance */
static unsigned long long g_inj_done = 0;               /* injections asked for, all instances */

/* Parses a spec; false (and *c untouched) when it is not one. */
static bool ar_inj_parse(const char* v, ArInjCfg* c) {
    if (!v) return false;
    char op[16] = {}, mode[16] = {};
    long long every = 0;
    int rank = 1, peer = -1;
    const int n = std::sscanf(v, "%15[^:]:%15[^:]:%lld:%d:%d", op, mode, &every, &rank, &peer);
    if (n < 3 || every < 1) return false;
    ArInjCfg r;
    if (!std::strcmp(op, "exact")) r.op = R4D_AR_OP_EXACT;
    else if (!std::strcmp(op, "hc")) r.op = R4D_AR_OP_HC_WRITE;
    else if (!std::strcmp(op, "ghc")) r.op = R4D_AR_OP_GATHER_HC;
    else if (!std::strcmp(op, "gather")) r.op = R4D_AR_OP_GATHER;
    else if (!std::strcmp(op, "wht6")) r.op = R4D_AR_OP_WHT6;
    else if (!std::strcmp(op, "hc6")) r.op = R4D_AR_OP_HC_WHT6;
    else if (!std::strcmp(op, "ghc6")) r.op = R4D_AR_OP_GHC_WHT6;
    else if (!std::strcmp(op, "hc4")) { r.op = R4D_AR_OP_W_HC; r.wide = true; }
    else if (!std::strcmp(op, "ghc4")) { r.op = R4D_AR_OP_W_GHC; r.wide = true; }
    else if (!std::strcmp(op, "exact4")) { r.op = R4D_AR_OP_W_ONESHOT; r.wide = true; }
    else if (!std::strcmp(op, "gather4")) { r.op = R4D_AR_OP_W_GATHER; r.wide = true; }
    else if (!std::strcmp(op, "ts4a")) { r.op = R4D_AR_OP_W_TWOSHOT; r.wide = true; r.phase = 0; }
    else if (!std::strcmp(op, "ts4b")) { r.op = R4D_AR_OP_W_TWOSHOT; r.wide = true; r.phase = 1; }
    else if (!std::strcmp(op, "ts4ca")) { r.op = R4D_AR_OP_W_CODED; r.wide = true; r.phase = 0; }
    else if (!std::strcmp(op, "ts4cb")) { r.op = R4D_AR_OP_W_CODED; r.wide = true; r.phase = 1; }
    else return false;
    if (!r.wide) {
        if (n > 4 || (n == 4 && (rank < 0 || rank > 1))) return false;
    } else {
        if (n >= 4 && (rank < 0 || rank > 3)) return false;
        if (n == 5 && (peer < 0 || peer > 3 || peer == rank)) return false;
    }
    if (!std::strcmp(mode, "lose")) r.mode = R4D_AR_INJ_LOSE;
    else if (!std::strcmp(mode, "dup")) r.mode = R4D_AR_INJ_DUP;
    else if (!std::strcmp(mode, "stuck")) r.mode = R4D_AR_INJ_STUCK;
    else if (!std::strcmp(mode, "noflag") || !std::strcmp(mode, "ctl-flag")) r.mode = R4D_AR_INJ_NOFLAG;
    else if (!std::strcmp(mode, "pair")) r.mode = R4D_AR_INJ_PAIR;
    else if (!std::strcmp(mode, "late")) r.mode = R4D_AR_INJ_LATE;
    else if (!std::strcmp(mode, "ctl-ack")) r.mode = R4D_AR_INJ_ACK;
    else if (!std::strcmp(mode, "ctl-rq")) r.mode = R4D_AR_INJ_RQ;
    else if (r.wide && !std::strcmp(mode, "multi")) r.mode = R4D_AR_INJ_MULTI;
    else if (r.wide && !std::strcmp(mode, "cross")) r.mode = R4D_AR_INJ_CROSS;
    else return false;
    /* The exchange's own modes (pair, late, ctl-ack, ctl-rq) need the exchange, which every op's
     * host twin has from stage 4 on (ar_hc_write stage 2, ar_gather_hc_write stage 3, the exact
     * all-reduce and the all-gather stage 4). Under BRILLIANCE_AR_RESEND=0 they act on stage 1's
     * kernels as lose (pair, late, stuck) or as nothing (ctl-ack, ctl-rq). */
    /* The wide kernels take every mode from W2/W3 on: the exchange's (pair, late, ctl-ack, ctl-rq)
     * and the two of width, multi (one writer, two peers: <peer> and the next rank after it that is
     * not the writer) and cross (two writers, one reader <peer>: <rank> and the next rank after it
     * that is not <peer>). The two-shot (ts4a, ts4b) has the exchange from W4, its coded form
     * (ts4ca, ts4cb) too, at the codec's granularity (dup and pair act as lose there,
     * r4d_ar_wide_ts.hip). Under BRILLIANCE_AR_RESEND=0 no op has it: there they act as lose (pair,
     * late, multi, cross) or as nothing (ctl-ack, ctl-rq), as the two-rank modes on stage 1's
     * kernels, and the coded form, which has no stop-only kernel, is not injected. */
    r.every = every;
    r.rank = n >= 4 ? rank : 1;
    r.peer = r.wide ? (n == 5 ? peer : (r.rank + 1) % 4) : -1;
    if (r.wide && r.mode == R4D_AR_INJ_CROSS) {
        r.rank2 = (r.rank + 1) % 4;
        if (r.rank2 == r.peer) r.rank2 = (r.rank2 + 1) % 4;
    }
    *c = r;
    return true;
}

/* Sets the switch from `spec` (null: from the environment) under g_inj_cfg_mu; counts restart. */
static int ar_inj_set(const char* spec, bool from_env) {
    std::lock_guard<std::mutex> lk(g_inj_cfg_mu);
    g_inj_count.clear();
    g_inj_done = 0;
    ArInjCfg c;
    bool on = false;
#if defined(_WIN32)
    const char* v = from_env ? std::getenv("BRILLIANCE_AR_INJECT") : spec;
    if (v && *v && std::strcmp(v, "0") && _stricmp(v, "off")) {
        on = ar_inj_parse(v, &c);
        if (!on)
            r4d_log(1, "r4d: BRILLIANCE_AR_INJECT=\"%s\" is not <op>:<mode>:<every>[:<rank>] "
                    "(op hc|ghc|exact|gather|wht6|hc6|ghc6, mode lose|dup|stuck|noflag|ctl-flag|"
                    "pair|late|ctl-ack|ctl-rq, every >= 1, rank 0|1) or <op>:<mode>:<every>[:<rank>"
                    "[:<peer>]] (op hc4|ghc4|exact4|gather4|ts4a|ts4b|ts4ca|ts4cb, mode lose|dup|stuck|noflag|"
                    "ctl-flag|pair|late|ctl-ack|ctl-rq|multi|cross, rank 0-3, peer 0-3 not the "
                    "rank); no injection", v);
        else if (c.wide && c.rank2 >= 0)
            r4d_log(1, "r4d: fault injection ON (BRILLIANCE_AR_INJECT=%s): ranks %d's and %d's "
                    "pushes to rank %d, in the same block, every %lld-th launch of the op on each "
                    "four-rank host instance", v, c.rank, c.rank2, c.peer, c.every);
        else if (c.wide)
            r4d_log(1, "r4d: fault injection ON (BRILLIANCE_AR_INJECT=%s): rank %d's push to "
                    "rank %d, every %lld-th launch of the op on each four-rank host instance",
                    v, c.rank, c.peer, c.every);
        else
            r4d_log(1, "r4d: fault injection ON (BRILLIANCE_AR_INJECT=%s): rank %d's push, "
                    "every %lld-th launch of the op on each two-rank host instance", v, c.rank,
                    c.every);
    }
#else
    (void)spec; (void)from_env;
#endif
    g_inj_cfg = c;
    g_inj_state.store(on ? 1 : 0, std::memory_order_release);
    return on ? 1 : 0;
}

/* The `drain` bits a launch of `op` on `in` carries (r4d_ar_seq.h): 0, or the mode in bits 32-35
 * and the injection's ordinal from bit 36, for every <every>-th launch on a host instance of the
 * chosen rank. */
static long long ar_inj_take(const ArInst* in, unsigned op) {
    int st = g_inj_state.load(std::memory_order_relaxed);
    if (st == 0) return 0;
    if (st < 0) st = ar_inj_set(nullptr, true);
    if (!st || !in || !in->host || in->world != 2) return 0;
    std::lock_guard<std::mutex> lk(g_inj_cfg_mu);
    const ArInjCfg& c = g_inj_cfg;
    if (c.wide || op != c.op || in->rank != c.rank || c.every < 1) return 0;
    if (++g_inj_count[(const void*)in] % c.every) return 0;
    return (long long)((unsigned long long)c.mode << 32 | (unsigned long long)(g_inj_done++) << 36);
}

/* The four-rank launchers' request (r4d_ar_wide_host.hip, inj_pick): 0, or the mode in bits 0-3, the
 * target peer's rank in 4-6, the phase in 7 and the injection's ordinal from bit 8, for every
 * <every>-th launch of `op` on a four-rank host instance of the chosen rank. `twohop`: the op is the
 * two-shot, whose launches the config's phase picks the hop of. */
static long long ar_inj_take_w(const ArInst* in, unsigned op, int twohop) {
    int st = g_inj_state.load(std::memory_order_relaxed);
    if (st == 0) return 0;
    if (st < 0) st = ar_inj_set(nullptr, true);
    if (!st || !in || !in->wide_x) return 0;
    std::lock_guard<std::mutex> lk(g_inj_cfg_mu);
    const ArInjCfg& c = g_inj_cfg;
    if (!c.wide || op != c.op || (in->rank != c.rank && in->rank != c.rank2) || c.every < 1 ||
        c.peer < 0)
        return 0;
    const long long cnt = ++g_inj_count[(const void*)in];
    if (cnt % c.every) return 0;
    const long long phase = twohop ? (long long)c.phase : 0;
    /* cross: both writers must pick the same block, so its ordinal is the instance's own count of
     * injections (the ranks launch in step); every other mode keeps the global count's. */
    const unsigned long long ord = c.rank2 >= 0 ? (unsigned long long)(cnt / c.every - 1) : g_inj_done;
    ++g_inj_done;
    return (long long)c.mode | ((long long)c.peer << 4) | (phase << 7) | (long long)(ord << 8);
}

static long long ar_inj_made() {
    if (g_inj_state.load(std::memory_order_relaxed) <= 0) return 0;
    std::lock_guard<std::mutex> lk(g_inj_cfg_mu);
    return (long long)g_inj_done;
}

/* ------------------------------------------------------------------ the exchange's switch
 *
 * Brilliance, fix C1 stages 2 to 4: the two-rank host twins of ar_hc_write and ar_gather_hc_write
 * (all three forms: the slot table, the inverse table, chained), of the plain exact all-reduce (its
 * plain and chained forms, and as the wht6 row's exact fallback) and of the two-rank all-gather run
 * the exchange (r4d_archk_exchange, r4d_ar_seq.h): a block whose check fails asks the peer to push
 * it again. So do the rotated wire's (tp-wire wht6) host twins: the wht6 all-reduce and the wht6
 * forms of ar_hc_write and ar_gather_hc_write. They had no stage 1 form: with the switch off they
 * run as before the exchange, with no bounded failure (a wrong push block waits for ever).
 * On by default. BRILLIANCE_AR_RESEND=0 (or off) runs stage 1's kernels instead, the same code as
 * before the exchange, which stop loudly on such a block; for a cost comparison in one build. Both
 * ranks are threads of this process, so they always agree. r4d_ar_resend_pin sets it from a test. */
static std::atomic<int> g_resend{ -1 };   /* -1 not read yet, 0 off, 1 on */

static bool ar_resend_on() {
#if !defined(_WIN32)
    return false;   /* as stage 1's bounded failure: elsewhere a wrong push traps (R4D_AR_GIVE_UP) */
#else
    int v = g_resend.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("BRILLIANCE_AR_RESEND");
        v = (e && *e && (!std::strcmp(e, "0") || !std::strcmp(e, "off") || !std::strcmp(e, "OFF"))) ? 0 : 1;
        int want = -1;
        if (g_resend.compare_exchange_strong(want, v) && !v) {
            r4d_log(1, "r4d: BRILLIANCE_AR_RESEND=%s: the two-rank host collectives "
                    "(ar_hc_write, ar_gather_hc_write, the exact all-reduce, the all-gather) and "
                    "the four-rank ones (W2-W4: ar_hc_write_4rank, ar_gather_hc_write_4rank, "
                    "ar_oneshot_4rank_exact, ar_gather_4rank, ar_twoshot_4rank_exact) run without "
                    "the resend (a wrong push block stops the engine, as in stage 1 and W1); the "
                    "four-rank coded two-shot (overlap 1) runs its default kernel, with no stop; on "
                    "tp-wire wht6, the wht6 all-reduce and the wht6 fused writes run as before the "
                    "exchange, with no stop (a wrong push block waits for ever)", e);
        }
        v = g_resend.load(std::memory_order_relaxed);
    }
    return v != 0;
#endif
}

/* Test hook (r4d_selftest): 1 on, 0 off, -1 from the environment again. Only between launches. */
extern "C" void r4d_ar_resend_pin(int on) { g_resend.store(on < 0 ? -1 : (on ? 1 : 0)); }

/* The `drain` bit asking a two-rank launcher for the exchange (R4D_AR_RESEND_BIT): a host twin
 * (the instance has a VRAM mirror) with the switch on; 0 otherwise. The launcher takes it only with
 * `stage` != 0 and strips it. */
static long long ar_resend_bit(const ArInst* in) {
    return in && in->host && in->my_stage && ar_resend_on() ? R4D_AR_RESEND_BIT : 0LL;
}

/* The rotated wire's `drain` bits: the exchange's bit as above and, only with it, an injection of
 * `op` (R4D_AR_OP_WHT6, _HC_WHT6, _GHC_WHT6). Their kernels have no stage 1 form, so an injected
 * launch without the exchange would wait for ever; under BRILLIANCE_AR_RESEND=0 they are not
 * injected, nor counted. */
static long long ar_wht6_bits(const ArInst* in, unsigned op) {
    const long long rs = ar_resend_bit(in);
    return rs ? (rs | ar_inj_take(in, op)) : 0LL;
}

/* Test hook (r4d_selftest): sets the injection as BRILLIANCE_AR_INJECT would ("" or "off" for none,
 * null to read the environment again); the launch counts start over. Returns 1 when on, 0 off. */
extern "C" int r4d_ar_inject_pin(const char* spec) {
    return spec ? ar_inj_set(*spec ? spec : "off", false) : ar_inj_set(nullptr, true);
}

/* Brilliance (r4d_log.h): the collectives' environment switches, read at rad_plugin_open, so their
 * W lines -- a bad BRILLIANCE_MAILBOX_MEM, BRILLIANCE_AR_RESEND=0, a bad BRILLIANCE_AR_INJECT or
 * fault injection on -- are logged at start and counted into the startup report's FINDINGS. Each is
 * read once and kept, as at the first collective that asked before; a value already read (or pinned
 * by a test) stays. */
extern "C" void r4d_ar_env_read(void) {
    (void)mailbox_mem_mode();
    (void)ar_resend_on();
    if (g_inj_state.load(std::memory_order_acquire) < 0) (void)ar_inj_set(nullptr, true);
}

/* Test hook (plugin_test): the three switches above read again at the next ask (the next
 * rad_plugin_open, or the first collective that asks), as in a fresh process. Only while no
 * collective is built. */
extern "C" void r4d_ar_env_unread(void) {
    g_mb_mem.store(-2);
    g_resend.store(-1);
    std::lock_guard<std::mutex> lk(g_inj_cfg_mu);
    g_inj_count.clear();
    g_inj_done = 0;
    g_inj_cfg = ArInjCfg{};
    g_inj_state.store(-1, std::memory_order_release);
}

/* Test hook (r4d_selftest): clears every two-rank host twin's fault record and count, and the
 * exchange's counters, and every four-rank host instance's record and counters (W1), so a test can
 * see the next one. Only between launches: a block giving up at
 * the same time would race it. */
extern "C" void r4d_ar_fault_clear() {
    std::lock_guard<std::mutex> lk(g_cache_mu);
    for (const CacheSlot& cs : g_cache) {
        const ArInst* in = cs.in;
        if (in && in->wide_x && in->my_flags && in->flag_nbw > 0) {
            /* W1: the record, the control line's count and injections, every counter. */
            volatile unsigned int* const f = (volatile unsigned int*)in->my_flags;
            const unsigned int nbw = (unsigned int)in->flag_nbw, ws = (unsigned int)in->world;
            for (unsigned int i = R4D_ARW_SRC_OFF(ws, nbw); i < R4D_ARW_REC_OFF(ws, nbw) + 2u * R4D_ARF_WORDS; ++i)
                f[i] = 0u;
            f[R4D_ARW_CTL_OFF(ws, nbw) + R4D_ARW_CTL_FAULTS] = 0u;
            f[R4D_ARW_CTL_OFF(ws, nbw) + R4D_ARW_CTL_INJEFF] = 0u;
            f[R4D_ARW_CTL_OFF(ws, nbw) + R4D_ARW_CTL_INJEFF2] = 0u;
            continue;
        }
        if (!in || !in->host || in->world != 2 || !in->my_flags || in->flag_nbw <= 0) continue;
        volatile unsigned int* const f = (volatile unsigned int*)in->my_flags;
        const unsigned int nbw = (unsigned int)in->flag_nbw;
        for (unsigned int i = 0; i < 2u * R4D_ARF_WORDS; ++i) f[R4D_AR2_REC_OFF(nbw) + i] = 0u;
        f[R4D_AR2_CTL_OFF(nbw) + R4D_AR2_CTL_FAULTS] = 0u;
        for (unsigned int i = 0; i < 4u * nbw; ++i) f[R4D_AR2_CNT_OFF(nbw) + i] = 0u;   /* stage 2's */
        for (unsigned int i = 0; i < 4u * nbw; ++i) f[R4D_AR2_CNT_GHC_OFF(nbw) + i] = 0u;   /* stage 3's */
        for (unsigned int i = 0; i < 4u * nbw; ++i) f[R4D_AR2_CNT_EXACT_OFF(nbw) + i] = 0u;   /* stage 4's */
        for (unsigned int i = 0; i < 4u * nbw; ++i) f[R4D_AR2_CNT_GATHER_OFF(nbw) + i] = 0u;
        for (unsigned int i = 0; i < 12u * nbw; ++i) f[R4D_AR2_CNT_WHT6_OFF(nbw) + i] = 0u;   /* wht6's three */
    }
}

extern "C" int r4d_rad_ar_gather_2rank(const RadArgs* a, RadStream s) {
    const ArInst* in = nullptr;
    ArCall c;
    long long row = 0, dpitch = 0, spitch = 0;
    int rc = ar_gather_wide_setup(a, 2, &in, &c, &row, &dpitch, &spitch);
    if (rc != RAD_OK) return rc;
    R4D_TRY("ar_gather_2rank",
        r4d_ar_gather_2rank(in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags,
                            in->my_seq, in->slot16, c.inp, c.out, c.numel, in->dtype, in->rank,
                            (long long)row, (long long)(uintptr_t)s, c.nblocks, c.nthreads,
                            c.drain | ar_inj_take(in, R4D_AR_OP_GATHER) | ar_resend_bit(in), c.acq,
                            (long long)dpitch,
                            (long long)spitch, in->my_stage, ar_chain(a, c.numel)));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_gather_3rank(const RadArgs* a, RadStream s) {
    const ArInst* in = nullptr;
    ArCall c;
    long long row = 0, dpitch = 0, spitch = 0;
    int rc = ar_gather_wide_setup(a, 3, &in, &c, &row, &dpitch, &spitch);
    if (rc != RAD_OK) return rc;
    R4D_TRY("ar_gather_3rank",
        r4d_ar_gather_3rank(in->peer_scratch[0], in->peer_scratch[1],
                            in->peer_flags[0], in->peer_flags[1],
                            in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                            c.inp, c.out, c.numel, in->dtype, in->rank, (long long)row,
                            (long long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq, c.pub,
                            (long long)dpitch, (long long)spitch));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_gather_4rank(const RadArgs* a, RadStream s) {
    const ArInst* in = nullptr;
    ArCall c;
    long long row = 0, dpitch = 0, spitch = 0;
    int rc = ar_gather_wide_setup(a, 4, &in, &c, &row, &dpitch, &spitch);
    if (rc != RAD_OK) return rc;
    if (in->wide_x) {   /* W1, the loud stop (r4d_ar_wide_host.hip); plain (never chained here) */
        R4D_TRY("ar_gather_4rank (host)",
            r4d_arw_gather(in->peer_scratch, in->peer_flags, in->my_scratch, in->my_flags,
                           in->my_seq, in->slot16, c.inp, c.out, c.numel, in->dtype, in->rank,
                           (long long)row, (long long)(uintptr_t)s, c.nblocks, c.nthreads,
                           c.drain, c.acq, c.pub, (long long)dpitch, (long long)spitch,
                           in->my_stage, ar_inj_take_w(in, R4D_AR_OP_W_GATHER, 0),
                           ar_resend_on() ? 1LL : 0LL));
        return RAD_OK;
    }
    /* One handshake, placement after it local: the release is the one-shot's. */
    const long long gate = link_gate(a), early = in->host ? 0LL : gate;
    R4D_TRY("ar_gather_4rank",
        r4d_ar_gather_4rank(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                            in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                            in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                            c.inp, c.out, c.numel, in->dtype, in->rank, (long long)row,
                            (long long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq, c.pub,
                            early, early ? in->gate_ctr : 0LL, (long long)dpitch, (long long)spitch,
                            ar_chain(a, c.numel)));
    return early ? RAD_LINK_ENDED : RAD_OK;
}

extern "C" int r4d_rad_ar_gather_8rank(const RadArgs* a, RadStream s) {
    const ArInst* in = nullptr;
    ArCall c;
    long long row = 0, dpitch = 0, spitch = 0;
    int rc = ar_gather_wide_setup(a, 8, &in, &c, &row, &dpitch, &spitch);
    if (rc != RAD_OK) return rc;
    R4D_TRY("ar_gather_8rank",
        r4d_ar_gather_8rank(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                            in->peer_scratch[3], in->peer_scratch[4], in->peer_scratch[5],
                            in->peer_scratch[6],
                            in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                            in->peer_flags[3], in->peer_flags[4], in->peer_flags[5],
                            in->peer_flags[6],
                            in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                            c.inp, c.out, c.numel, in->dtype, in->rank, (long long)row,
                            (long long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq, c.pub,
                            (long long)dpitch, (long long)spitch));
    return RAD_OK;
}

/* ------------------------------------------------------------------ 2-rank one-shot */

extern "C" int r4d_rad_ar_oneshot_2rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 2) return RAD_E_SHAPE;
    R4D_TRY("ar_oneshot_2rank_exact",
        r4d_ar_oneshot_2rank_exact(c.in->peer_scratch[0], c.in->my_scratch,
                                   c.in->peer_flags[0], c.in->my_flags, c.in->my_seq,
                                   c.in->slot16, c.inp, c.out, c.numel, c.in->dtype,
                                   (long long)(uintptr_t)s, c.nblocks, c.nthreads,
                                   c.drain | ar_inj_take(c.in, R4D_AR_OP_EXACT) | ar_resend_bit(c.in),
                                   c.acq, c.in->my_stage, ar_chain(a, c.numel)));
    return RAD_OK;
}

/* Below this size the compressed wire does not pay. A decode all-reduce (eight residual rows,
 * 80 KiB) already runs at ~92% of one-way link bandwidth on the exact kernel, leaving nothing for a
 * 2.4x smaller payload to recover against rotate, pack and unpack. A prefill chunk is 40x larger
 * and bandwidth-bound. So the row is chosen at declare and the wire per launch. */
enum { R4D_AR_WHT6_MIN_BYTES = 128 * 1024 };

/* Block ceiling for the compressed path: it moves three eighths of the bytes, so it needs more
 * blocks to saturate the same link. */
enum { R4D_AR_WHT6_NB_CAP = 48 };

/* The rotated wire's fused writes run 512 threads per block, not 1024. Their waves are
 * latency-bound on the gather's routed rows and the write's streams; at a 2048-row chunk, twice as
 * many resident blocks of half the size is 5-9% faster per call, while 96 or 192 blocks of 1024
 * do not help. */
enum { R4D_AR_WHT6_FUSED_THREADS = 512 };

/* The two-rank fused writes' coded push (r4d_ar_hc_write_coded_kernel: ~12.1 bits a bf16 value
 * instead of 16, the same bytes out) from this message size. Coding costs each block its encode and
 * the peer's decode and saves a quarter of the bytes, so it pays where a message's time is the
 * link's. On 2x R9700 (PCIe 4.0 P2P), rows of 2560 with four streams, N x [producer, reduction]
 * chained on both cards, coded against plain time a reduction, each at its grid below (the gather
 * form in brackets):
 *
 *     20 KiB  1.48 (1.48)    240 KiB  1.01 (1.04)    960 KiB  0.89 (0.75)      5 MiB  0.84 (0.82)
 *     65 KiB  1.14 (1.11)    320 KiB  0.97 (0.99)   1.25 MiB  0.86 (0.71)     10 MiB  0.82 (0.82)
 *    120 KiB  1.11 (1.08)    480 KiB  0.92 (0.95)    2.5 MiB  0.83 (0.84)   11.7 MiB  0.84 (0.84)
 *    160 KiB  1.05 (1.05)    640 KiB  0.89 (0.89)
 *
 * (20 KiB is the register form's step, 3.7 us against 5.5; 10 MiB a 2048-row chunk's, 533 us
 * against 436 and 627 against 513), and the same crossing with the link idle 100 us between
 * reductions, as in a decode step. The forms meet near 280 KiB, so the floor sits at 512 KiB,
 * where coding has begun to pay: a prefill chunk from 103 rows of 2560 is coded, and a decode or
 * verify step of eight streams (48 rows, 240 KiB) is not. A message that does not code at all
 * (every group raw) costs 14% more than plain at 240 KiB, 7% at 640 KiB and 1% at 10 MiB;
 * Qwen3.8-Flash-Next's prefill sends 0.8% (the mixers) and 0.02% (the experts) of its words raw,
 * 12.16 and 12.13 bits a value.
 *
 * `coded` (0 or 1), a launch parameter no declaration carries, pins one launch: r4d_selftest's
 * arms. */
enum { R4D_AR_CODED_MIN_BYTES = 512 * 1024 };

/* The coded form's grid. Its blocks code and decode in waves of 32 words, so it runs 1024 threads
 * a block whatever the row's tuned `nthreads` (at 512 a 10 MiB reduction takes 536 us against 435),
 * and with a quarter of the bytes off the link the memory behind it shows: up to 48 blocks instead
 * of 24 (the gather form at 2048 rows 559 us at 24 blocks and 513 at 48, at 192 rows 53 and 42;
 * 3% slower at 384 to 512 rows, nothing else). Part of the table's 0.75 and 0.71 is that grid: the
 * plain gather at 192 rows takes 59 us at 24 blocks and 49 at 48, where coding alone gives 0.87. */
enum { R4D_AR_CODED_NB_CAP = 48, R4D_AR_CODED_THREADS = 1024 };

static long long ar_coded_min_bytes() { return (long long)R4D_AR_CODED_MIN_BYTES; }

/* Whether a fused write of `words16` 16-byte words takes the coded push: asked for (the floor, or
 * the launch's `coded`) and fitting the form (r4d_ar_coded2_ok; the gather's launcher adds its
 * table's bound). Both ranks decide from the same M and n. */
static long long ar_coded(const RadArgs* a, const ArInst* in, long long words16) {
    if (in->host || !r4d_ar_coded2_ok(words16, in->slot16, in->my_stage)) return 0;
    const long long pin = r4d_geti_or(a, "coded", -1);
    if (pin >= 0) return pin != 0;
    return (long long)words16 * 16 >= ar_coded_min_bytes();
}

extern "C" int r4d_rad_ar_oneshot_2rank_wht6(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 2) return RAD_E_SHAPE;
    if (c.in->dtype == 2) return RAD_E_DTYPE;      /* the rotated wire is bf16/fp16 only */
    if (!c.in->locpack) return RAD_E_STATE;

    int group = 0, bits = 0, chunk = 0;
    r4d_ar_wht6_dims(&group, &bits, &chunk);

    /* Size gate: below the floor, or when the rotation's group does not tile the message, use the
     * exact kernel. Both share this instance's scratch, flags and sequence counters and one flag
     * value space, so mixing them on a stream is protocol-safe. A floor of 0 forces the rotated
     * wire. */
    const long long floor_b = r4d_pgeti_or(a->p, a->n_p, "min_bytes", R4D_AR_WHT6_MIN_BYTES);
    const long long msg_b   = (long long)c.numel * c.in->esize;
    if (group <= 0 || c.numel % group || msg_b < floor_b) {
        R4D_TRY("ar_oneshot_2rank_exact",
            r4d_ar_oneshot_2rank_exact(c.in->peer_scratch[0], c.in->my_scratch,
                                       c.in->peer_flags[0], c.in->my_flags, c.in->my_seq,
                                       c.in->slot16, c.inp, c.out, c.numel, c.in->dtype,
                                       (long long)(uintptr_t)s, c.nblocks, c.nthreads,
                                       c.drain | ar_inj_take(c.in, R4D_AR_OP_EXACT) | ar_resend_bit(c.in),
                                       c.acq, c.in->my_stage, ar_chain(a, c.numel)));
        return RAD_OK;
    }

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    if (cfgv.nblocks <= 0) {
        int nb = nb_for((long long)msg_b / 16, R4D_AR_WHT6_NB_CAP);
        if (nb > c.in->max_blocks) nb = c.in->max_blocks;
        c.nblocks = nb;
    }

    R4D_TRY("ar_oneshot_2rank_wht6",
        r4d_ar_oneshot_2rank_wht6(c.in->peer_scratch[0], c.in->my_scratch,
                                  c.in->peer_flags[0], c.in->my_flags, c.in->my_seq,
                                  c.in->locpack, c.in->slot_bytes, c.in->slot_bytes / 2,
                                  c.inp, c.out, c.numel, c.in->dtype,
                                  (long long)(uintptr_t)s, c.nblocks, c.nthreads,
                                  c.drain | ar_wht6_bits(c.in, R4D_AR_OP_WHT6), c.acq,
                                  c.in->my_stage));
    return RAD_OK;
}

/* ------------------------------------------------------------------ wide (3 / 4 / 8 rank)
 *
 * Peer arguments are in ascending global rank with the current rank removed, the order build()
 * filled peer_scratch/peer_flags in. */

/* The one-shots are safe in place: the thread that stores output word k is the one that read
 * input word k (for its push and its own term of the fold), and peers read only their own scratch
 * (r4d_ar_oneshot_Nrank_exact.hip). */
extern "C" int r4d_rad_ar_oneshot_3rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 3) return RAD_E_SHAPE;
    const ArInst* in = c.in;
    R4D_TRY("ar_oneshot_3rank_exact",
        r4d_ar_oneshot_3rank_exact(in->peer_scratch[0], in->peer_scratch[1],
                                   in->peer_flags[0], in->peer_flags[1],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_oneshot_4rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 4) return RAD_E_SHAPE;
    const ArInst* in = c.in;
    /* From `ts_min` elements, the exact two-shot on this instance: the same fold per element, so
     * the same bytes, for half the bytes a rank (a 16-row verify's message, 516 KiB a card at
     * 3x, is what the link waits on). It shares the wide family's flags, counters and slots, and
     * a shard pair fits any slot that holds the whole message. */
    const long long ts_min = r4d_cfg_ar_ts_min(a);
    const long long words16 = c.numel * in->esize / 16;
    if (ts_min > 0 && c.numel >= ts_min && words16 % in->world == 0 &&
        2 * (c.numel * in->esize) / in->world <= in->slot_bytes &&
        c.numel <= R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) {
        R4dArCfg cfgv;
        r4d_cfg_ar(a, &cfgv);
        int nb = cfgv.nblocks > 0 ? cfgv.nblocks
                                  : nb_for_shard(words16 / in->world,
                                                 in->nb_cap > 24 ? 24 : in->nb_cap);
        if (nb > in->max_blocks) nb = in->max_blocks;
        if (in->wide_x) {   /* W1, the loud stop (r4d_ar_wide_host.hip) */
            R4D_TRY("ar_oneshot_4rank_exact (two-shot, host)",
                r4d_arw_twoshot(in->peer_scratch, in->peer_flags, in->my_scratch, in->my_flags,
                                in->my_seq, in->slot16, c.inp, c.out, c.numel, in->dtype,
                                in->rank, (long long)(uintptr_t)s, nb, c.nthreads, c.drain,
                                c.acq, c.pub, in->my_stage,
                                ar_inj_take_w(in, R4D_AR_OP_W_TWOSHOT, 1),
                                ar_resend_on() ? 1LL : 0LL,
                                nullptr /* the band's decode-sized messages stay even (tp4-link-weight) */));
            return RAD_OK;
        }
        const long long gate = link_gate(a), early = in->host ? 0LL : gate;
        R4D_TRY("ar_oneshot_4rank_exact (two-shot)",
            r4d_ar_twoshot_4rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                       in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                       in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                       c.inp, c.out, c.numel, in->dtype, in->rank,
                                       (long long)(uintptr_t)s, nb, c.nthreads, c.drain, c.acq,
                                       c.pub, early, early ? in->gate_ctr : 0LL,
                                       ar_chain(a, c.numel)));
        return early ? RAD_LINK_ENDED : RAD_OK;
    }
    /* The wide release, the two-rank family's rule: over P2P the kernel ends the engine's hold
     * itself once the peers' pushes have landed (the kernel after the handshake is local VRAM
     * traffic); through the host mailbox the reduce reads the bytes across the link, so early
     * stays 0 and the engine releases by barrier. */
    if (in->wide_x) {   /* W1, the loud stop (r4d_ar_wide_host.hip) */
        R4D_TRY("ar_oneshot_4rank_exact (host)",
            r4d_arw_oneshot(in->peer_scratch, in->peer_flags, in->my_scratch, in->my_flags,
                            in->my_seq, in->slot16, c.inp, c.out, c.numel, in->dtype, in->rank,
                            (long long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                            c.pub, in->my_stage, ar_inj_take_w(in, R4D_AR_OP_W_ONESHOT, 0),
                            ar_resend_on() ? 1LL : 0LL));
        return RAD_OK;
    }
    const long long gate = link_gate(a), early = in->host ? 0LL : gate;
    R4D_TRY("ar_oneshot_4rank_exact",
        r4d_ar_oneshot_4rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                   in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub, early, early ? in->gate_ctr : 0LL));
    return early ? RAD_LINK_ENDED : RAD_OK;
}

extern "C" int r4d_rad_ar_oneshot_8rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 8) return RAD_E_SHAPE;
    const ArInst* in = c.in;
    R4D_TRY("ar_oneshot_8rank_exact",
        r4d_ar_oneshot_8rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                   in->peer_scratch[3], in->peer_scratch[4], in->peer_scratch[5],
                                   in->peer_scratch[6],
                                   in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                   in->peer_flags[3], in->peer_flags[4], in->peer_flags[5],
                                   in->peer_flags[6],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub));
    return RAD_OK;
}

/* The two-shots shard the message, so their block count scales with the shard.
 *
 * In place is safe (r4d_ar_twoshot_Nrank_exact.hip): a thread uses the same word offsets k in all
 * three phases (push reads word k of every peer's shard, reduce overwrites word k of its own,
 * assemble writes word k of every peer's shard), so each input word is overwritten only by the
 * thread that already read it, and peers read only their own scratch. The selector gives the
 * two-shot messages above the one-shot's size bound (r4d_rows.cpp, cArTsExact4). */
/* At three ranks the shards are uneven, so there is no divisibility check: the first
 * (words % 3) shards get one extra word and the slot holds two of the longest. The fold is the
 * one-shot's (fp32, ascending rank), so the band edge between the two rows changes no bits. */
extern "C" int r4d_rad_ar_twoshot_3rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 1);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 3) return RAD_E_SHAPE;
    if (c.numel > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) return RAD_E_SHAPE;
    const long long words16 = c.numel * c.in->esize / 16;
    if (words16 < c.in->world) return RAD_E_SHAPE;
    const long long sh_max = (words16 + c.in->world - 1) / c.in->world;
    if (2 * sh_max * 16 > c.in->slot_bytes) return RAD_E_SCRATCH;
    const ArInst* in = c.in;
    R4D_TRY("ar_twoshot_3rank_exact",
        r4d_ar_twoshot_3rank_exact(in->peer_scratch[0], in->peer_scratch[1],
                                   in->peer_flags[0], in->peer_flags[1],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_twoshot_4rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 1);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 4) return RAD_E_SHAPE;
    if (c.numel > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) return RAD_E_SHAPE;
    /* The message must shard evenly in whole 16-byte words and 2*shard must fit the slot. */
    const long long words16 = c.numel * c.in->esize / 16;
    if (words16 % c.in->world) return RAD_E_SHAPE;
    if (2 * (c.numel * c.in->esize) / c.in->world > c.in->slot_bytes) return RAD_E_SCRATCH;
    const ArInst* in = c.in;
    /* A reduction beside compute (overlap 1) carries its pushes coded at ~12.2 bits a bf16 value
     * (r4d_ar_twoshot_Nrank_exact.hip, the coded form), at 4 blocks of 1024 threads, wherever the
     * message fits that form; the same bytes either way. On 4x R9700 4 x 1024 gave the 31B's
     * prefill the most (+3.9%) against 8 x 512, 16 x 256 and 6 x 1024, whose blocks crowd the GEMM
     * beside them. `coded` 0, a launch parameter no declaration carries, keeps the plain wire:
     * r4d_selftest's plain arm. */
    const bool lane = r4d_geti_or(a, "overlap", 0) == 1;
    if (lane && r4d_geti_or(a, "coded", 1) != 0 &&
        r4d_ar_twoshot_coded_ok(c.numel, in->dtype, 4, in->slot16)) {
        if (in->wide_x && ar_resend_on()) {   /* W4, the resend (r4d_ar_wide_ts.hip); plain, never chained */
            R4D_TRY("ar_twoshot_4rank_exact (coded, host)",
                r4d_arw_twoshot_coded(in->peer_scratch, in->peer_flags, in->my_scratch, in->my_flags,
                                      in->my_seq, in->slot16, c.inp, c.out, c.numel, in->rank,
                                      (long long)(uintptr_t)s, 4, 1024, c.drain, c.acq, c.pub,
                                      in->my_stage, ar_inj_take_w(in, R4D_AR_OP_W_CODED, 1),
                                      ar4_weights_for(in, c.numel, true)));
            return RAD_OK;
        }
        const long long gate = link_gate(a), early = in->host ? 0LL : gate;
        R4D_TRY("ar_twoshot_4rank_exact (coded)",
            r4d_ar_twoshot_4rank_exact_coded(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                             in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                             in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                             c.inp, c.out, c.numel, in->rank, (long long)(uintptr_t)s,
                                             4, 1024, c.drain, c.acq, c.pub, early,
                                             early ? in->gate_ctr : 0LL, ar_chain(a, c.numel)));
        return early ? RAD_LINK_ENDED : RAD_OK;
    }
    /* W1, the loud stop (r4d_ar_wide_host.hip), for the plain form, and from W4 its resend; the
     * coded form above runs W4's resend, or under BRILLIANCE_AR_RESEND=0 its default kernel (no stop,
     * as before W4: W1 never covered it). */
    if (in->wide_x) {
        R4D_TRY("ar_twoshot_4rank_exact (host)",
            r4d_arw_twoshot(in->peer_scratch, in->peer_flags, in->my_scratch, in->my_flags,
                            in->my_seq, in->slot16, c.inp, c.out, c.numel, in->dtype, in->rank,
                            (long long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq, c.pub,
                            in->my_stage, ar_inj_take_w(in, R4D_AR_OP_W_TWOSHOT, 1),
                            ar_resend_on() ? 1LL : 0LL, ar4_weights_for(in, c.numel, false)));
        return RAD_OK;
    }
    /* The kernel ends the hold after its second handshake (its reduce is fused with the phase-2
     * push); through the host mailbox early stays 0 and the engine releases by barrier. */
    const long long gate = link_gate(a), early = in->host ? 0LL : gate;
    R4D_TRY("ar_twoshot_4rank_exact",
        r4d_ar_twoshot_4rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                   in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub, early, early ? in->gate_ctr : 0LL,
                                   ar_chain(a, c.numel)));
    return early ? RAD_LINK_ENDED : RAD_OK;
}

extern "C" int r4d_rad_ar_twoshot_8rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 1);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 8) return RAD_E_SHAPE;
    if (c.numel > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) return RAD_E_SHAPE;
    const ArInst* in = c.in;
    const int* weights = ar8_weights_for(in, c.numel);
    /* Even shards keep the four-rank rule (whole words a rank); weighted ones are uneven anyway. */
    const long long words16 = c.numel * in->esize / 16;
    if (!weights && words16 % in->world) return RAD_E_SHAPE;
    long long off[R4D_MAX_RANKS + 1];
    const long long sh_max = r4d_ar_shard_table(words16, in->world, weights, off);
    if (2 * sh_max * 16 > in->slot_bytes) return RAD_E_SCRATCH;
    /* Pieces only move the handshakes; the regions stay at 0 and sh_max, so scratch is as above. */
    const long long asked = r4d_pgeti_or(a->p, a->n_p, "pieces", in->pieces);
    if (asked < 0 || asked > R4D_AR_TS_MAX_PIECES) return RAD_E_INVAL;
    const long long pieces = ar8_pieces_for(in, c.numel, asked);
    /* Both for measuring a board (r4d_selftest --perfar8), never set by the engine: `timeline` a
     * device buffer of nblocks * R4D_AR_TL_WORDS int64 stamps (r4d.h), `form` a schedule other than
     * lockstep (r4d_ar_shards.h). Lockstep is the serving form: on eight R9700 the ring form takes
     * 2-6% less time for 8 and 16 MiB messages, but at TP8 the split form, which includes the ring
     * order, moves neither prefill nor decode beyond run-to-run noise. */
    const long long timeline = r4d_pgeti_or(a->p, a->n_p, "timeline", 0);
    const long long form = r4d_pgeti_or(a->p, a->n_p, "form", R4D_AR8_FORM_LOCKSTEP);
    if (form < R4D_AR8_FORM_LOCKSTEP || form > R4D_AR8_FORM_SPLIT) return RAD_E_INVAL;
    R4D_TRY("ar_twoshot_8rank_exact",
        r4d_ar_twoshot_8rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                   in->peer_scratch[3], in->peer_scratch[4], in->peer_scratch[5],
                                   in->peer_scratch[6],
                                   in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                   in->peer_flags[3], in->peer_flags[4], in->peer_flags[5],
                                   in->peer_flags[6],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub, weights, (long long)pieces, (long long)timeline, (long long)form));
    return RAD_OK;
}

/* The tiered-int8 wire. Its slot stride is in bytes, not 16-byte words: the wire is 38 bytes per
 * group of 32.
 *
 * It is the only lossy row at three and four ranks, so a lossy request resolves it at every size.
 * As in r4d_rad_ar_oneshot_2rank_wht6, a message under `min_bytes` (or not whole groups) goes to
 * the exact one-shot. The default floor is the two-rank wire's; blocks declare `min_bytes` from
 * --tp-wire-min-kb, and no four-rank crossover has been measured. In place is safe for the exact
 * two-shot's reason: a thread owns the same 8 elements of a shard in all phases and reads them
 * before writing them, and no peer reads this rank's input. */
/* Three ranks run over the uneven two-shot (r4d_ar_twoshot_Nrank_ti8.hip), so the group count
 * need not divide by three. */
extern "C" int r4d_rad_ar_twoshot_3rank_ti8(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 1);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 3) return RAD_E_SHAPE;
    if (c.in->dtype == 2) return RAD_E_DTYPE;

    int group = 0, group_bytes = 0;
    r4d_ar_ti8_dims(&group, &group_bytes);
    if (group <= 0) return RAD_E_DEVICE;
    const long long floor_b = r4d_pgeti_or(a->p, a->n_p, "min_bytes", R4D_AR_WHT6_MIN_BYTES);
    const long long n_groups = c.numel / group;
    if ((long long)c.numel * c.in->esize < floor_b || c.numel % group || n_groups < c.in->world)
        return r4d_rad_ar_oneshot_3rank_exact(a, s);
    /* The slot holds two wire images of the longest shard. */
    const long long sg = (n_groups + c.in->world - 1) / c.in->world;
    const long long wire = (sg * group_bytes + 15) / 16 * 16;
    if (2 * wire > c.in->slot_bytes) return RAD_E_SCRATCH;
    /* nthreads must be a multiple of the encoder's group width; the entry point throws otherwise. */
    if (c.nthreads % 32) return RAD_E_INVAL;

    const ArInst* in = c.in;
    R4D_TRY("ar_twoshot_3rank_ti8",
        r4d_ar_twoshot_3rank_ti8(in->peer_scratch[0], in->peer_scratch[1],
                                 in->peer_flags[0], in->peer_flags[1],
                                 in->my_scratch, in->my_flags, in->my_seq, in->slot_bytes,
                                 c.inp, c.out, c.numel, in->dtype, in->rank,
                                 (long long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                 c.pub));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_twoshot_4rank_ti8(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 1);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 4) return RAD_E_SHAPE;
    if (c.in->dtype == 2) return RAD_E_DTYPE;

    int group = 0, group_bytes = 0;
    r4d_ar_ti8_dims(&group, &group_bytes);
    if (group <= 0) return RAD_E_DEVICE;
    const long long floor_b = r4d_pgeti_or(a->p, a->n_p, "min_bytes", R4D_AR_WHT6_MIN_BYTES);
    if ((long long)c.numel * c.in->esize < floor_b || c.numel % ((long long)group * c.in->world))
        return r4d_rad_ar_oneshot_4rank_exact(a, s);
    /* The slot holds two wire images of one shard, each 16-byte aligned. */
    const long long wire = ((c.numel / c.in->world / group) * group_bytes + 15) / 16 * 16;
    if (2 * wire > c.in->slot_bytes) return RAD_E_SCRATCH;
    if (c.nthreads % 32) return RAD_E_INVAL;

    const ArInst* in = c.in;
    /* Released after the second handshake, as in the exact two-shot. */
    const long long gate = link_gate(a), early = in->host ? 0LL : gate;
    R4D_TRY("ar_twoshot_4rank_ti8",
        r4d_ar_twoshot_4rank_ti8(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                 in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                 in->my_scratch, in->my_flags, in->my_seq, in->slot_bytes,
                                 c.inp, c.out, c.numel, in->dtype, in->rank,
                                 (long long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                 c.pub, early, early ? in->gate_ctr : 0LL));
    return early ? RAD_LINK_ENDED : RAD_OK;
}

/* ------------------------------------------------------------------ the fused all-reduce + norm */

enum { AL_X = 0, AL_RESID, AL_W, AL_Q, AL_SCALE, AL_OUTBF16, AL_WPOST };

extern "C" int r4d_rad_ar_ln_had_quant_i8(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;

    const RadTensor* x  = r4d_opt(a, AL_X);
    const RadTensor* rs = r4d_opt(a, AL_RESID);
    const RadTensor* w  = r4d_opt(a, AL_W);
    const RadTensor* q  = r4d_opt(a, AL_Q);
    const RadTensor* sc = r4d_opt(a, AL_SCALE);
    const RadTensor* ob = r4d_opt(a, AL_OUTBF16);
    if (!x || !w || !q || !sc) return RAD_E_INVAL;
    if (x->rank < 2 || q->rank < 2) return RAD_E_SHAPE;
    /* A workgroup owns whole rows and walks them with vector loads, so nothing may be pitched. */
    if (!r4d_contig(x) || !r4d_contig(q) || !r4d_contig(w) || !r4d_contig(sc))
        return RAD_E_STRIDE;
    if (rs && !r4d_contig(rs)) return RAD_E_STRIDE;
    if (ob && !r4d_contig(ob)) return RAD_E_STRIDE;

    const long long K = x->shape[x->rank - 1];
    long long M = 1;
    for (uint32_t i = 0; i + 1 < x->rank; ++i) M *= x->shape[i];
    if (M < 1 || K < 1) return RAD_E_SHAPE;
    if (M * K > in->max_numel) return RAD_E_SHAPE;

    const long long had  = r4d_geti_or(a, "group", 0);
    const int       wire = (int)r4d_geti_or(a, "wire", 0);
    const float     eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float     wadd = (float)r4d_getf(a, "wadd", 0.0);

    /* Everything the entry point throws about, checked here. */
    if (K % 16) return RAD_E_SHAPE;
    if (had < 2 || (had & (had - 1))) return RAD_E_INVAL;
    if (had != 128 && had != 256 && had != 512) return RAD_E_SHAPE;
    if (K % had) return RAD_E_SHAPE;
    if (wire != 0 && wire != 6) return RAD_E_INVAL;
    if (wire == 6) {
        int stage_bits = 0, chunk_elems = 0, chunk_bytes = 0, scale_bytes = 0;
        r4d_ar_ln_wire_dims(&stage_bits, &chunk_elems, &chunk_bytes, &scale_bytes);
        if (chunk_elems <= 0 || K % chunk_elems) return RAD_E_SHAPE;
    }

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    int nthreads = cfgv.nthreads;
    if (nthreads != 128 && nthreads != 256 && nthreads != 512 && nthreads != 1024)
        return RAD_E_INVAL;
    /* The entry point clamps nblocks to M (a block owns whole rows); clamping here too keeps the
     * shim's count equal to the kernel's. */
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(M * K * 2 / 16, in->max_blocks);
    if (nblocks > (int)M) nblocks = (int)M;
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;   // a tuned count under R4D_AR_MAX_BLOCKS too
    /* LDS floor on the block count. The compressed wire stages every row a block owns, so a count
     * chosen for the push alone can exceed LDS (at K = 5120, more than two rows per block). Raise
     * the count to the kernel's floor; refuse shapes no count within the flag capacity serves.
     * Block count changes no bytes, since every reduction closes over one row. */
    const int nb_min = r4d_ar_ln_blocks_min((int)M, (int)K, (int)had, nthreads, wire);
    if (nb_min < 1 || nb_min > in->max_blocks) return RAD_E_SHAPE;
    if (nblocks < nb_min) nblocks = nb_min;

    R4D_TRY("ar_ln_had_quant_i8",
        r4d_ar_ln_had_quant_i8(in->peer_scratch[0], in->my_scratch, in->peer_flags[0],
                               in->my_flags, in->my_seq, in->slot16,
                               r4d_addr(x), r4d_addr(rs), r4d_addr(w), r4d_addr(ob),
                               r4d_addr(q), r4d_addr(sc),
                               (int)M, (int)K, (int)had, eps, wadd,
                               (long long)(uintptr_t)s, nblocks, nthreads, cfgv.drain, cfgv.acq,
                               wire, in->my_stage));
    return RAD_OK;
}

/* The fp8 counterpart: the int8 shim without the Hadamard rotation.
 *
 * `x` is inout here, unlike the int8 form (which keeps the reduced row in LDS): the reduced row is
 * stored back as the standalone all-reduce does, so other readers of `x` see the unfused pair's
 * bytes. This holds on both wires.
 *
 * `wire` is passed through: 0 for exact bf16, 6 for the rotated 6-bit payload. The architecture
 * chooses it as the standalone `all_reduce` row would (lossy wire allowed, group tiles the message,
 * size above --tp-wire-min-kb), and both ranks decide on the same T. A wire-6 chunk that would
 * straddle a row is refused as a shape. */
/* `hi8`: ar_rmsnorm_had_quant_i8g, the same with had_quant_act_i8g's quantiser on the norm's pass
 * (int8 codes, K whole scale groups of 128 or IQ8R's 256, no post-norm). */
static int rad_ar_rmsnorm_quant(const RadArgs* a, RadStream s, bool hi8) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;

    const RadTensor* x  = r4d_opt(a, AL_X);
    const RadTensor* rs = r4d_opt(a, AL_RESID);
    const RadTensor* w  = r4d_opt(a, AL_W);
    const RadTensor* q  = r4d_opt(a, AL_Q);
    const RadTensor* sc = r4d_opt(a, AL_SCALE);
    const RadTensor* ob = r4d_opt(a, AL_OUTBF16);
    /* ar_rmsnorm_quant_fp8's optional post-norm (Gemma 3's sandwich); ar_ln_had_quant_i8 has no
     * seventh operand, and this function serves only the fp8 op and its rotated int8 twin. */
    const RadTensor* wpo = r4d_opt(a, AL_WPOST);
    if (!x || !w || !q || !sc) return RAD_E_INVAL;
    if (hi8 && q->dtype != RAD_I8) return RAD_E_DTYPE;
    if (hi8 && wpo) return RAD_E_UNSUPPORTED;
    if (!r4d_contig(w)) return RAD_E_STRIDE;
    if (wpo && (!rs || !r4d_contig(wpo) || r4d_numel(wpo) != r4d_numel(w) || wpo->dtype != w->dtype))
        return RAD_E_SHAPE;

    /* Rank 2 everywhere, as in rmsnorm_quant_fp8: a norm has a gain per channel, so flattening
     * everything after dim 0 into the row would put one gain across several heads. */
    if (x->rank != 2 || q->rank != 2 || sc->rank != 2) return RAD_E_SHAPE;
    if (ob && ob->rank != 2) return RAD_E_SHAPE;
    const long long M = x->shape[0], K = x->shape[1];
    if (M < 1 || K < 1) return RAD_E_SHAPE;
    if (M * K > in->max_numel) return RAD_E_SHAPE;
    /* A row must be whole 16B words, since a block pushes its rows as words. */
    if (K % 8) return RAD_E_SHAPE;

    const long long g0 = r4d_quant_act_fp8_group();
    const long long grp = r4d_geti_or(a, "group", g0);
    if (grp != g0 && !(hi8 && grp == R4D_I8_GROUP_WIDE)) return RAD_E_SHAPE;
    const long long nkb = (K + grp - 1) / grp;
    if (hi8 && K % grp) return RAD_E_SHAPE;
    if (r4d_numel(w) != K) return RAD_E_SHAPE;
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (q->shape[0] != M || q->shape[1] != K) return RAD_E_SHAPE;
    if (sc->shape[0] != M || sc->shape[1] != nkb) return RAD_E_SHAPE;
    if (ob && (ob->shape[0] != M || ob->shape[1] != K)) return RAD_E_SHAPE;
    if (rs && (rs->rank != 2 || rs->shape[0] != M || rs->shape[1] != K)) return RAD_E_SHAPE;
    /* x is the message, so it must be tight; other operands may be pitched. */
    const long long xp = r4d_row_pitch(x), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    const long long yp = ob ? r4d_row_pitch(ob) : 0, rp = rs ? r4d_row_pitch(rs) : 0;
    if (xp < 0 || qp < 0 || sp < 0 || yp < 0 || rp < 0) return RAD_E_STRIDE;
    if (xp != K) return RAD_E_STRIDE;

    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);
    /* Gain dtype comes from the operand, as in rmsnorm_quant_fp8's shim. */
    const int w_f32 = w->dtype == RAD_F32;

    const int wire = (int)r4d_geti_or(a, "wire", 0);
    if (wire != 0 && wire != 6) return RAD_E_INVAL;
    if (wire == 6) {
        const int ce = r4d_ar_rmsnorm_quant_fp8_chunk_elems();
        /* Whole 64-element groups a row; the last 512-element chunk may be partial (padded). */
        if (ce <= 0 || K % 64) return RAD_E_SHAPE;
        /* This rank's packed-half side buffer exists for every bf16 instance; missing means a
         * broken init. */
        if (!in->locpack) return RAD_E_STATE;
        /* The padded message must fit: payload (3/4 byte an element) in the peer slot's half and
         * in locpack's, the bf16 scales in the slot's other half, the f32 ones in locpack after
         * half the slot (its allocation: a sixteenth of the slot plus a page). */
        const long long chunks = M * ((K + ce - 1) / ce), groups = chunks * (ce / 64);
        if (chunks * (ce * 6 / 8) > in->slot_bytes / 2 || groups * 2 > in->slot_bytes / 2 ||
            groups * 4 > in->slot_bytes / 16 + 4096)
            return RAD_E_SHAPE;
    }

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    /* The width is the standalone rmsnorm_quant_fp8's, r4d_act_threads(M, n) (1024 up to 32 rows,
     * 512 above): zero means that, resolved from M by the entry point. The sums fold in
     * kR4dSumLanes' partition at every width, so the width changes no bytes; it is not tuned here.
     * The tuned block count, drain and acq change no bytes either and are honoured. */
    const int nthreads = 0;
    /* One block per row, the norm's shape rather than the all-reduce's. The push saturates the
     * fabric at four blocks, but the norm serialises the rows a block owns, which costs more than
     * the saved dispatch; M is 8 to 64 at decode, a block count the fabric handles. nb_cap (24, the
     * push's number) is the wrong ceiling: at M = 64 it would serialise three rows per block.
     *
     * Cap at 256. The wire alone prefers ~64 blocks, but a 2048-row chunk at 64 gives each block 32
     * serial rows, costing several times what the wire gains; 256 is the combined knee and 512 is
     * within noise. At decode, nblocks is simply M.
     *
     * Block count changes no bytes: a block owns whole rows, the wire's 512-element chunk tiles a
     * row, and every reduction closes over one row. A tuned `nblocks` may name a smaller count; the
     * entry point refuses one whose LDS does not fit. The caps bound the flag array. */
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : (int)M;
    if (nblocks > (int)M) nblocks = (int)M;
    constexpr int kNbCap = 256;
    if (nblocks > kNbCap) nblocks = kNbCap;
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
    if (nblocks < 1) nblocks = 1;

    const int e = hi8
        ? r4d_ar_rmsnorm_had_quant_i8g(
              in->peer_scratch[0], in->my_scratch,
              in->peer_flags[0], in->my_flags, in->my_seq,
              in->slot16, r4d_addr(x), r4d_addr(rs), r4d_addr(w), r4d_addr(ob), r4d_addr(q),
              r4d_addr(sc), (int)M, (int)K, eps, wadd, w_f32,
              (int)rp, (int)yp, (int)qp, (int)sp,
              (long long)(uintptr_t)s, nblocks, nthreads, cfgv.drain, cfgv.acq, wire,
              in->locpack, in->slot_bytes / 2, in->my_stage, (int)grp,
              r4d_geti_or(a, "chain", 0) != 0 ? 1 : 0)
        : r4d_ar_rmsnorm_quant_fp8(
              in->peer_scratch[0], in->my_scratch,
              in->peer_flags[0], in->my_flags, in->my_seq,
              in->slot16, r4d_addr(x), r4d_addr(rs), r4d_addr(w), r4d_addr(ob), r4d_addr(q),
              r4d_addr(sc), (int)M, (int)K, eps, wadd, w_f32,
              (int)rp, (int)yp, (int)qp, (int)sp,
              (long long)(uintptr_t)s, nblocks, nthreads, cfgv.drain, cfgv.acq, wire,
              in->locpack, in->slot_bytes / 2, wpo ? r4d_addr(wpo) : 0, in->my_stage);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_ar_rmsnorm_quant_fp8(const RadArgs* a, RadStream s) {
    return rad_ar_rmsnorm_quant(a, s, false);
}

extern "C" int r4d_rad_ar_rmsnorm_had_quant_i8g(const RadArgs* a, RadStream s) {
    return rad_ar_rmsnorm_quant(a, s, true);
}

/* ar_rmsnorm: ar_rmsnorm_quant_fp8 without the quantiser, for a block whose linears read the bf16
 * norm output (arch/common's bf16 blocks). Its operands are that op's minus `q` and `scale`, with
 * `y` required; the exact wire only. The checks, the block count and the norm's width are
 * rad_ar_rmsnorm_quant's, for the same reasons. */
enum { AN_X = 0, AN_RESID, AN_W, AN_Y };

extern "C" int r4d_rad_ar_rmsnorm(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;

    const RadTensor* x  = r4d_opt(a, AN_X);
    const RadTensor* rs = r4d_opt(a, AN_RESID);
    const RadTensor* w  = r4d_opt(a, AN_W);
    const RadTensor* y  = r4d_opt(a, AN_Y);
    if (!x || !w || !y) return RAD_E_INVAL;
    if (!r4d_contig(w)) return RAD_E_STRIDE;
    if (x->rank != 2 || y->rank != 2) return RAD_E_SHAPE;
    const long long M = x->shape[0], K = x->shape[1];
    if (M < 1 || K < 1) return RAD_E_SHAPE;
    if (M * K > in->max_numel) return RAD_E_SHAPE;
    if (K % 8) return RAD_E_SHAPE;
    if (r4d_numel(w) != K) return RAD_E_SHAPE;
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (y->shape[0] != M || y->shape[1] != K) return RAD_E_SHAPE;
    if (rs && (rs->rank != 2 || rs->shape[0] != M || rs->shape[1] != K)) return RAD_E_SHAPE;
    const long long xp = r4d_row_pitch(x), yp = r4d_row_pitch(y);
    const long long rp = rs ? r4d_row_pitch(rs) : 0;
    if (xp < 0 || yp < 0 || rp < 0) return RAD_E_STRIDE;
    if (xp != K) return RAD_E_STRIDE;
    if (r4d_geti_or(a, "wire", 0) != 0) return RAD_E_UNSUPPORTED;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : (int)M;
    if (nblocks > (int)M) nblocks = (int)M;
    constexpr int kNbCap = 256;
    if (nblocks > kNbCap) nblocks = kNbCap;
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
    if (nblocks < 1) nblocks = 1;

    const int e = r4d_ar_rmsnorm_bf16(
        in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags, in->my_seq,
        in->slot16, r4d_addr(x), rs ? r4d_addr(rs) : 0, r4d_addr(w), r4d_addr(y), (int)M, (int)K,
        (float)r4d_getf(a, "eps", 1e-6), (float)r4d_getf(a, "wadd", 0.0), w->dtype == RAD_F32,
        (int)rp, (int)yp, (long long)(uintptr_t)s, nblocks, 0, cfgv.drain, cfgv.acq, in->my_stage,
        r4d_geti_or(a, "chain", 0) != 0 ? 1 : 0);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

/* The all-reduce and the sandwich-norm junction after it, int8 out: exact `all_reduce` of `x`, then
 * `rmsnorm_quant_i8` (post-norm of the reduced x, residual add, scalar, norm and the per-row int8
 * quantiser the w4a8 skinny GEMM reads). Operands are rmsnorm_quant_i8's with `x` inout, since the
 * reduced sum lands back in it. Exact wire only; a lossy deployment keeps the standalone
 * all-reduce. Width and block count as in ar_rmsnorm_quant_fp8. */
enum { AQ8_X = 0, AQ8_RES, AQ8_W, AQ8_Q, AQ8_S, AQ8_Y, AQ8_WP, AQ8_SC };

/* The operands and their rules, the same at every width. `grp`: quant_act_i8g's grid
 * (ar_rmsnorm_quant_i8g), a scale per (row, 128 columns) and K in whole groups. */
struct ArQ8Args {
    const RadTensor *x = nullptr, *rs = nullptr, *w = nullptr, *q = nullptr, *sc = nullptr;
    const RadTensor *ob = nullptr, *wp = nullptr, *sl = nullptr;
    long long M = 0, K = 0, yp = 0, rp = 0;
};

static int ar_q8_args(const RadArgs* a, const ArInst* in, ArQ8Args* o, bool grp = false) {
    const RadTensor* x  = o->x  = r4d_opt(a, AQ8_X);
    const RadTensor* rs = o->rs = r4d_opt(a, AQ8_RES);
    const RadTensor* w  = o->w  = r4d_opt(a, AQ8_W);
    const RadTensor* q  = o->q  = r4d_opt(a, AQ8_Q);
    const RadTensor* sc = o->sc = r4d_opt(a, AQ8_S);
    const RadTensor* ob = o->ob = r4d_opt(a, AQ8_Y);
    const RadTensor* wp = o->wp = r4d_opt(a, AQ8_WP);
    const RadTensor* sl = o->sl = r4d_opt(a, AQ8_SC);
    if (!x || !w || !q || !sc) return RAD_E_INVAL;
    if ((wp || sl) && !rs) return RAD_E_INVAL;
    if (!r4d_contig(w) || (wp && !r4d_contig(wp))) return RAD_E_STRIDE;
    if (x->dtype != RAD_BF16 || q->dtype != RAD_I8 || sc->dtype != RAD_F32) return RAD_E_DTYPE;
    long long M = 0, K = 0;
    if (r4d_rows_of(x, &M, &K) != RAD_OK) return RAD_E_SHAPE;
    if (M * K > in->max_numel) return RAD_E_SHAPE;
    if (K % 16 || (grp && K % 128)) return RAD_E_SHAPE;
    if (r4d_numel(w) != K || (w->dtype != RAD_F32 && w->dtype != RAD_BF16)) return RAD_E_DTYPE;
    if (wp && (r4d_numel(wp) != K || wp->dtype != w->dtype)) return RAD_E_SHAPE;
    if (sl && (r4d_numel(sl) != 1 || (sl->dtype != RAD_F32 && sl->dtype != RAD_BF16))) return RAD_E_SHAPE;
    if (r4d_numel(q) != M * K || !r4d_contig(q)) return RAD_E_SHAPE;
    if (r4d_numel(sc) != M * (grp ? K / 128 : 1) || !r4d_contig(sc)) return RAD_E_SHAPE;
    if (ob && r4d_numel(ob) != M * K) return RAD_E_SHAPE;
    if (rs && r4d_numel(rs) != M * K) return RAD_E_SHAPE;
    const long long xp = r4d_row_pitch(x), yp = ob ? r4d_row_pitch(ob) : 0, rp = rs ? r4d_row_pitch(rs) : 0;
    if (xp != K || yp < 0 || rp < 0) return RAD_E_STRIDE;   /* x is the message: tight */
    if (r4d_geti_or(a, "wire", 0) != 0) return RAD_E_INVAL;
    o->M = M; o->K = K; o->yp = yp; o->rp = rp;
    return RAD_OK;
}

static int rad_ar_rmsnorm_quant_i8(const RadArgs* a, RadStream s, bool grp) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;
    ArQ8Args v;
    const int arc = ar_q8_args(a, in, &v, grp);
    if (arc != RAD_OK) return arc;
    const long long M = v.M, K = v.K;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : (int)M;
    if (nblocks > (int)M) nblocks = (int)M;
    if (nblocks > 256) nblocks = 256;
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
    if (nblocks < 1) nblocks = 1;
    const int e = r4d_ar_rmsnorm_quant_i8(
        in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags, in->my_seq,
        in->slot16, r4d_addr(v.x), r4d_addr(v.rs), r4d_addr(v.w), r4d_addr(v.wp), r4d_addr(v.sl),
        v.sl ? (v.sl->dtype == RAD_F32) : 0, r4d_addr(v.ob), r4d_addr(v.q), r4d_addr(v.sc), (int)M,
        (int)K, (float)r4d_getf(a, "eps", 1e-6), (float)r4d_getf(a, "wadd", 0.0),
        v.w->dtype == RAD_F32, (int)v.rp, (int)v.yp, (long long)(uintptr_t)s, nblocks, cfgv.drain,
        cfgv.acq, grp ? 1 : 0, in->my_stage, r4d_geti_or(a, "chain", 0) != 0 ? 1 : 0);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_ar_rmsnorm_quant_i8(const RadArgs* a, RadStream s) {
    return rad_ar_rmsnorm_quant_i8(a, s, false);
}

/* The same at quant_act_i8g's grid (rmsnorm_quant_i8g's junction): two ranks only. A four-rank
 * deployment declares it, finds no row, and issues the all-reduce apart. */
extern "C" int r4d_rad_ar_rmsnorm_quant_i8g(const RadArgs* a, RadStream s) {
    return rad_ar_rmsnorm_quant_i8(a, s, true);
}

/* The same at four ranks, on the wide family's one-shot transport (r4d_rmsnorm_quant_i8.hip's
 * ar_rmsnorm_quant_i8_wide_kernel), the one ar_hc_write_4rank rides, with the junction as its
 * epilogue. The operands and the norm's width are the two-rank form's. The block count is one per
 * row, capped by the wide family's flag array in the launcher and by the instance here. The
 * instance is the four-rank family's {rank, 4, bf16, M_hi * n} (ar_hc_wide_init), so it is a
 * standalone all_reduce band's where the two bounds agree, which is safe on one stream for the
 * reason that init gives. The plugin issues it only at one-shot heights: each rank sends three
 * times the message, which the two-shot the standalone takes above them does not. */
extern "C" int r4d_rad_ar_rmsnorm_quant_i8_4rank(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 4) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;
    ArQ8Args v;
    const int arc = ar_q8_args(a, in, &v);
    if (arc != RAD_OK) return arc;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : (int)v.M;
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
    /* Above one-shot heights (the plugin's fold band at 5 rows and up), the `tail` axis's form:
     * the standalone two-shot over words on the blocks the standalone all_reduce would take for
     * this message (nb_for_shard), with the junction as its tail; its two counters are words of the
     * instance's zeroed sequence allocation past the landed counter (r4d_ar_seq.h). P2P only. */
    int tail_nbt = 0;
    if (r4d_cfg_ar_tail(a) && v.M >= 5 && !in->host) {
        const long long words16 = (long long)v.M * v.K / 8;
        tail_nbt = nb_for_shard(words16 / in->world, in->nb_cap > 24 ? 24 : in->nb_cap);
        if (tail_nbt > in->max_blocks) tail_nbt = in->max_blocks;
    }
    /* Eight words past the landed counter, at the allocation's bound (gate_ctr is at 2*maxb words):
     * in->max_blocks may be capped below it by R4D_AR_MAX_BLOCKS and would land in the parity counters. */
    const long long tail_ctr = in->gate_ctr + 8 * 4;
    /* `sp` (0 unless asked): the all-reduce split around the junction, P2P only. The host mailbox
     * runs the whole form, which leaves every byte the split one does. */
    const long long sp = r4d_geti_or(a, "sp", 0);
    if (sp < 0 || sp > 2) return RAD_E_INVAL;
    /* The junction after the last handshake is local, so either form releases there. */
    const long long gate = link_gate(a), early = in->host ? 0LL : gate;
    const int e = r4d_ar_rmsnorm_quant_i8_4rank(
        in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
        in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
        in->my_scratch, in->my_flags, in->my_seq, in->slot16,
        r4d_addr(v.x), r4d_addr(v.rs), r4d_addr(v.w), r4d_addr(v.wp), r4d_addr(v.sl),
        v.sl ? (v.sl->dtype == RAD_F32) : 0, r4d_addr(v.ob), r4d_addr(v.q), r4d_addr(v.sc),
        (int)v.M, (int)v.K, (float)r4d_getf(a, "eps", 1e-6), (float)r4d_getf(a, "wadd", 0.0),
        v.w->dtype == RAD_F32, (int)v.rp, (int)v.yp, (int)in->rank, (long long)(uintptr_t)s, nblocks,
        cfgv.drain, cfgv.acq, cfgv.pub, early, early ? in->gate_ctr : 0LL, tail_ctr, tail_nbt,
        r4d_geti_or(a, "chain", 0) != 0 ? 1 : 0, in->host ? 0 : (int)sp);
    if (e != 0) return RAD_E_SHAPE;
    return early ? RAD_LINK_ENDED : RAD_OK;
}

/* ar_rmsnorm_ihc_junction (WP16.2): the exact all-reduce of `a` [after the add] and Hy4-flash's iHC
 * junction over the reduced sum (r4d_ar_ihc_junction.hip). The operands are rmsnorm_ihc_junction's,
 * checked by the same function; `a` is the message, so it must be tight. A block owns whole rows:
 * one a row up to 256 blocks, as ar_rmsnorm_quant_i8's two-rank form. The exact wire's push is
 * coded (r4d_ar_arc2.h, lossless) from the fused writes' floor on P2P instances, rows of whole 256
 * values; the norms read the sum, so no lossy wire. The chained form at decode heights only. */
static int ar_ihc_junction_common(const RadArgs* a, const ArInst* in, r4d_ihc::R4dIhcJunction* j,
                                  R4dArCfg* cfgv, int* nblocks) {
    if (in->dtype != 0) return RAD_E_DTYPE;
    const int rc = r4d_ihc_junction_args(a, j);
    if (rc != RAD_OK) return rc;
    if (j->hc != 2 && j->hc != 4 && j->hc != 8) return RAD_E_UNSUPPORTED;
    if (j->ap != (long long)j->r.n) return RAD_E_STRIDE;
    if (j->M * (long long)j->r.n > in->max_numel) return RAD_E_SHAPE;
    r4d_cfg_ar(a, cfgv);
    int nb = cfgv->nblocks > 0 ? cfgv->nblocks : (int)j->M;
    if (nb > (int)j->M) nb = (int)j->M;
    if (nb > 256) nb = 256;
    if (nb > in->max_blocks) nb = in->max_blocks;
    *nblocks = nb < 1 ? 1 : nb;
    return RAD_OK;
}

extern "C" int r4d_rad_ar_rmsnorm_ihc_junction(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;
    r4d_ihc::R4dIhcJunction j;
    R4dArCfg cfgv;
    int nblocks = 1;
    const int rc = ar_ihc_junction_common(a, in, &j, &cfgv, &nblocks);
    if (rc != RAD_OK) return rc;
    const long long words16 = (long long)(j.M * (long long)j.r.n / 8);
    const long long coded = j.r.n % 256 == 0 ? ar_coded(a, in, words16) : 0;
    const int chain = r4d_geti_or(a, "chain", 0) != 0 && j.M <= 64 ? 1 : 0;
    /* K-13: at decode heights several blocks a row, each owning a run of the 16 waves (the launcher
     * splits the count into rows and parts). A caller's own block count is kept, and the launch stays
     * within the instance's cap; both ranks derive the same split from the same M and caps. */
    int parts = 0;
    if (cfgv.nblocks <= 0 && in->jparts && j.M >= 1 && j.M <= (long long)r4d_ihc::kSplitRows) {
        const int cap = in->nb_cap < in->max_blocks ? in->nb_cap : in->max_blocks;
        for (int b = (int)r4d_ihc::kSplitParts; b > 1; b >>= 1)
            if (j.M * b <= (long long)cap) { parts = b; break; }
        if (parts > 1) nblocks = (int)(j.M * (long long)parts);
    }
    const int e = r4d_ar_ihc_junction_2rank(in->peer_scratch[0], in->my_scratch, in->peer_flags[0],
                                            in->my_flags, in->my_seq, in->slot16, in->jparts, &j,
                                            (long long)(uintptr_t)s, nblocks, cfgv.drain, cfgv.acq,
                                            in->my_stage, chain, coded ? 1 : 0);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

/* Four ranks, on the wide one-shot (r4d_rad_ar_hc_wide_init's instance, shared with
 * ar_rmsnorm_quant_i8_4rank's key rule). One-shot only: a caller issues it at decode heights and
 * the unfused pair above them, whose all_reduce goes two-shot. */
extern "C" int r4d_rad_ar_rmsnorm_ihc_junction_4rank(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 4) return RAD_E_SHAPE;
    r4d_ihc::R4dIhcJunction j;
    R4dArCfg cfgv;
    int nblocks = 1;
    const int rc = ar_ihc_junction_common(a, in, &j, &cfgv, &nblocks);
    if (rc != RAD_OK) return rc;
    const int chain = r4d_geti_or(a, "chain", 0) != 0 && j.M <= 64 ? 1 : 0;
    const int e = r4d_ar_ihc_junction_4rank(
        in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2], in->peer_flags[0],
        in->peer_flags[1], in->peer_flags[2], in->my_scratch, in->my_flags, in->my_seq,
        in->slot16, &j, (int)in->rank, (long long)(uintptr_t)s, nblocks, cfgv.drain, cfgv.acq,
        cfgv.pub, chain);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

/* The all-reduce with the gated residual write: `all_reduce` of `y`, then `hc_write` of the reduced
 * `y` into `h`, in one kernel. A block whose output feeds a hyper-connection write skips its own
 * collective when this runs.
 *
 * `y` is the message, pushed as 16B words, so it must be tight. The streams may be pitched; n and
 * the pitch being multiples of eight keep every write whole 16B words.
 *
 * `wire` (0 exact bf16, 6 rotated 6-bit) belongs to the op, not the message: the architecture
 * declares one op per wire and issues the one ar_wire_is_exact picks for the step, decided on both
 * ranks from the same T. A wire-6 op is 6 at every size, with no floor. */
enum { AH_Y = 0, AH_INJ, AH_H };

/* ---- ar_gather_hc_write: moe_gather, then ar_hc_write, in one kernel. Operands:
 *   [0] y_expert  [1] expert_w  [2] sorted_tok  [3] shared?  [4] shared_gate?  [5] y  [6] inj  [7] h
 * The first five are moe_gather's and the last three ar_hc_write's. `wire` is as in ar_hc_write;
 * wire 6 builds its slot table in LDS at every size and needs no scratch. */
enum { AG_YE = 0, AG_EW, AG_SORTED, AG_SH, AG_SG, AG_Y, AG_INJ, AG_H };

/* ar_gather_hc_write's operand and shape rules at every width: moe_gather's five operands,
 * ar_hc_write's three, and the inverse table's scratch. */
struct GatherHcWriteArgs {
    const RadTensor *ye = nullptr, *ew = nullptr, *so = nullptr, *sh = nullptr, *sg = nullptr;
    const RadTensor *y = nullptr, *inj = nullptr, *h = nullptr;
    long long M = 0, n = 0, hc = 0, top_k = 0, irow = 0, hrow = 0;
    long long ye_ld = 0, ew_ld = 0, sh_ld = 0, sg_ld = 0;
    int sig = 0, wire = 0;
    bool has_inv = false;
};

static int ar_gather_hc_write_args(const RadArgs* a, const ArInst* in, GatherHcWriteArgs* o) {
    const RadTensor* ye  = r4d_opt(a, AG_YE);
    const RadTensor* ew  = r4d_opt(a, AG_EW);
    const RadTensor* so  = r4d_opt(a, AG_SORTED);
    const RadTensor* sh  = r4d_opt(a, AG_SH);
    const RadTensor* sg  = r4d_opt(a, AG_SG);
    const RadTensor* y   = r4d_opt(a, AG_Y);
    const RadTensor* inj = r4d_opt(a, AG_INJ);
    const RadTensor* h   = r4d_opt(a, AG_H);
    if (!ye || !ew || !so || !y || !inj || !h) return RAD_E_INVAL;
    if ((sh != nullptr) != (sg != nullptr)) return RAD_E_INVAL;
    if (y->rank != 2 || inj->rank != 2 || h->rank != 2) return RAD_E_SHAPE;
    const long long M = y->shape[0], n = y->shape[1];
    const long long hc = r4d_geti_or(a, "hc", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0);
    if (M < 1 || n < 1 || hc < 1 || hc > 8 || top_k < 1) return RAD_E_SHAPE;
    if (M * n > in->max_numel) return RAD_E_SHAPE;
    if (n % 8) return RAD_E_SHAPE;
    if (inj->shape[0] < M || inj->shape[1] != hc) return RAD_E_SHAPE;
    if (h->shape[0] < M || h->shape[1] != hc * n) return RAD_E_SHAPE;
    if (!r4d_contig(y)) return RAD_E_STRIDE;
    const long long irow = r4d_row_pitch(inj), hrow = r4d_row_pitch(h);
    if (irow < 0 || hrow < 0) return RAD_E_STRIDE;
    if (hrow % 8) return RAD_E_ALIGN;
    if (!r4d_aligned(y->data, 16) || !r4d_aligned(h->data, 16)) return RAD_E_ALIGN;

    /* moe_gather's side: its pitches and the shared-arm pair. */
    const long long T = M * top_k;
    long long Te = 0, Mw = 0;
    const long long ye_ld = r4d_rows_pitch(ye, n, &Te);
    const long long ew_ld = r4d_rows_pitch(ew, top_k, &Mw);
    if (ye_ld < 0 || ew_ld < 0 || Te < T || Mw < M) return RAD_E_STRIDE;
    if (so->dtype != RAD_I32 || !r4d_contig(so) || r4d_numel(so) < T) return RAD_E_SHAPE;
    if (ye->dtype != RAD_BF16 || ew->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (ye_ld % 8 || !r4d_aligned(ye->data, 16)) return RAD_E_ALIGN;
    long long sh_ld = 0, sg_ld = 0, shrows = 0, sgrows = 0;
    int sig = 0;
    if (sh) {
        if (sh->dtype != RAD_BF16 || sg->dtype != RAD_BF16) return RAD_E_DTYPE;
        sh_ld = r4d_rows_pitch(sh, n, &shrows);
        sg_ld = r4d_rows_pitch(sg, 1, &sgrows);
        if (sh_ld < 0 || sg_ld < 0 || shrows < M || sgrows < M) return RAD_E_STRIDE;
        const char* const act = r4d_gets(a, "act");
        if (act && !std::strcmp(act, "sigmoid"))  sig = 1;
        else if (act && std::strcmp(act, "none")) return RAD_E_UNSUPPORTED;
    }

    const int wire = (int)r4d_geti_or(a, "wire", 0);
    if (wire != 0 && wire != 6) return RAD_E_INVAL;

    o->ye = ye; o->ew = ew; o->so = so; o->sh = sh; o->sg = sg;
    o->y = y; o->inj = inj; o->h = h;
    o->M = M; o->n = n; o->hc = hc; o->top_k = top_k; o->irow = irow; o->hrow = hrow;
    o->ye_ld = ye_ld; o->ew_ld = ew_ld; o->sh_ld = sh_ld; o->sg_ld = sg_ld;
    o->sig = sig; o->wire = wire;
    o->has_inv = a->scratch && a->scratch_bytes >= T * (long long)sizeof(int);
    return RAD_OK;
}

extern "C" int r4d_rad_ar_gather_hc_write(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;

    GatherHcWriteArgs g;
    const int arc = ar_gather_hc_write_args(a, in, &g);
    if (arc != RAD_OK) return arc;
    const RadTensor* const ye = g.ye;
    const RadTensor* const ew = g.ew;
    const RadTensor* const so = g.so;
    const RadTensor* const sh = g.sh;
    const RadTensor* const sg = g.sg;
    const RadTensor* const y = g.y;
    const RadTensor* const inj = g.inj;
    const RadTensor* const h = g.h;
    const long long M = g.M, n = g.n, hc = g.hc, top_k = g.top_k, irow = g.irow, hrow = g.hrow;
    const long long ye_ld = g.ye_ld, ew_ld = g.ew_ld, sh_ld = g.sh_ld, sg_ld = g.sg_ld;
    const int sig = g.sig, wire = g.wire;
    const bool has_inv = g.has_inv;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long long words16 = (long long)(M * n * 2 / 16);
    if (wire == 6) {
        if (!in->locpack) return RAD_E_STATE;
        int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, R4D_AR_WHT6_NB_CAP);
        if (nblocks > in->max_blocks) nblocks = in->max_blocks;
        const long long gate = link_gate(a), early = in->host ? 0LL : gate;
        const int rc = r4d_ar_hc_write_wht6(
            in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags, in->my_seq,
            in->locpack, in->slot_bytes, in->slot_bytes / 2, r4d_addr(y), r4d_addr(inj),
            r4d_addr(h), (long long)M, (long long)n, (long long)hc, (long long)irow, (long long)hrow, r4d_addr(ye),
            r4d_addr(ew), r4d_addr(so), sh ? r4d_addr(sh) : 0LL, sh ? r4d_addr(sg) : 0LL,
            (long long)ye_ld, (long long)ew_ld, (long long)sh_ld, (long long)sg_ld, (long long)top_k, (long long)sig,
            (long long)(uintptr_t)s, nblocks, R4D_AR_WHT6_FUSED_THREADS,
            cfgv.drain | ar_wht6_bits(in, R4D_AR_OP_GHC_WHT6), cfgv.acq,
            early, early ? in->gate_ctr : 0LL, in->my_stage);
        if (rc != 0) return RAD_E_SHAPE;
        return early ? RAD_LINK_ENDED : RAD_OK;
    }
    const long long coded = ar_coded(a, in, words16);
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks
                                   : nb_for(words16, coded ? R4D_AR_CODED_NB_CAP : in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
#if defined(_WIN32)
    mbgeo_note(in, words16, nblocks, 0LL);
    mbcfg_say("ar_gather_hc_write", in, cfgv.drain);
#endif

    const long long gate = link_gate(a), early = in->host ? 0LL : gate;
    const int rc = r4d_ar_gather_hc_write(
        in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags, in->my_seq,
        in->slot16, r4d_addr(y), r4d_addr(inj), r4d_addr(h), (long long)M, (long long)n, (long long)hc,
        (long long)irow, (long long)hrow, r4d_addr(ye), r4d_addr(ew), r4d_addr(so),
        sh ? r4d_addr(sh) : 0LL, sh ? r4d_addr(sg) : 0LL, (long long)ye_ld, (long long)ew_ld, (long long)sh_ld,
        (long long)sg_ld, (long long)top_k, (long long)sig, (long long)(uintptr_t)s, nblocks,
        coded ? R4D_AR_CODED_THREADS : cfgv.nthreads,
        cfgv.drain | ar_inj_take(in, R4D_AR_OP_GATHER_HC) |
            (in->host && in->my_stage && ar_resend_on() ? R4D_AR_RESEND_BIT : 0LL),
        cfgv.acq,
        has_inv ? (long long)(uintptr_t)a->scratch : 0LL, early,
        early ? in->gate_ctr : 0LL, in->my_stage, (long long)r4d_geti_or(a, "chain", 0), coded);
    if (rc != 0) return RAD_E_SHAPE;
    return early ? RAD_LINK_ENDED : RAD_OK;
}

/* Four ranks, exact wire only: r4d_rad_ar_hc_write_4rank with the gather in front, on an instance
 * from r4d_rad_ar_hc_wide_init. Issued only on steps whose write takes the exact fused form
 * (ar_exact_write_takes, arch/common/rad_fp8.h) and whose gather is a single pass. The block count
 * is the write's rule, widened by the launcher until a block's slot table fits in LDS, or, when
 * that needs more blocks and scratch is present, kept and served from the inverse table. */
extern "C" int r4d_rad_ar_gather_hc_write_4rank(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 4) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;

    GatherHcWriteArgs g;
    const int arc = ar_gather_hc_write_args(a, in, &g);
    if (arc != RAD_OK) return arc;
    if (g.wire != 0) return RAD_E_UNSUPPORTED;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long long words16 = (long long)(g.M * g.n * 2 / 16);
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
    const long long gate = link_gate(a), early = in->host ? 0LL : gate;
    if (in->wide_x) {   /* W1, the loud stop (r4d_ar_wide_host.hip) */
        R4D_TRY("ar_gather_hc_write_4rank (host)",
            r4d_arw_gather_hc_write(
                in->peer_scratch, in->peer_flags, in->my_scratch, in->my_flags, in->my_seq,
                in->slot16, r4d_addr(g.y), r4d_addr(g.inj), r4d_addr(g.h), (long long)g.M,
                (long long)g.n, (long long)g.hc, (long long)g.irow, (long long)g.hrow,
                r4d_addr(g.ye), r4d_addr(g.ew), r4d_addr(g.so), g.sh ? r4d_addr(g.sh) : 0LL,
                g.sh ? r4d_addr(g.sg) : 0LL, (long long)g.ye_ld, (long long)g.ew_ld,
                (long long)g.sh_ld, (long long)g.sg_ld, (long long)g.top_k, (long long)g.sig,
                (long long)in->rank, (long long)(uintptr_t)s, nblocks, cfgv.nthreads, cfgv.drain,
                cfgv.acq, cfgv.pub, g.has_inv ? (long long)(uintptr_t)a->scratch : 0LL,
                in->my_stage, ar_inj_take_w(in, R4D_AR_OP_W_GHC, 0), ar_resend_on() ? 1LL : 0LL));
        return RAD_OK;
    }

    R4D_TRY("ar_gather_hc_write_4rank",
        r4d_ar_gather_hc_write_4rank(
            in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
            in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
            in->my_scratch, in->my_flags, in->my_seq, in->slot16,
            r4d_addr(g.y), r4d_addr(g.inj), r4d_addr(g.h), (long long)g.M, (long long)g.n, (long long)g.hc,
            (long long)g.irow, (long long)g.hrow, r4d_addr(g.ye), r4d_addr(g.ew), r4d_addr(g.so),
            g.sh ? r4d_addr(g.sh) : 0LL, g.sh ? r4d_addr(g.sg) : 0LL, (long long)g.ye_ld,
            (long long)g.ew_ld, (long long)g.sh_ld, (long long)g.sg_ld, (long long)g.top_k, (long long)g.sig,
            (long long)in->rank, (long long)(uintptr_t)s, nblocks, cfgv.nthreads, cfgv.drain, cfgv.acq,
            cfgv.pub, g.has_inv ? (long long)(uintptr_t)a->scratch : 0LL, early,
            early ? in->gate_ctr : 0LL));
    return early ? RAD_LINK_ENDED : RAD_OK;
}

/* Inverse slot table for the exact wire's prefill form: one int per (token, slot) at the band's M.
 * The rotated wire keeps its table in LDS. */
extern "C" int64_t r4d_rad_ar_gather_hc_write_scratch(const RadArgs* a) {
    if (r4d_geti_or(a, "wire", 0) == 6) return 0;
    const long long M = r4d_geti_or(a, "M", 0), k = r4d_geti_or(a, "top_k", 0);
    return M > 0 && k > 0 ? M * k * (int64_t)sizeof(int) : 0;
}

/* ar_hc_write's operand and shape rules at every width. */
struct HcWriteArgs {
    const RadTensor* y = nullptr;
    const RadTensor* inj = nullptr;
    const RadTensor* h = nullptr;
    long long M = 0, n = 0, hc = 0, irow = 0, hrow = 0;
    int wire = 0;
};

static int ar_hc_write_args(const RadArgs* a, const ArInst* in, HcWriteArgs* o) {
    const RadTensor* y   = r4d_opt(a, AH_Y);
    const RadTensor* inj = r4d_opt(a, AH_INJ);
    const RadTensor* h   = r4d_opt(a, AH_H);
    if (!y || !inj || !h) return RAD_E_INVAL;
    if (y->rank != 2 || inj->rank != 2 || h->rank != 2) return RAD_E_SHAPE;
    const long long M = y->shape[0], n = y->shape[1];
    const long long hc = r4d_geti_or(a, "hc", 0);
    if (M < 1 || n < 1 || hc < 1 || hc > 8) return RAD_E_SHAPE;
    if (M * n > in->max_numel) return RAD_E_SHAPE;
    if (n % 8) return RAD_E_SHAPE;
    if (inj->shape[0] < M || inj->shape[1] != hc) return RAD_E_SHAPE;
    if (h->shape[0] < M || h->shape[1] != hc * n) return RAD_E_SHAPE;
    if (!r4d_contig(y)) return RAD_E_STRIDE;
    const long long irow = r4d_row_pitch(inj), hrow = r4d_row_pitch(h);
    if (irow < 0 || hrow < 0) return RAD_E_STRIDE;
    if (hrow % 8) return RAD_E_ALIGN;
    if (!r4d_aligned(y->data, 16) || !r4d_aligned(h->data, 16)) return RAD_E_ALIGN;

    const int wire = (int)r4d_geti_or(a, "wire", 0);
    if (wire != 0 && wire != 6) return RAD_E_INVAL;

    o->y = y; o->inj = inj; o->h = h;
    o->M = M; o->n = n; o->hc = hc; o->irow = irow; o->hrow = hrow;
    o->wire = wire;
    return RAD_OK;
}

extern "C" int r4d_rad_ar_hc_write(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;

    HcWriteArgs w;
    const int arc = ar_hc_write_args(a, in, &w);
    if (arc != RAD_OK) return arc;
    const RadTensor* const y = w.y;
    const RadTensor* const inj = w.inj;
    const RadTensor* const h = w.h;
    const long long M = w.M, n = w.n, hc = w.hc, irow = w.irow, hrow = w.hrow;
    const int wire = w.wire;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long long words16 = (long long)(M * n * 2 / 16);
    if (wire == 6) {
        if (!in->locpack) return RAD_E_STATE;
        int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, R4D_AR_WHT6_NB_CAP);
        if (nblocks > in->max_blocks) nblocks = in->max_blocks;
        const long long gate = link_gate(a), early = in->host ? 0LL : gate;
        const int rc = r4d_ar_hc_write_wht6(
            in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags, in->my_seq,
            in->locpack, in->slot_bytes, in->slot_bytes / 2, r4d_addr(y), r4d_addr(inj),
            r4d_addr(h), (long long)M, (long long)n, (long long)hc, (long long)irow, (long long)hrow, 0LL, 0LL, 0LL, 0LL, 0LL,
            0LL, 0LL, 0LL, 0LL, 0LL, 0LL, (long long)(uintptr_t)s, nblocks, R4D_AR_WHT6_FUSED_THREADS,
            cfgv.drain | ar_wht6_bits(in, R4D_AR_OP_HC_WHT6), cfgv.acq, early,
            early ? in->gate_ctr : 0LL, in->my_stage);
        if (rc != 0) return RAD_E_SHAPE;
        return early ? RAD_LINK_ENDED : RAD_OK;
    }
    const long long coded = ar_coded(a, in, words16);
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks
                                   : nb_for(words16, coded ? R4D_AR_CODED_NB_CAP : in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
#if defined(_WIN32)
    if (mbgeo_on()) {
        /* The stall capture's operands (cap_run): host-side bookkeeping only. */
        MbGeo hca{ words16, nblocks, { 0, 0, 0, 0 }, 0 };
        hca.inj = r4d_addr(inj); hca.h = r4d_addr(h);
        hca.irow = irow; hca.hrow = hrow;
        hca.n = (int)n; hca.hc = (int)hc;
        mbgeo_note(in, words16, nblocks, r4d_addr(y), &hca);
    }
#endif
    /* Bit 0x100: the diagnostic push shadow (r4d_ar_hc_write_kernel), with the stall diagnostics. */
    long long drain_arg = cfgv.drain;
#if defined(_WIN32)
    if (in->host && in->my_stage) {
        if (push_shadow_mode() == 1) drain_arg |= 0x100;
        if (push_shadow_mode() == 2) drain_arg |= 0x200;
    }
    mbcfg_say("ar_hc_write", in, cfgv.drain);
#endif

    const long long gate = link_gate(a), early = in->host ? 0LL : gate;
    R4D_TRY("ar_hc_write",
        r4d_ar_hc_write(in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags,
                        in->my_seq, in->slot16, r4d_addr(y), r4d_addr(inj), r4d_addr(h),
                        (long long)M, (long long)n, (long long)hc, (long long)irow, (long long)hrow, (long long)(uintptr_t)s,
                        nblocks, coded ? R4D_AR_CODED_THREADS : cfgv.nthreads,
                        drain_arg | ar_inj_take(in, R4D_AR_OP_HC_WRITE) |
                            (in->host && in->my_stage && ar_resend_on() ? R4D_AR_RESEND_BIT : 0LL),
                        cfgv.acq, early,
                        early ? in->gate_ctr : 0LL, in->my_stage,
                        (long long)r4d_geti_or(a, "chain", 0), coded));
    return early ? RAD_LINK_ENDED : RAD_OK;
}

/* Four ranks, exact wire only: the wide one-shot with the write as its epilogue. The four-rank
 * lossy wire (the ti8 two-shot) has no fused form, so `wire` 6 is refused. Any message up to the
 * instance's bound is correct here as a one-shot; the architecture issues it on decode and verify
 * steps (ar_wide_write_exact, arch/common/rad_fp8.h), since a message the standalone bands would
 * send as a two-shot costs more traffic here.
 *
 * Block count follows the plain one-shot's rule (nb_for over the whole message under nb_cap);
 * words, flags and slot match the one-shot's, so the two can share an instance. */
extern "C" int r4d_rad_ar_hc_write_4rank(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 4) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;

    HcWriteArgs w;
    const int arc = ar_hc_write_args(a, in, &w);
    if (arc != RAD_OK) return arc;
    if (w.wire != 0) return RAD_E_UNSUPPORTED;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long long words16 = (long long)(w.M * w.n * 2 / 16);
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
    /* The two-rank fused write's release, at width: the reduce and the write behind the
     * handshake are local VRAM traffic, so the kernel ends the hold once the pushes land. */
    const long long gate = link_gate(a), early = in->host ? 0LL : gate;
    if (in->wide_x) {   /* W1, the loud stop (r4d_ar_wide_host.hip) */
        R4D_TRY("ar_hc_write_4rank (host)",
            r4d_arw_hc_write(in->peer_scratch, in->peer_flags, in->my_scratch, in->my_flags,
                             in->my_seq, in->slot16, r4d_addr(w.y), r4d_addr(w.inj),
                             r4d_addr(w.h), (long long)w.M, (long long)w.n, (long long)w.hc,
                             (long long)w.irow, (long long)w.hrow, (long long)in->rank,
                             (long long)(uintptr_t)s, nblocks, cfgv.nthreads, cfgv.drain,
                             cfgv.acq, cfgv.pub, in->my_stage,
                             ar_inj_take_w(in, R4D_AR_OP_W_HC, 0), ar_resend_on() ? 1LL : 0LL));
        return RAD_OK;
    }

    R4D_TRY("ar_hc_write_4rank",
        r4d_ar_hc_write_4rank(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                              in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                              in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                              r4d_addr(w.y), r4d_addr(w.inj), r4d_addr(w.h),
                              (long long)w.M, (long long)w.n, (long long)w.hc, (long long)w.irow, (long long)w.hrow,
                              (long long)in->rank, (long long)(uintptr_t)s, nblocks, cfgv.nthreads,
                              cfgv.drain, cfgv.acq, cfgv.pub, early,
                              early ? in->gate_ctr : 0LL));
    return early ? RAD_LINK_ENDED : RAD_OK;
}
