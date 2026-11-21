/* hipnative.cpp -- Brilliance: the device backend on Windows, on the HIP runtime's own streams.
 *
 * On Linux radiance drives the card through HSA queues it writes itself (core/device/aql.cpp): AQL
 * packets, its own SDMA queue for copy lists, recorded passes, chained launches, and fat-binary
 * registration hooks it exports so a plugin's launches become its packets. Windows' HIP runtime
 * sits on PAL and WDDM and offers none of that -- no ROCr, no AQL queue API, no /dev/kfd -- and a
 * DLL binds its HIP imports to amdhip64 itself, so there is nothing to interpose. This backend
 * completes hip.cpp's table (devices, properties, peer queries) with what HIP itself offers:
 * hipStream_t streams, hipEvent_t events, hipMemcpyAsync, pinned memory from hipHostMalloc, and
 * elastic memory from HIP's virtual memory API.
 *
 * A RadStream HERE IS A REAL hipStream_t, so a plugin's hipLaunchKernel on a RadStream goes
 * straight to amdhip64 on that stream, in order with everything this backend enqueues on it.
 *
 * NAMED "hipnative", NOT "aql". rad_dev_chain_enabled (core/device/dispatch.cpp) turns chained
 * launches on only for a backend named "aql": a chained form reads a record the AQL backend writes
 * into its packets, which a HIP launch never carries. Under any other name every row resolves its
 * plain kernel form.
 *
 * Adapted from Brilliance 0.2.2's core/device/hipnative.cpp, written against radiance 1.10's
 * table, to radiance 2.1's DeviceBackend. Each ROCm-on-Windows workaround is explained where it is
 * applied below. Left out on purpose, with the
 * entries null: recorded passes (HIP graph replay is deferred), the expert
 * prefetch, gates, the DMA lane's list engine and event, the link hold, host-backed mappings and
 * backing swaps. (map_file, null at first, is a file view registered with HIP since 2026-11-17.) dispatch.cpp and the callers above it treat each null as "the
 * backend has none", which is the honest answer here. The stall watch (stuck_ms) reports a
 * device that stopped completing work, and never ends the process (see "the stall watch").
 *
 * Compiles to nothing without RAD_HAVE_HIP, like hip.cpp, and is only in the Windows build
 * (win/cmake/RadWindows.cmake, brl_windows_finalize).
 */
#include "device/device.h"

#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP && defined(_WIN32)

#include "pin_ledger.h"
#include "stall_watch.h"
#include "coll_trace.h"
#include "device/aql.h"     /* the AQL test hooks this backend answers in aql.cpp's place */
#include "rad_format.h"     /* RAD_ALIGN_POOL */
#include "rad_internal.h"
#include "os/os.h"

#include <windows.h>
#include <io.h>         /* _get_osfhandle: map_file takes a CRT descriptor */
#include <winternl.h>
#include <d3dkmthk.h>
#include <dxgi1_4.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <string>
#include <vector>

/* win/device/hip_copy.hip: copies as kernels on the stream's compute queue (copy_kernels below). */
extern "C" int brl_hip_copy_ranges(void* dst, const void* src, uint64_t prefix,
                                   const RadCopyRange* ranges, int n, void* stream);
extern "C" int brl_hip_copy(void* dst, const void* src, uint64_t bytes, void* stream);
extern "C" int brl_hip_copy2d(void* dst, uint64_t dpitch, const void* src, uint64_t spitch,
                              uint64_t width, uint64_t height, void* stream);

/* DXGI for the per-process video-memory budgets, gdi32 for the kernel thunks (D3DKMT*) that count
 * every process's use of a card. Named here so the backend carries its own link needs. */
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "gdi32.lib")

namespace rad {
namespace {

/* ------------------------------------------------------------------ errors
 *
 * HIP KEEPS A PER-THREAD "LAST ERROR" that every failing call sets, and the launchers in the core
 * (sched/advance.hip, runtime/wtab.hip) and in the kernel libraries read it right after a launch
 * (hipGetLastError) to see whether THAT launch failed. On Linux the AQL backend owns
 * hipGetLastError and answers for its own launches only; here the runtime answers, so a failure
 * this backend has already handled -- an event not ready yet, a pointer query on memory HIP never
 * saw, an allocation refused -- must not be left behind for the next launch to report as its own.
 * Every handled failure clears it. */
inline void clear_sticky() { (void)hipGetLastError(); }

/* Maps a hipError_t to a coarse RAD_E_* class, as hip.cpp does; the detail goes to
 * rad_dev_last_error(). */
int hip_fail(hipError_t e, const char* who) {
    clear_sticky();
    device_set_error("%s: %s (%s)", who, hipGetErrorString(e), hipGetErrorName(e));
    switch (e) {
        case hipErrorOutOfMemory:      return RAD_E_NOMEM;
        case hipErrorInvalidValue:
        case hipErrorInvalidDevice:
        case hipErrorInvalidHandle:    return RAD_E_INVAL;
        case hipErrorNotSupported:     return RAD_E_UNSUPPORTED;
        default:                       return RAD_E_DEVICE;
    }
}
#define HIP_TRY(call)                                                            \
    do { hipError_t _e = (call);                                                 \
         if (_e != hipSuccess) return hip_fail(_e, #call); } while (0)
/* For void callers (frees, destroys): a failing teardown is still logged, since it means the
 * pointer was not what the caller thought. */
#define HIP_NOTE(call)                                                           \
    do { hipError_t _e = (call);                                                 \
         if (_e != hipSuccess) {                                                 \
             clear_sticky();                                                     \
             RAD_WARN("device: %s: %s", #call, hipGetErrorString(_e));           \
         } } while (0)

const DeviceBackend* g_base = nullptr;   /* hip.cpp's table */
DeviceBackend        g_nat;

bool env_on(const char* name, bool dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    return !(v[0] == '0' || _stricmp(v, "off") == 0 || _stricmp(v, "false") == 0 ||
             _stricmp(v, "no") == 0);
}

/* ------------------------------------------------------------------ the runtime's environment
 *
 * Set before the first HIP call (hipnative_backend_if_present runs before any), and only where the
 * operator has not set them: an operator's value always wins. Each has its reason beside
 * it below. Written to the CRT's table, which amdhip64 shares (one /MD CRT for
 * every module, RadWindows.cmake), and to the process block, which PAL may read instead.
 *
 * Deliberately NOT set: AMD_DIRECT_DISPATCH (=1 hung the PAL runtime) and GPU_MAX_HW_QUEUES (the
 * runtime's default measured faster on the R9700 than 8). Both are re-measured in M2. */
void set_env_default(const char* name, const char* value) {
    if (std::getenv(name)) return;
    _putenv_s(name, value);
    SetEnvironmentVariableA(name, value);
}

void runtime_defaults() {
    /* PAL caps a single allocation at a fraction of the card, and radiance's pools are single
     * large allocations. The Linux serve scripts set the same. */
    set_env_default("GPU_MAX_ALLOC_PERCENT", "100");
    set_env_default("GPU_SINGLE_ALLOC_PERCENT", "100");
    set_env_default("GPU_MAX_HEAP_SIZE", "100");
    /* rocm-systems#12571: on gfx1201 a copy issued behind hipEventRecord can run before the kernel
     * it follows when it goes to an SDMA engine. Copies on blit kernels instead are correct and
     * slower. On until a probe on the pinned runtime shows radiance's pattern does not trip it;
     * BRILLIANCE_SDMA=1 keeps the copy engines. With them on, the engine's small copies and the step
     * batch's uploads run as kernels on the compute queue (copy_kernels, below), which keeps the
     * drafters exact; the copy engines then carry only the bulk copies (specdec-sdma, 2026-11-20). */
    if (!env_on("BRILLIANCE_SDMA", false)) set_env_default("PAL_DISABLE_SDMA", "1");
    /* PAL keeps what a hipFree gives back in a resource cache of its own (clr palresource.cpp,
     * ResourceCache::addGpuMemory): every freed allocation smaller than an eighth of the card's
     * memory (3.98 GiB on a 32 GiB R9700), up to that eighth in all, for a later allocation of
     * about the same size. The cache is per card and nothing outside PAL can empty it: only an
     * allocation PAL fails to create frees it, and WDDM pages an oversubscribed one instead of
     * failing it. Windows counts the cached bytes as this process's use (DXGI CurrentUsage), so
     * the free figure the VRAM budget plans against (vram_free, below) loses them. An engine
     * that stops or unloads therefore leaves up to 3.98 GiB a card behind it, and the next load
     * on those cards plans that much less: Flash-Next at TP2, reloaded after an engine stop,
     * measured 4.66 GiB already held a card against 1.11 GiB at the first start, its KV pool,
     * activation arena and step staging, and failed with out of memory (pm/9-recover). Upstream's
     * fleet gives a backend's kept memory back before every load (trim_spares, vmem_trim); this
     * backend declares that it keeps none (vmem_trim null), and with the cache off that is
     * true. The engine allocates at a load, a park or restore and an elastic pool's change, never
     * on the step path (the allocation record below), and at TP2 no step measurably used the
     * cache.
     *
     * OFF FOR EVERY START, ONE CARD INCLUDED. It was off at first only with more than one device
     * visible, because one card's 4-stream
     * decode had read 3-13% slower with it off.
     * That cost did not reproduce on a rerun: no
     * hipMalloc or hipFree while one card decodes at 4 streams, only two grow-only pins, so nothing
     * is recycled per step, and over three pairs the cache off ran 3.975 against 4.014 ms per token,
     * within noise. With it off one card also reloads with its VRAM (the KV plan does not shrink)
     * and parks in 0.06-0.33 s instead of waiting out the fleet's 2 s free-figure wait. */
    const char* cache_why = std::getenv("GPU_RESOURCE_CACHE_SIZE") ? "the operator's" : "off: the default";
    set_env_default("GPU_RESOURCE_CACHE_SIZE", "0");
    /* The HIP runtime settings this process starts with, as the runtime will read them: the serve
     * log's ENVIRONMENT section names only RADIANCE_* variables, so a run that sets one of these
     * (a queue count, a flush mode) shows here that it reached the process. */
    std::string seen;
    for (const char* k : { "GPU_MAX_HW_QUEUES", "GPU_FLUSH_ON_EXECUTION", "AMD_OPT_FLUSH", "AMD_SERIALIZE_KERNEL",
                           "AMD_SERIALIZE_COPY", "AMD_DIRECT_DISPATCH", "PAL_DISABLE_SDMA",
                           "GPU_RESOURCE_CACHE_SIZE", "HIP_VISIBLE_DEVICES" }) {
        const char* v = std::getenv(k);
        seen += std::string(" ") + k + "=" + (v ? v : "(unset)");
        if (std::strcmp(k, "GPU_RESOURCE_CACHE_SIZE") == 0)
            seen += std::string(" (") + cache_why + ")";
    }
    RAD_INFO("device: HIP runtime settings in this process:%s", seen.c_str());
}

/* ------------------------------------------------------------------ the allocation record
 *
 * Every allocation this backend makes, by address: a free of the wrong kind is caught by name, a
 * pin is given back to the pin account at its free, and a fill or ranged copy into pageable
 * memory is refused as the AQL backend refuses it. Ordered, so an address inside an allocation
 * finds it (a caller hands over interior pointers: a ring's slot, a pool's piece). Read on the
 * step path under a shared lock, written only by allocations, which are never on it. */
struct Rec { int64_t bytes; int kind; };
std::shared_mutex             g_rec_mu;
std::map<uintptr_t, Rec>      g_recs;

void rec_add(void* p, int64_t bytes, int kind) {
    std::unique_lock<std::shared_mutex> lk(g_rec_mu);
    g_recs[(uintptr_t)p] = Rec{ bytes, kind };
}

bool rec_take(void* p, Rec* out) {
    std::unique_lock<std::shared_mutex> lk(g_rec_mu);
    auto it = g_recs.find((uintptr_t)p);
    if (it == g_recs.end()) return false;
    *out = it->second;
    g_recs.erase(it);
    return true;
}

/* The kind of the allocation holding `p`, or -1 when this backend did not make it (a pageable
 * allocation, an elastic range, memory a kernel library took from HIP itself). */
int rec_kind(const void* p) {
    if (!p) return -1;
    const uintptr_t a = (uintptr_t)p;
    std::shared_lock<std::shared_mutex> lk(g_rec_mu);
    auto it = g_recs.upper_bound(a);
    if (it == g_recs.begin()) return -1;
    --it;
    return a < it->first + (uintptr_t)std::max<int64_t>(it->second.bytes, 1) ? it->second.kind : -1;
}

/* ------------------------------------------------------------------ DXGI and the kernel's counters
 *
 * WDDM DOES NOT REFUSE AN ALLOCATION PAST THE CARD: it pages the excess to system memory, and every
 * kernel that touches it then runs at link speed with no error anywhere. So the free figure the
 * planner budgets against, and the room an allocation is checked for, is the smaller of HIP's and
 * DXGI's: the video-memory budget the OS gives this process on this adapter, less what the process
 * already uses. The adapter is found by the LUID HIP reports.
 *
 * FINDING THE ADAPTER ENTERS HIP (hipGetDeviceProperties, for the LUID); READING IT DOES NOT. Only a
 * caller already in the runtime -- props at start-up, an allocation's room check -- finds it
 * (`open`); vram_used, which a dashboard polls beside a running step, only reads what those found.
 * A failure to find it is not remembered: the next caller that may open tries again. */
constexpr int kMaxDev = 64;
std::mutex                  g_dx_mu;             /* serialises finding, never reading */
std::atomic<IDXGIAdapter3*> g_dx[kMaxDev];
std::atomic<uint64_t>       g_luid[kMaxDev];     /* the adapter's LUID; 0 until found */

uint64_t luid_bits(const LUID& l) { return (uint64_t)l.LowPart | ((uint64_t)(uint32_t)l.HighPart << 32); }

IDXGIAdapter3* adapter_for(int device, bool open) {
    if (device < 0 || device >= kMaxDev) return nullptr;
    IDXGIAdapter3* a = g_dx[device].load(std::memory_order_acquire);
    if (a || !open) return a;
    std::lock_guard<std::mutex> lk(g_dx_mu);
    a = g_dx[device].load(std::memory_order_acquire);
    if (a) return a;
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, device) != hipSuccess) {
        clear_sticky();
        return nullptr;
    }
    LUID luid;
    static_assert(sizeof luid == sizeof p.luid, "HIP's LUID is Windows' LUID");
    std::memcpy(&luid, p.luid, sizeof luid);
    if (luid_bits(luid) != 0) g_luid[device].store(luid_bits(luid), std::memory_order_release);
    IDXGIFactory4* f = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&f)) || !f) return nullptr;
    if (FAILED(f->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), (void**)&a))) a = nullptr;
    f->Release();
    if (a) g_dx[device].store(a, std::memory_order_release);
    return a;
}

/* This process's budget and use of the card's local memory. */
bool dxgi_local(int device, int64_t* budget, int64_t* usage, bool open) {
    IDXGIAdapter3* a = adapter_for(device, open);
    if (!a) return false;
    DXGI_QUERY_VIDEO_MEMORY_INFO mi{};
    if (FAILED(a->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mi))) return false;
    *budget = (int64_t)mi.Budget;
    *usage = (int64_t)mi.CurrentUsage;
    return true;
}

/* The shared (system memory) budget Windows gives this process on the adapter, or -1. */
int64_t shared_budget(int device) {
    IDXGIAdapter3* a = adapter_for(device, true);
    if (!a) return -1;
    DXGI_QUERY_VIDEO_MEMORY_INFO mi{};
    if (FAILED(a->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mi)) || !mi.Budget)
        return -1;
    return (int64_t)mi.Budget;
}

/* EVERY PROCESS'S use of the card's own memory, as the graphics kernel counts it: the bytes resident
 * in each of the adapter's memory segments, apertures (system memory the card sees through the
 * link) left out. The same figure Task Manager's "dedicated GPU memory" shows. A kernel thunk by
 * LUID: it takes no lock of the HIP runtime's. -1 when the kernel does not answer. */
int64_t d3dkmt_local_resident(uint64_t luid) {
    LUID l;
    l.LowPart = (DWORD)luid;
    l.HighPart = (LONG)(uint32_t)(luid >> 32);
    D3DKMT_QUERYSTATISTICS q;
    std::memset(&q, 0, sizeof q);
    q.Type = D3DKMT_QUERYSTATISTICS_ADAPTER;
    q.AdapterLuid = l;
    if (D3DKMTQueryStatistics(&q) != 0) return -1;
    const ULONG n = q.QueryResult.AdapterInformation.NbSegments;
    int64_t sum = 0;
    bool any = false;
    for (ULONG i = 0; i < n; ++i) {
        D3DKMT_QUERYSTATISTICS s;
        std::memset(&s, 0, sizeof s);
        s.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
        s.AdapterLuid = l;
        s.QuerySegment.SegmentId = i;
        if (D3DKMTQueryStatistics(&s) != 0) return -1;
        const D3DKMT_QUERYSTATISTICS_SEGMENT_INFORMATION& si = s.QueryResult.SegmentInformation;
        if (si.Aperture) continue;
#if (DXGKDDI_INTERFACE_VERSION >= DXGKDDI_INTERFACE_VERSION_WDDM2_9)
        if (si.SegmentProperties.SystemMemory) continue;   /* an integrated GPU's carve-out of RAM */
#endif
        sum += (int64_t)si.BytesResident;
        any = true;
    }
    return any ? sum : -1;
}

/* The card's user-mode driver version as DXGI reports it, packed a.b.c.d as four 16-bit fields
 * from the top (so packed versions compare as the versions do), or 0. */
uint64_t driver_version(int dev) {
    IDXGIAdapter3* a = adapter_for(dev, true);
    LARGE_INTEGER v{};
    if (!a || FAILED(a->CheckInterfaceSupport(__uuidof(IDXGIDevice), &v))) return 0;
    return (uint64_t)v.QuadPart;
}

/* TheRock#7221: drivers before 32.0.32015.2008 (Adrenalin 26.9.2) park an idle card's VRAM in
 * system memory. Brilliance 0.2.2 kept such cards awake with a fill every second; that keep-alive
 * is NOT carried into this port (retired). An older driver is named
 * once instead, so the cause of a slow first request after idling is on record. */
constexpr uint64_t kVramKeptFrom = (32ull << 48) | (0ull << 32) | (32015ull << 16) | 2008ull;

void check_driver(int dev) {
    static std::atomic<uint64_t> told{0};
    if (dev < 0 || dev >= kMaxDev || (told.fetch_or(1ull << dev) >> dev) & 1) return;
    const uint64_t v = driver_version(dev);
    if (v == 0 || v >= kVramKeptFrom) return;
    RAD_WARN("device: card %d's driver %u.%u.%u.%u parks an idle card's VRAM in system memory "
             "(TheRock#7221, fixed in 32.0.32015.2008, Adrenalin 26.9.2); this build does not keep "
             "cards awake, so the first request after an idle spell waits for it to come back. "
             "Update the AMD driver", dev, (unsigned)(v >> 48), (unsigned)(v >> 32) & 0xffff,
             (unsigned)(v >> 16) & 0xffff, (unsigned)v & 0xffff);
}

/* THE ROOM CHECK. The AQL backend refuses an allocation the driver has no room for before the
 * runtime is asked (aql.cpp, card_has_room); this does the same against the smaller of HIP's free
 * figure (for the current card only: hipMemGetInfo answers for no other) and the DXGI budget
 * room. `what` names the call refused.
 *
 * UNKNOWN AND FULL ARE DIFFERENT ANSWERS. A card neither HIP nor DXGI will describe is let through,
 * as the AQL backend lets it through. A DXGI room at or below zero is not unknown: the process is
 * already past its budget and WDDM is paging some of it now, so anything more is refused whatever
 * HIP's own figure says. */
bool card_has_room(int dev, int64_t bytes, const char* what) {
    int cur = -1;
    if (hipGetDevice(&cur) != hipSuccess) clear_sticky();
    int64_t avail = -1;   /* -1: no source could say */
    size_t fr = 0, tot = 0;
    if (cur == dev) {
        if (hipMemGetInfo(&fr, &tot) == hipSuccess) avail = (int64_t)fr;
        else clear_sticky();
    }
    int64_t budget = 0, usage = 0;
    const bool dx = dxgi_local(dev, &budget, &usage, true) && budget > 0;
    if (dx) {
        const int64_t room = budget - usage;
        if (room <= 0) {
            device_set_error("%s(%s) on card %d: this process already uses %s of the %s video-memory "
                             "budget Windows gives it on that card, so WDDM is already paging part of "
                             "it to system memory. Anything more would be paged too, without an "
                             "error, so the allocation is refused instead",
                             what, humanb(bytes).c_str(), dev, humanb(usage).c_str(),
                             humanb(budget).c_str());
            return false;
        }
        if (avail < 0 || room < avail) avail = room;
    }
    if (avail < 0) return true;
    constexpr int64_t kUnit = (int64_t)2 << 20;
    const int64_t need = (bytes + kUnit - 1) / kUnit * kUnit;
    if (need <= avail) return true;
    device_set_error("%s(%s) on card %d: the driver has %s left to give on it; this process already "
                     "uses %s of it. Windows would page the rest to system memory without an error, "
                     "so the allocation is refused instead. Memory another process took after the "
                     "VRAM budget was measured is the usual cause",
                     what, humanb(bytes).c_str(), dev, humanb(avail).c_str(),
                     dx ? humanb(usage).c_str() : "an unknown amount");
    return false;
}

/* ------------------------------------------------------------------ the pinned-memory limit
 *
 * EVERY PIN COUNTS ONCE FOR EVERY DEVICE THE PROCESS SEES. A portable hipHostMalloc is mapped onto
 * every visible device, and what Windows lets one process pin is the adapter's shared-memory budget
 * (DXGI's non-local segment group, about half to 80% of RAM) divided by that number of devices.
 * Measured by Brilliance 0.2.2 on a 99.7 GiB budget: two R9700s 50 GiB pinned and read, 51 hung;
 * with the integrated GPU visible too 32 read, 33 hung; one card 96 read. PAST IT NOTHING FAILS:
 * the pin succeeds and the first GPU operation after it blocks for good. So the allocation that
 * would cross the line is refused here, by name, before HIP is asked. The ceiling is the tested
 * driver's; another driver may move it.
 *
 * One account for the process (pin_ledger.h), held under one lock: pins are made at start-up and by
 * the pools, never on the step path. The margin is what the limit leaves for HIP's own pins and a
 * kernel library's (libr4d's collectives' mailboxes, which pin through HIP directly). Radiance 2.1's
 * plugin ABI has no hook for a library to report those, so they are covered by the margin alone;
 * the ledger's library side stays at zero until such a hook exists (Brilliance 0.2.2 had one,
 * rad_kernel_pin_hooks, outside the upstream ABI). */
constexpr int64_t kPinMargin = (int64_t)2 << 30;

struct PinLimit {
    int     devices = 0;
    int64_t budget = -1;     /* the smallest visible device's shared budget, or Microsoft's formula */
    int64_t limit = -1;      /* budget / devices - margin, or a lower BRILLIANCE_PIN_LIMIT_GIB; -1 unknown */
    bool    from_env = false; /* the limit is BRILLIANCE_PIN_LIMIT_GIB's */
    bool    read = false;
};
std::mutex g_pin_mu;
PinLedger  g_pins{ 0, 0, kPinMargin };
PinLimit   g_pin;

/* Read once, under g_pin_mu: the figure the OS states at start, not one that moves with memory
 * pressure. */
const PinLimit& pin_limit_locked() {
    if (g_pin.read) return g_pin;
    g_pin.read = true;
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) { clear_sticky(); n = 0; }
    g_pin.devices = n;
    int64_t b = -1;
    for (int d = 0; d < n; ++d) {
        const int64_t x = shared_budget(d);
        if (x > 0 && (b < 0 || x < b)) b = x;
    }
    if (b < 0) {
        /* No adapter answered: WDDM's own rule for the shared budget. */
        const int64_t t = os::mem_info().total;
        if (t > 0) b = std::min(t / 5 * 4, std::max(t - ((int64_t)16 << 30), t / 2));
    }
    g_pin.budget = b;
    const char* e = std::getenv("BRILLIANCE_PIN_LIMIT_GIB");
    const PinLimitChoice c = choose_pin_limit(b, n, kPinMargin, e);
    g_pin.limit = c.limit;
    g_pin.from_env = c.from_env;
    if (c.invalid)
        RAD_WARN("device: BRILLIANCE_PIN_LIMIT_GIB=%s is not a whole number of GiB above 0; it is "
                 "ignored", e);
    if (c.clamped)
        RAD_WARN("device: BRILLIANCE_PIN_LIMIT_GIB=%s asks for more than Windows lets this process "
                 "pin: %s, the %s shared-memory budget / %d device(s) - %s. Past that budget a pin "
                 "does not fail; the next GPU operation hangs. Using %s: the variable can only "
                 "lower the limit",
                 e, humanb(c.computed).c_str(), humanb(b).c_str(), n, humanb(kPinMargin).c_str(),
                 humanb(c.computed).c_str());
    RAD_INFO("device: this process may pin %s of host memory: %s%s shared-memory budget / %d "
             "device(s) - %s for HIP's and the kernel libraries' own pins",
             g_pin.limit >= 0 ? humanb(g_pin.limit).c_str() : "an unknown amount",
             c.from_env ? "BRILLIANCE_PIN_LIMIT_GIB; otherwise the " : "the ",
             humanb(b).c_str(), n, humanb(kPinMargin).c_str());
    return g_pin;
}

/* Takes `bytes` of the limit for a pin about to be made, or refuses it. The reservation comes
 * first so two ranks pinning at once cannot both pass on the same room; a failed pin gives it
 * back (unpin). */
bool pin_reserve(int64_t bytes) {
    std::lock_guard<std::mutex> lk(g_pin_mu);
    const PinLimit& l = pin_limit_locked();
    const int64_t before = g_pins.charged();
    if (g_pins.reserve_engine(bytes, l.limit)) return true;
    device_set_error("rad_dev_alloc(%s pinned): this process already pins %s, and Windows lets a "
                     "process pin about %s for its cards: the %s shared-memory budget divided by "
                     "the %d device(s) it sees, less %s%s. Past that the next GPU operation hangs "
                     "instead of failing, so the pin is refused. Lower --host-pool-mib, or hide the "
                     "devices this process does not use (HIP_VISIBLE_DEVICES)",
                     humanb(bytes).c_str(), humanb(before).c_str(), humanb(l.limit).c_str(),
                     humanb(l.budget).c_str(), l.devices, humanb(kPinMargin).c_str(),
                     l.from_env ? "; BRILLIANCE_PIN_LIMIT_GIB sets it here" : "");
    return false;
}

void unpin(int64_t bytes) {
    std::lock_guard<std::mutex> lk(g_pin_mu);
    g_pins.release_engine(bytes);
}

/* ------------------------------------------------------------------ memory
 *
 * Pinned host memory is hipHostMalloc, portable (every device sees it) and mapped (a kernel reads
 * it in place at the same address). Coherent by default: the Linux backend's pinned pools are
 * fine-grained, and the host-mailbox all-reduce and the zero-copy expert reads were measured against
 * that. BRILLIANCE_PINNED_COHERENT=0 asks for non-coherent memory once a measurement says a pool the
 * device only reads is cheaper so. RAD_MEM_HOST_PINNED and RAD_MEM_HOST_MAPPED are the same memory
 * here: on HIP a mapped pin's device pointer is its host pointer. */
namespace capture {   /* the stall capture's allocation notes, below the streams */
void note_alloc(void* p);
void forget(void* p);
}

void* n_alloc(int64_t bytes, int kind) {
    if (bytes <= 0) {
        device_set_error("rad_dev_alloc(%lld): size must be positive", (long long)bytes);
        return nullptr;
    }
    void* p = nullptr;
    hipError_t e = hipSuccess;
    switch (kind) {
        case RAD_MEM_DEVICE: {
            int dev = 0;
            if (hipGetDevice(&dev) != hipSuccess) clear_sticky();
            if (!card_has_room(dev, bytes, "rad_dev_alloc")) return nullptr;
            e = hipMalloc(&p, (size_t)bytes);
            break;
        }
        case RAD_MEM_HOST_PINNED:
        case RAD_MEM_HOST_MAPPED: {
            if (!pin_reserve(bytes)) return nullptr;
            const unsigned coh = env_on("BRILLIANCE_PINNED_COHERENT", true) ? hipHostMallocCoherent
                                                                            : hipHostMallocNonCoherent;
            e = hipHostMalloc(&p, (size_t)bytes, hipHostMallocPortable | hipHostMallocMapped | coh);
            if (e != hipSuccess || !p) unpin(bytes);
            break;
        }
        case RAD_MEM_HOST:
            /* Pageable, page-aligned so the same buffer can be an unbuffered-I/O target. */
            p = os::aligned_alloc(4096, (size_t)((bytes + 4095) / 4096 * 4096));
            if (!p) e = hipErrorOutOfMemory;
            break;
        default:
            device_set_error("rad_dev_alloc: unknown memory kind %d", kind);
            return nullptr;
    }
    if (e != hipSuccess || !p) {
        hip_fail(e == hipSuccess ? hipErrorOutOfMemory : e, "rad_dev_alloc");
        /* Within the pin limit a pin still needs two things free at the moment: physical memory,
         * since it stays resident, and COMMIT (RAM plus page file), which WDDM also charges with
         * this process's VRAM. The message says which ran short. */
        if (kind == RAD_MEM_HOST_PINNED || kind == RAD_MEM_HOST_MAPPED) {
            const os::MemInfo mi = os::mem_info();
            const bool commit = mi.commit_avail >= 0 && mi.commit_avail < bytes;
            device_set_error("rad_dev_alloc(%s pinned): %s. Now %s of RAM is available and %s of "
                             "commit (RAM + page file, %s in all, which this process's VRAM is "
                             "charged to as well)%s",
                             humanb(bytes).c_str(),
                             hipGetErrorString(e == hipSuccess ? hipErrorOutOfMemory : e),
                             humanb(mi.avail).c_str(), humanb(mi.commit_avail).c_str(),
                             humanb(mi.commit_limit).c_str(),
                             commit ? ": commit is what ran short; a larger page file raises it" : "");
        }
        return nullptr;
    }
    rec_add(p, bytes, kind);
    if (kind == RAD_MEM_HOST_PINNED || kind == RAD_MEM_HOST_MAPPED) capture::note_alloc(p);
    return p;
}

bool unmap_file(void* p);   /* map_file's free, below */

void n_free(void* p, int kind) {
    if (!p) return;
    Rec r{};
    if (!rec_take(p, &r)) {
        RAD_WARN("rad_dev_free(%p, kind=%d): not an allocation of the device layer; leaked rather "
                 "than freed with the wrong allocator", p, kind);
        return;
    }
    /* Freed by what it is, which the record knows; the caller's kind is only checked. */
    if (r.kind != kind)
        RAD_WARN("rad_dev_free(%p): freed as kind %d, allocated as kind %d", p, kind, r.kind);
    switch (r.kind) {
        case RAD_MEM_DEVICE:      HIP_NOTE(hipFree(p)); break;
        case RAD_MEM_HOST_PINNED:
        case RAD_MEM_HOST_MAPPED:
            capture::forget(p);
            if (!unmap_file(p)) HIP_NOTE(hipHostFree(p));
            unpin(r.bytes);
            break;
        case RAD_MEM_HOST:        os::aligned_free(p); break;
        default: break;
    }
}

/* ------------------------------------------------------------------ map_file
 *
 * [foff, foff + bytes) of a CRT descriptor's file, mapped shared and pinned for every card, as
 * core/device/device.h states the contract: a view of the file's own section (MapViewOfFile), then
 * hipHostRegister, portable and mapped, which locks the view's pages in place. UNLIKE A hipHostMalloc
 * PIN, A REGISTERED VIEW HAS ITS OWN DEVICE ADDRESS on Windows (measured 2026-11-17: the view at
 * 0x209e1e60000, the cards' at 0x20ac00000), where Linux's contract has one address. The engine
 * already asks: the n-gram gather reads rad_dev_device_ptr of the copy (Engine::copy_table's caller
 * in engine_bringup.cpp), and copies take either address. It counts against the pin limit like any
 * pin. Freed by rad_dev_free(p, RAD_MEM_HOST_MAPPED): unregistered, unmapped,
 * the file left alone. For --ngram-placement ram=DIR (core/format/table_share.h), where engines in
 * other processes map the same file. */
struct FileMap { HANDLE section; };
std::mutex                 g_fmap_mu;
std::map<uintptr_t, FileMap> g_fmaps;

bool unmap_file(void* p) {
    FileMap f{};
    {
        std::lock_guard<std::mutex> lk(g_fmap_mu);
        auto it = g_fmaps.find((uintptr_t)p);
        if (it == g_fmaps.end()) return false;
        f = it->second;
        g_fmaps.erase(it);
    }
    HIP_NOTE(hipHostUnregister(p));
    if (!UnmapViewOfFile(p))
        RAD_WARN("rad_dev_free(%p): UnmapViewOfFile failed (error %lu)", p, GetLastError());
    CloseHandle(f.section);
    return true;
}

void* n_map_file(int fd, int64_t foff, int64_t bytes) {
    if (fd < 0 || foff < 0 || bytes <= 0) {
        device_set_error("rad_dev_map_file(%d, %lld, %lld): a file, a non-negative offset and a "
                         "positive size are needed", fd, (long long)foff, (long long)bytes);
        return nullptr;
    }
    const HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) {
        device_set_error("rad_dev_map_file: descriptor %d is not an open file", fd);
        return nullptr;
    }
    const int64_t len = (bytes + 4095) / 4096 * 4096;
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(h, &size) || size.QuadPart < foff + bytes) {
        device_set_error("rad_dev_map_file(%s at +%s): the file holds %s, short of the range",
                         humanb(bytes).c_str(), humanb(foff).c_str(), humanb(size.QuadPart).c_str());
        return nullptr;
    }
    const HANDLE sec = CreateFileMappingW(h, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    if (!sec) {
        device_set_error("rad_dev_map_file(%s): CreateFileMapping failed (error %lu)",
                         humanb(bytes).c_str(), GetLastError());
        return nullptr;
    }
    ULARGE_INTEGER off;
    off.QuadPart = (uint64_t)foff;
    void* p = MapViewOfFile(sec, FILE_MAP_READ | FILE_MAP_WRITE, off.HighPart, off.LowPart, (SIZE_T)bytes);
    if (!p) {
        device_set_error("rad_dev_map_file(%s at +%s): MapViewOfFile failed (error %lu)",
                         humanb(bytes).c_str(), humanb(foff).c_str(), GetLastError());
        CloseHandle(sec);
        return nullptr;
    }
    auto undo = [&] {
        UnmapViewOfFile(p);
        CloseHandle(sec);
    };
    if (!pin_reserve(len)) {
        undo();
        return nullptr;
    }
    hipError_t e = hipHostRegister(p, (size_t)len, hipHostRegisterPortable | hipHostRegisterMapped);
    if (e != hipSuccess) {
        clear_sticky();
        device_set_error("rad_dev_map_file(%s): hipHostRegister of the file's view: %s",
                         humanb(bytes).c_str(), hipGetErrorString(e));
        unpin(len);
        undo();
        return nullptr;
    }
    void* dp = nullptr;
    e = hipHostGetDevicePointer(&dp, p, 0);
    if (e != hipSuccess || !dp) {
        clear_sticky();
        device_set_error("rad_dev_map_file(%s): the cards have no address for the file's view (%s)",
                         humanb(bytes).c_str(), hipGetErrorString(e == hipSuccess ? hipErrorInvalidValue : e));
        HIP_NOTE(hipHostUnregister(p));
        unpin(len);
        undo();
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(g_fmap_mu);
        g_fmaps[(uintptr_t)p] = FileMap{ sec };
    }
    rec_add(p, len, RAD_MEM_HOST_MAPPED);
    capture::note_alloc(p);
    return p;
}

/* hip.cpp's queries can fail harmlessly (a pointer HIP never saw); their answers are final, so
 * the sticky error they leave is cleared. */
void* n_host_ptr(void* p) {
    void* r = g_base->host_ptr(p);
    clear_sticky();
    return r;
}

void* n_device_ptr(void* p) {
    void* r = g_base->device_ptr(p);
    clear_sticky();
    return r;
}

/* ------------------------------------------------------------------ the stall watch
 *
 * dev_stuck_ms for this backend. The AQL backend watches its queues' read indices; HIP hides its
 * queues, so here progress is what the host sees of the device: a blocking wait (stream or event
 * synchronize) that returns, an event query that finds its work done, or a stream's progress
 * beacon that moved (namespace beacon below; the decisions are win/device/stall_watch.h's, which
 * win/tests/stall_watch_test.cpp drives). While some wait is
 * blocked and nothing has made progress for kStallMs, the device is stalled: stuck_ms is the time
 * since the last progress (/health answers 503 past the server's limit), and the verdict is logged
 * once a stall by a watcher thread, since the waiting thread is inside the driver.
 *
 * THE PROCESS IS NOT ENDED ON WINDOWS (scheduler.stuck-exit is reported, not acted on). Reviewed
 * for the port: Windows' own timeout detection and recovery (TDR) resets a GPU that stops
 * answering, and the runtime then fails the waits instead of hanging; and ending a process with
 * work on a card is what took the driver down under Brilliance 0.2.2 (2026-11-08). So the stall is
 * named and /health reports it; a supervisor or a person decides. */
/* THE LIVE STREAMS, for the stall watch's kick below: every stream this backend made, with its card
 * (the HIP index this process sees) and the order it was made in, which is all a log line can name
 * a stream by. */
namespace streams {
constexpr int kFrames = 6;
struct Info {
    int      card = -1;
    uint64_t seq = 0;
    bool     high = false;
    void*    made_at[kFrames] = {};   /* the creator's return addresses, innermost first */
};
std::mutex                    mu;
std::map<hipStream_t, Info>   live;
uint64_t                      next_seq = 0;

void add(hipStream_t s, bool high) {
    int card = -1;
    if (hipGetDevice(&card) != hipSuccess) (void)hipGetLastError();
    Info in;
    in.card = card;
    in.high = high;
    /* Skips this function and n_stream_create; the dispatch layer and its caller follow. */
    (void)CaptureStackBackTrace(2, kFrames, in.made_at, nullptr);
    std::lock_guard<std::mutex> lk(mu);
    in.seq = ++next_seq;
    live[s] = in;
}

/* " module+0xoffset" for each frame, the form win/tools/stacks/resolve-stacks.py names. */
std::string where(void* const* pcs, int n) {
    std::string out;
    for (int i = 0; i < n && pcs[i]; ++i) {
        HMODULE m = nullptr;
        char path[MAX_PATH] = {};
        char b[MAX_PATH + 32];
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)pcs[i], &m) &&
            GetModuleFileNameA(m, path, sizeof path)) {
            const char* base = std::strrchr(path, '\\');
            std::snprintf(b, sizeof b, " %s+0x%llx", base ? base + 1 : path,
                          (unsigned long long)((uintptr_t)pcs[i] - (uintptr_t)m));
        } else {
            std::snprintf(b, sizeof b, " %p", pcs[i]);
        }
        out += b;
    }
    return out;
}

void remove(hipStream_t s) {
    std::lock_guard<std::mutex> lk(mu);
    live.erase(s);
}
}  // namespace streams

/* THE STALL CAPTURE (BRILLIANCE_STALL_CAPTURE=1; libr4d's cap_run, r4d_ar_entry.hip): at a stall,
 * libr4d looks for the bytes of a host-mailbox block that matches no check word in every pinned
 * host buffer the engine owns. Those are this backend's: the allocation record holds them all
 * (the expert pool, the KV host tier, the prefix cache's host storage, staging rings), and with the
 * capture on each pinned allocation also keeps its creator's return addresses, taken when it is
 * made (start-up and the pools, never the step path), so the capture can name it. At a stall the
 * list goes to libr4d before the dump (hand_over, from stall::kick). Off, nothing here runs but
 * one test per pinned allocation. */
namespace capture {
constexpr int kFrames = 8;
bool on() {
    static const bool v = env_on("BRILLIANCE_STALL_CAPTURE", false);
    return v;
}
std::mutex                                        mu;
std::map<uintptr_t, std::vector<void*>>           made_at;

void note_alloc(void* p) {
    if (!on() || !p) return;
    std::vector<void*> f(kFrames, nullptr);
    /* Skips this function and n_alloc / n_map_file; the dispatch layer and its caller follow. */
    (void)CaptureStackBackTrace(2, kFrames, f.data(), nullptr);
    std::lock_guard<std::mutex> lk(mu);
    made_at[(uintptr_t)p] = std::move(f);
}

void forget(void* p) {
    if (!on() || !p) return;
    std::lock_guard<std::mutex> lk(mu);
    made_at.erase((uintptr_t)p);
}

void hand_over() {
    if (!on()) return;
    using Fn = void (*)(const long long*, const char* const*, int);
    HMODULE m = GetModuleHandleA("libr4d.dll");
    const Fn fn = m ? (Fn)(void*)GetProcAddress(m, "r4d_win_capture_ranges") : nullptr;
    if (!fn) {
        RAD_WARN("device: stall capture: libr4d's r4d_win_capture_ranges is not available; its origin "
                 "search will cover the collectives' mailboxes only");
        return;
    }
    std::vector<long long> pairs;
    std::vector<std::string> labels;
    int64_t total = 0;
    {
        std::shared_lock<std::shared_mutex> lk(g_rec_mu);
        std::lock_guard<std::mutex> lk2(mu);
        for (const auto& r : g_recs) {
            if (r.second.kind != RAD_MEM_HOST_PINNED && r.second.kind != RAD_MEM_HOST_MAPPED) continue;
            pairs.push_back((long long)r.first);
            pairs.push_back((long long)r.second.bytes);
            total += r.second.bytes;
            char b[160];
            std::snprintf(b, sizeof b, "pinned allocation %p (%s, %s), made at", (void*)r.first,
                          humanb(r.second.bytes).c_str(),
                          r.second.kind == RAD_MEM_HOST_PINNED ? "host-pinned" : "host-mapped");
            std::string l = b;
            auto w = made_at.find(r.first);
            l += w != made_at.end() ? streams::where(w->second.data(), (int)w->second.size()) : " (not noted)";
            labels.push_back(std::move(l));
        }
    }
    std::vector<const char*> lp;
    for (const std::string& l : labels) lp.push_back(l.c_str());
    fn(pairs.data(), lp.data(), (int)labels.size());
    RAD_WARN("device: stall capture: %zu pinned host allocation(s), %s in all, handed to libr4d's origin "
             "search", labels.size(), humanb(total).c_str());
}
}  // namespace capture

namespace slog {   /* the stream op log, below the events */
std::string dump_all();
void forget(const void* s);
std::string copy_overlaps();
}

/* THE COLLECTIVES' FAULT RECORDS (fix C1 stage 1; libr4d's r4d_ar_seq.h and r4d_ar_entry.hip). A
 * two-rank host mailbox's block that waited past its deadline (a push that arrived wrong, a flag that
 * never came) records a fault in pinned host memory and ends instead of spinning for ever. The
 * engine polls the records after every step (dev_coll_fault_poll); this finds libr4d's exports by
 * name, as the stall dump does (coll_trace.cpp), and logs the counters once a minute when any moved
 * (from the stall watch's thread, never the step path). */
namespace mbox {
using PollFn = int (*)(char*, int);
using CountFn = int (*)(long long*, int);
std::atomic<void*> g_poll{nullptr}, g_count{nullptr};

/* libr4d's export `name`, cached once found; null while libr4d is not loaded or lacks it. */
void* find(std::atomic<void*>& slot, const char* name) {
    void* f = slot.load(std::memory_order_acquire);
    if (f) return f;
    HMODULE m = GetModuleHandleA("libr4d.dll");
    f = m ? (void*)GetProcAddress(m, name) : nullptr;
    if (f) slot.store(f, std::memory_order_release);
    return f;
}

int poll(char* buf, int cap) {
    const PollFn fn = (PollFn)find(g_poll, "r4d_ar_fault_poll");
    return fn ? fn(buf, cap) : 0;
}

/* r4d_ar_mailbox_counters' entries (libr4d's R4D_AR_MB_COUNTERS): the two-rank host twins' 31, then
 * the four-rank host instances' (W1, the wide loud stop) from 31, then their exchange's (W2/W3, the
 * resend) from 53, then the coded two-shot's (W4) from 81. An older libr4d returns fewer. */
constexpr int kCounters = 87;

int counters(int64_t* out, int n) {
    const CountFn fn = (CountFn)find(g_count, "r4d_ar_mailbox_counters");
    if (!fn || !out || n <= 0) return 0;
    long long v[kCounters] = { 0 };
    const int k = fn(v, n < kCounters ? n : kCounters);
    for (int i = 0; i < k; ++i) out[i] = (int64_t)v[i];
    return k;
}

/* The four-rank half of the minute line (W1): a fault stops the engine, so the line is for the
 * late passes (a push that landed after its flag; harmless, but a link's health shows there first)
 * and for the faults of a run that kept going under injection. Counts are named by source rank, so a
 * rank whose link degrades (GPU 4 on its riser, design section 7) is not hidden in a total. */
void minute_wide(const int64_t* v, int k) {
    static int64_t last[kCounters] = { 0 };
    if (k < 53 || v[31] <= 0) return;
    bool moved = false;
    for (int i = 32; i < k && i < kCounters; ++i) moved = moved || v[i] != last[i];
    if (!moved) return;
    std::memcpy(last, v, sizeof last);
    if (k >= 81 && (v[53] | v[54] | v[55] | v[56])) {
        /* The exchange (W2/W3): a wrong or late block is resent, not stopped on, so this is where a
         * real one shows, named by the rank whose pushes failed and by collective. */
        RAD_WARN("r4d: four-rank host mailbox, since start: %lld block(s) resent (%lld request(s)), "
                 "%lld control re-store(s), %lld ack(s) never came; requests by source rank 0/1/2/3: "
                 "%lld/%lld/%lld/%lld; resends in hc_write %lld, gather_hc_write %lld, exact %lld, "
                 "gather %lld, two-shot %lld, coded two-shot %lld", (long long)v[54], (long long)v[53],
                 (long long)v[55], (long long)v[56], (long long)v[57], (long long)v[58],
                 (long long)v[59], (long long)v[60], (long long)v[62], (long long)v[66],
                 (long long)v[70], (long long)v[74], (long long)v[78], k >= 87 ? (long long)v[82] : 0LL);
    }
    char src[160] = "";
    std::snprintf(src, sizeof src, "by source rank 0/1/2/3: faults %lld/%lld/%lld/%lld, late "
                  "%lld/%lld/%lld/%lld", (long long)v[35], (long long)v[36], (long long)v[37],
                  (long long)v[38], (long long)v[39], (long long)v[40], (long long)v[41],
                  (long long)v[42]);
    char inj[96] = "";
    if (v[34] > 0) std::snprintf(inj, sizeof inj, "; %lld injection(s) that changed bytes", (long long)v[34]);
    RAD_WARN("r4d: four-rank host mailbox, since start over %lld instance(s): %lld fault(s), %lld late "
             "pass(es) (a push that landed after its flag); %s (faults in hc_write %lld, "
             "gather_hc_write %lld, exact %lld, gather %lld, two-shot %lld, coded two-shot %lld)%s",
             (long long)v[31], (long long)v[32], (long long)v[33], src, (long long)v[43],
             (long long)v[44], (long long)v[45], (long long)v[46], (long long)v[47],
             k >= 87 ? (long long)v[85] : 0LL, inj);
}

/* Once a minute, from the stall watch's thread: a line when any counter moved since the last. The
 * resend (fix C1 stages 2 to 4) makes a lost or misdirected line a counted resend instead of a
 * stop, so this line is where a real one shows, with the share of each collective other than
 * ar_hc_write that moved (ar_gather_hc_write, stage 3; the exact all-reduce and the all-gather,
 * stage 4; on tp-wire wht6, the wht6 all-reduce and the wht6 fused writes); under
 * BRILLIANCE_AR_INJECT it names the injections too. */
void minute() {
    static int64_t last[31] = { 0 };
    int64_t v[kCounters] = { 0 };
    const int k = counters(v, kCounters);
    minute_wide(v, k);
    if (k < 6 || v[0] <= 0) return;
    bool moved = false;
    for (int i = 1; i < 31; ++i) moved = moved || v[i] != last[i];
    if (!moved) return;
    std::memcpy(last, v, sizeof last);   /* the two-rank part, v[0..30] */
    char ghc[800] = "";
    static const struct { int at; const char* name; } kShare[] = {
        { 7, "ar_gather_hc_write" }, { 11, "ar_oneshot_2rank_exact" }, { 15, "ar_gather_2rank" },
        { 19, "ar_oneshot_2rank_wht6" }, { 23, "ar_hc_write (wht6)" },
        { 27, "ar_gather_hc_write (wht6)" } };
    size_t gl = 0;
    for (const auto& sh : kShare) {
        const int at = sh.at;
        if (k < at + 4 || !(v[at] | v[at + 1] | v[at + 2] | v[at + 3]) || gl >= sizeof ghc) continue;
        const int w = std::snprintf(ghc + gl, sizeof ghc - gl, " (in %s: %lld resent on %lld request(s), "
                                    "%lld re-stored, %lld ack(s) never came)", sh.name,
                                    (long long)v[at + 1], (long long)v[at], (long long)v[at + 2],
                                    (long long)v[at + 3]);
        if (w > 0) gl += (size_t)w;
    }
    char inj[96] = "";
    if (v[6] > 0) std::snprintf(inj, sizeof inj, "; %lld injected (BRILLIANCE_AR_INJECT)", (long long)v[6]);
    const char* what = (v[1] > 0 || v[3] > 0)
        ? "; GPU->host writes were lost or misdirected (SUPPORT.md, known issue 1)" : "";
    RAD_WARN("r4d: two-rank host mailbox, since start over %lld instance(s): %lld fault(s), %lld "
             "block(s) resent on %lld request(s), %lld control word(s) re-stored, %lld ack(s) "
             "never came%s%s%s", (long long)v[0], (long long)v[1], (long long)v[3], (long long)v[2],
             (long long)v[4], (long long)v[5], ghc, inj, what);
}
}  // namespace mbox

/* THE PROGRESS BEACONS (win/device/stall_watch.h): behind every BRILLIANCE_STALL_BEACONS-th op the
 * engine issues on a stream (default 64; 0 turns them off), the stream writes a fresh serial into
 * its own word of pinned host memory with hipStreamWriteValue64 -- the BRILLIANCE_SLOG_MARK=write
 * mechanism, a command processor write that makes HIP submit nothing -- and the stall watch reads
 * the words with a plain load. A step that keeps the card busy for many seconds (a 32K-token
 * prefill on one card) then shows the watch its progress, where before only the step's final wait
 * returning did, and the watch called it a stall.
 *
 * One pinned block for the process (portable, mapped, coherent: every card writes it, the host
 * sees each write), made with the first stream, never freed: kSlots words a cache line apart. A
 * stream takes a slot when it is made and gives it back when destroyed; a write still queued on a
 * destroyed stream lands in a slot another stream may hold, which reads as one reading of progress
 * at most. Past kSlots live streams, the rest have no beacon (said once). Cost: a lock and one
 * write every `every` ops; nothing on the ops between. */
namespace beacon {
constexpr int    kSlots = 64;
constexpr size_t kStride = 8;   /* uint64_t words: one slot a 64-byte line */
std::mutex                      mu;
volatile uint64_t*              block = nullptr;
bool                            failed = false;
std::vector<int>                free_slots;
std::map<hipStream_t, int>      slot_of;
std::atomic<uint64_t>           serial{0};

uint32_t every() {
    static const uint32_t v = [] {
        const char* e = std::getenv("BRILLIANCE_STALL_BEACONS");
        if (!e || !*e) return 64u;
        char* end = nullptr;
        const unsigned long n = std::strtoul(e, &end, 10);
        if (end == e || *end || n > 1000000ul) {
            RAD_WARN("device: BRILLIANCE_STALL_BEACONS=%s is not a whole number of ops (0 turns the "
                     "stall watch's beacons off); 64 is used", e);
            return 64u;
        }
        if (n == 0)
            RAD_WARN("device: the stall watch's progress beacons are off (BRILLIANCE_STALL_BEACONS=0): "
                     "a step that keeps a card busy past 5 s reads as a stall");
        return (uint32_t)n;
    }();
    return v;
}

/* A new stream's slot; the block is made with the first. */
void attach(hipStream_t s) {
    if (every() == 0) return;
    std::lock_guard<std::mutex> lk(mu);
    if (!block && !failed) {
        void* p = nullptr;
        const hipError_t e = hipHostMalloc(&p, kSlots * kStride * sizeof(uint64_t),
                                           hipHostMallocPortable | hipHostMallocMapped |
                                               hipHostMallocCoherent);
        if (e != hipSuccess || !p) {
            (void)hipGetLastError();
            failed = true;
            RAD_WARN("device: the stall watch's progress beacons are off: their pinned block could "
                     "not be made (%s), so a step that keeps a card busy past 5 s reads as a stall",
                     hipGetErrorString(e));
            return;
        }
        std::memset(p, 0, kSlots * kStride * sizeof(uint64_t));
        block = (volatile uint64_t*)p;
        for (int i = kSlots - 1; i >= 0; --i) free_slots.push_back(i);
    }
    if (!block) return;
    if (free_slots.empty()) {
        static bool said = false;
        if (!said) {
            said = true;
            RAD_WARN("device: more than %d live streams: the ones past it have no stall-watch beacon",
                     kSlots);
        }
        return;
    }
    slot_of[s] = free_slots.back();
    free_slots.pop_back();
}

void detach(hipStream_t s) {
    std::lock_guard<std::mutex> lk(mu);
    const auto it = slot_of.find(s);
    if (it == slot_of.end()) return;
    free_slots.push_back(it->second);
    slot_of.erase(it);
}

/* An op the engine issued on `s` (hipnative_note_launch): a beacon behind every every()-th. */
void on_launch(hipStream_t s) {
    static thread_local BeaconCadence cadence{ every(), 0 };
    if (!cadence.due()) return;
    volatile uint64_t* word = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu);
        const auto it = slot_of.find(s);
        if (it == slot_of.end()) return;
        word = block + (size_t)it->second * kStride;
    }
    const uint64_t v = serial.fetch_add(1, std::memory_order_relaxed) + 1;
    const hipError_t r = hipStreamWriteValue64(s, (void*)word, v, 0);
    if (r != hipSuccess) {
        (void)hipGetLastError();
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            RAD_WARN("device: a stall-watch beacon could not go on its stream (%s); that stream's "
                     "long steps can read as a stall", hipGetErrorString(r));
    }
}

/* Every slot's word now, into out[kSlots]; 0 when there is no block. */
size_t read(uint64_t* out) {
    std::lock_guard<std::mutex> lk(mu);
    if (!block) return 0;
    for (int i = 0; i < kSlots; ++i) out[i] = block[(size_t)i * kStride];
    return (size_t)kSlots;
}
}  // namespace beacon

namespace stall {
constexpr int64_t kStallMs = 5000;
std::mutex                    mu;
StallWatch                    w(kStallMs);   /* under mu */
std::once_flag                watcher_once;

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* A reported stall that progress ended: k is its kick's time (StallWatch::progress). Only the
 * timing is known here. The markers may have made HIP submit held work, or the stall may have
 * ended on its own (a held stream let go, a slow step finishing, a driver reset); the watch
 * cannot tell which, so the line names no cause (2026-11-20, the held-stream test: let go about
 * 0.15 s after the kick, the line then claimed unsubmitted work). */
void say_resumed(int64_t k, int64_t t) {
    if (k > 0)
        RAD_WARN("device: the device resumed %lld ms after the stall watch's markers (the streams "
                 "named above); the watch does not know what ended the stall",
                 (long long)(t - k));
}

void progress() {
    const int64_t t = now_ms();
    int64_t k = 0;
    {
        std::lock_guard<std::mutex> lk(mu);
        k = w.progress(t);
    }
    say_resumed(k, t);
}

/* THE KICK. HIP on Windows batches what is queued on a stream and submits it later; work another
 * stream or card waits for (an event, a collective's mailbox) can therefore sit unsubmitted while
 * everything that would submit it waits too -- the device idle, nothing failing (2026-11-16, Flash-
 * Next at TP2: both ranks in hipEventSynchronize, every GPU engine at 0%). On a stall, every live
 * stream is queried, and one with work still queued gets a marker event, either of which makes HIP
 * submit what it holds. Harmless to correct work (a marker orders nothing), and the log says which
 * streams were behind; if the device then resumes, progress() says so. Run on a thread of its own,
 * so a runtime call that blocks here cannot silence the watcher. */
void kick() {
    std::vector<std::pair<hipStream_t, streams::Info>> v;
    {
        std::lock_guard<std::mutex> lk(streams::mu);
        v.assign(streams::live.begin(), streams::live.end());
    }
    std::string busy;
    int n_busy = 0;
    for (const auto& sv : v) {
        const hipError_t q = hipStreamQuery(sv.first);
        if (q == hipSuccess) continue;
        (void)hipGetLastError();
        if (q != hipErrorNotReady) continue;
        ++n_busy;
        char b[128];
        std::snprintf(b, sizeof b, "\n    #%llu card %d%s, stream %p, made at",
                      (unsigned long long)sv.second.seq, sv.second.card,
                      sv.second.high ? ", high priority" : "", (void*)sv.first);
        busy += b;
        busy += streams::where(sv.second.made_at, streams::kFrames);
        if (sv.second.card < 0 || hipSetDevice(sv.second.card) != hipSuccess) {
            (void)hipGetLastError();
            continue;
        }
        hipEvent_t e = nullptr;
        if (hipEventCreateWithFlags(&e, hipEventDisableTiming) != hipSuccess) {
            (void)hipGetLastError();
            continue;
        }
        if (hipEventRecord(e, sv.first) != hipSuccess) (void)hipGetLastError();
        (void)hipEventDestroy(e);
    }
    {
        std::lock_guard<std::mutex> lk(mu);
        w.kicked(now_ms());
    }
    RAD_WARN("device: stall watch: %d of %zu stream(s) still have work queued:%s. Each was queried "
             "and given a marker event, which makes HIP submit what it holds back",
             n_busy, v.size(), n_busy ? busy.c_str() : " none");
    /* The stall capture's search list, before the dump starts it (BRILLIANCE_STALL_CAPTURE). */
    capture::hand_over();
    /* What each rank issued last across ranks: where two ranks' tails differ, they diverged. */
    ctrace::dump();
    /* Each stream's last records and waits: a wait cycle between streams shows here. */
    RAD_WARN("device: stream op log (R record, W wait, C copy; numbers are each stream's own op "
             "count):%s", slog::dump_all().c_str());
    /* Whether any copy or memset this backend issued wrote into a collective's host mailbox. */
    const std::string ov = slog::copy_overlaps();
    if (!ov.empty()) RAD_WARN("device: copies into the host mailboxes:%s", ov.c_str());
}

/* The beacons are read first: a word that moved is progress, which a long step that keeps the card
 * busy shows here and nowhere else until its wait returns. */
int64_t stuck_ms() {
    uint64_t words[beacon::kSlots];
    const size_t n = beacon::read(words);
    const int64_t t = now_ms();
    StallWatch::Verdict v;
    int64_t k = 0, last_beacon = -1;
    uint64_t moves = 0;
    {
        std::lock_guard<std::mutex> lk(mu);
        if (n) k = w.observe(words, n, t);
        v = w.check(t);
        moves = w.beacon_moves();
        last_beacon = w.last_beacon();
    }
    say_resumed(k, t);
    if (!v.report) return v.ms;
    char beacons[160];
    if (beacon::every() == 0 || n == 0)
        std::snprintf(beacons, sizeof beacons, "the progress beacons are off, so a step that keeps "
                      "a card busy this long would read the same");
    else if (last_beacon < 0)
        std::snprintf(beacons, sizeof beacons, "no stream's progress beacon has moved since start");
    else
        std::snprintf(beacons, sizeof beacons, "no stream's progress beacon has moved for %lld s "
                      "(%llu readings found one moving before)", (long long)((t - last_beacon) / 1000),
                      (unsigned long long)moves);
    RAD_ERR("device: nothing has completed on the device for %lld s while a host wait is "
            "blocked on it, and %s: a hung kernel, work HIP never submitted, or a device the "
            "driver lost. On Windows the process is not ended for it (scheduler.stuck-exit is not "
            "armed: TDR resets a GPU that stops answering, and ending a process with work on a "
            "card has crashed the driver); /health reports the stall",
            (long long)(v.ms / 1000), beacons);
    std::thread(kick).detach();
    return v.ms;
}

/* One blocking wait, from entry to return. The watcher starts with the first wait. */
struct Wait {
    uint64_t id;
    Wait() {
        std::call_once(watcher_once, [] {
            std::thread([] {
                for (int tick = 1;; ++tick) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    (void)stuck_ms();
                    if (tick % 60 == 0) mbox::minute();   /* the mailbox counters, if any moved */
                }
            }).detach();
        });
        std::lock_guard<std::mutex> lk(mu);
        id = w.wait_begin(now_ms());
    }
    ~Wait() {
        const int64_t t = now_ms();
        int64_t k = 0;
        {
            std::lock_guard<std::mutex> lk(mu);
            k = w.wait_end(id, t);
        }
        say_resumed(k, t);
    }
};
}  // namespace stall

int64_t n_stuck_ms() { return stall::stuck_ms(); }
int n_coll_fault_poll(char* buf, int cap) { return mbox::poll(buf, cap); }
int n_coll_counters(int64_t* out, int n) { return mbox::counters(out, n); }

/* ------------------------------------------------------------------ streams
 *
 * Non-blocking: no implicit ordering against the null stream, which the engine never uses
 * (device.h, contract note 2) and plugin bring-up code still does (a synchronous hipMemset). Work a
 * library puts on the null stream is therefore NOT ordered before work on these streams; the
 * library waits for it itself. Stream
 * priorities are asked for so a runtime that honours them gets them; HIP on Windows has accepted
 * and ignored them so far. */
int n_stream_create(RadStream* out, int high_priority) {
    if (!out) return RAD_E_INVAL;
    hipStream_t s = nullptr;
    if (high_priority) {
        int lo = 0, hi = 0;
        if (hipDeviceGetStreamPriorityRange(&lo, &hi) == hipSuccess &&
            hipStreamCreateWithPriority(&s, hipStreamNonBlocking, hi) == hipSuccess) {
            streams::add(s, true);
            beacon::attach(s);
            *out = (RadStream)s;
            return RAD_OK;
        }
        clear_sticky();   /* handled: a plain stream below */
    }
    HIP_TRY(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    streams::add(s, false);
    beacon::attach(s);
    *out = (RadStream)s;
    return RAD_OK;
}

void n_stream_destroy(RadStream s) {
    if (!s) return;
    streams::remove((hipStream_t)s);
    beacon::detach((hipStream_t)s);
    slog::forget(s);
    HIP_NOTE(hipStreamDestroy((hipStream_t)s));
}

int n_stream_sync(RadStream s) {
    if (!s) {
        device_set_error("rad_stream_sync: the null stream (the engine never uses it)");
        return RAD_E_INVAL;
    }
    stall::Wait w;
    HIP_TRY(hipStreamSynchronize((hipStream_t)s));
    return RAD_OK;
}

/* ------------------------------------------------------------------ events
 *
 * A host-wait event blocks the waiting thread in the driver (hipEventBlockingSync), as the AQL
 * backend's interrupt waits do, rather than spinning a core per rank thread.
 * BRILLIANCE_EVENT_SPIN=1 spins instead, for the decode latency a sleeping wake-up costs; a
 * measurement decides the default. RAD_EVENT_LOCAL is accepted and ignored: a HIP event's release is
 * system-wide, so there is no cheaper scope to pick (device.h allows exactly that).
 *
 * A HIP EVENT BELONGS TO A CARD. The runtime refuses to record an event made on one card onto
 * another card's stream (hipErrorInvalidHandle), while a RadEvent belongs to no card: the AQL
 * backend's events are signals any queue completes. So an event records the card it was made on,
 * and a record onto another card's stream remakes it there first. That is a slow path taken once
 * per move, never in a steady step, where each event stays with its stream. */
struct NEv {
    hipEvent_t ev = nullptr;
    unsigned   flags = 0;
    int        dev = -1;
    /* Where it was last recorded: the stream and that stream's op number (slog below). */
    std::atomic<void*>    last_stream{nullptr};
    std::atomic<uint64_t> last_serial{0};
};

/* THE STREAM OP LOG, for a stall between streams: each stream's last records, waits and copies,
 * and for a wait, where its event had last been recorded when the wait was made. A stall where
 * stream A waits on a record B made after B's own wait on A is a cycle the log shows directly;
 * one with no such cycle in the log is the runtime binding a wait differently (stall::kick prints
 * it). Kernel launches go to HIP from the plugins and are not here; the collective trace has the
 * collectives. Cheap: a lookup under a lock and a few stores per record, wait or copy. */
namespace slog {
constexpr int kRing = 4096;   /* deep enough that launches (K) leave the step's records in view */
struct Op {
    uint64_t serial = 0;
    char     kind = 0;            /* R record, W wait, C copy, L copy list, G gathered ranges,
                                     K a kernel launch the engine issued (BRILLIANCE_SLOG_LAUNCHES) */
    const void* ev = nullptr;
    const void* rec_stream = nullptr;
    uint64_t rec_serial = 0;
    int64_t  bytes = 0;
    const char* op = nullptr;     /* K: the op's and the kernel's names, which outlive the log */
    const char* kernel = nullptr;
    /* K with completion marks on (mark_mode): the event recorded right behind the launch, or for
     * the "write" mode any non-null value; null where no mark was placed. */
    const void* mark = nullptr;
};
struct Log {
    uint64_t n = 0;
    Op       ring[kRing];
    /* The completion marks' own events, one per ring slot, made on the slot's first marked launch
     * and re-recorded each time a launch lands in that slot again (so a slot's event always
     * belongs to the K op the slot holds). */
    hipEvent_t kev[kRing] = {};
    /* The "write" mode's word in pinned host memory: the serial of the last launch the stream
     * completed, written by the stream itself behind each launch. */
    volatile uint64_t* word = nullptr;
    bool               word_failed = false;
};
std::mutex                           mu;
std::map<const void*, Log>           logs;

/* THE COMPLETION MARKS. With BRILLIANCE_SLOG_LAUNCHES=1 each launch the engine issues is logged
 * by name (K), and by default a mark goes on the stream right behind it, so a stall dump can name
 * the LAST launch each stream completed and the FIRST it did not. BRILLIANCE_SLOG_MARK picks how:
 *   event (default)  an event recorded behind each launch, one per ring slot, queried by the dump.
 *                    A record also makes HIP submit what it holds (hipnative_submit), so this
 *                    submits after every launch, as GPU_FLUSH_ON_EXECUTION=1 does;
 *   write            hipStreamWriteValue64 of the launch's serial into a word of pinned host
 *                    memory, one word a stream: the stream's own progress, without an event;
 *   off              names only (the log as of d15089f).
 * Off by default with the launch log; nothing here runs unless BRILLIANCE_SLOG_LAUNCHES is set. */
int mark_mode() {
    static const int m = [] {
        const char* v = std::getenv("BRILLIANCE_SLOG_MARK");
        if (!v || !*v || _stricmp(v, "event") == 0 || _stricmp(v, "1") == 0) return 1;
        if (_stricmp(v, "write") == 0) return 2;
        return 0;
    }();
    return m;
}

uint64_t note(const void* s, char kind, const void* ev, const void* rs, uint64_t rser, int64_t bytes) {
    std::lock_guard<std::mutex> lk(mu);
    Log& L = logs[s];
    const uint64_t k = ++L.n;
    Op& o = L.ring[k % kRing];
    o.serial = k;
    o.kind = kind;
    o.ev = ev;
    o.rec_stream = rs;
    o.rec_serial = rser;
    o.bytes = bytes;
    o.op = nullptr;
    o.kernel = nullptr;
    return k;
}

uint64_t note_launch(const void* s, const char* op, const char* kernel) {
    Log* L = nullptr;
    uint64_t k = 0;
    hipEvent_t ev = nullptr;
    volatile uint64_t* word = nullptr;
    const int mode = mark_mode();
    {
        std::lock_guard<std::mutex> lk(mu);
        L = &logs[s];
        k = ++L->n;
        Op& o = L->ring[k % kRing];
        o.serial = k;
        o.kind = 'K';
        o.ev = nullptr;
        o.rec_stream = nullptr;
        o.rec_serial = 0;
        o.bytes = 0;
        o.op = op;
        o.kernel = kernel;
        o.mark = nullptr;
        if (mode == 1) ev = L->kev[k % kRing];
        if (mode == 2 && !L->word_failed) word = L->word;
        if (mode == 2 && !L->word && !L->word_failed) {
            void* p = nullptr;
            if (hipHostMalloc(&p, 64, hipHostMallocMapped) == hipSuccess && p) {
                std::memset(p, 0, 64);
                L->word = word = (volatile uint64_t*)p;
            } else {
                (void)hipGetLastError();
                L->word_failed = true;
            }
        }
    }
    /* The mark goes on the stream outside the lock: a record costs about 15 us, and the ranks'
     * launches would otherwise queue behind each other's. Only the issuing thread touches this
     * slot's event (a stream is issued from one thread). */
    const void* placed = nullptr;
    if (mode == 1) {
        if (!ev && hipEventCreateWithFlags(&ev, hipEventDisableTiming) != hipSuccess) {
            (void)hipGetLastError();
            ev = nullptr;
        }
        if (ev && hipEventRecord(ev, (hipStream_t)s) == hipSuccess) placed = ev;
        else (void)hipGetLastError();
    } else if (mode == 2 && word) {
        const hipError_t r = hipStreamWriteValue64((hipStream_t)s, (void*)word, k, 0);
        if (r == hipSuccess) placed = (const void*)word;
        else (void)hipGetLastError();
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            RAD_WARN("device: launch marks (BRILLIANCE_SLOG_MARK=write): the first %s", r == hipSuccess
                     ? "went on its stream" : hipGetErrorString(r));
    } else if (mode == 2) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            RAD_WARN("device: launch marks (BRILLIANCE_SLOG_MARK=write): no pinned word for the "
                     "stream, so no marks");
    }
    if (mode == 1 && !placed) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) RAD_WARN("device: launch marks (event): a record failed");
    }
    if (mode == 1 || placed) {
        std::lock_guard<std::mutex> lk(mu);
        if (mode == 1) L->kev[k % kRing] = ev;
        Op& o = L->ring[k % kRing];
        if (o.serial == k) o.mark = placed;
    }
    return k;
}

/* THE COPY LOG (BRILLIANCE_SLOG_LAUNCHES=1): the destination and size of every copy and memset this
 * backend issues, in one ring for the process, so a stall can test them against the collectives'
 * host mailboxes (r4d_win_mailbox_ranges, libr4d). Kernel writes into host memory are not here. */
struct CopyRec { const void* dst; int64_t bytes; const void* stream; int64_t ms; char kind; };
constexpr int kCopies = 65536;
std::mutex   copy_mu;
CopyRec      copies[kCopies];
uint64_t     n_copies = 0;

bool copy_log_on() {
    static const bool on = env_on("BRILLIANCE_SLOG_LAUNCHES", false);
    return on;
}

void note_copy(const void* dst, int64_t bytes, const void* s, char kind) {
    if (!copy_log_on() || !dst || bytes <= 0) return;
    const int64_t t = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lk(copy_mu);
    copies[n_copies % kCopies] = CopyRec{ dst, bytes, s, t, kind };
    ++n_copies;
}

std::string copy_overlaps() {
    if (!copy_log_on()) return "";
    using RangesFn = int (*)(long long*, int);
    HMODULE m = GetModuleHandleA("libr4d.dll");
    const RangesFn fn = m ? (RangesFn)(void*)GetProcAddress(m, "r4d_win_mailbox_ranges") : nullptr;
    if (!fn) return " (libr4d's mailbox ranges are not available)";
    long long r[128];
    const int nr = fn(r, 64);
    const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lk(copy_mu);
    const uint64_t from = n_copies > (uint64_t)kCopies ? n_copies - kCopies : 0;
    uint64_t hits = 0;
    std::string out, first;
    char b[200];
    for (uint64_t i = from; i < n_copies; ++i) {
        const CopyRec& c = copies[i % kCopies];
        const uintptr_t d0 = (uintptr_t)c.dst, d1 = d0 + (uintptr_t)c.bytes;
        for (int j = 0; j < nr; ++j) {
            const uintptr_t m0 = (uintptr_t)r[2 * j], m1 = m0 + (uintptr_t)r[2 * j + 1];
            if (d0 < m1 && m0 < d1) {
                if (++hits <= 8) {
                    std::snprintf(b, sizeof b, " [%c %p +%lld into %p+%lld, %lld ms ago]", c.kind,
                                  c.dst, (long long)c.bytes, (void*)m0, (long long)(m1 - m0),
                                  (long long)(now - c.ms));
                    first += b;
                }
                break;
            }
        }
    }
    const int64_t span = n_copies > from ? now - copies[from % kCopies].ms : 0;
    std::snprintf(b, sizeof b, " %llu of the last %llu copies and memsets (%lld ms) wrote into the %d "
                  "mailbox range(s)", (unsigned long long)hits, (unsigned long long)(n_copies - from),
                  (long long)span, nr);
    out = b;
    return out + first;
}

void forget(const void* s) {
    std::lock_guard<std::mutex> lk(mu);
    auto it = logs.find(s);
    if (it == logs.end()) return;
    for (hipEvent_t& e : it->second.kev)
        if (e) (void)hipEventDestroy(e);
    if (it->second.word) (void)hipHostFree((void*)it->second.word);
    logs.erase(it);
}

/* The last `n` ops of stream `s`, oldest first; streams are named by `name(stream)`. */
template <class Name>
std::string tail(const void* s, int n, const Name& name) {
    std::lock_guard<std::mutex> lk(mu);
    auto it = logs.find(s);
    if (it == logs.end()) return " (nothing logged)";
    const Log& L = it->second;
    std::string out;
    char b[160];
    const uint64_t from = L.n > (uint64_t)n ? L.n - (uint64_t)n + 1 : 1;
    std::snprintf(b, sizeof b, " %llu op(s);", (unsigned long long)L.n);
    out += b;
    for (uint64_t k = from; k <= L.n; ++k) {
        const Op& o = L.ring[k % kRing];
        if (o.serial != k) continue;
        if (o.kind == 'R')
            std::snprintf(b, sizeof b, " [%llu R ev %p]", (unsigned long long)k, o.ev);
        else if (o.kind == 'W')
            std::snprintf(b, sizeof b, " [%llu W ev %p, recorded on %s at %llu]", (unsigned long long)k,
                          o.ev, name(o.rec_stream).c_str(), (unsigned long long)o.rec_serial);
        else
            std::snprintf(b, sizeof b, " [%llu %c %lld B]", (unsigned long long)k, o.kind,
                          (long long)o.bytes);
        out += b;
    }
    return out;
}

/* Live events, so the dump queries only events that still exist. */
std::mutex                  ev_mu;
std::set<const void*>       ev_live;
void ev_add(const void* e) { std::lock_guard<std::mutex> lk(ev_mu); ev_live.insert(e); }
void ev_del(const void* e) { std::lock_guard<std::mutex> lk(ev_mu); ev_live.erase(e); }

/* Where stream `s`'s execution stands: the last logged record found complete and the first found
 * pending, judged by records whose event has not been re-recorded since (a re-recorded event says
 * nothing about the older record). 0 where none qualifies. Queries HIP; under `mu`. */
void frontier(const void* s, const Log& L, uint64_t* done, uint64_t* pending) {
    *done = 0;
    *pending = 0;
    const uint64_t from = L.n > (uint64_t)kRing ? L.n - kRing + 1 : 1;
    std::lock_guard<std::mutex> lk(ev_mu);
    for (uint64_t k = from; k <= L.n; ++k) {
        const Op& o = L.ring[k % kRing];
        if (o.serial != k || o.kind != 'R' || !ev_live.count(o.ev)) continue;
        const NEv* e = (const NEv*)o.ev;
        if (e->last_stream.load() != s || e->last_serial.load() != k) continue;
        /* Not an event a thread may be synchronizing on: HIP holds an event's lock for the whole
         * of hipEventSynchronize, and a query would wait behind the stall it is reporting. */
        if (e->flags & RAD_EVENT_HOST_WAIT) continue;
        const hipError_t q = hipEventQuery(e->ev);
        if (q == hipSuccess) *done = k;
        else {
            (void)hipGetLastError();
            if (!*pending && q == hipErrorNotReady) *pending = k;
        }
    }
}

std::string dump_all() {
    std::vector<std::pair<hipStream_t, streams::Info>> v;
    {
        std::lock_guard<std::mutex> lk(streams::mu);
        v.assign(streams::live.begin(), streams::live.end());
    }
    auto name = [&](const void* s) -> std::string {
        for (const auto& sv : v)
            if (sv.first == s) return "#" + std::to_string(sv.second.seq);
        char b[32];
        std::snprintf(b, sizeof b, "%p", s);
        return b;
    };
    std::lock_guard<std::mutex> lk(mu);
    std::map<const void*, std::pair<uint64_t, uint64_t>> fr;
    for (const auto& sv : v) {
        auto it = logs.find(sv.first);
        if (it == logs.end()) continue;
        uint64_t d = 0, p = 0;
        frontier(sv.first, it->second, &d, &p);
        fr[sv.first] = {d, p};
    }
    std::string out;
    char b[200];
    for (const auto& sv : v) {
        auto it = logs.find(sv.first);
        if (it == logs.end()) continue;
        const Log& L = it->second;
        const auto f = fr[sv.first];
        std::snprintf(b, sizeof b, "\n    %s (card %d): %llu op(s); records complete through %llu, "
                      "first pending record at %llu; waits after %llu:",
                      name(sv.first).c_str(), sv.second.card, (unsigned long long)L.n,
                      (unsigned long long)f.first, (unsigned long long)f.second,
                      (unsigned long long)f.first);
        out += b;
        const uint64_t from = L.n > (uint64_t)kRing ? L.n - kRing + 1 : 1;
        int shown = 0;
        for (uint64_t k = std::max(from, f.first + 1); k <= L.n && shown < 16; ++k) {
            const Op& o = L.ring[k % kRing];
            if (o.serial != k || o.kind != 'W') continue;
            const auto src = fr.count(o.rec_stream) ? fr[o.rec_stream] : std::make_pair(0ull, 0ull);
            std::snprintf(b, sizeof b, " [%llu waits on %s's %llu (%s: complete through %llu)]",
                          (unsigned long long)k, name(o.rec_stream).c_str(),
                          (unsigned long long)o.rec_serial,
                          o.rec_serial <= src.first ? "done" : "NOT DONE",
                          (unsigned long long)src.first);
            out += b;
            ++shown;
        }
        if (!shown) out += " none logged";
        /* With launches logged: every op past the last completed record, in order, so the dump
         * names what the stream stopped on (a kernel, a copy, a wait). */
        bool any_k = false;
        for (uint64_t k = from; k <= L.n && !any_k; ++k) any_k = L.ring[k % kRing].kind == 'K';
        if (!any_k) continue;
        /* The stream stopped somewhere between the last record found complete and the first found
         * pending (records whose event was re-recorded since are not judged, so the two can be far
         * apart). The ops just before the first pending record are where it stopped. */
        auto list = [&](uint64_t lo, uint64_t hi) {
            for (uint64_t k = std::max(from, lo); k <= std::min(hi, L.n); ++k) {
                const Op& o = L.ring[k % kRing];
                if (o.serial != k) continue;
                if (o.kind == 'K')
                    std::snprintf(b, sizeof b, " [%llu K %s / %s]", (unsigned long long)k,
                                  o.op ? o.op : "?", o.kernel ? o.kernel : "?");
                else if (o.kind == 'W')
                    std::snprintf(b, sizeof b, " [%llu W %s's %llu]", (unsigned long long)k,
                                  name(o.rec_stream).c_str(), (unsigned long long)o.rec_serial);
                else if (o.kind == 'R')
                    std::snprintf(b, sizeof b, " [%llu R%s]", (unsigned long long)k,
                                  k == f.second ? " FIRST PENDING" : "");
                else
                    std::snprintf(b, sizeof b, " [%llu %c %lld B]", (unsigned long long)k, o.kind,
                                  (long long)o.bytes);
                out += b;
            }
        };
        /* With completion marks: the last launch the stream completed and the first it did not,
         * from the launches' own marks rather than the records'. A stream runs in order, so any
         * marked launch found done past the first pending one is reported as an anomaly. */
        uint64_t k_done = 0, k_pend = 0, n_marked = 0, n_late = 0;
        if (L.word) {
            const uint64_t w = *L.word;
            for (uint64_t k = from; k <= L.n; ++k) {
                const Op& o = L.ring[k % kRing];
                if (o.serial != k || o.kind != 'K' || !o.mark) continue;
                ++n_marked;
                if (k <= w) k_done = k;
                else if (!k_pend) k_pend = k;
            }
        } else {
            for (uint64_t k = from; k <= L.n; ++k) {
                const Op& o = L.ring[k % kRing];
                if (o.serial != k || o.kind != 'K' || !o.mark) continue;
                ++n_marked;
                const hipError_t q = hipEventQuery((hipEvent_t)o.mark);
                if (q == hipSuccess) {
                    k_done = k;
                    if (k_pend) ++n_late;
                } else {
                    (void)hipGetLastError();
                    if (!k_pend && q == hipErrorNotReady) k_pend = k;
                }
            }
        }
        if (n_marked) {
            auto kname = [&](uint64_t k) -> std::string {
                if (!k) return "none in the log";
                const Op& o = L.ring[k % kRing];
                char nb[200];
                std::snprintf(nb, sizeof nb, "%llu %s / %s", (unsigned long long)k,
                              o.op ? o.op : "?", o.kernel ? o.kernel : "?");
                return nb;
            };
            out += "\n        LAUNCH MARKS (";
            out += L.word ? "write" : "event";
            out += "): last completed [" + kname(k_done) + "], first NOT completed [" +
                   kname(k_pend) + "]";
            std::snprintf(b, sizeof b, "; %llu marked launch(es) in view, %llu done after the "
                          "first pending%s", (unsigned long long)n_marked,
                          (unsigned long long)n_late, n_late ? " (OUT OF ORDER)" : "");
            out += b;
            if (k_pend || k_done) {
                const uint64_t c = k_pend ? k_pend : k_done;
                out += "\n        ops around it (* completed, ! not):";
                for (uint64_t k = std::max(from, c > 40 ? c - 40 : 1); k <= std::min(c + 8, L.n); ++k) {
                    const Op& o = L.ring[k % kRing];
                    if (o.serial != k) continue;
                    if (o.kind == 'K') {
                        const char* st = !o.mark ? "" : (k <= k_done && (!k_pend || k < k_pend)) ? "*" : "!";
                        std::snprintf(b, sizeof b, " [%llu%s K %s / %s]", (unsigned long long)k, st,
                                      o.op ? o.op : "?", o.kernel ? o.kernel : "?");
                    } else if (o.kind == 'W')
                        std::snprintf(b, sizeof b, " [%llu W %s's %llu]", (unsigned long long)k,
                                      name(o.rec_stream).c_str(), (unsigned long long)o.rec_serial);
                    else if (o.kind == 'R')
                        std::snprintf(b, sizeof b, " [%llu R]", (unsigned long long)k);
                    else
                        std::snprintf(b, sizeof b, " [%llu %c %lld B]", (unsigned long long)k, o.kind,
                                      (long long)o.bytes);
                    out += b;
                }
            }
            continue;
        }
        out += "\n        ops after the last complete record:";
        list(f.first + 1, f.first + 8);
        if (f.second > f.first + 8) {
            out += "\n        ops up to the first pending record:";
            list(std::max(f.first + 9, f.second > 64 ? f.second - 64 : 1), f.second + 4);
        }
    }
    return out;
}
}  // namespace slog

unsigned hip_event_flags(unsigned flags) {
    return ((flags & RAD_EVENT_HOST_WAIT) && !env_on("BRILLIANCE_EVENT_SPIN", false))
               ? hipEventBlockingSync
               : hipEventDefault;
}

int n_event_create(RadEvent* out, unsigned flags) {
    if (!out) return RAD_E_INVAL;
    auto* e = new (std::nothrow) NEv;
    if (!e) return RAD_E_NOMEM;
    e->flags = flags;
    if (hipGetDevice(&e->dev) != hipSuccess) { clear_sticky(); e->dev = -1; }
    const hipError_t r = hipEventCreateWithFlags(&e->ev, hip_event_flags(flags));
    if (r != hipSuccess) {
        delete e;
        return hip_fail(r, "hipEventCreateWithFlags");
    }
    slog::ev_add(e);
    *out = (RadEvent)e;
    return RAD_OK;
}

void n_event_destroy(RadEvent ev) {
    auto* e = (NEv*)ev;
    if (!e) return;
    slog::ev_del(e);
    HIP_NOTE(hipEventDestroy(e->ev));
    delete e;
}

/* Remakes `e` on card `dev`. Waits already enqueued on the old event keep the runtime's own
 * reference to its marker, so destroying it here does not release them early. */
int event_move(NEv* e, int dev) {
    int cur = -1;
    HIP_TRY(hipGetDevice(&cur));
    if (cur != dev) HIP_TRY(hipSetDevice(dev));
    hipEvent_t fresh = nullptr;
    const hipError_t r = hipEventCreateWithFlags(&fresh, hip_event_flags(e->flags));
    if (cur != dev) HIP_NOTE(hipSetDevice(cur));
    if (r != hipSuccess) return hip_fail(r, "hipEventCreateWithFlags (moving an event to its stream's card)");
    HIP_NOTE(hipEventDestroy(e->ev));
    e->ev = fresh;
    e->dev = dev;
    return RAD_OK;
}

int n_event_record(RadEvent ev, RadStream s) {
    auto* e = (NEv*)ev;
    if (!e || !s) {
        device_set_error("rad_event_record: null %s", e ? "stream" : "event");
        return RAD_E_INVAL;
    }
    const hipError_t r = hipEventRecord(e->ev, (hipStream_t)s);
    if (r != hipSuccess) {
        clear_sticky();
        int sd = -1;
        if (hipStreamGetDevice((hipStream_t)s, &sd) != hipSuccess || sd < 0 || sd == e->dev)
            return hip_fail(r, "hipEventRecord");
        RAD_TRY(event_move(e, sd));
        HIP_TRY(hipEventRecord(e->ev, (hipStream_t)s));
    }
    const uint64_t k = slog::note(s, 'R', e, s, 0, 0);
    e->last_serial.store(k, std::memory_order_relaxed);
    e->last_stream.store(s, std::memory_order_release);
    return RAD_OK;
}

int n_event_wait(RadStream s, RadEvent ev) {
    auto* e = (NEv*)ev;
    if (!e || !s) {
        device_set_error("rad_event_wait: null %s", e ? "stream" : "event");
        return RAD_E_INVAL;
    }
    HIP_TRY(hipStreamWaitEvent((hipStream_t)s, e->ev, 0));
    (void)slog::note(s, 'W', e, e->last_stream.load(std::memory_order_acquire),
                     e->last_serial.load(std::memory_order_relaxed), 0);
    return RAD_OK;
}

int n_event_query(RadEvent ev) {
    auto* e = (NEv*)ev;
    if (!e) return RAD_E_INVAL;
    const hipError_t r = hipEventQuery(e->ev);
    if (r == hipSuccess) {
        stall::progress();
        return 1;
    }
    if (r == hipErrorNotReady) {
        clear_sticky();
        return 0;
    }
    return hip_fail(r, "hipEventQuery");
}

int n_event_sync(RadEvent ev) {
    auto* e = (NEv*)ev;
    if (!e) return RAD_E_INVAL;
    if (!(e->flags & RAD_EVENT_HOST_WAIT)) {
        /* The contract every backend keeps (device.h, rad_event_create_as). */
        device_set_error("rad_event_sync: the event was made without RAD_EVENT_HOST_WAIT");
        return RAD_E_INVAL;
    }
    stall::Wait w;
    HIP_TRY(hipEventSynchronize(e->ev));
    return RAD_OK;
}

int n_event_elapsed_ms(RadEvent a, RadEvent b, float* out) {
    auto* ea = (NEv*)a;
    auto* eb = (NEv*)b;
    if (!ea || !eb || !out) return RAD_E_INVAL;
    HIP_TRY(hipEventElapsedTime(out, ea->ev, eb->ev));
    return RAD_OK;
}

/* A HIP launch acquires at system scope already, so host writes made before it are visible to it;
 * there is nothing to widen. */
void n_host_wrote(void) {}

/* ------------------------------------------------------------------ copies and fills */

/* False, with the error set, when [p, p + bytes) starts in an elastic reservation and is not backed
 * to its end. Every copy and fill asks it for each end before HIP is called; it is defined with the
 * elastic memory below, which says why. */
bool vm_backed(const void* p, int64_t bytes, const char* what);
/* Whether `p` lies in an elastic reservation, backed or not. */
bool vm_elastic(const void* p);

/* COPIES AS KERNELS WHILE THE RUNTIME'S COPY ENGINES ARE ON (specdec-sdma, 2026-11-20).
 *
 * With BRILLIANCE_SDMA=1 hipMemcpyAsync goes to an SDMA engine, and on gfx1201 with HIP 7.16 a copy
 * issued behind hipEventRecord can run before the kernels ahead of it on the same stream have
 * finished (rocm-systems#12571's class). The engine's per-step metadata lives in single device
 * buffers that stream order alone protects: the step batch's staging slab (sched/batch.h, Set), a
 * plugin's derived buffers (BatchBuilder::run_derivations), the coupled drafter's draws. An upload
 * that overtakes the pass still reading the previous step's values changes that pass's result: a
 * drafter's acceptance read as 0 where the host counted 7 stopped the engine on its first requests.
 * So while the copy engines are on, every copy up to kKernelCopyMax bytes between memory the card
 * reaches (device memory, pinned or mapped host memory) runs as a kernel on the stream's compute
 * queue (win/device/hip_copy.hip), which the queue orders with the kernels around it; larger copies
 * -- the expert mover's uploads, what the copy engines are for -- and copies of pageable memory keep
 * the runtime's. With the copy engines off (the default) the runtime's copies are blit kernels on
 * the compute queue already, and nothing changes.
 *
 * BRILLIANCE_COPY_KERNELS: auto (default, as above), on (also with the copy engines off), off (never;
 * the runtime's copies only, for an A/B). */
constexpr int64_t kKernelCopyMax = 256 << 10;

bool copy_kernels() {
    static const bool on = [] {
        const char* v = std::getenv("BRILLIANCE_COPY_KERNELS");
        const char* pal = std::getenv("PAL_DISABLE_SDMA");   /* runtime_defaults has run by now */
        const bool sdma = !(pal && *pal && std::strcmp(pal, "0") != 0);
        const bool forced_on = v && (std::strcmp(v, "on") == 0 || std::strcmp(v, "1") == 0);
        const bool forced_off = v && (std::strcmp(v, "off") == 0 || std::strcmp(v, "0") == 0);
        const bool k = forced_on || (!forced_off && sdma);
        RAD_INFO("device: small copies and the step batch's ranged uploads run as %s (copy engines "
                 "%s; BRILLIANCE_COPY_KERNELS=%s)",
                 k ? "kernels on the stream's compute queue" : "the runtime's hipMemcpyAsync",
                 sdma ? "on" : "off", v && *v ? v : "auto");
        return k;
    }();
    return on;
}

/* Whether a kernel can read or write `p` where it stands: device memory, or pinned or mapped host
 * memory. Pageable memory (this backend's RAD_MEM_HOST, or memory HIP never registered) cannot. */
bool card_reaches(const void* p) {
    const int k = rec_kind(p);
    if (k == RAD_MEM_DEVICE || k == RAD_MEM_HOST_PINNED || k == RAD_MEM_HOST_MAPPED) return true;
    if (k == RAD_MEM_HOST) return false;
    hipPointerAttribute_t a{};
    if (hipPointerGetAttributes(&a, p) != hipSuccess) {
        clear_sticky();
        return false;
    }
    return a.type == hipMemoryTypeDevice || a.type == hipMemoryTypeHost;
}

int n_memcpy_async(void* dst, const void* src, int64_t bytes, RadStream s) {
    if (bytes == 0) return RAD_OK;
    if (!dst || !src || bytes < 0 || !s) {
        device_set_error("rad_memcpy_async(%p <- %p, %lld): null pointer, negative size or null stream",
                         dst, src, (long long)bytes);
        return RAD_E_INVAL;
    }
    if (!vm_backed(dst, bytes, "rad_memcpy_async") || !vm_backed(src, bytes, "rad_memcpy_async"))
        return RAD_E_INVAL;
    if (bytes <= kKernelCopyMax && copy_kernels() && card_reaches(dst) && card_reaches(src)) {
        if (brl_hip_copy(dst, src, (uint64_t)bytes, s) < 0)
            return hip_fail(hipGetLastError(), "brl_copy (a small copy's kernel)");
        (void)slog::note(s, 'C', nullptr, nullptr, 0, bytes);
        slog::note_copy(dst, bytes, s, 'C');
        return RAD_OK;
    }
    HIP_TRY(hipMemcpyAsync(dst, src, (size_t)bytes, hipMemcpyDefault, (hipStream_t)s));
    (void)slog::note(s, 'C', nullptr, nullptr, 0, bytes);
    slog::note_copy(dst, bytes, s, 'C');
    return RAD_OK;
}

int n_memcpy_2d_async(void* dst, int64_t dpitch, const void* src, int64_t spitch, int64_t width,
                      int64_t height, RadStream s) {
    if (width == 0 || height == 0) return RAD_OK;
    if (!dst || !src || !s || width < 0 || height < 0 || dpitch < width || spitch < width) {
        device_set_error("rad_memcpy_2d_async: null pointer or stream, or a pitch below the width");
        return RAD_E_INVAL;
    }
    if (!vm_backed(dst, (height - 1) * dpitch + width, "rad_memcpy_2d_async") ||
        !vm_backed(src, (height - 1) * spitch + width, "rad_memcpy_2d_async"))
        return RAD_E_INVAL;
    if ((height - 1) * dpitch + width <= kKernelCopyMax && (height - 1) * spitch + width <= kKernelCopyMax &&
        copy_kernels() && card_reaches(dst) && card_reaches(src)) {
        if (brl_hip_copy2d(dst, (uint64_t)dpitch, src, (uint64_t)spitch, (uint64_t)width,
                           (uint64_t)height, s) < 0)
            return hip_fail(hipGetLastError(), "brl_copy2d (a small pitched copy's kernel)");
        (void)slog::note(s, 'C', nullptr, nullptr, 0, width * height);
        slog::note_copy(dst, (height - 1) * dpitch + width, s, '2');
        return RAD_OK;
    }
    hipError_t e = hipMemcpy2DAsync(dst, (size_t)dpitch, src, (size_t)spitch, (size_t)width,
                                    (size_t)height, hipMemcpyDefault, (hipStream_t)s);
    /* IN ELASTIC MEMORY THE RUNTIME'S PITCHED COPY STOPS AT ONE MAPPING. It checks the rectangle's
     * extent, (height - 1) * pitch + width, against the mapping the address lies in
     * (validateMemoryObject, hip_memory.cpp: validateRegion on the sub-buffer), where the plain
     * copy checks against the whole reservation. So a rectangle longer than a mapping, 2 MiB in a
     * pool and 64 MiB in a paged allocation, is refused with hipErrorInvalidValue, however well it
     * is backed: a KV checkpoint's layer rows (kv.cpp) are one. Such a copy goes row by row
     * through the plain copy, on the same stream and so in the same order; rows that touch are
     * one copy. */
    if (e == hipErrorInvalidValue && (vm_elastic(dst) || vm_elastic(src))) {
        clear_sticky();
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            RAD_INFO("device: a pitched copy longer than one mapping of elastic memory goes row by "
                     "row (first: %lld rows of %s, %s apart); the runtime's own stops at a mapping",
                     (long long)height, humanb(width).c_str(),
                     humanb(vm_elastic(dst) ? dpitch : spitch).c_str());
        char* d = static_cast<char*>(dst);
        const char* f = static_cast<const char*>(src);
        e = hipSuccess;
        if (dpitch == width && spitch == width)
            e = hipMemcpyAsync(d, f, (size_t)(width * height), hipMemcpyDefault, (hipStream_t)s);
        else
            for (int64_t r = 0; r < height && e == hipSuccess; ++r)
                e = hipMemcpyAsync(d + r * dpitch, f + r * spitch, (size_t)width, hipMemcpyDefault,
                                   (hipStream_t)s);
    }
    if (e != hipSuccess) return hip_fail(e, "hipMemcpy2DAsync");
    (void)slog::note(s, 'C', nullptr, nullptr, 0, width * height);
    slog::note_copy(dst, (height - 1) * dpitch + width, s, '2');
    return RAD_OK;
}

/* PAGEABLE MEMORY IS REFUSED, as the AQL backend refuses it (device.h, contract note 1):
 * RAD_MEM_HOST is memory the device has never been shown. Kept identical so a path that works here
 * works on Linux. Checked against the record before HIP is asked, so the refusal does not depend on
 * what this runtime version does with such a pointer. */
int n_memset_async(void* dst, int value, int64_t bytes, RadStream s) {
    if (bytes == 0) return RAD_OK;
    if (!dst || bytes < 0 || !s) {
        device_set_error("rad_memset_async(%p, %lld): null pointer, negative size or null stream",
                         dst, (long long)bytes);
        return RAD_E_INVAL;
    }
    if (rec_kind(dst) == RAD_MEM_HOST) {
        device_set_error("rad_memset_async(%p, %lld): the destination is not device-accessible "
                         "-- RAD_MEM_HOST is pageable and the GPU has never seen it. Use "
                         "RAD_MEM_HOST_PINNED or RAD_MEM_HOST_MAPPED, or memset it on the host.",
                         dst, (long long)bytes);
        return RAD_E_UNSUPPORTED;
    }
    if (!vm_backed(dst, bytes, "rad_memset_async")) return RAD_E_INVAL;
    HIP_TRY(hipMemsetAsync(dst, value, (size_t)bytes, (hipStream_t)s));
    slog::note_copy(dst, bytes, s, 'M');
    return RAD_OK;
}

/* Whether the host may read `p` in place: this backend's host memory, or memory HIP says is the
 * host's. The ranged copy reads its range list on the host (below), so a list in device memory is
 * refused instead of faulting the process. */
bool host_readable(const void* p) {
    const int k = rec_kind(p);
    if (k == RAD_MEM_HOST_PINNED || k == RAD_MEM_HOST_MAPPED || k == RAD_MEM_HOST) return true;
    if (k == RAD_MEM_DEVICE) return false;
    hipPointerAttribute_t a{};
    if (hipPointerGetAttributes(&a, p) != hipSuccess) {
        clear_sticky();
        return true;   /* HIP never saw it: ordinary process memory */
    }
    /* Unregistered is ordinary process memory too; only the card's own memory is out of reach. */
    return a.type != hipMemoryTypeDevice && a.type != hipMemoryTypeArray;
}

/* THE RANGED COPY. While the copy engines are on (copy_kernels above), one kernel on the stream's
 * compute queue (win/device/hip_copy.hip), as on Linux (the core's aql_blit.hip, left out of this
 * build with the rest of the AQL backend): the step batch's staging slab is single and stream order
 * must hold for it. The kernel reads the range list when it runs, from the pinned memory beside the
 * source, which the contract allows.
 *
 * Otherwise hipMemcpyAsync calls on the same stream (blit kernels on the compute queue with the
 * copy engines off): the prefix, then each range in list order (adjacent ranges, contiguous at both
 * ends, merged into one call). Same stream, so every piece reads the source as it stands when the
 * stream reaches it, and a range written over the prefix lands after it, as the kernel's contract
 * has it. The list is read here, at the call: it stays unchanged from the call until the stream
 * passes the copy, and its callers write it on the host before the call (sched/batch.cpp,
 * kvstream/stage.cpp). The cost is a runtime call per merged range instead of one dispatch.
 *
 * The checks are the AQL backend's (aql.cpp, a_memcpy_ranges_async), which the host backend makes
 * too, so a caller bug fails the same way everywhere: a prefix not a multiple of 16 or a base off 16
 * bytes is refused, and so is pageable memory. Every range is checked before any copy is issued, so
 * a refused call enqueues nothing. */
int n_memcpy_ranges_async(void* dst, const void* src, int64_t prefix, const RadCopyRange* ranges,
                          int n, RadStream s) {
    if (!s) {
        device_set_error("rad_memcpy_ranges_async: null stream");
        return RAD_E_INVAL;
    }
    if (prefix < 0 || (prefix & 15) || n < 0 || (n > 0 && !ranges)) {
        device_set_error("rad_memcpy_ranges_async: prefix %lld (a multiple of 16), %d ranges",
                         (long long)prefix, n);
        return RAD_E_INVAL;
    }
    if (prefix == 0 && n == 0) return RAD_OK;
    if (!dst || !src || rec_kind(dst) == RAD_MEM_HOST || rec_kind(src) == RAD_MEM_HOST ||
        (n > 0 && (rec_kind(ranges) == RAD_MEM_HOST || !host_readable(ranges))) ||
        (((uintptr_t)dst | (uintptr_t)src) & 15)) {
        device_set_error("rad_memcpy_ranges_async(%p <- %p, ranges %p): every address must be one "
                         "the card reaches, the range list one the host reads too, and both bases "
                         "16-byte aligned", dst, src, (const void*)ranges);
        return RAD_E_UNSUPPORTED;
    }
    for (int i = 0; i < n; ++i) {
        const RadCopyRange& r = ranges[i];
        if (r.src < 0 || r.dst < 0 || r.bytes < 0 || ((r.src | r.dst | r.bytes) & 3)) {
            device_set_error("rad_memcpy_ranges_async: range %d (%lld -> %lld, %lld bytes) is not "
                             "three non-negative multiples of four", i, (long long)r.src,
                             (long long)r.dst, (long long)r.bytes);
            return RAD_E_INVAL;
        }
    }
    char* d = static_cast<char*>(dst);
    const char* f = static_cast<const char*>(src);
    /* With the rest of the checks, before anything is issued. */
    if (prefix > 0 && (!vm_backed(d, prefix, "rad_memcpy_ranges_async") ||
                       !vm_backed(f, prefix, "rad_memcpy_ranges_async")))
        return RAD_E_INVAL;
    for (int i = 0; i < n; ++i)
        if (ranges[i].bytes > 0 &&
            (!vm_backed(d + ranges[i].dst, ranges[i].bytes, "rad_memcpy_ranges_async") ||
             !vm_backed(f + ranges[i].src, ranges[i].bytes, "rad_memcpy_ranges_async")))
            return RAD_E_INVAL;
    hipStream_t hs = (hipStream_t)s;
    if (copy_kernels()) {
        if (brl_hip_copy_ranges(d, f, (uint64_t)prefix, ranges, n, s) < 0)
            return hip_fail(hipGetLastError(), "brl_copy_ranges (the ranged copy's kernel)");
        if (prefix > 0) slog::note_copy(d, prefix, s, 'G');
        for (int i = 0; i < n; ++i)
            if (ranges[i].bytes > 0) slog::note_copy(d + ranges[i].dst, ranges[i].bytes, s, 'G');
        (void)slog::note(s, 'G', nullptr, nullptr, 0, prefix);
        return RAD_OK;
    }
    if (prefix > 0) HIP_TRY(hipMemcpyAsync(d, f, (size_t)prefix, hipMemcpyDefault, hs));
    if (prefix > 0) slog::note_copy(d, prefix, s, 'G');
    int64_t cs = 0, cd = 0, cb = 0;   /* the merged range being gathered */
    for (int i = 0; i <= n; ++i) {
        if (i < n) {
            const RadCopyRange& r = ranges[i];
            if (r.bytes == 0) continue;
            if (cb > 0 && cs + cb == r.src && cd + cb == r.dst) {
                cb += r.bytes;
                continue;
            }
        }
        if (cb > 0) HIP_TRY(hipMemcpyAsync(d + cd, f + cs, (size_t)cb, hipMemcpyDefault, hs));
        if (cb > 0) slog::note_copy(d + cd, cb, s, 'G');
        if (i < n) {
            cs = ranges[i].src;
            cd = ranges[i].dst;
            cb = ranges[i].bytes;
        }
    }
    (void)slog::note(s, 'G', nullptr, nullptr, 0, prefix);
    return RAD_OK;
}

/* Copy lists segment by segment on the stream: this backend has no DMA queue of its own. The list
 * is read at the call, as the contract says. RAD_COPY_YIELD has nothing to yield to (no link hold
 * here), so it is accepted and has no effect. Checked whole before anything is issued. */
int n_memcpy_list_async(const RadCopySeg* segs, int n, unsigned flags, RadStream s) {
    (void)flags;
    if (!s || n < 0 || (n > 0 && !segs)) {
        device_set_error("rad_memcpy_list_async: %d segments, list %p, stream %p", n,
                         (const void*)segs, (void*)s);
        return RAD_E_INVAL;
    }
    for (int i = 0; i < n; ++i)
        if (segs[i].bytes < 0 || (segs[i].bytes > 0 && (!segs[i].src || !segs[i].dst))) {
            device_set_error("rad_memcpy_list_async: segment %d (%p <- %p, %lld bytes)", i,
                             segs[i].dst, segs[i].src, (long long)segs[i].bytes);
            return RAD_E_INVAL;
        }
    for (int i = 0; i < n; ++i)
        if (segs[i].bytes > 0 && (!vm_backed(segs[i].dst, segs[i].bytes, "rad_memcpy_list_async") ||
                                  !vm_backed(segs[i].src, segs[i].bytes, "rad_memcpy_list_async")))
            return RAD_E_INVAL;
    int64_t total = 0;
    for (int i = 0; i < n; ++i)
        if (segs[i].bytes > 0) {
            HIP_TRY(hipMemcpyAsync(segs[i].dst, segs[i].src, (size_t)segs[i].bytes,
                                   hipMemcpyDefault, (hipStream_t)s));
            slog::note_copy(segs[i].dst, segs[i].bytes, s, 'L');
            total += segs[i].bytes;
        }
    if (total > 0) (void)slog::note(s, 'L', nullptr, nullptr, 0, total);
    return RAD_OK;
}

/* What carries the copy lists, for the start-up report: never a DMA queue of the process's own. */
int n_dma_lane(int device, char* out, int cap) {
    (void)device;
    if (!out || cap <= 0) return RAD_E_INVAL;
    const char* sdma = std::getenv("PAL_DISABLE_SDMA");
    std::snprintf(out, (size_t)cap, "%s",
                  sdma && sdma[0] == '1'
                      ? "none: segment by segment with hipMemcpyAsync, on blit kernels "
                        "(PAL_DISABLE_SDMA=1, rocm-systems#12571)"
                      : "none: segment by segment with hipMemcpyAsync, on the runtime's copy engines");
    return RAD_OK;
}

/* ------------------------------------------------------------------ elastic memory
 *
 * HIP's virtual memory API has every piece of the contract (device.h): reserve address space, back
 * a range, unback it, grant access. It is ON when the card supports it and exactly one device is
 * visible, and OFF where it cannot be trusted: with more than one device visible, hipMemMap commits
 * the whole reservation on this runtime (rocm-systems#13232), which would turn every elastic pool
 * into one sized whole that also claims its address space's worth of memory. Off, the entries
 * answer granularity 0 and pools are sized once (a supported configuration, device.h), and the
 * paging layer maps nothing in chunks. BRILLIANCE_VMM=0 forces it off and =1 forces it on; unset or
 * "auto" follows the device count (see vm_enabled below).
 *
 * What this backend does not have: host-backed mappings and their swap (vmem_commit_host,
 * vmem_swap), so parking keeps no host copies and a parked chunk goes to the disk tier or back to
 * its container; and vmem_commit_units and vmem_trim, which are optional.
 *
 * Each mapping is its own record (start -> handle, bytes), because the contract hands vmem_decommit
 * a run of whole mappings to unmap together (VMem::decommit, paging.cpp's chunks): each is unmapped
 * by its own size and its handle released, so none leaks. A range that does not start on a mapping
 * or does not end on one is refused before anything is unmapped. */
std::mutex                  g_vm_mu;
std::atomic<int>            g_vm_state{-1};    /* -1 unknown, 0 off, 1 on */
std::atomic<int64_t>        g_vm_gran{0};
struct VmRes { void* raw; size_t raw_bytes; int64_t bytes; int device; };
struct VmMap { hipMemGenericAllocationHandle_t h; int64_t bytes; };
std::map<uintptr_t, VmRes>  g_vm_res;          /* by the aligned base handed out */
std::map<uintptr_t, VmMap>  g_vm_maps;         /* by mapping start */
/* The two maps change under g_vm_mu AND g_vm_look_mu; g_vm_look_mu is held only for the insert or
 * erase itself. So a reader holding either sees them whole, and vm_backed, which every copy calls,
 * takes g_vm_look_mu alone and never waits behind an unmap (a device sync, under g_vm_mu). */
std::mutex                  g_vm_look_mu;
std::atomic<int>            g_vm_res_n{0};     /* reservations held; 0: no copy needs the lookup */

/* A COPY INTO AN ELASTIC RANGE THAT IS NOT BACKED IS REFUSED HERE, because the runtime does not
 * refuse it. An address in a reservation with no mapping under it resolves to no memory object, so
 * HIP takes it for pageable host memory and stages the copy on the CPU; the reservation is address
 * space the process has reserved and not committed, and the staging copy faults (0xC0000005 inside
 * amdhip64 on a runtime thread: device_test's host-fill case, 2026-11-16 and 11-17). With several
 * cards visible the runtime commits host memory behind the whole reservation (rocm-systems#13232),
 * and the same copy would land there silently. A copy that starts in a mapping and runs past its
 * end into a hole passes HIP's check too, which looks at the first mapping only.
 *
 * So: a range that starts inside a reservation must be covered by mappings to its end. Checked when
 * the copy is issued, not when the stream runs it: a caller that unbacks the range in between is
 * not caught. The cost is one atomic load in a process with no reservation, otherwise a lock held
 * for two map lookups. */
bool vm_backed(const void* p, int64_t bytes, const char* what) {
    if (!p || bytes <= 0 || g_vm_res_n.load(std::memory_order_relaxed) == 0) return true;
    const uintptr_t a = (uintptr_t)p, b = a + (uintptr_t)bytes;
    uintptr_t at = a;
    {
        std::lock_guard<std::mutex> lk(g_vm_look_mu);
        auto r = g_vm_res.upper_bound(a);
        if (r == g_vm_res.begin()) return true;
        --r;
        if (a >= r->first + (uintptr_t)r->second.bytes) return true;   /* not elastic memory */
        auto m = g_vm_maps.upper_bound(a);
        if (m != g_vm_maps.begin()) {
            /* From the mapping that starts at or before `a`, while each one continues the run. */
            for (--m; m != g_vm_maps.end() && at < b && m->first <= at; ++m) {
                const uintptr_t end = m->first + (uintptr_t)m->second.bytes;
                if (end <= at) break;
                at = end;
            }
        }
    }
    if (at >= b) return true;
    device_set_error("%s(%p, %s): elastic memory that is not backed at %p (nothing is mapped "
                     "there, or it was parked); the copy is refused", what, p, humanb(bytes).c_str(),
                     (void*)at);
    return false;
}

bool vm_elastic(const void* p) {
    if (!p || g_vm_res_n.load(std::memory_order_relaxed) == 0) return false;
    const uintptr_t a = (uintptr_t)p;
    std::lock_guard<std::mutex> lk(g_vm_look_mu);
    auto r = g_vm_res.upper_bound(a);
    return r != g_vm_res.begin() && a < (--r)->first + (uintptr_t)r->second.bytes;
}

hipMemAllocationProp vm_prop(int device) {
    hipMemAllocationProp p{};
    p.type = hipMemAllocationTypePinned;
    p.location.type = hipMemLocationTypeDevice;
    p.location.id = device;
    return p;
}

bool vm_enabled() {
    const int st = g_vm_state.load();
    if (st >= 0) return st == 1;
    int n = 0, dev = 0, sup = 0;
    /* A failed query is an answer here (no cards, no VMM), so its sticky error is cleared. */
    if (hipGetDeviceCount(&n) != hipSuccess) clear_sticky();
    if (hipGetDevice(&dev) != hipSuccess) clear_sticky();
    bool supported = false;
    if (hipDeviceGetAttribute(&sup, hipDeviceAttributeVirtualMemoryManagementSupported, dev) == hipSuccess)
        supported = sup != 0;
    else
        clear_sticky();
    const char* v = std::getenv("BRILLIANCE_VMM");
    const bool forced = v && *v && _stricmp(v, "auto") != 0;
    /* ON with exactly one device visible, OFF with more, unless BRILLIANCE_VMM says otherwise.
     *
     * One device: qualified on driver 32.0.32015.12012 / HIP 7.16 (2026-11-17, device_test on one
     * R9700: every remap and disk-tier parking case, and 1,000 cycles of granules remapped at
     * random). The "wrong bytes from multi-chunk reads" of 2026-11-16 that had it off were refused
     * copies: the commit mapped without granting access, and PAL refuses every copy into such a
     * mapping (n_vmem_commit grants it now). The crash that followed was a copy into an unmapped
     * range, which vm_backed refuses.
     *
     * More than one: the first mapping of a reservation commits its whole size as host memory
     * (rocm-systems#13232, open), and a peer access grant crashed in hipMemSetAccess (Brilliance
     * 0.2.2, two cards). Not qualified; BRILLIANCE_VMM=1 turns it on there for that work only. */
    const bool on = supported && (forced ? env_on("BRILLIANCE_VMM", false) : n == 1);
    if (!supported)
        RAD_INFO("device: HIP virtual memory is not supported here; pools are sized whole");
    else if (!on && forced)
        RAD_INFO("device: HIP virtual memory is off (BRILLIANCE_VMM=%s); pools are sized whole", v);
    else if (!on)
        RAD_INFO("device: HIP virtual memory is off with %d devices visible (the first mapping commits "
                 "its whole reservation, rocm-systems#13232); pools are sized whole. It is on with one "
                 "device visible", n);
    else if (n != 1)
        RAD_WARN("device: HIP virtual memory is on (BRILLIANCE_VMM=%s) with %d devices visible, which "
                 "is not qualified: the first mapping commits its whole reservation "
                 "(rocm-systems#13232)", v, n);
    else
        RAD_INFO("device: HIP virtual memory is on (%s)", forced ? "BRILLIANCE_VMM" : "one device visible");
    int expect = -1;
    g_vm_state.compare_exchange_strong(expect, on ? 1 : 0);
    return g_vm_state.load() == 1;
}

int64_t n_vmem_granularity(void) {
    if (!vm_enabled()) return 0;
    if (const int64_t g = g_vm_gran.load(std::memory_order_relaxed)) return g;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) clear_sticky();
    hipMemAllocationProp p = vm_prop(dev);
    size_t g = 0;
    if (hipMemGetAllocationGranularity(&g, &p, hipMemAllocationGranularityRecommended) != hipSuccess) {
        clear_sticky();   /* no granule: pools are sized whole, a supported configuration */
        return 0;
    }
    if (!g) return 0;
    /* Never finer than the pool alignment: the commit is per granule and a pool is gigabytes. */
    const int64_t gr = (int64_t)std::max<size_t>(g, (size_t)RAD_ALIGN_POOL);
    g_vm_gran.store(gr, std::memory_order_relaxed);
    return gr;
}

/* The reservation holding [p, p + bytes), or null. Under g_vm_mu. */
const VmRes* vm_res_of(const void* p, int64_t bytes) {
    const uintptr_t a = (uintptr_t)p;
    auto it = g_vm_res.upper_bound(a);
    if (it == g_vm_res.begin()) return nullptr;
    --it;
    return a + (uintptr_t)bytes <= it->first + (uintptr_t)it->second.bytes ? &it->second : nullptr;
}

/* THE ALIGNMENT IS NOT HIP'S TO KEEP ON WINDOWS: hipMemAddressReserve returns ranges aligned only to
 * its own granule whatever alignment it is asked for (Brilliance 0.2.2's device_test saw a 2 MiB
 * request come back 384 KiB off). So a reservation that comes back misaligned is retaken one
 * alignment larger and the aligned range inside it is handed out; the real one is remembered for
 * the release. */
void* n_vmem_reserve(int64_t bytes, int64_t align) {
    if (bytes <= 0) {
        device_set_error("vmem_reserve(%lld): size must be positive", (long long)bytes);
        return nullptr;
    }
    if (align <= 0) align = n_vmem_granularity();
    if (align <= 0) {
        device_set_error("vmem_reserve: HIP virtual memory is off");
        return nullptr;
    }
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) clear_sticky();
    void* p = nullptr;
    hipError_t e = hipMemAddressReserve(&p, (size_t)bytes, (size_t)align, nullptr, 0);
    if (e != hipSuccess) {
        hip_fail(e, "hipMemAddressReserve");
        return nullptr;
    }
    void* raw = p;
    size_t raw_bytes = (size_t)bytes;
    if ((uintptr_t)p % (uintptr_t)align != 0) {
        HIP_NOTE(hipMemAddressFree(p, (size_t)bytes));
        raw_bytes = (size_t)(bytes + align);
        e = hipMemAddressReserve(&raw, raw_bytes, (size_t)align, nullptr, 0);
        if (e != hipSuccess) {
            hip_fail(e, "hipMemAddressReserve (aligned retry)");
            return nullptr;
        }
        p = (void*)(((uintptr_t)raw + (uintptr_t)align - 1) / (uintptr_t)align * (uintptr_t)align);
    }
    std::lock_guard<std::mutex> lk(g_vm_mu);
    std::lock_guard<std::mutex> look(g_vm_look_mu);
    g_vm_res[(uintptr_t)p] = VmRes{ raw, raw_bytes, bytes, dev };
    g_vm_res_n.store((int)g_vm_res.size(), std::memory_order_relaxed);
    return p;
}

/* Unmaps and releases every mapping in [a, b). Under g_vm_mu. Returns the first failure; a
 * mapping whose unmap failed stays recorded, since it is still backed. */
int vm_unmap_span(uintptr_t a, uintptr_t b) {
    auto it = g_vm_maps.lower_bound(a);
    while (it != g_vm_maps.end() && it->first < b) {
        const hipError_t e = hipMemUnmap((void*)it->first, (size_t)it->second.bytes);
        if (e != hipSuccess) return hip_fail(e, "hipMemUnmap");
        const hipError_t r = hipMemRelease(it->second.h);
        if (r != hipSuccess) {
            /* Unmapped already: the memory is gone from the address space either way. */
            clear_sticky();
            RAD_WARN("device: hipMemRelease after an unmap at %p: %s", (void*)it->first,
                     hipGetErrorString(r));
        }
        std::lock_guard<std::mutex> look(g_vm_look_mu);
        it = g_vm_maps.erase(it);
    }
    return RAD_OK;
}

void n_vmem_release(void* base, int64_t bytes) {
    if (!base) return;
    std::lock_guard<std::mutex> lk(g_vm_mu);
    auto it = g_vm_res.find((uintptr_t)base);
    if (it == g_vm_res.end()) {
        RAD_WARN("vmem_release(%p): not a reservation of this backend", base);
        return;
    }
    const VmRes r = it->second;
    /* The contract hands over an empty range; one that is not would leak its backing with the
     * address space, so what is still mapped goes first, said once. */
    auto m = g_vm_maps.lower_bound((uintptr_t)base);
    if (m != g_vm_maps.end() && m->first < (uintptr_t)base + (uintptr_t)r.bytes) {
        RAD_WARN("vmem_release(%p, %s): still backed; unmapped first", base, humanb(bytes).c_str());
        if (vm_unmap_span((uintptr_t)base, (uintptr_t)base + (uintptr_t)r.bytes) < 0) {
            RAD_WARN("vmem_release(%p): an unmap failed, so the address space is kept: %s", base,
                     rad_dev_last_error());
            return;
        }
    }
    {
        std::lock_guard<std::mutex> look(g_vm_look_mu);
        g_vm_res.erase(it);
        g_vm_res_n.store((int)g_vm_res.size(), std::memory_order_relaxed);
    }
    HIP_NOTE(hipMemAddressFree(r.raw, r.raw_bytes));
}

/* Backs [addr, addr + bytes) as ONE mapping of new physical memory on the reservation's card. A
 * commit is an allocation like any other, so it passes the same room check first -- the AQL backend
 * checks before every commit too -- or growing an elastic pool past the WDDM budget would page it,
 * silently, a granule at a time.
 *
 * Returns 1: the contract's "recycled memory that may hold what its last mapping left". Neither HIP
 * nor PAL promises that hipMemCreate's memory reads zero (PAL hands out memory it freed for this
 * process before), so 0 would be a promise this backend cannot keep. 1 costs the caller a clear it
 * would have needed anyway (paging.cpp zeroes such chunks; a pool's contents are written before
 * they are read). */
int n_vmem_commit(void* addr, int64_t bytes) {
    if (!addr || bytes <= 0) return RAD_E_INVAL;
    int dev = -1;
    {
        std::lock_guard<std::mutex> lk(g_vm_mu);
        if (const VmRes* r = vm_res_of(addr, bytes)) dev = r->device;
        auto m = g_vm_maps.upper_bound((uintptr_t)addr + (uintptr_t)bytes - 1);
        if (m != g_vm_maps.begin()) {
            --m;
            if (m->first + (uintptr_t)m->second.bytes > (uintptr_t)addr) {
                device_set_error("vmem_commit(%p, %s): overlaps the mapping at %p", addr,
                                 humanb(bytes).c_str(), (void*)m->first);
                return RAD_E_INVAL;
            }
        }
    }
    if (dev < 0) {
        device_set_error("vmem_commit(%p, %s): not inside a reservation of this backend", addr,
                         humanb(bytes).c_str());
        return RAD_E_INVAL;
    }
    if (!card_has_room(dev, bytes, "vmem commit")) return RAD_E_NOMEM;
    hipMemAllocationProp p = vm_prop(dev);
    hipMemGenericAllocationHandle_t h{};
    HIP_TRY(hipMemCreate(&h, (size_t)bytes, &p, 0));
    const hipError_t e = hipMemMap(addr, (size_t)bytes, 0, h, 0);
    if (e != hipSuccess) {
        HIP_NOTE(hipMemRelease(h));
        return hip_fail(e, "hipMemMap");
    }
    /* Access with the mapping, as the AQL backend grants it (a_vmem_commit), since callers commit and
     * then copy with no vmem_protect between (device_test's remap case, the paging layer's chunks
     * before its own protect). PAL refuses every copy into a mapping without access, hipErrorUnknown
     * from ihipMemcpy_validate_memory, which is what read as "multi-chunk reads came back wrong":
     * the copies never ran. On PAL access is a flag on each mapping (paldevice.cpp SetAccess), so a
     * chunk's own grant does not meet the adjacent-granule refusal device.h describes. Only the owning
     * card: a peer grant crashed in hipMemSetAccess with two cards (Brilliance 0.2.2). */
    hipMemAccessDesc acc{};
    acc.location.type = hipMemLocationTypeDevice;
    acc.location.id = dev;
    acc.flags = hipMemAccessFlagsProtReadWrite;
    const hipError_t ae = hipMemSetAccess(addr, (size_t)bytes, &acc, 1);
    if (ae != hipSuccess) {
        HIP_NOTE(hipMemUnmap(addr, (size_t)bytes));
        HIP_NOTE(hipMemRelease(h));
        return hip_fail(ae, "hipMemSetAccess");
    }
    std::lock_guard<std::mutex> lk(g_vm_mu);
    std::lock_guard<std::mutex> look(g_vm_look_mu);
    g_vm_maps[(uintptr_t)addr] = VmMap{ h, bytes };
    return 1;
}

int n_vmem_decommit(void* addr, int64_t bytes) {
    if (!addr || bytes <= 0) return RAD_E_INVAL;
    const uintptr_t a = (uintptr_t)addr, b = a + (uintptr_t)bytes;
    std::lock_guard<std::mutex> lk(g_vm_mu);
    /* Exactly a run of whole mappings, or nothing is unmapped. */
    uintptr_t at = a;
    for (auto it = g_vm_maps.find(a); it != g_vm_maps.end() && it->first == at && at < b; ++it)
        at += (uintptr_t)it->second.bytes;
    if (at != b) {
        device_set_error("vmem_decommit(%p, %s): not a run of whole mappings of this backend "
                         "(they cover %s from its start)", addr, humanb(bytes).c_str(),
                         humanb((int64_t)(at - a)).c_str());
        return RAD_E_INVAL;
    }
    return vm_unmap_span(a, b);
}

int n_vmem_protect(void* addr, int64_t bytes) {
    if (!addr || bytes <= 0) return RAD_E_INVAL;
    int dev = -1;
    {
        std::lock_guard<std::mutex> lk(g_vm_mu);
        if (const VmRes* r = vm_res_of(addr, bytes)) dev = r->device;
    }
    if (dev < 0) {
        device_set_error("vmem_protect(%p, %s): not inside a reservation of this backend", addr,
                         humanb(bytes).c_str());
        return RAD_E_INVAL;
    }
    hipMemAccessDesc d{};
    d.location.type = hipMemLocationTypeDevice;
    d.location.id = dev;
    d.flags = hipMemAccessFlagsProtReadWrite;
    HIP_TRY(hipMemSetAccess(addr, (size_t)bytes, &d, 1));
    return RAD_OK;
}

/* ------------------------------------------------------------------ devices */

int n_count(void) {
    const int n = g_base->count();
    if (n <= 0) clear_sticky();
    return n;
}

int n_set(int device) {
    const int r = g_base->set(device);
    if (r < 0) clear_sticky();
    return r;
}

int n_integrated(int device) {
    const int r = g_base->integrated(device);
    clear_sticky();
    return r;
}

/* hip.cpp's props, then what it reads from Linux files read the Windows way. Its sysfs and
 * /proc/meminfo reads find nothing here and fall back by themselves; what they would have added:
 *
 *  - THE FREE FIGURE, clamped to DXGI's budget room. A process past its budget has no room, not
 *    "an unknown amount": the planner then sizes nothing more onto the card.
 *  - AN INTEGRATED GPU's free figure bounded by the RAM Windows has available, which its
 *    "device memory" is (hip.cpp bounds it by MemAvailable).
 *
 * It also finds the card's adapter, so vram_used can answer from here on (device.h). */
int n_props(int device, RadDeviceProps* out) {
    const int pr = g_base->props(device, out);
    clear_sticky();
    RAD_TRY(pr);
    int64_t budget = 0, usage = 0;
    if (!out->is_host_backend && dxgi_local(device, &budget, &usage, true) && budget > 0) {
        const int64_t room = budget > usage ? budget - usage : 0;
        if (room < out->vram_free) out->vram_free = room;
    }
    if (out->unified) {
        const int64_t avail = os::mem_info().avail;
        if (avail >= 0 && avail < out->vram_free) out->vram_free = avail;
    } else {
        check_driver(device);
    }
    return RAD_OK;
}

/* Every process's VRAM use on the card (d3dkmt_local_resident), never entering HIP; unsupported
 * until props has found the card's LUID, as the contract allows. */
int n_vram_used(int device, int64_t* out) {
    if (!out || device < 0 || device >= kMaxDev) return RAD_E_INVAL;
    const uint64_t luid = g_luid[device].load(std::memory_order_acquire);
    if (!luid) return RAD_E_UNSUPPORTED;
    const int64_t v = d3dkmt_local_resident(luid);
    if (v < 0) {
        device_set_error("vram_used(%d): the graphics kernel gave no segment statistics", device);
        return RAD_E_UNSUPPORTED;
    }
    *out = v;
    return RAD_OK;
}

/* WHETHER A CARD GRANTS FINE-GRAINED MEMORY, asked once a card. The peer all-reduce keeps the
 * scratch and flags a peer writes in fine-grained memory and reads them with no acquire fence,
 * which is sound there and nowhere else; whether PAL grants it on an R9700 under WDDM is not
 * measured. A card that refuses has no use for peer access. */
bool fine_grained_ok(int device) {
    static std::atomic<int> known[kMaxDev];   /* 0 unasked, 1 granted, -1 refused */
    if (device < 0 || device >= kMaxDev) return false;
    if (const int k = known[device].load()) return k > 0;
    int prev = 0;
    if (hipGetDevice(&prev) != hipSuccess || hipSetDevice(device) != hipSuccess) {
        clear_sticky();
        return false;   /* unknown is not granted; not latched, the next caller asks again */
    }
    void* p = nullptr;
    const hipError_t e = hipExtMallocWithFlags(&p, 64 << 10, hipDeviceMallocFinegrained);
    if (e == hipSuccess) HIP_NOTE(hipFree(p));
    HIP_NOTE(hipSetDevice(prev));
    clear_sticky();
    if (e != hipSuccess)
        RAD_WARN("device: card %d refuses fine-grained memory: %s (%s). Counted as having no peer "
                 "access, so collectives use host memory", device, hipGetErrorString(e),
                 hipGetErrorName(e));
    known[device].store(e == hipSuccess ? 1 : -1);
    return e == hipSuccess;
}

/* PAL'S YES IS NOT THE LAST WORD ON WINDOWS. hipDeviceCanAccessPeer reports what PAL's multi-GPU
 * query says the pair could do; whether hipDeviceEnablePeerAccess then succeeds under WDDM is a
 * separate question (Brilliance 0.2.2 saw PAL say yes and WDDM refuse), and the engine picks peer
 * rows or host-memory rows from this answer alone. So a yes is tried: fine-grained memory on both
 * cards, then an enable, undone at once when this probe made it, because the contract is "asked,
 * not enabled" (device.h) and the library enables the pair for itself. A refusal anywhere is no
 * peer access. This backend claims no direct path it has not seen enabled. */
int n_peer_access(int self, int peer) {
    const int r = g_base->peer_access(self, peer);
    clear_sticky();
    if (r != 1 || self == peer) return r;
    if (!fine_grained_ok(self) || !fine_grained_ok(peer)) return 0;
    int prev = 0;
    HIP_TRY(hipGetDevice(&prev));
    HIP_TRY(hipSetDevice(self));
    const hipError_t e = hipDeviceEnablePeerAccess(peer, 0);
    if (e == hipSuccess) HIP_NOTE(hipDeviceDisablePeerAccess(peer));
    /* A failed restore leaves this thread on the wrong card; hip.cpp's d_enable_peer says so the
     * same way. */
    HIP_NOTE(hipSetDevice(prev));
    clear_sticky();
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
        RAD_WARN("device: the driver reports card %d can reach card %d's memory, but enabling it "
                 "failed: %s (%s). Counted as no peer access", self, peer, hipGetErrorString(e),
                 hipGetErrorName(e));
        return 0;
    }
    return 1;
}

/* hip.cpp's enable answers RAD_OK on hipErrorPeerAccessAlreadyEnabled and leaves that error behind
 * in the runtime, where the next launch on this thread would read it as its own. */
int n_enable_peer(int self, int peer) {
    const int r = g_base->enable_peer(self, peer);
    clear_sticky();
    return r;
}

int n_pci_address(int device, char* out, int cap) {
    const int r = g_base->pci_address(device, out, cap);
    if (r < 0) clear_sticky();
    return r;
}

int n_runtime_version(char* out, int cap) {
    const int r = g_base->runtime_version(out, cap);
    if (r < 0) clear_sticky();
    return r;
}

/* An integrated GPU beside discrete cards is named once: HIP shows it to the process unless
 * HIP_VISIBLE_DEVICES hides it, and visible it costs twice even though the engine never runs on it
 * (engine_bringup.cpp skips integrated devices beside discrete ones). Every portable pin is mapped
 * onto it too, which divides the pin limit by one more device, and with it present the elastic
 * memory rule (more than one device) turns VMM off. TheRock#6224: beside a discrete card it can also
 * abort a hipFree. */
void warn_visible_igpu(int n) {
    int igpu = -1, discrete = 0;
    for (int d = 0; d < n; ++d) {
        int v = 0;
        if (hipDeviceGetAttribute(&v, hipDeviceAttributeIntegrated, d) != hipSuccess) {
            clear_sticky();
            continue;
        }
        if (v) { if (igpu < 0) igpu = d; }
        else ++discrete;
    }
    if (igpu >= 0 && discrete > 0)
        RAD_WARN("device: HIP shows integrated GPU %d beside %d discrete card(s). The engine does "
                 "not run on it, but it divides the pinned-memory limit, turns elastic memory off "
                 "(rocm-systems#13232) and can abort a free (TheRock#6224). Hide it with "
                 "HIP_VISIBLE_DEVICES set to the discrete cards' HIP indices",
                 igpu, discrete);
}

}  /* namespace */

/* rad_dev_submit on this backend (device.h): an event recorded on `s` makes HIP submit what it has
 * queued there -- the cheapest call that does (Brilliance 0.2.2's probe: about 15 us, against 28
 * for hipStreamQuery). One event a thread, never waited on, destroyed with its thread. */
int64_t hipnative_pin_limit() {
    std::lock_guard<std::mutex> lk(g_pin_mu);
    return pin_limit_locked().limit;
}

/* rad_dev_note_launch on this backend (device.h): the launch into its stream's op log, only when
 * BRILLIANCE_SLOG_LAUNCHES is set -- a lock and a few stores per launch, plus the completion mark
 * behind it (slog::mark_mode, BRILLIANCE_SLOG_MARK), for hunting a stall. Off, one test of a
 * static. Always, the stall watch's progress beacon (beacon::on_launch): a thread-local count per
 * op, and a lock and one hipStreamWriteValue64 every BRILLIANCE_STALL_BEACONS-th. */
void hipnative_note_launch(RadStream s, const char* op, const char* kernel) {
    static const bool on = env_on("BRILLIANCE_SLOG_LAUNCHES", false);
    if (on) (void)slog::note_launch(s, op, kernel);
    beacon::on_launch((hipStream_t)s);   /* the stall watch's progress beacon, every 64th op */
}

void hipnative_submit(RadStream s) {
    struct Ev {
        hipEvent_t ev = nullptr;
        ~Ev() { if (ev) (void)hipEventDestroy(ev); }
    };
    static thread_local Ev e;
    if (!e.ev && hipEventCreateWithFlags(&e.ev, hipEventDisableTiming) != hipSuccess) {
        e.ev = nullptr;
        clear_sticky();
        return;
    }
    if (hipEventRecord(e.ev, (hipStream_t)s) != hipSuccess) clear_sticky();
}

/* The backend Windows runs in place of AQL (core/device/dispatch.cpp, select_backend). Null when
 * the machine shows no device, so a HIP build still runs the host backend there. */
const DeviceBackend* hipnative_backend_if_present(void) {
    static const DeviceBackend* const made = []() -> const DeviceBackend* {
        runtime_defaults();
        g_base = hip_backend_if_present();
        if (!g_base) return nullptr;
        g_nat = *g_base;
        g_nat.name = "hipnative";

        g_nat.count = n_count;
        g_nat.set = n_set;
        g_nat.props = n_props;
        g_nat.enable_peer = n_enable_peer;
        g_nat.alloc = n_alloc;
        g_nat.free = n_free;
        g_nat.host_ptr = n_host_ptr;
        g_nat.device_ptr = n_device_ptr;

        g_nat.stream_create = n_stream_create;
        g_nat.stream_destroy = n_stream_destroy;
        g_nat.stream_sync = n_stream_sync;
        g_nat.event_create = n_event_create;
        g_nat.event_destroy = n_event_destroy;
        g_nat.event_record = n_event_record;
        g_nat.event_wait = n_event_wait;
        g_nat.event_query = n_event_query;
        g_nat.event_sync = n_event_sync;
        g_nat.event_elapsed_ms = n_event_elapsed_ms;
        g_nat.host_wrote = n_host_wrote;

        g_nat.memcpy_async = n_memcpy_async;
        g_nat.memcpy_2d_async = n_memcpy_2d_async;
        g_nat.memset_async = n_memset_async;
        g_nat.memcpy_ranges_async = n_memcpy_ranges_async;
        g_nat.memcpy_list_async = n_memcpy_list_async;
        g_nat.dma_lane = n_dma_lane;

        g_nat.vmem_granularity = n_vmem_granularity;
        g_nat.vmem_reserve = n_vmem_reserve;
        g_nat.vmem_release = n_vmem_release;
        g_nat.vmem_commit = n_vmem_commit;
        g_nat.vmem_decommit = n_vmem_decommit;
        g_nat.vmem_protect = n_vmem_protect;

        g_nat.vram_used = n_vram_used;
        g_nat.integrated = n_integrated;
        g_nat.peer_access = n_peer_access;
        g_nat.pci_address = n_pci_address;
        g_nat.runtime_version = n_runtime_version;

        /* PLUGIN DEVICE CODE: no count, so the loader takes every library's code as present
         * (dev_code_mark 0, dev_code_runs 1). On Linux the executable exports the fat-binary
         * registration hooks and counts every plugin's code objects (aql_code.cpp); a Windows DLL
         * registers its code with amdhip64 directly, so this process never sees a registration.
         * A library whose code does not run on the card fails at its first launch instead, by name
         * (hipErrorInvalidDeviceFunction or hipErrorNoBinaryForGpu, read by the launcher). Faking a
         * count would claim a check this backend does not make. */
        g_nat.code_mark = nullptr;
        g_nat.code_runs = nullptr;

        /* Null, each because the backend has no such thing (header comment): recorded passes,
         * link hold and gate, gates, expert prefetch, uncached memory, DMA event,
         * host-backed mappings and swaps, kept queues and kept memory, launch-ahead control.
         * Not stuck_ms: HIP hides its queues, so the stall watch above (n_stuck_ms) times a stall
         * from what the host sees, and /health answers 503 "stuck" once nothing has completed for
         * a minute while a wait is blocked (kStuckMs, core/server/admin.cpp). It is reported,
         * never fatal: scheduler.stuck-exit does not end the process on this backend. */
        g_nat.tape_begin = nullptr;
        g_nat.tape_mark = nullptr;
        g_nat.tape_pause = nullptr;
        g_nat.tape_end = nullptr;
        g_nat.tape_len = nullptr;
        g_nat.tape_play = nullptr;
        g_nat.tape_diff = nullptr;
        g_nat.tape_destroy = nullptr;
        g_nat.tape_record = nullptr;
        g_nat.tape_set_overlap = nullptr;
        g_nat.link_hold = nullptr;
        g_nat.link_gate = nullptr;
        g_nat.xp_supported = nullptr;
        g_nat.xp_stream_create = nullptr;
        g_nat.ride = nullptr;
        g_nat.ride_wait = nullptr;
        g_nat.ride_release = nullptr;
        g_nat.xp_list = nullptr;
        g_nat.xp_behind = nullptr;
        g_nat.alloc_uncached = nullptr;
        g_nat.free_uncached = nullptr;
        g_nat.stream_gate = nullptr;
        g_nat.gate_open = nullptr;
        g_nat.gate_room = nullptr;
        g_nat.event_dma = nullptr;
        g_nat.map_file = n_map_file;   /* a file view registered with HIP, above */
        g_nat.stuck_ms = n_stuck_ms;   /* the stall watch, above: reported, never fatal here */
        g_nat.vmem_commit_host = nullptr;
        g_nat.vmem_swap = nullptr;
        g_nat.vmem_commit_units = nullptr;
        g_nat.vmem_trim = nullptr;
        g_nat.peer_launch = nullptr;
        g_nat.cards_claim = nullptr;
        g_nat.stream_counts = nullptr;
        g_nat.stream_trim = nullptr;
        g_nat.coll_fault_poll = n_coll_fault_poll;   /* libr4d's fault records, above */
        g_nat.coll_counters = n_coll_counters;

        warn_visible_igpu(n_count());
        return &g_nat;
    }();
    return made;
}

/* ------------------------------------------------------------------ the AQL test hooks
 *
 * aql.cpp defines these, and a HIP build's device tests call them (tests/device_test.cpp,
 * tests/host_kargs_main.h); the Windows build leaves aql.cpp out, so they are answered here with
 * what is true of this backend. */

/* No kernel argument rings to place: HIP owns kernel arguments here. Refused, so `--host-kargs`
 * cannot claim to test a placement this backend does not have. */
int aql_force_host_kargs(void) {
    device_set_error("aql_force_host_kargs: the native HIP backend has no kernel argument rings");
    return RAD_E_UNSUPPORTED;
}

namespace aql {
/* No ride tokens on the native HIP backend (rad_ride is unsupported), so none is reusable. */
bool ride_reusable(void*) { return false; }
/* The chained-launch and argument-ring probes (upstream 8f2e511f, 946a6f9e): no chained launches,
 * no prefetch stream and no argument ring here, so refused as aql.cpp's own non-HIP build refuses
 * them. The device tests that call them skip first on any backend but "aql". */
int chain_probe(void*, unsigned*, unsigned*, unsigned, unsigned) { return RAD_E_UNSUPPORTED; }
int chain_marks(void*, unsigned*, unsigned) { return RAD_E_UNSUPPORTED; }
int chain_asks(void*, const void*, void*, uint64_t, unsigned*, unsigned) { return RAD_E_UNSUPPORTED; }
}  /* namespace aql */

}  /* namespace rad */

#endif /* RAD_HAVE_HIP && _WIN32 */
