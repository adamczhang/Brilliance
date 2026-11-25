/* gpu_win32.cpp -- Brilliance: os::gpu_samples (win/os/os.h), what the driver reports of each
 * hardware GPU, for the dashboard's per-card figures (core/fleet/fleet.cpp, card_telemetry).
 *
 * WHERE THE FIGURES COME FROM. Linux reads the amdgpu driver's sysfs: gpu_busy_percent, hwmon power
 * and the junction temperature, all three the card's own power-management firmware (SMU) metrics.
 * Windows has no sysfs. Two sources, both readable without admin rights:
 *
 * - AMD's ADL (AMD Display Library), whose runtime atiadlxx.dll the AMD driver installs in
 *   System32. Its performance-metrics log (ADL2_New_QueryPMLogData_Get) is the same SMU metrics
 *   Linux reads: board power in watts, GFX activity in percent, and the hotspot (junction)
 *   temperature. It is the source for those three on an AMD card. ADL's SDK headers are MIT
 *   (win/licenses/adl-sdk.LICENSE); the few declarations it needs are below, and nothing of AMD's
 *   is shipped -- the DLL is the driver's, loaded at run time from System32 only (never from the
 *   application directory or PATH, so no planted copy is loaded), and a machine without it simply
 *   reports less.
 *   AMD's newer ADLX is NOT used: its SDK licence (reviewed 2026-11-16, again 2026-11-21) allows
 *   distribution only in object code under an end-user licence that forbids redistribution and
 *   reverse engineering, which an open-source engine cannot carry.
 * - Windows itself, for every WDDM 2.4+ adapter, through D3DKMT (the Windows SDK's d3dkmthk.h,
 *   gdi32): the adapter's performance data (temperature, fan, memory clock, power as a share of a
 *   rating it does not name) and each engine node's clock. Vendor-neutral; the temperature where
 *   ADL gives none.
 *
 * WHAT IS NOT TAKEN FROM WINDOWS, and why, so a dashboard never shows a plausible wrong number:
 * - busy: the per-engine running time Windows accounts (Task Manager's GPU Engine counters) reads
 *   0% for a HIP kernel spinning on an R9700 for 15 s (measured 2026-11-16): HIP's compute work is
 *   not seen by that accounting. Busy comes from ADL's GFX activity, the firmware's own count,
 *   or is not reported.
 * - watts: D3DKMT reports power as tenths of a percent of a rating it does not report, so
 *   power_frac stays a fraction; watts come from ADL or are not reported.
 * - vram_used: the engine reports its own residency per card already.
 *
 * Matching: the adapter's PCI address (D3DKMT's KMTQAITYPE_ADAPTERADDRESS, ADL's AdapterInfo bus,
 * device and function) against the engine's spelling of the card's address ("0000:03:00.0",
 * core/device/hip.cpp d_pci_address). Adapters are enumerated once; their handles and the ADL
 * context live for the process. A D3DKMT query takes microseconds; an ADL metrics read 1-4 ms a
 * card (measured on the test machine's R9700s), so an ADL reading is kept for kAdlKeepMs and reused. */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winternl.h>
#include <d3dkmthk.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <tlhelp32.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "os.h"

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "pdh.lib")

namespace rad {
namespace os {
namespace {

/* ---- ADL: the declarations this file needs, from AMD's ADL SDK (MIT; adl_sdk.h, adl_defines.h,
 * adl_structures.h at GPUOpen-LibrariesAndSDKs/display-library 84c881cb). Copyright (c) 2016 - 2022
 * Advanced Micro Devices, Inc. The licence text is win/licenses/adl-sdk.LICENSE. */
constexpr int kAdlMaxPath = 256;            /* ADL_MAX_PATH */
struct AdlAdapterInfo {                     /* AdapterInfo, its Windows layout */
    int  iSize;
    int  iAdapterIndex;
    char strUDID[kAdlMaxPath];
    int  iBusNumber;
    int  iDeviceNumber;
    int  iFunctionNumber;
    int  iVendorID;
    char strAdapterName[kAdlMaxPath];
    char strDisplayName[kAdlMaxPath];
    int  iPresent;
    int  iExist;
    char strDriverPath[kAdlMaxPath];
    char strDriverPathExt[kAdlMaxPath];
    char strPNPString[kAdlMaxPath];
    int  iOSDisplayIndex;
};
struct AdlSensor { int supported; int value; };   /* ADLSingleSensorData */
constexpr int kAdlMaxSensors = 256;                /* ADL_PMLOG_MAX_SENSORS */
struct AdlPmLog { int size; AdlSensor sensors[kAdlMaxSensors]; };   /* ADLPMLogDataOutput */
/* ADL_PMLOG_SENSORS: the three read here. */
constexpr int kAdlActivityGfx = 19;   /* ADL_PMLOG_INFO_ACTIVITY_GFX, percent */
constexpr int kAdlHotspot     = 27;   /* ADL_PMLOG_TEMPERATURE_HOTSPOT, degrees C */
constexpr int kAdlBoardPower  = 73;   /* ADL_PMLOG_BOARD_POWER, watts */
typedef void* (__stdcall* AdlMalloc)(int);              /* ADL_MAIN_MALLOC_CALLBACK */
typedef int (*AdlCreate)(AdlMalloc, int, void**);       /* ADL2_Main_Control_Create */
typedef int (*AdlCount)(void*, int*);                   /* ADL2_Adapter_NumberOfAdapters_Get */
typedef int (*AdlInfo)(void*, AdlAdapterInfo*, int);    /* ADL2_Adapter_AdapterInfo_Get */
typedef int (*AdlQuery)(void*, int, AdlPmLog*);         /* ADL2_New_QueryPMLogData_Get */
constexpr int kAdlOk = 0;                               /* ADL_OK */
/* ---- end of the ADL declarations */

/* ADL hands back what it allocates through this; it is never freed (one adapter list a process). */
void* __stdcall adl_alloc(int n) { return std::malloc(n > 0 ? (size_t)n : 1); }

constexpr int64_t kAdlKeepMs = 500;   /* an ADL reading is reused for this long */

struct Adapter {
    D3DKMT_HANDLE handle = 0;
    std::string   bdf;        /* "0000:bb:dd.f", the engine's spelling */
    uint64_t      luid = 0;   /* LowPart | HighPart << 32, as the counters' instance names carry it */
    int           adl = -1;   /* its ADL adapter index, -1 when ADL does not know it */
    /* The last ADL reading and when it was taken (steady clock, ms; -1 before the first). */
    int64_t       adl_at_ms = -1;
    double        adl_busy = -1, adl_power_w = -1, adl_temp_c = -1;
};

std::mutex           g_mu;
bool                 g_enumerated = false;
std::vector<Adapter> g_adapters;
void*                g_adl = nullptr;   /* the ADL context, null without ADL */
AdlQuery             g_adl_query = nullptr;

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* ADL's index for each adapter, by PCI address, when the AMD driver's ADL is installed. Under g_mu,
 * once. A failure anywhere leaves ADL unused, and the D3DKMT figures stand alone. */
void adl_open() {
    HMODULE h = LoadLibraryExW(L"atiadlxx.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!h) return;
    const auto create = (AdlCreate)(void*)GetProcAddress(h, "ADL2_Main_Control_Create");
    const auto count  = (AdlCount)(void*)GetProcAddress(h, "ADL2_Adapter_NumberOfAdapters_Get");
    const auto info   = (AdlInfo)(void*)GetProcAddress(h, "ADL2_Adapter_AdapterInfo_Get");
    const auto query  = (AdlQuery)(void*)GetProcAddress(h, "ADL2_New_QueryPMLogData_Get");
    void* ctx = nullptr;
    if (!create || !count || !info || !query || create(adl_alloc, 1, &ctx) != kAdlOk || !ctx) return;
    int n = 0;
    if (count(ctx, &n) != kAdlOk || n <= 0) return;
    std::vector<AdlAdapterInfo> ai((size_t)n);
    for (AdlAdapterInfo& a : ai) a.iSize = (int)sizeof a;
    if (info(ctx, ai.data(), (int)(sizeof(AdlAdapterInfo) * ai.size())) != kAdlOk) return;
    /* One GPU has an ADL index per display output; the first for its address serves. */
    for (Adapter& ad : g_adapters)
        for (const AdlAdapterInfo& a : ai) {
            char bdf[32];
            std::snprintf(bdf, sizeof bdf, "0000:%02x:%02x.%x", a.iBusNumber & 0xff, a.iDeviceNumber & 0x1f,
                          a.iFunctionNumber & 0x7);
            if (ad.bdf == bdf) { ad.adl = a.iAdapterIndex; break; }
        }
    g_adl = ctx;
    g_adl_query = query;
}

/* The ADL figures of `ad` into `s`, each only when the card reports it. Under g_mu. */
void adl_read(Adapter& ad, GpuSample& s) {
    if (!g_adl || ad.adl < 0) return;
    const int64_t t = now_ms();
    if (ad.adl_at_ms < 0 || t - ad.adl_at_ms >= kAdlKeepMs) {
        AdlPmLog pm{};
        pm.size = (int)sizeof pm;
        ad.adl_busy = ad.adl_power_w = ad.adl_temp_c = -1;
        if (g_adl_query(g_adl, ad.adl, &pm) == kAdlOk) {
            const AdlSensor& act = pm.sensors[kAdlActivityGfx];
            const AdlSensor& pw  = pm.sensors[kAdlBoardPower];
            const AdlSensor& hot = pm.sensors[kAdlHotspot];
            if (act.supported && act.value >= 0 && act.value <= 100) ad.adl_busy = act.value / 100.0;
            if (pw.supported && pw.value >= 0) ad.adl_power_w = pw.value;
            /* A sensor that reads 0 C is not reading. */
            if (hot.supported && hot.value > 0) ad.adl_temp_c = hot.value;
        }
        ad.adl_at_ms = t;
    }
    if (ad.adl_busy >= 0) s.busy = ad.adl_busy;
    if (ad.adl_power_w >= 0) s.power_w = ad.adl_power_w;
    if (ad.adl_temp_c > 0) s.temp_c = ad.adl_temp_c;
}

/* One query of adapter `h`; false when the driver does not answer it. */
template <class T>
bool query(D3DKMT_HANDLE h, KMTQUERYADAPTERINFOTYPE type, T* out) {
    D3DKMT_QUERYADAPTERINFO q{};
    q.hAdapter = h;
    q.Type = type;
    q.pPrivateDriverData = out;
    q.PrivateDriverDataSize = (UINT)sizeof *out;
    return D3DKMTQueryAdapterInfo(&q) == 0;
}

/* Every adapter with a PCI address, once. Software and remote adapters have none (bus ~0u) and are
 * closed at once. Under g_mu. */
void enumerate() {
    if (g_enumerated) return;
    g_enumerated = true;
    D3DKMT_ENUMADAPTERS2 ea{};
    if (D3DKMTEnumAdapters2(&ea) != 0 || ea.NumAdapters == 0) return;
    std::vector<D3DKMT_ADAPTERINFO> info(ea.NumAdapters);
    ea.pAdapters = info.data();
    if (D3DKMTEnumAdapters2(&ea) != 0) return;
    for (UINT i = 0; i < ea.NumAdapters; ++i) {
        D3DKMT_ADAPTERADDRESS a{};
        if (!query(info[i].hAdapter, KMTQAITYPE_ADAPTERADDRESS, &a) || a.BusNumber == ~0u) {
            D3DKMT_CLOSEADAPTER c{ info[i].hAdapter };
            (void)D3DKMTCloseAdapter(&c);
            continue;
        }
        /* The segment (PCI domain) is not reported; every bus on a desktop board is domain 0. */
        char bdf[32];
        std::snprintf(bdf, sizeof bdf, "0000:%02x:%02x.%x", a.BusNumber & 0xff, a.DeviceNumber & 0x1f,
                      a.FunctionNumber & 0x7);
        Adapter ad;
        ad.handle = info[i].hAdapter;
        ad.bdf = bdf;
        ad.luid = (uint64_t)info[i].AdapterLuid.LowPart |
                  ((uint64_t)(uint32_t)info[i].AdapterLuid.HighPart << 32);
        g_adapters.push_back(std::move(ad));
    }
    adl_open();
}

std::string utf8(const wchar_t* w) {
    if (!w || !*w) return std::string();
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return std::string();
    std::string s((size_t)n, 'x');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    s.resize((size_t)n - 1);   /* the terminator the call wrote */
    return s;
}

/* Every process's dedicated use of every adapter, from the counters: luid -> pid -> bytes. False
 * when the counters do not answer (no GPU counter set on this system, or PDH failed). */
bool dedicated_by_process(std::map<uint64_t, std::map<uint32_t, int64_t>>* out) {
    PDH_HQUERY q = nullptr;
    if (PdhOpenQueryW(nullptr, 0, &q) != ERROR_SUCCESS) return false;
    struct Closer { PDH_HQUERY q; ~Closer() { PdhCloseQuery(q); } } closer{ q };
    PDH_HCOUNTER c = nullptr;
    /* The English name, whatever the system's language. */
    if (PdhAddEnglishCounterW(q, L"\\GPU Process Memory(*)\\Dedicated Usage", 0, &c) != ERROR_SUCCESS)
        return false;
    if (PdhCollectQueryData(q) != ERROR_SUCCESS) return false;
    DWORD bytes = 0, count = 0;
    const DWORD fmt = PDH_FMT_LARGE | PDH_FMT_NOSCALE;
    PDH_STATUS st = PdhGetFormattedCounterArrayW(c, fmt, &bytes, &count, nullptr);
    if (st == ERROR_SUCCESS && count == 0) return true;   /* no process holds any GPU memory */
    if (st != (PDH_STATUS)PDH_MORE_DATA || bytes == 0) return false;
    /* Instances can appear between the two calls: a little more room, and one retry. */
    std::vector<unsigned char> buf;
    for (int tries = 0; tries < 2; ++tries) {
        bytes += 4096;
        buf.assign(bytes, 0);
        st = PdhGetFormattedCounterArrayW(c, fmt, &bytes, &count,
                                          reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data()));
        if (st != (PDH_STATUS)PDH_MORE_DATA) break;
    }
    if (st != ERROR_SUCCESS) return false;
    const auto* items = reinterpret_cast<const PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data());
    for (DWORD i = 0; i < count; ++i) {
        const PDH_FMT_COUNTERVALUE& v = items[i].FmtValue;
        if (v.CStatus != PDH_CSTATUS_VALID_DATA && v.CStatus != PDH_CSTATUS_NEW_DATA) continue;
        uint32_t pid = 0;
        uint64_t luid = 0;
        if (!gpu_process_instance(utf8(items[i].szName), &pid, &luid)) continue;
        (*out)[luid][pid] += (int64_t)v.largeValue;
    }
    return true;
}

/* Every running process's executable name, by pid, from one snapshot. */
std::map<uint32_t, std::string> process_names() {
    std::map<uint32_t, std::string> names;
    const HANDLE h = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (h == INVALID_HANDLE_VALUE) return names;
    PROCESSENTRY32W e{};
    e.dwSize = sizeof e;
    for (BOOL ok = Process32FirstW(h, &e); ok; ok = Process32NextW(h, &e))
        names[(uint32_t)e.th32ProcessID] = utf8(e.szExeFile);
    CloseHandle(h);
    return names;
}

bool wanted(const std::vector<std::string>* only, const std::string& bdf) {
    if (!only) return true;
    for (const std::string& s : *only)
        if (s == bdf) return true;
    return false;
}

}  // namespace

bool gpu_process_instance(const std::string& instance, uint32_t* pid, uint64_t* luid) {
    unsigned long p = 0, hi = 0, lo = 0;
    int phys = 0, end = 0;
    if (std::sscanf(instance.c_str(), "pid_%lu_luid_0x%lx_0x%lx_phys_%d%n", &p, &hi, &lo, &phys, &end) != 4 ||
        end <= 0 || (size_t)end != instance.size())
        return false;
    *pid = (uint32_t)p;
    *luid = ((uint64_t)(uint32_t)hi << 32) | (uint32_t)lo;
    return true;
}

std::vector<GpuHolders> gpu_vram_holders(const std::vector<std::string>& bdfs) {
    std::vector<GpuHolders> out(bdfs.size());
    std::vector<uint64_t> luids(bdfs.size(), 0);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        enumerate();
        for (size_t i = 0; i < bdfs.size(); ++i) {
            out[i].bdf = bdfs[i];
            for (const Adapter& ad : g_adapters)
                if (ad.bdf == bdfs[i]) luids[i] = ad.luid;
        }
    }
    /* The counters' read and the snapshot outside the lock: they take far longer than a sample. */
    std::map<uint64_t, std::map<uint32_t, int64_t>> by;
    if (!dedicated_by_process(&by)) return out;
    const std::map<uint32_t, std::string> names = process_names();
    const uint32_t self = (uint32_t)GetCurrentProcessId();
    for (size_t i = 0; i < bdfs.size(); ++i) {
        if (luids[i] == 0) continue;   /* no adapter at that address: unknown */
        out[i].known = true;
        const auto it = by.find(luids[i]);
        if (it == by.end()) continue;
        for (const auto& [pid, b] : it->second) {
            if (pid == self || b <= 0) continue;
            const auto n = names.find(pid);
            out[i].procs.push_back(GpuProcVram{ pid, n == names.end() ? std::string() : n->second, b });
        }
    }
    return out;
}

std::vector<GpuSample> gpu_samples(const std::vector<std::string>* only) {
    std::lock_guard<std::mutex> lk(g_mu);
    enumerate();
    std::vector<GpuSample> out;
    for (Adapter& ad : g_adapters) {
        if (!wanted(only, ad.bdf)) continue;
        GpuSample s;
        s.bdf = ad.bdf;
        D3DKMT_ADAPTER_PERFDATA pd{};
        pd.PhysicalAdapterIndex = 0;
        if (query(ad.handle, KMTQAITYPE_ADAPTERPERFDATA, &pd)) {
            /* A driver that fills none of it leaves zeros; a card is never at 0.0 C or 0 Hz. */
            if (pd.Temperature > 0) s.temp_c = pd.Temperature / 10.0;
            if (pd.MemoryFrequency > 0) s.mem_mhz = (double)pd.MemoryFrequency / 1e6;
            if (pd.MaxMemoryFrequency > 0) s.mem_max_mhz = (double)pd.MaxMemoryFrequency / 1e6;
            if (pd.FanRPM > 0) s.fan_rpm = pd.FanRPM;
            /* Zero is a real reading here: an idle R9700 reports 0.0% (measured). */
            if (pd.Temperature > 0) s.power_frac = pd.Power / 1000.0;
        }
        D3DKMT_NODE_PERFDATA nd{};
        nd.NodeOrdinal = 0;
        nd.PhysicalAdapterIndex = 0;
        if (query(ad.handle, KMTQAITYPE_NODEPERFDATA, &nd)) {
            if (nd.MaxFrequency > 0) {
                s.gfx_mhz = (double)nd.Frequency / 1e6;
                s.gfx_max_mhz = (double)nd.MaxFrequency / 1e6;
            }
        }
        /* After D3DKMT: ADL's junction temperature, where it has one, replaces the adapter's. */
        adl_read(ad, s);
        out.push_back(std::move(s));
    }
    return out;
}

}  // namespace os
}  // namespace rad
