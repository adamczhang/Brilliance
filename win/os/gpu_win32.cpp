/* gpu_win32.cpp -- Brilliance: os::gpu_samples (win/os/os.h), what the graphics kernel reports of
 * each hardware GPU, for the dashboard's per-card figures (core/fleet/fleet.cpp, card_telemetry).
 *
 * WHERE THE FIGURES COME FROM. Linux reads the amdgpu driver's sysfs: gpu_busy_percent, hwmon power
 * and the junction temperature. Windows has no sysfs. The vendor's own telemetry API (AMD's ADLX)
 * reads all three exactly, but its SDK licence forbids redistributing its headers in source form and
 * asks for a restrictive end-user licence (reviewed 2026-11-16), which a source-distributed port of
 * an open-source engine cannot carry. So this reads what Windows itself offers for every WDDM 2.4+
 * adapter, through D3DKMT (the Windows SDK's d3dkmthk.h, gdi32): the adapter's performance data
 * (temperature, fan, memory clock, power as a share of its rated power) and each engine node's clock.
 *
 * WHAT IT DOES NOT REPORT, and why, so a dashboard never shows a plausible wrong number:
 * - busy: the per-engine running time Windows accounts (Task Manager's GPU Engine counters) reads
 *   0% for a HIP kernel spinning on an R9700 for 15 s (measured 2026-11-16):
 *   HIP's compute work is not seen by that accounting, so a busy figure from it would say idle
 *   while the card works.
 * - vram_used: the engine reports its own residency per card already.
 * - watts: the adapter reports power as tenths of a percent of its rated power, not watts; the
 *   rating it is a share of is not reported, so power_frac stays a fraction.
 *
 * Matching: the adapter's PCI address (KMTQAITYPE_ADAPTERADDRESS) against the engine's spelling of
 * the card's address ("0000:03:00.0", core/device/hip.cpp d_pci_address). Adapters are enumerated
 * once; their handles live for the process. Queries take microseconds and allocate nothing but the
 * returned vector. */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winternl.h>
#include <d3dkmthk.h>

#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include "os.h"

#pragma comment(lib, "gdi32.lib")

namespace rad {
namespace os {
namespace {

struct Adapter {
    D3DKMT_HANDLE handle = 0;
    std::string   bdf;    /* "0000:bb:dd.f", the engine's spelling */
};

std::mutex           g_mu;
bool                 g_enumerated = false;
std::vector<Adapter> g_adapters;

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
        g_adapters.push_back(Adapter{ info[i].hAdapter, bdf });
    }
}

bool wanted(const std::vector<std::string>* only, const std::string& bdf) {
    if (!only) return true;
    for (const std::string& s : *only)
        if (s == bdf) return true;
    return false;
}

}  // namespace

std::vector<GpuSample> gpu_samples(const std::vector<std::string>* only) {
    std::lock_guard<std::mutex> lk(g_mu);
    enumerate();
    std::vector<GpuSample> out;
    for (const Adapter& ad : g_adapters) {
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
        out.push_back(std::move(s));
    }
    return out;
}

}  // namespace os
}  // namespace rad
