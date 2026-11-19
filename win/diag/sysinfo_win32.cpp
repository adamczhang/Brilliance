/* Brilliance: the live machine's facts for the startup report and --diagnose, from Windows.
 *
 * core/diag/sysinfo.cpp reads /proc and /sys under a root, which Windows does not have, so with no
 * root (the live machine) a Windows build asks Win32 instead: the processor and its cores from the
 * kernel's processor relationships, memory from GlobalMemoryStatusEx, the release from the
 * registry and RtlGetVersion, the active power plan, and every display-class device from SetupAPI
 * with its PCI address, IDs, service (driver) and PCIe link. What Windows has no counterpart for --
 * transparent huge pages, TTM, cwsr_enable, the kernel command line, io_uring -- stays unknown, as
 * the report already prints a field it could not read. A tree under a root (the tests' fixtures)
 * is still read by the upstream code. */
#include "diag/diag.h"
#include "rad_internal.h"

#include <windows.h>
#include <winternl.h>
#include <setupapi.h>
#include <cfgmgr32.h>
#include <devguid.h>
#include <initguid.h>   /* the property keys below are defined here (selectany), not imported */
#include <devpkey.h>
#include <pciprop.h>
#include <powrprof.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "cfgmgr32.lib")
#pragma comment(lib, "powrprof.lib")

namespace rad {
namespace diag {

namespace {

std::string narrow(const wchar_t* w) {
    if (!w || !*w) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::string s((size_t)n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}

std::string reg_string(HKEY root, const wchar_t* key, const wchar_t* value) {
    wchar_t buf[512];
    DWORD bytes = sizeof buf;
    if (RegGetValueW(root, key, value, RRF_RT_REG_SZ, nullptr, buf, &bytes) != ERROR_SUCCESS) return {};
    return narrow(buf);
}

/* Cores and logical processors, over every processor group. */
void read_cpus(HostFacts* h) {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<unsigned char> buf(len);
    if (len && GetLogicalProcessorInformationEx(
                   RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &len)) {
        int cores = 0, threads = 0;
        for (DWORD off = 0; off < len;) {
            const auto* p = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(buf.data() + off);
            if (p->Relationship == RelationProcessorCore) {
                ++cores;
                for (WORD g = 0; g < p->Processor.GroupCount; ++g)
                    threads += (int)__builtin_popcountll((unsigned long long)p->Processor.GroupMask[g].Mask);
            }
            off += p->Size;
        }
        if (cores > 0) h->cores = cores;
        if (threads > 0) h->threads = threads;
    }
    h->cpu_model = trim(reg_string(HKEY_LOCAL_MACHINE,
                                   L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                                   L"ProcessorNameString"));
}

/* The release: "10.0.26300" as the kernel's, "Windows 11 Pro 25H2" as the distribution's. The
 * registry's ProductName still says Windows 10 on Windows 11, whose builds start at 22000. */
void read_release(HostFacts* h) {
    using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
    DWORD build = 0;
    if (HMODULE nt = GetModuleHandleW(L"ntdll.dll")) {
        if (auto fn = (RtlGetVersionFn)(void*)GetProcAddress(nt, "RtlGetVersion")) {
            OSVERSIONINFOW v{};
            v.dwOSVersionInfoSize = sizeof v;
            if (fn(&v) == 0) {
                build = v.dwBuildNumber;
                char b[64];
                std::snprintf(b, sizeof b, "%lu.%lu.%lu", v.dwMajorVersion, v.dwMinorVersion,
                              v.dwBuildNumber);
                h->kernel = b;
            }
        }
    }
    const wchar_t* cv = L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";
    std::string product = reg_string(HKEY_LOCAL_MACHINE, cv, L"ProductName");
    if (build >= 22000 && product.rfind("Windows 10", 0) == 0) product.replace(0, 10, "Windows 11");
    const std::string disp = reg_string(HKEY_LOCAL_MACHINE, cv, L"DisplayVersion");
    h->distro = product + (disp.empty() ? "" : " " + disp);
}

/* Swap, as Windows has it: the page files' size and the part holding pages now
 * (SystemPageFileInformation, the figures Task Manager and pagefile counters show). Commit charge
 * is not swap -- most of it is never written out -- so it is not used here. */
void read_page_files(HostFacts* h) {
    struct PageFileInfo {          /* SYSTEM_PAGEFILE_INFORMATION */
        ULONG          NextEntryOffset;
        ULONG          TotalSize;  /* pages */
        ULONG          TotalInUse; /* pages */
        ULONG          PeakUsage;  /* pages */
        UNICODE_STRING PageFileName;
    };
    using QueryFn = LONG(WINAPI*)(ULONG, PVOID, ULONG, PULONG);
    HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    auto q = nt ? (QueryFn)(void*)GetProcAddress(nt, "NtQuerySystemInformation") : nullptr;
    if (!q) return;
    std::vector<unsigned char> buf(64 * 1024);
    ULONG got = 0;
    if (q(18 /* SystemPageFileInformation */, buf.data(), (ULONG)buf.size(), &got) != 0) return;
    int64_t total = 0, used = 0;
    for (size_t off = 0; got && off + sizeof(PageFileInfo) <= got;) {
        const auto* p = (const PageFileInfo*)(buf.data() + off);
        total += (int64_t)p->TotalSize * h->page_size;
        used += (int64_t)p->TotalInUse * h->page_size;
        if (!p->NextEntryOffset) break;
        off += p->NextEntryOffset;
    }
    h->swap_total = total;
    h->swap_free = total - used;
}

/* The active power plan, where Linux has a cpufreq governor. */
void read_power_plan(HostFacts* h) {
    GUID* scheme = nullptr;
    if (PowerGetActiveScheme(nullptr, &scheme) != ERROR_SUCCESS || !scheme) return;
    DWORD bytes = 0;
    if (PowerReadFriendlyName(nullptr, scheme, nullptr, nullptr, nullptr, &bytes) == ERROR_SUCCESS && bytes) {
        std::vector<unsigned char> name(bytes);
        if (PowerReadFriendlyName(nullptr, scheme, nullptr, nullptr, name.data(), &bytes) == ERROR_SUCCESS)
            h->cpu_governor = "power plan " + narrow((const wchar_t*)name.data());
    }
    LocalFree(scheme);
    h->cpu_driver = "Windows";
}

std::string reg_prop(HDEVINFO set, SP_DEVINFO_DATA* d, DWORD prop) {
    wchar_t buf[1024];
    DWORD type = 0;
    if (!SetupDiGetDeviceRegistryPropertyW(set, d, prop, &type, (PBYTE)buf, sizeof buf - sizeof(wchar_t), nullptr))
        return {};
    buf[(sizeof buf / sizeof buf[0]) - 1] = 0;
    return narrow(buf);   /* the first string of a REG_MULTI_SZ */
}

/* The kernel's link-speed code (pciprop.h: 1 = 2.5 GT/s ... 5 = 32 GT/s, 6 = 64) as the strings
 * the Linux report prints. */
std::string link_speed(DWORD code) {
    switch (code) {
    case 1: return "2.5 GT/s PCIe";
    case 2: return "5.0 GT/s PCIe";
    case 3: return "8.0 GT/s PCIe";
    case 4: return "16.0 GT/s PCIe";
    case 5: return "32.0 GT/s PCIe";
    case 6: return "64.0 GT/s PCIe";
    default: return {};
    }
}

/* A devnode's 32-bit property, 0 when it has none. */
DWORD node_u32(DEVINST n, const DEVPROPKEY& key) {
    DEVPROPTYPE type = 0;
    DWORD v = 0;
    ULONG size = sizeof v;
    if (CM_Get_DevNode_PropertyW(n, &key, &type, (PBYTE)&v, &size, 0) != CR_SUCCESS ||
        type != DEVPROP_TYPE_UINT32)
        return 0;
    return v;
}

/* A devnode's PCI address, "" when it is not a PCI-enumerated device (the root complex is ACPI's,
 * and so are its parents). */
std::string node_bdf(DEVINST n) {
    wchar_t id[MAX_DEVICE_ID_LEN];
    if (CM_Get_Device_IDW(n, id, MAX_DEVICE_ID_LEN, 0) != CR_SUCCESS || wcsncmp(id, L"PCI\\", 4) != 0)
        return {};
    DEVPROPTYPE type = 0;
    DWORD bus = 0, addr = 0;
    ULONG size = sizeof bus;
    if (CM_Get_DevNode_PropertyW(n, &DEVPKEY_Device_BusNumber, &type, (PBYTE)&bus, &size, 0) != CR_SUCCESS)
        return {};
    size = sizeof addr;
    if (CM_Get_DevNode_PropertyW(n, &DEVPKEY_Device_Address, &type, (PBYTE)&addr, &size, 0) != CR_SUCCESS)
        return {};
    char b[32];
    std::snprintf(b, sizeof b, "0000:%02lx:%02lx.%lx", bus, (addr >> 16) & 0xffff, addr & 0xffff);
    return b;
}

/* The device's PCIe path, root port first and its own link last, as Linux's report lists it:
 * every PCI ancestor (the root port, a slot's bridge, a card's own switch ports). Windows states
 * link speed and width for endpoints only, not for bridges, so the hops above the card are named
 * with their links unknown. That matters on a card like the R9700, which carries a switch on its
 * board: its own link is to that switch (32 GT/s x16), and the slot's link -- the one a riser or
 * an x4 slot narrows -- is the upstream switch port's, which Windows does not report. */
std::vector<PciHop> link_chain(DEVINST dev) {
    std::vector<PciHop> chain;
    for (DEVINST n = dev; n;) {
        const DWORD cs = node_u32(n, DEVPKEY_PciDevice_CurrentLinkSpeed);
        const DWORD cw = node_u32(n, DEVPKEY_PciDevice_CurrentLinkWidth);
        const std::string bdf = node_bdf(n);
        if (!bdf.empty() || cs || cw) {
            PciHop h;
            h.bdf = bdf;
            h.cur_speed = link_speed(cs);
            h.max_speed = link_speed(node_u32(n, DEVPKEY_PciDevice_MaxLinkSpeed));
            h.cur_width = cw ? (int)cw : -1;
            const DWORD mw = node_u32(n, DEVPKEY_PciDevice_MaxLinkWidth);
            h.max_width = mw ? (int)mw : -1;
            chain.insert(chain.begin(), h);
        }
        DEVINST parent = 0;
        if (CM_Get_Parent(&parent, n, 0) != CR_SUCCESS) break;
        n = parent;
    }
    return chain;
}

/* The adapter's memory as its driver states it: HardwareInformation.qwMemorySize in the driver
 * key (a WDDM display driver writes it), -1 when absent. */
int64_t driver_vram(HDEVINFO set, SP_DEVINFO_DATA* d) {
    HKEY k = SetupDiOpenDevRegKey(set, d, DICS_FLAG_GLOBAL, 0, DIREG_DRV, KEY_READ);
    if (k == INVALID_HANDLE_VALUE) return -1;
    unsigned long long v = 0;
    DWORD bytes = sizeof v, type = 0;
    int64_t out = -1;
    if (RegQueryValueExW(k, L"HardwareInformation.qwMemorySize", nullptr, &type, (LPBYTE)&v, &bytes) ==
            ERROR_SUCCESS && (type == REG_QWORD || type == REG_BINARY) && bytes == sizeof v)
        out = (int64_t)v;
    RegCloseKey(k);
    return out;
}

/* Every present display-class device: its PCI address (domain 0; Windows reports bus, device and
 * function), vendor and device IDs, the service bound to it, its PCIe path, the VRAM its driver
 * states and its NUMA node. `integrated` stays false here; the engine fills in the name, rank and
 * device of the cards it uses from the runtime. */
void read_cards(std::vector<CardFacts>* cards) {
    HDEVINFO set = SetupDiGetClassDevsW(&GUID_DEVCLASS_DISPLAY, nullptr, nullptr, DIGCF_PRESENT);
    if (set == INVALID_HANDLE_VALUE) return;
    SP_DEVINFO_DATA d{};
    d.cbSize = sizeof d;
    for (DWORD i = 0; SetupDiEnumDeviceInfo(set, i, &d); ++i) {
        CardFacts c;
        const std::string hw = reg_prop(set, &d, SPDRP_HARDWAREID);   /* PCI\VEN_1002&DEV_7551&... */
        if (hw.rfind("PCI\\", 0) != 0) continue;
        const size_t v = hw.find("VEN_"), dv = hw.find("DEV_");
        if (v != std::string::npos) c.vendor = "0x" + hw.substr(v + 4, 4);
        if (dv != std::string::npos) c.pci_device = "0x" + hw.substr(dv + 4, 4);
        for (char& ch : c.vendor) ch = (char)std::tolower((unsigned char)ch);
        for (char& ch : c.pci_device) ch = (char)std::tolower((unsigned char)ch);
        DWORD bus = 0, addr = 0, type = 0;
        const bool have_bus = SetupDiGetDeviceRegistryPropertyW(set, &d, SPDRP_BUSNUMBER, &type,
                                                                (PBYTE)&bus, sizeof bus, nullptr);
        const bool have_addr = SetupDiGetDeviceRegistryPropertyW(set, &d, SPDRP_ADDRESS, &type,
                                                                 (PBYTE)&addr, sizeof addr, nullptr);
        if (have_bus && have_addr) {
            char b[32];
            std::snprintf(b, sizeof b, "0000:%02lx:%02lx.%lx", bus, (addr >> 16) & 0xffff, addr & 0xffff);
            c.bdf = b;
        }
        c.driver = reg_prop(set, &d, SPDRP_SERVICE);
        c.chain = link_chain(d.DevInst);
        c.vram_total = driver_vram(set, &d);
        {
            DEVPROPTYPE type = 0;
            INT32 node = -1;
            if (SetupDiGetDevicePropertyW(set, &d, &DEVPKEY_Device_Numa_Node, &type, (PBYTE)&node,
                                          sizeof node, nullptr, 0) && type == DEVPROP_TYPE_UINT32)
                c.numa_node = (int)node;
        }
        cards->push_back(std::move(c));
    }
    SetupDiDestroyDeviceInfoList(set);
    std::sort(cards->begin(), cards->end(),
              [](const CardFacts& a, const CardFacts& b) { return a.bdf < b.bdf; });
}

}  /* namespace */

void collect_windows(Facts* f) {
    HostFacts& h = f->host;
    SYSTEM_INFO si{};
    GetNativeSystemInfo(&si);
    if (si.dwPageSize) h.page_size = (int64_t)si.dwPageSize;
    read_cpus(&h);
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    if (GlobalMemoryStatusEx(&m)) {
        h.mem_total = (int64_t)m.ullTotalPhys;
        h.mem_available = (int64_t)m.ullAvailPhys;
    }
    read_page_files(&h);
    read_release(&h);
    read_power_plan(&h);
    h.iommu = "unknown";
    h.iommu_why = "not read on Windows";
    read_cards(&f->cards);
}

}  /* namespace diag */
}  /* namespace rad */
