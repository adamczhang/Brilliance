/* os_gpu_test.cpp -- Brilliance: os::gpu_samples (win/os/gpu_win32.cpp) on the machine it runs on.
 *
 * Reads the graphics kernel's adapter data, its per-process memory counters and, where the AMD
 * driver installs it, ADL's metrics only: it starts no device work and needs no GPU (on a machine
 * without a hardware adapter every check holds trivially). Checks the contract the
 * dashboard relies on: PCI addresses in the engine's spelling, filtering by `only`, and readings
 * that are either -1 ("not reported") or physically plausible -- never a made-up 0. And the one the
 * engine's startup relies on (gpu_vram_holders): one reading per address asked, this process left
 * out, an unknown address unknown; it prints who holds VRAM on each card, which makes it the live
 * check of that reading. */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>   /* GetCurrentProcessId */

#include <chrono>
#include <cstdio>
#include <regex>
#include <string>
#include <vector>

#include "os.h"

static int g_fail = 0;
#define CHECK(c, ...)                                         \
    do {                                                      \
        if (!(c)) {                                           \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);  \
            std::printf(__VA_ARGS__);                         \
            std::printf("\n");                                \
            ++g_fail;                                         \
        }                                                     \
    } while (0)

int main() {
    using rad::os::GpuSample;
    const std::vector<GpuSample> all = rad::os::gpu_samples();
    const std::regex bdf("^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\\.[0-7]$");
    for (const GpuSample& g : all) {
        std::printf("  %s: temp %.1f C, power %.3f of rating, %.0f W, busy %.2f, fan %.0f rpm, gfx %.0f/%.0f MHz, "
                    "mem %.0f/%.0f MHz\n",
                    g.bdf.c_str(), g.temp_c, g.power_frac, g.power_w, g.busy, g.fan_rpm, g.gfx_mhz,
                    g.gfx_max_mhz, g.mem_mhz, g.mem_max_mhz);
        CHECK(std::regex_match(g.bdf, bdf), "address '%s' is not in the engine's spelling", g.bdf.c_str());
        CHECK(g.temp_c == -1 || (g.temp_c > 0 && g.temp_c < 150), "%s: temperature %.1f", g.bdf.c_str(), g.temp_c);
        CHECK(g.power_frac == -1 || (g.power_frac >= 0 && g.power_frac <= 2), "%s: power share %.3f",
              g.bdf.c_str(), g.power_frac);
        CHECK(g.gfx_mhz == -1 || (g.gfx_mhz >= 0 && g.gfx_max_mhz > 0), "%s: engine clock %.0f of %.0f",
              g.bdf.c_str(), g.gfx_mhz, g.gfx_max_mhz);
        CHECK(g.power_w == -1 || (g.power_w >= 0 && g.power_w < 2000), "%s: board power %.0f W",
              g.bdf.c_str(), g.power_w);
        CHECK(g.busy == -1 || (g.busy >= 0 && g.busy <= 1), "%s: busy %.2f", g.bdf.c_str(), g.busy);
        CHECK(g.vram_used == -1, "%s: vram is not reported on Windows", g.bdf.c_str());
    }
    /* `only` returns exactly the named cards, and asks none it was not given. */
    if (!all.empty()) {
        const std::vector<std::string> one{ all.back().bdf };
        const std::vector<GpuSample> got = rad::os::gpu_samples(&one);
        CHECK(got.size() == 1 && got[0].bdf == one[0], "only={%s}: %zu sample(s)", one[0].c_str(), got.size());
    }
    const std::vector<std::string> none{ "0000:ff:1f.7" };
    CHECK(rad::os::gpu_samples(&none).empty(), "an address no adapter has returned a sample");

    /* Every other process's VRAM by card (gpu_vram_holders), the live reading behind the engine's
     * startup report and settle check (core/mem/vram_others.h). First the counter instances' names. */
    uint32_t pid = 0;
    uint64_t luid = 0;
    CHECK(rad::os::gpu_process_instance("pid_19340_luid_0x00000000_0x000181BB_phys_0", &pid, &luid) &&
              pid == 19340 && luid == 0x181BBull,
          "instance name: pid %u luid 0x%llx", pid, (unsigned long long)luid);
    CHECK(rad::os::gpu_process_instance("pid_7_luid_0x00000001_0x0000abcd_phys_1", &pid, &luid) && pid == 7 &&
              luid == 0x10000ABCDull,
          "instance name with a high LUID part: pid %u luid 0x%llx", pid, (unsigned long long)luid);
    for (const char* bad : { "_Total", "pid_7_luid_0x00000000_phys_0", "pid_7_luid_0x00000000_0x000181BB_phys_0x",
                             "luid_0x00000000_0x000181BB_phys_0", "" })
        CHECK(!rad::os::gpu_process_instance(bad, &pid, &luid), "'%s' parsed as an instance", bad);
    /* Then the reading of every adapter, and an address no adapter has, which is unknown. */
    std::vector<std::string> ask;
    for (const GpuSample& g : all) ask.push_back(g.bdf);
    ask.push_back(none[0]);
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<rad::os::GpuHolders> held = rad::os::gpu_vram_holders(ask);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK(held.size() == ask.size(), "%zu reading(s) for %zu address(es)", held.size(), ask.size());
    const uint32_t self = (uint32_t)GetCurrentProcessId();
    size_t known = 0;
    for (size_t i = 0; i < held.size() && i < ask.size(); ++i) {
        const rad::os::GpuHolders& h = held[i];
        CHECK(h.bdf == ask[i], "reading %zu is for '%s', asked '%s'", i, h.bdf.c_str(), ask[i].c_str());
        known += h.known ? 1 : 0;
        int64_t total = 0;
        std::string who;
        for (const rad::os::GpuProcVram& p : h.procs) {
            CHECK(p.pid != self, "%s: this process is listed", h.bdf.c_str());
            CHECK(p.bytes > 0, "%s: pid %u holds %lld bytes", h.bdf.c_str(), p.pid, (long long)p.bytes);
            total += p.bytes;
            char line[512];
            std::snprintf(line, sizeof line, "%s%s (pid %u) %.1f MiB", who.empty() ? "" : ", ",
                          p.name.empty() ? "?" : p.name.c_str(), p.pid, p.bytes / 1048576.0);
            who += line;
        }
        CHECK(h.known || h.procs.empty(), "%s: unknown, yet %zu holder(s)", h.bdf.c_str(), h.procs.size());
        std::printf("  %s: other processes' VRAM %s%.1f MiB%s%s\n", h.bdf.c_str(), h.known ? "" : "unknown, ",
                    total / 1048576.0, who.empty() ? "" : ": ", who.c_str());
    }
    if (!held.empty()) CHECK(!held.back().known, "an address no adapter has read as known");
    std::printf("  read in %.0f ms; %zu of %zu adapter(s) known\n", ms, known, all.size());

    std::printf("%zu adapter(s); %s\n", all.size(), g_fail ? "FAILED" : "ok");
    return g_fail ? 1 : 0;
}
