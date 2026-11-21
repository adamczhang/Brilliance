/* os_gpu_test.cpp -- Brilliance: os::gpu_samples (win/os/gpu_win32.cpp) on the machine it runs on.
 *
 * Reads the graphics kernel's adapter data and, where the AMD driver installs it, ADL's metrics
 * only: it starts no device work and needs no GPU (on a machine without a hardware adapter every
 * check holds trivially). Checks the contract the
 * dashboard relies on: PCI addresses in the engine's spelling, filtering by `only`, and readings
 * that are either -1 ("not reported") or physically plausible -- never a made-up 0. */
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
    std::printf("%zu adapter(s); %s\n", all.size(), g_fail ? "FAILED" : "ok");
    return g_fail ? 1 : 0;
}
