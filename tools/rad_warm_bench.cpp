/* rad-warm-bench -- Brilliance (ngram-warm-tune): time --ngram-warm's read of a container's n-gram
 * table into the OS file cache, alone, with no engine and no card.
 *
 * One run is: an optional eviction (another container read through the cache first, so the table's
 * pages are the oldest and are given back to it), the warm read in the mode asked for, timed, and
 * an optional second pass that tells whether the pages stayed: a cached pass reads at memory
 * speed, far above the drive's. It prints one JSON line a read, for scripts, and a summary line.
 *
 * The cache it reads into is the machine's: a measurement, so it runs alone on the machine (run
 * class M). It reads and keeps nothing; no lock, no pin. A run needs the table's size free in the
 * standby list to mean anything, and the eviction only works if it reads more than the standby list
 * holds (Windows: \Memory\Standby Cache * Bytes). Without elevation there is no purge of the
 * standby list, which is why it evicts by reading.
 */
#include "format/radfile.h"
#include "format/recipe.h"
#include "format/table_warm.h"
#include "rad_internal.h"
#include "iface.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace rad;

namespace {

void usage() {
    std::fputs(
        "rad-warm-bench -- time the n-gram table's read into the OS file cache (--ngram-warm)\n"
        "\n"
        "usage: rad-warm-bench [options] <model.rad>\n"
        "\n"
        "  --glob GLOB        the weights to read (default *ple_ngram*), every plane\n"
        "  --io SPEC          how: io=pread|handle|prefetch,readers=N,piece=SIZE over the default\n"
        "                     (io=handle,readers=16,piece=64M); `alpha3` is the first form\n"
        "                     (io=pread,readers=8,piece=16M)\n"
        "  --evict FILE.rad   first read FILE through the cache, so the table's pages go back\n"
        "  --evict-gib N      how much of it (default 120)\n"
        "  --verify           then read the table a second time (io=handle), timed\n"
        "  --no-warm          only the eviction: read nothing of the model (with --evict)\n"
        "  --label TEXT       carried into the JSON lines\n"
        "  -h, --help\n"
        "\n"
        "Output: one JSON line a read, {\"label\",\"what\",\"io\",\"bytes\",\"secs\",\"gbps\",\"t0\",\"t1\"}, `what`\n"
        "being evict, warm or verify. Exit 0, or 1 on a failed read.\n",
        stdout);
}

/* Seconds since the epoch, to line a read up with a typeperf log. */
double unix_now() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

int timed(const char* label, const char* what, const RadFile& f,
          const std::vector<std::pair<int64_t, int64_t>>& rg, const TableWarmOpts& o, double* gbps) {
    int64_t bytes = 0;
    double secs = 0;
    std::string why;
    const double t0 = unix_now();
    const int rc = table_warm_read(f, rg, &bytes, &secs, &why, o);
    const double t1 = unix_now();
    const double g = secs > 0 ? (double)bytes / secs / 1e9 : 0.0;
    if (gbps) *gbps = g;
    std::printf("{\"label\":\"%s\",\"what\":\"%s\",\"io\":\"%s\",\"bytes\":%lld,\"secs\":%.3f,"
                "\"gbps\":%.3f,\"t0\":%.3f,\"t1\":%.3f%s}\n",
                label, what, table_warm_opts_str(o).c_str(), (long long)bytes, secs, g, t0, t1,
                rc == RAD_OK ? "" : ",\"failed\":true");
    std::fflush(stdout);
    if (rc != RAD_OK) std::fprintf(stderr, "rad-warm-bench: %s read failed: %s\n", what, why.c_str());
    return rc;
}

}  /* namespace */

int main(int argc, char** argv) {
    std::string path, glob = "*ple_ngram*", evict, label = "run";
    int64_t evict_gib = 120;
    bool verify = false, no_warm = false;
    TableWarmOpts o;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "rad-warm-bench: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--glob") glob = next("--glob");
        else if (a == "--io") {
            const std::string s = next("--io");
            if (s == "alpha3") { o = table_warm_opts_alpha3(); continue; }
            std::string why;
            if (table_warm_opts_parse(s, &o, &why) != RAD_OK) {
                std::fprintf(stderr, "rad-warm-bench: --io: %s\n", why.c_str());
                return 2;
            }
        }
        else if (a == "--evict") evict = next("--evict");
        else if (a == "--evict-gib")
            evict_gib = rad_flag_int("rad-warm-bench", "--evict-gib", next("--evict-gib"), 1, 4096);
        else if (a == "--verify") verify = true;
        else if (a == "--no-warm") no_warm = true;
        else if (a == "--label") label = next("--label");
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "rad-warm-bench: unrecognised argument '%s'. Try --help.\n", a.c_str());
            return 2;
        } else if (path.empty()) path = a;
        else { std::fprintf(stderr, "rad-warm-bench: more than one model given\n"); return 2; }
    }
    if (path.empty()) { usage(); return 2; }

    RadFile f;
    if (f.open(path.c_str()) < 0) return 1;
    std::vector<std::pair<int64_t, int64_t>> rg;
    for (int64_t i = 0; i < f.entry_count(); ++i) {
        const RadFileEntry& e = f.entry(i);
        if (!recipe_glob(glob.c_str(), f.str(e.name))) continue;
        for (uint32_t k = 0; k < e.n_planes; ++k) {
            const RadFilePlane& p = f.plane(e, (int)k);
            rg.emplace_back((int64_t)p.offset, (int64_t)p.bytes);
        }
    }
    rg = ngram_warm_ranges(std::move(rg));
    int64_t total = 0;
    for (const auto& x : rg) total += x.second;
    if (total <= 0) {
        std::fprintf(stderr, "rad-warm-bench: no weight in %s matches '%s'\n", path.c_str(), glob.c_str());
        return 1;
    }
    std::fprintf(stderr, "rad-warm-bench: %s: %s in %zu range(s) match '%s'\n", path.c_str(),
                 humanb(total).c_str(), rg.size(), glob.c_str());

    if (!evict.empty()) {
        RadFile ev;
        if (ev.open(evict.c_str()) < 0) return 1;
        const int64_t n = std::min<int64_t>(ev.size(), evict_gib << 30);
        if (timed(label.c_str(), "evict", ev, { { 0, n } }, TableWarmOpts{}, nullptr) != RAD_OK)
            return 1;
    }
    if (no_warm) return 0;
    double warm = 0, again = 0;
    if (timed(label.c_str(), "warm", f, rg, o, &warm) != RAD_OK) return 1;
    if (verify && timed(label.c_str(), "verify", f, rg, TableWarmOpts{}, &again) != RAD_OK) return 1;
    std::fprintf(stderr, "rad-warm-bench: %s: warm %.2f GB/s%s\n", label.c_str(), warm,
                 verify ? fmt(", second pass %.2f GB/s", again).c_str() : "");
    return 0;
}
