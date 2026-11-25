/* heat_replay_test.cpp -- rad-heat-replay's per-layer floor (--layer-min), run as its own process on
 * a synthetic route trace in core/place/rtrace.h's format, written here.
 *
 * The floor models a mover that keeps every layer at N resident units (backlog M-16, audit E-M6):
 * the thin layers are raised at the start -- the units coming off the layers above N, so the slab
 * does not grow -- and a swap whose demotion would take a layer under N is abandoned. The trace's
 * snapshot puts all sixteen residents in layers 0 and 1, the pair routed every dispatch, and layers
 * 2 and 3 are routed one dispatch in eight and hold nothing: a floor of 3 has to take units out of
 * the hot pair to fill the quiet one, and the replay has to miss more for it. Each of the three
 * ways the floor can be wrong has its check: a layer left under N at the start, a slab that changed
 * size, and a hit rate that did not move. A floor the slab cannot hold is clamped to the one it
 * can, so it neither overshoots nor freezes every demotion.
 *
 * Everything runs on the host backend (the tool touches no card), so this is a unit entry; each
 * case runs the tool as its own process, a few tens of milliseconds each.
 */
#include "rad_test.h"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

/* Little-endian, as rtrace.h writes it: this machine's own byte order is the format's. */
template <class T> void put(std::string& s, T v) { s.append(reinterpret_cast<const char*>(&v), sizeof v); }

/* 4 layers x 8 experts of one MiB each, all movable; tiers after rad::Tier: VRAM 0, HostPinned 1. */
std::string trace_bytes() {
    const int32_t nl = 4, ne = 8, ndis = 300;
    std::string s;
    s.append("RADRTRC1", 8);
    put(s, nl);
    put(s, ne);
    for (int32_t k = 0; k < nl * ne; ++k) put(s, (int64_t)(1 << 20));
    for (int32_t k = 0; k < nl * ne; ++k) s.push_back((char)1);

    /* The snapshot at dispatch 0: the hot pair (layers 0 and 1) resident, the quiet one pooled. */
    s.push_back('T');
    put(s, (int32_t)0);
    for (int32_t l = 0; l < nl; ++l)
        for (int32_t e = 0; e < ne; ++e) s.push_back((char)(l < 2 ? 0 : 1));

    uint64_t routed = 0, routed_res = 0;
    for (int32_t d = 0; d < ndis; ++d) {
        /* Layers 0 and 1 are routed every dispatch; 2 and 3 one dispatch in eight. The floor's
         * residents in the quiet pair are then mostly wasted, and a replay misses more for them. */
        const bool quiet = d % 8 != 0;
        for (int32_t l = 0; l < nl; ++l) {
            if (quiet && l >= 2) continue;
            /* Six distinct experts a layer, one row each, walking the layer by dispatch. */
            std::string pairs;
            int32_t n = 0;
            for (int32_t i = 0; i < 6; ++i) {
                const uint16_t e = (uint16_t)((d + i * 3 + l) % ne);
                if (i && e == (uint16_t)((d + (i - 1) * 3 + l) % ne)) continue;
                put(pairs, e);
                put(pairs, (uint16_t)1);
                ++n;
                ++routed;
                if (l < 2) ++routed_res;
            }
            s.push_back('R');
            put(s, d);
            put(s, (int32_t)0);
            put(s, (int16_t)l);
            put(s, (int16_t)1);
            put(s, (int32_t)(l < 2 ? 0 : n));
            put(s, n);
            s += pairs;
        }
        s.push_back('D');
        put(s, d);
        put(s, (int32_t)0);
        put(s, routed);
        put(s, routed_res);
        put(s, (uint64_t)0);
        put(s, (uint64_t)0);
        put(s, (uint64_t)0);
        put(s, (uint64_t)0);
        put(s, (uint64_t)0);
    }
    return s;
}

struct Out {
    int code = -1;
    std::string text;
};

#if defined(_WIN32)
/* Brilliance: _popen hands the line to cmd.exe, which takes neither VAR=value prefixes nor single
 * quotes (kbench_mutant_test.cpp does the same): the variables are set with cmd's own `set`, and
 * the paths are double-quoted. _pclose returns the tool's exit code. */
Out run(const std::string& path, const std::string& args) {
    std::string a = args;
    std::replace(a.begin(), a.end(), '\'', '"');
    std::string bin = RAD_HEAT_REPLAY_BIN;
    std::replace(bin.begin(), bin.end(), '/', '\\');
    const std::string cmd = "set \"ROCR_VISIBLE_DEVICES=99\" & set \"OMP_NUM_THREADS=2\" & "
                            "set \"OMP_WAIT_POLICY=passive\" & \"" + bin + "\" \"" + path + "\" " + a +
                            " 2>&1";
    Out o;
    FILE* p = _popen(cmd.c_str(), "rb");
    if (!p) return o;
    char buf[4096];
    size_t got;
    while ((got = std::fread(buf, 1, sizeof buf, p)) > 0) o.text.append(buf, got);
    o.code = _pclose(p);
    return o;
}
#else
Out run(const std::string& path, const std::string& args) {
    const std::string cmd = "ROCR_VISIBLE_DEVICES=99 OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive " +
                            std::string(RAD_HEAT_REPLAY_BIN) + " '" + path + "' " + args + " 2>&1";
    Out o;
    FILE* p = ::popen(cmd.c_str(), "r");
    if (!p) return o;
    char buf[4096];
    size_t got;
    while ((got = std::fread(buf, 1, sizeof buf, p)) > 0) o.text.append(buf, got);
    const int st = ::pclose(p);
    o.code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    return o;
}
#endif

/* The replay's hit rate, from "replay: routed R missed M (hit H; heat's own count ...)". */
bool hit_of(const Out& o, double* hit) {
    const size_t at = o.text.find("missed ");
    if (at == std::string::npos) return false;
    const size_t p = o.text.find("(hit ", at);
    if (p == std::string::npos) return false;
    return std::sscanf(o.text.c_str() + p, "(hit %lf", hit) == 1;
}

/* "residents a layer: start MIN..MAX (median ...)", the range over the layers with a movable unit. */
bool start_range(const Out& o, long* lo, long* hi) {
    const size_t at = o.text.find("residents a layer: start ");
    if (at == std::string::npos) return false;
    return std::sscanf(o.text.c_str() + at, "residents a layer: start %ld..%ld", lo, hi) == 2;
}

/* The slab the run started from, "start resident N units": the floor must not move it. */
bool start_resident(const Out& o, long* n) {
    const size_t at = o.text.find(", start resident ");
    if (at == std::string::npos) return false;
    return std::sscanf(o.text.c_str() + at, ", start resident %ld units", n) == 1;
}

#define CHECK_RUN(o, cond)                                                                \
    do { const bool _ok = (cond); CHECK(_ok);                                              \
         if (!_ok) std::fprintf(stderr, "    exit %d\n%s\n", (o).code, (o).text.c_str()); } \
    while (0)

struct Fixture {
    std::string dir, path;
    bool ok = false;

    Fixture() {
        /* Brilliance: under the test's own directory (RAD_TEST_TMPDIR, rad_test.h), not /tmp. */
        dir = radtest::tmp_path("heatreplay");
        if (::mkdir(dir.c_str(), 0777) != 0 && errno != EEXIST) return;
        path = dir + "/synthetic.rt";
        const std::string b = trace_bytes();
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return;
        const bool wrote = std::fwrite(b.data(), 1, b.size(), f) == b.size();
        std::fclose(f);
        ok = wrote;
    }
    ~Fixture() {
        if (!dir.empty()) {
            std::remove(path.c_str());
            ::rmdir(dir.c_str());
        }
    }
};

}  // namespace

/* Without --layer-min the slab keeps the snapshot's own shape: nothing is raised, and the hot pair
 * keeps what heat gives it. */
TEST(the_replay_without_a_floor_keeps_the_snapshots_shape) {
    Fixture fx;
    REQUIRE(fx.ok);
    const Out o = run(fx.path, "");
    CHECK_RUN(o, o.code == 0);
    long lo = -1, hi = -1, res0 = -1;
    CHECK_RUN(o, start_range(o, &lo, &hi));
    CHECK_RUN(o, start_resident(o, &res0));
    CHECK_EQ(lo, 0L);
    CHECK_EQ(hi, 8L);
    CHECK_EQ(res0, 16L);
    CHECK_RUN(o, o.text.find("floor:") == std::string::npos);
}

/* A floor of 3: every layer holds at least 3 residents, the quiet pair is filled from the hot pair,
 * the slab is still 16 units, and the replay misses more than it did without the floor. */
TEST(a_layer_floor_fills_the_cold_layers_from_the_hot_ones_and_costs_hits) {
    Fixture fx;
    REQUIRE(fx.ok);
    const Out base = run(fx.path, "");
    const Out o = run(fx.path, "--layer-min 3");
    CHECK_RUN(o, o.code == 0);
    long lo = -1, hi = -1, res0 = -1;
    CHECK_RUN(o, start_range(o, &lo, &hi));
    CHECK_RUN(o, start_resident(o, &res0));
    CHECK(lo >= 3);
    CHECK_EQ(hi, 5L);   /* 8 + 8 = 16 residents, 4 a layer when flat: 3, 3, 5, 5 */
    CHECK_EQ(res0, 16L);
    CHECK_RUN(o, o.text.find("floor: 3 units a layer (achieved 3 at the start, enforced)") !=
                    std::string::npos);

    double h0 = 0, h = 0;
    CHECK_RUN(base, hit_of(base, &h0));
    CHECK_RUN(o, hit_of(o, &h));
    std::fprintf(stderr, "    hit %.4f with the floor, %.4f without\n", h, h0);
    CHECK(h < h0);
}

/* A floor the slab cannot hold is clamped to the one it can: 16 units over four layers is four a
 * layer, and a request for 100 raises every layer to exactly four rather than overshooting or
 * leaving the run frozen. */
TEST(a_floor_past_what_the_slab_holds_is_clamped) {
    Fixture fx;
    REQUIRE(fx.ok);
    const Out o = run(fx.path, "--layer-min 100");
    CHECK_RUN(o, o.code == 0);
    long lo = -1, hi = -1, res0 = -1;
    CHECK_RUN(o, start_range(o, &lo, &hi));
    CHECK_RUN(o, start_resident(o, &res0));
    CHECK_EQ(lo, 4L);
    CHECK_EQ(hi, 4L);
    CHECK_EQ(res0, 16L);
    CHECK_RUN(o, o.text.find("floor: 100 units a layer (achieved 4 at the start, enforced)") !=
                    std::string::npos);
    double h = 0;
    CHECK_RUN(o, hit_of(o, &h));
    CHECK(h > 0.0);
}

RAD_TEST_MAIN()
