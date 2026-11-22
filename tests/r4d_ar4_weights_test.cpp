/* r4d_ar4_weights_test.cpp -- the four-rank host-mailbox two-shot's link weights
 * (libr4d/r4d_ar4_weights.h, Brilliance tp4-link-weight).
 *
 *   Setting   BRILLIANCE_AR4_SHARD_WEIGHTS: auto, off/even, four weights by rank, bus:weight pairs;
 *             anything else refused
 *   Rule      the fast tier reduces: the test machine's links (Gen5 x4, Gen5 x8, x8, Gen4 x4 by rank) give both
 *             risers no shard and the x8 pair a half each, whatever bus 32 trained at; equal (or
 *             nearly equal) links, a tier of one and an unknown rate keep the even split
 *   Tables    the plain and coded (whole-wave) tables, their fit in the even build's slot, and the
 *             even table equal to r4d_ar_shard_table's
 *   Model     a four-rank host model of both two-shot forms that follows the kernels' block
 *             partition, per-shard limits and slot regions (r4d_arw_span, r4d_arw_lim): the output
 *             bits under the test machine's and other tables equal the even split's and a direct fold; every
 *             word reaches every rank once; every push lands inside its region of the slot; and each
 *             push check covers exactly the words its push wrote (so the protection's check words
 *             agree, and a damaged word is always seen)
 */
#include "rad_test.h"

#include "r4d_ar_shards.h"
#include "r4d_ar4_weights.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

std::string ws(const int* w) {
    return std::to_string(w[0]) + "," + std::to_string(w[1]) + "," + std::to_string(w[2]) + "," +
           std::to_string(w[3]);
}

std::string from_rates(double a, double b, double c, double d) {
    const double g[4] = { a, b, c, d };
    int w[4] = { -1, -1, -1, -1 };
    if (!r4d_ar4_weights_from_rates(g, w)) return "even";
    return ws(w);
}

/* The test machine's links as 0.2.x's probe measured them (GB/s): Gen5 x8 28, Gen5 x4 14, the Gen4 x4 riser 7.
 * Rank order on the 0.3.0 benches was buses 6, 3, 9, 32. */
constexpr double kX8 = 28.0, kX4 = 14.0, kRiser = 7.0;

}  // namespace

TEST(the_setting_parses_auto_off_ranks_and_buses) {
    CHECK_EQ(r4d_ar4_spec_parse(nullptr).kind, (int)R4D_AR4_SPEC_AUTO);
    CHECK_EQ(r4d_ar4_spec_parse("").kind, (int)R4D_AR4_SPEC_AUTO);
    CHECK_EQ(r4d_ar4_spec_parse("auto").kind, (int)R4D_AR4_SPEC_AUTO);
    CHECK_EQ(r4d_ar4_spec_parse("off").kind, (int)R4D_AR4_SPEC_EVEN);
    CHECK_EQ(r4d_ar4_spec_parse("even").kind, (int)R4D_AR4_SPEC_EVEN);
    const R4DAr4Spec r = r4d_ar4_spec_parse("1,1,1,0");
    REQUIRE(r.kind == R4D_AR4_SPEC_RANKS);
    CHECK_EQ(ws(r.w), std::string("1,1,1,0"));
    const R4DAr4Spec b = r4d_ar4_spec_parse("3:2,9:2,6:1,32:0");
    REQUIRE(b.kind == R4D_AR4_SPEC_BUSES);
    CHECK_EQ(b.n, 4);
    const int buses[4] = { 6, 3, 9, 32 };   /* rank order */
    int w[4];
    REQUIRE(r4d_ar4_spec_by_bus(b, buses, w));
    CHECK_EQ(ws(w), std::string("1,2,2,0"));
    /* a named bus no rank is on is allowed; a rank whose bus is not named is not */
    const R4DAr4Spec extra = r4d_ar4_spec_parse("3:1,9:1,6:1,32:0,12:4");
    REQUIRE(extra.kind == R4D_AR4_SPEC_BUSES);
    CHECK(r4d_ar4_spec_by_bus(extra, buses, w));
    const R4DAr4Spec missing = r4d_ar4_spec_parse("3:1,9:1,6:1");
    REQUIRE(missing.kind == R4D_AR4_SPEC_BUSES);
    CHECK(!r4d_ar4_spec_by_bus(missing, buses, w));
    const int unknown[4] = { 6, 3, -1, 32 };
    CHECK(!r4d_ar4_spec_by_bus(b, unknown, w));
    const R4DAr4Spec zero = r4d_ar4_spec_parse("3:0,9:0,6:0,32:1");
    REQUIRE(zero.kind == R4D_AR4_SPEC_BUSES);
    const int only3[4] = { 6, 3, 9, 3 };
    CHECK(!r4d_ar4_spec_by_bus(r4d_ar4_spec_parse("3:0,9:0,6:0,32:1"), only3, w));
    for (const char* bad : { "1,1,1", "1,1,1,0,1", "0,0,0,0", "1, 1,1,0", "-1,1,1,1", "1,1,1,0x",
                             "65536,1,1,1", "on", "3:1,3:1,6:1,32:0", "3:1,9:1,6:1,32:", "3=1",
                             "256:1,9:1,6:1,32:0", "3:0,9:0,6:0,32:0", "1,1,1,0,", ",1,1,1,0" })
        CHECK_EQ(r4d_ar4_spec_parse(bad).kind, (int)R4D_AR4_SPEC_BAD);
}

TEST(reds_links_give_both_risers_no_shard_and_the_x8_pair_a_half_each) {
    /* buses 6, 3, 9, 32: the step breakdown found both risers about 60-70 ms a step heavier alike */
    CHECK_EQ(from_rates(kX4, kX8, kX8, kRiser), std::string("0,1,1,0"));
    /* bus 32 trained at Gen5 x4: the same table */
    CHECK_EQ(from_rates(kX4, kX8, kX8, kX4), std::string("0,1,1,0"));
    /* the rule follows the rates, not the rank: any order */
    CHECK_EQ(from_rates(kX8, kRiser, kX4, kX8), std::string("1,0,0,1"));
    CHECK_EQ(from_rates(kRiser, kX4, kX8, kX8), std::string("0,0,1,1"));
    /* as a probe reads them, a little off the nominal rates */
    CHECK_EQ(from_rates(13.6, 26.9, 27.3, 6.4), std::string("0,1,1,0"));
    CHECK_EQ(from_rates(14.2, 28.4, 23.0, 7.1), std::string("0,1,1,0"));   /* 23.0: inside the tier */
    /* each riser carries the message once each way instead of 1.5 times; each x8 card twice */
    const int w[4] = { 0, 1, 1, 0 };
    CHECK_NEAR(r4d_ar4_link_load(nullptr, 3), 1.5, 1e-12);
    CHECK_NEAR(r4d_ar4_link_load(w, 3), 1.0, 1e-12);
    CHECK_NEAR(r4d_ar4_link_load(w, 0), 1.0, 1e-12);
    CHECK_NEAR(r4d_ar4_link_load(w, 1), 2.0, 1e-12);
}

TEST(three_fast_links_share_the_work_and_one_fast_link_keeps_it_even) {
    CHECK_EQ(from_rates(kX8, kX8, kX8, kRiser), std::string("1,1,1,0"));
    CHECK_EQ(from_rates(kX4, kX8, kX4, kX4), std::string("even"));   /* a tier of one */
}

TEST(equal_links_keep_the_even_split) {
    CHECK_EQ(from_rates(kX8, kX8, kX8, kX8), std::string("even"));
    CHECK_EQ(from_rates(kRiser, kRiser, kRiser, kRiser), std::string("even"));
    /* a probe's scatter on four equal links */
    CHECK_EQ(from_rates(28.4, 28.1, 27.9, 28.3), std::string("even"));
    CHECK_EQ(from_rates(26.0, 28.4, 27.0, 28.0), std::string("even"));
}

TEST(an_unknown_rate_keeps_the_even_split) {
    CHECK_EQ(from_rates(0.0, kX8, kX8, kRiser), std::string("even"));
    CHECK_EQ(from_rates(kX4, kX8, -1.0, kRiser), std::string("even"));
    CHECK_EQ(from_rates(kX4, kX8, NAN, kRiser), std::string("even"));
}

TEST(the_even_table_is_the_shard_tables) {
    for (long long n16 : { 4LL, 5LL, 7LL, 1000LL, 655360LL, 1966080LL, 1966083LL }) {
        long long off[5];
        const long long mx = r4d_ar_shard_table(n16, 4, nullptr, off);
        R4DArwTab t;
        r4d_arw_tab(n16, nullptr, 1, &t);
        for (int s = 0; s <= 4; ++s) CHECK_EQ((long long)t.off[s], off[s]);
        CHECK_EQ((long long)t.sh_max, mx);
        const int w[4] = { 1, 1, 1, 0 };
        const long long mxw = r4d_ar_shard_table(n16, 4, w, off);
        r4d_arw_tab(n16, w, 1, &t);
        for (int s = 0; s <= 4; ++s) CHECK_EQ((long long)t.off[s], off[s]);
        CHECK_EQ((long long)t.sh_max, mxw);
    }
}

TEST(reds_tables_fit_the_even_builds_slot_for_a_pipelined_piece) {
    /* Flash-Next TP4: a 6,144-row instance (2,560 wide, bf16), pipelined pieces of 2,048 rows; the
     * shipped table on the test machine (0,1,1,0) and a three-card one (1,1,1,0) */
    const long long slot16 = 6144LL * 2560 * 2 / 16;
    const long long piece16 = 2048LL * 2560 * 2 / 16;
    const int tables[2][4] = { { 0, 1, 1, 0 }, { 1, 1, 1, 0 } };
    for (const auto& w : tables) {
        R4DArwTab t;
        r4d_arw_tab(piece16, w, 32, &t);
        CHECK_EQ(t.off[4], (int)piece16);
        for (int s = 0; s < 4; ++s) CHECK_EQ((t.off[s + 1] - t.off[s]) % 32, 0);
        for (int s = 0; s < 4; ++s)
            if (!w[s]) CHECK_EQ(t.off[s + 1] - t.off[s], 0);   /* a rank weighted 0 owns nothing */
        CHECK(r4d_arw_tab_fits(t, true, slot16));
        CHECK(r4d_arw_tab_fits(t, false, slot16));
        /* the plain form fits a whole message (two halves at most); the coded form of a whole
         * 6,144-row message does not, and runs even, with the same bits */
        r4d_arw_tab(slot16, w, 1, &t);
        CHECK(r4d_arw_tab_fits(t, false, slot16));
        r4d_arw_tab(slot16, w, 32, &t);
        CHECK(!r4d_arw_tab_fits(t, true, slot16));
    }
    R4DArwTab t;
    r4d_arw_tab(slot16, nullptr, 32, &t);
    CHECK(r4d_arw_tab_fits(t, true, slot16));
    /* a table giving one rank everything never fits the plain form of a full message */
    const int one[4] = { 1, 0, 0, 0 };
    r4d_arw_tab(slot16, one, 1, &t);
    CHECK(!r4d_arw_tab_fits(t, false, slot16));
}

TEST(the_coded_region_restates_the_kernels) {
    /* r4d_ar_wide_ts.hip's arwc_region_bytes; tw.hip static_asserts the same at three sizes */
    CHECK_EQ(r4d_arw_coded_region_bytes(1), (size_t)256);
    CHECK_EQ(r4d_arw_coded_region_bytes(32), (size_t)((12 * 32 + 4 + 15) / 16 * 16 + 16 * 32 + 255) / 256 * 256);
    CHECK(2 * r4d_arw_coded_region_bytes(163840) <= (size_t)655360 * 16);
}

/* ---- the four-rank host model ----------------------------------------------------------------- */
namespace {

uint16_t f2bf(float f) {   /* round to nearest even, as __hip_bfloat16's conversion */
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x7fffffu)) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct W16 { uint16_t v[8]; };   /* one 16-byte word: eight bf16 values */

/* One push or check of a block: words [lo, hi) of a shard, from rank `src` to rank `dst`, in a hop. */
struct Range { int hop, b, src, dst, lo, hi; };

struct Model {
    std::vector<W16> out[4];
    std::vector<Range> pushes, checks;
    bool in_bounds = true;    /* every push inside its region, region B inside the slot */
    int assembled_twice = 0;  /* output words written more than once */
};

/* The two-shot over table `t` as the kernels run it: nb blocks splitting sh_max (whole waves when
 * coded), each with its own limit per shard; hop 1 pushes this rank's words of every other shard to
 * that shard's owner (region A of the owner's slot for this source), the owner checks and folds
 * (fp32, ascending rank), hop 2 pushes the folded shard to every peer's region B, every peer checks
 * and assembles. The coded form's regions are laid out as the kernel does (capacity sh_max, region
 * B at its coded region's size); the codec is lossless, so the model carries the words themselves. */
Model run(const std::vector<W16> (&in)[4], const R4DArwTab& t, bool coded, int nb, long long slot16) {
    Model m;
    const int unit = coded ? 32 : 1;
    const int SL = t.sh_max;
    const size_t region_b_off = coded ? r4d_arw_coded_region_bytes(SL) : (size_t)SL * 16;
    const size_t region_bytes = coded ? r4d_arw_coded_region_bytes(SL) : (size_t)SL * 16;
    if (region_b_off + region_bytes > (size_t)slot16 * 16) m.in_bounds = false;
    /* slot[dst][src]: region A then B, as words (the coded payload's size is checked above) */
    std::vector<W16> slotA[4][4], slotB[4][4];
    for (int d = 0; d < 4; ++d)
        for (int s = 0; s < 4; ++s) { slotA[d][s].assign((size_t)SL, W16{}); slotB[d][s].assign((size_t)SL, W16{}); }
    const long long n16 = (long long)in[0].size();
    for (int r = 0; r < 4; ++r) m.out[r].assign((size_t)n16, W16{});
    std::vector<int> writes[4];
    for (int r = 0; r < 4; ++r) writes[r].assign((size_t)n16, 0);

    for (int b = 0; b < nb; ++b) {
        int w0 = 0, w1 = 0;
        r4d_arw_span(SL, nb, b, unit, &w0, &w1);
        /* hop 1: rank r pushes its words of shard `o` to o; o checks what it reads of its shard */
        for (int r = 0; r < 4; ++r)
            for (int o = 0; o < 4; ++o) {
                if (o == r) continue;
                const int lim = r4d_arw_lim(t.off, w1, o);
                m.pushes.push_back({ 1, b, r, o, w0, lim > w0 ? lim : w0 });
                for (int k = w0; k < lim; ++k) {
                    if (k >= SL) m.in_bounds = false;
                    else slotA[o][r][(size_t)k] = in[r][(size_t)(t.off[o] + k)];
                }
                const int my = r4d_arw_lim(t.off, w1, o);   /* the owner's check: its own limit */
                m.checks.push_back({ 1, b, r, o, w0, my > w0 ? my : w0 });
            }
        /* the fold and hop 2: rank o reduces its shard and pushes it to every peer */
        for (int o = 0; o < 4; ++o) {
            const int lim = r4d_arw_lim(t.off, w1, o);
            for (int k = w0; k < lim; ++k) {
                W16 res;
                for (int j = 0; j < 8; ++j) {
                    float acc = 0.0f;
                    for (int r = 0; r < 4; ++r)
                        acc += bf2f(r == o ? in[r][(size_t)(t.off[o] + k)].v[j] : slotA[o][r][(size_t)k].v[j]);
                    res.v[j] = f2bf(acc);
                }
                m.out[o][(size_t)(t.off[o] + k)] = res;
                ++writes[o][(size_t)(t.off[o] + k)];
                for (int p = 0; p < 4; ++p)
                    if (p != o) slotB[p][o][(size_t)k] = res;
            }
            for (int p = 0; p < 4; ++p)
                if (p != o) m.pushes.push_back({ 2, b, o, p, w0, lim > w0 ? lim : w0 });
        }
        /* hop 2's check and assemble: rank p reads shard o's words from o's region B */
        for (int p = 0; p < 4; ++p)
            for (int o = 0; o < 4; ++o) {
                if (o == p) continue;
                const int lim = r4d_arw_lim(t.off, w1, o);
                m.checks.push_back({ 2, b, o, p, w0, lim > w0 ? lim : w0 });
                for (int k = w0; k < lim; ++k) {
                    m.out[p][(size_t)(t.off[o] + k)] = slotB[p][o][(size_t)k];
                    ++writes[p][(size_t)(t.off[o] + k)];
                }
            }
    }
    for (int r = 0; r < 4; ++r)
        for (long long k = 0; k < n16; ++k) m.assembled_twice += writes[r][(size_t)k] != 1;
    return m;
}

/* Contributions with the codec's hard cases: zeros, subnormals, infs and NaNs, wide exponent
 * ranges, values that cancel. */
void contributions(long long n16, unsigned seed, std::vector<W16> (&in)[4]) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-4.0f, 4.0f);
    for (int r = 0; r < 4; ++r) {
        in[r].assign((size_t)n16, W16{});
        for (long long k = 0; k < n16; ++k)
            for (int j = 0; j < 8; ++j) {
                const unsigned pick = g() % 64;
                uint16_t v;
                if (pick == 0) v = 0;
                else if (pick == 1) v = 0x0001;              /* subnormal */
                else if (pick == 2) v = 0x7f80;              /* inf */
                else if (pick == 3) v = (uint16_t)(0x3f80 + (g() % 128));
                else if (pick == 4) v = (uint16_t)(0x0080 + (g() % 0x7e00));
                else v = f2bf(u(g) * std::ldexp(1.0f, (int)(g() % 9) - 4));
                in[r][(size_t)k].v[j] = v;
            }
    }
}

bool same_bits(const std::vector<W16>& a, const std::vector<W16>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(W16)) == 0);
}

/* The direct fold of every word (the one-shot's): what every table must give. */
std::vector<W16> direct(const std::vector<W16> (&in)[4]) {
    std::vector<W16> o(in[0].size());
    for (size_t k = 0; k < o.size(); ++k)
        for (int j = 0; j < 8; ++j) {
            float acc = 0.0f;
            for (int r = 0; r < 4; ++r) acc += bf2f(in[r][k].v[j]);
            o[k].v[j] = f2bf(acc);
        }
    return o;
}

/* Every push check covers exactly the words its push wrote (the same block, hop, pair and range),
 * so the check words the writer sends match what the reader hashes, and no word is used unchecked. */
bool checks_match_pushes(const Model& m) {
    std::map<std::string, std::pair<int, int>> pu;
    auto key = [](const Range& r) {
        return std::to_string(r.hop) + "/" + std::to_string(r.b) + "/" + std::to_string(r.src) + ">" +
               std::to_string(r.dst);
    };
    for (const Range& r : m.pushes) pu[key(r)] = { r.lo, r.hi };
    if (pu.size() != m.pushes.size() || m.checks.size() != m.pushes.size()) return false;
    for (const Range& c : m.checks) {
        const auto it = pu.find(key(c));
        if (it == pu.end() || it->second.first != c.lo || it->second.second != c.hi) return false;
    }
    return true;
}

}  // namespace

TEST(every_table_gives_the_even_splits_bits_on_four_host_ranks) {
    const int tables[][4] = { { 0, 1, 1, 0 }, { 1, 1, 1, 0 }, { 2, 2, 1, 0 }, { 0, 0, 0, 1 },
                              { 1, 0, 0, 0 }, { 4, 4, 1, 1 }, { 3, 1, 2, 7 }, { 1, 1, 1, 1 } };
    /* sizes in 16-byte words, whole waves a rank for the coded form; slots as large as the message */
    for (long long n16 : { 128LL, 384LL, 4096LL, 32768LL + 128LL }) {
        std::vector<W16> in[4];
        contributions(n16, (unsigned)n16, in);
        const std::vector<W16> want = direct(in);
        for (bool coded : { false, true }) {
            if (coded && n16 % 128) continue;
            const long long slot16 = 4 * n16;   /* room for any table, so every one is exercised */
            R4DArwTab even;
            r4d_arw_tab(n16, nullptr, coded ? 32 : 1, &even);
            for (int nb : { 1, 3, 4, 8 }) {
                const Model e = run(in, even, coded, nb, slot16);
                for (int r = 0; r < 4; ++r) CHECK(same_bits(e.out[r], want));
                for (const auto& w : tables) {
                    R4DArwTab t;
                    r4d_arw_tab(n16, w, coded ? 32 : 1, &t);
                    const Model m = run(in, t, coded, nb, slot16);
                    CHECK(m.in_bounds);
                    CHECK_EQ(m.assembled_twice, 0);
                    CHECK(checks_match_pushes(m));
                    for (int r = 0; r < 4; ++r) {
                        if (!same_bits(m.out[r], e.out[r]))
                            ::radtest::fail(__FILE__, __LINE__,
                                            "table " + ws(w) + (coded ? " coded" : " plain") + " n16 " +
                                                std::to_string(n16) + " nb " + std::to_string(nb) +
                                                " rank " + std::to_string(r) + ": bits differ from even");
                        ++::radtest::checks();
                    }
                }
            }
        }
    }
}

TEST(a_rank_with_no_shard_still_sends_and_receives_everything) {
    const long long n16 = 1024;
    std::vector<W16> in[4];
    contributions(n16, 7u, in);
    const std::vector<W16> want = direct(in);
    const int tables[2][4] = { { 1, 1, 1, 0 }, { 0, 1, 1, 0 } };
    for (const auto& w : tables) {
        R4DArwTab t;
        r4d_arw_tab(n16, w, 32, &t);
        const Model m = run(in, t, true, 4, n16);
        for (int z = 0; z < 4; ++z) {
            if (w[z]) continue;
            REQUIRE(t.off[z + 1] - t.off[z] == 0);
            long long sent1 = 0, recv2 = 0, sent2 = 0;
            for (const Range& p : m.pushes) {
                if (p.hop == 1 && p.src == z) sent1 += p.hi - p.lo;
                if (p.hop == 2 && p.dst == z) recv2 += p.hi - p.lo;
                if (p.hop == 2 && p.src == z) sent2 += p.hi - p.lo;
            }
            /* its link carries the message once each way: its whole contribution out, the result in */
            CHECK_EQ(sent1, n16);
            CHECK_EQ(recv2, n16);
            CHECK_EQ(sent2, 0LL);
            CHECK(same_bits(m.out[z], want));
        }
    }
}

TEST(the_coded_blocks_split_whole_waves) {
    for (int sh : { 32, 96, 4096, 21856 })
        for (int nb : { 1, 2, 3, 4, 7 }) {
            int prev = 0;
            for (int b = 0; b < nb; ++b) {
                int w0 = 0, w1 = 0;
                r4d_arw_span(sh, nb, b, 32, &w0, &w1);
                CHECK_EQ(w0 % 32, 0);
                if (w1 > w0) {
                    CHECK_EQ(w0, prev);
                    prev = w1;
                }
            }
            CHECK_EQ(prev, sh);
        }
}

RAD_TEST_MAIN()
