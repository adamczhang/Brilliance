/* ngram_warm_test.cpp -- Brilliance: when --ngram-warm reads the n-gram table into the file cache.
 *
 * The warm (format/table_warm.h) reads a 47.68 GiB table at load on the machine it was made for, so
 * reading it at the wrong time costs real memory and time, and skipping it when it should run costs
 * about 9% of one-stream decode. The decision is a pure function of what the engine knows after the
 * weight load, and every case here is one of its branches:
 *
 *   - it reads only when asked, for a table on the disk tier read through the container's mapping;
 *   - a copy in RAM, a table on the card, Linux's direct-read row cache and disk=DIR's own file
 *     are skips with their reason, not reads;
 *   - it never forces: without the table's size plus the floor available, or with the available
 *     memory unreadable, it skips;
 *   - the ranks' ranges are read once: every rank publishes the same table.
 *
 * Host only: no card, no model, no file is read. */
#include "rad_test.h"
#include "rad_core.h"
#include "fleet/config.h"
#include "format/table_warm.h"

#include <string>
#include <utility>
#include <vector>

using namespace rad;

namespace {

constexpr int64_t GiB = (int64_t)1 << 30;
constexpr int64_t kTable = (int64_t)(47.68 * (double)GiB);   /* Flash-Next's table */

/* The case the option was made for: two cards, the table on the disk tier, 80 GiB available. */
NgramWarmFacts flashnext_2card() {
    NgramWarmFacts f;
    f.asked = true;
    f.table_bytes = kTable;
    f.resolved = TablePlacement::Disk;
    f.avail = 80 * GiB;
    f.floor = kNgramWarmFloor;
    return f;
}

bool says(const NgramWarmPlan& p, const char* what) { return p.why.find(what) != std::string::npos; }

}  // namespace

/* ---- when it reads ---------------------------------------------------------------------- */

TEST(the_disk_tier_with_room_is_warmed) {
    const NgramWarmPlan p = ngram_warm_plan(flashnext_2card());
    CHECK(p.warm);
    CHECK(says(p, "read from the drive"));
}

TEST(exactly_the_table_and_the_floor_is_enough) {
    NgramWarmFacts f = flashnext_2card();
    f.avail = kTable + kNgramWarmFloor;
    CHECK(ngram_warm_plan(f).warm);
    f.avail -= 1;
    CHECK(!ngram_warm_plan(f).warm);
}

TEST(the_floor_is_24_gib) { CHECK_EQ(kNgramWarmFloor, 24 * GiB); }

/* ---- when it does not ------------------------------------------------------------------- */

TEST(off_unless_asked) {
    NgramWarmFacts f = flashnext_2card();
    f.asked = false;
    const NgramWarmPlan p = ngram_warm_plan(f);
    CHECK(!p.warm);
    CHECK(says(p, "off"));
}

TEST(a_model_without_a_table_is_not_warmed) {
    NgramWarmFacts f = flashnext_2card();
    f.table_bytes = 0;
    const NgramWarmPlan p = ngram_warm_plan(f);
    CHECK(!p.warm);
    CHECK(says(p, "no n-gram table"));
}

TEST(a_copy_in_ram_or_on_the_card_is_not_warmed) {
    NgramWarmFacts f = flashnext_2card();
    f.resolved = TablePlacement::Ram;
    NgramWarmPlan p = ngram_warm_plan(f);
    CHECK(!p.warm);
    CHECK(says(p, "a copy in RAM"));
    f.resolved = TablePlacement::Vram;
    p = ngram_warm_plan(f);
    CHECK(!p.warm);
    CHECK(says(p, "on the card"));
}

TEST(direct_reads_skip_the_page_cache_so_are_not_warmed) {
    NgramWarmFacts f = flashnext_2card();
    f.direct_reads = true;
    const NgramWarmPlan p = ngram_warm_plan(f);
    CHECK(!p.warm);
    CHECK(says(p, "direct I/O"));
}

TEST(a_table_in_its_own_file_is_not_warmed) {
    NgramWarmFacts f = flashnext_2card();
    f.own_file = true;
    const NgramWarmPlan p = ngram_warm_plan(f);
    CHECK(!p.warm);
    CHECK(says(p, "disk=DIR"));
}

TEST(short_of_the_table_and_the_floor_it_skips_with_the_figures) {
    NgramWarmFacts f = flashnext_2card();
    f.avail = 60 * GiB;   /* the table fits, the floor beside it does not */
    const NgramWarmPlan p = ngram_warm_plan(f);
    CHECK(!p.warm);
    CHECK(says(p, "60.00 GiB"));
    CHECK(says(p, "47.68 GiB"));
    CHECK(says(p, "24.00 GiB to leave free"));
}

TEST(a_larger_headroom_raises_the_floor) {
    NgramWarmFacts f = flashnext_2card();
    f.floor = 40 * GiB;   /* --host-headroom-mib above 24 GiB */
    CHECK(!ngram_warm_plan(f).warm);   /* 47.68 + 40 > 80 */
}

TEST(unreadable_memory_is_a_skip_not_a_read) {
    NgramWarmFacts f = flashnext_2card();
    f.avail = -1;
    const NgramWarmPlan p = ngram_warm_plan(f);
    CHECK(!p.warm);
    CHECK(says(p, "cannot be read"));
}

/* ---- what it reads ---------------------------------------------------------------------- */

TEST(every_rank_publishes_the_same_table_and_it_is_read_once) {
    const auto r = ngram_warm_ranges({ { 4096, kTable }, { 4096, kTable } });
    CHECK_EQ(r.size(), size_t(1));
    CHECK_EQ(r.at(0).first, (int64_t)4096);
    CHECK_EQ(r.at(0).second, kTable);
}

TEST(the_table_and_its_scale_stay_two_ranges_unless_they_touch) {
    /* The scale is two bytes elsewhere in the container. */
    auto r = ngram_warm_ranges({ { 1 << 20, 2 }, { 64 << 20, 1 << 20 }, { 1 << 20, 2 } });
    CHECK_EQ(r.size(), size_t(2));
    CHECK_EQ(r.at(0).first, (int64_t)(1 << 20));
    CHECK_EQ(r.at(0).second, (int64_t)2);
    CHECK_EQ(r.at(1).first, (int64_t)(64 << 20));
    /* Touching and overlapping ranges merge; empty and negative ones are dropped. */
    r = ngram_warm_ranges({ { 100, 50 }, { 150, 50 }, { 120, 10 }, { 500, 0 }, { -8, 16 } });
    CHECK_EQ(r.size(), size_t(1));
    CHECK_EQ(r.at(0).first, (int64_t)100);
    CHECK_EQ(r.at(0).second, (int64_t)100);
    CHECK(ngram_warm_ranges({}).empty());
}

/* ---- the setting ------------------------------------------------------------------------ */

TEST(the_flag_turns_it_on_and_the_default_is_off) {
    Config off;
    CHECK(!off.ngram_warm);
    Argv a({ "--model", "/m/x.rad", "--ngram-warm" });
    Config c;
    CHECK_EQ(config_parse(a.argc(), a.argv.data(), &c), RAD_OK);
    CHECK(c.ngram_warm);
}

TEST(the_profile_key_is_ngram_warm_true_or_false) {
    fleet::FleetConfig on;
    CHECK_EQ(fleet::parse("models:\n  q:\n    model: /m/q.rad\n    ngram-warm: true\n", "p.yaml", &on),
             RAD_OK);
    CHECK(on.models.at(0).cfg.ngram_warm);
    fleet::FleetConfig off;
    CHECK_EQ(fleet::parse("models:\n  q:\n    model: /m/q.rad\n    ngram-warm: false\n", "p.yaml",
                          &off), RAD_OK);
    CHECK(!off.models.at(0).cfg.ngram_warm);
    fleet::FleetConfig bad;
    CHECK(fleet::parse("models:\n  q:\n    model: /m/q.rad\n    ngram-warm: yes\n", "p.yaml", &bad) !=
          RAD_OK);
}

RAD_TEST_MAIN()
