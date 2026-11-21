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
#include "format/radfile.h"
#include "format/table_warm.h"

#include <cstdio>
#include <cstdint>

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

/* ---- how it reads (ngram-warm-tune) ------------------------------------------------------- */

TEST(the_default_reads_through_a_handle_per_reader_and_alpha3_is_kept) {
    const TableWarmOpts d;
    CHECK(d.io == WarmIo::Handle);
    CHECK_EQ(d.readers, 16);
    CHECK_EQ(d.piece, (int64_t)64 << 20);
    const TableWarmOpts a = table_warm_opts_alpha3();
    CHECK(a.io == WarmIo::Pread);
    CHECK_EQ(a.readers, 8);
    CHECK_EQ(a.piece, (int64_t)16 << 20);
    CHECK_EQ(table_warm_opts_str(d), std::string("io=handle readers=16 piece=64.00 MiB"));
}

TEST(the_io_spec_sets_any_subset_and_refuses_the_rest) {
    TableWarmOpts o;
    std::string why;
    CHECK_EQ(table_warm_opts_parse("io=pread,readers=8,piece=16M", &o, &why), RAD_OK);
    CHECK(o.io == WarmIo::Pread);
    CHECK_EQ(o.readers, 8);
    CHECK_EQ(o.piece, (int64_t)16 << 20);
    /* A subset keeps the rest; spaces separate as commas do; MiB and G spell sizes. */
    CHECK_EQ(table_warm_opts_parse("io=prefetch piece=1GiB", &o, &why), RAD_OK);
    CHECK(o.io == WarmIo::Prefetch);
    CHECK_EQ(o.readers, 8);
    CHECK_EQ(o.piece, (int64_t)1 << 30);
    CHECK_EQ(table_warm_opts_parse("", &o, &why), RAD_OK);
    CHECK(o.io == WarmIo::Prefetch);
    /* Refused, and `o` left as it was. */
    for (const char* bad : { "io=seq", "readers=0", "readers=65", "piece=512K", "piece=2G",
                             "piece=12Q", "threads=4", "readers=" }) {
        why.clear();
        CHECK_EQ(table_warm_opts_parse(bad, &o, &why), RAD_E_INVAL);
        CHECK(!why.empty());
    }
    CHECK(o.io == WarmIo::Prefetch);
    CHECK_EQ(o.piece, (int64_t)1 << 30);
}

/* A real read of a small container's weight range in each mode: every byte of the range is
 * counted once, whatever the split into readers and pieces, and a range past the end fails with
 * what was read before it. */
TEST(each_mode_reads_the_whole_range_once) {
    const std::string path = radtest::tmp_path("ngram_warm_read.rad");
    {
        RadWriter w;
        CHECK_OK(w.begin(path.c_str()));
        w.set_arch("radtest");
        RadWriter::Weight x;
        x.name = "blk.1.ple_ngram.weight";
        x.enc = rad_enc_plain(RAD_BF16);
        x.rank = 2;
        x.shape[0] = 1536;    /* 1536 x 2048 bf16: 6 MiB, three 2 MiB-aligned spans of 2 MiB */
        x.shape[1] = 2048;
        x.layer = 1;
        std::vector<uint8_t> plane((size_t)1536 * 2048 * 2, 0x5a);
        const void* p[1] = { plane.data() };
        CHECK(w.add_weight(x, p) >= 0);
        CHECK_OK(w.finish());
    }
    RadFile f;
    REQUIRE(f.open(path.c_str()) >= 0);
    const RadFileEntry* e = nullptr;
    for (int64_t i = 0; i < f.entry_count(); ++i)
        if (std::string(f.str(f.entry(i).name)) == "blk.1.ple_ngram.weight") e = &f.entry(i);
    REQUIRE(e != nullptr);
    const RadFilePlane& pl = f.plane(*e, 0);
    const std::vector<std::pair<int64_t, int64_t>> rg = { { (int64_t)pl.offset, (int64_t)pl.bytes } };
    for (const char* spec : { "io=pread,readers=8,piece=16M", "io=handle,readers=3,piece=1M",
                              "io=handle,readers=16,piece=64M", "io=prefetch,readers=4,piece=1M" }) {
        TableWarmOpts o;
        std::string why;
        REQUIRE_EQ(table_warm_opts_parse(spec, &o, &why), RAD_OK);
        int64_t bytes = -1;
        double secs = -1;
        CHECK_EQ(table_warm_read(f, rg, &bytes, &secs, &why, o), RAD_OK);
        CHECK_EQ(bytes, (int64_t)pl.bytes);
        CHECK(secs >= 0);
        CHECK(why.empty());
    }
    /* Past the end: an error and a reason (pread and handle; prefetch reads nothing itself). */
    for (const char* spec : { "io=pread", "io=handle" }) {
        TableWarmOpts o;
        std::string why;
        REQUIRE_EQ(table_warm_opts_parse(spec, &o, &why), RAD_OK);
        int64_t bytes = -1;
        double secs = -1;
        const std::vector<std::pair<int64_t, int64_t>> past = { { f.size() - 4096, 1 << 20 } };
        CHECK(table_warm_read(f, past, &bytes, &secs, &why, o) < 0);
        CHECK(!why.empty());
        CHECK(bytes < (int64_t)(1 << 20));
    }
    f.close();
    std::remove(path.c_str());
}

RAD_TEST_MAIN()
