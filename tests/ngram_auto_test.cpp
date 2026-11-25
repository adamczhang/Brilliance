/* ngram_auto_test.cpp -- Brilliance: where --ngram-placement auto puts an n-gram table
 * (core/place/ngram_auto.h), on the figures of the 2026-11-22 incident.
 *
 * That afternoon a one-card Flash-Next start on the test machine (125.52 GiB of RAM) planned its 47.68 GiB
 * table as a pinned copy on top of a 42 GiB expert pool with 99.79 GiB available, keeping only
 * the 4 GiB --host-headroom-mib to spare; the next GPU call hung after the pin. The cases:
 *
 *   - with the 4 GiB headroom alone (the rule before) the copy fit, which is what auto chose;
 *   - with the floor (max of 24 GiB and the headroom) it does not, so auto reads the file, and
 *     not on a freshly booted, idle test machine either;
 *   - the pin limit did not refuse it on one card (96.67 GiB) and does on two (46.83 GiB);
 *   - on Windows a table larger than a quarter of RAM is read from the file even where it fits,
 *     a small table is still pinned, and Linux keeps the copy;
 *   - the card comes first, and with no host gather and no room there is nowhere.
 *
 * Host only: arithmetic on figures handed in; no card, no model, no file. */
#include "rad_test.h"
#include "place/ngram_auto.h"

#include <cstdint>

using namespace rad;

namespace {

constexpr int64_t kMiB = (int64_t)1 << 20;
constexpr int64_t kGiB = (int64_t)1 << 30;
constexpr int64_t gib(double g) { return (int64_t)(g * (double)kGiB); }

/* The test machine on 11-22, from the incident's server logs. */
constexpr int64_t kTotal = gib(125.52);      /* "RAM 125.52 GiB" */
constexpr int64_t kTable = gib(47.68);       /* the n-gram table */
constexpr int64_t kHeadroom = 4 * kGiB;      /* --host-headroom-mib's default */
/* Still to be allocated after the table at plan time: the 43000 MiB pinned pool (41.99 GiB), the
 * embedding (1.18), the prefix cache's host tier (1.00), the checkpoint store and arenas (943.80
 * MiB) and the encoder towers (857 MiB): 45.94 GiB. */
constexpr int64_t kStill = gib(45.94);
constexpr int64_t kAvailAtPlan = gib(99.79);  /* the RAM log's first sample, 2:45:55 PM */
constexpr int64_t kAvailIdle = gib(111.7);    /* after the reboot, nothing running */
/* What amdgpu_pin_cap hands the planner: the backend's limit (99.67 GiB budget / devices - 2 GiB)
 * less the GiB kept for the load's staging windows. */
constexpr int64_t kBudget = gib(99.67);
constexpr int64_t kCapOneCard = kBudget / 1 - 2 * kGiB - kGiB;   /* 96.67 GiB */
constexpr int64_t kCapTwoCards = kBudget / 2 - 2 * kGiB - kGiB;  /* 46.83 GiB */

int64_t aligned(int64_t b) { return (b + 2 * kMiB - 1) / (2 * kMiB) * (2 * kMiB); }

NgramAutoIn red_one_card(bool room) {
    NgramAutoIn in;
    in.device = true;          /* libr4d gathers on the card */
    in.host = true;            /* libavx gathers on the host */
    in.card_holds = false;     /* "the card cannot hold the other weights with it" */
    in.ram_room = room;
    in.file_preferred = true;  /* Windows */
    in.need = aligned(kTable);
    in.total = kTotal;
    return in;
}

}  /* namespace */

TEST(the_floor_is_24_gib_or_the_headroom_if_larger) {
    CHECK_EQ(ngram_auto_margin(kHeadroom), 24 * kGiB);
    CHECK_EQ(ngram_auto_margin(0), 24 * kGiB);
    CHECK_EQ(ngram_auto_margin(32 * kGiB), 32 * kGiB);
    CHECK_EQ(kHostRamFloor, 24 * kGiB);
}

TEST(on_11_22_the_headroom_alone_let_the_copy_fit) {
    /* 99.79 - 45.94 - 4 = 49.85 GiB for a 47.69 GiB copy: the rule before chose ram, leaving about
     * 2 GiB above the headroom. An explicit `ram` still keeps only the headroom. */
    CHECK(!host_mem_short(kAvailAtPlan, kStill, kHeadroom, aligned(kTable)));
}

TEST(on_11_22_the_floor_sends_the_table_to_the_file) {
    const int64_t m = ngram_auto_margin(kHeadroom);
    /* 99.79 - 45.94 - 24 = 29.85 GiB, short of 47.69. */
    CHECK(host_mem_short(kAvailAtPlan, kStill, m, aligned(kTable)));
    CHECK(ngram_auto_choose(red_one_card(false)) == NgramAutoTier::Disk);
}

TEST(an_idle_red_cannot_pin_it_on_one_card_either) {
    /* 111.7 - 45.94 - 24 = 41.76 GiB, still short: one-card Flash-Next never pins its table. */
    CHECK(host_mem_short(kAvailIdle, kStill, ngram_auto_margin(kHeadroom), aligned(kTable)));
}

TEST(the_pin_limit_allowed_it_on_one_card_and_refuses_it_on_two) {
    CHECK_EQ(kCapOneCard, gib(99.67) - 3 * kGiB);
    /* One card: 45.94 + 47.69 = 93.63 GiB under 96.67. The limit was not what was passed. */
    CHECK(!pin_cap_short(kCapOneCard, 0, kStill, aligned(kTable)));
    /* Two cards: the table alone is past 46.83 GiB, whatever else is pinned. */
    CHECK(kCapTwoCards < aligned(kTable));
    CHECK(pin_cap_short(kCapTwoCards, 0, 0, aligned(kTable)));
    /* No limit known (Linux, or a backend with none): never short. */
    CHECK(!pin_cap_short(-1, 0, kStill, aligned(kTable)));
    CHECK(!pin_cap_short(0, 0, kStill, aligned(kTable)));
}

TEST(on_windows_a_table_over_a_quarter_of_ram_is_read_from_the_file_even_where_it_fits) {
    /* 47.69 GiB is 38% of 125.52 GiB. */
    CHECK(ngram_pin_too_large(aligned(kTable), kTotal));
    CHECK(ngram_auto_choose(red_one_card(true)) == NgramAutoTier::Disk);
    /* A 512 GiB host pins it when the floor allows. */
    NgramAutoIn big = red_one_card(true);
    big.total = 512 * kGiB;
    CHECK(ngram_auto_choose(big) == NgramAutoTier::Ram);
    /* A small table on the test machine is still pinned when the floor allows. */
    NgramAutoIn small = red_one_card(true);
    small.need = 8 * kGiB;
    CHECK(ngram_auto_choose(small) == NgramAutoTier::Ram);
    /* Exactly a quarter is not over it. */
    CHECK(!ngram_pin_too_large(kTotal / 4, kTotal));
    /* Unreadable RAM never prefers the file on size. */
    CHECK(!ngram_pin_too_large(aligned(kTable), -1));
}

TEST(linux_keeps_the_copy_where_it_fits) {
    NgramAutoIn in = red_one_card(true);
    in.file_preferred = false;
    CHECK(ngram_auto_choose(in) == NgramAutoTier::Ram);
}

TEST(without_a_host_gather_the_copy_is_the_only_tier_off_the_card) {
    /* No disk tier to prefer: a copy that fits is pinned whatever its size. */
    NgramAutoIn in = red_one_card(true);
    in.host = false;
    CHECK(ngram_auto_choose(in) == NgramAutoTier::Ram);
    in.ram_room = false;
    CHECK(ngram_auto_choose(in) == NgramAutoTier::Nowhere);
}

TEST(the_card_comes_first) {
    NgramAutoIn in = red_one_card(true);
    in.card_holds = true;
    CHECK(ngram_auto_choose(in) == NgramAutoTier::Vram);
    /* A card that would hold it but has no gather kernel does not. */
    in.device = false;
    CHECK(ngram_auto_choose(in) == NgramAutoTier::Disk);
}

TEST(unreadable_memory_is_not_short) {
    CHECK(!host_mem_short(-1, kStill, 24 * kGiB, aligned(kTable)));
}

TEST(the_load_time_recheck_on_11_22_figures) {
    /* At the pin, after the pools (still = 0): the RAM log read 12.40 GiB available at 2:46:07 PM,
     * before the copy. Auto's recheck refuses it and the table is read from the file. */
    CHECK(host_mem_short(gib(12.40), 0, ngram_auto_margin(kHeadroom), aligned(kTable)));
    /* A copy with room beside the floor passes. */
    CHECK(!host_mem_short(gib(80.0), 0, ngram_auto_margin(kHeadroom), aligned(kTable)));
}
RAD_TEST_MAIN()
