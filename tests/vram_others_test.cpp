/* Brilliance: the VRAM other processes hold on the serving cards (core/mem/vram_others.h): the
 * startup report's text and the settle check's verdict, on readings made up here. The live reading
 * (os::gpu_vram_holders) is Windows' and has its own check in os_gpu_test.
 *
 * The case behind it, 2026-11-22: T3 Code held about 720 MiB of the bus-32 R9700 when a four-card
 * Flash-Next server read the card, took about 256 MiB more while it loaded, and the card settled at
 * 7.28 MiB free against 263.45 MiB kept (160 MiB --gpu-headroom-mib, 103.45 MiB for later). The
 * start then failed on "step batch staging: could not reserve 12.43 MiB", naming nobody. */
#include "rad_test.h"

#include "mem/vram_others.h"

#include <string>
#include <vector>

using namespace rad;

static const int64_t MiB = 1024 * 1024;

static VramHolders known(std::vector<VramHolder> p) {
    VramHolders h;
    h.known = true;
    h.procs = std::move(p);
    return h;
}

static const VramHolder kT3{ 19340, "T3 Code (Nightly).exe", 720 * MiB };
static const VramHolder kHold{ 4242, "red-linkhold.exe", 16 * MiB };

TEST(the_holders_text_names_the_largest_first_and_says_none_and_unknown_apart) {
    CHECK_EQ(vram_holders_text(VramHolders{}), std::string("unknown"));
    CHECK_EQ(vram_holders_text(known({})), std::string("none"));
    CHECK_EQ(vram_holders_text(known({ VramHolder{ 7, "x.exe", 0 } })), std::string("none"));
    CHECK_EQ(vram_holders_text(known({ kHold, kT3 })),
             std::string("T3 Code (Nightly).exe (pid 19340) 720.00 MiB, red-linkhold.exe (pid 4242) 16.00 MiB"));
    /* A process that exited between the counters and the snapshot has no name. */
    CHECK_EQ(vram_holders_text(known({ VramHolder{ 99, "", 2 * MiB } })), std::string("a process (pid 99) 2.00 MiB"));
    const VramHolders many = known({ kT3, kHold, VramHolder{ 1, "a.exe", 3 * MiB }, VramHolder{ 2, "b.exe", 1 * MiB } });
    CHECK_EQ(vram_holders_text(many, 2),
             std::string("T3 Code (Nightly).exe (pid 19340) 720.00 MiB, red-linkhold.exe (pid 4242) 16.00 MiB, "
                         "and 2 more (4.00 MiB)"));
    CHECK_EQ(vram_holders_total(many), 740 * MiB);
}

TEST(growth_counts_a_new_process_whole_and_ignores_shrinking_and_unknown_readings) {
    const VramHolders then = known({ kT3, kHold });
    VramHolder t3 = kT3;
    t3.bytes += 256 * MiB;
    VramHolder hold = kHold;
    hold.bytes -= 4 * MiB;
    const VramHolder fresh{ 555, "browser.exe", 40 * MiB };
    const std::vector<VramGrowth> g = vram_holders_grown(then, known({ t3, hold, fresh }));
    REQUIRE_EQ(g.size(), (size_t)2);
    CHECK_EQ(g[0].now.pid, (uint32_t)19340);   /* largest growth first */
    CHECK_EQ(g[0].grew, 256 * MiB);
    CHECK_EQ(g[1].now.pid, (uint32_t)555);
    CHECK_EQ(g[1].grew, 40 * MiB);
    CHECK_EQ(vram_growth_total(g), 296 * MiB);
    CHECK(vram_holders_grown(VramHolders{}, known({ t3 })).empty());
    CHECK(vram_holders_grown(then, VramHolders{}).empty());
}

TEST(the_11_22_four_card_start_is_refused_naming_t3_code) {
    VramHolder t3 = kT3;
    t3.bytes += 256 * MiB;
    /* Settled at 7.28 MiB free with 103.45 MiB kept for later: no room left; the allowance is
     * the engine's own 160 MiB; the warning's "at least 353". */
    const int64_t room = (int64_t)(7.28 * MiB) - (int64_t)(103.45 * MiB);
    const int64_t taken = (int64_t)(256.17 * MiB);   /* 263.45 MiB kept, 7.28 MiB free */
    const VramSettleVerdict v = vram_settle_verdict(3, "0000:20:00.0", room, 160 * MiB, taken, 353,
                                                    known({ kT3, kHold }), known({ t3, kHold }));
    CHECK(v.refuse);
    CHECK_EQ(v.grew, 256 * MiB);
    CHECK(has(v.message, "card 3 (0000:20:00.0): other processes took 256.00 MiB more of its VRAM after "
                         "the budget read the card"));
    CHECK(has(v.message, "leaves 0 B for a kernel's first launch"));
    CHECK(has(v.message, "under the 160.00 MiB this engine needs"));
    CHECK(has(v.message, "T3 Code (Nightly).exe (pid 19340) holds 976.00 MiB now, 256.00 MiB more"));
    CHECK(!has(v.message, "red-linkhold"));   /* it did not grow */
    CHECK(has(v.message, "Close it, or set its graphics preference to another GPU"));
    CHECK(has(v.message, "--gpu-headroom-mib to at least 353"));
    CHECK(has(v.message, "Startup stops here rather than fail on a later allocation"));
}

TEST(two_processes_that_grew_are_both_named) {
    VramHolder t3 = kT3;
    t3.bytes += 100 * MiB;
    const VramHolder fresh{ 555, "browser.exe", 90 * MiB };
    const VramSettleVerdict v = vram_settle_verdict(0, "0000:03:00.0", 10 * MiB, 160 * MiB, 200 * MiB, 310,
                                                    known({ kT3 }), known({ t3, fresh }));
    CHECK(v.refuse);
    CHECK(has(v.message, "T3 Code (Nightly).exe (pid 19340) holds 820.00 MiB now, 100.00 MiB more, "
                         "browser.exe (pid 555) holds 90.00 MiB now, 90.00 MiB more"));
    CHECK(has(v.message, "Close them, or set their graphics preference"));
}

TEST(no_refusal_unless_the_card_is_short_because_of_the_others) {
    VramHolder t3 = kT3;
    t3.bytes += 64 * MiB;
    const VramHolders then = known({ kT3 }), now = known({ t3 });
    /* Not short: the growth came out of the headroom and the allowance is still there. */
    VramSettleVerdict v = vram_settle_verdict(1, "0000:09:00.0", 200 * MiB, 160 * MiB, 64 * MiB, 160, then, now);
    CHECK(!v.refuse);
    CHECK_EQ(v.grew, 64 * MiB);
    CHECK(v.message.empty());
    /* Short, but 200 MiB went missing and they took 64 of it: mostly an allocation the budget does
     * not count, and the earlier warning stands. */
    v = vram_settle_verdict(1, "0000:09:00.0", 50 * MiB, 160 * MiB, 200 * MiB, 270, then, now);
    CHECK(!v.refuse);
    /* Short, and they took half of what went missing: refused. */
    v = vram_settle_verdict(1, "0000:09:00.0", 50 * MiB, 160 * MiB, 128 * MiB, 270, then, now);
    CHECK(v.refuse);
    /* Short, nobody grew. */
    v = vram_settle_verdict(1, "0000:09:00.0", 10 * MiB, 160 * MiB, 250 * MiB, 310, then, then);
    CHECK(!v.refuse);
    CHECK_EQ(v.grew, (int64_t)0);
    /* Short, readings unknown (another platform, or the counters did not answer). */
    v = vram_settle_verdict(1, "0000:09:00.0", 10 * MiB, 160 * MiB, 250 * MiB, 310, VramHolders{}, VramHolders{});
    CHECK(!v.refuse);
    /* A holding at the budget read is never refused by itself: the plan counted it (a second
     * engine on the same cards, docs/GUIDE.md section 8). */
    const VramHolder engine{ 777, "radiance.exe", 28 * 1024 * MiB };
    v = vram_settle_verdict(1, "0000:09:00.0", 300 * MiB, 160 * MiB, 0, 160, known({ engine }), known({ engine }));
    CHECK(!v.refuse);
    v = vram_settle_verdict(1, "0000:09:00.0", 10 * MiB, 160 * MiB, 250 * MiB, 310, known({ engine }),
                            known({ engine }));
    CHECK(!v.refuse);
}

TEST(the_startup_report_lists_every_card_and_is_silent_when_nothing_is_known) {
    const std::vector<int> cards{ 0, 1, 2, 3 };
    const std::vector<std::string> bdfs{ "0000:06:00.0", "0000:03:00.0", "0000:09:00.0", "0000:20:00.0" };
    const std::vector<int64_t> r160(4, 160);
    CHECK_EQ(vram_holders_report(cards, bdfs, std::vector<VramHolders>(4), r160), std::string());
    const std::string none = vram_holders_report(cards, bdfs, { known({}), known({}), known({}), known({}) }, r160);
    CHECK_EQ(none, std::string("VRAM other processes hold on the serving cards: none (card 0 (0000:06:00.0) none; "
                               "card 1 (0000:03:00.0) none; card 2 (0000:09:00.0) none; card 3 (0000:20:00.0) none)"));
    const std::string some =
        vram_holders_report(cards, bdfs, { known({}), known({}), VramHolders{}, known({ kT3, kHold }) }, r160);
    CHECK(has(some, "card 2 (0000:09:00.0) unknown"));
    CHECK(has(some, "card 3 (0000:20:00.0) 736.00 MiB: T3 Code (Nightly).exe (pid 19340) 720.00 MiB, "
                    "red-linkhold.exe (pid 4242) 16.00 MiB"));
    CHECK(has(some, "comes out of the 160 MiB --gpu-headroom-mib, and startup stops naming them"));
    /* Several cards and no card of its own yet: the line says how to keep more on the busy one. */
    CHECK(has(some, "can keep more free on its own with memory.card-headroom-mib, {CARD: MiB}"));
}

/* Brilliance (issue #1): with memory.card-headroom-mib naming one card, the report gives each
 * card's own reserve and no longer suggests the key. One card: no per-card hint. */
TEST(the_startup_report_gives_each_cards_own_reserve) {
    const std::vector<int> cards{ 0, 1 };
    const std::vector<std::string> bdfs{ "0000:03:00.0", "0000:07:00.0" };
    const std::string s = vram_holders_report(cards, bdfs, { known({ kHold }), known({ kT3 }) }, { 160, 1536 },
                                              { false, true });
    CHECK(has(s, "comes out of each card's reserve (card 0 160 MiB, card 1 1536 MiB "
                 "(memory.card-headroom-mib))"));
    CHECK(!has(s, "can keep more free on its own"));
    const std::string one = vram_holders_report({ 0 }, { "0000:03:00.0" }, { known({ kT3 }) }, { 160 });
    CHECK(has(one, "comes out of the 160 MiB --gpu-headroom-mib"));
    CHECK(!has(one, "memory.card-headroom-mib"));
}

/* On a start of several cards the refusal's advice is the card's own reserve, not a headroom that
 * every card would pay. */
TEST(a_multi_card_refusal_names_the_cards_own_reserve) {
    VramHolder t3 = kT3;
    t3.bytes += 1100 * MiB;
    const VramSettleVerdict v = vram_settle_verdict(1, "0000:07:00.0", 10 * MiB, 160 * MiB, 1130 * MiB, 1290,
                                                    known({ kT3 }), known({ t3 }), true);
    CHECK(v.refuse);
    CHECK(has(v.message, "set memory.card-headroom-mib to {1: 1290} to leave other processes that much "
                         "more room on this card alone, while the other cards keep theirs"));
    CHECK(!has(v.message, "--gpu-headroom-mib"));
}

TEST(cards_with_no_address_are_not_read) {
    /* The host backend's cards have no PCI address: nothing is read, everything is unknown, and
     * the engine logs nothing. */
    const std::vector<VramHolders> h = read_vram_holders({ "", "" });
    REQUIRE_EQ(h.size(), (size_t)2);
    CHECK(!h[0].known);
    CHECK(!h[1].known);
    CHECK(read_vram_holders({}).empty());
}

RAD_TEST_MAIN()
