/* Brilliance: the native HIP backend's stall watch (win/device/stall_watch.h), driven with a clock
 * of the test's own, so no card is needed:
 *   - a long healthy step whose beacons keep moving is never a stall, however long its wait
 *     (2026-11-20: 32K-token prefills on Gemma 4 12B and 31B logged false E lines);
 *   - the same step without beacons reads as one, as it did before them;
 *   - a real stall is still reported, once, stall_ms after the beacons stop: a hung kernel or
 *     work never submitted, with or without work that drained before it;
 *   - a reported stall that the device comes back from says so, and a later stall is reported
 *     again;
 *   - frozen beacons with nobody waiting are not a stall;
 *   - the issuing thread's cadence puts a beacon behind every `every`-th op. */
#include "rad_test.h"

#include "stall_watch.h"

#include <cstdint>
#include <vector>

namespace {

constexpr int64_t kStall = 5000;

/* Four slots, as the backend reads them: slot 0 the busy stream, slot 1 an idle one, 2 and 3
 * unused (0). */
struct Card {
    std::vector<uint64_t> words = std::vector<uint64_t>(4, 0);
    uint64_t serial = 0;
    void beacon(size_t slot = 0) { words[slot] = ++serial; }
};

}  /* namespace */

TEST(a_long_step_whose_beacons_move_is_never_a_stall) {
    rad::StallWatch w(kStall);
    Card c;
    c.beacon(1);   /* the idle stream's last beacon, from before */
    (void)w.observe(c.words.data(), c.words.size(), 0);
    const uint64_t id = w.wait_begin(1000);   /* one host wait for the whole prefill */
    for (int64_t t = 1000; t <= 26000; t += 1000) {   /* a 25 s step, read once a second */
        if (t % 2000 == 0) c.beacon(0);               /* a beacon every 2 s of GPU time */
        bool moved = false;
        CHECK_EQ(w.observe(c.words.data(), c.words.size(), t, &moved), (int64_t)0);
        CHECK_EQ(moved, t % 2000 == 0);
        const rad::StallWatch::Verdict v = w.check(t);
        CHECK_EQ(v.ms, (int64_t)0);
        CHECK(!v.report);
    }
    CHECK_EQ(w.wait_end(id, 26500), (int64_t)0);
    CHECK_EQ(w.waiting(), (size_t)0);
    CHECK_EQ(w.beacon_moves(), (uint64_t)13 + 1);   /* 13 in the step, 1 at the start */
}

TEST(the_same_step_without_beacons_reads_as_a_stall) {
    rad::StallWatch w(kStall);
    const uint64_t id = w.wait_begin(1000);
    CHECK_EQ(w.check(5999).ms, (int64_t)0);
    const rad::StallWatch::Verdict at = w.check(6000);
    CHECK_EQ(at.ms, (int64_t)5000);
    CHECK(at.report);
    const rad::StallWatch::Verdict later = w.check(9000);
    CHECK_EQ(later.ms, (int64_t)8000);
    CHECK(!later.report);   /* reported once a stall */
    (void)w.wait_end(id, 9500);
    CHECK_EQ(w.check(9500).ms, (int64_t)0);
}

TEST(a_hung_step_is_reported_once_stall_ms_after_its_beacons_stop) {
    rad::StallWatch w(kStall);
    Card c;
    const uint64_t id = w.wait_begin(0);
    /* Queued work drains for 3 s, beacons moving, then a kernel hangs (or the rest was never
     * submitted): the words stop. */
    int64_t t = 0;
    for (; t <= 3000; t += 500) {
        c.beacon(0);
        (void)w.observe(c.words.data(), c.words.size(), t);
        CHECK(!w.check(t).report);
    }
    const int64_t last_move = 3000;
    int reports = 0;
    int64_t reported_at = -1;
    for (t = last_move + 1000; t <= 70000; t += 1000) {
        bool moved = true;
        (void)w.observe(c.words.data(), c.words.size(), t, &moved);   /* the same words */
        CHECK(!moved);
        const rad::StallWatch::Verdict v = w.check(t);
        if (t - last_move < kStall) {
            CHECK_EQ(v.ms, (int64_t)0);
        } else {
            CHECK_EQ(v.ms, t - last_move);
        }
        if (v.report) {
            ++reports;
            reported_at = t;
        }
    }
    CHECK_EQ(reports, 1);
    CHECK_EQ(reported_at, last_move + kStall);
    CHECK(w.check(70000).ms > 60000);   /* /health's "stuck" limit is a minute */
    CHECK_EQ(w.last_beacon(), last_move);
    (void)id;
}

TEST(a_stall_before_any_beacon_is_reported_from_the_wait_start) {
    rad::StallWatch w(kStall);
    Card c;
    (void)w.observe(c.words.data(), c.words.size(), 0);   /* all slots 0: nothing yet */
    const uint64_t id = w.wait_begin(2000);
    for (int64_t t = 2000; t < 7000; t += 1000) {
        (void)w.observe(c.words.data(), c.words.size(), t);
        CHECK(!w.check(t).report);
    }
    const rad::StallWatch::Verdict v = w.check(7000);
    CHECK(v.report);
    CHECK_EQ(v.ms, (int64_t)5000);
    CHECK_EQ(w.last_beacon(), (int64_t)-1);
    CHECK_EQ(w.beacon_moves(), (uint64_t)0);
    (void)id;
}

TEST(a_beacon_that_moved_before_the_wait_does_not_hide_a_stall_in_it) {
    rad::StallWatch w(kStall);
    Card c;
    c.beacon(0);
    (void)w.observe(c.words.data(), c.words.size(), 10000);
    const uint64_t id = w.wait_begin(20000);   /* the last progress was 10 s before the wait */
    CHECK_EQ(w.check(24999).ms, (int64_t)0);
    CHECK(w.check(25000).report);
    (void)id;
}

TEST(a_reported_stall_the_device_resumes_from_says_so_and_a_later_one_is_reported_again) {
    rad::StallWatch w(kStall);
    Card c;
    const uint64_t id = w.wait_begin(0);
    CHECK(w.check(5000).report);
    w.kicked(5100);                            /* the markers went on the streams */
    c.beacon(0);                               /* and the device moved */
    CHECK_EQ(w.observe(c.words.data(), c.words.size(), 5400), (int64_t)5100);
    CHECK_EQ(w.check(5400).ms, (int64_t)0);
    /* The kick is told once; the next progress is just progress. */
    CHECK_EQ(w.progress(5500), (int64_t)0);
    /* Still in the same wait, the device stops again. */
    CHECK(!w.check(10400).report);
    const rad::StallWatch::Verdict again = w.check(10500);
    CHECK(again.report);
    CHECK_EQ(again.ms, (int64_t)5000);
    CHECK_EQ(w.wait_end(id, 11000), (int64_t)0);   /* not kicked this time */
}

TEST(frozen_beacons_with_nobody_waiting_are_not_a_stall) {
    rad::StallWatch w(kStall);
    Card c;
    c.beacon(0);
    (void)w.observe(c.words.data(), c.words.size(), 0);
    for (int64_t t = 0; t <= 120000; t += 10000) {
        (void)w.observe(c.words.data(), c.words.size(), t);
        CHECK_EQ(w.check(t).ms, (int64_t)0);
    }
    /* A wait that ends inside the limit leaves nothing behind. */
    const uint64_t id = w.wait_begin(130000);
    CHECK_EQ(w.check(134000).ms, (int64_t)0);
    (void)w.wait_end(id, 134500);
    CHECK_EQ(w.check(200000).ms, (int64_t)0);
}

TEST(two_waits_the_oldest_counts_and_any_progress_resets) {
    rad::StallWatch w(kStall);
    const uint64_t a = w.wait_begin(0);
    const uint64_t b = w.wait_begin(3000);
    CHECK(w.check(5000).report);               /* from a's start */
    (void)w.wait_end(b, 6000);                 /* b returned: progress */
    CHECK_EQ(w.check(10999).ms, (int64_t)0);
    CHECK(w.check(11000).report);              /* a is still blocked, 5 s after b's return */
    (void)a;
}

TEST(a_slot_given_back_and_rewritten_is_progress_and_unwritten_slots_are_not) {
    rad::StallWatch w(kStall);
    std::vector<uint64_t> words = { 0, 0, 0, 0 };
    bool moved = true;
    (void)w.observe(words.data(), words.size(), 0, &moved);
    CHECK(!moved);
    words[2] = 7;                               /* a write that landed */
    (void)w.observe(words.data(), words.size(), 100, &moved);
    CHECK(moved);
    (void)w.observe(words.data(), words.size(), 200, &moved);
    CHECK(!moved);                              /* the same value read again */
    words.push_back(0);                         /* a longer reading: the new slot unwritten */
    (void)w.observe(words.data(), words.size(), 300, &moved);
    CHECK(!moved);
}

TEST(the_cadence_puts_a_beacon_behind_every_nth_op_and_none_when_off) {
    rad::BeaconCadence c{ 64, 0 };
    int due = 0, first = -1;
    for (int op = 1; op <= 640; ++op)
        if (c.due()) {
            ++due;
            if (first < 0) first = op;
        }
    CHECK_EQ(due, 10);
    CHECK_EQ(first, 64);
    rad::BeaconCadence off{ 0, 0 };
    for (int op = 0; op < 1000; ++op) CHECK(!off.due());
    rad::BeaconCadence each{ 1, 0 };
    for (int op = 0; op < 10; ++op) CHECK(each.due());
}

RAD_TEST_MAIN()
