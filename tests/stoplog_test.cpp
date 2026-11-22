/* Brilliance: the stop journal (core/util/stoplog.h) and the rank barrier's release kind
 * (core/runtime/rank_barrier.h), without a card or a model:
 *   - off unless armed: nothing is written, every call is harmless;
 *   - every record is on the disk when the call returns, read back while the journal is still
 *     open (what survives a hard lock is what was written through);
 *   - a step records its begin and its end with its time and whether it kept its bound;
 *   - a step past its bound is named by the watcher while it still runs, and again at twice the
 *     bound, so a step that never ends is still in the journal;
 *   - a step still open at the close is named; nothing is written after the close;
 *   - BRILLIANCE_STOP_LOG naming a directory puts radiance-stop-<pid>.log in it;
 *   - the barrier tells a full release from an abort, including an abort that comes after a full
 *     release and before the waiter wakes (the split the stop journal reports). */
#include "rad_test.h"

#include "runtime/rank_barrier.h"
#include "util/stoplog.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace sl = rad::stoplog;

namespace {

std::string read_all(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::string> lines_of(const std::string& p) {
    std::vector<std::string> v;
    std::istringstream in(read_all(p));
    for (std::string l; std::getline(in, l);) v.push_back(l);
    return v;
}

/* The KIND and name of a record: the fifth and sixth words. */
struct Rec {
    std::string kind, name, detail;
};
Rec parse(const std::string& line) {
    std::istringstream in(line);
    std::string date, time, t, tid;
    Rec r;
    in >> date >> time >> t >> tid >> r.kind >> r.name;
    std::getline(in, r.detail);
    if (!r.detail.empty() && r.detail[0] == ' ') r.detail.erase(0, 1);
    return r;
}

int find(const std::vector<std::string>& ls, const std::string& kind, const std::string& name,
         int from = 0) {
    for (int i = from; i < (int)ls.size(); ++i) {
        const Rec r = parse(ls[(size_t)i]);
        if (r.kind == kind && r.name == name) return i;
    }
    return -1;
}

void set_env(const char* k, const std::string& v) {
#if defined(_WIN32)
    _putenv_s(k, v.c_str());
#else
    if (v.empty()) unsetenv(k);
    else setenv(k, v.c_str(), 1);
#endif
}

}  // namespace

TEST(off_unless_armed) {
    sl::reset_for_test();
    set_env("BRILLIANCE_STOP_LOG", "");
    CHECK(!sl::arm_from_env());
    CHECK(!sl::active());
    sl::begin("nothing");
    sl::note("x", "%d", 1);
    { sl::Step s("y", 10); }
    sl::finish("nothing");
    CHECK(!sl::begun());
    CHECK(sl::path().empty());
}

TEST(each_record_is_on_the_disk_when_the_call_returns) {
    sl::reset_for_test();
    const std::string p = radtest::tmp_path("stoplog_written.log");
    REQUIRE(sl::arm(p));
    CHECK(sl::active());
    CHECK_EQ(sl::path(), p);
    /* Read back while the journal is open, after each call. */
    auto ls = lines_of(p);
    REQUIRE_EQ(ls.size(), (size_t)1);
    CHECK_EQ(parse(ls[0]).kind, std::string("ARM"));
    sl::begin("Ctrl-Break (test)");
    ls = lines_of(p);
    REQUIRE_EQ(ls.size(), (size_t)2);
    CHECK_EQ(parse(ls[1]).kind, std::string("STOP"));
    CHECK_EQ(parse(ls[1]).name, std::string("begin"));
    CHECK_EQ(parse(ls[1]).detail, std::string("Ctrl-Break (test)"));
    sl::begin("again");
    CHECK_EQ(parse(lines_of(p).back()).name, std::string("again"));
    sl::note("rank.exit", "rank %d: %s", 3, "line\nbreaks become spaces");
    ls = lines_of(p);
    REQUIRE_EQ(ls.size(), (size_t)4);
    CHECK_EQ(parse(ls[3]).detail, std::string("rank 3: line breaks become spaces"));
    sl::finish("test done");
    sl::reset_for_test();
}

TEST(a_step_records_its_time_and_bound) {
    sl::reset_for_test();
    const std::string p = radtest::tmp_path("stoplog_steps.log");
    REQUIRE(sl::arm(p));
    sl::begin("test");
    {
        sl::Step a(std::string("model.m one.drain"), 5000);
        CHECK_EQ(parse(lines_of(p).back()).kind, std::string("BEGIN"));
        /* A space in a name is written as '_': the name stays one word. */
        CHECK_EQ(parse(lines_of(p).back()).name, std::string("model.m_one.drain"));
    }
    {
        sl::Step b("fleet.front_stop", 20);
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        b.end("the listener closed late");
        b.end("twice is once");
    }
    sl::finish("test done");
    const auto ls = lines_of(p);
    const int ea = find(ls, "END", "model.m_one.drain");
    const int eb = find(ls, "END", "fleet.front_stop");
    REQUIRE(ea > 0);
    REQUIRE(eb > ea);
    CHECK_EQ(parse(ls[(size_t)ea]).detail.rfind("ok ", 0), (size_t)0);
    CHECK(parse(ls[(size_t)ea]).detail.find("bound 5000 ms done") != std::string::npos);
    CHECK_EQ(parse(ls[(size_t)eb]).detail.rfind("over ", 0), (size_t)0);
    CHECK(parse(ls[(size_t)eb]).detail.find("the listener closed late") != std::string::npos);
    CHECK_EQ(find(ls, "END", "fleet.front_stop", eb + 1), -1);   /* ended once */
    CHECK_EQ(parse(ls.back()).kind, std::string("CLOSE"));
    sl::reset_for_test();
}

TEST(a_step_past_its_bound_is_named_while_it_runs) {
    sl::reset_for_test();
    const std::string p = radtest::tmp_path("stoplog_wait.log");
    REQUIRE(sl::arm(p));
    sl::begin("test");
    {
        sl::Step s("model.m.rank1.ctx_shutdown", 100);
        /* Past 1x and 2x the bound: the watcher (every 250 ms) has named it at least once by
         * 600 ms, while the step is still open. */
        std::this_thread::sleep_for(std::chrono::milliseconds(900));
        const auto ls = lines_of(p);
        const int w = find(ls, "WAIT", "model.m.rank1.ctx_shutdown");
        CHECK(w > 0);
        CHECK_EQ(find(ls, "END", "model.m.rank1.ctx_shutdown"), -1);
        if (w > 0) CHECK(parse(ls[(size_t)w]).detail.find("bound 100 ms") != std::string::npos);
    }
    sl::finish("test done");
    const auto ls = lines_of(p);
    const int w1 = find(ls, "WAIT", "model.m.rank1.ctx_shutdown");
    CHECK(w1 > 0);
    CHECK(find(ls, "WAIT", "model.m.rank1.ctx_shutdown", w1 + 1) > w1);   /* again at 2x */
    sl::reset_for_test();
}

TEST(a_step_open_at_the_close_is_named_and_nothing_follows) {
    sl::reset_for_test();
    const std::string p = radtest::tmp_path("stoplog_open.log");
    REQUIRE(sl::arm(p));
    sl::begin("test");
    auto* never = new sl::Step("model.m.stream_probe", 60000);
    sl::finish("closing with a step open");
    const size_t n = lines_of(p).size();
    sl::note("late", "after the close");
    delete never;   /* its end after the close writes nothing */
    const auto ls = lines_of(p);
    CHECK_EQ(ls.size(), n);
    const int o = find(ls, "NOTE", "model.m.stream_probe");
    REQUIRE(o > 0);
    CHECK(parse(ls[(size_t)o]).detail.find("still open at close") != std::string::npos);
    CHECK_EQ(parse(ls.back()).kind, std::string("CLOSE"));
    CHECK_EQ(find(ls, "NOTE", "late"), -1);
    sl::reset_for_test();
}

TEST(a_directory_gets_a_journal_named_for_the_process) {
    sl::reset_for_test();
    const std::string d = radtest::tmp_path("stoplog_dir");
    std::filesystem::create_directories(d);
    set_env("BRILLIANCE_STOP_LOG", d);
    REQUIRE(sl::arm_from_env());
    const std::string p = sl::path();
    CHECK(p.find("radiance-stop-") != std::string::npos);
    CHECK(std::filesystem::exists(p));
    CHECK(std::filesystem::equivalent(std::filesystem::path(p).parent_path(), d));
    CHECK(sl::arm_from_env());   /* once on, later calls keep it */
    sl::finish("test done");
    set_env("BRILLIANCE_STOP_LOG", "");
    sl::reset_for_test();
}

TEST(the_barrier_tells_a_full_release_from_an_abort) {
    /* A full release: both parties see true. */
    {
        rad::RankBarrier b(2);
        std::atomic<int> got{-1};
        std::thread t([&] { got.store(b.arrive_and_wait() ? 1 : 0); });
        CHECK(b.arrive_and_wait());
        t.join();
        CHECK_EQ(got.load(), 1);
    }
    /* An abort releases a waiter that was not joined by every party: false, and every later
     * arrival returns false at once. */
    {
        rad::RankBarrier b(3);
        std::atomic<int> got{-1};
        std::thread t([&] { got.store(b.arrive_and_wait() ? 1 : 0); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        b.abort();
        t.join();
        CHECK_EQ(got.load(), 0);
        CHECK(!b.arrive_and_wait());
    }
    /* A full release followed at once by an abort, before the waiter wakes: still a full
     * release for it, as for every party of that turn. The thread arrives first (most rounds;
     * the 200 us head start) and waits; this thread's arrival turns the generation and aborts
     * back to back, usually before the waiter has woken. Before 2026-11-22 such a waiter could
     * not tell this from an abort; whichever way the scheduler wakes it now, the answer is the
     * turn's. */
    for (int k = 0; k < 200; ++k) {
        rad::RankBarrier b(2);
        std::atomic<int> got{-1};
        std::atomic<bool> in{false};
        std::thread t([&] {
            in.store(true);
            got.store(b.arrive_and_wait() ? 1 : 0);
        });
        while (!in.load()) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        const bool mine = b.arrive_and_wait();
        b.abort();
        t.join();
        /* With two parties the second arrival turns, so both are full releases whatever the
         * abort's timing. */
        CHECK(mine);
        CHECK_EQ(got.load(), 1);
    }
}

/* Brilliance: the stop's close() (Engine::serve). Parties arrive in pairs, a release then a join, as
 * the scheduler and the rank threads do. */
TEST(the_stop_lets_an_open_join_turn_and_releases_nothing_after) {
    /* No step open: close() is abort(). A party waiting at the release gets false, and so does
     * every later arrival. */
    {
        rad::RankBarrier b(3);
        std::atomic<int> got{-1};
        std::thread t([&] { got.store(b.arrive_and_wait() ? 1 : 0); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        b.close();
        t.join();
        CHECK_EQ(got.load(), 0);
        CHECK(!b.arrive_and_wait());
        b.close();                       /* a second close changes nothing */
        CHECK(!b.arrive_and_wait());
    }
    /* A step released and its join open: the join waits for the slow party and turns as a full
     * turn for all three; the next release is refused to every party. Here the slow party is this
     * thread, which arrives at the join only after the close. */
    for (int k = 0; k < 50; ++k) {
        rad::RankBarrier b(3);
        std::atomic<int> released{0};
        std::atomic<int> join_a{-1}, join_s{-1}, next_a{-1}, next_s{-1};
        auto party = [&](std::atomic<int>& join, std::atomic<int>& next) {
            if (b.arrive_and_wait()) released.fetch_add(1);
            join.store(b.arrive_and_wait() ? 1 : 0);
            next.store(b.arrive_and_wait() ? 1 : 0);
        };
        std::thread a([&] { party(join_a, next_a); });
        std::thread s([&] { party(join_s, next_s); });
        CHECK(b.arrive_and_wait());      /* the release turns */
        while (released.load() < 2) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(k % 5));
        b.close();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CHECK_EQ(join_a.load(), -1);     /* the join is still open: the close did not end it */
        CHECK_EQ(join_s.load(), -1);
        CHECK(b.arrive_and_wait());      /* the slow party reaches the join, and it turns */
        a.join();
        s.join();
        CHECK_EQ(join_a.load(), 1);
        CHECK_EQ(join_s.load(), 1);
        CHECK_EQ(next_a.load(), 0);      /* no release after the stop */
        CHECK_EQ(next_s.load(), 0);
        CHECK(!b.arrive_and_wait());
    }
}

RAD_TEST_MAIN()
