/* Brilliance: the native HIP backend's stall watch on a card (win/device/stall_watch.h; the
 * decisions themselves are stall_watch_test's). Two cases, one card, nothing spun on the GPU:
 *   - a healthy batch that keeps the card busy for about 8 s behind ONE host wait, as a 32K-token
 *     prefill does, is never a stall: its progress beacons move. With BRILLIANCE_STALL_BEACONS=0
 *     (the watch as it was), the same batch reads as one, which shows the case reaches the
 *     condition the beacons fix;
 *   - a stream held still (a host function that blocks it, so the card sits idle: no hung kernel,
 *     no TDR) behind a host wait IS a stall: dev_stuck_ms passes 5 s, then reads 0 once the hold
 *     is let go and the wait returns.
 * Skips with no card or on another backend. */
#include "rad_test.h"

#include "rad_internal.h"
#include "device/device.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#if defined(_WIN32) && defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
#include <hip/hip_runtime.h>
#define STALL_CARD_TEST 1
#endif

namespace {

using Clock = std::chrono::steady_clock;

int64_t ms_since(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t).count();
}

bool need_native_card() {
#if defined(STALL_CARD_TEST)
    const char* name = rad::device_backend_name();
    if (!rad::device_is_host() && name && std::strcmp(name, "hipnative") == 0) return true;
#endif
    radtest::skip("needs the native HIP backend on a card");
    return false;
}

bool beacons_off() {
    const char* e = std::getenv("BRILLIANCE_STALL_BEACONS");
    return e && std::strcmp(e, "0") == 0;
}

/* dev_stuck_ms read every 100 ms on a thread of its own; the most it read. */
struct Poll {
    std::atomic<bool>    stop{false};
    std::atomic<int64_t> most{0};
    std::thread          t;
    Poll() {
        t = std::thread([this] {
            while (!stop.load()) {
                const int64_t v = rad::dev_stuck_ms();
                if (v > most.load()) most.store(v);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }
    ~Poll() {
        stop.store(true);
        t.join();
    }
};

struct Stream {
    RadStream s = nullptr;
    Stream() { rad_stream_create(&s, 0); }
    ~Stream() { if (s) rad_stream_destroy(s); }
};

#if defined(STALL_CARD_TEST)
/* The hold: a host function on the stream waits here until let go, at most 30 s. */
struct Hold {
    std::mutex              mu;
    std::condition_variable cv;
    bool                    go = false;
    bool                    entered = false;
    static void fn(void* p) {
        auto* h = (Hold*)p;
        std::unique_lock<std::mutex> lk(h->mu);
        h->entered = true;
        h->cv.notify_all();
        h->cv.wait_for(lk, std::chrono::seconds(30), [h] { return h->go; });
    }
    void let_go() {
        std::lock_guard<std::mutex> lk(mu);
        go = true;
        cv.notify_all();
    }
};
#endif

}  /* namespace */

TEST(a_long_healthy_batch_behind_one_wait_is_not_a_stall) {
    if (!need_native_card()) return;
    const int64_t bytes = (int64_t)256 << 20;
    void* p = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
    REQUIRE(p != nullptr);
    Stream s;
    REQUIRE(s.s != nullptr);
    /* What one fill costs on this card, to size the batch at about 8 s. */
    const auto c0 = Clock::now();
    for (int i = 0; i < 64; ++i) {
        CHECK_OK(rad_memset_async(p, i & 0xff, bytes, s.s));
        rad_dev_note_launch(s.s, "stall_watch_fill", nullptr);
    }
    CHECK_OK(rad_stream_sync(s.s));
    const double per_op = (double)ms_since(c0) / 64.0;
    const int64_t n = per_op > 0.0 ? (int64_t)(8000.0 / per_op) : 20000;
    const int64_t ops = n < 1000 ? 1000 : (n > 400000 ? 400000 : n);
    std::fprintf(stderr, "stall_watch_card_test: %.3f ms a fill, %lld fills\n", per_op, (long long)ops);

    int64_t most = 0, took = 0;
    {
        Poll poll;
        const auto t0 = Clock::now();
        for (int64_t i = 0; i < ops; ++i) {
            CHECK_OK(rad_memset_async(p, (int)(i & 0xff), bytes, s.s));
            rad_dev_note_launch(s.s, "stall_watch_fill", nullptr);
        }
        CHECK_OK(rad_stream_sync(s.s));   /* the one host wait */
        took = ms_since(t0);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        most = poll.most.load();
    }
    std::fprintf(stderr, "stall_watch_card_test: the batch took %lld ms behind one wait; the stall "
                 "watch read at most %lld ms\n", (long long)took, (long long)most);
    CHECK(took > 6000);   /* long enough that the watch as it was calls it a stall */
    if (beacons_off()) CHECK(most >= 5000);
    else CHECK_EQ(most, (int64_t)0);
    rad_dev_free(p, RAD_MEM_DEVICE);
}

TEST(a_stream_held_still_behind_a_wait_is_a_stall_until_it_is_let_go) {
    if (!need_native_card()) return;
#if defined(STALL_CARD_TEST)
    const int64_t bytes = (int64_t)1 << 20;
    void* p = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
    REQUIRE(p != nullptr);
    Stream s;
    REQUIRE(s.s != nullptr);
    /* Work that finishes first: its beacons move, then nothing does. */
    for (int i = 0; i < 256; ++i) {
        CHECK_OK(rad_memset_async(p, i & 0xff, bytes, s.s));
        rad_dev_note_launch(s.s, "stall_watch_fill", nullptr);
    }
    Hold hold;
    REQUIRE_EQ((int)hipLaunchHostFunc((hipStream_t)s.s, &Hold::fn, &hold), (int)hipSuccess);
    /* Work behind the hold, beacons included: none of it may run until the hold is let go. */
    for (int i = 0; i < 256; ++i) {
        CHECK_OK(rad_memset_async(p, i & 0xff, bytes, s.s));
        rad_dev_note_launch(s.s, "stall_watch_fill", nullptr);
    }
    int64_t most = 0, at = -1;
    int rc = -1;
    {
        Poll poll;
        const auto t0 = Clock::now();
        std::thread waiter([&] { rc = rad_stream_sync(s.s); });
        while (ms_since(t0) < 12000 && poll.most.load() < 5000)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        most = poll.most.load();
        at = ms_since(t0);
        {
            std::lock_guard<std::mutex> lk(hold.mu);
            CHECK(hold.entered);
        }
        hold.let_go();
        waiter.join();
    }
    std::fprintf(stderr, "stall_watch_card_test: held stream: the watch read %lld ms after %lld ms\n",
                 (long long)most, (long long)at);
    CHECK_OK(rc);
    CHECK(most >= 5000);
    CHECK(at < 9000);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    CHECK_EQ(rad::dev_stuck_ms(), (int64_t)0);   /* the wait returned: nobody is blocked */
    rad_dev_free(p, RAD_MEM_DEVICE);
#endif
}

RAD_TEST_MAIN()
