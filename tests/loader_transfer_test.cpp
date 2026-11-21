/* Exercise the actual loader ring with local completion-failure injection. No serving hooks. */
#include "rad_test.h"
#include "engine_priv.h"
#if defined(_WIN32)
#include "os/os.h"   /* Brilliance: no fork; a child is this binary again, running one case */
#include <cstdlib>
#include <cstring>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
enum Fault { None, Create, Record, Wait, Copy, Fence, CopyFence };
Fault fault = None;
int fences = 0;
int event_create(RadEvent* event) {
    return fault == Create ? RAD_E_DEVICE : rad_event_create(event);
}
int event_record(RadEvent event, RadStream stream) {
    const int status = rad_event_record(event, stream);
    return fault == Record ? RAD_E_DEVICE : status; // failure after submission
}
int event_sync(RadEvent event) {
    return fault == Wait || (fault == Fence || fault == CopyFence) ? RAD_E_DEVICE : rad_event_sync(event);
}
int stream_sync(RadStream stream) {
    ++fences;
    return (fault == Fence || fault == CopyFence) ? RAD_E_DEVICE : rad_stream_sync(stream);
}
int copy_async(void* dst, const void* src, int64_t bytes, RadStream stream) {
    const int status = rad_memcpy_async(dst, src, bytes, stream);
    return (fault == Copy || fault == CopyFence) ? RAD_E_DEVICE : status; // partial submission cannot be undone safely
}
}
#define rad_event_create event_create
#define rad_event_record event_record
#define rad_event_sync event_sync
#define rad_stream_sync stream_sync
#define rad_memcpy_async copy_async
#define RAD_LOADER_INTERNALS_TEST 1
#include "../core/engine_bringup.cpp"
#undef RAD_LOADER_INTERNALS_TEST
#undef rad_event_create
#undef rad_event_record
#undef rad_event_sync
#undef rad_stream_sync
#undef rad_memcpy_async

TEST(failed_completion_events_drain_before_resolving_ring_ownership) {
    if (!rad::device_is_host()) { radtest::skip("host backend only"); return; }
    for (Fault mode : { Create, Record, Wait, Copy }) {
        fault = None;
        fences = 0;
        rad::loader_test::PinRing ring;
        ring.start(1, true);
        auto piece = ring.take(4096);
        REQUIRE(bool(piece));
        std::memset(piece.p, 0x5a, 4096);
        std::vector<uint8_t> destination(4096, 0);
        if (mode == Copy) fault = mode;
        CHECK_EQ(ring.send(piece, destination.data(), 0, 4096, 0), mode == Copy ? RAD_E_DEVICE : RAD_OK);
        fault = mode;
        const int given = ring.give(piece);
        CHECK_EQ(given, (mode == Wait || mode == Copy) ? RAD_OK : RAD_E_DEVICE);
        CHECK(!piece);
        CHECK_EQ(ring.drain(), RAD_E_DEVICE);
        CHECK_EQ(fences, 1);
        CHECK(std::all_of(destination.begin(), destination.end(), [](uint8_t b) { return b == 0x5a; }));
        fault = None;
    }
}
#if defined(_WIN32)
namespace {
/* Brilliance: this binary again, running only the case `name`, with `extra` added to its
 * environment (format_test's pattern). Its exit code; what it wrote to stdout and stderr in *said. */
int run_case_alone(const char* name, const std::string& extra, std::string* said) {
    std::vector<std::string> env;
    for (char** e = _environ; e && *e; ++e)
        if (std::strncmp(*e, "RAD_TEST_FILTER=", 16) != 0) env.push_back(*e);
    env.push_back(extra);
    rad::os::Captured child;
    if (!child.start({ rad::os::exe_path(), name }, env)) return -1;
    if (!child.read_all(said, 60000)) child.kill();
    return child.wait();
}
}  // namespace

/* Brilliance: the forked child of the case below, in a process of its own. Windows has no fork, so
 * that case starts this binary again for this case alone, with the fault in the environment. Like
 * the Linux child it sends and gives one piece under the fault, then drains: the fence must end the
 * process with exit 70 (fatal) before the ring can recycle the piece. Any other run skips it. */
TEST(a_ring_whose_upload_cannot_be_proven_complete_in_a_process_of_its_own) {
    const char* m = std::getenv("RADTEST_LOADER_FAULT");
    if (!m)
        SKIP("the child that ambiguous_upload_or_failed_fallback_fence_cannot_recycle_the_ring "
             "starts; nothing to do on its own");
    if (!rad::device_is_host()) std::_Exit(97);
    const Fault mode = (Fault)std::atoi(m);
    fault = None;
    rad::loader_test::PinRing ring;
    ring.start(1, true);
    auto piece = ring.take(4096);
    if (!piece) std::_Exit(98);
    std::vector<uint8_t> destination(4096);
    if (mode == CopyFence) fault = mode;
    (void)ring.send(piece, destination.data(), 0, 4096, 0);
    (void)ring.give(piece);
    fault = mode;
    (void)ring.drain();
    std::_Exit(99);
}
#endif

TEST(ambiguous_upload_or_failed_fallback_fence_cannot_recycle_the_ring) {
    if (!rad::device_is_host()) { radtest::skip("host backend only"); return; }
    for (Fault mode : { CopyFence, Fence }) {
#if defined(_WIN32)
        /* Brilliance: no fork. The case above, alone in a child; the fence's fatal() exits 70. */
        std::string said;
        const int status = run_case_alone(
            "a_ring_whose_upload_cannot_be_proven_complete_in_a_process_of_its_own",
            "RADTEST_LOADER_FAULT=" + std::to_string((int)mode), &said);
        CHECK_EQ(status, 70);
        CHECK(said.find(mode == CopyFence ? "load: failed upload did not drain"
                                          : "load: upload stream did not drain") != std::string::npos);
#else
        const pid_t child = fork();
        REQUIRE(child >= 0);
        if (child == 0) {
            fault = None;
            rad::loader_test::PinRing ring;
            ring.start(1, true);
            auto piece = ring.take(4096);
            if (!piece) _exit(98);
            std::vector<uint8_t> destination(4096);
            if (mode == CopyFence) fault = mode;
            (void)ring.send(piece, destination.data(), 0, 4096, 0);
            (void)ring.give(piece);
            fault = mode;
            (void)ring.drain();
            _exit(99);
        }
        int status = 0;
        REQUIRE_EQ(waitpid(child, &status, 0), child);
        REQUIRE(WIFEXITED(status));
        CHECK_EQ(WEXITSTATUS(status), 70);
#endif
    }
}
RAD_TEST_MAIN()
