/* stoplog.cpp -- Brilliance: the stop journal (see stoplog.h). */
#include "util/stoplog.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace rad::stoplog {

namespace {

using clock = std::chrono::steady_clock;

struct Open {
    char              name[160] = {};
    clock::time_point t0{};
    int64_t           bound_ms = 0;
    int64_t           next_wait_ms = 0;   /* the watcher's next report, doubling from the bound */
    bool              used = false;
};

constexpr int kSlots = 64;

/* Never destroyed: a watcher still running while the process exits, or a record from a thread the
 * exit did not join, finds it whole. */
struct Journal {
    std::mutex              mu;          /* one record at a time, and the open steps */
    std::condition_variable cv;          /* wakes the watcher to end */
    std::string             path;
    clock::time_point       armed_at{};
#if defined(_WIN32)
    HANDLE                  h = INVALID_HANDLE_VALUE;
#else
    int                     fd = -1;
#endif
    bool                    closed = false;
    bool                    watching = false;
    std::thread             watcher;
    Open                    steps[kSlots];
};

std::atomic<bool> g_on{false};
std::atomic<bool> g_begun{false};
std::atomic<bool> g_atexit{false};

Journal& J() {
    static Journal* j = new Journal();
    return *j;
}

unsigned long thread_id() {
#if defined(_WIN32)
    return (unsigned long)GetCurrentThreadId();
#elif defined(SYS_gettid)
    return (unsigned long)syscall(SYS_gettid);
#else
    return 0;
#endif
}

/* Under J().mu. Writes `line` and pushes it to the disk before returning. */
void put_locked(const char* line, size_t n) {
    Journal& j = J();
#if defined(_WIN32)
    if (j.h == INVALID_HANDLE_VALUE) return;
    DWORD w = 0;
    (void)WriteFile(j.h, line, (DWORD)n, &w, nullptr);
    (void)FlushFileBuffers(j.h);
#else
    if (j.fd < 0) return;
    for (size_t off = 0; off < n;) {
        const ssize_t w = ::write(j.fd, line + off, n - off);
        if (w <= 0) break;
        off += (size_t)w;
    }
    (void)::fdatasync(j.fd);
#endif
}

/* Under J().mu. One record: the prefix, then `kind`, `name` and the formatted detail. */
void record_locked(const char* kind, const char* name, const char* fmt, va_list ap) {
    Journal& j = J();
    if (j.closed) return;
    char line[1024];
    const auto now_sys = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now_sys);
    const int ms = (int)(std::chrono::duration_cast<std::chrono::milliseconds>(
                             now_sys.time_since_epoch()).count() % 1000);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    const double t = std::chrono::duration<double, std::milli>(clock::now() - j.armed_at).count();
    int n = std::snprintf(line, sizeof line,
                          "%04d-%02d-%02d %02d:%02d:%02d.%03d t=%.1f tid=%lu %s %s",
                          tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min,
                          tm.tm_sec, ms, t, thread_id(), kind, name && *name ? name : "-");
    if (n < 0) return;
    if ((size_t)n >= sizeof line) n = (int)sizeof line - 2;
    /* The name is one word: a space in it would read as the detail's start. */
    if (name && *name) {
        const size_t len = std::strlen(name);
        for (int i = n - (int)std::min(len, (size_t)n); i < n; ++i)
            if (line[i] == ' ' || line[i] == '\t') line[i] = '_';
    }
    if (fmt && *fmt) {
        line[n++] = ' ';
        const int m = std::vsnprintf(line + n, sizeof line - (size_t)n - 1, fmt, ap);
        if (m > 0) n = std::min<int>(n + m, (int)sizeof line - 2);
    }
    /* One line a record, whatever the detail held. */
    for (int i = 0; i < n; ++i)
        if (line[i] == '\n' || line[i] == '\r') line[i] = ' ';
    line[n++] = '\n';
    put_locked(line, (size_t)n);
}

void record_locked_f(const char* kind, const char* name, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    record_locked(kind, name, fmt, ap);
    va_end(ap);
}

void watch_loop() {
    Journal& j = J();
    std::unique_lock<std::mutex> lk(j.mu);
    while (!j.closed) {
        j.cv.wait_for(lk, std::chrono::milliseconds(250));
        if (j.closed) break;
        const auto now = clock::now();
        for (Open& s : j.steps) {
            if (!s.used) continue;
            const int64_t el = std::chrono::duration_cast<std::chrono::milliseconds>(now - s.t0).count();
            if (el < s.next_wait_ms) continue;
            record_locked_f("WAIT", s.name, "%lld ms so far, bound %lld ms", (long long)el,
                            (long long)s.bound_ms);
            s.next_wait_ms = std::max<int64_t>(s.next_wait_ms * 2, 1);
        }
    }
}

void at_exit() { finish("process exit (atexit); the runtime libraries unload after this"); }

void close_locked() {
    Journal& j = J();
    j.closed = true;
#if defined(_WIN32)
    if (j.h != INVALID_HANDLE_VALUE) CloseHandle(j.h);
    j.h = INVALID_HANDLE_VALUE;
#else
    if (j.fd >= 0) ::close(j.fd);
    j.fd = -1;
#endif
}

}  // namespace

namespace {
#if defined(_WIN32)
std::wstring wide(const std::string& p) {
    const int wn = MultiByteToWideChar(CP_UTF8, 0, p.c_str(), -1, nullptr, 0);
    std::wstring w(wn > 0 ? (size_t)wn - 1 : 0, L'\0');
    if (wn > 1) MultiByteToWideChar(CP_UTF8, 0, p.c_str(), -1, w.data(), wn);
    return w;
}
#endif

bool is_dir(const std::string& p) {
#if defined(_WIN32)
    const DWORD a = GetFileAttributesW(wide(p).c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

unsigned long process_id() {
#if defined(_WIN32)
    return (unsigned long)GetCurrentProcessId();
#else
    return (unsigned long)getpid();
#endif
}
}  // namespace

bool arm(const std::string& p) {
    if (g_on.load(std::memory_order_acquire)) return true;
    Journal& j = J();
    std::lock_guard<std::mutex> lk(j.mu);
#if defined(_WIN32)
    j.h = CreateFileW(wide(p).c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                      nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (j.h == INVALID_HANDLE_VALUE) return false;
#else
    j.fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC, 0644);
    if (j.fd < 0) return false;
#endif
    j.path = p;
    j.closed = false;
    j.armed_at = clock::now();
    record_locked_f("ARM", "journal", "pid=%lu written through to the disk record by record",
                    process_id());
    g_on.store(true, std::memory_order_release);
    if (!g_atexit.exchange(true)) std::atexit(at_exit);
    return true;
}

bool arm_from_env() {
    if (g_on.load(std::memory_order_acquire)) return true;
    const char* v = std::getenv("BRILLIANCE_STOP_LOG");
    if (!v || !*v) return false;
    std::string p = v;
    if (is_dir(p)) {
        if (p.back() != '/' && p.back() != '\\') p += '/';
        p += "radiance-stop-" + std::to_string(process_id()) + ".log";
    }
    if (!arm(p)) {
        std::fprintf(stderr, "W BRILLIANCE_STOP_LOG=%s: the stop journal cannot be opened; it is off\n", v);
        return false;
    }
    return true;
}

bool active() { return g_on.load(std::memory_order_relaxed); }

std::string path() {
    if (!active()) return {};
    std::lock_guard<std::mutex> lk(J().mu);
    return J().path;
}

bool begun() { return g_begun.load(std::memory_order_acquire); }

void begin(const char* cause) {
    if (!active()) return;
    Journal& j = J();
    std::lock_guard<std::mutex> lk(j.mu);
    if (j.closed) return;
    const bool first = !g_begun.exchange(true);
    record_locked_f("STOP", first ? "begin" : "again", "%s", cause ? cause : "");
    if (first && !j.watching) {
        j.watching = true;
        j.watcher = std::thread(watch_loop);
    }
}

void note(const char* name, const char* fmt, ...) {
    if (!active()) return;
    Journal& j = J();
    std::lock_guard<std::mutex> lk(j.mu);
    va_list ap;
    va_start(ap, fmt);
    record_locked("NOTE", name, fmt, ap);
    va_end(ap);
}

Step::Step(const char* name, int64_t bound_ms) {
    if (!active()) return;
    Journal& j = J();
    std::lock_guard<std::mutex> lk(j.mu);
    if (j.closed) return;
    for (int i = 0; i < kSlots; ++i) {
        if (j.steps[i].used) continue;
        Open& s = j.steps[i];
        std::snprintf(s.name, sizeof s.name, "%s", name ? name : "-");
        s.t0 = clock::now();
        s.bound_ms = bound_ms;
        s.next_wait_ms = std::max<int64_t>(bound_ms, 1);
        s.used = true;
        slot_ = i;
        break;
    }
    record_locked_f("BEGIN", name, "bound %lld ms", (long long)bound_ms);
}

void Step::end(const char* outcome) {
    if (slot_ < 0) return;
    Journal& j = J();
    std::lock_guard<std::mutex> lk(j.mu);
    Open& s = j.steps[slot_];
    slot_ = -1;
    if (!s.used) return;
    const double el = std::chrono::duration<double, std::milli>(clock::now() - s.t0).count();
    record_locked_f("END", s.name, "%s %.1f ms bound %lld ms %s",
                    el <= (double)s.bound_ms ? "ok" : "over", el, (long long)s.bound_ms,
                    outcome && *outcome ? outcome : "done");
    s.used = false;
}

void finish(const char* why) {
    if (!active()) return;
    Journal& j = J();
    std::thread w;
    {
        std::lock_guard<std::mutex> lk(j.mu);
        if (j.closed) return;
        /* A step still open here never ended: name it before the journal closes. */
        const auto now = clock::now();
        for (Open& s : j.steps) {
            if (!s.used) continue;
            record_locked_f("NOTE", s.name, "still open at close after %lld ms",
                            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - s.t0).count());
            s.used = false;
        }
        record_locked_f("CLOSE", "journal", "%s", why ? why : "");
        close_locked();
        w = std::move(j.watcher);
        j.watching = false;
    }
    j.cv.notify_all();
    if (w.joinable()) {
        if (w.get_id() == std::this_thread::get_id()) w.detach();
        else w.join();
    }
    g_on.store(false, std::memory_order_release);
}

void reset_for_test() {
    finish("reset");
    Journal& j = J();
    std::lock_guard<std::mutex> lk(j.mu);
    for (Open& s : j.steps) s.used = false;
    j.path.clear();
    j.closed = false;
    g_begun.store(false, std::memory_order_release);
    g_on.store(false, std::memory_order_release);
}

}  // namespace rad::stoplog
