/* stallwatch.cpp -- see stallwatch.h. */
#include "util/stallwatch.h"

#include "rad_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#if defined(_WIN32)
/* Brilliance: no execinfo; a Windows thread's stack is copied while it is suspended (stack_of). */
#else
#include <execinfo.h>
#endif
#include <fstream>
#include <mutex>
#include <sstream>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#include "os/os.h"   /* Brilliance: UTF-8 module names and thread descriptions (win/os) */
#endif

namespace rad {

namespace {

#if defined(_WIN32)
/* Brilliance: no /proc to read lines from; what Windows says about a thread is asked of it. */
#else
std::string read_first_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    if (f) std::getline(f, s);
    return s;
}
#endif

/* KFD ioctl numbers (type 'K') from <linux/kfd_ioctl.h>, listed by number because the build's
 * header can be older than the running driver. */
const char* kfd_ioctl_name(unsigned nr) {
    static const char* const k[] = {
        nullptr, "GET_VERSION", "CREATE_QUEUE", "DESTROY_QUEUE", "SET_MEMORY_POLICY",
        "GET_CLOCK_COUNTERS", "GET_PROCESS_APERTURES", "UPDATE_QUEUE", "CREATE_EVENT",
        "DESTROY_EVENT", "SET_EVENT", "RESET_EVENT", "WAIT_EVENTS", "DBG_REGISTER",
        "DBG_UNREGISTER", "DBG_ADDRESS_WATCH", "DBG_WAVE_CONTROL", "SET_SCRATCH_BACKING_VA",
        "GET_TILE_CONFIG", "SET_TRAP_HANDLER", "GET_PROCESS_APERTURES_NEW", "ACQUIRE_VM",
        "ALLOC_MEMORY_OF_GPU", "FREE_MEMORY_OF_GPU", "MAP_MEMORY_TO_GPU", "UNMAP_MEMORY_FROM_GPU",
        "SET_CU_MASK", "GET_QUEUE_WAVE_STATE", "GET_DMABUF_INFO", "IMPORT_DMABUF",
        "ALLOC_QUEUE_GWS", "SMI_EVENTS", "SVM", "SET_XNACK_MODE", "CRIU_OP", "AVAILABLE_MEMORY",
        "EXPORT_DMABUF", "RUNTIME_ENABLE", "DBG_TRAP", "CREATE_PROCESS", "PROFILER",
    };
    return nr < sizeof k / sizeof k[0] ? k[nr] : nullptr;
}

/* The system calls a waiting thread is found in, by their x86-64 numbers. */
const char* syscall_name(long nr) {
#if defined(__x86_64__)
    switch (nr) {
        case 0: return "read";         case 1: return "write";       case 2: return "open";
        case 3: return "close";        case 4: return "stat";        case 7: return "poll";
        case 9: return "mmap";         case 10: return "mprotect";   case 11: return "munmap";
        case 16: return "ioctl";       case 17: return "pread64";    case 18: return "pwrite64";
        case 23: return "select";      case 28: return "madvise";    case 35: return "nanosleep";
        case 61: return "wait4";       case 149: return "mlock";     case 202: return "futex";
        case 230: return "clock_nanosleep";                          case 232: return "epoll_wait";
        case 237: return "mbind";      case 257: return "openat";    case 262: return "newfstatat";
        case 270: return "pselect6";   case 271: return "ppoll";     case 281: return "epoll_pwait";
        case 332: return "statx";      case 426: return "io_uring_enter"; case 437: return "openat2";
        default: return nullptr;
    }
#else
    (void)nr;
    return nullptr;
#endif
}

#if defined(_WIN32)
/* Brilliance: the lines decoded are Linux's, so the number is its x86-64 one; Windows has no
 * SYS_ioctl, and no such line of its own. */
bool is_ioctl(long nr) { return nr == 16; }
#else
bool is_ioctl(long nr) { return nr == SYS_ioctl; }
#endif

#if defined(_WIN32)
/* ---- Brilliance: the stack of a thread that waits, copied while it is suspended ----
 *
 * Windows has no signal a thread can be made to take, so the thread is suspended, its registers
 * and the top of its stack are copied, and it is resumed. Nothing between the suspend and the
 * resume takes a user-mode lock, since the suspended thread may hold any of them (the heap's, the
 * loader's). The frames are then picked out of the copy: the instruction pointer, and every word
 * that points just past a call instruction in a loaded image. A scan, not an unwind -- an unwind
 * looks up function tables under the loader's lock -- so a stale return address left on the stack
 * can appear; the first frames are the thread's own.
 *
 * A suspension does not end a wait early, as a signal can on Linux: the kernel resumes the wait
 * afterwards, so a thread in a driver call is not left out. But reading its registers waits until
 * the kernel can deliver that request to the thread, and a thread blocked on a driver's lock may
 * not allow it for as long as the lock is held. So the copy is made by a capture thread, and the
 * report waits for it at most 2 s, as the Linux report waits for its signal. That thread is
 * started with the first watch, while nothing is stalled: a thread started during a stall can
 * itself wait on the loader lock the stalled thread holds. */
constexpr int    kMaxFrames = 48;
constexpr size_t kStackCopy = 64 * 1024;

/* The request a watch hands the capture thread, and its answer. One request is out at a time;
 * one the capture thread is still stuck on is answered late and ignored. */
struct StackGrab {
    std::timed_mutex        one;             /* held by the watch whose request is out */
    std::mutex              mu;
    std::condition_variable cv;
    uint64_t                asked = 0;       /* request numbers */
    uint64_t                taken = 0;
    uint64_t                answered = 0;
    DWORD                   tid = 0;
    uintptr_t               ip = 0;          /* the answer to `answered` */
    std::vector<uint8_t>    stack;           /* the stack from the stack pointer upward */
    const char*             why = nullptr;   /* or why there is none */
};
StackGrab*     g_grab = nullptr;   /* never freed: the capture thread lives as long as the process */
std::once_flag g_install_once;

/* Suspends thread `tid` of this process, copies its instruction pointer and up to `cap` bytes of
 * its stack into `copy`, and resumes it. Null, or why it could not. */
const char* copy_thread(DWORD tid, uint8_t* copy, size_t cap, uintptr_t* ip, size_t* copied) {
#if defined(_M_X64) || defined(__x86_64__)
    HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
                              THREAD_QUERY_LIMITED_INFORMATION,
                          FALSE, tid);
    if (!h) return "the thread could not be opened";
    /* A thread id is reused once its thread has gone, and may then name another process's. */
    if (GetProcessIdOfThread(h) != GetCurrentProcessId()) {
        CloseHandle(h);
        return "the thread has exited";
    }
    const char* why = nullptr;
    if (SuspendThread(h) == (DWORD)-1) {
        why = "the thread could not be suspended";
    } else {
        CONTEXT ctx{};
        ctx.ContextFlags = CONTEXT_CONTROL;
        /* SuspendThread only asks; reading the registers is what waits for the thread to stop. */
        if (!GetThreadContext(h, &ctx)) {
            why = "the thread's registers could not be read";
        } else {
            *ip = (uintptr_t)ctx.Rip;
            /* The committed stack from the stack pointer up. A guard page at the stack pointer
             * (a frame not yet touched) is stepped over: reading it would consume the guard the
             * thread's stack grows by. */
            uintptr_t at = (uintptr_t)ctx.Rsp;
            MEMORY_BASIC_INFORMATION mb{};
            for (int i = 0; i < 2 && VirtualQuery((const void*)at, &mb, sizeof mb); ++i) {
                const uintptr_t end = (uintptr_t)mb.BaseAddress + mb.RegionSize;
                if (mb.State == MEM_COMMIT && !(mb.Protect & (PAGE_GUARD | PAGE_NOACCESS))) {
                    *copied = std::min<size_t>(cap, (size_t)(end - at)) & ~(size_t)7;
                    std::memcpy(copy, (const void*)at, *copied);
                    break;
                }
                at = end;
            }
        }
        ResumeThread(h);
    }
    CloseHandle(h);
    return why;
#else
    (void)tid; (void)copy; (void)cap; (void)ip; (void)copied;
    return "no stack on this architecture";
#endif
}

void capture_loop(StackGrab* g) {
    std::vector<uint8_t> copy(kStackCopy);   /* allocated here, never while a thread is suspended */
    uint64_t done = 0;
    for (;;) {
        DWORD tid = 0;
        {
            std::unique_lock<std::mutex> lk(g->mu);
            g->cv.wait(lk, [&] { return g->asked != done; });
            done = g->taken = g->asked;
            tid = g->tid;
        }
        uintptr_t ip = 0;
        size_t copied = 0;
        const char* why = copy_thread(tid, copy.data(), copy.size(), &ip, &copied);
        {
            std::lock_guard<std::mutex> lk(g->mu);
            g->answered = done;
            g->ip = ip;
            g->why = why;
            g->stack.assign(copy.data(), copy.data() + copied);
        }
        g->cv.notify_all();
    }
}

void install_stack_handler() {
    std::call_once(g_install_once, [] {
        g_grab = new StackGrab;
        std::thread(capture_loop, g_grab).detach();
    });
}

/* Whether `ret` is the address just after a call instruction in a loaded image's code: a direct
 * call, or an indirect one through a register, [reg], [reg+disp] or [rip+disp] (an import). */
bool after_call(uintptr_t ret) {
    MEMORY_BASIC_INFORMATION mb{};
    if (ret < 8 || !VirtualQuery((const void*)(ret - 7), &mb, sizeof mb) ||
        mb.State != MEM_COMMIT || mb.Type != MEM_IMAGE || (mb.Protect & PAGE_GUARD))
        return false;
    const DWORD exec = PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                       PAGE_EXECUTE_WRITECOPY;
    if (!(mb.Protect & exec) || (uintptr_t)mb.BaseAddress + mb.RegionSize < ret) return false;
    const uint8_t* p = (const uint8_t*)ret;
    if (p[-5] == 0xE8) return true;                              /* call rel32 */
    if (p[-6] == 0xFF && (p[-5] & 0x38) == 0x10) return true;    /* call [rip/reg+disp32] */
    if (p[-3] == 0xFF && (p[-2] & 0x38) == 0x10) return true;    /* call [reg+disp8] */
    if (p[-2] == 0xFF && (p[-1] & 0x38) == 0x10) return true;    /* call reg, call [reg] */
    return false;
}

std::string stack_of(pid_t tid) {
    install_stack_handler();
    StackGrab* g = g_grab;
    std::unique_lock<std::timed_mutex> one(g->one, std::chrono::seconds(2));
    if (!one.owns_lock()) return "no stack: another watch's request did not finish within 2 s";
    uintptr_t ip = 0;
    std::vector<uint8_t> stack;
    {
        std::unique_lock<std::mutex> lk(g->mu);
        const uint64_t want = ++g->asked;
        g->tid = (DWORD)tid;
        g->cv.notify_all();
        if (!g->cv.wait_for(lk, std::chrono::seconds(2), [&] { return g->answered == want; }))
            return g->taken == want
                       ? "no stack: the thread did not stop within 2 s, as a wait inside a driver "
                         "can keep it from doing"
                       : "no stack: the capture is still waiting for an earlier thread to stop";
        if (g->why) return g->why;
        ip = g->ip;
        stack.swap(g->stack);
    }
    std::string s = stall_module_offset(ip);
    int n = 1;
    for (size_t o = 0; o + 8 <= stack.size() && n < kMaxFrames; o += 8) {
        uintptr_t v;
        std::memcpy(&v, stack.data() + o, 8);
        if (!after_call(v)) continue;
        s += " " + stall_module_offset(v);
        ++n;
    }
    return s;
}
#else
/* ---- the stack of a thread that waits in user space, taken by a signal it handles itself ---- */

constexpr int kMaxFrames = 48;
std::atomic<pid_t> g_want_tid{0};
std::atomic<int>   g_have{0};
void*              g_frames[kMaxFrames];
int                g_n_frames = 0;

void on_stack_signal(int) {
    if ((pid_t)syscall(SYS_gettid) != g_want_tid.load(std::memory_order_acquire)) return;
    g_n_frames = backtrace(g_frames, kMaxFrames);
    g_have.store(1, std::memory_order_release);
}

/* A real-time signal nothing else in the process uses (glibc keeps the first few for itself). */
int stack_signal() { return SIGRTMIN + 7; }

std::once_flag g_install_once;
void install_stack_handler() {
    std::call_once(g_install_once, [] {
        /* backtrace() loads its unwinder on first use; done here, outside any handler. */
        void* prime[2];
        (void)backtrace(prime, 2);
        struct sigaction sa {};
        sa.sa_handler = on_stack_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(stack_signal(), &sa, nullptr);
    });
}

std::string stack_of(pid_t tid) {
    install_stack_handler();
    g_have.store(0, std::memory_order_relaxed);
    g_want_tid.store(tid, std::memory_order_release);
    if (syscall(SYS_tgkill, getpid(), tid, stack_signal()) != 0) {
        g_want_tid.store(0);
        return "the thread could not be signalled";
    }
    for (int i = 0; i < 200 && !g_have.load(std::memory_order_acquire); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    g_want_tid.store(0, std::memory_order_release);
    if (!g_have.load(std::memory_order_acquire))
        return "no stack: the thread did not take the signal within 2 s";
    std::string s;
    /* Frame 0 and 1 are the handler and the signal trampoline. */
    for (int i = 2; i < g_n_frames; ++i)
        s += (s.empty() ? "" : " ") + stall_module_offset((uintptr_t)g_frames[i]);
    return s;
}
#endif

}  /* namespace */

#if defined(_WIN32)
/* Brilliance: the image that maps `addr` and its offset from the image's base: the module+offset
 * a debugger prints and a lookup in the image's PDB takes. Asked of the memory manager
 * (VirtualQuery, GetMappedFileNameW), not the loader, whose lock a stalled thread may hold. An
 * executable is named without ".exe", as Linux names it. */
std::string stall_module_offset(uintptr_t addr) {
    MEMORY_BASIC_INFORMATION mb{};
    wchar_t w[1024];
    DWORD n = 0;
    if (VirtualQuery((const void*)addr, &mb, sizeof mb) && mb.State == MEM_COMMIT &&
        mb.Type == MEM_IMAGE)
        n = GetMappedFileNameW(GetCurrentProcess(), (void*)addr, w, (DWORD)(sizeof w / sizeof w[0]));
    if (n == 0) return fmt("0x%llx", (unsigned long long)addr);
    std::string name = os::path_basename(os::narrow(std::wstring(w, n)));
    if (name.size() > 4 && _stricmp(name.c_str() + name.size() - 4, ".exe") == 0)
        name.resize(name.size() - 4);
    return fmt("%s+0x%llx", name.c_str(),
               (unsigned long long)(addr - (uintptr_t)mb.AllocationBase));
}
#else
std::string stall_module_offset(uintptr_t addr) {
    /* Relative to the module's load base (lowest mapping start minus its file offset), which is
     * what addr2line takes for PIC code. The file offset differs where the linker places a
     * segment at an address unlike its file position, as lld does. */
    struct Map { unsigned long lo, hi, off; std::string path; };
    std::vector<Map> maps;
    {
        std::ifstream f("/proc/self/maps");
        std::string line;
        while (std::getline(f, line)) {
            Map m{};
            char path[512] = "";
            if (std::sscanf(line.c_str(), "%lx-%lx %*s %lx %*s %*s %511s", &m.lo, &m.hi, &m.off,
                            path) < 3)
                continue;
            m.path = path;
            maps.push_back(std::move(m));
        }
    }
    for (const Map& m : maps) {
        if (addr < m.lo || addr >= m.hi) continue;
        if (m.path.empty() || m.path[0] != '/') return fmt("0x%lx", (unsigned long)addr);
        unsigned long base = m.lo - m.off;
        for (const Map& o : maps)
            if (o.path == m.path && o.lo - o.off < base) base = o.lo - o.off;
        const size_t slash = m.path.rfind('/');
        return fmt("%s+0x%lx", m.path.c_str() + slash + 1, (unsigned long)(addr - base));
    }
    return fmt("0x%lx", (unsigned long)addr);
}
#endif

std::string stall_decode_syscall(const std::string& line) {
    if (line.empty()) return "unknown";
    if (line == "running") return "running";
    std::istringstream in(line);
    long nr = 0;
    if (!(in >> nr)) return "unknown";
    if (nr < 0) return "not in a system call";
    std::vector<unsigned long> a;
    std::string w;
    while (in >> w) a.push_back(std::strtoul(w.c_str(), nullptr, 16));
    /* Six arguments, then the stack pointer and the program counter. */
    const unsigned long pc = a.size() >= 8 ? a[7] : 0;
    std::string s;
    if (is_ioctl(nr) && a.size() >= 2) {
        const long fd = (long)a[0];
        const unsigned long req = a[1];
        char target[256] = "";
        const ssize_t n = readlink(fmt("/proc/self/fd/%ld", fd).c_str(), target, sizeof target - 1);
        if (n > 0) target[n] = 0; else std::snprintf(target, sizeof target, "fd %ld", fd);
        const unsigned type = (unsigned)((req >> 8) & 0xff), num = (unsigned)(req & 0xff);
        const char* kfd = type == 'K' ? kfd_ioctl_name(num) : nullptr;
        s = kfd ? fmt("ioctl(%s, AMDKFD_IOC_%s)", target, kfd)
                : fmt("ioctl(%s, request 0x%lx: type '%c' number 0x%x)", target, req,
                      type >= 0x20 && type < 0x7f ? (char)type : '?', num);
    } else if (const char* name = syscall_name(nr)) {
        s = name;
    } else {
        s = fmt("system call %ld", nr);
    }
    if (pc) s += ", called from " + stall_module_offset(pc);
    return s;
}

#if defined(_WIN32)
/* Brilliance: Windows keeps no per-thread state, wait channel or system call that a process can
 * read, so a thread is named by its id and its description, and its stack says where it waits. A
 * thread in a driver call is not spared the stack: a suspension does not end its wait. */
std::string stall_describe_thread(pid_t tid, bool stack) {
    std::string s = fmt("tid %d", (int)tid);
    if (HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)tid)) {
        PWSTR d = nullptr;
        if (GetProcessIdOfThread(h) == GetCurrentProcessId() &&
            SUCCEEDED(GetThreadDescription(h, &d)) && d && *d)
            s += " (" + os::narrow(d) + ")";
        if (d) LocalFree(d);
        CloseHandle(h);
    }
    if (stack) s += "\n    stack: " + stack_of(tid);
    return s;
}
#else
std::string stall_describe_thread(pid_t tid, bool stack) {
    const std::string dir = fmt("/proc/self/task/%d/", (int)tid);
    const std::string stat = read_first_line(dir + "stat");
    /* The state is the field after the parenthesised name, which may itself hold spaces. */
    char state = '?';
    const size_t rp = stat.rfind(')');
    if (rp != std::string::npos && rp + 2 < stat.size()) state = stat[rp + 2];
    const std::string wchan = read_first_line(dir + "wchan");
    const std::string sc = read_first_line(dir + "syscall");
    std::string s = fmt("tid %d (%s), state %c", (int)tid, read_first_line(dir + "comm").c_str(),
                        state);
    if (!wchan.empty() && wchan != "0") s += ", waiting in the kernel at " + wchan;
    s += ", " + stall_decode_syscall(sc);
    long nr = -1;
    std::sscanf(sc.c_str(), "%ld", &nr);
    /* A thread in a driver call is not signalled: the call already says where it waits, and the
     * signal could end the wait early instead of restarting it. */
    if (stack && !is_ioctl(nr) && state != 'D') s += "\n    stack: " + stack_of(tid);
    return s;
}
#endif

struct StallWatch::Impl {
    std::string             phase;
    uint64_t                quiet_ns = 0;
    uint64_t                poll_ns = 0;
    pid_t                   holder = 0;
    std::mutex              mu;
    std::condition_variable cv;
    bool                    stop = false;
    std::thread             th;

    void report(uint64_t silent_ns) {
#if defined(_WIN32)
        /* Brilliance: no thread's system call can be read, so the threads in a driver call cannot
         * be picked out and there are no "also in a driver call" lines; the holder's stack names
         * the module it waits in. */
        std::string others;
        int n_others = 0;
#else
        const pid_t me = (pid_t)syscall(SYS_gettid);
        std::string others;
        int n_others = 0;
        if (DIR* d = opendir("/proc/self/task")) {
            while (dirent* e = readdir(d)) {
                const pid_t t = (pid_t)std::atoi(e->d_name);
                if (t <= 0 || t == holder || t == me) continue;
                long nr = -1;
                std::sscanf(read_first_line(fmt("/proc/self/task/%d/syscall", (int)t)).c_str(),
                            "%ld", &nr);
                if (!is_ioctl(nr)) continue;
                if (n_others++ < 8) others += "\n  also in a driver call: " +
                                              stall_describe_thread(t, false);
            }
            closedir(d);
        }
#endif
        if (n_others > 8) others += fmt("\n  and %d more threads in a driver call", n_others - 8);
        RAD_WARN("%s: nothing has been logged for %.0f s since the line above. The thread doing "
                 "the %s is %s%s\n  A report of a %s that stops here should include these lines.",
                 phase.c_str(), (double)silent_ns / 1e9, phase.c_str(),
                 stall_describe_thread(holder, true).c_str(), others.c_str(), phase.c_str());
    }

    void run(const std::atomic<uint64_t>& beat) {
        uint64_t since = rad_mono_ns();       /* the start of the current silence */
        uint64_t own = 0;                     /* the time of this watch's own last report */
        uint64_t next = quiet_ns;             /* the silence that is reported next */
        std::unique_lock<std::mutex> lk(mu);
        while (!cv.wait_for(lk, std::chrono::nanoseconds(poll_ns), [&] { return stop; })) {
            const uint64_t last = std::max(log_last_line_ns(),
                                           beat.load(std::memory_order_relaxed));
            if (last > since && last != own) { since = last; next = quiet_ns; }
            const uint64_t now = rad_mono_ns();
            if (now - since < next) continue;
            lk.unlock();
            report(now - since);
            lk.lock();
            own = log_last_line_ns();
            next *= 2;
        }
    }
};

StallWatch::StallWatch(const char* phase, uint64_t quiet_ms) : p_(std::make_unique<Impl>()) {
    p_->phase = phase ? phase : "phase";
    p_->quiet_ns = quiet_ms * 1000000ull;
    /* A quarter of the quiet interval, so a report is at most that late, and at most a second. */
    p_->poll_ns = std::max<uint64_t>(1000000ull, std::min<uint64_t>(1000000000ull,
                                                                    p_->quiet_ns / 4));
#if defined(_WIN32)
    /* Brilliance: the Windows thread id, the one OpenThread takes. */
    p_->holder = (pid_t)GetCurrentThreadId();
#else
    p_->holder = (pid_t)syscall(SYS_gettid);
#endif
    install_stack_handler();
    p_->th = std::thread([this] { p_->run(beat_); });
}

void StallWatch::progress() { beat_.store(rad_mono_ns(), std::memory_order_relaxed); }

StallWatch::~StallWatch() {
    {
        std::lock_guard<std::mutex> lk(p_->mu);
        p_->stop = true;
    }
    p_->cv.notify_all();
    if (p_->th.joinable()) p_->th.join();
}

}  /* namespace rad */
