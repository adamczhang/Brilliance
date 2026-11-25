/* Brilliance: the operating-system services whose Windows meaning differs from POSIX, as
 * explicit calls.
 *
 * win/compat makes the POSIX functions that carry over to Windows unchanged simply work. This
 * header is for the ones that do not: positional I/O at 64-bit offsets, unbuffered (direct) I/O
 * with its alignment rules, file mapping, address-space reservation with a commit you can take
 * back, aligned allocation whose pointer must be freed by its own function, the facts about
 * memory and paths a POSIX call would answer from /proc or $HOME. A radiance source reaches these
 * from an `#if defined(_WIN32)` branch, so its Linux code is left exactly as radiance wrote it
 * (docs/WINDOWS-PORT.md section 8).
 *
 * Conventions, chosen so a Windows branch reads like the POSIX code beside it: paths are UTF-8;
 * a failing call returns -1 (or null, or an invalid object) AND sets errno, so the branch reports
 * it with the same strerror(errno) the POSIX branch uses. No <windows.h> here: this header is
 * included by sources that also include radiance's own headers, and windows.h's macros (min, max,
 * ERROR, near, far, small...) do not belong in them. */
#pragma once
#if defined(_WIN32)

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "job_limit.h"

namespace rad::os {

/* ----------------------------------------------------------------------------- strings */
std::wstring widen(const std::string& utf8);
std::string  narrow(const std::wstring& utf16);
/* The system's message for a Win32 error code (GetLastError() when `err` is 0). */
std::string  win32_message(unsigned long err = 0);

/* ----------------------------------------------------------------------------- files
 * A file handle with positional I/O. Always opened with FILE_SHARE_READ | WRITE | DELETE, so it
 * behaves like a POSIX descriptor towards other openers: it can be renamed over or deleted while
 * open, and another reader is not refused. read_at/write_at are thread-safe on one File (each
 * carries its own offset, as pread/pwrite do) and move any length, in pieces under 1 GiB. */
class File {
public:
    enum : unsigned {
        Read     = 1u << 0,
        Write    = 1u << 1,
        Create   = 1u << 2,   /* create if missing */
        Trunc    = 1u << 3,   /* truncate to zero (implies Create) */
        Excl     = 1u << 4,   /* fail if it exists (with Create) */
        Direct   = 1u << 5,   /* FILE_FLAG_NO_BUFFERING: offsets, lengths, buffers sector-aligned */
        Private  = 1u << 6,   /* a new file readable by its owner only (owner-only DACL) */
        DeleteOnClose = 1u << 7,
        Random   = 1u << 8,   /* FILE_FLAG_RANDOM_ACCESS: no read-ahead */
        Sequential = 1u << 9, /* FILE_FLAG_SEQUENTIAL_SCAN */
    };

    File() = default;
    ~File() { close(); }
    File(File&& o) noexcept : h_(o.h_), flags_(o.flags_) { o.h_ = nullptr; }
    File& operator=(File&& o) noexcept;
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    /* Open `path`. On failure valid() is false and errno is set. */
    static File open(const std::string& path, unsigned flags);

    bool     valid() const { return h_ != nullptr; }
    unsigned flags() const { return flags_; }
    void     close();

    /* Bytes moved, which is short only at end of file; -1 with errno on an error. */
    int64_t read_at(void* buf, int64_t n, int64_t off) const;
    int64_t write_at(const void* buf, int64_t n, int64_t off) const;
    /* The whole range or -1: a short read is EIO. */
    int     read_exact(void* buf, int64_t n, int64_t off) const;
    int     write_exact(const void* buf, int64_t n, int64_t off) const;

    int64_t size() const;                 /* -1 with errno on failure */
    int     set_size(int64_t bytes) const;/* grow (zero-filled) or shrink */
    int     sync() const;                 /* FlushFileBuffers */
    /* Sparse: a hole reads as zeros and takes no space. Best effort (0 where unsupported). */
    int     set_sparse() const;
    int     punch_hole(int64_t off, int64_t len) const;
    /* The alignment unbuffered I/O on this file requires (physical sector size); 4096 if unknown. */
    int64_t direct_alignment() const;
    /* Volume serial and 128-bit file id: equal for the same file under any path. */
    bool    same_file(const File& other) const;
    void*   native() const { return h_; }
    /* A second, independent handle on the same file (DuplicateHandle). */
    File    dup() const;

private:
    void*    h_ = nullptr;
    unsigned flags_ = 0;
};

/* ----------------------------------------------------------------------------- mappings
 * A read-only view of a whole file (CreateFileMappingW + MapViewOfFile). Views of 100+ GiB are
 * fine on x64. The file stays readable and deletable by others while mapped. */
class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile() { unmap(); }
    MappedFile(MappedFile&& o) noexcept;
    MappedFile& operator=(MappedFile&& o) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    /* Map all of `path` read-only. False with errno on failure. A zero-length file maps to a
     * valid, empty view (data() null, size() 0). */
    bool map_readonly(const std::string& path);
    void unmap();

    const uint8_t* data() const { return (const uint8_t*)base_; }
    int64_t        size() const { return size_; }
    bool           valid() const { return base_ != nullptr || mapped_empty_; }

private:
    void*   base_ = nullptr;
    int64_t size_ = 0;
    bool    mapped_empty_ = false;
};

/* Prefetch a range of a mapping into memory (PrefetchVirtualMemory): madvise(MADV_WILLNEED). */
void prefetch(const void* addr, int64_t len);

/* ----------------------------------------------------------------------------- memory */
/* Aligned heap memory. The pointer must be released with aligned_free, never free(). */
void*   aligned_alloc(size_t align, size_t bytes);
void    aligned_free(void* p);

/* Anonymous committed read-write pages (VirtualAlloc). `large` asks for large pages and falls
 * back to normal ones when the privilege is missing. */
void*   anon_alloc(int64_t bytes, bool large = false);
void    anon_free(void* p);

/* Address space without memory, then memory in it. Windows charges a commit against RAM plus
 * page file at once (there is no overcommit), and a reservation is released only whole. */
void*   vmem_reserve(int64_t bytes, int64_t align);
int     vmem_commit(void* addr, int64_t bytes);    /* read-write, zero-filled */
int     vmem_decommit(void* addr, int64_t bytes);  /* back to reserved; next commit is zeroed */
int     vmem_protect_rw(void* addr, int64_t bytes);
void    vmem_release(void* base);

/* Lock pages in RAM, raising the working-set bounds first (the Windows limit on locked memory). */
int     lock_memory(const void* addr, int64_t bytes);
void    unlock_memory(const void* addr, int64_t bytes);

int64_t page_size();
struct MemInfo {
    int64_t total = -1;       /* physical RAM */
    int64_t avail = -1;       /* available to allocate without paging */
    int64_t commit_limit = -1;/* RAM + page file */
    int64_t commit_avail = -1;
};
MemInfo mem_info();
/* This process's working set and private commit, bytes. */
int64_t process_rss();
/* Brilliance (issue #1): this process's private commit (PrivateUsage), bytes; -1 unreadable. */
int64_t process_commit();
/* Brilliance (issue #1): the memory limits of the Windows job object this process runs in
 * (job_limit.h); in_job false and no limits outside one. */
JobMemoryLimit job_memory_limit();

/* Brilliance: every AMD adapter's negotiated PCIe lane width, by PCI bus number.
 *
 * Windows reports no link for the slot of a card that carries its own PCIe switch (the R9700 does):
 * the GPU's own link is the one inside the card and reads x16 in any slot, and the switch's upstream
 * port has no link properties at all. AMD's driver knows the slot's width, and its ADL library
 * (atiadlxx.dll, installed with every AMD driver) says it without admin rights or a GPU call:
 * ADL2_Adapter_ChipSetInfo_Get's iCurrentPCIELaneWidth. It read 8, 8, 4, 4 on four R9700s whose
 * copy rates were 28.5, 28.5, 14.4, 14.4 GB/s. It does not say the negotiated speed, so a caller
 * that needs rates takes the generation as equal across the cards.
 *
 * Empty when the library, one of its calls or an adapter's answer is missing: never a guess. */
struct PciLanes { int bus = -1; int width = 0; };
std::vector<PciLanes> amd_lane_widths();

/* ----------------------------------------------------------------------------- paths */
std::string exe_path();             /* this executable, UTF-8 */
std::string exe_dir();
std::string local_app_data();       /* %LOCALAPPDATA% */
std::string program_data();         /* %ProgramData% */
std::string temp_dir();             /* GetTempPathW, no trailing separator */
std::string system_dir();           /* %SystemRoot%\System32: where a system tool is started from */
constexpr char kPathListSep = ';';
/* The plugin home when nothing names one: <exe>\..\share\brilliance in an install, else
 * <exe>\..\radiance_home in a build tree, else the configured RAD_DEFAULT_HOME. */
std::string default_home();
/* The part after the last '/' or '\\'. */
std::string path_basename(const std::string& p);
/* The file's size and last-write time as seconds and nanoseconds since 1970 (100 ns resolution):
 * what stat's st_size and st_mtim give on Linux, where _stat64 keeps whole seconds only. False
 * with errno when the file cannot be read. */
bool        file_size_mtime(const std::string& p, int64_t* size, int64_t* sec, int64_t* nsec);
/* The absolute, normalised UTF-8 path (GetFullPathNameW): realpath without the link resolution,
 * which is what a name built from it needs. `p` itself when it cannot be resolved. */
std::string full_path(const std::string& p);
bool        is_dir(const std::string& p);
bool        is_file(const std::string& p);
/* mkdir -p. `owner_only` gives each directory it creates an owner-only, inherited DACL. */
int         make_dirs(const std::string& p, bool owner_only = false);
/* rename(2): replaces an existing target atomically on one volume. */
int         rename_replace(const std::string& from, const std::string& to);
/* unlink(2), including a file another handle still has open (POSIX delete semantics). */
int         remove_file(const std::string& p);
/* The other processes that have `p` open (Restart Manager): what radiance answers from /proc on
 * Linux before it rewrites a container another engine is serving. This process is left out. */
std::vector<long> processes_using(const std::string& p);

/* ----------------------------------------------------------------------------- shared files
 * What the n-gram table share (core/format/table_share.cpp, --ngram-placement ram=DIR) takes from
 * open(2), flock(2) and st_ino, on CRT descriptors: the table's file reaches rad_dev_map_file as an
 * fd, and pread, pwrite, fstat, ftruncate and close work on it, as on Linux. */

/* open(2) of a file that several processes hold and the last of them deletes: a descriptor over a
 * handle opened FILE_SHARE_READ | WRITE | DELETE -- the CRT's open() leaves DELETE out, and the
 * leaving holder could then not remove the file while a newcomer has it open -- not inherited by
 * children. `create` makes the file if it is missing, with FILE_ATTRIBUTE_TEMPORARY: the cache
 * manager keeps a temporary file's pages in memory while memory allows rather than writing them
 * out, which is as near as a Windows directory comes to a tmpfs. A volume without POSIX delete
 * keeps a deleted file at its name, refusing every open, until its last handle closes; a create
 * waits up to two seconds for that. -1 with errno. */
int  open_shared_fd(const std::string& path, bool write, bool create);

/* flock(2) on a descriptor from open_shared_fd: LockSh or LockEx, with LockNb not to wait, or
 * LockUn. A whole-file advisory lock held by the descriptor's open file -- every open is a holder
 * of its own, in one process or several -- and let go when the descriptor is closed or the process
 * ends, however it ends.
 *
 * It is LockFileEx on one byte at 2^62, past the end of any file. Windows enforces a byte-range
 * lock against ReadFile and WriteFile inside its range -- from every other handle, and for a shared
 * lock from its own as well -- so a lock over the data would stop the filler writing the table.
 * Nothing reads or writes at 2^62, so there the lock is advisory, as flock's is. A lock asked for
 * replaces the one the descriptor holds, which is let go first: flock(2) converts a lock the same
 * way, and says the conversion is not atomic. 0, or -1 with errno: EWOULDBLOCK under LockNb when
 * another holder's lock is in the way. */
enum : int { LockSh = 1, LockEx = 2, LockNb = 4, LockUn = 8 };   /* flock(2)'s values */
int  flock_fd(int fd, int op);

/* Whether `path` names the file `fd` has open, now (volume serial and file id): what POSIX code
 * compares st_dev and st_ino for, or st_nlink with 0, since st_ino carries no identity on Windows.
 * False when nothing is at `path`, or the file there is a deleted one still open somewhere. */
bool fd_names(int fd, const std::string& path);

/* A file's revision as POSIX code reads it from stat(2): st_dev, st_ino, st_size, and st_mtim and
 * st_ctim to the nanosecond. Here from the file's handle: the volume serial number, the file id
 * (FILE_ID_INFO's 128 bits folded to 64; only compared), the size, and the last-write and change
 * times (100 ns resolution). _stat64 has no inode, keeps whole seconds, and its st_ctime is the
 * creation time, so a file rewritten in place within one second would keep its revision. False with
 * errno when the file cannot be read or is not a regular file (EISDIR for a directory). */
struct FileStamp {
    uint64_t device = 0, inode = 0, bytes = 0;
    int64_t  mtime_s = 0, ctime_s = 0;
    int32_t  mtime_ns = 0, ctime_ns = 0;
};
bool file_stamp_fd(int fd, FileStamp* out);
bool file_stamp(const std::string& path, FileStamp* out);

/* ----------------------------------------------------------------------------- processes */
/* One argument quoted so CommandLineToArgvW -- and so the CRT, and the SCM starting a service --
 * splits it back unchanged: backslashes are literal except before a quote and at the end, where
 * they double. Left bare when it has no space, tab or quote and is not empty. */
std::string quote_arg(const std::string& arg);

/* Start `cmdline` (UTF-8, each argument quote_arg'ed) with `env_extra` lines "NAME=value" over
 * this process's environment -- a name already set takes the new value, compared without regard
 * to case as Windows compares them -- and wait at most `timeout_ms` (-1 forever). Returns the
 * exit code (an NTSTATUS such as 0xC0000005 when it crashed), -1 with errno when it could not
 * start, and -2 on a timeout (the child is terminated). */
long long   spawn_and_wait(const std::string& cmdline, const std::string& env_extra = "",
                           long timeout_ms = -1);

/* Whether a process with id `pid` is running now: what POSIX code asks with kill(pid, 0). A process
 * that has exited but whose handle another process still holds counts as gone. */
bool        process_alive(long long pid);

/* A child whose stdout and stderr share one pipe its parent reads: what the tests' fork, dup2 and
 * pipe2 do on Linux (tests/tiny_model.h).
 *
 * THE CHILD CAN BE STOPPED THE WAY SIGTERM STOPS IT (interrupt). It is started in a process group
 * of its own, on this process's console -- attaching to the parent's console first when this
 * process has none -- so a CTRL_BREAK_EVENT for that group reaches it and nothing else: the engine
 * takes it, through on_terminate(), as SIGTERM, drains, and exits with its own status. Where no
 * console can be shared (this process was started detached and its parent has none either), the
 * child gets a console of its own without a window (CREATE_NO_WINDOW), and interrupt() says why it
 * cannot reach it rather than killing it.
 *
 * Only the pipe's write end and this process's standard input are inherited (a handle list), so a
 * socket or file this process has open -- inheritable or not -- never stays open in the child. */
class Captured {
public:
    Captured() = default;
    ~Captured();
    Captured(const Captured&) = delete;
    Captured& operator=(const Captured&) = delete;

    /* Start argv[0] (a path, UTF-8) with `env` as its WHOLE environment ("NAME=value" entries).
     * False with errno when it cannot be started. */
    bool start(const std::vector<std::string>& argv, const std::vector<std::string>& env);
    /* Append what the child writes to `out` until it closes its end of the pipe, or for at most
     * `timeout_ms` when that is positive; false when the time ran out. */
    bool read_all(std::string* out, int64_t timeout_ms);
    /* Up to `n` bytes of what the child writes, blocking until some arrive: read(2) on the Linux
     * child's pipe. 0 at the end of its output -- the child, and anything that inherited its
     * output, has exited; -1 with errno on an error, EBADF once close_read() has run. Never
     * EINTR. For a drain thread; close_read() and wait() must not run while it is inside. */
    int64_t read_some(void* buf, size_t n);
    /* Close this process's end of the pipe: close(2) on the Linux child's fd. */
    void close_read();
    /* The child's process id from start() until wait() has collected it, else -1. */
    long pid() const { return proc_ ? pid_ : -1; }
    /* Ask the child to end as kill(pid, SIGTERM) asks on Linux: CTRL_BREAK_EVENT to its process
     * group. True when the event was sent or the child had already exited (nothing left to ask);
     * false with errno, and the reason in *why, when it cannot be delivered -- the child shares no
     * console with this process -- or nothing was started. Never kills: that is kill()'s. */
    bool interrupt(std::string* why = nullptr);
    /* Whether interrupt() will reach a child start() makes now, attaching to the parent's console
     * if this process has none; when not, the reason in *why. */
    static bool can_interrupt(std::string* why = nullptr);
    /* End the child now (TerminateProcess), as SIGKILL does. */
    void kill();
    /* Wait for the child to exit and return its exit code (an NTSTATUS such as 0xC0000005 when it
     * crashed); -1 when nothing was started. */
    int  wait();

private:
    void* proc_ = nullptr;
    void* rd_ = nullptr;
    long  pid_ = -1;
    bool  console_ = false;   /* started on this process's console, so interrupt() reaches it */
    std::string no_console_;  /* when not, why */
};

/* Start `cmdline` without waiting for it: a handle for terminate_and_wait, or null with errno. */
void*       spawn(const std::string& cmdline);
void        terminate_and_wait(void* proc);

/* Call `fn` once on Ctrl-C, Ctrl-Break or the console's closing (on the handler's own thread). A
 * second such event is left to the default handler, which ends the process: the SA_RESETHAND
 * behaviour of the POSIX code. Log-off and shut-down do not arrive here -- Windows sends them to a
 * process that has gdi32 or user32 loaded, as the engine does, only as window messages -- so a
 * console server is ended at shut-down without its drain; the service is told (PRESHUTDOWN). */
void        on_terminate(void (*fn)(int sig));

/* Console: UTF-8 output and ANSI escape processing on stdout/stderr where they are consoles. */
void        console_init();

/* Deliver the termination on_terminate() registered for, as if Ctrl-C had arrived: what the
 * service's STOP control calls, so a service stop is the same graceful drain as a console one.
 * True when it was delivered now. False when nothing has registered yet -- the engine is still
 * reading its configuration or probing the cards -- and then the request is KEPT: the on_terminate()
 * that registers delivers it at once, and the engine stops before it loads anything. Never an
 * exit: a process that may have started work on a GPU is not ended from outside its own drain. */
bool        request_terminate(int sig);

/* SIGHUP's Windows counterpart: `fn` is called, on a watcher thread, each time the named event
 * reload_event_name(this pid) is set -- by win\tools\reload-server.ps1, or anything that can open
 * it. Global\ where the process may create it (a service), else Local\ (a console server in the
 * user's session); the reloader tries both. Once per process. */
void        on_reload(void (*fn)(int sig));
std::string reload_event_name(long long pid, bool global);

/* How many CPUs were busy, on average, over the next `window_ms` (GetSystemTimes; it sleeps that
 * long): Windows has no load average, and this is the figure diagnose mode's busy-host warning
 * compares with the CPU count where Linux reads /proc/loadavg. -1 when it cannot be read. */
double      busy_cpus(int window_ms);

/* ----------------------------------------------------------------------------- latency
 * Once, at the start of every Brilliance executable that serves:
 *   - opt the process out of power throttling (EcoQoS). Windows 11 otherwise parks cores under a
 *     background-looking process, and IGNORES its timer-resolution request (the Phase 0 probe saw
 *     timeBeginPeriod(1) do nothing until the process opted out);
 *   - then ask for a 1 ms timer, so the waits that do sleep sleep about as long as they asked.
 * BRILLIANCE_PRIORITY=high|above|normal sets the process priority class (default above-normal).
 * BRILLIANCE_CPUS confines the process to some logical processors: "l3:N" (the Nth L3 domain --
 * a 9950X3D CCD), "vcache" (the domain with the largest L3) or a list like "0-7,16-23". */
void        process_init();

/* Where this layer's own log lines go (process_init's "W process: ..." and "I process: ..."):
 * `level` numbered as the core's rad::Log (0 E, 1 W, 2 I, 3 D, 4 T), `line` without its level tag.
 * Unset, a line is printed to stderr as "<tag> <line>", as before. radiance sets it to the core's
 * logger (rad::log_line_at) before process_init, so these W lines are counted into the startup
 * report's FINDINGS like every other (log_warn_lines). Set once, before process_init. */
using LogSink = void (*)(int level, const char* line);
void        set_log_sink(LogSink sink);
/* One line through the sink, or to stderr: process_init's lines, and a test's. */
void        log_os_line(int level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

/* Wait about `us` microseconds without the 15.6 ms (or, after process_init, ~1-2 ms) granularity
 * of a sleep: up to half a millisecond is spent yielding the core, longer is a high-resolution
 * waitable timer. For backoff loops whose first steps are microseconds. */
void        sleep_us(int64_t us);

/* ----------------------------------------------------------------------------- GPU telemetry
 * What the driver reports of every hardware GPU (win/os/gpu_win32.cpp). From the graphics kernel's
 * adapter performance data (D3DKMT, WDDM 2.4+), for any vendor: engine and memory clocks,
 * temperature, fan, and power as a share of the adapter's rating. From AMD's ADL, where the AMD
 * driver installs it: board power in watts, GFX activity (busy) and the junction temperature, the
 * firmware metrics Linux reads from sysfs. -1 is "not reported" -- a driver that fills none of it
 * leaves all of it at -1, never 0.
 *
 * NEVER FROM WINDOWS' COUNTERS: Windows' per-engine running time reads 0% for a HIP kernel
 * spinning on an R9700 (measured 2026-11-16), so busy is ADL's or -1. vram_used stays -1: the
 * engine accounts its own VRAM. AMD's newer ADLX is not used: its SDK licence does not fit an
 * open-source engine (gpu_win32.cpp).
 *
 * Cards are named by PCI address as the engine spells them (dev_pci_address). `only`, when given,
 * names the cards wanted: the others are neither returned nor asked. Thread-safe; microseconds a
 * call. Off the step path: its caller is the dashboard's per-card figures (fleet.cpp). */
struct GpuSample {
    std::string bdf;                   /* "0000:03:00.0" */
    std::string name;                  /* the adapter's description */
    double  busy        = -1;          /* 0..1, ADL's GFX activity; -1 without ADL (see above) */
    double  gfx_mhz     = -1;          /* engine 0's clock now, and the driver's "max" -- the R9700's */
    double  gfx_max_mhz = -1;          /* is its 2350 MHz game clock, which it boosts well past */
    double  mem_mhz     = -1;          /* memory clock now, and its ceiling */
    double  mem_max_mhz = -1;
    double  temp_c      = -1;
    double  power_frac  = -1;          /* 0..1 of the adapter's rated power */
    double  power_w     = -1;          /* board power in watts, ADL's; -1 without ADL */
    double  fan_rpm     = -1;
    int64_t vram_used   = -1;          /* not reported on Windows (see above) */
};
std::vector<GpuSample> gpu_samples(const std::vector<std::string>* only = nullptr);

/* Every OTHER process's use of each named card's own memory (win/os/gpu_win32.cpp), for the
 * engine's startup report and its settle check (core/mem/vram_others.h). From the graphics kernel's
 * per-process counters, "\GPU Process Memory(*)\Dedicated Usage" (Task Manager's "dedicated GPU
 * memory"), readable without admin rights; the counter instances name the adapter by LUID, which
 * the adapter enumeration maps to the card's PCI address. This process is left out. Names come from
 * a process snapshot ("" when a process has exited meanwhile). Nothing here enters a GPU runtime.
 *
 * One entry per address asked, in that order. `known` is false for an address no adapter has, or
 * when the counters do not answer. Thread-safe; tens of milliseconds a call or more (the counters'
 * enumeration): startup and its checks only, never a request's path. */
struct GpuProcVram {
    uint32_t    pid = 0;
    std::string name;
    int64_t     bytes = 0;
};
struct GpuHolders {
    std::string              bdf;
    bool                     known = false;
    std::vector<GpuProcVram> procs;     /* those holding more than 0 bytes on the card */
};
std::vector<GpuHolders> gpu_vram_holders(const std::vector<std::string>& bdfs);
/* A counter instance's name, "pid_19340_luid_0x00000000_0x000181BB_phys_0", as pid and LUID (high
 * 32 bits from the first number). False for any other shape. */
bool gpu_process_instance(const std::string& instance, uint32_t* pid, uint64_t* luid);

/* ----------------------------------------------------------------------------- service
 * `radiance --service NAME --config FILE [--log FILE] [--env-from-file VAR=FILE]...
 * [--stop-timeout SECONDS]` is what the Service Control Manager starts (win/SERVICE.md;
 * win\serve.ps1 -Install registers it). It runs `run_engine` -- radiance's own main -- with
 * `--config FILE` on the service's thread, its output in the log, and turns STOP, PRESHUTDOWN and
 * SHUTDOWN into request_terminate(SIGTERM): the drain Ctrl-Break starts in a console. Started from
 * a console instead, it runs there exactly as the service would. Returns the process exit code.
 * service_wanted: whether the command line asks for it (a --service anywhere on it). */
bool        service_wanted(int argc, char** argv);
int         service_main(int argc, char** argv, int (*run_engine)(int, char**));
/* stdout and stderr -- the CRT streams, descriptors 1 and 2, and the standard handles children
 * inherit -- to the end of `path`, all through ONE handle opened for appending only, both streams
 * unbuffered. Works in a process started with no console, as the SCM starts a service. 0, or -1
 * with errno set. */
int         redirect_output(const std::string& path);

}   // namespace rad::os

#endif /* _WIN32 */
