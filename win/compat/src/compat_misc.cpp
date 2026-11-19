// Brilliance POSIX layer: the small functions (see win/compat/include/brl_compat.h for what the
// layer is and is not for).

#include <windows.h>
#include <winioctl.h>
#include <io.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <sys/stat.h>

#include <mutex>
#include <string>

#include "brl_compat.h"
#include "brl_prelude.h"
#include "unistd.h"
#include "sched.h"
#include "sys/resource.h"
#include "sys/ioctl.h"
#include "sys/time.h"
#include "sys/statvfs.h"
#include "pthread.h"
#include "compat_internal.h"

namespace brl {

HANDLE osfhandle(int fd) {
    intptr_t h;
    {
        QuietCrt quiet;
        h = _get_osfhandle(fd);
    }
    if (h == -1 || h == -2) {   // -2: a standard stream with no console behind it
        errno = EBADF;
        return INVALID_HANDLE_VALUE;
    }
    return (HANDLE)h;
}

}   // namespace brl

namespace {

// One call moves at most this much: ReadFile/WriteFile take a DWORD length, and the CRT's
// _read/_write an unsigned int.
constexpr DWORD kIoPiece = 1u << 30;

void at_offset(OVERLAPPED* ov, long long off) {
    ov->Offset = (DWORD)(unsigned long long)off;
    ov->OffsetHigh = (DWORD)((unsigned long long)off >> 32);
}

std::wstring widen(const char* s) {
    const int n = s ? MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0) : 0;
    if (n <= 0) return {};
    std::wstring w((size_t)n - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
    return w;
}

// A FILE THAT GROWS PAST ITS END GROWS BY A HOLE, which is what Linux gives for an ftruncate that
// extends a file or a pwrite that lands past its end: the gap is a hole, and a later write anywhere
// in it costs that write. NTFS without the sparse attribute allocates the growth but zero-fills from
// the last written byte (the valid data length) to wherever the next write lands, synchronously
// inside that write. Measured on the D: NM790 (WINDOWS-PORT.md 9.33): one 1 MiB pwrite 64 GiB into
// a new file took 15-19 s and wrote 68 GB, 4.4 GB/s of zeros; sparse, it took no time and wrote 1 MiB.
// So a file is made sparse before it grows. The price is that a sparse file is allocated as it is
// written back, in 64 KiB units, rather than up front -- 16 GiB written by 32 threads in shuffled
// 64 MiB pieces came to ~7,000 extents; the callers that want one extent (the prefix-cache tier)
// size their files through os::File, not here. Best effort: a volume without sparse files (FAT,
// exFAT) keeps the plain growth.
void grow_by_a_hole(HANDLE h, long long to) {
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(h, &size) || to <= size.QuadPart) return;
    FILE_BASIC_INFO bi{};
    if (GetFileInformationByHandleEx(h, FileBasicInfo, &bi, sizeof bi) &&
        (bi.FileAttributes & FILE_ATTRIBUTE_SPARSE_FILE))
        return;
    FILE_SET_SPARSE_BUFFER sparse{TRUE};
    DWORD ret = 0;
    (void)DeviceIoControl(h, FSCTL_SET_SPARSE, &sparse, sizeof sparse, nullptr, 0, &ret, nullptr);
}

// ftruncate's Windows meaning. SetFileInformationByHandle sets the end of file in one call and
// leaves the file pointer alone; the CRT's _chsize_s instead WRITES the zeros a growth adds, 4 KiB
// at a time through the pointer, so kld's 15 GB create_sized (core/kld.cpp) wrote 15 GB of zeros
// before its first row. The growth is a hole (grow_by_a_hole): kld's first row, near the end of its
// file, would otherwise still write every zero before it.
int set_end_of_file(HANDLE h, long long len) {
    if (len < 0) {
        errno = EINVAL;
        return -1;
    }
    grow_by_a_hole(h, len);
    FILE_END_OF_FILE_INFO eof{};
    eof.EndOfFile.QuadPart = len;
    if (!SetFileInformationByHandle(h, FileEndOfFileInfo, &eof, sizeof eof)) {
        errno = brl_errno_from_win32(GetLastError());
        return -1;
    }
    return 0;
}

// POSITIONAL I/O ON A CRT DESCRIPTOR. The CRT opens every file SYNCHRONOUSLY, and on a synchronous
// handle the I/O manager runs one request at a time per file object and leaves its file pointer at
// the end of each, offset or not. POSIX pread/pwrite do neither: table_copy runs sixteen readers on
// one container descriptor (core/format/table_copy.cpp), and the container writer mixes a
// sequential write() of the spill with pwrite()s of the header on descriptors sharing a pointer.
//
// So pread reads through a handle of its own: ReOpenFile on the descriptor's handle with
// FILE_FLAG_OVERLAPPED, for that one call. It is the same file -- a reopen of that handle, not of a
// path that may since name another -- its file object has no position to move and no lock to queue
// on, and it is closed before pread returns, so no handle outlives the call: close(), unlink() and
// rename() after it see exactly the handles the caller made. NOT CACHED PER DESCRIPTOR, because
// the CRT's close() cannot be hooked -- a macro named `close` would rename RadFile::close and every
// other member of that name -- so a cache would hold the file open after the caller closed it and
// could not tell a reused descriptor from the old one. The reopen costs an open and a close, 25-50
// us through the volume's filter drivers on the machine this was written on: five times a 4 KiB
// read from the cache, and nothing against what table_copy moves per call -- sixteen readers over
// a cached GiB went from 4-5 GB/s to 7.5 with it.
//
// pwrite stays on the descriptor's own handle, with its position put back after the call: a reopen
// is granted whatever the file's ACL allows, so a pwrite through one would succeed on a read-only
// descriptor. A handle that cannot be reopened (a pipe, a device) is read that way too.
//
// The save, the write and the restore are ONE STEP PER FILE, under a lock chosen by the file's
// identity (volume serial and file index, which every descriptor of the file shares, dups
// included). Interleaved across threads, one would restore a position another's write had already
// moved and leave the pointer at the end of some write: rad-convert pwrites planes from every
// thread through a dup of its output and then writes the header sequentially from where the
// pointer is, so the header landed in the middle of the weights and offset 0 stayed zeros. The
// lock costs nothing a synchronous file object does not already cost: the I/O manager runs its
// requests one at a time.
std::mutex& io_lock_for(HANDLE h) {
    static std::mutex locks[64];
    uint64_t key = (uint64_t)(uintptr_t)h;
    BY_HANDLE_FILE_INFORMATION fi;
    if (GetFileInformationByHandle(h, &fi))
        key = ((uint64_t)fi.dwVolumeSerialNumber << 40) ^
              (((uint64_t)fi.nFileIndexHigh << 32) | fi.nFileIndexLow);
    key ^= key >> 31;
    key *= 0x9e3779b97f4a7c15ull;
    key ^= key >> 29;
    return locks[key % 64];
}

template <class Io>
ssize_t on_crt_handle(HANDLE h, long long off, Io io) {
    std::lock_guard<std::mutex> lk(io_lock_for(h));
    LARGE_INTEGER zero{}, cur{};
    const bool have_pos = SetFilePointerEx(h, zero, &cur, FILE_CURRENT) != 0;
    OVERLAPPED ov{};
    at_offset(&ov, off);
    DWORD moved = 0;
    const BOOL ok = io(h, &ov, &moved);
    const DWORD e = ok ? 0 : GetLastError();
    if (have_pos) SetFilePointerEx(h, cur, nullptr, FILE_BEGIN);
    if (!ok) {
        if (e == ERROR_HANDLE_EOF) return 0;
        errno = brl_errno_from_win32(e);
        return -1;
    }
    return (ssize_t)moved;
}

}   // namespace

extern "C" {

int brl_errno_from_win32(unsigned long err) {
    switch (err) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
    case ERROR_INVALID_DRIVE:
    case ERROR_MOD_NOT_FOUND:       return ENOENT;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION:
    case ERROR_LOCK_VIOLATION:
    case ERROR_USER_MAPPED_FILE:    return EACCES;
    case ERROR_FILE_EXISTS:
    case ERROR_ALREADY_EXISTS:      return EEXIST;
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:
    case ERROR_COMMITMENT_LIMIT:
    case ERROR_NO_SYSTEM_RESOURCES: return ENOMEM;
    case ERROR_DISK_FULL:
    case ERROR_HANDLE_DISK_FULL:    return ENOSPC;
    case ERROR_INVALID_HANDLE:      return EBADF;
    case ERROR_INVALID_PARAMETER:   return EINVAL;
    case ERROR_NOT_SUPPORTED:       return ENOTSUP;
    case ERROR_DIR_NOT_EMPTY:       return ENOTEMPTY;
    case ERROR_FILENAME_EXCED_RANGE:
    case ERROR_BUFFER_OVERFLOW:     return ENAMETOOLONG;
    case ERROR_HANDLE_EOF:          return 0;
    default:                        return EIO;
    }
}

long brl_sysconf(int name) {
    switch (name) {
    case _SC_PAGESIZE: {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (long)si.dwPageSize;
    }
    case _SC_PHYS_PAGES:
    case _SC_AVPHYS_PAGES: {
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof ms;
        if (!GlobalMemoryStatusEx(&ms)) return -1;
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        const unsigned long long b = name == _SC_PHYS_PAGES ? ms.ullTotalPhys : ms.ullAvailPhys;
        return (long)(b / si.dwPageSize);
    }
    case _SC_NPROCESSORS_ONLN:
    case _SC_NPROCESSORS_CONF:
        return (long)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    default:
        errno = EINVAL;
        return -1;
    }
}

// A SLEEP THAT SLEEPS AS LONG AS IT WAS ASKED. Sleep() and the default waitable timer round up to
// the system timer tick, 15.6 ms unless something raised the resolution, so a 20 us backoff in a
// polling loop becomes a 15 ms stall. A high-resolution waitable timer (Windows 10 1803+) honours
// sub-millisecond intervals without changing the process-wide timer resolution.
int brl_usleep(unsigned long long usec) {
    if (usec == 0) {
        SwitchToThread();
        return 0;
    }
    static thread_local HANDLE t = nullptr;
    if (!t) {
        t = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                   TIMER_ALL_ACCESS);
        if (!t) t = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
    }
    if (!t) {
        Sleep((DWORD)((usec + 999) / 1000));
        return 0;
    }
    LARGE_INTEGER due;
    due.QuadPart = -(LONGLONG)(usec * 10);   // relative, in 100 ns units
    if (!SetWaitableTimer(t, &due, 0, nullptr, nullptr, FALSE)) {
        Sleep((DWORD)((usec + 999) / 1000));
        return 0;
    }
    WaitForSingleObject(t, INFINITE);
    return 0;
}

int brl_fsync(int fd) {
    brl::QuietCrt quiet;
    return _commit(fd);
}

// A negative offset is refused before it reaches Windows, which reads 0xFFFFFFFF'FFFFFFFF in an
// OVERLAPPED as "at the end of the file": a pwrite there would append.
ssize_t pread(int fd, void* buf, size_t n, long long off) {
    HANDLE h = brl::osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    if (off < 0) {
        errno = EINVAL;
        return -1;
    }
    if (n == 0) return 0;
    const DWORD want = (DWORD)(n > kIoPiece ? kIoPiece : n);
    HANDLE r = ReOpenFile(h, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                          FILE_FLAG_OVERLAPPED);
    if (r == INVALID_HANDLE_VALUE)
        return on_crt_handle(h, off, [&](HANDLE x, OVERLAPPED* ov, DWORD* got) {
            return ReadFile(x, buf, want, got, ov);
        });
    OVERLAPPED ov{};
    at_offset(&ov, off);
    DWORD got = 0;
    BOOL ok = ReadFile(r, buf, want, nullptr, &ov);
    DWORD e = ok ? 0 : GetLastError();
    if (ok || e == ERROR_IO_PENDING) {   // the only request on this handle, so it is waited on
        ok = GetOverlappedResult(r, &ov, &got, TRUE);
        e = ok ? 0 : GetLastError();
    }
    CloseHandle(r);
    if (!ok) {
        if (e == ERROR_HANDLE_EOF) return 0;
        errno = brl_errno_from_win32(e);
        return -1;
    }
    return (ssize_t)got;
}

// A pwrite past the end leaves a hole (grow_by_a_hole), decided under the file's lock so no other
// pwrite moves the end between the look and the write. rad-convert's sealed writer
// (core/format/radfile.cpp) never sizes its output: its threads pwrite planes at their final
// offsets as each weight is done, and a late layer's small weight lands near the end within
// seconds. Into a plain file, that one write zero-fills everything before it, inside one
// WriteFile, with every other thread waiting on the lock: a 473 GiB conversion's output stood
// 364 GB long seconds after its start, and that run hard-locked the machine twice.
ssize_t pwrite(int fd, const void* buf, size_t n, long long off) {
    HANDLE h = brl::osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    if (off < 0) {
        errno = EINVAL;
        return -1;
    }
    const DWORD want = (DWORD)(n > kIoPiece ? kIoPiece : n);
    return on_crt_handle(h, off, [&](HANDLE x, OVERLAPPED* ov, DWORD* put) {
        grow_by_a_hole(x, off);
        return WriteFile(x, buf, want, put, ov);
    });
}

ssize_t brl_read(int fd, void* buf, size_t n) {
    brl::QuietCrt quiet;
    return (ssize_t)_read(fd, buf, (unsigned)(n > kIoPiece ? kIoPiece : n));
}

ssize_t brl_write(int fd, const void* buf, size_t n) {
    brl::QuietCrt quiet;
    return (ssize_t)_write(fd, buf, (unsigned)(n > kIoPiece ? kIoPiece : n));
}

int ftruncate(int fd, long long len) {
    HANDLE h = brl::osfhandle(fd);
    return h == INVALID_HANDLE_VALUE ? -1 : set_end_of_file(h, len);
}

long long brl_lseek64(int fd, long long off, int whence) {
    brl::QuietCrt quiet;
    return _lseeki64(fd, off, whence);
}

ssize_t readlink(const char* path, char* buf, size_t n) {
    if (!path || strcmp(path, "/proc/self/exe") != 0) {
        errno = EINVAL;   // not a symbolic link: the only answer POSIX has for that
        return -1;
    }
    wchar_t w[32768];
    const DWORD len = GetModuleFileNameW(nullptr, w, (DWORD)(sizeof w / sizeof w[0]));
    if (!len) {
        errno = EIO;
        return -1;
    }
    const int m = WideCharToMultiByte(CP_UTF8, 0, w, (int)len, buf, (int)n, nullptr, nullptr);
    if (m <= 0) {
        errno = ENAMETOOLONG;
        return -1;
    }
    return (ssize_t)m;   // like POSIX: not NUL-terminated
}

int brl_sched_yield(void) {
    SwitchToThread();
    return 0;
}

int getrlimit(int resource, struct rlimit* rl) {
    if (!rl) {
        errno = EINVAL;
        return -1;
    }
    rl->rlim_cur = rl->rlim_max = RLIM_INFINITY;
    (void)resource;
    return 0;
}

int brl_ioctl_winsize(int fd, unsigned long req, struct winsize* ws) {
    if (req != TIOCGWINSZ || !ws) {
        errno = ENOTTY;
        return -1;
    }
    HANDLE h = brl::osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    CONSOLE_SCREEN_BUFFER_INFO ci;
    if (!GetConsoleScreenBufferInfo(h, &ci)) {
        errno = ENOTTY;
        return -1;
    }
    ws->ws_col = (unsigned short)(ci.srWindow.Right - ci.srWindow.Left + 1);
    ws->ws_row = (unsigned short)(ci.srWindow.Bottom - ci.srWindow.Top + 1);
    ws->ws_xpixel = ws->ws_ypixel = 0;
    return 0;
}

int brl_gettimeofday(struct timeval* tv, void* /*tz*/) {
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    ULARGE_INTEGER u;
    u.LowPart = ft.dwLowDateTime;
    u.HighPart = ft.dwHighDateTime;
    const unsigned long long us = (u.QuadPart - 116444736000000000ULL) / 10;   // since 1970
    tv->tv_sec = (long)(us / 1000000);
    tv->tv_usec = (long)(us % 1000000);
    return 0;
}

// ------------------------------------------------------------------------- the prelude's functions
// setenv(name, "", 1) UNSETS the variable rather than setting it empty: the UCRT keeps no empty
// variables (_putenv_s with "" is its way to remove one), so getenv() answers NULL afterwards where
// POSIX answers "". Every reader in the tree takes an empty variable for an absent one (config_test
// holds RADIANCE_KERNELS to it), so here the two mean the same; refusing with EINVAL instead would
// turn that way of clearing a variable into a failure.
int setenv(const char* name, const char* value, int overwrite) {
    if (!name || !*name || strchr(name, '=')) {
        errno = EINVAL;
        return -1;
    }
    if (!overwrite && getenv(name)) return 0;
    return _putenv_s(name, value ? value : "") == 0 ? 0 : -1;
}

int unsetenv(const char* name) {
    if (!name || !*name || strchr(name, '=')) {
        errno = EINVAL;
        return -1;
    }
    return _putenv_s(name, "") == 0 ? 0 : -1;
}

struct tm* gmtime_r(const time_t* t, struct tm* out) { return gmtime_s(out, t) == 0 ? out : nullptr; }
struct tm* localtime_r(const time_t* t, struct tm* out) { return localtime_s(out, t) == 0 ? out : nullptr; }

char* strndup(const char* s, size_t n) {
    const size_t len = strnlen(s, n);
    char* p = (char*)malloc(len + 1);
    if (!p) return nullptr;
    memcpy(p, s, len);
    p[len] = 0;
    return p;
}

char* strtok_r(char* s, const char* delim, char** save) { return strtok_s(s, delim, save); }

// A directory junction at `link` naming `target`: a mount-point reparse point, which needs no
// privilege where a symbolic link needs Developer Mode or an elevated token.
static bool make_junction(const char* link, const char* target) {
    char full[MAX_PATH * 4];
    if (!GetFullPathNameA(target, sizeof full, full, nullptr)) return false;
    wchar_t wfull[MAX_PATH * 4];
    const int nw = MultiByteToWideChar(CP_ACP, 0, full, -1, wfull, MAX_PATH * 4);
    if (nw <= 1) return false;
    const std::wstring print(wfull), sub = L"\\??\\" + print;
    if (!CreateDirectoryA(link, nullptr)) return false;
    HANDLE h = CreateFileA(link, GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                           FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) { RemoveDirectoryA(link); return false; }
    // REPARSE_DATA_BUFFER's mount-point form: the tag, the data length, a reserved word, then the
    // substitute name and the print name, each NUL-terminated, their offsets and lengths in bytes.
    const USHORT sb = (USHORT)(sub.size() * sizeof(wchar_t)), pb = (USHORT)(print.size() * sizeof(wchar_t));
    std::string buf(16 + sb + 2 + pb + 2, '\0');
    char* p = &buf[0];
    const DWORD tag = IO_REPARSE_TAG_MOUNT_POINT;
    const USHORT data_len = (USHORT)(buf.size() - 8), sub_off = 0, print_off = (USHORT)(sb + 2);
    memcpy(p, &tag, 4);
    memcpy(p + 4, &data_len, 2);
    memcpy(p + 8, &sub_off, 2);
    memcpy(p + 10, &sb, 2);
    memcpy(p + 12, &print_off, 2);
    memcpy(p + 14, &pb, 2);
    memcpy(p + 16, sub.data(), sb);
    memcpy(p + 16 + sb + 2, print.data(), pb);
    DWORD got = 0;
    const BOOL ok = DeviceIoControl(h, FSCTL_SET_REPARSE_POINT, p, (DWORD)buf.size(), nullptr, 0,
                                    &got, nullptr);
    CloseHandle(h);
    if (!ok) RemoveDirectoryA(link);
    return ok != 0;
}

int symlink(const char* target, const char* linkpath) {
    if (!target || !linkpath) { errno = EINVAL; return -1; }
    const DWORD attr = GetFileAttributesA(target);
    if (attr == INVALID_FILE_ATTRIBUTES) { errno = ENOENT; return -1; }
    const bool dir = (attr & FILE_ATTRIBUTE_DIRECTORY) != 0;
    if (CreateSymbolicLinkA(linkpath, target, (dir ? SYMBOLIC_LINK_FLAG_DIRECTORY : 0) |
                                                  SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE))
        return 0;
    if (dir ? make_junction(linkpath, target) : CreateHardLinkA(linkpath, target, nullptr) != 0) return 0;
    errno = GetLastError() == ERROR_ALREADY_EXISTS ? EEXIST : EPERM;
    return -1;
}

// POSIX getline: the line with its '\n', NUL-terminated, in a buffer grown with realloc, as glibc's
// is, so a caller frees it with free(); -1 at end of file with nothing read.
long long getline(char** lineptr, size_t* n, FILE* f) {
    if (!lineptr || !n || !f) { errno = EINVAL; return -1; }
    size_t len = 0;
    int c;
    while ((c = getc(f)) != EOF) {
        if (!*lineptr || len + 2 > *n) {
            size_t cap = *n ? *n * 2 : 128;
            while (cap < len + 2) cap *= 2;
            char* p = (char*)realloc(*lineptr, cap);
            if (!p) { errno = ENOMEM; return -1; }
            *lineptr = p;
            *n = cap;
        }
        (*lineptr)[len++] = (char)c;
        if (c == '\n') break;
    }
    if (len == 0) return -1;
    (*lineptr)[len] = 0;
    return (long long)len;
}

static void ensure_tmp_root(const char* tmpl) {
    if (tmpl && (tmpl[0] == '/' || tmpl[0] == '\\') && _strnicmp(tmpl + 1, "tmp", 3) == 0 &&
        (tmpl[4] == '/' || tmpl[4] == '\\'))
        CreateDirectoryA("\\tmp", nullptr);
}

// The XXXXXX tail, filled from a counter and the process id until a name is free.
static bool fill_template(char* tmpl, unsigned attempt) {
    const size_t n = strlen(tmpl);
    if (n < 6 || strcmp(tmpl + n - 6, "XXXXXX") != 0) return false;
    static const char abc[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    unsigned long long v = ((unsigned long long)GetCurrentProcessId() << 20) ^ GetTickCount64() ^
                           ((unsigned long long)attempt * 2654435761u);
    for (size_t i = n - 6; i < n; ++i) {
        tmpl[i] = abc[v % 36];
        v /= 36;
    }
    return true;
}

char* mkdtemp(char* tmpl) {
    ensure_tmp_root(tmpl);
    const size_t n = tmpl ? strlen(tmpl) : 0;
    for (unsigned a = 0; a < 1000; ++a) {
        if (!fill_template(tmpl, a)) {
            errno = EINVAL;
            return nullptr;
        }
        if (CreateDirectoryA(tmpl, nullptr)) return tmpl;
        if (GetLastError() != ERROR_ALREADY_EXISTS) {
            errno = brl_errno_from_win32(GetLastError());
            return nullptr;
        }
        memcpy(tmpl + n - 6, "XXXXXX", 6);
    }
    errno = EEXIST;
    return nullptr;
}

int mkstemp(char* tmpl) {
    ensure_tmp_root(tmpl);
    const size_t n = tmpl ? strlen(tmpl) : 0;
    for (unsigned a = 0; a < 1000; ++a) {
        if (!fill_template(tmpl, a)) {
            errno = EINVAL;
            return -1;
        }
        const int fd = _open(tmpl, _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY, _S_IREAD | _S_IWRITE);
        if (fd >= 0) return fd;
        if (errno != EEXIST) return -1;
        memcpy(tmpl + n - 6, "XXXXXX", 6);
    }
    errno = EEXIST;
    return -1;
}

int truncate(const char* path, long long len) {
    const std::wstring w = widen(path);
    HANDLE h = w.empty() ? INVALID_HANDLE_VALUE
                         : CreateFileW(w.c_str(), GENERIC_WRITE,
                                       FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                       nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        errno = w.empty() ? ENOENT : brl_errno_from_win32(GetLastError());
        return -1;
    }
    const int rc = set_end_of_file(h, len);
    CloseHandle(h);
    return rc;
}

}   // extern "C"

// ------------------------------------------------------------------------------- thread naming
pthread_t pthread_self(void) { return (pthread_t)GetCurrentThread(); }

int pthread_setname_np(pthread_t thread, const char* name) {
    if (!name) return EINVAL;
    const HRESULT hr = SetThreadDescription((HANDLE)thread, widen(name).c_str());
    return SUCCEEDED(hr) ? 0 : EINVAL;
}

// ------------------------------------------------------------------------------- free space
int statvfs(const char* path, struct statvfs* out) {
    if (!path || !out) { errno = EINVAL; return -1; }
    ULARGE_INTEGER avail{}, total{}, free_all{};
    if (!GetDiskFreeSpaceExW(widen(path).c_str(), &avail, &total, &free_all)) {
        errno = brl_errno_from_win32(GetLastError());
        return -1;
    }
    *out = {};
    out->f_bsize = out->f_frsize = 1;
    out->f_blocks = total.QuadPart;
    out->f_bfree = free_all.QuadPart;
    out->f_bavail = avail.QuadPart;
    out->f_namemax = 255;
    return 0;
}