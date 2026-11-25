// Brilliance: core/os/os.h on Win32. See the header for what belongs here and the conventions.

#if defined(_WIN32)

#include "os/os.h"

#include <windows.h>
#include <winioctl.h>
#include <psapi.h>
#include <sddl.h>
#include <shlobj.h>
#include <knownfolders.h>
#include <restartmanager.h>
#include <io.h>
#include <fcntl.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

extern "C" int brl_errno_from_win32(unsigned long err);   // win/compat

namespace rad::os {

namespace {

int fail(DWORD err = GetLastError()) {
    errno = brl_errno_from_win32(err);
    if (errno == 0) errno = EIO;
    return -1;
}

// The largest single ReadFile/WriteFile: the length is a DWORD, and keeping a piece under 1 GiB
// also keeps one call from monopolising the volume queue.
constexpr int64_t kIoPiece = int64_t(1) << 30;

// A handle a thread keeps for its life and closes when it ends.
struct ThreadHandle {
    HANDLE h;
    explicit ThreadHandle(HANDLE x) : h(x) {}
    ~ThreadHandle() {
        if (h) CloseHandle(h);
    }
};

// One event per thread for overlapped I/O. Handles are opened FILE_FLAG_OVERLAPPED so that
// concurrent positional reads on one handle really run concurrently -- a synchronous handle
// serialises every operation on the file object, and radiance reads a container from 16 threads.
HANDLE io_event() {
    static thread_local ThreadHandle ev(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    return ev.h;
}

// An owner-only security descriptor: radiance's 0600/0700. The prefix-cache tier holds other
// clients' prompts, so "private" is a security property here, not a nicety.
struct OwnerOnlySd {
    PSECURITY_DESCRIPTOR sd = nullptr;
    SECURITY_ATTRIBUTES  sa{};
    explicit OwnerOnlySd(bool inherit) {
        const wchar_t* sddl = inherit ? L"D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)"
                                      : L"D:P(A;;FA;;;OW)(A;;FA;;;SY)";
        if (ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, SDDL_REVISION_1, &sd, nullptr)) {
            sa.nLength = sizeof sa;
            sa.lpSecurityDescriptor = sd;
            sa.bInheritHandle = FALSE;
        }
    }
    ~OwnerOnlySd() {
        if (sd) LocalFree(sd);
    }
    SECURITY_ATTRIBUTES* get() { return sd ? &sa : nullptr; }
};

}   // namespace

// ------------------------------------------------------------------------------- strings
std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w((size_t)(n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}

std::string narrow(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s((size_t)(n > 0 ? n : 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

std::string win32_message(unsigned long err) {
    if (err == 0) err = GetLastError();
    wchar_t* msg = nullptr;
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, err, 0, (LPWSTR)&msg, 0, nullptr);
    std::string m = msg ? narrow(msg) : "Win32 error " + std::to_string(err);
    if (msg) LocalFree(msg);
    while (!m.empty() && (m.back() == '\n' || m.back() == '\r' || m.back() == ' ' || m.back() == '.'))
        m.pop_back();
    return m;
}

/* os.h: this layer's lines go to the core's logger once radiance has set it, so their W lines are
 * counted (FINDINGS); before that, or in a tool that never sets it, to stderr as they always were. */
static std::atomic<LogSink> g_log_sink{ nullptr };
void set_log_sink(LogSink sink) { g_log_sink.store(sink); }

void log_os_line(int level, const char* fmt, ...) {
    char buf[2048];
    std::string big;
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int n = std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n < 0) buf[0] = '\0';
    if (n >= (int)sizeof buf) {
        big.resize((size_t)n + 1);
        std::vsnprintf(big.data(), big.size(), fmt, ap2);
        big.resize((size_t)n);
    }
    va_end(ap2);
    const char* msg = big.empty() ? buf : big.c_str();
    if (const LogSink sink = g_log_sink.load()) {
        sink(level, msg);
        return;
    }
    std::fprintf(stderr, "%c %s\n", level >= 0 && level <= 4 ? "EWIDT"[level] : '?', msg);
}

// ------------------------------------------------------------------------------- File
File& File::operator=(File&& o) noexcept {
    if (this != &o) {
        close();
        h_ = o.h_;
        flags_ = o.flags_;
        o.h_ = nullptr;
    }
    return *this;
}

File File::open(const std::string& path, unsigned flags) {
    File f;
    DWORD access = 0;
    if (flags & Read) access |= GENERIC_READ;
    if (flags & Write) access |= GENERIC_WRITE;
    if (flags & DeleteOnClose) access |= DELETE;
    if (!access) access = GENERIC_READ;

    DWORD disp = OPEN_EXISTING;
    if ((flags & Create) && (flags & Excl)) disp = CREATE_NEW;
    else if (flags & Trunc) disp = CREATE_ALWAYS;   // Trunc implies Create (os.h)
    else if (flags & Create) disp = OPEN_ALWAYS;

    DWORD attrs = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
    if (flags & Direct) attrs |= FILE_FLAG_NO_BUFFERING;
    if (flags & DeleteOnClose) attrs |= FILE_FLAG_DELETE_ON_CLOSE;
    if (flags & Random) attrs |= FILE_FLAG_RANDOM_ACCESS;
    if (flags & Sequential) attrs |= FILE_FLAG_SEQUENTIAL_SCAN;

    OwnerOnlySd sd(false);
    HANDLE h = CreateFileW(widen(path).c_str(), access,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           (flags & Private) ? sd.get() : nullptr, disp, attrs, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        fail();
        return f;
    }
    f.h_ = h;
    f.flags_ = flags;
    return f;
}

void File::close() {
    if (h_) {
        CloseHandle((HANDLE)h_);
        h_ = nullptr;
    }
}

int64_t File::read_at(void* buf, int64_t n, int64_t off) const {
    if (!h_) {
        errno = EBADF;
        return -1;
    }
    int64_t done = 0;
    while (done < n) {
        const DWORD want = (DWORD)std::min<int64_t>(n - done, kIoPiece);
        OVERLAPPED ov{};
        const uint64_t at = (uint64_t)(off + done);
        ov.Offset = (DWORD)at;
        ov.OffsetHigh = (DWORD)(at >> 32);
        ov.hEvent = io_event();
        DWORD got = 0;
        if (!ReadFile((HANDLE)h_, (char*)buf + done, want, nullptr, &ov)) {
            const DWORD e = GetLastError();
            if (e == ERROR_HANDLE_EOF) break;
            if (e != ERROR_IO_PENDING) return done ? done : fail(e);
        }
        if (!GetOverlappedResult((HANDLE)h_, &ov, &got, TRUE)) {
            const DWORD e = GetLastError();
            if (e == ERROR_HANDLE_EOF) break;
            return done ? done : fail(e);
        }
        done += got;
        if (got < want) break;   // end of file
    }
    return done;
}

int64_t File::write_at(const void* buf, int64_t n, int64_t off) const {
    if (!h_) {
        errno = EBADF;
        return -1;
    }
    int64_t done = 0;
    while (done < n) {
        const DWORD want = (DWORD)std::min<int64_t>(n - done, kIoPiece);
        OVERLAPPED ov{};
        const uint64_t at = (uint64_t)(off + done);
        ov.Offset = (DWORD)at;
        ov.OffsetHigh = (DWORD)(at >> 32);
        ov.hEvent = io_event();
        DWORD put = 0;
        if (!WriteFile((HANDLE)h_, (const char*)buf + done, want, nullptr, &ov)) {
            const DWORD e = GetLastError();
            if (e != ERROR_IO_PENDING) return done ? done : fail(e);
        }
        if (!GetOverlappedResult((HANDLE)h_, &ov, &put, TRUE)) return done ? done : fail();
        done += put;
        if (put == 0) {
            errno = EIO;
            return done ? done : -1;
        }
    }
    return done;
}

int File::read_exact(void* buf, int64_t n, int64_t off) const {
    const int64_t got = read_at(buf, n, off);
    if (got < 0) return -1;
    if (got != n) {
        errno = EIO;
        return -1;
    }
    return 0;
}

int File::write_exact(const void* buf, int64_t n, int64_t off) const {
    const int64_t put = write_at(buf, n, off);
    if (put < 0) return -1;
    if (put != n) {
        errno = EIO;
        return -1;
    }
    return 0;
}

int64_t File::size() const {
    LARGE_INTEGER s;
    if (!h_ || !GetFileSizeEx((HANDLE)h_, &s)) return fail();
    return s.QuadPart;
}

int File::set_size(int64_t bytes) const {
    FILE_END_OF_FILE_INFO eof;
    eof.EndOfFile.QuadPart = bytes;
    if (!h_ || !SetFileInformationByHandle((HANDLE)h_, FileEndOfFileInfo, &eof, sizeof eof)) return fail();
    return 0;
}

int File::sync() const {
    if (!h_ || !FlushFileBuffers((HANDLE)h_)) return fail();
    return 0;
}

int File::set_sparse() const {
    DWORD ret = 0;
    FILE_SET_SPARSE_BUFFER sb{TRUE};
    OVERLAPPED ov{};
    ov.hEvent = io_event();
    if (!DeviceIoControl((HANDLE)h_, FSCTL_SET_SPARSE, &sb, sizeof sb, nullptr, 0, &ret, &ov)) {
        if (GetLastError() != ERROR_IO_PENDING || !GetOverlappedResult((HANDLE)h_, &ov, &ret, TRUE))
            return 0;   // best effort: a file system without sparse files just stores the zeros
    }
    return 0;
}

int File::punch_hole(int64_t off, int64_t len) const {
    FILE_ZERO_DATA_INFORMATION z;
    z.FileOffset.QuadPart = off;
    z.BeyondFinalZero.QuadPart = off + len;
    DWORD ret = 0;
    OVERLAPPED ov{};
    ov.hEvent = io_event();
    if (!DeviceIoControl((HANDLE)h_, FSCTL_SET_ZERO_DATA, &z, sizeof z, nullptr, 0, &ret, &ov)) {
        if (GetLastError() != ERROR_IO_PENDING || !GetOverlappedResult((HANDLE)h_, &ov, &ret, TRUE))
            return fail();
    }
    return 0;
}

int64_t File::direct_alignment() const {
    FILE_STORAGE_INFO si{};
    if (h_ && GetFileInformationByHandleEx((HANDLE)h_, FileStorageInfo, &si, sizeof si)) {
        const int64_t a = std::max<int64_t>(si.LogicalBytesPerSector,
                                            si.FileSystemEffectivePhysicalBytesPerSectorForAtomicity);
        if (a > 0) return a;
    }
    return 4096;
}

bool File::same_file(const File& other) const {
    FILE_ID_INFO a{}, b{};
    if (!h_ || !other.h_) return false;
    if (!GetFileInformationByHandleEx((HANDLE)h_, FileIdInfo, &a, sizeof a)) return false;
    if (!GetFileInformationByHandleEx((HANDLE)other.h_, FileIdInfo, &b, sizeof b)) return false;
    return a.VolumeSerialNumber == b.VolumeSerialNumber &&
           std::memcmp(&a.FileId, &b.FileId, sizeof a.FileId) == 0;
}

File File::dup() const {
    File f;
    HANDLE d = nullptr;
    if (!h_ || !DuplicateHandle(GetCurrentProcess(), (HANDLE)h_, GetCurrentProcess(), &d, 0, FALSE,
                                DUPLICATE_SAME_ACCESS)) {
        fail();
        return f;
    }
    f.h_ = d;
    f.flags_ = flags_;
    return f;
}

// ------------------------------------------------------------------------------- MappedFile
MappedFile::MappedFile(MappedFile&& o) noexcept
    : base_(o.base_), size_(o.size_), mapped_empty_(o.mapped_empty_) {
    o.base_ = nullptr;
    o.size_ = 0;
    o.mapped_empty_ = false;
}

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
    if (this != &o) {
        unmap();
        base_ = o.base_;
        size_ = o.size_;
        mapped_empty_ = o.mapped_empty_;
        o.base_ = nullptr;
        o.size_ = 0;
        o.mapped_empty_ = false;
    }
    return *this;
}

bool MappedFile::map_readonly(const std::string& path) {
    unmap();
    HANDLE h = CreateFileW(widen(path).c_str(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        fail();
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(h, &sz)) {
        fail();
        CloseHandle(h);
        return false;
    }
    if (sz.QuadPart == 0) {
        CloseHandle(h);
        mapped_empty_ = true;
        return true;
    }
    HANDLE m = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) {
        fail();
        CloseHandle(h);
        return false;
    }
    void* v = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    const DWORD e = GetLastError();
    // The view holds its own references to the section and the file; neither handle is needed.
    CloseHandle(m);
    CloseHandle(h);
    if (!v) {
        fail(e);
        return false;
    }
    base_ = v;
    size_ = sz.QuadPart;
    return true;
}

void MappedFile::unmap() {
    if (base_) UnmapViewOfFile(base_);
    base_ = nullptr;
    size_ = 0;
    mapped_empty_ = false;
}

void prefetch(const void* addr, int64_t len) {
    if (!addr || len <= 0) return;
    WIN32_MEMORY_RANGE_ENTRY r;
    r.VirtualAddress = const_cast<void*>(addr);
    r.NumberOfBytes = (SIZE_T)len;
    (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &r, 0);
}

// ------------------------------------------------------------------------------- memory
void* aligned_alloc(size_t align, size_t bytes) {
    void* p = _aligned_malloc(bytes ? bytes : 1, align < sizeof(void*) ? sizeof(void*) : align);
    if (!p) errno = ENOMEM;
    return p;
}

void aligned_free(void* p) { _aligned_free(p); }

namespace {
bool enable_lock_privilege() {
    static std::atomic<int> state{0};   // 0 unknown, 1 granted, -1 refused
    int s = state.load();
    if (s) return s > 0;
    HANDLE tok = nullptr;
    bool ok = false;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
        TOKEN_PRIVILEGES tp{};
        tp.PrivilegeCount = 1;
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        if (LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid) &&
            AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr) &&
            GetLastError() == ERROR_SUCCESS)
            ok = true;
        CloseHandle(tok);
    }
    state.store(ok ? 1 : -1);
    return ok;
}
}   // namespace

void* anon_alloc(int64_t bytes, bool large) {
    if (large && enable_lock_privilege()) {
        const SIZE_T lp = GetLargePageMinimum();
        if (lp) {
            const SIZE_T n = (SIZE_T)((bytes + (int64_t)lp - 1) / (int64_t)lp * (int64_t)lp);
            if (void* p = VirtualAlloc(nullptr, n, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE))
                return p;
        }
    }
    void* p = VirtualAlloc(nullptr, (SIZE_T)bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!p) fail();
    return p;
}

void anon_free(void* p) {
    if (p) VirtualFree(p, 0, MEM_RELEASE);
}

void* vmem_reserve(int64_t bytes, int64_t align) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    if (align <= (int64_t)si.dwAllocationGranularity) {
        void* p = VirtualAlloc(nullptr, (SIZE_T)bytes, MEM_RESERVE, PAGE_NOACCESS);
        if (!p) fail();
        return p;
    }
    // An aligned reservation in one call (VirtualAlloc2, Windows 10 1803+), looked up at run time
    // so the SDK's NTDDI level does not decide whether this builds.
    using VA2 = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    static VA2 va2 = (VA2)(void*)GetProcAddress(GetModuleHandleW(L"kernelbase.dll"), "VirtualAlloc2");
    if (va2) {
        MEM_ADDRESS_REQUIREMENTS req{};
        req.Alignment = (SIZE_T)align;
        MEM_EXTENDED_PARAMETER par{};
        par.Type = MemExtendedParameterAddressRequirements;
        par.Pointer = &req;
        if (void* p = va2(nullptr, nullptr, (SIZE_T)bytes, MEM_RESERVE, PAGE_NOACCESS, &par, 1)) return p;
    }
    // Fallback: find an aligned hole by over-reserving, then reserve exactly inside it. Another
    // thread can take the hole in between, so it retries.
    for (int tries = 0; tries < 16; ++tries) {
        char* p = (char*)VirtualAlloc(nullptr, (SIZE_T)(bytes + align), MEM_RESERVE, PAGE_NOACCESS);
        if (!p) {
            fail();
            return nullptr;
        }
        char* a = (char*)(((uintptr_t)p + (uintptr_t)align - 1) & ~((uintptr_t)align - 1));
        VirtualFree(p, 0, MEM_RELEASE);
        if (void* q = VirtualAlloc(a, (SIZE_T)bytes, MEM_RESERVE, PAGE_NOACCESS)) return q;
    }
    errno = ENOMEM;
    return nullptr;
}

int vmem_commit(void* addr, int64_t bytes) {
    if (!VirtualAlloc(addr, (SIZE_T)bytes, MEM_COMMIT, PAGE_READWRITE)) return fail();
    return 0;
}

int vmem_decommit(void* addr, int64_t bytes) {
    if (!VirtualFree(addr, (SIZE_T)bytes, MEM_DECOMMIT)) return fail();
    return 0;
}

int vmem_protect_rw(void* addr, int64_t bytes) {
    DWORD old = 0;
    if (!VirtualProtect(addr, (SIZE_T)bytes, PAGE_READWRITE, &old)) return fail();
    return 0;
}

void vmem_release(void* base) {
    if (base) VirtualFree(base, 0, MEM_RELEASE);
}

int lock_memory(const void* addr, int64_t bytes) {
    if (VirtualLock(const_cast<void*>(addr), (SIZE_T)bytes)) return 0;
    if (GetLastError() != ERROR_WORKING_SET_QUOTA) return fail();
    // Locked pages count against the minimum working set; raise both bounds by what is asked
    // and retry once, as llama.cpp does.
    SIZE_T mn = 0, mx = 0;
    DWORD fl = 0;
    if (!GetProcessWorkingSetSizeEx(GetCurrentProcess(), &mn, &mx, &fl)) return fail();
    const SIZE_T add = (SIZE_T)bytes + (SIZE_T)(16u << 20);
    if (!SetProcessWorkingSetSizeEx(GetCurrentProcess(), mn + add, mx + add,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE))
        return fail();
    if (!VirtualLock(const_cast<void*>(addr), (SIZE_T)bytes)) return fail();
    return 0;
}

void unlock_memory(const void* addr, int64_t bytes) { VirtualUnlock(const_cast<void*>(addr), (SIZE_T)bytes); }

int64_t page_size() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}

MemInfo mem_info() {
    MemInfo m;
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) {
        m.total = (int64_t)ms.ullTotalPhys;
        m.avail = (int64_t)ms.ullAvailPhys;
        m.commit_limit = (int64_t)ms.ullTotalPageFile;
        m.commit_avail = (int64_t)ms.ullAvailPageFile;
    }
    return m;
}

int64_t process_rss() {
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof pmc)) return -1;
    return (int64_t)pmc.WorkingSetSize;
}

int64_t process_commit() {
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof pmc)) return -1;
    return (int64_t)pmc.PrivateUsage;
}

/* Brilliance (issue #1): queried without a handle, Windows answers for the job this process runs
 * in, the innermost when jobs nest. A limit is read only where its flag is set. */
JobMemoryLimit job_memory_limit() {
    JobMemoryLimit j;
    BOOL in = FALSE;
    if (!IsProcessInJob(GetCurrentProcess(), nullptr, &in) || !in) return j;
    j.in_job = true;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
    if (!QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &li, sizeof li, nullptr))
        return j;
    const DWORD f = li.BasicLimitInformation.LimitFlags;
    if (f & JOB_OBJECT_LIMIT_JOB_MEMORY) j.job = (int64_t)li.JobMemoryLimit;
    if (f & JOB_OBJECT_LIMIT_PROCESS_MEMORY) j.process = (int64_t)li.ProcessMemoryLimit;
    return j;
}

// ------------------------------------------------------------------------------- paths
std::string exe_path() {
    std::vector<wchar_t> buf(1024);
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(), (DWORD)buf.size());
        if (n == 0) return {};
        if (n < buf.size()) return narrow(std::wstring(buf.data(), n));
        buf.resize(buf.size() * 2);
    }
}

std::string exe_dir() {
    std::string p = exe_path();
    const size_t s = p.find_last_of("\\/");
    return s == std::string::npos ? std::string(".") : p.substr(0, s);
}

static std::string known_folder(const KNOWNFOLDERID& id) {
    PWSTR w = nullptr;
    std::string s;
    if (SUCCEEDED(SHGetKnownFolderPath(id, KF_FLAG_CREATE, nullptr, &w))) s = narrow(w);
    if (w) CoTaskMemFree(w);
    return s;
}

std::string local_app_data() { return known_folder(FOLDERID_LocalAppData); }
std::string program_data() { return known_folder(FOLDERID_ProgramData); }

std::string temp_dir() {
    wchar_t buf[MAX_PATH + 2];
    DWORD n = GetTempPathW(MAX_PATH + 1, buf);
    std::string s = n ? narrow(std::wstring(buf, n)) : std::string(".");
    while (s.size() > 3 && (s.back() == '\\' || s.back() == '/')) s.pop_back();
    return s;
}

std::string system_dir() {
    wchar_t buf[MAX_PATH + 1];
    const UINT n = GetSystemDirectoryW(buf, MAX_PATH + 1);
    return n && n <= MAX_PATH ? narrow(std::wstring(buf, n)) : std::string("C:\\Windows\\System32");
}

std::string default_home() {
    const std::string up = exe_dir() + "/..";   // forward slashes: GetFullPathNameW normalises
    for (const char* rel : {"/share/brilliance", "/radiance_home", "/share/radiance"}) {
        const std::string h = up + rel;
        if (is_dir(h)) {
            // Sized by asking: the executable is long-path aware, so the answer can exceed any
            // fixed buffer.
            const std::wstring wh = widen(h);
            const DWORD need = GetFullPathNameW(wh.c_str(), 0, nullptr, nullptr);
            if (!need) return h;
            std::wstring full(need, L'\0');
            const DWORD n = GetFullPathNameW(wh.c_str(), need, full.data(), nullptr);
            return n && n < need ? narrow(full.substr(0, n)) : h;
        }
    }
#ifdef RAD_DEFAULT_HOME
    return RAD_DEFAULT_HOME;
#else
    return ".";
#endif
}

std::string path_basename(const std::string& p) {
    const size_t s = p.find_last_of("\\/");
    return s == std::string::npos ? p : p.substr(s + 1);
}

bool file_size_mtime(const std::string& p, int64_t* size, int64_t* sec, int64_t* nsec) {
    WIN32_FILE_ATTRIBUTE_DATA a{};
    if (!GetFileAttributesExW(widen(p).c_str(), GetFileExInfoStandard, &a)) {
        fail();
        return false;
    }
    /* FILETIME: 100 ns units since 1601-01-01; 11644473600 s separate that from 1970-01-01. */
    const uint64_t t = ((uint64_t)a.ftLastWriteTime.dwHighDateTime << 32) | a.ftLastWriteTime.dwLowDateTime;
    const int64_t since1970 = (int64_t)t - 116444736000000000ll;
    if (size) *size = (int64_t)(((uint64_t)a.nFileSizeHigh << 32) | a.nFileSizeLow);
    if (sec) *sec = since1970 / 10000000;
    if (nsec) *nsec = (since1970 % 10000000) * 100;
    return true;
}

std::string full_path(const std::string& p) {
    const std::wstring w = widen(p);
    const DWORD n = GetFullPathNameW(w.c_str(), 0, nullptr, nullptr);
    if (n == 0) return p;
    std::wstring out((size_t)n, L'\0');
    const DWORD got = GetFullPathNameW(w.c_str(), n, out.data(), nullptr);
    if (got == 0 || got >= n) return p;
    out.resize(got);
    return narrow(out);
}

bool is_dir(const std::string& p) {
    const DWORD a = GetFileAttributesW(widen(p).c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

bool is_file(const std::string& p) {
    const DWORD a = GetFileAttributesW(widen(p).c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

int make_dirs(const std::string& p, bool owner_only) {
    if (p.empty()) {
        errno = EINVAL;
        return -1;
    }
    if (is_dir(p)) return 0;
    const size_t s = p.find_last_of("\\/");
    if (s != std::string::npos && s > 0 && !(s == 2 && p[1] == ':')) {
        const std::string parent = p.substr(0, s);
        if (!parent.empty() && !is_dir(parent) && make_dirs(parent, owner_only) < 0) return -1;
    }
    OwnerOnlySd sd(true);
    if (!CreateDirectoryW(widen(p).c_str(), owner_only ? sd.get() : nullptr)) {
        const DWORD e = GetLastError();
        if (e == ERROR_ALREADY_EXISTS && is_dir(p)) return 0;
        return fail(e);
    }
    return 0;
}

int rename_replace(const std::string& from, const std::string& to) {
    // POSIX rename semantics first (Windows 10 1709+ on NTFS): the target may be open elsewhere,
    // as it may on Linux. MoveFileExW, which refuses an open target, is the fallback.
    HANDLE h = CreateFileW(widen(from).c_str(), DELETE | SYNCHRONIZE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        const std::wstring wt = widen(to);
        std::vector<char> buf(sizeof(FILE_RENAME_INFO) + (wt.size() + 1) * sizeof(wchar_t));
        auto* ri = (FILE_RENAME_INFO*)buf.data();
        ri->Flags = 0x1 /* REPLACE_IF_EXISTS */ | 0x2 /* POSIX_SEMANTICS */;
        ri->RootDirectory = nullptr;
        ri->FileNameLength = (DWORD)(wt.size() * sizeof(wchar_t));
        std::memcpy(ri->FileName, wt.c_str(), (wt.size() + 1) * sizeof(wchar_t));
        const BOOL ok = SetFileInformationByHandle(h, (FILE_INFO_BY_HANDLE_CLASS)22 /* FileRenameInfoEx */,
                                                   ri, (DWORD)buf.size());
        CloseHandle(h);
        if (ok) return 0;
    }
    if (!MoveFileExW(widen(from).c_str(), widen(to).c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return fail();
    return 0;
}

int remove_file(const std::string& p) {
    HANDLE h = CreateFileW(widen(p).c_str(), DELETE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (h == INVALID_HANDLE_VALUE) return fail();
    // POSIX delete: the name goes now, the data when the last handle closes -- unlink(2).
    struct { DWORD Flags; } di{0x1 /* DELETE */ | 0x2 /* POSIX_SEMANTICS */};
    BOOL ok = SetFileInformationByHandle(h, (FILE_INFO_BY_HANDLE_CLASS)21 /* FileDispositionInfoEx */,
                                         &di, sizeof di);
    if (!ok) {
        FILE_DISPOSITION_INFO d{TRUE};
        ok = SetFileInformationByHandle(h, FileDispositionInfo, &d, sizeof d);
    }
    const DWORD e = GetLastError();
    CloseHandle(h);
    if (!ok) return fail(e);
    return 0;
}

std::vector<long> processes_using(const std::string& p) {
    std::vector<long> who;
    DWORD session = 0;
    WCHAR key[CCH_RM_SESSION_KEY + 1] = {};
    if (RmStartSession(&session, 0, key) != ERROR_SUCCESS) return who;
    const std::wstring w = widen(p);
    LPCWSTR files[] = {w.c_str()};
    if (RmRegisterResources(session, 1, files, 0, nullptr, 0, nullptr) == ERROR_SUCCESS) {
        UINT need = 0, have = 0;
        DWORD reasons = 0;
        std::vector<RM_PROCESS_INFO> info;
        DWORD rc = RmGetList(session, &need, &have, nullptr, &reasons);
        while (rc == ERROR_MORE_DATA) {
            info.resize(need);
            have = need;
            rc = RmGetList(session, &need, &have, info.data(), &reasons);
        }
        if (rc == ERROR_SUCCESS) {
            const DWORD me = GetCurrentProcessId();
            for (UINT i = 0; i < have; ++i)
                if (info[i].Process.dwProcessId != me) who.push_back((long)info[i].Process.dwProcessId);
        }
    }
    RmEndSession(session);
    return who;
}

// ------------------------------------------------------------------------------- shared files
namespace {
// The byte flock_fd locks: past the end of any file this port makes or maps (os.h).
constexpr uint64_t kFlockByte = uint64_t(1) << 62;

void at_flock_byte(OVERLAPPED* ov) {
    *ov = OVERLAPPED{};
    ov->Offset = (DWORD)kFlockByte;
    ov->OffsetHigh = (DWORD)(kFlockByte >> 32);
}

// Volume serial and file id of an open handle: FILE_ID_INFO where the file system has 128-bit ids
// (NTFS, ReFS), else the 64-bit index every local file system reports.
bool file_identity(HANDLE h, uint64_t* volume, unsigned char id[16]) {
    FILE_ID_INFO fi{};
    if (GetFileInformationByHandleEx(h, FileIdInfo, &fi, sizeof fi)) {
        *volume = fi.VolumeSerialNumber;
        std::memcpy(id, &fi.FileId, 16);
        return true;
    }
    BY_HANDLE_FILE_INFORMATION bi{};
    if (!GetFileInformationByHandle(h, &bi)) return false;
    *volume = bi.dwVolumeSerialNumber;
    const uint64_t index = ((uint64_t)bi.nFileIndexHigh << 32) | bi.nFileIndexLow;
    std::memset(id, 0, 16);
    std::memcpy(id, &index, sizeof index);
    return true;
}
}   // namespace

}   // namespace rad::os
namespace brl { HANDLE osfhandle(int fd); }   // win/compat: the handle, or EBADF without a crash
namespace rad::os {

int open_shared_fd(const std::string& path, bool write, bool create) {
    const std::wstring w = widen(path);
    const DWORD access = GENERIC_READ | (write ? GENERIC_WRITE : 0);
    HANDLE h = INVALID_HANDLE_VALUE;
    for (int tries = 0;; ++tries) {
        h = CreateFileW(w.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                        nullptr, create ? OPEN_ALWAYS : OPEN_EXISTING,
                        create ? FILE_ATTRIBUTE_TEMPORARY : FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) break;
        const DWORD e = GetLastError();
        // STATUS_DELETE_PENDING arrives as ERROR_ACCESS_DENIED: a file deleted without POSIX
        // semantics, at its name until the last handle on it closes (os.h). 40 x 50 ms.
        if (!create || e != ERROR_ACCESS_DENIED || tries >= 40) return fail(e);
        Sleep(50);
    }
    const int fd = _open_osfhandle((intptr_t)h, (write ? _O_RDWR : _O_RDONLY) | _O_BINARY | _O_NOINHERIT);
    if (fd < 0) {
        CloseHandle(h);
        errno = EMFILE;
        return -1;
    }
    return fd;
}

int flock_fd(int fd, int op) {
    const HANDLE h = brl::osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;   // errno EBADF
    const int kind = op & (LockSh | LockEx | LockUn);
    if (kind != LockSh && kind != LockEx && kind != LockUn) {
        errno = EINVAL;
        return -1;
    }
    // Whatever this descriptor holds goes first (ERROR_NOT_LOCKED: it held nothing). A handle's own
    // shared lock would otherwise stand in the way of its exclusive one, and a second shared lock
    // on the same byte would need a second unlock.
    OVERLAPPED ov;
    at_flock_byte(&ov);
    if (!UnlockFileEx(h, 0, 1, 0, &ov)) {
        const DWORD e = GetLastError();
        if (e != ERROR_NOT_LOCKED) return fail(e);
    }
    if (kind == LockUn) return 0;
    const DWORD flags = (kind == LockEx ? LOCKFILE_EXCLUSIVE_LOCK : 0) |
                        ((op & LockNb) ? LOCKFILE_FAIL_IMMEDIATELY : 0);
    at_flock_byte(&ov);
    if (LockFileEx(h, flags, 0, 1, 0, &ov)) return 0;
    DWORD e = GetLastError();
    if (e == ERROR_IO_PENDING) {   // a handle opened for overlapped I/O: wait as flock would
        DWORD n = 0;
        if (GetOverlappedResult(h, &ov, &n, TRUE)) return 0;
        e = GetLastError();
    }
    if (e == ERROR_LOCK_VIOLATION) {
        errno = EWOULDBLOCK;
        return -1;
    }
    return fail(e);
}

bool fd_names(int fd, const std::string& path) {
    const HANDLE h = brl::osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return false;
    const HANDLE p = CreateFileW(widen(path).c_str(), FILE_READ_ATTRIBUTES,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (p == INVALID_HANDLE_VALUE) return false;
    uint64_t va = 0, vb = 0;
    unsigned char ia[16], ib[16];
    const bool same = file_identity(h, &va, ia) && file_identity(p, &vb, ib) && va == vb &&
                      std::memcmp(ia, ib, sizeof ia) == 0;
    CloseHandle(p);
    return same;
}

namespace {
// FILETIME-style 100 ns units since 1601-01-01 as seconds and nanoseconds since 1970-01-01.
void stamp_time(int64_t t, int64_t* sec, int32_t* nsec) {
    const int64_t since1970 = t - 116444736000000000ll;
    int64_t s = since1970 / 10000000, r = since1970 % 10000000;
    if (r < 0) { r += 10000000; --s; }
    *sec = s;
    *nsec = (int32_t)(r * 100);
}

bool stamp_of(HANDLE h, FileStamp* out) {
    FILE_BASIC_INFO b{};
    FILE_STANDARD_INFO st{};
    uint64_t volume = 0;
    unsigned char id[16];
    if (!GetFileInformationByHandleEx(h, FileBasicInfo, &b, sizeof b) ||
        !GetFileInformationByHandleEx(h, FileStandardInfo, &st, sizeof st) ||
        !file_identity(h, &volume, id)) {
        fail();
        return false;
    }
    if (st.Directory || GetFileType(h) != FILE_TYPE_DISK) {
        errno = st.Directory ? EISDIR : EINVAL;
        return false;
    }
    uint64_t lo = 0, hi = 0;
    std::memcpy(&lo, id, 8);
    std::memcpy(&hi, id + 8, 8);
    out->device = volume;
    out->inode = lo ^ hi;
    out->bytes = (uint64_t)st.EndOfFile.QuadPart;
    stamp_time(b.LastWriteTime.QuadPart, &out->mtime_s, &out->mtime_ns);
    stamp_time(b.ChangeTime.QuadPart, &out->ctime_s, &out->ctime_ns);
    return true;
}
}   // namespace

bool file_stamp_fd(int fd, FileStamp* out) {
    const HANDLE h = brl::osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return false;
    return stamp_of(h, out);
}

bool file_stamp(const std::string& path, FileStamp* out) {
    const HANDLE h = CreateFileW(widen(path).c_str(), FILE_READ_ATTRIBUTES,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                 OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        fail();
        return false;
    }
    const bool ok = stamp_of(h, out);
    CloseHandle(h);
    return ok;
}

// ------------------------------------------------------------------------------- processes
std::string quote_arg(const std::string& s) {
    if (!s.empty() && s.find_first_of(" \t\n\v\"") == std::string::npos) return s;
    std::string out = "\"";
    for (size_t i = 0;; ++i) {
        size_t nbs = 0;
        while (i < s.size() && s[i] == '\\') {
            ++i;
            ++nbs;
        }
        if (i == s.size()) {
            out.append(nbs * 2, '\\');   // so the closing quote stays a quote
            break;
        }
        if (s[i] == '"') {
            out.append(nbs * 2 + 1, '\\');
            out += '"';
        } else {
            out.append(nbs, '\\');
            out += s[i];
        }
    }
    out += '"';
    return out;
}

namespace {
// The name of an environment entry, "=C:" style drive entries included (their name starts at the
// '=' and ends at the next one).
std::wstring env_name(const std::wstring& entry) {
    const size_t eq = entry.find(L'=', 1);
    return entry.substr(0, eq == std::wstring::npos ? entry.size() : eq);
}
bool env_name_eq(const std::wstring& a, const std::wstring& b) {
    return CompareStringOrdinal(a.c_str(), (int)a.size(), b.c_str(), (int)b.size(), TRUE) == CSTR_EQUAL;
}
}   // namespace

long long spawn_and_wait(const std::string& cmdline, const std::string& env_extra, long timeout_ms) {
    std::vector<wchar_t> env;
    if (!env_extra.empty()) {
        // Windows looks a name up by the FIRST entry that has it, so an override appended after
        // the inherited copy would never be seen: the inherited entry is dropped instead. The
        // block is kept in the case-insensitive order CreateProcess documents.
        std::vector<std::wstring> add;
        for (size_t start = 0; start < env_extra.size();) {
            size_t end = env_extra.find('\n', start);
            if (end == std::string::npos) end = env_extra.size();
            std::wstring line = widen(env_extra.substr(start, end - start));
            if (!line.empty() && line.back() == L'\r') line.pop_back();
            if (line.find(L'=', 1) != std::wstring::npos) {
                const std::wstring n = env_name(line);   // a later line for the same name wins
                add.erase(std::remove_if(add.begin(), add.end(),
                                         [&](const std::wstring& a) { return env_name_eq(env_name(a), n); }),
                          add.end());
                add.push_back(std::move(line));
            }
            start = end + 1;
        }
        std::vector<std::wstring> all;
        wchar_t* cur = GetEnvironmentStringsW();
        for (const wchar_t* p = cur; p && *p; p += wcslen(p) + 1) {
            std::wstring e(p);
            bool overridden = false;
            for (const std::wstring& a : add) overridden |= env_name_eq(env_name(e), env_name(a));
            if (!overridden) all.push_back(std::move(e));
        }
        if (cur) FreeEnvironmentStringsW(cur);
        for (std::wstring& a : add) all.push_back(std::move(a));
        std::stable_sort(all.begin(), all.end(), [](const std::wstring& a, const std::wstring& b) {
            const std::wstring na = env_name(a), nb = env_name(b);
            return CompareStringOrdinal(na.c_str(), (int)na.size(), nb.c_str(), (int)nb.size(), TRUE) ==
                   CSTR_LESS_THAN;
        });
        for (const std::wstring& e : all) env.insert(env.end(), e.c_str(), e.c_str() + e.size() + 1);
        env.push_back(L'\0');
    }
    // THE CHILD AND EVERYTHING IT STARTS LIVE IN A JOB THAT DIES WITH THIS PROCESS. hfget.ps1 starts
    // curl.exe; a service stopped while it downloads ends this process with TerminateProcess, and
    // without the job PowerShell and curl would go on writing the model cache as SYSTEM, racing the
    // next start's own fetch of the same files. A timeout ends the whole tree, not only its root.
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof li);
    }
    // ONLY THE STANDARD HANDLES ARE INHERITED, so a child that outlives a step cannot hold this
    // process's files open (the service log among them, which the next start rotates).
    std::vector<HANDLE> inherit;
    HANDLE std3[3] = {GetStdHandle(STD_INPUT_HANDLE), GetStdHandle(STD_OUTPUT_HANDLE), GetStdHandle(STD_ERROR_HANDLE)};
    for (HANDLE& h : std3) {
        DWORD fl = 0;
        if (!h || h == INVALID_HANDLE_VALUE || !GetHandleInformation(h, &fl) || !(fl & HANDLE_FLAG_INHERIT)) {
            h = nullptr;
            continue;
        }
        if (std::find(inherit.begin(), inherit.end(), h) == inherit.end()) inherit.push_back(h);
    }
    SIZE_T asz = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &asz);
    std::vector<unsigned char> abuf(asz);
    auto* attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(abuf.data());
    bool have_attrs = InitializeProcThreadAttributeList(attrs, 1, 0, &asz) != 0;
    if (have_attrs && !inherit.empty() &&
        !UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit.data(),
                                   inherit.size() * sizeof(HANDLE), nullptr, nullptr)) {
        DeleteProcThreadAttributeList(attrs);
        have_attrs = false;
    }

    std::wstring cmd = widen(cmdline);
    STARTUPINFOEXW si{};
    si.StartupInfo.cb = have_attrs ? sizeof si : sizeof si.StartupInfo;
    si.StartupInfo.dwFlags = inherit.empty() ? 0 : STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = std3[0];
    si.StartupInfo.hStdOutput = std3[1];
    si.StartupInfo.hStdError = std3[2];
    si.lpAttributeList = have_attrs ? attrs : nullptr;
    PROCESS_INFORMATION pi{};
    const DWORD cf = CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED | (have_attrs ? EXTENDED_STARTUPINFO_PRESENT : 0);
    const BOOL made = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, have_attrs && !inherit.empty(), cf,
                                     env.empty() ? nullptr : env.data(), nullptr, &si.StartupInfo, &pi);
    const DWORD made_err = GetLastError();
    if (have_attrs) DeleteProcThreadAttributeList(attrs);
    if (!made) {
        if (job) CloseHandle(job);
        return fail(made_err);
    }
    if (job && !AssignProcessToJobObject(job, pi.hProcess)) {
        CloseHandle(job);   // left alone rather than not started: the child still runs, unguarded
        job = nullptr;
    }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    const DWORD w = WaitForSingleObject(pi.hProcess, timeout_ms < 0 ? INFINITE : (DWORD)timeout_ms);
    if (w == WAIT_TIMEOUT) {
        if (job) TerminateJobObject(job, 0xC000013A /* STATUS_CONTROL_C_EXIT */);
        else TerminateProcess(pi.hProcess, 0xC000013A);
        WaitForSingleObject(pi.hProcess, 5000);
        CloseHandle(pi.hProcess);
        if (job) CloseHandle(job);
        return -2;
    }
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    if (job) CloseHandle(job);   // and with it anything the child left running
    return (long long)code;
}

void* spawn(const std::string& cmdline) {
    std::wstring cmd = widen(cmdline);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr,
                        nullptr, &si, &pi)) {
        fail();
        return nullptr;
    }
    CloseHandle(pi.hThread);
    return pi.hProcess;
}

namespace {
std::mutex g_console_mu;

// Whether this process has a console a child can share, attaching to the parent's when it has
// none. GetConsoleProcessList is the test, not GetConsoleWindow: a console without a window -- what a
// process started CREATE_NO_WINDOW has, as many test runners start tests -- is still a console that
// Ctrl-Break travels through, and GetConsoleWindow is null there.
bool share_console(std::string* why) {
    std::lock_guard<std::mutex> lk(g_console_mu);
    DWORD ids[1];
    if (GetConsoleProcessList(ids, 1) > 0) return true;
    // Attaching may point the standard handles at the console; the ones this process was started
    // with (the pipes a test runner reads) are put back, so its own output goes where it went.
    const DWORD which[3] = {STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE};
    HANDLE saved[3];
    for (int i = 0; i < 3; ++i) saved[i] = GetStdHandle(which[i]);
    const BOOL attached = AttachConsole(ATTACH_PARENT_PROCESS);
    const DWORD err = GetLastError();
    for (int i = 0; i < 3; ++i)
        if (attached && saved[i] && saved[i] != INVALID_HANDLE_VALUE) SetStdHandle(which[i], saved[i]);
    if (attached || GetConsoleProcessList(ids, 1) > 0) return true;
    if (why)
        *why = "this process has no console and could not attach to its parent's (" + win32_message(err) +
               "): it was started detached -- DETACHED_PROCESS, a service, a GUI launcher -- so Ctrl-Break, "
               "Windows' SIGTERM, cannot reach a child";
    return false;
}
}   // namespace

Captured::~Captured() {
    if (proc_) { kill(); wait(); }
    if (rd_) CloseHandle((HANDLE)rd_);
}

bool Captured::start(const std::vector<std::string>& argv, const std::vector<std::string>& env) {
    if (argv.empty()) { errno = EINVAL; return false; }
    // A process group of its own, on this process's console, so interrupt() reaches the child and
    // only it. Without a console to share, a console of its own without a window (no window pops
    // up): then the child cannot be interrupted, and interrupt() says why.
    no_console_.clear();
    console_ = share_console(&no_console_);
    SECURITY_ATTRIBUTES sa{ sizeof sa, nullptr, TRUE };
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 1 << 16)) { fail(); return false; }
    /* The read end stays in this process only. */
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    std::string cmd;
    for (const std::string& a : argv) cmd += (cmd.empty() ? "" : " ") + quote_arg(a);
    std::wstring block;   /* the environment: NAME=value\0 ... \0 */
    for (const std::string& e : env) { block += widen(e); block.push_back(L'\0'); }
    block.push_back(L'\0');
    // ONLY THE PIPE'S WRITE END AND STDIN ARE INHERITED. bInheritHandles alone hands the child every
    // inheritable handle this process holds -- a socket() is inheritable by default -- and a
    // test's open connection, or another child's pipe being set up on another thread, would stay
    // open in the server: the FD_CLOEXEC of the Linux code, made explicit.
    HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    DWORD fl = 0;
    if (!in || in == INVALID_HANDLE_VALUE || !GetHandleInformation(in, &fl) || !(fl & HANDLE_FLAG_INHERIT))
        in = nullptr;
    HANDLE inherit[2] = {wr, in};
    SIZE_T asz = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &asz);
    std::vector<unsigned char> abuf(asz);
    auto* attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(abuf.data());
    if (!InitializeProcThreadAttributeList(attrs, 1, 0, &asz)) {
        const DWORD e = GetLastError();
        CloseHandle(rd);
        CloseHandle(wr);
        fail(e);
        return false;
    }
    if (!UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit,
                                   (in ? 2 : 1) * sizeof(HANDLE), nullptr, nullptr)) {
        const DWORD e = GetLastError();
        DeleteProcThreadAttributeList(attrs);
        CloseHandle(rd);
        CloseHandle(wr);
        fail(e);
        return false;
    }
    STARTUPINFOEXW si{};
    si.StartupInfo.cb = sizeof si;
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = in;
    si.StartupInfo.hStdOutput = wr;
    si.StartupInfo.hStdError = wr;
    si.lpAttributeList = attrs;
    PROCESS_INFORMATION pi{};
    std::wstring wcmd = widen(cmd);
    const DWORD cf = CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT | CREATE_NEW_PROCESS_GROUP |
                     (console_ ? 0 : CREATE_NO_WINDOW);
    const BOOL ok = CreateProcessW(widen(argv[0]).c_str(), wcmd.data(), nullptr, nullptr, TRUE, cf,
                                   block.data(), nullptr, &si.StartupInfo, &pi);
    const DWORD err = GetLastError();
    DeleteProcThreadAttributeList(attrs);
    CloseHandle(wr);   /* only the child holds the write end now, so its exit ends the reads */
    if (!ok) { CloseHandle(rd); fail(err); return false; }
    CloseHandle(pi.hThread);
    proc_ = pi.hProcess;
    pid_ = (long)pi.dwProcessId;
    rd_ = rd;
    return true;
}

bool Captured::read_all(std::string* out, int64_t timeout_ms) {
    if (!rd_) return true;
    const ULONGLONG end = GetTickCount64() + (ULONGLONG)(timeout_ms > 0 ? timeout_ms : 0);
    char buf[4096];
    for (;;) {
        if (timeout_ms > 0) {
            /* An anonymous pipe has no overlapped reads, so the wait is a poll of what is buffered. */
            DWORD avail = 0;
            if (!PeekNamedPipe((HANDLE)rd_, nullptr, 0, nullptr, &avail, nullptr)) return true;   /* closed */
            if (avail == 0) {
                if (GetTickCount64() >= end) return false;
                Sleep(5);
                continue;
            }
        }
        DWORD n = 0;
        if (!ReadFile((HANDLE)rd_, buf, sizeof buf, &n, nullptr)) return true;
        /* TRUE with nothing read is the child's zero-byte write (ReadFile's documentation), not
         * the end of its output: that is ERROR_BROKEN_PIPE. */
        out->append(buf, n);
    }
}

int64_t Captured::read_some(void* buf, size_t n) {
    if (!rd_) { errno = EBADF; return -1; }
    const DWORD want = (DWORD)std::min<size_t>(n, size_t(1) << 30);
    for (;;) {
        DWORD got = 0;
        if (ReadFile((HANDLE)rd_, buf, want, &got, nullptr)) {
            if (got > 0 || want == 0) return (int64_t)got;
            continue;   /* a zero-byte write by the child: nothing yet, and not the end */
        }
        const DWORD err = GetLastError();
        if (err == ERROR_BROKEN_PIPE) return 0;   /* every write end is closed: the end of the output */
        return fail(err);
    }
}

void Captured::close_read() {
    if (rd_) CloseHandle((HANDLE)rd_);
    rd_ = nullptr;
}

bool Captured::interrupt(std::string* why) {
    if (!proc_) {
        errno = ESRCH;
        if (why) *why = "no child is running";
        return false;
    }
    if (WaitForSingleObject((HANDLE)proc_, 0) == WAIT_OBJECT_0) return true;   /* already gone */
    if (!console_) {
        errno = ENOTSUP;
        if (why) *why = no_console_;
        return false;
    }
    // CTRL_BREAK_EVENT, not CTRL_C_EVENT: a new process group starts with Ctrl-C ignored. The
    // engine's handler (on_terminate) takes Ctrl-Break as SIGTERM.
    if (GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, (DWORD)pid_)) return true;
    const DWORD err = GetLastError();
    if (WaitForSingleObject((HANDLE)proc_, 0) == WAIT_OBJECT_0) return true;   /* it exited meanwhile */
    if (why) *why = "GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, " + std::to_string(pid_) + "): " + win32_message(err);
    fail(err);
    return false;
}

bool Captured::can_interrupt(std::string* why) {
    return share_console(why);
}

void Captured::kill() {
    if (proc_) TerminateProcess((HANDLE)proc_, 1);
}

int Captured::wait() {
    if (!proc_) return -1;
    WaitForSingleObject((HANDLE)proc_, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess((HANDLE)proc_, &code);
    CloseHandle((HANDLE)proc_);
    proc_ = nullptr;
    pid_ = -1;
    if (rd_) { CloseHandle((HANDLE)rd_); rd_ = nullptr; }
    return (int)code;
}

bool process_alive(long long pid) {
    if (pid <= 0 || pid > 0xFFFFFFFFll) return false;
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    if (!h) return GetLastError() == ERROR_ACCESS_DENIED;   /* exists, but not ours to open */
    DWORD code = 0;
    const bool running = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
    CloseHandle(h);
    return running;
}

void terminate_and_wait(void* proc) {
    if (!proc) return;
    TerminateProcess((HANDLE)proc, 1);
    WaitForSingleObject((HANDLE)proc, 10000);
    CloseHandle((HANDLE)proc);
}

namespace {
std::atomic<void (*)(int)> g_term_fn{nullptr};
std::atomic<int>           g_term_seen{0};
std::atomic<int>           g_term_pending{0};   // a request_terminate() before on_terminate()
HANDLE                     g_term_done = nullptr;

// The registered function, once per process whichever way the termination came.
void deliver_term(void (*fn)(int), int sig) {
    if (g_term_seen.fetch_add(1) == 0) fn(sig);
}

void signal_term_done() {
    if (g_term_done) SetEvent(g_term_done);
}

BOOL WINAPI console_handler(DWORD type) {
    if (g_term_seen.fetch_add(1) > 0) return FALSE;   // second event: the default handler ends us
    const int sig = type == CTRL_C_EVENT ? SIGINT : SIGTERM;
    if (auto fn = g_term_fn.load()) fn(sig);
    // Closing the console ends the process as soon as this handler returns; give the engine the
    // few seconds Windows allows to finish its step and release the cards. (Log-off and shut-down
    // are never delivered here: the engine loads gdi32 for the GPU telemetry, and Windows sends
    // those to a console process with gdi32 or user32 loaded only as window messages. A server that
    // must drain at shut-down runs as the service, whose PRESHUTDOWN control is.)
    if (type == CTRL_CLOSE_EVENT || type == CTRL_LOGOFF_EVENT || type == CTRL_SHUTDOWN_EVENT)
        WaitForSingleObject(g_term_done, 4500);
    return TRUE;
}
}   // namespace

void on_terminate(void (*fn)(int sig)) {
    g_term_fn.store(fn);
    if (!g_term_done) {
        g_term_done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        std::atexit(signal_term_done);
    }
    SetConsoleCtrlHandler(console_handler, TRUE);
    // A service stopped while the engine was still starting asked before there was anyone to
    // tell: it is told now, and stops before it loads anything (request_terminate).
    if (const int sig = g_term_pending.exchange(0)) deliver_term(fn, sig);
}

bool request_terminate(int sig) {
    // Kept first, then the function looked for; on_terminate stores the function, then takes what
    // was kept. Whichever of the two runs second finds the other's write, and the exchange gives
    // the request to exactly one of them.
    g_term_pending.store(sig);
    auto fn = g_term_fn.load();
    if (!fn) return false;
    if (const int s = g_term_pending.exchange(0)) deliver_term(fn, s);
    return true;
}

std::string reload_event_name(long long pid, bool global) {
    return std::string(global ? "Global\\" : "Local\\") + "Brilliance-Reload-" + std::to_string(pid);
}

void on_reload(void (*fn)(int sig)) {
    static std::atomic<bool> started{false};
    if (!fn || started.exchange(true)) return;
    // Global first, so a service in session 0 can be reached from an administrator's console;
    // creating a Global name needs SeCreateGlobalPrivilege, which a service has and a user's console
    // process does not, so a console server falls back to its session's Local name. Auto-reset: one
    // Set is one reload, as one SIGHUP is.
    const long long pid = (long long)GetCurrentProcessId();
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, widen(reload_event_name(pid, true)).c_str());
    if (!ev) ev = CreateEventW(nullptr, FALSE, FALSE, widen(reload_event_name(pid, false)).c_str());
    if (!ev) return;
    std::thread([ev, fn] {
        for (;;) {
            if (WaitForSingleObject(ev, INFINITE) != WAIT_OBJECT_0) return;
            fn(1 /* SIGHUP's number on Linux */);
        }
    }).detach();
}

double busy_cpus(int window_ms) {
    auto ticks = [](FILETIME f) { return ((uint64_t)f.dwHighDateTime << 32) | f.dwLowDateTime; };
    FILETIME i0, k0, u0, i1, k1, u1;
    if (!GetSystemTimes(&i0, &k0, &u0)) return -1;
    Sleep((DWORD)std::max(window_ms, 1));
    if (!GetSystemTimes(&i1, &k1, &u1)) return -1;
    // Kernel time includes idle time, as GetSystemTimes reports it.
    const uint64_t idle = ticks(i1) - ticks(i0);
    const uint64_t all = (ticks(k1) - ticks(k0)) + (ticks(u1) - ticks(u0));
    if (all == 0) return -1;
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    return (double)(all - std::min(idle, all)) / (double)all * (double)si.dwNumberOfProcessors;
}

namespace {
/* The L3 domains of this machine, lowest logical processor first: on a 9950X3D the two CCDs, the
 * V-Cache one with the larger L3. Group 0 only -- 64 logical processors, more than any desktop part
 * this targets has. */
struct L3Domain { uint64_t mask = 0; uint64_t bytes = 0; };
std::vector<L3Domain> l3_domains() {
    std::vector<L3Domain> out;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
    std::vector<unsigned char> buf(len);
    if (!len || !GetLogicalProcessorInformationEx(
                    RelationCache, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len))
        return out;
    for (DWORD off = 0; off < len;) {
        auto* e = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
        if (e->Relationship == RelationCache && e->Cache.Level == 3 && e->Cache.GroupMask.Group == 0 &&
            (e->Cache.Type == CacheUnified || e->Cache.Type == CacheData)) {
            bool seen = false;
            for (const L3Domain& d : out) seen |= d.mask == (uint64_t)e->Cache.GroupMask.Mask;
            if (!seen) out.push_back({ (uint64_t)e->Cache.GroupMask.Mask, (uint64_t)e->Cache.CacheSize });
        }
        off += e->Size;
    }
    std::sort(out.begin(), out.end(), [](const L3Domain& a, const L3Domain& b) {
        return (a.mask & (~a.mask + 1)) < (b.mask & (~b.mask + 1));
    });
    return out;
}

std::string cpu_list(uint64_t m) {
    std::string s;
    for (int i = 0; i < 64;) {
        if (!(m >> i & 1)) { ++i; continue; }
        int j = i;
        while (j + 1 < 64 && (m >> (j + 1) & 1)) ++j;
        s += (s.empty() ? "" : ",") + std::to_string(i) + (j > i ? "-" + std::to_string(j) : "");
        i = j + 1;
    }
    return s;
}

/* BRILLIANCE_CPUS: the logical processors this process runs on -- its rank threads, the HIP
 * runtime's threads and OpenMP's alike. "l3:N" is the Nth L3 domain, "vcache" the one with the
 * largest L3 (the 9950X3D's V-Cache CCD), anything else a list such as "0-7,16-23". Which domain
 * serves best is the probe's to say (its hot-path CCD measurement, docs/WINDOWS-PORT.md 4.7);
 * unset, Windows schedules over every processor as usual. */
void apply_cpus() {
    const char* v = std::getenv("BRILLIANCE_CPUS");
    if (!v || !*v) return;
    const std::vector<L3Domain> l3 = l3_domains();
    uint64_t want = 0;
    std::string how;
    if (_strnicmp(v, "l3:", 3) == 0) {
        const int k = std::atoi(v + 3);
        if (k >= 0 && k < (int)l3.size()) want = l3[(size_t)k].mask;
        how = "L3 domain " + std::to_string(k);
    } else if (_stricmp(v, "vcache") == 0) {
        const L3Domain* best = nullptr;
        for (const L3Domain& d : l3) if (!best || d.bytes > best->bytes) best = &d;
        if (best) want = best->mask;
        how = "the largest L3";
    } else {
        for (const char* s = v; *s;) {
            char* e = nullptr;
            const long a = std::strtol(s, &e, 10);
            long b = a;
            if (e == s) { want = 0; break; }
            if (*e == '-') { const char* t = e + 1; b = std::strtol(t, &e, 10); if (e == t) { want = 0; break; } }
            for (long i = a; i <= b && i < 64; ++i) if (i >= 0) want |= 1ull << i;
            s = *e == ',' ? e + 1 : e;
            if (*e && *e != ',') { want = 0; break; }
        }
        how = "as listed";
    }
    DWORD_PTR proc = 0, sys = 0;
    GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);
    want &= (uint64_t)sys;
    std::string doms;
    for (size_t i = 0; i < l3.size(); ++i)
        doms += (i ? "; " : "") + std::string("l3:") + std::to_string(i) + " = " + cpu_list(l3[i].mask) +
                " (" + std::to_string(l3[i].bytes >> 20) + " MiB)";
    if (!want) {
        log_os_line(1, "process: BRILLIANCE_CPUS=%s names no processor of this machine; "
                    "running on all of them. L3 domains: %s", v, doms.empty() ? "none" : doms.c_str());
        return;
    }
    if (!SetProcessAffinityMask(GetCurrentProcess(), (DWORD_PTR)want)) {
        log_os_line(1, "process: BRILLIANCE_CPUS=%s: %s", v, win32_message().c_str());
        return;
    }
    log_os_line(2, "process: running on CPUs %s (BRILLIANCE_CPUS=%s, %s). L3 domains: %s",
                cpu_list(want).c_str(), v, how.c_str(), doms.c_str());
}
}   // namespace

void process_init() {
    console_init();
    apply_cpus();
    // Power throttling off, for execution speed and -- the part the probe found -- for the timer
    // resolution, which Windows 11 otherwise ignores for this process.
    PROCESS_POWER_THROTTLING_STATE pt{};
    pt.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    pt.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | 0x4 /* IGNORE_TIMER_RESOLUTION */;
    pt.StateMask = 0;
    (void)SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pt, sizeof pt);
    using TBP = UINT(WINAPI*)(UINT);
    if (HMODULE wm = LoadLibraryW(L"winmm.dll"))
        if (auto tbp = (TBP)(void*)GetProcAddress(wm, "timeBeginPeriod")) tbp(1);
    // Never realtime: anything but "high" or "normal" is above-normal, with a warning.
    DWORD cls = ABOVE_NORMAL_PRIORITY_CLASS;
    const char* name = "above-normal";
    if (const char* p = std::getenv("BRILLIANCE_PRIORITY"); p && *p) {
        if (_stricmp(p, "high") == 0) { cls = HIGH_PRIORITY_CLASS; name = "high"; }
        else if (_stricmp(p, "normal") == 0) { cls = NORMAL_PRIORITY_CLASS; name = "normal"; }
        else if (_stricmp(p, "above") != 0)
            log_os_line(1, "process: BRILLIANCE_PRIORITY=%s is not high, above or normal; using above", p);
    }
    if (SetPriorityClass(GetCurrentProcess(), cls))
        log_os_line(2, "process: priority class %s (BRILLIANCE_PRIORITY), power throttling off, 1 ms timer", name);
    else
        log_os_line(1, "process: cannot set priority class %s: %s", name, win32_message().c_str());
}

void sleep_us(int64_t us) {
    if (us <= 0) {
        SwitchToThread();
        return;
    }
    if (us <= 500) {
        LARGE_INTEGER f, t0, t;
        QueryPerformanceFrequency(&f);
        QueryPerformanceCounter(&t0);
        const LONGLONG ticks = us * f.QuadPart / 1000000;
        do {
            for (int i = 0; i < 32; ++i) YieldProcessor();
            SwitchToThread();
            QueryPerformanceCounter(&t);
        } while (t.QuadPart - t0.QuadPart < ticks);
        return;
    }
    static thread_local ThreadHandle timer_h(
        CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS));
    HANDLE timer = timer_h.h;
    if (!timer) {
        Sleep((DWORD)((us + 999) / 1000));
        return;
    }
    LARGE_INTEGER due;
    due.QuadPart = -(LONGLONG)(us * 10);
    if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, INFINITE);
    else Sleep((DWORD)((us + 999) / 1000));
}

namespace {
UINT g_console_cp = 0;
void restore_console_cp() {
    if (g_console_cp) SetConsoleOutputCP(g_console_cp);
}
}   // namespace

void console_init() {
    // The console is shared with the shell that started us: its code page goes back at exit, or
    // cmd's chcp would stay 65001 after the server is gone.
    static bool once = false;
    if (!once) {
        once = true;
        g_console_cp = GetConsoleOutputCP();
        if (g_console_cp && g_console_cp != CP_UTF8) std::atexit(restore_console_cp);
    }
    SetConsoleOutputCP(CP_UTF8);
    for (DWORD id : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE}) {
        HANDLE h = GetStdHandle(id);
        DWORD mode = 0;
        if (h && h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode))
            SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
}

/* ---- the slot's lane width, from AMD's ADL (os.h) -------------------------------------------- */
namespace {

/* ADL returns what it allocates through this, and frees nothing itself. */
void* __stdcall adl_alloc(int n) { return std::malloc(n > 0 ? (size_t)n : 1); }

/* AdapterInfo on Windows (ADL SDK, adl_structures.h): iSize, iAdapterIndex, strUDID[256],
 * iBusNumber, iDeviceNumber, iFunctionNumber, iVendorID, strAdapterName[256], strDisplayName[256],
 * iPresent, iExist, three more 256-byte strings, iOSDisplayIndex. Read by offset, so no SDK header
 * is needed for two integers. */
constexpr int kAdlAdapterInfoBytes = 1572;
constexpr int kAdlAdapterIndexOff  = 4;
constexpr int kAdlBusNumberOff     = 264;

}   // namespace

std::vector<PciLanes> amd_lane_widths() {
    std::vector<PciLanes> out;
    HMODULE h = LoadLibraryA("atiadlxx.dll");
    if (!h) return out;
    typedef void* (__stdcall *AllocCb)(int);
    typedef int (*CreateFn)(AllocCb, int, void**);
    typedef int (*DestroyFn)(void*);
    typedef int (*CountFn)(void*, int*);
    typedef int (*InfoFn)(void*, void*, int);
    typedef int (*ChipFn)(void*, int, int*);
    const auto create  = (CreateFn)(void*)GetProcAddress(h, "ADL2_Main_Control_Create");
    const auto destroy = (DestroyFn)(void*)GetProcAddress(h, "ADL2_Main_Control_Destroy");
    const auto count   = (CountFn)(void*)GetProcAddress(h, "ADL2_Adapter_NumberOfAdapters_Get");
    const auto info    = (InfoFn)(void*)GetProcAddress(h, "ADL2_Adapter_AdapterInfo_Get");
    const auto chip    = (ChipFn)(void*)GetProcAddress(h, "ADL2_Adapter_ChipSetInfo_Get");
    void* ctx = nullptr;
    /* 0: every adapter, not only those with a display connected (served cards are headless). */
    if (create && destroy && count && info && chip && create(adl_alloc, 0, &ctx) == 0 && ctx) {
        int n = 0;
        if (count(ctx, &n) == 0 && n > 0 && n <= 1024) {
            std::vector<char> buf((size_t)n * kAdlAdapterInfoBytes, 0);
            if (info(ctx, buf.data(), n * kAdlAdapterInfoBytes) == 0) {
                for (int i = 0; i < n; ++i) {
                    const char* a = buf.data() + (size_t)i * kAdlAdapterInfoBytes;
                    int idx = 0, bus = 0;
                    std::memcpy(&idx, a + kAdlAdapterIndexOff, sizeof idx);
                    std::memcpy(&bus, a + kAdlBusNumberOff, sizeof bus);
                    /* An adapter appears once per display output; one answer a bus. */
                    bool seen = false;
                    for (const PciLanes& p : out) seen = seen || p.bus == bus;
                    if (seen) continue;
                    /* ADLChipSetInfo: iBusType, iBusSpeedType, iMaxPCIELaneWidth,
                     * iCurrentPCIELaneWidth, iSupportedAGPSpeeds, iCurrentAGPSpeed. */
                    int cs[6] = { 0 };
                    if (chip(ctx, idx, cs) == 0 && cs[3] > 0) {
                        PciLanes p;
                        p.bus = bus;
                        p.width = cs[3];
                        out.push_back(p);
                    }
                }
            }
        }
        destroy(ctx);
    }
    FreeLibrary(h);
    return out;
}

}   // namespace rad::os

#endif   // _WIN32
