// Brilliance POSIX layer: <sys/mman.h> over the Win32 memory manager. See the header for exactly
// which uses are supported and why the rest are refused.

#include <windows.h>
#include <io.h>
#include <errno.h>

#include <map>
#include <mutex>
#include <string>

#include "brl_compat.h"
#include "sys/mman.h"
#include "fcntl.h"
#include "compat_internal.h"

namespace {

void* fail(int e) {
    errno = e;
    return MAP_FAILED;
}

int fail_i(int e) {
    errno = e;
    return -1;
}

// THE WORKING SET IS WINDOWS' LIMIT ON LOCKED MEMORY: VirtualLock fails with a quota error once the
// locked pages would exceed the process's minimum working set, so mlock raises the minimum (and the
// maximum with it) by what it locks when it has to. munlock gives back what its mlock added and no
// more -- a lock that fit under the old minimum added nothing -- so a process that locks and
// unlocks a table in turn does not ratchet its working set up for good. Keyed by the locked range's
// start, which is how the callers name a range to munlock.
std::mutex                  g_ws_mu;
std::map<uintptr_t, SIZE_T> g_ws_added;

bool shift_working_set(SIZE_T add, bool up) {
    SIZE_T mn = 0, mx = 0;
    DWORD fl = 0;
    if (!GetProcessWorkingSetSizeEx(GetCurrentProcess(), &mn, &mx, &fl)) return false;
    if (!up && (mn < add || mx < add)) return false;
    return SetProcessWorkingSetSizeEx(GetCurrentProcess(), up ? mn + add : mn - add,
                                      up ? mx + add : mx - add,
                                      QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE) != 0;
}

// The pages [addr, addr + len) touches, as VirtualLock counts them.
SIZE_T page_span(const void* addr, size_t len) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const uintptr_t pg = si.dwPageSize, a = (uintptr_t)addr;
    return (SIZE_T)(((a + len + pg - 1) & ~(pg - 1)) - (a & ~(pg - 1)));
}

}   // namespace

extern "C" {

void* mmap(void* addr, size_t len, int prot, int flags, int fd, long long off) {
    if (len == 0) return fail(EINVAL);
    if (flags & MAP_FIXED) return fail(EINVAL);   // see the header: use os::vmem_* for this
    (void)addr;

    if (flags & MAP_ANONYMOUS) {
        if (prot == PROT_NONE) {
            void* p = VirtualAlloc(nullptr, len, MEM_RESERVE, PAGE_NOACCESS);
            return p ? p : fail(ENOMEM);
        }
        const DWORD pr = (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY;
        void* p = VirtualAlloc(nullptr, len, MEM_RESERVE | MEM_COMMIT, pr);
        return p ? p : fail(ENOMEM);
    }

    // A read-only view of a file. Writable file mappings are not something radiance makes.
    if (prot & PROT_WRITE) return fail(EINVAL);
    HANDLE fh = brl::osfhandle(fd);
    if (fh == INVALID_HANDLE_VALUE) return fail(EBADF);
    // A MAPPED FILE CAN STILL BE DELETED OR REPLACED, as on Linux. Windows allows that only when the
    // file object the view keeps alive was opened with FILE_SHARE_DELETE, and the CRT's open() asks
    // for no such thing. So the view is made from a second handle on the same file that does: once
    // the caller closes fd -- at once, as POSIX code does after mmap -- nothing the mapping holds
    // stands in the way of removing a temporary directory or renaming a new file over this one.
    HANDLE share = INVALID_HANDLE_VALUE;
    {
        std::wstring path(1024, L'\0');
        DWORD n = GetFinalPathNameByHandleW(fh, path.data(), (DWORD)path.size(), FILE_NAME_NORMALIZED);
        if (n >= path.size()) {
            path.assign((size_t)n + 1, L'\0');
            n = GetFinalPathNameByHandleW(fh, path.data(), (DWORD)path.size(), FILE_NAME_NORMALIZED);
        }
        if (n > 0 && n < path.size())
            share = CreateFileW(path.c_str(), GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    }
    HANDLE m = CreateFileMappingW(share != INVALID_HANDLE_VALUE ? share : fh, nullptr, PAGE_READONLY,
                                  0, 0, nullptr);
    if (share != INVALID_HANDLE_VALUE) CloseHandle(share);   // the section holds the file now
    if (!m) return fail(brl_errno_from_win32(GetLastError()));
    const unsigned long long o = (unsigned long long)off;
    void* v = MapViewOfFile(m, FILE_MAP_READ, (DWORD)(o >> 32), (DWORD)o, len);
    const DWORD e = GetLastError();
    // The view keeps the section and the file alive; the caller may close fd at once, as on Linux.
    CloseHandle(m);
    if (!v) return fail(brl_errno_from_win32(e));
    if (flags & MAP_POPULATE) {
        WIN32_MEMORY_RANGE_ENTRY r{v, len};
        (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &r, 0);
    }
    return v;
}

int munmap(void* addr, size_t len) {
    MEMORY_BASIC_INFORMATION mbi{};
    if (!addr || !len || !VirtualQuery(addr, &mbi, sizeof mbi)) return fail_i(EINVAL);
    // Only a whole region: Windows releases a view or an allocation by its base, never a piece. So
    // the length must cover exactly the pages mmap gave -- the regions that share the base, which
    // mprotect may have split -- or the call would free more than it names (a short length) or
    // claim to free what it did not (a long one).
    if (mbi.AllocationBase != addr) return fail_i(EINVAL);
    SIZE_T extent = 0;
    for (const char* p = (const char*)addr;
         VirtualQuery(p, &mbi, sizeof mbi) && mbi.AllocationBase == addr && mbi.State != MEM_FREE;
         p += mbi.RegionSize)
        extent += mbi.RegionSize;
    if (page_span(addr, len) != extent) return fail_i(EINVAL);
    VirtualQuery(addr, &mbi, sizeof mbi);
    if (mbi.Type == MEM_MAPPED) return UnmapViewOfFile(addr) ? 0 : fail_i(EINVAL);
    return VirtualFree(addr, 0, MEM_RELEASE) ? 0 : fail_i(EINVAL);
}

int mprotect(void* addr, size_t len, int prot) {
    DWORD old = 0;
    const DWORD pr = prot == PROT_NONE         ? PAGE_NOACCESS
                     : (prot & PROT_WRITE)      ? PAGE_READWRITE
                                                : PAGE_READONLY;
    if (prot != PROT_NONE) {
        // Access to reserved-only pages needs a commit first, which is what a POSIX mprotect over
        // PROT_NONE anonymous memory amounts to.
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(addr, &mbi, sizeof mbi) && mbi.State == MEM_RESERVE)
            return VirtualAlloc(addr, len, MEM_COMMIT, pr) ? 0 : fail_i(ENOMEM);
    }
    return VirtualProtect(addr, len, pr, &old) ? 0 : fail_i(EINVAL);
}

int madvise(void* addr, size_t len, int advice) {
    if (advice == MADV_WILLNEED && addr && len) {
        WIN32_MEMORY_RANGE_ENTRY r{addr, len};
        (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &r, 0);
    }
    return 0;
}

int posix_madvise(void* addr, size_t len, int advice) { return madvise(addr, len, advice); }

int mlock(const void* addr, size_t len) {
    if (VirtualLock(const_cast<void*>(addr), len)) return 0;
    if (GetLastError() != ERROR_WORKING_SET_QUOTA) return fail_i(ENOMEM);
    std::lock_guard<std::mutex> lk(g_ws_mu);
    const SIZE_T add = page_span(addr, len) + (SIZE_T)(16u << 20);
    if (!shift_working_set(add, true)) return fail_i(ENOMEM);
    if (!VirtualLock(const_cast<void*>(addr), len)) {
        shift_working_set(add, false);
        return fail_i(ENOMEM);
    }
    g_ws_added[(uintptr_t)addr] += add;
    return 0;
}

// MLOCK_ONFAULT has no Windows equivalent: the whole range is locked, which faults it in.
int mlock2(const void* addr, size_t len, unsigned /*flags*/) { return mlock(addr, len); }

int munlock(const void* addr, size_t len) {
    VirtualUnlock(const_cast<void*>(addr), len);
    std::lock_guard<std::mutex> lk(g_ws_mu);
    auto it = g_ws_added.find((uintptr_t)addr);
    if (it != g_ws_added.end()) {
        shift_working_set(it->second, false);
        g_ws_added.erase(it);
    }
    return 0;
}

int posix_fadvise(int fd, long long off, long long len, int advice) {
    (void)fd; (void)off; (void)len; (void)advice;
    return 0;   // advisory; the Windows cache manager takes no per-range hints
}

// A temporary file in %TEMP%, opened delete-on-close and shared every way, so a child handed it as
// its stdout writes into it, the parent reads it back by offset (pread reopens it), and it is gone
// once both have closed it. The handle is created non-inheritable: a child gets it only through a
// spawn's dup2. `name` labels a Linux memfd in /proc and has no use here.
int memfd_create(const char* name, unsigned flags) {
    (void)name; (void)flags;
    wchar_t dir[MAX_PATH + 1];
    const DWORD n = GetTempPathW(MAX_PATH + 1, dir);
    if (n == 0 || n > MAX_PATH) return fail_i(ENOENT);
    wchar_t path[MAX_PATH + 1];
    if (!GetTempFileNameW(dir, L"mfd", 0, path)) return fail_i(brl_errno_from_win32(GetLastError()));
    HANDLE h = CreateFileW(path, GENERIC_READ | GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE,
                           nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        const DWORD e = GetLastError();
        DeleteFileW(path);
        return fail_i(brl_errno_from_win32(e));
    }
    const int fd = _open_osfhandle((intptr_t)h, _O_RDWR | _O_BINARY);
    if (fd < 0) {
        CloseHandle(h);
        return fail_i(EMFILE);
    }
    return fd;
}

}   // extern "C"
