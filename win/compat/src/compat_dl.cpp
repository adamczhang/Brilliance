// Brilliance POSIX layer: <dlfcn.h> and <dirent.h> over the Win32 loader and FindFirstFileExW.

#include <windows.h>
#include <psapi.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "brl_compat.h"
#include "dlfcn.h"
#include "dirent.h"

namespace {

std::wstring widen(const char* s) {
    if (!s) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    if (n <= 0) return {};
    std::wstring w((size_t)n - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
    return w;
}

std::string narrow(const wchar_t* w) {
    if (!w) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) return {};
    std::string s((size_t)n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

thread_local std::string t_dlerr;
thread_local bool        t_dlerr_set = false;

void set_dlerror(const char* what, const char* arg, DWORD err) {
    wchar_t* msg = nullptr;
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                       FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, err, 0, (LPWSTR)&msg, 0, nullptr);
    std::string m = msg ? narrow(msg) : std::string("Win32 error ") + std::to_string(err);
    if (msg) LocalFree(msg);
    while (!m.empty() && (m.back() == '\n' || m.back() == '\r' || m.back() == ' ')) m.pop_back();
    t_dlerr = std::string(arg ? arg : "") + ": " + what + ": " + m;
    t_dlerr_set = true;
}

// A NAME WITH A SEPARATOR IS A PATH, RELATIVE TO THE CURRENT DIRECTORY, as dlopen takes any name
// containing a '/'. LoadLibraryExW takes a relative path as a name to SEARCH for -- the
// executable's directory first -- so "plugins/x.dll" would load a same-named file beside the
// executable over the one the caller pointed at. It is made absolute first (GetFullPathNameW, the
// current directory at the call).
std::wstring full_path(const std::wstring& p) {
    const DWORD n = GetFullPathNameW(p.c_str(), 0, nullptr, nullptr);
    if (!n) return p;
    std::wstring f((size_t)n, L'\0');
    const DWORD k = GetFullPathNameW(p.c_str(), n, f.data(), nullptr);
    if (!k || k >= n) return p;
    f.resize(k);
    return f;
}

}   // namespace

extern "C" {

void* dlopen(const char* path, int flags) {
    if (!path) return (void*)GetModuleHandleW(nullptr);
    std::wstring w = widen(path);
    for (auto& c : w)
        if (c == L'/') c = L'\\';
    const bool is_path = w.find(L'\\') != std::wstring::npos;
    if (is_path) w = full_path(w);
    // A Linux plugin path names "libfoo.so"; its Windows build is "libfoo.dll" in the same place.
    // Taking the one for the other when the .so does not exist keeps every path radiance spells --
    // in its tests, its docs' --kernels examples, an operator's habits -- meaning the same plugin.
    if (w.size() > 3 && _wcsicmp(w.c_str() + w.size() - 3, L".so") == 0 &&
        GetFileAttributesW(w.c_str()) == INVALID_FILE_ATTRIBUTES)
        w = w.substr(0, w.size() - 3) + L".dll";
    // A missing dependency must be an error the caller reports, not a modal dialog that hangs a
    // service with nobody to click it. This thread's mode only: the process's belongs to the host.
    DWORD old_mode = 0;
    const bool moded = SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX, &old_mode) != 0;
    HMODULE h = nullptr;
    if (flags & RTLD_NOLOAD) {
        GetModuleHandleExW(0, w.c_str(), &h);
    } else {
        // A bare name is searched for -- the executable's directory, System32, AddDllDirectory's --
        // and never the current directory. LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR needs a full path.
        h = LoadLibraryExW(w.c_str(), nullptr,
                           is_path ? (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
                                   : LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    }
    const DWORD err = GetLastError();
    if (moded) SetThreadErrorMode(old_mode, nullptr);
    if (!h) set_dlerror("cannot load", path, err);
    return (void*)h;
}

void* dlsym(void* handle, const char* name) {
    FARPROC p = nullptr;
    if (handle == RTLD_DEFAULT) {
        // The executable first, then every module in load order: the nearest Windows has to a
        // global symbol scope.
        p = GetProcAddress(GetModuleHandleW(nullptr), name);
        if (!p) {
            HMODULE mods[1024];
            DWORD need = 0;
            if (EnumProcessModules(GetCurrentProcess(), mods, sizeof mods, &need)) {
                const DWORD n = need / sizeof(HMODULE);
                for (DWORD i = 0; i < n && i < 1024 && !p; ++i) p = GetProcAddress(mods[i], name);
            }
        }
    } else {
        p = GetProcAddress((HMODULE)handle, name);
    }
    if (!p) set_dlerror("undefined symbol", name, GetLastError());
    return (void*)p;
}

int dlclose(void* handle) {
    if (!handle || handle == (void*)GetModuleHandleW(nullptr)) return 0;
    if (!FreeLibrary((HMODULE)handle)) {
        set_dlerror("cannot unload", "", GetLastError());
        return -1;
    }
    return 0;
}

char* dlerror(void) {
    if (!t_dlerr_set) return nullptr;
    t_dlerr_set = false;
    return &t_dlerr[0];
}

int dladdr(const void* addr, Dl_info* info) {
    HMODULE h = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)addr, &h))
        return 0;
    static thread_local std::string name;
    wchar_t buf[32768];
    const DWORD n = GetModuleFileNameW(h, buf, (DWORD)(sizeof buf / sizeof buf[0]));
    name = n ? narrow(buf) : std::string();
    info->dli_fname = name.c_str();
    info->dli_fbase = (void*)h;
    info->dli_sname = nullptr;
    info->dli_saddr = nullptr;
    return 1;
}

// ------------------------------------------------------------------------------- dirent
struct brl_DIR {
    HANDLE           h = INVALID_HANDLE_VALUE;
    WIN32_FIND_DATAW fd{};
    bool             first = true;
    struct dirent    ent{};
};

DIR* opendir(const char* path) {
    std::wstring w = widen(path);
    if (w.empty()) {
        errno = ENOENT;
        return nullptr;
    }
    if (w.back() != L'\\' && w.back() != L'/') w += L'\\';
    w += L'*';
    auto* d = new brl_DIR;
    d->h = FindFirstFileExW(w.c_str(), FindExInfoBasic, &d->fd, FindExSearchNameMatch, nullptr,
                            FIND_FIRST_EX_LARGE_FETCH);
    if (d->h == INVALID_HANDLE_VALUE) {
        errno = brl_errno_from_win32(GetLastError());
        delete d;
        return nullptr;
    }
    return d;
}

struct dirent* readdir(DIR* d) {
    if (!d) return nullptr;
    if (!d->first) {
        if (!FindNextFileW(d->h, &d->fd)) return nullptr;
    }
    d->first = false;
    const std::string n = narrow(d->fd.cFileName);
    strncpy(d->ent.d_name, n.c_str(), sizeof d->ent.d_name - 1);
    d->ent.d_name[sizeof d->ent.d_name - 1] = 0;
    const DWORD a = d->fd.dwFileAttributes;
    d->ent.d_type = (a & FILE_ATTRIBUTE_REPARSE_POINT) ? DT_LNK
                    : (a & FILE_ATTRIBUTE_DIRECTORY)   ? DT_DIR
                                                       : DT_REG;
    return &d->ent;
}

int closedir(DIR* d) {
    if (!d) return -1;
    if (d->h != INVALID_HANDLE_VALUE) FindClose(d->h);
    delete d;
    return 0;
}

}   // extern "C"
