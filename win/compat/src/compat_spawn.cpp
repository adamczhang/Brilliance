// Brilliance POSIX layer: posix_spawn and waitpid over CreateProcessW (see spawn.h).

#include <windows.h>
#include <errno.h>
#include <io.h>
#include <string.h>

#include <algorithm>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#include "brl_compat.h"
#include "spawn.h"
#include "sys/wait.h"

namespace {

std::wstring widen(const char* s) {
    if (!s) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    if (n <= 0) return {};
    std::wstring w((size_t)n - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n);
    return w;
}

// One argument quoted as the CRT will split it back into argv.
void append_quoted(std::wstring& cmd, const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos) {
        cmd += a;
        return;
    }
    cmd += L'"';
    for (size_t i = 0;; ++i) {
        size_t nbs = 0;
        while (i < a.size() && a[i] == L'\\') {
            ++i;
            ++nbs;
        }
        if (i == a.size()) {
            cmd.append(nbs * 2, L'\\');
            break;
        }
        if (a[i] == L'"') {
            cmd.append(nbs * 2 + 1, L'\\');
            cmd += L'"';
        } else {
            cmd.append(nbs, L'\\');
            cmd += a[i];
        }
    }
    cmd += L'"';
}

std::mutex                          g_mu;
std::unordered_map<pid_t, HANDLE>   g_children;

// posix_spawn_file_actions_t points at one of these: the dup2s, in order.
struct Actions {
    std::vector<std::pair<int, int>> dup2;   // (fd, newfd), newfd 0..2
};

// An inheritable duplicate of `h` for the child, or null when `h` is not a handle.
HANDLE inheritable(HANDLE h) {
    if (!h || h == INVALID_HANDLE_VALUE) return nullptr;
    HANDLE d = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), h, GetCurrentProcess(), &d, 0, TRUE,
                         DUPLICATE_SAME_ACCESS))
        return nullptr;
    return d;
}

int spawn(pid_t* pid, const char* path, char* const argv[], char* const envp[],
          const Actions* actions = nullptr) {
    std::wstring cmd;
    for (int i = 0; argv && argv[i]; ++i) {
        if (i) cmd += L' ';
        append_quoted(cmd, widen(argv[i]));
    }
    std::vector<wchar_t> env;
    if (envp) {
        for (int i = 0; envp[i]; ++i) {
            const std::wstring e = widen(envp[i]);
            env.insert(env.end(), e.begin(), e.end());
            env.push_back(L'\0');
        }
        env.push_back(L'\0');
    }
    STARTUPINFOW si{};
    si.cb = sizeof si;
    // With dup2 actions the child's three standard handles are given explicitly: each is this
    // process's own, or the descriptor a dup2 names, as an inheritable duplicate closed again once
    // the child has it (posix_spawn's dup2 likewise leaves the parent's descriptors as they were).
    HANDLE std3[3] = { nullptr, nullptr, nullptr };
    if (actions && !actions->dup2.empty()) {
        const DWORD which[3] = { STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE };
        HANDLE src[3];
        for (int i = 0; i < 3; ++i) src[i] = GetStdHandle(which[i]);
        for (const auto& a : actions->dup2) {
            const HANDLE h = (HANDLE)_get_osfhandle(a.first);
            if (h == INVALID_HANDLE_VALUE) return EBADF;
            src[a.second] = h;
        }
        for (int i = 0; i < 3; ++i) std3[i] = inheritable(src[i]);
        si.dwFlags |= STARTF_USESTDHANDLES;
        si.hStdInput = std3[0];
        si.hStdOutput = std3[1];
        si.hStdError = std3[2];
    }
    PROCESS_INFORMATION pi{};
    const std::wstring app = widen(path);
    const BOOL made =
        CreateProcessW(app.empty() ? nullptr : app.c_str(), cmd.empty() ? nullptr : &cmd[0], nullptr,
                       nullptr, TRUE, CREATE_UNICODE_ENVIRONMENT, envp ? env.data() : nullptr, nullptr,
                       &si, &pi);
    const DWORD err = made ? 0 : GetLastError();
    for (HANDLE h : std3)
        if (h) CloseHandle(h);
    if (!made) return brl_errno_from_win32(err);
    CloseHandle(pi.hThread);
    std::lock_guard<std::mutex> lk(g_mu);
    g_children[(pid_t)pi.dwProcessId] = pi.hProcess;
    if (pid) *pid = (pid_t)pi.dwProcessId;
    return 0;
}

}  // namespace

extern "C" {

int posix_spawn_file_actions_init(posix_spawn_file_actions_t* fa) {
    if (!fa) return EINVAL;
    *fa = new (std::nothrow) Actions;
    return *fa ? 0 : ENOMEM;
}

int posix_spawn_file_actions_destroy(posix_spawn_file_actions_t* fa) {
    if (!fa) return EINVAL;
    delete static_cast<Actions*>(*fa);
    *fa = nullptr;
    return 0;
}

int posix_spawn_file_actions_adddup2(posix_spawn_file_actions_t* fa, int fd, int newfd) {
    if (!fa || !*fa || fd < 0 || newfd < 0 || newfd > 2) return EINVAL;
    static_cast<Actions*>(*fa)->dup2.emplace_back(fd, newfd);
    return 0;
}

int posix_spawn(pid_t* pid, const char* path, const posix_spawn_file_actions_t* actions,
                const posix_spawnattr_t* attr, char* const argv[], char* const envp[]) {
    if (attr && *attr) return ENOTSUP;
    return spawn(pid, path, argv, envp,
                 actions && *actions ? static_cast<const Actions*>(*actions) : nullptr);
}

int posix_spawnp(pid_t* pid, const char* file, const posix_spawn_file_actions_t* actions,
                 const posix_spawnattr_t* attr, char* const argv[], char* const envp[]) {
    if ((actions && *actions) || (attr && *attr)) return ENOTSUP;
    return spawn(pid, nullptr, argv, envp);   // CreateProcessW searches PATH for argv[0]
    (void)file;
}

pid_t waitpid(pid_t pid, int* status, int options) {
    if (pid == -1) {
        // Any child: the first of them to end. WaitForMultipleObjects takes 64 at a time, so more
        // children than that are polled a group at a time.
        for (;;) {
            std::vector<pid_t> ids;
            std::vector<HANDLE> hs;
            {
                std::lock_guard<std::mutex> lk(g_mu);
                for (const auto& c : g_children) { ids.push_back(c.first); hs.push_back(c.second); }
            }
            if (hs.empty()) {
                errno = ECHILD;
                return -1;
            }
            const bool one_pass = (options & WNOHANG) != 0;
            const DWORD wait_ms = one_pass ? 0 : (hs.size() <= MAXIMUM_WAIT_OBJECTS ? INFINITE : 10);
            for (size_t at = 0; at < hs.size(); at += MAXIMUM_WAIT_OBJECTS) {
                const DWORD n = (DWORD)std::min<size_t>(MAXIMUM_WAIT_OBJECTS, hs.size() - at);
                const DWORD w = WaitForMultipleObjects(n, hs.data() + at, FALSE, wait_ms);
                if (w < WAIT_OBJECT_0 + n) return waitpid(ids[at + (w - WAIT_OBJECT_0)], status, 0);
                if (w == WAIT_FAILED) {
                    errno = brl_errno_from_win32(GetLastError());
                    return -1;
                }
            }
            if (one_pass) return 0;
        }
    }
    HANDLE h = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_children.find(pid);
        if (it == g_children.end()) {
            errno = ECHILD;
            return -1;
        }
        h = it->second;
    }
    const DWORD w = WaitForSingleObject(h, (options & WNOHANG) ? 0 : INFINITE);
    if (w == WAIT_TIMEOUT) return 0;
    DWORD code = 0;
    GetExitCodeProcess(h, &code);
    if (status) *status = (int)code;
    std::lock_guard<std::mutex> lk(g_mu);
    CloseHandle(h);
    g_children.erase(pid);
    return pid;
}

}  // extern "C"
