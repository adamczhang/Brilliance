/* brl-stacks.cpp -- Brilliance: every thread's stack of a running process, from outside it.
 *
 *   brl-stacks <pid>
 *
 * For a server that stopped making progress (a step that hangs, /health reporting "stuck"), where
 * the engine's own watchdog prints only the step thread. Each thread is suspended for the moment
 * its stack is read and resumed at once; nothing is changed and nothing is ended. (A process with
 * work on a card is never killed: under Brilliance 0.2.2 that took the driver down.)
 *
 * A frame prints as module+offset, and as the nearest symbol DbgHelp knows (exports, or a PDB
 * beside the module). radiance.exe ships without one: resolve-stacks.py resolves its offsets
 * against radiance.map, which the build writes beside it (win/tools/stacks/README.md).
 *
 * Windows only; built with the toolchain's clang++ (see the README). */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dbghelp.h>
#include <psapi.h>
#include <tlhelp32.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "psapi.lib")

namespace {

struct Mod {
    DWORD64 base = 0, size = 0;
    std::string name;
};

std::vector<Mod> modules(HANDLE p) {
    std::vector<Mod> out;
    HMODULE hm[2048];
    DWORD need = 0;
    if (!EnumProcessModulesEx(p, hm, sizeof hm, &need, LIST_MODULES_ALL)) return out;
    for (DWORD i = 0; i < need / sizeof(HMODULE) && i < 2048; ++i) {
        MODULEINFO mi{};
        char nm[MAX_PATH] = {};
        if (!GetModuleInformation(p, hm[i], &mi, sizeof mi)) continue;
        GetModuleBaseNameA(p, hm[i], nm, sizeof nm);
        out.push_back({(DWORD64)mi.lpBaseOfDll, (DWORD64)mi.SizeOfImage, nm});
    }
    return out;
}

std::string where(HANDLE p, const std::vector<Mod>& mods, DWORD64 pc) {
    char buf[1024];
    std::string s;
    for (const Mod& m : mods)
        if (pc >= m.base && pc < m.base + m.size) {
            std::snprintf(buf, sizeof buf, "%s+0x%llx", m.name.c_str(),
                          (unsigned long long)(pc - m.base));
            s = buf;
            break;
        }
    if (s.empty()) {
        std::snprintf(buf, sizeof buf, "0x%llx", (unsigned long long)pc);
        s = buf;
    }
    alignas(SYMBOL_INFO) char sb[sizeof(SYMBOL_INFO) + 512];
    SYMBOL_INFO* si = (SYMBOL_INFO*)sb;
    si->SizeOfStruct = sizeof(SYMBOL_INFO);
    si->MaxNameLen = 511;
    DWORD64 disp = 0;
    if (SymFromAddr(p, pc, &disp, si)) {
        std::snprintf(buf, sizeof buf, "  (%s+0x%llx)", si->Name, (unsigned long long)disp);
        s += buf;
    }
    return s;
}

std::string thread_name(HANDLE t) {
    using Fn = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    static Fn fn = (Fn)(void*)GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription");
    if (!fn) return "";
    PWSTR w = nullptr;
    std::string out;
    if (SUCCEEDED(fn(t, &w)) && w) {
        char b[256];
        if (WideCharToMultiByte(CP_UTF8, 0, w, -1, b, sizeof b, nullptr, nullptr) > 0) out = b;
        LocalFree(w);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: brl-stacks <pid>\n");
        return 2;
    }
    const DWORD pid = (DWORD)std::strtoul(argv[1], nullptr, 10);
    HANDLE p = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    if (!p) {
        std::fprintf(stderr, "brl-stacks: cannot open process %lu (error %lu)\n", pid, GetLastError());
        return 1;
    }
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_FAIL_CRITICAL_ERRORS);
    if (!SymInitialize(p, nullptr, TRUE))
        std::fprintf(stderr, "brl-stacks: SymInitialize failed (error %lu); offsets only\n", GetLastError());
    const std::vector<Mod> mods = modules(p);

    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "brl-stacks: no thread snapshot (error %lu)\n", GetLastError());
        return 1;
    }
    THREADENTRY32 te{};
    te.dwSize = sizeof te;
    int n = 0;
    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
        if (te.th32OwnerProcessID != pid) continue;
        HANDLE t = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_LIMITED_INFORMATION,
                              FALSE, te.th32ThreadID);
        if (!t) {
            std::printf("thread %lu: cannot open (error %lu)\n\n", te.th32ThreadID, GetLastError());
            continue;
        }
        const std::string nm = thread_name(t);
        std::vector<DWORD64> pcs;
        if (SuspendThread(t) != (DWORD)-1) {
            CONTEXT c{};
            c.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(t, &c)) {
                STACKFRAME64 f{};
                f.AddrPC.Offset = c.Rip;
                f.AddrPC.Mode = AddrModeFlat;
                f.AddrFrame.Offset = c.Rbp;
                f.AddrFrame.Mode = AddrModeFlat;
                f.AddrStack.Offset = c.Rsp;
                f.AddrStack.Mode = AddrModeFlat;
                for (int i = 0; i < 64; ++i) {
                    if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, p, t, &f, &c, nullptr,
                                     SymFunctionTableAccess64, SymGetModuleBase64, nullptr))
                        break;
                    if (!f.AddrPC.Offset) break;
                    pcs.push_back(f.AddrPC.Offset);
                }
            }
            ResumeThread(t);   /* before any symbol lookup: the thread is held only for the walk */
        }
        CloseHandle(t);
        std::printf("thread %lu%s%s%s\n", te.th32ThreadID, nm.empty() ? "" : " \"", nm.c_str(),
                    nm.empty() ? "" : "\"");
        for (DWORD64 pc : pcs) std::printf("    %s\n", where(p, mods, pc).c_str());
        std::printf("\n");
        ++n;
    }
    CloseHandle(snap);
    std::printf("%d thread(s) of process %lu\n", n, pid);
    SymCleanup(p);
    CloseHandle(p);
    return 0;
}
