// Brilliance: binds a plugin's imports of the host ABI to whichever executable loaded it.
//
// A plugin DLL imports rad_issue, rad_decl_* and the rest from "brilliance_host.dll", a module
// that does not exist: the name is only what the import library (cmake/BrillianceWindows.cmake)
// records. The import is DELAY-LOADED, so nothing resolves it when the plugin loads; on the first
// call the delay-load helper asks this hook before it would search for the DLL, and the hook
// answers with the running executable, which exports those functions from a .def file. The helper
// then GetProcAddress()es each one there.
//
// This is what the ELF build gets from the dynamic linker for free -- a plugin binds to radiance,
// rad-convert or a test, whichever dlopen'ed it -- and it is why plugin sources need no change.
// An executable that hosts plugins without exporting a function a plugin imports would fail at
// that plugin's first call to it, so the plugin loader reads the plugin's delay-import table
// against the executable's exports before it registers anything, and refuses the plugin naming
// what is missing (core/plugin/loader.cpp), as RTLD_NOW does on Linux. A harness that loads a
// plugin itself, past the loader, skips that check; for it the failure hook below names the
// function before the helper raises its exception.

#if defined(_WIN32)
#include <windows.h>
#include <delayimp.h>
#include <stdio.h>
#include <string.h>

extern "C" {

static bool is_host(PDelayLoadInfo info) {
    return info && info->szDll && _stricmp(info->szDll, "brilliance_host.dll") == 0;
}

// A COUNTED reference: the helper FreeLibrary()s the handle it was given when another thread
// resolved the same module first, and an uncounted one would be released from under the process.
static FARPROC WINAPI rad_host_delay_hook(unsigned notify, PDelayLoadInfo info) {
    if (notify != dliNotePreLoadLibrary || !is_host(info)) return nullptr;
    HMODULE exe = nullptr;
    return GetModuleHandleExW(0, nullptr, &exe) ? reinterpret_cast<FARPROC>(exe) : nullptr;
}

// The module's path in UTF-8 (the console and the logs take UTF-8), left as it was on failure.
static void module_name(HMODULE m, char* out, int cap) {
    wchar_t w[MAX_PATH];
    const DWORD n = GetModuleFileNameW(m, w, MAX_PATH);
    const int k = n && n < MAX_PATH
                      ? WideCharToMultiByte(CP_UTF8, 0, w, (int)n, out, cap - 1, nullptr, nullptr)
                      : 0;
    if (k > 0) out[k] = 0;
}

// A host function the executable does not export. There is nothing to bind the call to -- any
// stand-in would return something the plugin takes for an answer -- so the helper's exception
// (0xC06D007F) still ends the call; what this adds is the function's and the plugin's names on
// stderr, where the bare exception code names neither.
static FARPROC WINAPI rad_host_delay_failure(unsigned notify, PDelayLoadInfo info) {
    if (notify != dliFailGetProc || !is_host(info)) return nullptr;
    char ord[16];
    const char* fn = info->dlp.fImportByName ? info->dlp.szProcName : ord;
    if (!info->dlp.fImportByName) snprintf(ord, sizeof ord, "#%lu", (unsigned long)info->dlp.dwOrdinal);
    HMODULE self = nullptr;
    char plugin[3 * MAX_PATH] = "a plugin", exe[3 * MAX_PATH] = "this executable";
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&rad_host_delay_failure), &self))
        module_name(self, plugin, sizeof plugin);
    module_name(nullptr, exe, sizeof exe);
    fprintf(stderr, "brilliance: %s calls the host function %s, which %s does not export; an "
                    "executable that loads plugins exports the host ABI (brl_win_host_exports)\n",
            plugin, fn, exe);
    fflush(stderr);
    return nullptr;
}

// delayimp.h declares both `const`; the helper reads them once per unresolved import.
const PfnDliHook __pfnDliNotifyHook2 = rad_host_delay_hook;
const PfnDliHook __pfnDliFailureHook2 = rad_host_delay_failure;

}   // extern "C"
#endif
