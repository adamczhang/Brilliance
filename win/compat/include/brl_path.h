/* Brilliance: a Windows path spelled the way radiance's sources take paths apart.
 *
 * Radiance finds a file beside another by cutting the path at its last '/': r4d_selftest opens
 * "libref.so" beside the plugin under test that way. A path typed or scripted on Windows
 * (D:\...\kernels\libr4d.dll) has no '/', so the cut leaves no directory, the bare name is searched
 * for beside the executable instead, and every row that checks a kernel against libref failed with
 * "reference plugin unavailable" or "oracle missing" (2026-11-20, st1: 27 rows, 409 FAIL lines).
 *
 * The Win32 loader and the CRT take '/' and '\' alike, and the POSIX layer's dlopen turns '/' back
 * into '\' itself, so a program that takes a path on its command line converts it once, here, and
 * radiance's code reads it as written. C++ only. */
#pragma once

#if !defined(_WIN32)
#error "win/compat is the Windows build's POSIX layer and must not be on a POSIX include path"
#endif

#include <string>

/* `p` with every '\' made '/'. A UNC prefix (\\server\share) becomes //server/share, which the
 * loader and the CRT also take. */
inline std::string brl_forward_slashes(std::string p) {
    for (char& c : p)
        if (c == '\\') c = '/';
    return p;
}
