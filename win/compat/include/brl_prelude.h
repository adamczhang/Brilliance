/* Brilliance: force-included into every C, C++ and HIP translation unit of the Windows build
 * (-include, cmake/BrillianceWindows.cmake).
 *
 * It declares the handful of POSIX functions that live in standard headers on Linux -- <stdlib.h>,
 * <string.h>, <time.h> -- and that the UCRT does not have. Declaring them here rather than
 * shadowing those standard headers keeps the C library's own headers untouched. Nothing in this
 * file may include <windows.h>: it is in every translation unit. */
#pragma once
#if defined(_WIN32)

/* OFF_T IS 64 BITS, AS ON EVERY PLATFORM RADIANCE RUNS ON. The UCRT's is 32 bits, and every
 * `(off_t)offset` cast in the tree would truncate a container offset past 2 GiB before any call
 * saw it. The UCRT defines off_t in one place (ucrt/sys/types.h, under _OFF_T_DEFINED) and uses it
 * in none of its own declarations -- its functions take `long` or `_off_t` by name -- so defining
 * the guard first and giving off_t 64 bits changes user code only. _off_t keeps the CRT's 32-bit
 * meaning because struct stat and _lseek are declared with it. */
#ifndef _OFF_T_DEFINED
#define _OFF_T_DEFINED
typedef long      _off_t;
typedef long long off_t;
#endif

#include <stddef.h>
#include <stdio.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

int        setenv(const char* name, const char* value, int overwrite);
int        unsetenv(const char* name);
struct tm* gmtime_r(const time_t* t, struct tm* out);
struct tm* localtime_r(const time_t* t, struct tm* out);
char*      strndup(const char* s, size_t n);
char*      strtok_r(char* s, const char* delim, char** save);
/* POSIX getline (the return is ssize_t's width, which brl_compat.h makes intptr_t). */
long long  getline(char** lineptr, size_t* n, FILE* stream);
/* Temporary files and directories from a "...XXXXXX" template, replaced in place. A template under
 * "/tmp" means the \tmp directory of the current drive on Windows; it is created if missing, the
 * way every POSIX system guarantees /tmp exists. */
char*      mkdtemp(char* tmpl);
int        mkstemp(char* tmpl);
int        truncate(const char* path, long long len);

#ifdef __cplusplus
}
#endif

#endif /* _WIN32 */
