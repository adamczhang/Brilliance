/* Brilliance: the POSIX compatibility layer for Windows -- what every header in this directory
 * shares.
 *
 * WHAT THIS LAYER IS FOR. It lets radiance's sources include and call the POSIX functions whose
 * meaning carries over to Windows unchanged -- dlopen, opendir, isatty, sysconf, usleep,
 * gmtime_r, setenv -- so those files compile as radiance wrote them and merge cleanly when
 * radiance changes them.
 *
 * Some needed more than a rename and are emulated where radiance's callers use them as Linux does:
 * off_t is 64 bits here (the UCRT's is 32), and pread/pwrite, lseek, ftruncate and read/write take
 * 64-bit offsets and counts; mmap maps read-only views and anonymous memory (a writable or shared
 * file mapping is refused with EINVAL); a bad descriptor is EBADF, not the CRT's fast-fail.
 *
 * WHAT IT IS NOT FOR. Anything whose meaning does NOT carry over is left out on purpose, so that a
 * call to it is a compile error that points at the site rather than an emulation that is subtly
 * wrong: O_DIRECT, fork, signals beyond Ctrl-C, posix_memalign (its pointer cannot be passed to
 * free()). Those sites use core/os/os.h explicitly.
 *
 * The directory is on the include path of the Windows build only (cmake/BrillianceWindows.cmake),
 * and the implementations are the brl_compat library every target links. */
#pragma once

#if !defined(_WIN32)
#error "win/compat is the Windows build's POSIX layer and must not be on a POSIX include path"
#endif

#include <stddef.h>
#include <stdint.h>

#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef intptr_t ssize_t;
#endif
#ifndef _PID_T_DEFINED
#define _PID_T_DEFINED
typedef int pid_t;   /* a Windows process id */
#endif

#ifdef __cplusplus
#define BRL_C_BEGIN extern "C" {
#define BRL_C_END   }
#else
#define BRL_C_BEGIN
#define BRL_C_END
#endif

BRL_C_BEGIN
/* errno from a Win32 error code, for the few emulations that fail through GetLastError. */
int brl_errno_from_win32(unsigned long err);
BRL_C_END
