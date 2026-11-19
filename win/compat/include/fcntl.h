/* Brilliance POSIX layer: <fcntl.h>.
 *
 * O_CLOEXEC maps to _O_NOINHERIT, which is its meaning. O_DIRECT is deliberately NOT defined: an
 * unbuffered read on Windows needs sector-aligned offsets, lengths and buffers and a handle opened
 * with FILE_FLAG_NO_BUFFERING, which no flag on the CRT's open() can ask for, so a site that wants
 * it is a compile error that leads to os::File. Files open in binary mode by default in every
 * Brilliance executable (binmode.obj, cmake/BrillianceWindows.cmake), as on Linux. */
#pragma once
#include "brl_compat.h"
#include_next <fcntl.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC _O_NOINHERIT
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif
#ifndef O_NOATIME
#define O_NOATIME 0
#endif

/* posix_fadvise: advisory, and the Windows cache manager takes no per-range hints, so it is
 * accepted and ignored (0), which an advisory call may do. */
#define POSIX_FADV_NORMAL     0
#define POSIX_FADV_RANDOM     1
#define POSIX_FADV_SEQUENTIAL 2
#define POSIX_FADV_WILLNEED   3
#define POSIX_FADV_DONTNEED   4
#define POSIX_FADV_NOREUSE    5
BRL_C_BEGIN
int posix_fadvise(int fd, long long off, long long len, int advice);
BRL_C_END
