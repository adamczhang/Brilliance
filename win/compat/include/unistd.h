/* Brilliance POSIX layer: <unistd.h>, the parts whose meaning carries over (see brl_compat.h).
 *
 * The UCRT already declares the POSIX names close, read, write, dup, dup2, isatty, access,
 * unlink, rmdir and getpid in <io.h>/<process.h>/<direct.h>; this header pulls those in and adds
 * the rest. pread, pwrite, ftruncate and lseek take a 64-bit off_t (brl_prelude.h makes off_t 64
 * bits for user code), and lseek is redirected so the UCRT's 32-bit one is never reached.
 * readlink answers exactly one question, "/proc/self/exe", because that is the one radiance asks;
 * there are no symlinks to read otherwise. symlink makes the nearest thing an unprivileged process
 * may: a symbolic link where Developer Mode allows one, else a junction for a directory and a hard
 * link for a file (same volume) -- what radiance's tests link to stand in for a plugin home.
 *
 * A BAD DESCRIPTOR IS -1 WITH EBADF from every function here, as POSIX has it, where the UCRT's
 * own end the process (0xC0000417).
 *
 * READ AND WRITE MOVE AT MOST 1 GiB A CALL and return ssize_t. The UCRT's take an unsigned int
 * count: a size_t above 4 GiB is cut to its low 32 bits without a word, one above INT_MAX ends the
 * process, and the int they return cannot hold a count above 2 GiB. A macro cannot redirect them
 * -- `read` and `write` are member names everywhere (std::ostream::write, Sink::read) -- so C++
 * gets overloads that any integer count other than unsigned int prefers to the UCRT's (an
 * unsigned int count cannot pass 4 GiB). close stays the UCRT's for the same reason, and because
 * an exact match on (int) cannot be outbid. */
#pragma once
#include "brl_compat.h"

#include <io.h>
#include <process.h>
#include <direct.h>
#include <stdlib.h>

#ifndef F_OK
#define F_OK 0
#endif
#ifndef W_OK
#define W_OK 2
#endif
#ifndef R_OK
#define R_OK 4
#endif
#ifndef X_OK
#define X_OK 0      /* the UCRT's access() rejects 1; "exists" is the closest honest answer */
#endif

#ifndef STDIN_FILENO
#define STDIN_FILENO  0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#endif

#define _SC_PAGESIZE          1
#define _SC_PAGE_SIZE         _SC_PAGESIZE
#define _SC_PHYS_PAGES        2
#define _SC_AVPHYS_PAGES      3
#define _SC_NPROCESSORS_ONLN  4
#define _SC_NPROCESSORS_CONF  5

BRL_C_BEGIN
long      brl_sysconf(int name);
int       brl_usleep(unsigned long long usec);
int       brl_fsync(int fd);
ssize_t   pread(int fd, void* buf, size_t n, long long off);
ssize_t   pwrite(int fd, const void* buf, size_t n, long long off);
ssize_t   brl_read(int fd, void* buf, size_t n);
ssize_t   brl_write(int fd, const void* buf, size_t n);
/* Growing a file makes it sparse, as ftruncate does on Linux (win/compat/src/compat_misc.cpp). */
int       ftruncate(int fd, long long len);
long long brl_lseek64(int fd, long long off, int whence);
ssize_t   readlink(const char* path, char* buf, size_t n);
int       symlink(const char* target, const char* linkpath);
BRL_C_END

#ifdef __cplusplus
#include <type_traits>
extern "C++" {
template <class N, typename std::enable_if<std::is_integral<N>::value, int>::type = 0>
inline ssize_t read(int fd, void* buf, N n) { return brl_read(fd, buf, (size_t)n); }
template <class N, typename std::enable_if<std::is_integral<N>::value, int>::type = 0>
inline ssize_t write(int fd, const void* buf, N n) { return brl_write(fd, buf, (size_t)n); }
}
#endif

#define sysconf brl_sysconf
#define usleep  brl_usleep
#define fsync   brl_fsync
#define lseek   brl_lseek64
