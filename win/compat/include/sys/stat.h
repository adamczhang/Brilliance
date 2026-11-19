/* Brilliance POSIX layer: <sys/stat.h> with 64-bit sizes.
 *
 * The UCRT's `struct stat` holds the size in 32 bits, and a model container is 114 GiB. `stat`
 * and `fstat` are therefore redirected to the _stat64 family: `struct stat st; stat(p, &st)`
 * becomes `struct _stat64 st; _stat64(p, &st)`, so every existing call site reads a 64-bit size
 * without being edited. This is safe in this tree because `stat` is never used as an ordinary
 * identifier (checked when the layer was written).
 *
 * st_dev and st_ino carry no file identity on Windows (st_ino is 0), so "is this the same file"
 * must not be asked of them: such sites use os::same_file. */
#pragma once
#include "brl_compat.h"
#include_next <sys/stat.h>
#include <direct.h>

#define stat  _stat64
#define fstat _fstat64

#ifndef S_ISDIR
#define S_ISDIR(m)  (((m) & _S_IFMT) == _S_IFDIR)
#define S_ISREG(m)  (((m) & _S_IFMT) == _S_IFREG)
#define S_ISCHR(m)  (((m) & _S_IFMT) == _S_IFCHR)
#define S_ISFIFO(m) (((m) & _S_IFMT) == _S_IFIFO)
#define S_ISLNK(m)  (0)
#endif
#ifndef S_IRUSR
#define S_IRUSR _S_IREAD
#define S_IWUSR _S_IWRITE
#define S_IXUSR 0
#define S_IRWXU (S_IRUSR | S_IWUSR)
#define S_IRGRP 0
#define S_IWGRP 0
#define S_IROTH 0
#define S_IWOTH 0
#endif

#ifdef __cplusplus
/* POSIX mkdir takes a mode; the permission bits have no Windows meaning (a private directory is
 * made private by os::make_dir with an owner-only DACL, not here). */
inline int mkdir(const char* path, int /*mode*/) { return _mkdir(path); }
#endif
