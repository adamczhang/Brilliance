/* Brilliance POSIX layer: <sys/statvfs.h>, the free-space question.
 *
 * radiance asks statvfs how much room a directory's volume has (the disk tier, the park files).
 * GetDiskFreeSpaceExW answers the same question: the bytes available to this caller, which honours
 * quotas as f_bavail does, and the volume's total. They are reported in 1-byte "blocks"
 * (f_frsize = f_bsize = 1), so every product a caller forms (f_bavail * f_frsize) is the byte count
 * whatever block size it assumed. The file-count and flag fields have no Windows meaning and read
 * 0. */
#pragma once
#include "brl_compat.h"

typedef unsigned long long fsblkcnt_t;
typedef unsigned long long fsfilcnt_t;

struct statvfs {
    unsigned long f_bsize;
    unsigned long f_frsize;
    fsblkcnt_t    f_blocks;
    fsblkcnt_t    f_bfree;
    fsblkcnt_t    f_bavail;
    fsfilcnt_t    f_files;
    fsfilcnt_t    f_ffree;
    fsfilcnt_t    f_favail;
    unsigned long f_fsid;
    unsigned long f_flag;
    unsigned long f_namemax;
};

BRL_C_BEGIN
/* `path` is UTF-8 and may name a file or a directory on the volume. 0, or -1 with errno. */
int statvfs(const char* path, struct statvfs* out);
BRL_C_END
