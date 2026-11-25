/* Brilliance POSIX layer: <sys/mman.h>, for the uses whose meaning carries over exactly.
 *
 * SUPPORTED, because radiance's mappings are of these kinds and nothing else:
 *   - a read-only view of a file descriptor (PROT_READ, MAP_PRIVATE or MAP_SHARED), whole or from
 *     an allocation-granularity-aligned offset: a container, a checkpoint, a GGUF, an imatrix;
 *   - anonymous read-write memory (MAP_ANONYMOUS), and anonymous PROT_NONE address space;
 *   - munmap of a WHOLE region previously returned by mmap;
 *   - madvise/posix_madvise as hints (WILLNEED prefetches; the rest have no Windows equivalent and
 *     are accepted and ignored, which is what an advisory call may do);
 *   - mlock/mlock2/munlock over VirtualLock, raising the working set first when the lock needs it
 *     and munlock lowering it again by what that mlock raised;
 *   - memfd_create as a file of its own to read, write and hand to a child (r4d_selftest's per-row
 *     output buffers): a temporary file deleted when its last handle closes. Its descriptor is not
 *     inherited (MFD_CLOEXEC is what Windows does anyway); it is not meant to be mapped writable.
 *
 * REFUSED with EINVAL, so a use is a visible failure rather than a quiet divergence: MAP_FIXED,
 * writable file mappings, a partial munmap (a length that does not cover exactly the pages mmap
 * returned), mprotect beyond toggling an anonymous region's access.
 * Code that needs those (the host backend's elastic memory) uses core/os/os.h's vmem_* instead.
 *
 * Windows commits anonymous memory at mmap time (MAP_NORESERVE has no equivalent): the commit
 * charge is taken against RAM plus page file at once, never on first touch. */
#pragma once
#include "brl_compat.h"

#define PROT_NONE  0x0
#define PROT_READ  0x1
#define PROT_WRITE 0x2
#define PROT_EXEC  0x4

#define MAP_SHARED    0x01
#define MAP_PRIVATE   0x02
#define MAP_FIXED     0x10
#define MAP_ANONYMOUS 0x20
#define MAP_ANON      MAP_ANONYMOUS
#define MAP_NORESERVE 0x4000
#define MAP_POPULATE  0x8000
#define MAP_HUGETLB   0x40000
#define MAP_FAILED    ((void*)-1)

#define MADV_NORMAL     0
#define MADV_RANDOM     1
#define MADV_SEQUENTIAL 2
#define MADV_WILLNEED   3
#define MADV_DONTNEED   4
#define MADV_DONTFORK   10
#define MADV_DOFORK     11
#define MADV_HUGEPAGE   14
#define MADV_NOHUGEPAGE 15
#define POSIX_MADV_NORMAL     MADV_NORMAL
#define POSIX_MADV_RANDOM     MADV_RANDOM
#define POSIX_MADV_SEQUENTIAL MADV_SEQUENTIAL
#define POSIX_MADV_WILLNEED   MADV_WILLNEED
#define POSIX_MADV_DONTNEED   MADV_DONTNEED

#define MLOCK_ONFAULT 0x01

#define MFD_CLOEXEC 0x0001U

BRL_C_BEGIN
void* mmap(void* addr, size_t len, int prot, int flags, int fd, long long off);
int   munmap(void* addr, size_t len);
int   mprotect(void* addr, size_t len, int prot);
int   madvise(void* addr, size_t len, int advice);
int   posix_madvise(void* addr, size_t len, int advice);
int   mlock(const void* addr, size_t len);
int   mlock2(const void* addr, size_t len, unsigned flags);
int   munlock(const void* addr, size_t len);
int   memfd_create(const char* name, unsigned flags);

/* Brilliance's own, not POSIX: many ranges of mapped memory read in with one PrefetchVirtualMemory
 * call, where Linux code would madvise(MADV_WILLNEED) each (libavx avx_ngram.cpp: the disk-tier
 * n-gram gather's rows and its read-ahead's). The call reads the ranges' pages in parallel and
 * returns once they are in memory, but does not map them into the process: each page's first touch
 * is still a soft fault. Here, not in win/os, because plugins built outside the tree link this
 * layer only. brl_prefetch_range has WIN32_MEMORY_RANGE_ENTRY's layout (checked in
 * compat_mman.cpp), so the array is passed as is. 0, also for n == 0; or -1 with errno and, when
 * `why` is given, the reason in up to `cap` bytes. Adapted from wmmunn's rad_compat_prefetch. */
typedef struct {
    void*  addr;
    size_t bytes;
} brl_prefetch_range;
int   brl_prefetch_ranges(const brl_prefetch_range* ranges, size_t n, char* why, size_t cap);
BRL_C_END
