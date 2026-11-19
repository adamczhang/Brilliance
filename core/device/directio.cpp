/* directio.cpp -- see directio.h for why the page cache is not wanted here. */
#include "directio.h"
#include "../rad_internal.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace rad {
namespace {

/* Bounce buffer size; allocated on first misaligned use only. */
constexpr int64_t kBounceBytes = 1 << 20;

int64_t align_down(int64_t v, int64_t a) { return v / a * a; }

#if defined(_WIN32)
/* Brilliance: the same three primitives over an os::File, ReadFile/WriteFile at 64-bit offsets.
 * EINTR does not exist there, and os::File already moves any length in pieces. */
int64_t pread_some(const os::File& f, void* dst, int64_t bytes, int64_t off) {
    const int64_t got = f.read_at(dst, bytes, off);
    return got < 0 ? (int64_t)RAD_E_IO : got;
}
int pread_exact(const os::File& f, void* dst, int64_t bytes, int64_t off) {
    if (f.read_exact(dst, bytes, off) != 0) return RAD_E_IO;
    dio_progress.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
    return RAD_OK;
}
int pwrite_exact(const os::File& f, const void* src, int64_t bytes, int64_t off) {
    return f.write_exact(src, bytes, off) == 0 ? RAD_OK : RAD_E_IO;
}
#else

/* Reads up to `bytes`, stopping at end of file. Returns the byte count, or a negative error. */
int64_t pread_some(int fd, void* dst, int64_t bytes, int64_t off) {
    int64_t done = 0;
    while (done < bytes) {
        ssize_t r = ::pread(fd, (uint8_t*)dst + done, (size_t)(bytes - done), (off_t)(off + done));
        if (r < 0) {
            if (errno == EINTR) continue;
            return RAD_E_IO;
        }
        if (r == 0) break;
        done += r;
    }
    return done;
}

/* Fill `bytes` exactly. A short pread inside the file is a torn read, not a smaller one: end of
 * file inside a range the caller said was there is an error. */
int pread_exact(int fd, void* dst, int64_t bytes, int64_t off) {
    const int64_t got = pread_some(fd, dst, bytes, off);
    if (got < 0) return (int)got;
    if (got != bytes) return RAD_E_IO;
    dio_progress.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
    return RAD_OK;
}

int pwrite_exact(int fd, const void* src, int64_t bytes, int64_t off) {
    int64_t done = 0;
    while (done < bytes) {
        ssize_t w = ::pwrite(fd, (const uint8_t*)src + done, (size_t)(bytes - done),
                             (off_t)(off + done));
        if (w <= 0) {
            if (w < 0 && errno == EINTR) continue;
            return RAD_E_IO;
        }
        done += w;
    }
    return RAD_OK;
}
#endif  /* !_WIN32 */

}  /* namespace */

DirectFile::~DirectFile() { close(); }

int DirectFile::open(const char* path, int flags) {
    if (!path || !*path) return RAD_E_INVAL;
    close();

#if defined(_WIN32)
    /* Brilliance: FILE_FLAG_NO_BUFFERING is O_DIRECT's Windows twin, with the same rules --
     * sector-aligned offset, length and buffer -- which alignment() (4096) meets for every sector
     * size in use. Opened read-write whenever writing, for the read-modify-write in write_at. The
     * 0600 of the POSIX open is an owner-only DACL: the tiers hold other clients' prompts. */
    unsigned of = os::File::Read | os::File::Private;
    if (flags & RAD_DIO_WRITE)  of |= os::File::Write;
    if (flags & RAD_DIO_CREATE) of |= os::File::Create;
    if (flags & RAD_DIO_TRUNC)  of |= os::File::Trunc | os::File::Create;
    f_ = os::File::open(path, of | os::File::Direct);
    direct_ = f_.valid();
    if (!f_.valid() && (errno == EINVAL || errno == ENOTSUP)) {
        if (flags & RAD_DIO_ALLOW_BUFFERED) {
            f_ = os::File::open(path, of);
            direct_ = false;
            if (f_.valid())
                RAD_WARN("directio: %s does not support unbuffered I/O; opened buffered. Reads will "
                         "go through the file cache, which is a second copy of every byte.", path);
        } else {
            RAD_ERR("directio: cannot open %s unbuffered: %s. The tier directory must be on a "
                    "volume that supports it.", path, std::strerror(errno));
            return RAD_E_IO;
        }
    }
    if (!f_.valid()) {
        RAD_ERR("directio: cannot open %s: %s", path, std::strerror(errno));
        return RAD_E_IO;
    }
    fd_ = 0;
    return RAD_OK;
#else
    const bool want_write = (flags & RAD_DIO_WRITE) != 0;
    /* O_RDWR whenever writing: a partial-block write reads the block first (see write_at). */
    int oflags = want_write ? O_RDWR : O_RDONLY;
    if (flags & RAD_DIO_CREATE) oflags |= O_CREAT;
    if (flags & RAD_DIO_TRUNC)  oflags |= O_TRUNC;

    fd_ = ::open(path, oflags | O_DIRECT, 0600);
    direct_ = fd_ >= 0;

    if (fd_ < 0 && (errno == EINVAL || errno == ENOTSUP || errno == EOPNOTSUPP)) {
        /* tmpfs and some network mounts refuse O_DIRECT outright. */
        if (flags & RAD_DIO_ALLOW_BUFFERED) {
            fd_ = ::open(path, oflags, 0600);
            direct_ = false;
            if (fd_ >= 0)
                RAD_WARN("directio: %s does not support O_DIRECT; opened buffered. Reads will go "
                         "through the page cache, which is a second copy of every byte and evicts "
                         "the weight pool's own pages.", path);
        } else {
            RAD_ERR("directio: cannot open %s with O_DIRECT: %s. The tier directory must be on a "
                    "filesystem that supports it.", path, std::strerror(errno));
            return RAD_E_IO;
        }
    }
    if (fd_ < 0) {
        RAD_ERR("directio: cannot open %s: %s", path, std::strerror(errno));
        return RAD_E_IO;
    }
    return RAD_OK;
#endif  /* !_WIN32 */
}

void DirectFile::close() {
#if defined(_WIN32)
    /* Brilliance: the handle, and a bounce from os::aligned_alloc, which free() cannot take. */
    f_.close();
    fd_ = -1;
    direct_ = false;
    os::aligned_free(bounce_);
    bounce_ = nullptr;
    bounce_bytes_ = 0;
#else
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    direct_ = false;
    std::free(bounce_);
    bounce_ = nullptr;
    bounce_bytes_ = 0;
#endif
}

int64_t DirectFile::size() const {
    if (fd_ < 0) return -1;
#if defined(_WIN32)
    return f_.size();   /* Brilliance */
#endif
    struct stat st{};
    if (::fstat(fd_, &st) != 0) return -1;
    return (int64_t)st.st_size;
}

int DirectFile::bounce_ready() {
    if (bounce_) return RAD_OK;
    void* p = nullptr;
#if defined(_WIN32)
    p = os::aligned_alloc((size_t)alignment(), (size_t)kBounceBytes);   /* Brilliance */
    if (!p) return RAD_E_NOMEM;
#else
    if (posix_memalign(&p, (size_t)alignment(), (size_t)kBounceBytes) != 0 || !p)
        return RAD_E_NOMEM;
#endif
    bounce_ = (uint8_t*)p;
    bounce_bytes_ = kBounceBytes;
    return RAD_OK;
}

/* ------------------------------------------------------------------ read */

int DirectFile::read_at(void* dst, int64_t off, int64_t bytes) {
    if (fd_ < 0) return RAD_E_STATE;
    if (!dst || off < 0 || bytes < 0) return RAD_E_INVAL;
    if (bytes == 0) return RAD_OK;

    const int64_t A = alignment();
    const bool aligned = (off % A) == 0 && (bytes % A) == 0 &&
                         ((uintptr_t)dst % (uintptr_t)A) == 0;

    /* Fast path: one pread straight into the caller's pinned buffer. Every .rad extent lands here
     * (movement units and rad_dev_alloc pools are both RAD_ALIGN_UNIT-aligned). */
    if (aligned) return pread_exact(io(), dst, bytes, off);

    /* O_DIRECT rejects unaligned transfers with EINVAL, so read an aligned superset through the
     * bounce and copy out. This serves file tails and header/footer reads, never the mover; a
     * step-path read landing here indicates a layout bug. */
    RAD_TRY(bounce_ready());

    const int64_t lo = align_down(off, A);
    const int64_t hi = align_up(off + bytes, A);
    for (int64_t base = lo; base < hi; base += bounce_bytes_) {
        const int64_t span = std::min(bounce_bytes_, hi - base);
        const int64_t s = std::max(off, base);
        const int64_t e = std::min(off + bytes, base + span);
        if (e <= s) continue;

        const int64_t got = pread_some(io(), bounce_, span, base);
        if (got < 0) return (int)got;
        /* A short read is fine only if it still covers the requested slice. */
        if (got < e - base) return RAD_E_IO;
        std::memcpy((uint8_t*)dst + (s - off), bounce_ + (s - base), (size_t)(e - s));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ write */

/* Aligned writes go straight through; anything else is a read-modify-write of whole blocks. */
int DirectFile::write_at(const void* src, int64_t off, int64_t bytes) {
    if (fd_ < 0) return RAD_E_STATE;
    if (!src || off < 0 || bytes < 0) return RAD_E_INVAL;
    if (bytes == 0) return RAD_OK;

    const int64_t A = alignment();
    const bool aligned = (off % A) == 0 && (bytes % A) == 0 &&
                         ((uintptr_t)src % (uintptr_t)A) == 0;
    if (aligned) return pwrite_exact(io(), src, bytes, off);

    RAD_TRY(bounce_ready());
    const int64_t old_size = size();
    if (old_size < 0) return RAD_E_IO;

    const int64_t lo = align_down(off, A);
    const int64_t hi = align_up(off + bytes, A);
    for (int64_t base = lo; base < hi; base += bounce_bytes_) {
        const int64_t span = std::min(bounce_bytes_, hi - base);
        /* Read-modify-write the whole chunk, including fully covered interior blocks. The extra
         * reads are deliberate: this path is for correctness at boundaries, not throughput. */
        const int64_t got = pread_some(io(), bounce_, span, base);
        if (got < 0) return (int)got;
        if (got < span) std::memset(bounce_ + got, 0, (size_t)(span - got));

        const int64_t s = std::max(off, base);
        const int64_t e = std::min(off + bytes, base + span);
        if (e > s)
            std::memcpy(bounce_ + (s - base), (const uint8_t*)src + (s - off), (size_t)(e - s));
        RAD_TRY(pwrite_exact(io(), bounce_, span, base));
    }

    /* The last chunk padded the file to a block boundary; truncate back to the real end. */
    const int64_t want = std::max(old_size, off + bytes);
#if defined(_WIN32)
    if (size() > want && f_.set_size(want) != 0) return RAD_E_IO;   /* Brilliance: the handle's own */
#else
    if (size() > want && ::ftruncate(fd_, (off_t)want) != 0) return RAD_E_IO;
#endif
    return RAD_OK;
}

}  /* namespace rad */
