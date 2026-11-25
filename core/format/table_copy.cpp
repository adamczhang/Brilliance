/* table_copy.cpp -- see table_copy.h. */
#include "table_copy.h"

#include "../rad_internal.h"
#if defined(_WIN32)
#include "os/os.h"   /* Brilliance: os::File and the replacing rename (win/os) */
#endif

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <thread>
#include <vector>

namespace rad {

int table_copy_check(const uint8_t* src, int64_t n, const uint8_t* buf, std::string* why) {
    if (!why || !src || !buf || n <= 0) return RAD_E_INVAL;
    const int64_t probe = std::min<int64_t>(n, 4096);
    bool same = std::memcmp(buf, src, (size_t)probe) == 0 &&
                std::memcmp(buf + n - probe, src + n - probe, (size_t)probe) == 0;
    for (int64_t o = 0; same && o + 4096 <= n; o += std::max<int64_t>(n / 64, 4096))
        same = std::memcmp(buf + o, src + o, 4096) == 0;
    if (!same) {
        *why = "the copy does not match the container's mapping";
        return RAD_E_IO;
    }
    return RAD_OK;
}

/* The table's readers: `readers` threads (no more than the table has 64 MiB pieces), each reading
 * its own contiguous span in 64 MiB pieces and dropping the container's pages behind it. A drive
 * serves the queued reads at full rate, and spans keep each reader sequential. */
static void table_readers(const RadFile& file, int64_t foff, uint8_t* buf, int64_t n, int readers,
                          std::atomic<int>& err) {
    const int64_t piece = (int64_t)64 << 20;
    const int64_t huge = (int64_t)2 << 20;
    const int64_t pieces = (n + piece - 1) / piece;
    const int nt = (int)std::min<int64_t>(
        std::max(1, (int)std::min<uint32_t>((uint32_t)readers, std::thread::hardware_concurrency())),
        std::max<int64_t>(1, pieces));
    const int64_t span = ((n + nt - 1) / nt + huge - 1) & ~(huge - 1);
    std::vector<std::thread> rd;
    for (int i = 0; i < nt; ++i) {
        const int64_t lo = (int64_t)i * span, hi = std::min(n, lo + span);
        if (lo >= hi) break;
        rd.emplace_back([&, lo, hi] {
            for (int64_t o = lo; o < hi && err.load() == RAD_OK; o += piece) {
                const int64_t b = std::min(piece, hi - o);
                const int rc = file.read_at(buf + o, b, foff + o);
                if (rc < 0) { err.store(rc); return; }
                file.drop_range(foff + o, b);
            }
        });
    }
    for (std::thread& t : rd) t.join();
}

struct TableFill::State {
    std::atomic<int>      err{RAD_OK};
    std::atomic<uint64_t> began{0}, ended{0};
};

TableFill::~TableFill() {
    if (th_.joinable()) th_.join();
}

TableFill& TableFill::operator=(TableFill&& o) noexcept {
    if (this == &o) return *this;
    if (th_.joinable()) th_.join();
    st_ = std::move(o.st_);
    src_ = o.src_;
    dst_ = o.dst_;
    n_ = o.n_;
    th_ = std::move(o.th_);
    o.src_ = nullptr;
    o.dst_ = nullptr;
    o.n_ = 0;
    return *this;
}

int TableFill::start(const RadFile& file, const uint8_t* src, int64_t n, int64_t foff, uint8_t* dst,
                     int readers) {
    if (!src || !dst || n <= 0 || readers < 1) return RAD_E_INVAL;
    if (th_.joinable()) return RAD_E_STATE;
    src_ = src;
    dst_ = dst;
    n_ = n;
    st_ = std::make_shared<State>();
    const std::shared_ptr<State> st = st_;
    th_ = std::thread([&file, foff, buf = dst, n, readers, st] {
        st->began.store(rad_mono_ns());
        table_readers(file, foff, buf, n, readers, st->err);
        st->ended.store(rad_mono_ns());
    });
    return RAD_OK;
}

int TableFill::wait(std::string* why) {
    if (!why) return RAD_E_INVAL;
    if (!st_ || !src_ || !dst_ || n_ <= 0) return RAD_E_STATE;
    if (th_.joinable()) th_.join();
    const int err = st_->err.load();
    if (err != RAD_OK) {
        *why = fmt("reading the container failed (%s)", rad_strerror(err));
        return RAD_E_IO;
    }
    return table_copy_check(src_, n_, dst_, why);
}

double TableFill::seconds() const {
    if (!st_) return 0;
    const uint64_t began = st_->began.load(), ended = st_->ended.load();
    return ended > began ? (double)(ended - began) / 1e9 : 0;
}

int table_copy_fill(const RadFile& file, const uint8_t* src, int64_t n, int64_t foff,
                    uint8_t* buf, std::string* why) {
    if (!why || !src || !buf || n <= 0) return RAD_E_INVAL;
    TableFill fill;
    const int rc = fill.start(file, src, n, foff, buf, 16);
    if (rc != RAD_OK) {
        *why = "the table's readers could not be started";
        return rc;
    }
    return fill.wait(why);
}

int table_copy_reserve(int64_t n, TableCopy* out, std::string* why) {
    if (!out || !why || n <= 0) return RAD_E_INVAL;
    *out = TableCopy{};
    const size_t huge = (size_t)2 << 20;
    const size_t sz = ((size_t)n + huge - 1) & ~(huge - 1);
    void* m = ::mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                     -1, 0);
    if (m == MAP_FAILED) {
        *why = fmt("mmap of %s: %s", humanb((int64_t)sz).c_str(), std::strerror(errno));
        return RAD_E_NOMEM;
    }
    (void)::madvise(m, sz, MADV_HUGEPAGE);
    if (::mlock2(m, sz, MLOCK_ONFAULT) != 0) {
        *why = fmt("mlock2 of %s: %s", humanb((int64_t)sz).c_str(), std::strerror(errno));
        ::munmap(m, sz);
        return RAD_E_NOMEM;
    }
    out->base = m;
    out->bytes = sz;
    return RAD_OK;
}

int table_copy_make(const RadFile& file, const uint8_t* src, int64_t n, int64_t foff,
                    TableCopy* out, std::string* why) {
    if (!out || !why || !src || n <= 0) return RAD_E_INVAL;
    TableCopy tc;
    int rc = table_copy_reserve(n, &tc, why);
    if (rc != RAD_OK) return rc;
    rc = table_copy_fill(file, src, n, foff, static_cast<uint8_t*>(tc.base), why);
    if (rc != RAD_OK) {
        table_copy_free(&tc);
        return rc;
    }
    *out = tc;
    return RAD_OK;
}

int table_file_open(const RadFile& file, const uint8_t* src, int64_t n, int64_t foff,
                    const std::string& path, TableCopy* out, bool* filled, std::string* why) {
    if (!out || !filled || !why || !src || n <= 0 || path.empty()) return RAD_E_INVAL;
    *out = TableCopy{};
    *filled = false;
    struct stat st;
#if defined(_WIN32)
    /* Brilliance: Windows maps no file writable here (win/compat's mmap refuses it, see
     * sys/mman.h), so the copy is written, not stored through a mapping: the same bytes, read from
     * the container in pieces and written at their offsets, then flushed and put in place with a
     * replacing rename. The read-only mapping below is the same on both. */
    if (::stat(path.c_str(), &st) != 0 || st.st_size != n) {
        const std::string tmp = path + ".part";
        int rc = RAD_OK;
        {
            const os::File f = os::File::open(tmp, os::File::Write | os::File::Create | os::File::Trunc);
            if (!f.valid()) {
                *why = fmt("creating %s: %s", tmp.c_str(), std::strerror(errno));
                return RAD_E_IO;
            }
            const int64_t piece = (int64_t)64 << 20;
            std::vector<uint8_t> buf((size_t)std::min(n, piece));
            for (int64_t o = 0; o < n && rc == RAD_OK; o += piece) {
                const int64_t b = std::min(piece, n - o);
                rc = file.read_at(buf.data(), b, foff + o);
                if (rc < 0) { *why = fmt("reading the container failed (%s)", rad_strerror(rc)); rc = RAD_E_IO; break; }
                if (std::memcmp(buf.data(), src + o, (size_t)b) != 0) {
                    *why = "the copy does not match the container's mapping";
                    rc = RAD_E_IO;
                    break;
                }
                if (f.write_exact(buf.data(), b, o) != 0) {
                    *why = fmt("writing %s: %s", tmp.c_str(), std::strerror(errno));
                    rc = RAD_E_IO;
                    break;
                }
                file.drop_range(foff + o, b);
            }
            if (rc == RAD_OK && f.sync() != 0) {
                *why = fmt("writing %s: %s", tmp.c_str(), std::strerror(errno));
                rc = RAD_E_IO;
            }
        }
        if (rc == RAD_OK && os::rename_replace(tmp, path) != 0) {
            *why = fmt("renaming %s: %s", tmp.c_str(), std::strerror(errno));
            rc = RAD_E_IO;
        }
        if (rc != RAD_OK) {
            ::unlink(tmp.c_str());
            return rc;
        }
        *filled = true;
    }
#else
    if (::stat(path.c_str(), &st) != 0 || st.st_size != n) {
        const std::string tmp = path + ".part";
        const int fd = ::open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0) {
            *why = fmt("creating %s: %s", tmp.c_str(), std::strerror(errno));
            return RAD_E_IO;
        }
        void* m = MAP_FAILED;
        if (::ftruncate(fd, (off_t)n) == 0)
            m = ::mmap(nullptr, (size_t)n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) {
            *why = fmt("sizing %s to %s: %s", tmp.c_str(), humanb(n).c_str(), std::strerror(errno));
            ::close(fd);
            ::unlink(tmp.c_str());
            return RAD_E_IO;
        }
        int rc = table_copy_fill(file, src, n, foff, static_cast<uint8_t*>(m), why);
        if (rc == RAD_OK && (::msync(m, (size_t)n, MS_SYNC) != 0 || ::fsync(fd) != 0)) {
            *why = fmt("writing %s: %s", tmp.c_str(), std::strerror(errno));
            rc = RAD_E_IO;
        }
        ::munmap(m, (size_t)n);
        ::close(fd);
        if (rc == RAD_OK && ::rename(tmp.c_str(), path.c_str()) != 0) {
            *why = fmt("renaming %s: %s", tmp.c_str(), std::strerror(errno));
            rc = RAD_E_IO;
        }
        if (rc != RAD_OK) {
            ::unlink(tmp.c_str());
            return rc;
        }
        *filled = true;
    }
#endif
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        *why = fmt("opening %s: %s", path.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    void* m = ::mmap(nullptr, (size_t)n, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) {
        *why = fmt("mapping %s: %s", path.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    if (!*filled) {
        const int rc = table_copy_check(src, n, static_cast<const uint8_t*>(m), why);
        if (rc != RAD_OK) {
            ::munmap(m, (size_t)n);
            *why = fmt("%s does not match the container (%s); remove it and it is filled again",
                       path.c_str(), why->c_str());
            return rc;
        }
    }
    *out = TableCopy{ m, (size_t)n };
    return RAD_OK;
}

void table_copy_free(TableCopy* c) {
    if (!c || !c->base) return;
    ::munlock(c->base, c->bytes);
    ::munmap(c->base, c->bytes);
    *c = TableCopy{};
}

}  /* namespace rad */
