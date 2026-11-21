/* table_warm.cpp -- see table_warm.h. */
#include "table_warm.h"

#include "../rad_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include "os/os.h"   /* Brilliance: os::File, os::prefetch (win/os) */
#else
#include <sys/mman.h>
#endif

namespace rad {

NgramWarmPlan ngram_warm_plan(const NgramWarmFacts& f) {
    NgramWarmPlan p;
    if (!f.asked) {
        p.why = "off (ngram-warm not set)";
        return p;
    }
    if (f.table_bytes <= 0) {
        p.why = "this model has no n-gram table read from the container";
        return p;
    }
    if (f.resolved != TablePlacement::Disk) {
        p.why = fmt("the n-gram table is %s, not read from the drive, so there is nothing to warm",
                    f.resolved == TablePlacement::Ram ? "a copy in RAM" : "on the card");
        return p;
    }
    if (f.direct_reads) {
        p.why = "the gather reads the table's rows with direct I/O through its own row cache, "
                "which the page cache does not serve";
        return p;
    }
    if (f.own_file) {
        p.why = "the table is read from its own file (ngram-placement: disk=DIR), which the warm "
                "does not read";
        return p;
    }
    const int64_t floor = std::max<int64_t>(f.floor, 0);
    if (f.avail < 0) {
        p.why = "the host's available memory cannot be read, so the table's room cannot be checked";
        return p;
    }
    if (f.avail < f.table_bytes + floor) {
        p.why = fmt("%s of host memory is available, short of the table's %s and %s to leave free",
                    humanb(f.avail).c_str(), humanb(f.table_bytes).c_str(), humanb(floor).c_str());
        return p;
    }
    p.warm = true;
    p.why = fmt("the table (%s) is read from the drive and %s of host memory is available, at "
                "least its size and %s more", humanb(f.table_bytes).c_str(),
                humanb(f.avail).c_str(), humanb(floor).c_str());
    return p;
}

std::vector<std::pair<int64_t, int64_t>> ngram_warm_ranges(std::vector<std::pair<int64_t, int64_t>> r) {
    std::vector<std::pair<int64_t, int64_t>> k;   /* as [begin, end) while merging */
    k.reserve(r.size());
    for (const auto& x : r)
        if (x.first >= 0 && x.second > 0) k.emplace_back(x.first, x.first + x.second);
    std::sort(k.begin(), k.end());
    size_t m = 0;
    for (size_t i = 0; i < k.size(); ++i) {
        if (m && k[i].first <= k[m - 1].second) k[m - 1].second = std::max(k[m - 1].second, k[i].second);
        else k[m++] = k[i];
    }
    k.resize(m);
    for (auto& x : k) x.second -= x.first;        /* back to (offset, length) */
    return k;
}

TableWarmOpts table_warm_opts_alpha3() {
    TableWarmOpts o;
    o.io = WarmIo::Pread;
    o.readers = 8;
    o.piece = (int64_t)16 << 20;
    return o;
}

/* "64", "64M", "64MiB", "1G": bytes. */
static bool warm_size(const std::string& v, int64_t* out) {
    size_t i = 0;
    int64_t n = 0;
    for (; i < v.size() && v[i] >= '0' && v[i] <= '9'; ++i) {
        if (n > ((int64_t)1 << 40)) return false;
        n = n * 10 + (v[i] - '0');
    }
    if (i == 0) return false;
    int sh = 0;
    if (i < v.size()) {
        const char u = v[i];
        if (u == 'k' || u == 'K') sh = 10;
        else if (u == 'm' || u == 'M') sh = 20;
        else if (u == 'g' || u == 'G') sh = 30;
        else return false;
        ++i;
        if (i < v.size() && (v[i] == 'i' || v[i] == 'I')) ++i;
        if (i < v.size() && (v[i] == 'b' || v[i] == 'B')) ++i;
        if (i != v.size()) return false;
    }
    *out = n << sh;
    return true;
}

int table_warm_opts_parse(const std::string& spec, TableWarmOpts* o, std::string* why) {
    if (!o || !why) return RAD_E_INVAL;
    TableWarmOpts t = *o;
    size_t p = 0;
    while (p < spec.size()) {
        size_t q = spec.find_first_of(", ", p);
        if (q == std::string::npos) q = spec.size();
        const std::string kv = spec.substr(p, q - p);
        p = q + 1;
        if (kv.empty()) continue;
        const size_t eq = kv.find('=');
        const std::string k = kv.substr(0, eq);
        const std::string v = eq == std::string::npos ? std::string() : kv.substr(eq + 1);
        if (k == "io") {
            if (v == "pread") t.io = WarmIo::Pread;
            else if (v == "handle") t.io = WarmIo::Handle;
            else if (v == "prefetch") t.io = WarmIo::Prefetch;
            else {
                *why = fmt("io must be pread, handle or prefetch, not '%s'", v.c_str());
                return RAD_E_INVAL;
            }
        } else if (k == "readers") {
            int64_t n = 0;
            if (!warm_size(v, &n) || n < 1 || n > 64) {
                *why = fmt("readers must be 1 to 64, not '%s'", v.c_str());
                return RAD_E_INVAL;
            }
            t.readers = (int)n;
        } else if (k == "piece") {
            int64_t n = 0;
            if (!warm_size(v, &n) || n < ((int64_t)1 << 20) || n > ((int64_t)1 << 30)) {
                *why = fmt("piece must be 1M to 1G, not '%s'", v.c_str());
                return RAD_E_INVAL;
            }
            t.piece = n;
        } else {
            *why = fmt("unknown key '%s' (io, readers, piece)", k.c_str());
            return RAD_E_INVAL;
        }
    }
    *o = t;
    return RAD_OK;
}

std::string table_warm_opts_str(const TableWarmOpts& o) {
    const char* io = o.io == WarmIo::Pread ? "pread" : o.io == WarmIo::Handle ? "handle" : "prefetch";
    return fmt("io=%s readers=%d piece=%s", io, o.readers, humanb(o.piece).c_str());
}

/* One reader's span [lo, hi) of the container, in `o.piece` steps. */
static int warm_span(const RadFile& file, const TableWarmOpts& o, int64_t lo, int64_t hi,
                     const std::atomic<int>& err, std::atomic<int64_t>& done) {
    if (o.io == WarmIo::Prefetch) {
        const uint8_t* base = file.base();
        if (!base) return RAD_E_STATE;
        for (int64_t at = lo; at < hi && err.load() == RAD_OK; at += o.piece) {
            const int64_t b = std::min(o.piece, hi - at);
#if defined(_WIN32)
            os::prefetch(base + at, b);
#else
            /* A page-aligned start, as madvise requires. */
            const uintptr_t a = (uintptr_t)(base + at) & ~(uintptr_t)4095;
            posix_madvise((void*)a, (size_t)((uintptr_t)(base + at + b) - a), POSIX_MADV_WILLNEED);
#endif
            done.fetch_add(b);
        }
        return RAD_OK;
    }
    std::vector<uint8_t> buf((size_t)std::min(o.piece, hi - lo));
#if defined(_WIN32)
    if (o.io == WarmIo::Handle) {
        /* No access hint: read-ahead follows this file object along the span. */
        const os::File f = os::File::open(file.path(), os::File::Read);
        if (!f.valid()) return RAD_E_IO;
        for (int64_t at = lo; at < hi && err.load() == RAD_OK; at += o.piece) {
            const int64_t b = std::min(o.piece, hi - at);
            if (f.read_at(buf.data(), b, at) != b) return RAD_E_IO;
            done.fetch_add(b);
        }
        return RAD_OK;
    }
#endif
    /* pread, and `handle` where every read already shares one file object (POSIX). */
    for (int64_t at = lo; at < hi && err.load() == RAD_OK; at += o.piece) {
        const int64_t b = std::min(o.piece, hi - at);
        const int rc = file.read_at(buf.data(), b, at);
        if (rc < 0) return rc;
        done.fetch_add(b);
    }
    return RAD_OK;
}

int table_warm_read(const RadFile& file, const std::vector<std::pair<int64_t, int64_t>>& ranges,
                    int64_t* bytes, double* secs, std::string* why, const TableWarmOpts& o) {
    if (!bytes || !secs || !why) return RAD_E_INVAL;
    *bytes = 0;
    *secs = 0;
    const auto t0 = std::chrono::steady_clock::now();
    /* Readers on contiguous spans, as table_copy_fill reads: a drive serves several queued reads
     * at full rate, and each reader stays sequential. Nothing waits on this read but the start,
     * and the buffers are held only while it runs. */
    const unsigned want = (unsigned)std::max(o.readers, 1);
    const int nt = (int)std::max(1u, std::min(want, std::thread::hardware_concurrency()));
    const int64_t align = (int64_t)2 << 20;
    std::atomic<int> err{RAD_OK};
    std::atomic<int64_t> done{0};
    for (const auto& rg : ranges) {
        const int64_t off = rg.first, n = rg.second;
        if (n <= 0) continue;
        const int64_t span = ((n + nt - 1) / nt + align - 1) & ~(align - 1);
        std::vector<std::thread> rd;
        try {
            for (int i = 0; i < nt; ++i) {
                const int64_t lo = (int64_t)i * span, hi = std::min(n, lo + span);
                if (lo >= hi) break;
                rd.emplace_back([&, lo, hi] {
                    int rc = RAD_OK;
                    try {
                        rc = warm_span(file, o, off + lo, off + hi, err, done);
                    } catch (const std::exception&) {
                        rc = RAD_E_NOMEM;   /* the scratch buffer */
                    }
                    if (rc < 0) {
                        int ok = RAD_OK;
                        err.compare_exchange_strong(ok, rc);
                    }
                });
            }
        } catch (const std::exception& e) {
            err.store(RAD_E_STATE);
            *why = fmt("starting a reader: %s", e.what());
        }
        for (std::thread& t : rd) t.join();
        if (err.load() != RAD_OK) break;
    }
    *bytes = done.load();
    *secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (err.load() != RAD_OK) {
        if (why->empty()) *why = fmt("reading the container failed (%s)", rad_strerror(err.load()));
        return err.load();
    }
    return RAD_OK;
}

}  /* namespace rad */
