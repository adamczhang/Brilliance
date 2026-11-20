/* table_warm.cpp -- see table_warm.h. */
#include "table_warm.h"

#include "../rad_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <thread>
#include <vector>

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

int table_warm_read(const RadFile& file, const std::vector<std::pair<int64_t, int64_t>>& ranges,
                    int64_t* bytes, double* secs, std::string* why) {
    if (!bytes || !secs || !why) return RAD_E_INVAL;
    *bytes = 0;
    *secs = 0;
    const auto t0 = std::chrono::steady_clock::now();
    /* Readers on contiguous spans, as table_copy_fill reads: a drive serves several queued reads
     * at full rate, and each reader stays sequential. Eight, not sixteen: nothing waits on this
     * read but the start, and the buffers are held only while it runs. */
    const int nt = (int)std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    const int64_t piece = (int64_t)16 << 20;
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
                    try {
                        std::vector<uint8_t> buf((size_t)std::min(piece, hi - lo));
                        for (int64_t o = lo; o < hi && err.load() == RAD_OK; o += piece) {
                            const int64_t b = std::min(piece, hi - o);
                            const int rc = file.read_at(buf.data(), b, off + o);
                            if (rc < 0) { err.store(rc); return; }
                            done.fetch_add(b);
                        }
                    } catch (const std::exception&) {
                        err.store(RAD_E_NOMEM);   /* the scratch buffer */
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
