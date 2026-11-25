/* rtrace.h -- RADIANCE_ROUTE_TRACE=FILE: rank 0's routing, as the heat engine is credited with it,
 * for tools/heat_replay.cpp (expert streaming X0.1, kernels/experts-streaming.md §7).
 *
 * Written on the rank thread at Engine::tiering_dispatch, where heat.observe is called; nothing is
 * read back, nothing changes what runs. Little-endian binary, in order:
 *
 *   header  char magic[8] "RADRTRC1"; int32 n_layer, n_expert;
 *           int64 bytes[n_layer * n_expert]   (the unit's bytes on this rank, 0: no unit)
 *           uint8 movable[n_layer * n_expert]
 *   'T'     int32 dispatch; uint8 tier[n_layer * n_expert]   (rad::Tier, the heat engine's view)
 *           at the first dispatch and every kSnapEvery dispatches
 *   'R'     int32 dispatch, hist_step; int16 layer, reps; int32 live_missed, n;
 *           n x (uint16 expert, uint16 count)                  (one credited histogram)
 *   'D'     int32 dispatch, swaps; uint64 routed, routed_res, promotions, demotions,
 *           early_published, h2d_bytes, paced_units            (cumulative, after the dispatch)
 *
 * `live_missed` counts the row's routed experts the heat engine saw off the device when it credited
 * the row: the live run's definition of a miss, for the replay's validation. A few MB a minute at
 * eight clients. */
#pragma once
#include "heat.h"
#include "mover.h"
#include "planner.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace rad {

class RouteTrace {
public:
    static constexpr int kSnapEvery = 256;

    ~RouteTrace() { close(); }

    int open(const std::string& path, const Plan& plan) {
        f_ = std::fopen(path.c_str(), "wb");
        if (!f_) return RAD_E_IO;
        std::setvbuf(f_, nullptr, _IOFBF, 1 << 22);
        nl_ = ne_ = 0;
        for (const MoveUnit& mu : plan.units)
            if (mu.layer >= 0 && mu.expert >= 0) {
                nl_ = std::max(nl_, mu.layer + 1);
                ne_ = std::max(ne_, mu.expert + 1);
            }
        const size_t n = (size_t)nl_ * (size_t)ne_;
        unit_.assign(n, -1);
        std::vector<int64_t> bytes(n, 0);
        std::vector<uint8_t> mov(n, 0);
        for (size_t u = 0; u < plan.units.size(); ++u) {
            const MoveUnit& mu = plan.units[u];
            if (mu.layer < 0 || mu.expert < 0) continue;
            const size_t k = (size_t)mu.layer * (size_t)ne_ + (size_t)mu.expert;
            unit_[k] = (int32_t)u;
            bytes[k] = mu.bytes;
            mov[k] = mu.movable ? 1 : 0;
        }
        std::fwrite("RADRTRC1", 1, 8, f_);
        put(nl_);
        put(ne_);
        std::fwrite(bytes.data(), sizeof(int64_t), n, f_);
        std::fwrite(mov.data(), 1, n, f_);
        tiers_.resize(n);
        return RAD_OK;
    }
    bool on() const { return f_ != nullptr; }
    int n_expert() const { return ne_; }

    void row(int dispatch, int hist_step, int layer, int reps, const int32_t* counts, int32_t n_ex,
             const HeatEngine& heat) {
        if (!f_ || layer < 0 || layer >= nl_) return;
        if (dispatch != snapped_ && (snapped_ < 0 || dispatch - snapped_ >= kSnapEvery)) snap(dispatch, heat);
        pairs_.clear();
        int32_t missed = 0;
        const int32_t n = std::min(n_ex, ne_);
        for (int32_t e = 0; e < n; ++e) {
            if (counts[e] <= 0) continue;
            pairs_.push_back((uint16_t)e);
            pairs_.push_back((uint16_t)std::min(counts[e], 65535));
            const int32_t u = unit_[(size_t)layer * (size_t)ne_ + (size_t)e];
            if (u >= 0 && heat.tier(u) != Tier::VRAM) ++missed;
        }
        std::fputc('R', f_);
        put((int32_t)dispatch);
        put((int32_t)hist_step);
        put((int16_t)layer);
        put((int16_t)std::min(reps, 32767));
        put(missed);
        put((int32_t)(pairs_.size() / 2));
        std::fwrite(pairs_.data(), sizeof(uint16_t), pairs_.size(), f_);
    }

    void dispatch_end(int dispatch, int swaps, const HeatEngine& heat, const MoverStats& ms) {
        if (!f_) return;
        const HeatStats hs = heat.stats();
        std::fputc('D', f_);
        put((int32_t)dispatch);
        put((int32_t)swaps);
        put((uint64_t)hs.routed);
        put((uint64_t)hs.routed_resident);
        put((uint64_t)ms.promotions);
        put((uint64_t)ms.demotions);
        put((uint64_t)ms.early_published);
        put((uint64_t)ms.h2d_bytes);
        put((uint64_t)ms.paced_units);
    }

    void close() {
        if (f_) std::fclose(f_);
        f_ = nullptr;
    }

private:
    std::FILE* f_ = nullptr;
    int32_t nl_ = 0, ne_ = 0;
    int snapped_ = -1;
    std::vector<int32_t> unit_;
    std::vector<uint8_t> tiers_;
    std::vector<uint16_t> pairs_;

    template <class T> void put(T v) { std::fwrite(&v, sizeof v, 1, f_); }
    void snap(int dispatch, const HeatEngine& heat) {
        for (size_t k = 0; k < unit_.size(); ++k)
            tiers_[k] = unit_[k] >= 0 ? (uint8_t)heat.tier(unit_[k]) : (uint8_t)Tier::VRAM;
        std::fputc('T', f_);
        put((int32_t)dispatch);
        std::fwrite(tiers_.data(), 1, tiers_.size(), f_);
        snapped_ = dispatch;
    }
};

}  /* namespace rad */
