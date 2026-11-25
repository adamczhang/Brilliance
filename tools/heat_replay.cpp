/* heat_replay -- replays a RADIANCE_ROUTE_TRACE file (core/place/rtrace.h) through the heat engine
 * (core/place/heat.cpp, unchanged) and a model of the mover, for expert streaming X0.2
 * (kernels/experts-streaming.md §7.2).
 *
 *   rad-heat-replay TRACE [--lat N] [--decay X] [--hysteresis X] [--min-gain X] [--moves N]
 *                         [--credit-cap X] [--presence] [--resident N] [--layer-min N]
 *                         [--from D] [--to D] [--oracle]
 *
 * The model: one slab class, the trace's first tier snapshot as the starting resident set and its
 * size as the slab (the live run's lent-slot fills and drops are not modelled, so its resident count
 * drifts; the snapshots are printed beside it). Each dispatch credits its rows as the live engine
 * did, asks heat.step() for swaps (the move budget is the only limit; no slot starvation) and
 * publishes each swap `--lat` passes after the dispatch that proposed it: 2 is today's order (copy
 * behind pass d+1, publish behind it, first read by pass d+2), 1 is X1 (a) (published before pass
 * d+1 is played). A row's misses are counted against the residency in force for its pass (its
 * hist_step). `--oracle` admits every unit routed in pass p into VRAM for pass p+1 at no cost
 * (unbounded, the bound of "copy after the first miss").
 *
 * Prints, per run: dispatches, routed and missed units a dispatch (the replay's, against the live
 * engine's own count at credit time), promotions a dispatch, readmits, link MB a dispatch (streamed
 * misses and promotion copies), and the residents a layer at the start and at the end (min..max,
 * median, sd over the layers that hold a movable unit). With --resident the slab is cut or grown to
 * N units first (coldest = highest expert index), a capacity sweep. With --layer-min N no layer is
 * left under N resident units: the starting set is raised to the highest floor this slab can hold
 * on every layer, never above N (the units come from the layers above it, the fullest first,
 * highest expert index first, so the total does not move, and a floor the slab cannot hold is
 * clamped rather than overshot), and a swap whose demotion would take a layer under that floor is
 * abandoned (heat.abandon), the model of a mover that keeps a per-layer floor (M-16, E-M6). */
#include "place/heat.h"
#include "place/planner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <vector>

using namespace rad;

namespace {

struct Row { int32_t dispatch, hist; int16_t layer, reps; int32_t live_missed; std::vector<uint16_t> ec; };
struct Disp { int32_t dispatch, swaps; uint64_t routed, routed_res, promotions, demotions, early, h2d, paced; };

struct Trace {
    int32_t nl = 0, ne = 0;
    std::vector<int64_t> bytes;
    std::vector<uint8_t> movable;
    std::vector<std::pair<int32_t, std::vector<uint8_t>>> snaps;
    std::vector<Row> rows;
    std::vector<Disp> disp;
};

template <class T> bool get(FILE* f, T* v) { return std::fread(v, sizeof *v, 1, f) == 1; }

bool load(const char* path, Trace* t) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::perror(path); return false; }
    char magic[8];
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "RADRTRC1", 8)) {
        std::fprintf(stderr, "%s: not a route trace\n", path);
        return false;
    }
    get(f, &t->nl);
    get(f, &t->ne);
    const size_t n = (size_t)t->nl * (size_t)t->ne;
    t->bytes.resize(n);
    t->movable.resize(n);
    if (std::fread(t->bytes.data(), 8, n, f) != n || std::fread(t->movable.data(), 1, n, f) != n) return false;
    for (;;) {
        const int c = std::fgetc(f);
        if (c == EOF) break;
        if (c == 'T') {
            int32_t d;
            if (!get(f, &d)) break;
            std::vector<uint8_t> tier(n);
            if (std::fread(tier.data(), 1, n, f) != n) break;
            t->snaps.emplace_back(d, std::move(tier));
        } else if (c == 'R') {
            Row r;
            int32_t k;
            if (!get(f, &r.dispatch) || !get(f, &r.hist) || !get(f, &r.layer) || !get(f, &r.reps) ||
                !get(f, &r.live_missed) || !get(f, &k))
                break;
            r.ec.resize((size_t)k * 2);
            if (std::fread(r.ec.data(), 2, r.ec.size(), f) != r.ec.size()) break;
            t->rows.push_back(std::move(r));
        } else if (c == 'D') {
            Disp d;
            if (!get(f, &d.dispatch) || !get(f, &d.swaps) || !get(f, &d.routed) || !get(f, &d.routed_res) ||
                !get(f, &d.promotions) || !get(f, &d.demotions) || !get(f, &d.early) || !get(f, &d.h2d) ||
                !get(f, &d.paced))
                break;
            t->disp.push_back(d);
        } else {
            std::fprintf(stderr, "%s: bad record '%c' after %zu rows (truncated trace?)\n", path, c,
                         t->rows.size());
            break;
        }
    }
    std::fclose(f);
    return true;
}

struct Opt {
    int lat = 2;
    HeatParams hp{};
    bool presence = false, oracle = false;
    int64_t resident = -1, layer_min = 0;
    int32_t from = INT32_MIN, to = INT32_MAX;
};

/* The residents a layer over the layers that hold a movable unit: min..max (median, sd), the
 * exit report's per-layer table as one line (M-16). */
std::string layer_line(const Trace& t, const std::vector<int64_t>& res) {
    std::vector<int64_t> r;
    for (int32_t l = 0; l < t.nl; ++l) {
        bool has = false;
        for (int32_t e = 0; e < t.ne && !has; ++e) {
            const size_t k = (size_t)l * (size_t)t.ne + e;
            has = t.bytes[k] > 0 && t.movable[k];
        }
        if (has) r.push_back(res[(size_t)l]);
    }
    if (r.empty()) return "no movable layer";
    std::sort(r.begin(), r.end());
    double mean = 0;
    for (int64_t x : r) mean += (double)x;
    mean /= (double)r.size();
    double var = 0;
    for (int64_t x : r) var += ((double)x - mean) * ((double)x - mean);
    var /= (double)r.size();
    return fmt("%lld..%lld (median %lld, sd %.0f)", (long long)r.front(), (long long)r.back(),
               (long long)r[r.size() / 2], std::sqrt(var));
}

}  /* namespace */

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s TRACE [--lat N] [--decay X] [--hysteresis X] [--min-gain X] "
                     "[--moves N] [--credit-cap X] [--presence] [--resident N] [--layer-min N] "
                     "[--from D] [--to D] [--oracle]\n",
                     argv[0]);
        return 2;
    }
    Opt o;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--lat") o.lat = std::atoi(val());
        else if (a == "--decay") o.hp.decay = std::strtof(val(), nullptr);
        else if (a == "--hysteresis") o.hp.hysteresis = std::strtof(val(), nullptr);
        else if (a == "--min-gain") o.hp.min_gain = std::strtof(val(), nullptr);
        else if (a == "--moves") o.hp.moves_per_step = std::atoi(val());
        else if (a == "--credit-cap") o.hp.credit_cap = std::strtof(val(), nullptr);
        else if (a == "--presence") o.presence = true;
        else if (a == "--oracle") o.oracle = true;
        else if (a == "--resident") o.resident = std::atoll(val());
        else if (a == "--layer-min") o.layer_min = std::atoll(val());
        else if (a == "--from") o.from = std::atoi(val());
        else if (a == "--to") o.to = std::atoi(val());
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    Trace t;
    if (!load(argv[1], &t)) return 1;
    if (t.snaps.empty() || t.rows.empty()) { std::fprintf(stderr, "empty trace\n"); return 1; }
    const size_t n = (size_t)t.nl * (size_t)t.ne;

    /* The starting tiers: the last snapshot at or before --from (the first otherwise). */
    size_t si = 0;
    for (size_t i = 0; i < t.snaps.size(); ++i) if (t.snaps[i].first <= o.from) si = i;
    std::vector<uint8_t> tier = t.snaps[si].second;
    int64_t res0 = 0;
    for (size_t k = 0; k < n; ++k) res0 += t.bytes[k] > 0 && tier[k] == (uint8_t)Tier::VRAM;
    if (o.resident >= 0) {
        /* Cut or grow by expert index, round-robin over layers, movable units only. */
        int64_t have = res0;
        for (int32_t e = t.ne - 1; e >= 0 && have > o.resident; --e)
            for (int32_t l = 0; l < t.nl && have > o.resident; ++l) {
                const size_t k = (size_t)l * t.ne + e;
                if (t.bytes[k] > 0 && t.movable[k] && tier[k] == (uint8_t)Tier::VRAM) { tier[k] = (uint8_t)Tier::HostPinned; --have; }
            }
        for (int32_t e = 0; e < t.ne && have < o.resident; ++e)
            for (int32_t l = 0; l < t.nl && have < o.resident; ++l) {
                const size_t k = (size_t)l * t.ne + e;
                if (t.bytes[k] > 0 && t.movable[k] && tier[k] != (uint8_t)Tier::VRAM) { tier[k] = (uint8_t)Tier::VRAM; ++have; }
            }
        res0 = have;
    }

    /* --layer-min: no layer under N resident units at any dispatch, at the same total. Every layer
     * is raised to the highest floor this slab can hold, never above N (units come off the layers
     * above it, the fullest first, the highest expert index first), so what the floor costs is the
     * hit rate of the residents it moves, not a bigger slab. `floor_achieved` is that floor: a trace
     * whose slab cannot hold N everywhere is clamped to what it reaches, not overshot. */
    std::vector<int64_t> layer_res((size_t)t.nl, 0);
    for (int32_t l = 0; l < t.nl; ++l)
        for (int32_t e = 0; e < t.ne; ++e) {
            const size_t k = (size_t)l * (size_t)t.ne + e;
            layer_res[(size_t)l] += t.bytes[k] > 0 && t.movable[k] && tier[k] == (uint8_t)Tier::VRAM;
        }
    int64_t floor_achieved = 0;
    if (o.layer_min > 0) {
        /* The highest floor this slab can hold on every layer: raising the thin layers to F is paid
         * for by the layers above F, so F is feasible while sum of max(0, res - F) >= sum of
         * max(0, F - res), which is monotone in F. Bisect for it, never above the request: a floor
         * the slab cannot hold is clamped to the one it can, not overshot and not left frozen. */
        auto feasible = [&](int64_t f) {
            int64_t supply = 0, demand = 0;
            for (int32_t l = 0; l < t.nl; ++l) {
                bool has = false;
                for (int32_t e = 0; e < t.ne && !has; ++e) {
                    const size_t k = (size_t)l * (size_t)t.ne + e;
                    has = t.bytes[k] > 0 && t.movable[k];
                }
                if (!has) continue;
                const int64_t v = layer_res[(size_t)l];
                if (v > f) supply += v - f;
                else demand += f - v;
            }
            return supply >= demand;
        };
        for (int64_t lo = 0, hi = o.layer_min; lo < hi;) {
            const int64_t mid = lo + (hi - lo + 1) / 2;
            if (feasible(mid)) lo = mid;
            else hi = mid - 1;
            floor_achieved = lo;
        }
        for (int32_t l = 0; l < t.nl; ++l) {
            while (layer_res[(size_t)l] < floor_achieved) {
                int32_t e_in = -1;
                for (int32_t e = 0; e < t.ne && e_in < 0; ++e) {
                    const size_t k = (size_t)l * (size_t)t.ne + e;
                    if (t.bytes[k] > 0 && t.movable[k] && tier[k] != (uint8_t)Tier::VRAM) e_in = e;
                }
                if (e_in < 0) break;
                int32_t l_out = -1;
                for (int32_t l2 = 0; l2 < t.nl; ++l2)
                    if (layer_res[(size_t)l2] > floor_achieved &&
                        (l_out < 0 || layer_res[(size_t)l2] > layer_res[(size_t)l_out]))
                        l_out = l2;
                if (l_out < 0) break;
                int32_t e_out = -1;
                for (int32_t e = t.ne - 1; e >= 0 && e_out < 0; --e) {
                    const size_t k = (size_t)l_out * (size_t)t.ne + e;
                    if (t.bytes[k] > 0 && t.movable[k] && tier[k] == (uint8_t)Tier::VRAM) e_out = e;
                }
                if (e_out < 0) break;
                tier[(size_t)l * (size_t)t.ne + e_in] = (uint8_t)Tier::VRAM;
                tier[(size_t)l_out * (size_t)t.ne + e_out] = (uint8_t)Tier::HostPinned;
                ++layer_res[(size_t)l];
                --layer_res[(size_t)l_out];
            }
        }
    }
    const std::string layer_start = layer_line(t, layer_res);
    /* What the floor binds: the requested one, or the one this slab can hold on every layer. */
    const int64_t floor_ = floor_achieved;

    /* A plan of one unit per (layer, expert), one slab class. */
    Plan plan;
    plan.dynamic = true;
    std::vector<int32_t> unit_of(n, -1);
    for (size_t k = 0; k < n; ++k) {
        if (t.bytes[k] <= 0) continue;
        MoveUnit mu;
        mu.layer = (int32_t)(k / (size_t)t.ne);
        mu.expert = (int32_t)(k % (size_t)t.ne);
        mu.slab_class = 0;
        mu.bytes = t.bytes[k];
        mu.movable = t.movable[k] != 0;
        mu.tier = (Tier)tier[k] == Tier::VRAM ? Tier::VRAM : Tier::HostPinned;
        unit_of[k] = (int32_t)plan.units.size();
        plan.units.push_back(mu);
    }
    HeatEngine heat;
    if (heat.init(plan, o.hp) < 0 || !heat.enabled()) { std::fprintf(stderr, "heat init failed\n"); return 1; }

    struct Pub { int32_t pass; Swap s; };
    std::deque<Pub> pend;
    /* The floor's bookkeeping: the committed residents a layer and the delta of the swaps accepted
     * but not yet published, so a batch of swaps cannot walk a layer under N between two passes. */
    std::vector<int64_t> layer_pend((size_t)t.nl, 0);
    uint64_t floor_refused = 0;
    std::vector<int32_t> dense((size_t)t.ne);
    uint64_t routed = 0, missed = 0, live_missed = 0, rows_n = 0, promos = 0, miss_bytes = 0, promo_bytes = 0;
    uint64_t oracle_adm = 0;
    std::vector<uint8_t> ever_demoted(plan.units.size(), 0);
    uint64_t readmits = 0;
    std::map<int32_t, std::vector<size_t>> by_disp;
    for (size_t i = 0; i < t.rows.size(); ++i)
        if (t.rows[i].dispatch >= o.from && t.rows[i].dispatch <= o.to) by_disp[t.rows[i].dispatch].push_back(i);
    std::vector<uint8_t> vram(plan.units.size());
    for (size_t u = 0; u < plan.units.size(); ++u) vram[u] = plan.units[u].tier == Tier::VRAM;
    /* --oracle: units admitted for the pass after the one that routed them. */
    std::vector<int32_t> oracle_until(plan.units.size(), INT32_MIN);
    int32_t last_pass = INT32_MIN;
    auto publish_to = [&](int32_t pass) {
        while (!pend.empty() && pend.front().pass <= pass) {
            const Swap s = pend.front().s;
            pend.pop_front();
            if (s.promote >= 0) {
                heat.set_tier(s.promote, Tier::VRAM);
                heat.set_busy(s.promote, false);
                vram[(size_t)s.promote] = 1;
                if (ever_demoted[(size_t)s.promote]) ++readmits;
                if (plan.units[(size_t)s.promote].movable) {
                    const size_t l = (size_t)plan.units[(size_t)s.promote].layer;
                    ++layer_res[l];
                    --layer_pend[l];
                }
            }
            if (s.demote >= 0) {
                heat.set_tier(s.demote, Tier::HostPinned);
                heat.set_busy(s.demote, false);
                vram[(size_t)s.demote] = 0;
                ever_demoted[(size_t)s.demote] = 1;
                if (plan.units[(size_t)s.demote].movable) {
                    const size_t l = (size_t)plan.units[(size_t)s.demote].layer;
                    --layer_res[l];
                    ++layer_pend[l];
                }
            }
        }
    };
    int64_t n_disp = 0;
    std::vector<Swap> swaps;
    for (auto& [d, idx] : by_disp) {
        ++n_disp;
        for (size_t i : idx) {
            const Row& r = t.rows[i];
            if (r.hist > last_pass) last_pass = r.hist;
            publish_to(r.hist);
            std::fill(dense.begin(), dense.end(), 0);
            for (size_t j = 0; j + 1 < r.ec.size(); j += 2) {
                const uint16_t e = r.ec[j], c = r.ec[j + 1];
                if (e >= t.ne) continue;
                dense[e] = o.presence ? 1 : c;
                const int32_t u = unit_of[(size_t)r.layer * t.ne + e];
                if (u < 0) continue;
                const uint64_t w = (uint64_t)std::max<int>(1, r.reps);
                routed += w;
                const bool hit = vram[(size_t)u] || (o.oracle && oracle_until[(size_t)u] >= r.hist);
                if (!hit) { missed += w; miss_bytes += w * (uint64_t)plan.units[(size_t)u].bytes; }
                if (o.oracle && !vram[(size_t)u]) {
                    if (oracle_until[(size_t)u] < r.hist) ++oracle_adm;
                    oracle_until[(size_t)u] = r.hist + 1;
                }
            }
            live_missed += (uint64_t)r.live_missed;
            ++rows_n;
            heat.observe(r.layer, dense.data(), t.ne, r.reps > 1 ? (float)r.reps : 1.0f);
        }
        swaps.clear();
        heat.step(&swaps);
        for (const Swap& s : swaps) {
            if (s.demote >= 0 && floor_ > 0) {
                const size_t l = (size_t)plan.units[(size_t)s.demote].layer;
                if (layer_res[l] + layer_pend[l] <= floor_) {
                    heat.abandon(s);
                    ++floor_refused;
                    continue;
                }
            }
            pend.push_back(Pub{ d + o.lat, s });
            if (s.promote >= 0) {
                ++promos;
                promo_bytes += (uint64_t)plan.units[(size_t)s.promote].bytes;
                if (plan.units[(size_t)s.promote].movable)
                    ++layer_pend[(size_t)plan.units[(size_t)s.promote].layer];
            }
            if (s.demote >= 0 && plan.units[(size_t)s.demote].movable)
                --layer_pend[(size_t)plan.units[(size_t)s.demote].layer];
        }
    }
    publish_to(INT32_MAX);
    const HeatStats hs = heat.stats();
    /* The live run's own counters over the same dispatches. */
    const Disp* d0 = nullptr;
    const Disp* d1 = nullptr;
    for (const Disp& d : t.disp) {
        if (d.dispatch < o.from || d.dispatch > o.to) continue;
        if (!d0) d0 = &d;
        d1 = &d;
    }
    const double nd = (double)std::max<int64_t>(1, n_disp);
    std::printf("trace %s: %d layers x %d experts, %zu rows, %lld dispatches (%d..%d), start resident %lld units\n",
                argv[1], t.nl, t.ne, (size_t)rows_n, (long long)n_disp, by_disp.empty() ? 0 : by_disp.begin()->first,
                by_disp.empty() ? 0 : by_disp.rbegin()->first, (long long)res0);
    std::printf("params: lat %d decay %.3f hysteresis %.2f min-gain %.2f moves %d credit-cap %.0f%s%s\n", o.lat,
                (double)o.hp.decay, (double)o.hp.hysteresis, (double)o.hp.min_gain, o.hp.moves_per_step,
                (double)o.hp.credit_cap, o.presence ? " presence" : "", o.oracle ? " ORACLE" : "");
    std::printf("residents a layer: start %s, end %s\n", layer_start.c_str(),
                layer_line(t, layer_res).c_str());
    if (o.layer_min > 0)
        std::printf("floor: %lld units a layer (achieved %lld at the start, enforced), %llu swaps "
                    "abandoned on it\n",
                    (long long)o.layer_min, (long long)floor_, (unsigned long long)floor_refused);
    std::printf("replay: routed %.1f missed %.1f (hit %.4f; heat's own count %.4f) promotions %.2f "
                "readmits %.1f%% link MB a dispatch: stream %.1f promote %.1f\n",
                routed / nd, missed / nd, routed ? 1.0 - (double)missed / routed : 0.0,
                hs.routed ? (double)hs.routed_resident / hs.routed : 0.0, promos / nd,
                promos ? 100.0 * readmits / promos : 0.0, miss_bytes / nd / 1e6, promo_bytes / nd / 1e6);
    if (o.oracle) std::printf("oracle admissions %.1f a dispatch (%.1f MB)\n", oracle_adm / nd,
                              oracle_adm / nd * (plan.units.empty() ? 0 : plan.units[0].bytes) / 1e6);
    if (d0 && d1 && d1 != d0) {
        const double ld = (double)std::max(1, d1->dispatch - d0->dispatch);
        std::printf("live:   routed %.1f missed %.1f (hit %.4f; rows' live_missed %.1f) promotions %.2f "
                    "early-published %.2f h2d MB %.1f paced %.1f  [over dispatches %d..%d]\n",
                    (d1->routed - d0->routed) / ld, ((d1->routed - d0->routed) - (d1->routed_res - d0->routed_res)) / ld,
                    d1->routed > d0->routed ? (double)(d1->routed_res - d0->routed_res) / (d1->routed - d0->routed) : 0.0,
                    live_missed / nd, (d1->promotions - d0->promotions) / ld, (d1->early - d0->early) / ld,
                    (d1->h2d - d0->h2d) / ld / 1e6, (d1->paced - d0->paced) / ld, d0->dispatch, d1->dispatch);
    }
    for (const auto& [d, tr] : t.snaps) {
        if (d < o.from || d > o.to) continue;
        int64_t r = 0;
        for (size_t k = 0; k < n; ++k) r += t.bytes[k] > 0 && tr[k] == (uint8_t)Tier::VRAM;
        std::printf("  live resident at dispatch %d: %lld units\n", d, (long long)r);
    }
    return 0;
}
