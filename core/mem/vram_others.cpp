/* vram_others.cpp -- Brilliance: the VRAM other processes hold on the serving cards (vram_others.h). */
#include "mem/vram_others.h"

#include "rad_internal.h"

#include <algorithm>
#include <map>

#if defined(_WIN32)
#include "os/os.h"   /* os::gpu_vram_holders (win/os/gpu_win32.cpp) */
#endif

namespace rad {

std::vector<VramHolders> read_vram_holders(const std::vector<std::string>& bdfs) {
    std::vector<VramHolders> out(bdfs.size());
#if defined(_WIN32)
    std::vector<std::string> ask;
    for (const std::string& b : bdfs)
        if (!b.empty()) ask.push_back(b);
    if (ask.empty()) return out;
    const std::vector<os::GpuHolders> got = os::gpu_vram_holders(ask);
    for (size_t i = 0; i < bdfs.size(); ++i)
        for (const os::GpuHolders& g : got)
            if (!bdfs[i].empty() && g.bdf == bdfs[i] && g.known) {
                out[i].known = true;
                for (const os::GpuProcVram& p : g.procs)
                    out[i].procs.push_back(VramHolder{ p.pid, p.name, p.bytes });
            }
#endif
    return out;
}

int64_t vram_holders_total(const VramHolders& h) {
    int64_t t = 0;
    for (const VramHolder& p : h.procs) t += std::max<int64_t>(p.bytes, 0);
    return t;
}

static std::string who(const VramHolder& p) {
    return fmt("%s (pid %u)", p.name.empty() ? "a process" : p.name.c_str(), (unsigned)p.pid);
}

std::string vram_holders_text(const VramHolders& h, size_t most) {
    if (!h.known) return "unknown";
    std::vector<VramHolder> v;
    for (const VramHolder& p : h.procs)
        if (p.bytes > 0) v.push_back(p);
    if (v.empty()) return "none";
    std::stable_sort(v.begin(), v.end(),
                     [](const VramHolder& a, const VramHolder& b) { return a.bytes > b.bytes; });
    std::string s;
    int64_t rest = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i < most) {
            if (!s.empty()) s += ", ";
            s += who(v[i]) + " " + humanb(v[i].bytes);
        } else {
            rest += v[i].bytes;
        }
    }
    if (v.size() > most)
        s += fmt(", and %zu more (%s)", v.size() - most, humanb(rest).c_str());
    return s;
}

std::vector<VramGrowth> vram_holders_grown(const VramHolders& then, const VramHolders& now) {
    std::vector<VramGrowth> g;
    if (!then.known || !now.known) return g;
    std::map<uint32_t, int64_t> before;
    for (const VramHolder& p : then.procs) before[p.pid] += std::max<int64_t>(p.bytes, 0);
    for (const VramHolder& p : now.procs) {
        const auto it = before.find(p.pid);
        const int64_t d = p.bytes - (it == before.end() ? 0 : it->second);
        if (d > 0) g.push_back(VramGrowth{ p, d });
    }
    std::stable_sort(g.begin(), g.end(), [](const VramGrowth& a, const VramGrowth& b) { return a.grew > b.grew; });
    return g;
}

int64_t vram_growth_total(const std::vector<VramGrowth>& g) {
    int64_t t = 0;
    for (const VramGrowth& x : g) t += x.grew;
    return t;
}

VramSettleVerdict vram_settle_verdict(int card, const std::string& bdf, int64_t room, int64_t allowance,
                                      int64_t taken, int64_t need_head_mib, const VramHolders& at_read,
                                      const VramHolders& now, bool per_card) {
    VramSettleVerdict v;
    const std::vector<VramGrowth> g = vram_holders_grown(at_read, now);
    v.grew = vram_growth_total(g);
    /* Short, and mostly because of them: their growth is at least half of what went missing after
     * the budget read. A shortfall they did not cause (a driver or plugin allocation the budget
     * does not count) is not theirs to be named for, and keeps the settle check's warning. */
    if (room >= allowance || v.grew <= 0 || 2 * v.grew < taken) return v;
    v.refuse = true;
    std::string names;
    for (const VramGrowth& x : g) {
        if (!names.empty()) names += ", ";
        names += fmt("%s holds %s now, %s more", who(x.now).c_str(), humanb(x.now.bytes).c_str(),
                     humanb(x.grew).c_str());
    }
    v.message = fmt(
        "card %d%s: other processes took %s more of its VRAM after the budget read the card, which "
        "leaves %s for a kernel's first launch to load its code object and size its scratch in -- "
        "under the %s this engine needs: %s. Startup stops here rather than fail on a later "
        "allocation. Close %s, or set %s graphics preference to another GPU (Windows Settings > "
        "System > Display > Graphics) and restart %s; or %s",
        card, bdf.empty() ? "" : fmt(" (%s)", bdf.c_str()).c_str(), humanb(v.grew).c_str(),
        humanb(std::max<int64_t>(room, 0)).c_str(), humanb(allowance).c_str(), names.c_str(),
        g.size() == 1 ? "it" : "them", g.size() == 1 ? "its" : "their", g.size() == 1 ? "it" : "them",
        per_card
            ? fmt("set memory.card-headroom-mib to {%d: %lld} to leave other processes that much more "
                  "room on this card alone, while the other cards keep theirs", card,
                  (long long)need_head_mib).c_str()
            : fmt("set --gpu-headroom-mib to at least %lld to leave other processes that much more "
                  "room", (long long)need_head_mib).c_str());
    return v;
}

std::string vram_holders_report(const std::vector<int>& cards, const std::vector<std::string>& bdfs,
                                const std::vector<VramHolders>& h, const std::vector<int64_t>& headroom_mib,
                                const std::vector<bool>& named) {
    bool any_known = false, any_held = false;
    for (const VramHolders& x : h) {
        any_known = any_known || x.known;
        any_held = any_held || (x.known && vram_holders_total(x) > 0);
    }
    if (!any_known) return "";
    std::string s;
    for (size_t i = 0; i < h.size(); ++i) {
        if (!s.empty()) s += "; ";
        s += fmt("card %d", i < cards.size() ? cards[i] : (int)i);
        if (i < bdfs.size() && !bdfs[i].empty()) s += " (" + bdfs[i] + ")";
        const int64_t t = vram_holders_total(h[i]);
        s += h[i].known && t > 0 ? fmt(" %s: %s", humanb(t).c_str(), vram_holders_text(h[i]).c_str())
                                 : " " + vram_holders_text(h[i]);
    }
    if (!any_held)
        return "VRAM other processes hold on the serving cards: none (" + s + ")";
    /* Each card's reserve, and where it comes from. One figure when every card keeps the same one
     * from --gpu-headroom-mib, as before. */
    auto reserve = [&](size_t i) { return i < headroom_mib.size() ? headroom_mib[i] : (int64_t)0; };
    auto is_named = [&](size_t i) { return i < named.size() && named[i]; };
    bool uniform = true;
    for (size_t i = 0; i < h.size(); ++i)
        uniform = uniform && !is_named(i) && reserve(i) == reserve(0);
    std::string out;
    if (uniform) {
        out = fmt("VRAM other processes hold on the serving cards: %s. The budget counts it as already "
                  "held; what they take after this reading comes out of the %lld MiB "
                  "--gpu-headroom-mib, and startup stops naming them if that leaves a card short",
                  s.c_str(), (long long)reserve(0));
    } else {
        std::string per;
        for (size_t i = 0; i < h.size(); ++i) {
            if (!per.empty()) per += ", ";
            per += fmt("card %d %lld MiB%s", i < cards.size() ? cards[i] : (int)i, (long long)reserve(i),
                       is_named(i) ? " (memory.card-headroom-mib)" : "");
        }
        out = fmt("VRAM other processes hold on the serving cards: %s. The budget counts it as already "
                  "held; what they take after this reading comes out of each card's reserve (%s), "
                  "and startup stops naming them if that leaves a card short",
                  s.c_str(), per.c_str());
    }
    /* Brilliance (issue #1): on a start of several cards, one card that others use (the one that
     * drives the desktop) can keep more in reserve alone; raising --gpu-headroom-mib would cost
     * every card the same. Said only while no card has its own reserve yet. */
    bool any_named = false;
    for (size_t i = 0; i < h.size(); ++i) any_named = any_named || is_named(i);
    if (h.size() > 1 && !any_named)
        out += ". A card that drives a display or runs other GPU applications can keep more free "
               "on its own with memory.card-headroom-mib, {CARD: MiB}, while the others keep "
               "--gpu-headroom-mib";
    return out;
}

}  // namespace rad
