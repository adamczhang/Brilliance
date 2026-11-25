/* vram_others.h -- Brilliance: the VRAM other processes hold on the cards an engine serves from.
 *
 * THE CASE (2026-11-22). A desktop app with its GPU set to an R9700 (T3 Code, an Electron app, held
 * about 720 MiB on bus 32) took about 256 MiB more of that card while a four-card Flash-Next server
 * loaded. The budget had counted what the app held when it read the card (`already held`,
 * vram_budget.h), so the plan fitted; the growth came out of the --gpu-headroom-mib, the card
 * settled at 7 MiB free, and the start failed later on an allocation ("step batch staging: could
 * not reserve 12.43 MiB") that named nobody.
 *
 * WHAT IS DONE ABOUT IT (core/engine_bringup.cpp):
 *   - at the budget read (Engine::plan), every other process's use of each serving card is read
 *     and reported by name, pid and MiB, so the log says who shares the card;
 *   - at the settle check after every allocation, a card short of its first-launch allowance is
 *     read again. When other processes' growth since the budget read is what made it short (the
 *     card would not be short without it), the start stops there, naming each process that grew,
 *     instead of failing later on an out-of-memory error. A shortfall the others did not cause
 *     keeps the earlier warning.
 *
 * A holding at the budget read is not refused: the plan counted it, and two engines on one set of
 * cards (docs/GUIDE.md section 8) are a supported layout in which the second starts beside the
 * first's pools. Only what is taken after the reading can break a fitted plan, and the headroom is
 * the room the plan leaves for it.
 *
 * The reading is Windows' (os::gpu_vram_holders, win/os/gpu_win32.cpp: the graphics kernel's
 * per-process "Dedicated Usage" counters, readable without admin rights). Elsewhere it is not read,
 * and nothing here changes what the engine does. This file is the portable part: the text and the
 * decision, tested on the host (tests/vram_others_test.cpp). */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rad {

/* One other process's use of one card's own memory. */
struct VramHolder {
    uint32_t    pid = 0;
    std::string name;        /* the executable's file name, or "" when it could not be read */
    int64_t     bytes = 0;
};

/* Every other process's use of each card, in the order the cards were asked for. `known` is false
 * where it could not be read (no reading on this platform, an adapter the kernel does not report,
 * a counter that did not answer): unknown is not "none". */
struct VramHolders {
    bool                    known = false;
    std::vector<VramHolder> procs;
};

/* The cards' readings: Windows' counters (os::gpu_vram_holders), one per PCI address in `bdfs` (the
 * engine's spelling, dev_pci_address); every entry unknown elsewhere or for an empty address. */
std::vector<VramHolders> read_vram_holders(const std::vector<std::string>& bdfs);

int64_t vram_holders_total(const VramHolders& h);

/* "T3 Code (Nightly).exe (pid 19340) 704.00 MiB, red-linkhold.exe (pid 4242) 16.00 MiB": largest
 * first, at most `most` named, the rest as "and N more (X)". "none" for an empty known reading,
 * "unknown" for an unknown one. */
std::string vram_holders_text(const VramHolders& h, size_t most = 6);

/* One process's growth between two readings of a card: its use now, and how much more than at the
 * first reading (a process absent then grew by all of its use). */
struct VramGrowth {
    VramHolder now;
    int64_t    grew = 0;
};
/* The processes that hold more now than then, largest growth first. Empty when either reading is
 * unknown: growth cannot be told from an unknown reading. */
std::vector<VramGrowth> vram_holders_grown(const VramHolders& then, const VramHolders& now);
int64_t vram_growth_total(const std::vector<VramGrowth>& g);

/* The settle check's call for one card (engine_bringup.cpp). `room` is what the card has left for
 * first launches (free less what is kept for later); `allowance` the engine's own need for them;
 * `taken` what went missing after the budget read the card (what the budget kept, less what is
 * free now); `need_head_mib` the --gpu-headroom-mib that would have kept the card whole. Refuses
 * when the card is short (room < allowance) and other processes' growth since the budget read is at
 * least half of `taken`: they, not an allocation the budget does not count, are the main cause.
 * (Not "short without them": on 11-22 the card would have had its allowance to within a fraction of
 * a MiB, and the readings are not that exact.) The message names the card, each process that grew,
 * and what to do. No refusal leaves the message empty.
 *
 * `per_card`: the start serves on more than one card, so the advice names this card's own reserve
 * (memory.card-headroom-mib, {card: need_head_mib}) rather than raising --gpu-headroom-mib on every
 * card, which the plan would charge to every rank (Brilliance, issue #1). */
struct VramSettleVerdict {
    bool        refuse = false;
    int64_t     grew = 0;      /* the other processes' growth on this card, whatever the verdict */
    std::string message;
};
VramSettleVerdict vram_settle_verdict(int card, const std::string& bdf, int64_t room, int64_t allowance,
                                      int64_t taken, int64_t need_head_mib, const VramHolders& at_read,
                                      const VramHolders& now, bool per_card = false);

/* The budget read's report line for the serving cards (one entry a card, in rank order, with its
 * device index and address), or "" when no card's reading is known. `headroom_mib` is each card's
 * reserve in the same order (vram_budget_headroom_mib); a card whose reserve memory.card-headroom-mib
 * sets is flagged in `named`. With several cards and one of them held by others, the line names
 * memory.card-headroom-mib as the way to give that card alone more room. */
std::string vram_holders_report(const std::vector<int>& cards, const std::vector<std::string>& bdfs,
                                const std::vector<VramHolders>& h, const std::vector<int64_t>& headroom_mib,
                                const std::vector<bool>& named = {});

}  // namespace rad
