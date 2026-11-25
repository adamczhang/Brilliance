# Four ranks on a PCIe fabric: what pays and what doesn't

Every TP4 lever tried on 4x R9700 (PCIe Gen4 x16, P2P, behind PCIe switches, 210 W cap), and what
decided it. Use it before re-attacking the same ground on another card or fabric. Unless a row says
otherwise the numbers are paired within-session windows (bar 0.5% plus a sign test) at
`prefill-span: 2`, `tp-wire: exact`, `placement: all_vram`, iq4r MoE, `max-num-batched-tokens:
4096`, `checkpoint-interval: 2048`; the Hy4-flash figures also carry `num-speculative-tokens: 5`
(DFly2). Where a number is a BetterBench `--quick` median of five passes, the run's own IQR is the
error bar and category spreads of 7-21% are acceptance noise, not signal.

What the fabric is here, measured: `link-capped` (16.0 of 32.0 GT/s x16 on all four cards, so Gen4
lanes behind Gen4 switches), `power-cap-below-default` (210 W against the card's 300 W). Both are
not ours to change; every number below is therefore the *capped*, Gen4-switched column, and each
link- or clock-bound lever reads smaller than it would unfettered.

Shorthand: `pp` prefill; `c8` eight concurrent streams; `M` rows in a collective; `ti8` the int8
peer wire (`tp-wire: wht6`'s coding); `hc_write` the fused all-reduce + residual-write row.

## What shipped

- **`prefill-span: 2`**: overlaps the two half-chunk all-reduces with compute through the second
  lane's rows. +5-11% pp on Gen5 lanes, +2.0-2.8% pp on the capped Gen4 fabric here (Hy4-flash,
  BetterBench `--quick`, 5 passes: 16k 4359.9 -> 4447.2 t/s, 32k 4277.7 -> 4397.7; 2k and 8k inside
  0.6%), +1.6% c8. It needs `checkpoint-interval: 2048` and `all_vram` (placement flips its sign)
  and a budget that holds the span (`max-num-batched-tokens: 4096` for span 2).
- **`tp-wire: exact` with the span on** (reversal of the earlier `wht6` recommendation): `wht6`
  earns +0.5-0.8% pp over `exact` once the span is on (8k 4487 -> 4579, 32k 4398 -> 4430) and
  +0.3% at c8, and it costs output quality. At one stream the wire makes no difference at all:
  `wht6` 161.5 tok/s against `exact` 161.6 on the same prompt (greedy, 200 tokens, three runs). A
  six-row step is dispatch-bound, not byte-bound, so the lossy coding buys nothing where a single
  stream spends its time.
- **`ts_min` 16384 (was 32768) for the four-rank exact one-shot**: the speculating decode's verify
  at depth 5 is six rows of 3072, 18432 elements, which the old default kept on the one-shot. As a
  two-shot it decodes 131.2 tok/s against 129.5, +1.3% (greedy probes, three runs each, +-0.1%;
  `ts_min=65536`, which also keeps those six rows one-shot, lands on 128.9, so the edge is the row
  count and not the run; the shipped default, with no pin at all, reproduces it: 131.2 tok/s median
  of five greedy runs against 129.4 for the old default). A two-shot sends 1.5N a rank instead of
  3N for a second handshake, and 18432 elements is past that trade's break-even.
- **The fused iHC junction at four ranks, up to eight rows**: the all-reduce happens inside the
  norm's kernel, so it holds the CUs while it waits on the link. That pays where the step is
  launch-bound and loses where it is link-bound, and the crossover is measured: six rows (one
  sequence's depth-5 verify) 132.8 tok/s against 131.0 chained, +1.4%; twelve rows (two sequences)
  213.1 against 221.6, -3.8%; twenty-four -6.2%; forty-eight -9% (0.9% run-to-run, and the greedy
  output is bit-identical with and without). `arch/hyv4flash` therefore declares the twin up to 8
  rows and keeps the chain (add, standalone exact all_reduce, plain junction) declared for every
  height above, so `Junction::step` picks per step. This replaces the old
  `RADIANCE_HY4_JUNCTION_TP4` gate, which was off because the twin "has not run on four cards" — it
  has now: `=1` restores the whole verify band, `=0` never fuses at four ranks. c8 528.2 against
  516.4 chained, i.e. the split costs nothing where the old gate protected. (`ar_*_4rank` rows release the link gate themselves, as the
  two-rank rows do). Without it every yielding DMA (stager copies, expert prefetch) idles behind
  each collective: removing it costs 3.7% at c8. Any new collective row must honour the gate.
- **Four-rank selftest coverage**: the wide-collective and fused-write cases at world 4.
- **The second-lane rows at world 4** (`ar_oneshot_4rank_exact_lane`,
  `ar_rmsnorm_ihc_junction_4rank_lane`): the rows a collective resolves to when the caller says
  `overlap: 1`. Both are the wide entries and the wide kernels -- the junction already takes whole
  rows a block, so its one a row *is* the narrow launch. Nothing in `core/` reads an `overlap` param
  at all: a plugin declares and issues the second-lane rows from its own pipelined prefill loop, as
  `qwen35_fp8`, `qwen35_w8a8`, `gemma3` and `gemma4` do. The span does not use them either --
  `core/sched/geometry.cpp` gives the *chunk* two intervals, so the span's gain comes from fewer
  chunk boundaries, not from those rows. Since the pipelined prefill above, `arch/hyv4flash` is the
  fifth plugin that does (the `overlap 1` twin `Junction::declare` puts beside the plain row, issued
  only for a step's pieces and only with `RADIANCE_HY4_PIPE=1`), and it is the only user of the
  world-4 rows.

## What was tried and lost

| lever | result | why it is closed |
|---|---|---|
| widen the MoE fast band to four classes | no change | the cap only bites 3-bit formats; inert for 4-bit experts |
| fused all-reduce + `hc_write` on the exact-wire band | never selected | `wht6` takes the lossy band, and the fused row is exact-only. Never timed in its own territory |
| per-class grid split for the MoE GEMM | not constructible in one launch | a run's class is not a property of `blockIdx.y`. A two-launch split was never tried |
| fused `ti8` two-shot + `hc_write` | c8 -0.43%, c16 sign-split | the saved launches sit at the ~4 us launch floor; the epilogue serialises behind the all-reduce spin |
| `RADIANCE_HY4_JUNCTION_TP4=1` (fuse every verify height) | c1 +1.0%, c2 -3.8%, c4 -6.2%, c8 -9% | a fused collective computes inside the norm's kernel, so the CUs idle while it waits on the link; only a one-sequence step is launch-bound enough to pay. Kept as an env override, not a default |
| `ts_min: 0` (never hand a message to the two-shot) | c8 -8% (488 against 531, three runs each) | the two-shot's 1.5N a rank already pays for its second handshake from 6 rows on; this closes open lever 4 (a one-shot `ti8` band) by arithmetic too: it would send 1.78N with one sync against the two-shot's 0.89N with two, which is the trade that just lost |
| `nblocks` 2 and 8 on the four-rank rows (default 4) | c8 530 and 492 against 531, c1 unchanged | handshake-bound: 2 blocks is the default's equal and 8 is worse, so the block count is not where a six-row or 48-row message spends its time |
| `drain=1` (vmcnt instead of s_wait_storecnt), `acq=1` (fence after the flag), `nthreads` 512 and 256 on the four-rank rows | c1 132.4 / 132.5 / 132.4 / 132.9 against 132.7 default; c8 515-523 against 530.7, all inside its ±8% | none of the three reaches the flag path's cost: the tune is accepted through every four-rank kernel (a bogus value is refused by name, so the sweep did apply), and one sequence's step is bound by the launch and the handshake, not by how the wait is spelled or how many threads walk the rows |
| `tp-wire: wht6` on top of the span | +0.5-0.8% pp, +0.3% c8 | it costs output quality for less than a percent once the span is on; at one stream it is exactly free (161.5 against 161.6 tok/s), because a six-row step is dispatch-bound |
| more all-reduce blocks (16 -> 32 -> 64) | GB/s falls (M=2048: 34.5 -> 31.4; M=128: 18.4 -> 7.4) | handshake-bound, not launch-starved |
| `tp-wire: exact` everywhere | -7.6 to -8.3% at span off | exact wins only at decode steps of 25 tokens or fewer, which it already serves |
| rank-load balancer | +1.3% at 64K only | changes output bits; not worth it |
| step-graph capture | none for the trunk | trunk submission is device backpressure, so the host is hidden. Evidence is a synthetic ring, not an engine capture |
| `ti8` MoE gather row | premise false | on the lossy wire the gather's all-reduce already rides `ti8` |
| two-shot transport with the iHC junction as its tail, at prefill heights | not built; ~1% by arithmetic | 84 junctions a step, each worth one launch (~4-4.6 us, `--perfops`) plus one dependency gap: about 0.4 ms of a 30-40 ms prefill chunk. The transport does not change (1.5N either way); only the epilogue rides along. The kernel that does it for `ar_rmsnorm_quant_i8`'s junction (`r4d_rmsnorm_quant_i8.hip`, `ar_rmsnorm_quant_i8_tail_kernel`) is ~250 lines of two-hop staging that has no counterpart for the junction's post/h tables. Not worth an unmeasured kernel |
| faster `ti8` codec | none possible | encode is 4% of the row (36 of ~900 us); the peer push runs at ~60 GB/s, PCIe Gen5 x16 line rate. The "22% tax" against exact is the link |

Only a lever that ran correctly and still lost to a measured mechanism is closed for good: the
block count, the `ti8` fusion and the codec. The fused exact-wire rows, the two-launch class split
and draft-pass graph capture were never built or never timed where they would pay.

## Open levers, ranked

1. **Split the attention core too** (`hyv4flash_attn.h`). Pieces 0's junction and the last piece
   are still exposed because the arms between them are whole-step. Three of four halves hidden
   against today's one of two is +3% over the pipeline already built (see the section below for the
   numbers, the absorbed penalty it has to earn back, and the `row0` argument already in libr4d);
   the cross-layer handoff that would hide the fourth is another +3% at most and needs the layers to
   overlap. **Built so far: the libr4d `row0` argument only** -- see the section below for where it
   stopped and what the measured power envelope says about the ceiling.
2. **The dense pipelined prefill at four ranks** for `qwen35_fp8` / `qwen35_w8a8`: their declare
   gates on `g.world == 2`. Two traps are now known and cheap to avoid (the whole-step junction
   must not run beside its pieces; every collective, exposed pieces included, has to ride one
   lane), so this is a measurement, not research.
3. **Fused gather/write rows for exact-wire configs**, where they still have territory.
4. **The span floor at TP4 with tiered experts**: eight checkpoint intervals, priced on Flash-Next at
   TP2 only.
5. **Draft-pass graph capture**, if low-concurrency latency matters: the host gap is real at c1/c2
   and hidden at c8.

Closed by measurement, so they do not come back as ideas: the cut of the two-piece pipeline (six
cut points within 0.6%, see below -- it is launches and the arithmetic of the split, not the routed
pass's M); a four-piece pipeline on an 8192-row budget (the chunk is interval-capped, so there is no
8192-row step to split, and by the split's arithmetic any number of pieces hides the same half of
the boundaries); a wider token budget for decode (c8 519.9 against 519.6 shipped, c1 133.5 against
134.2 -- flat); always-one-shot (`ts_min: 0` loses 8% at c8); junction fusion above 8 rows (loses 9%
at c8); `nblocks` pinning at 1 (text bit-identical, no gain); and any instruction-level work on the
four-rank kernels themselves -- the link ceiling is 37.5 GB/s a rank and every smaller message is
handshake-bound, which is what made the pipeline the lever it is.

## What the four-rank collectives can and cannot gain

`r4d_selftest --perfar4 <numel> [iters] [nblocks]` runs the real rows on four cards, one thread a
rank, and checks the fold on every rank. bf16, 300 launches a point, the slowest rank's span:

| message | one-shot | GB/s a rank | exact two-shot | GB/s a rank |
|---|--:|--:|--:|--:|
| 6 rows of 3072 (36 KiB, one sequence's verify) | 20.9 us | 5.3 | **14.2 us** | 3.9 |
| 12 rows (72 KiB) | 17.6 | 12.6 | 16.5 | 6.7 |
| 48 rows (288 KiB, c8) | 40.5 | 21.9 | 38.3 | 11.6 |
| 512 rows (3 MiB) | **282.0** | 33.5 | 289.8 | 16.3 |
| 2048 rows (12 MiB) | 1025.2 | 36.8 | **1014.6** | 18.6 |
| 4096 rows (24 MiB) | 2015.5 | **37.5** | 1978.3 | 19.1 |

Four things follow, and they close the question "can the kernel itself be made faster":

- **The link ceiling is 37.5 GB/s a rank** (three peers, 24 MiB). Everything below that is latency,
  not bytes: at c8's 288 KiB the collective runs at 21.9 GB/s, at the verify's 36 KiB at 5.3. A
  six-row message spends ~10 of its ~14 us in the handshake and the launch while moving 54 KiB. No
  instruction-level change removes that; halving the bytes (what the two-shot does) is what pays,
  which is why 36 KiB prefers it 14.2 against 20.9 (-32%). That is the row-level reason for the
  `ts_min: 16384` default, which until now rested on end-to-end decode alone (+1.3%).
- **Above 512 rows the two-shot's traffic advantage stops mattering**: 3 MiB it loses 2.8%, 12 and
  24 MiB it wins 1-2%. Its per-rank byte rate is half the one-shot's, so what it really buys is one
  fewer DMA burst a block at tiny sizes, not less traffic.
- **A tiny message wants few blocks**: at 6 rows the two-shot's block count 1 / 2 / 4 / 8 runs 13.2 /
  14.1 / 17.0 / 23.5 us, monotone, because each block's push is its own burst; block 1 also beats
  the rule's 14.2 by 7%. The rule stays the default (pinning it end-to-end changes nothing
  measurable: greedy 192-token probe, `RADIANCE_TUNE=all_reduce:nblocks=1`, 134.7 against 134.6 t/s
  at c1 and 503 against 505 at c8, c1 text bit-identical), and 1 is now in the axis list so a
  sweep can reach it. `ts_min` is not a row parameter -- the adapter reads it to choose the row, so
  `--perfar4` cannot pin it.
- **What is left is the count**: ~4 us a launch plus ~8 us of handshake, 60-84 times a decode step
  (84 x 14 us is 1.2 ms of an 18 ms step). Removing launches is the pipelined prefill's job (the
  `*_4rank_lane` rows), not a kernel's -- and at four ranks it bought 13.5% with decode flat,
  see below.

## The pipelined prefill at four ranks: built, +13.5% with decode untouched, off by default

A prefill step's two collectives per layer (the attention junction and the FFN junction) sit where
nothing runs beside them, so their ~1 ms each is a hole in the step. `arch/hyv4flash` now splits a
wide chunk into two row pieces and rides the second piece's junction on lane 1 beside the first
piece's arms: `Model::pipe_on`, `Junction::step(c, r0, rows, beside)` (the `overlap 1` twin rows in
`libr4d`), `Ffn::body(c, T, r0, n)` and `rad_buf_concurrent` on the two pieces' transients. Off
unless `RADIANCE_HY4_PIPE=1`.

Hy4-flash, one 10948-token prompt, three cold runs each, same session, `max-num-batched-tokens:
4096`, `prefill-span: 2`, `placement: all_vram`:

| config | pp t/s | against shipped |
|---|--:|--:|
| shipped: `checkpoint-interval: 2048`, the span's overlap | 4813.1 / 4833.8 / 4818.0 | -- |
| `checkpoint-interval: 4096`, pieces off | 4847.4 / 4868.5 / 4881.4 | +1.0% |
| `checkpoint-interval: 4096`, pieces on lane 0 (no overlap) | 4692.3 | -2.5% |
| `checkpoint-interval: 4096`, pieces on lane 1 (pipelined) | 4998.1 / 5007.7 / 5010.4 | +3.9% |
| `checkpoint-interval: 4096`, `max-num-batched-tokens: 8192`, lane 1 | **5136.3 / 5124.7** | **+6.2%** |

A second session, rebuilt from the tidied sources, measured 4881.4 against 5010.4 t/s (+2.6%), so
the pair holds at +2.6 to +3.1% over its own baseline. The last row is the config to use: 5130.5
against 4825.9 t/s, alternating runs in one session, i.e. **+6.2% with decode untouched** (c8 520.8
and 519.9 without the pipeline against 521.4 and 518.7 with it, c1 133.6 against 133.8).

So the pieces cost 3.2% and the second lane returns 6.5%, and the pipeline's own baseline is the
same step run whole.

**Re-measured on the current base**, after the rebase onto the G-3 launch fusions (a step's routed
arm now runs one fused `swiglu_clamped_had_quant_i8g` pass instead of two). Same prompt, three
warm runs each, one session, `RADIANCE_HY4_PIPE` the only difference between the last two rows:

| config | pp t/s | c1 t/s | c8 t/s | against shipped |
|---|--:|--:|--:|--:|
| shipped: `checkpoint-interval: 2048`, the span's overlap | 4913.5 | 148.0 | 553.2 | -- |
| `checkpoint-interval: 4096`, `max-num-batched-tokens: 8192`, pieces off | 4978.1 | 146.3 | 550.3 | +1.3% |
| the same, `RADIANCE_HY4_PIPE=1` | **5577.5** | 147.2 | 551.8 | **+13.5%** |

A second session of the pipelined arm measured 5622.4 / 147.8 / 553.2, so the band is 5568 to 5622
t/s. The pipeline is +12.0% over its own baseline there, against +3.9% on the older base: the
pieces are 2048 rows a piece at an 8192-token budget, and a 2048-row piece sits a whole shape band
below the 4096-row step, so the step's second half gains from the split as well. Decode is
untouched in the sense that matters -- c1 is 146.3 to 148.0 t/s across the three arms (the same
`e56cb2263f` for 192 greedy tokens) and c8 550.3 / 551.8 inside its +-1.5% noise. Prefill is still
not bit-identical: 448a9473ec without the pieces (the same hash the shipped config gives) against
e7001a8285 with them.

**A row floor in the declare cost the arena its levels, not just the plugin a branch.** `pipe_on`
was computed with `T > 1024` and the narrow twin of each junction was declared only when it was
set, so the graph a pipelined deployment declared at 8192 rows carried six more reductions than the
one it declared at 256 or 1024 -- and the arena sizes each level it lends by declaring the graph
again at a smaller step. Measured before the fix, on this box:

    W activation arena: no level for 256-token steps on rank 0 -- the sizing declare failed:
      invalid argument
    W activation arena: no level for 1024-token steps on rank 0 -- the sizing declare failed:
      invalid argument
    I activation arena: 4096 tokens use 2.03 GiB of 3.93 GiB ...

So two of the three levels were dropped, steps below 4096 rows could not lend their arena back to
the expert slab, and boot held ~1.9 GiB it did not need -- which on a box sized for the loan is an
out-of-memory boot, and the KV pool loses the same. The row floor is the step's business
(`n0 >= RADIANCE_HY4_PIPE_MIN`), not the graph's: the twin is declared wherever the library serves
it, `pipe_on` keeps only the env, the rank count and the library check (it clears itself if the
twin did not resolve), and the levels come back:

| arm | 256 tokens | 1024 tokens | 4096 tokens | arena | the two pools |
|---|--:|--:|--:|--:|--:|
| shipped (2048-row steps) | 196.72 MiB | 556.32 MiB | 1.01 GiB of 1.95 | 1.95 GiB | -- |
| pieces off | 260.80 MiB | 620.40 MiB | 2.01 GiB of 3.88 | 3.88 GiB | 107.47 GiB |
| pieces on | 262.30 MiB | 626.40 MiB | 2.03 GiB of 3.93 | 3.93 GiB | 107.15 GiB |

The cut costs 0.05 GiB of arena and 0.32 GiB of the pools, not 1.9 GiB. And it was worth more than
the pipeline: with the levels dropped, the *decode* arm of a pipelined deployment was 540.9 against
552.1 t/s (c8, -2%, at the edge of the noise), which the level below 4096 explains -- after the fix
it is 551.8 against 550.3. `tests/hyv4flash_arch_test.cpp` keeps it that way:
`a_sizing_declare_at_four_ranks_matches_with_the_pipeline_on_and_off` declares at 64, 512 and 2048
rows with the pipeline off and on and compares the op and buffer lists (207 against 201 ops without
the fix, the six twins being the difference).

That the wider budget's arm fuses at all took a fix: its routed band tops out at 131072 slots, and
`libr4d`'s two swiglu-quant rows carried a `C_LE("M", 65535)` from a 16-bit row count that neither
kernel has any more, so the declare resolved to no kernel and every rank refused to start. The cap
is gone (`libr4d/r4d_rows.cpp`), and the fused pass is byte-identical to the pair at both heights
(prefill 448a9473ec at M 65536 and at M 131072). What it needs is a step whose rows fit in *one* checkpoint interval: a chunk is
capped at `checkpoint-interval` before the token budget, and the span can only overlap intervals
inside one step, so a step that is one interval has no overlap of its own. `checkpoint-interval:
4096` is that shape (the startup line names the bound: `prefill chunks of at most 4096`); at the
shipped 2048 the pipeline cannot run at all, because a 1024-row piece leaves the shape band its
kernels were picked from.

**The cut does not matter.** Six cuts, same session, step of 4096 rows, three cold runs each
(`RADIANCE_HY4_PIPE_ROWS`): 1280+2816 4997.2, 1536+2560 4998.2, 1792+2304 4995.5, 2048+2048 4993.7,
2304+1792 5001.1, 2560+1536 4970.6 t/s -- flat within 0.6%. So the 3.2% a piece costs is not the
routed pass's row count (a 1280-row expert pass costs what a 2048-row pass costs), it is the extra
launches plus the arithmetic of the split: whatever the cut, exactly half of the two boundaries is
hidden and half is exposed. The half cut stays, being the only one with both pieces inside the band.

**A wider budget pays only with the pipeline.** `max-num-batched-tokens: 8192` on top of the 4096
interval adds +2.4% (5130.5 against 5007.7) while the serial step gains nothing from it (4924.5
against 4870), and decode is flat either way. The likely reason is the second 8192-row step of a
long prompt, which the span keeps whole but the pipeline splits as 4096+4096.

Two mistakes cost 9 of the 12 points on the way, both worth knowing:

- **The whole-step junction still ran beside its pieces**: `if (jm) l.j_mid.step(c, T)` stayed, so
every layer added the connection and reduced a second time. That alone was the -12% piece cost.
- **A piece on lane 0 ends the lane-1 run**: the ranks have to agree on one lane for every
collective (`abi/rad_runtime.h`), so the exposed pieces ride lane 1 too. With them on lane 0 the
overlap measured +0.8% instead of +6.5%.

It is not bit-identical, and cannot be: the cut splits the routed pass, so each expert's GEMM sees
half the rows and rounds its own sum. A 32-token greedy continuation of the same prompt hashes
`2ef357978b` pipelined against `418c9054a0` whole; the lane itself is clean (the same pieces on lane
0 hash `2ef357978b` too). Decode is untouched: c1 134.2 against 134.1 t/s, c8 521 against 522.

The lever left is the exposed half. Piece 0's junction and the last piece still wait, because the
arms between them are whole-step; splitting the attention core as well would put them under GEMMs
too. Worked out and started, not finished:

- **What the split has to beat.** At an absorbed prefill the four boundaries of a layer hide three
  of four halves (piece 0 of the attention needs nothing beside it, and only the last junction piece
  stays exposed), i.e. 1.5 of the 2 ms a layer's reductions cost against today's 1.0 -- +3% over the
  6.2% below. A cross-layer handoff (the last junction piece beside the *next* layer's attention
  piece 0) would hide the fourth, +6% at most.
- **It is only possible absorbed.** 30 of the 42 layers slide, and a fresh prompt's sliding layers
  run the non-absorbed dense form, whose queries need every key of the step: not splittable by rows
  at all. `RADIANCE_HY4_PREFILL=absorbed` sends them through the paged form instead, and costs 2.4%
  on its own: 4822.4 against 4939.2 t/s, both without the pipeline (with the two-piece pipeline
  4999.2 against 5110.2, so the same 2.2% gap). The split therefore has to earn the absorbed penalty
  back before it earns anything.
- **What it needs.** `mla_attn_paged_sink` derives a query's absolute position from
  `used - qlen + i`, which is the *last* `qlen` rows' formula: a piece at rows `[row0, row0+rows)`
  of a one-sequence step needs `+ row0`. Added to libr4d as an argument with a 0 default (14 bytes
  in `SinkMeta`, `static_assert` keeps it padding-free, bit-identical while nothing passes it) and
  built. The selection form needs nothing: `sel`/`len` are row tables.
- **What it still needs.** `Attn::front(c, batch, r0, n)` and `Attn::core(..., row0)`: the piece's
  own planes are piece-local, only the layer input, the position and slot operands, the output rows
  and the selection rows move with `r0`. The twin declare carries `row0 = pipe_rows(max_tok)`, so
  the piece path runs only on a step of exactly the declared height (the 2756-row tail of a 10948
  token prompt stays on the two-piece path). Then the within-layer schedule: `front(0)+front(1)`,
  the indexer whole, `core(0)`, `core(1)` beside `j_mid(0)`, `body(0)` beside `j_mid(1)`,
  `body(1)` beside `j_end(0)`, `j_end(1)` exposed.
  **Where it stopped:** the libr4d argument is in and built, the arch side is not started, and the
  `row0` selftest case (a piece launch against the whole-step launch's rows `[r0, R)`) is unwritten.
  The argument is inert until then: nothing passes it, so every launch is the launch it was.

## Why the next +36% is not a scheduling question

The target set after the pipeline work was 7000 t/s of prefill, 45% over the 4829 t/s the shipped
config measures on a 10948-token prompt, and 300 t/s of decode on one stream, 2.2x over the 134.2 it
measures. Where that would have to come from was measured once a second during a prefill
(`rocm-smi --showuse --showpower --showclocks`, four cards):

| | idle | during prefill |
|---|--:|--:|
| GPU use | 3 % | **100 % on all four** |
| package power | 13-16 W | **180-268 W** (mean ~220) |
| sclk | 41 MHz | at level S throughout |

So the step is at the fabric's power envelope, not in bubbles: the pipeline's +6.2% (on the base it
was measured on; +13.2% against the shipped config after the rebase) is what scheduling had left, and the ~3% the attention split would add is the rest of it. 7000 t/s would
need 45% more work per watt, which is the expert path's math (4-bit decode and GEMM shape), not its
schedule. The decode target says the same thing from the other side: 300 t/s on one stream is
3.3 ms/token against today's 7.4 (134.2 t/s), 2.2x, and a one-stream step is dispatch- and
link-bound long before it is FLOP-bound.

## Instruments

- `r4d_selftest --perfar4 <numel> [iters] [nblocks]`: the four-rank collectives on four cards,
  fold-checked, `nblocks` pinnable; numel 0 runs the shipped sizes.
- `debug-graph: true` lists which row each op resolves to. It lists declared bands, so confirm
  against the running step.
- `r4d_selftest --perfops` times small ops (every elementwise op costs ~4-4.6 us regardless of
  payload); `--perfhc` and the GEMM/MoE modes time the rest.
- `rad-kbench` for kernel-against-reference replay; BetterBench's fixed profile decides promotion.
- The power/clock sampler behind the table above: `rocm-smi --showuse --showpower --showclocks`
  once a second, averaged over the four cards, while a `pprobe` run holds the prefill.
- `RADIANCE_HY4_PIPE=1` the pipelined prefill, `=0` off; `RADIANCE_HY4_PIPE_LANE=0` issues the same
  pieces on lane 0 (the piece cost without the overlap), `RADIANCE_HY4_PIPE_ROWS=<n>` moves the cut
  and `RADIANCE_HY4_PIPE_MIN=<n>` the floor a piece must clear. A `hyv4flash: pipelined prefill, N +
  M rows` line on stderr names the shape that took it.
