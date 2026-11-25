# Quantising Hy4-flash: what to run, in this tree

For whoever quantises `tencent/Hy4-flash` (bf16, 148 shards, 440 GB). Everything here is the short
form; the reference is [ARCHITECTURES.md, hyv4flash](ARCHITECTURES.md#hyv4flash-tencent-hy4-flash).
No real Hy4-flash weight has been through this tree: it was developed on random-weight dummies against the
transformers port (PR 50903, commit c2ec8505), so the first real run is a measurement, not a regression.

## What changed in the weight names since the first plugin commit (ce9ae1ff)

Convert once with the current tree; a container made earlier does not load. The declared set grew from
43699 to 43753 weights (`rad-convert --plan-only` over the release's 1964 tensors):

| before | now | why |
|---|---|---|
| `blk.l.attn_q_b` | `blk.l.attn_q_nope` + `blk.l.attn_q_rope` | the rope part of each head is rotated on its own; two linears over rows of the same `q_b_proj` (each head's 192 leading / 64 trailing rows). 42 layers |
| `blk.l.index_query` | `blk.l.index_query_nope` + `blk.l.index_query_rope` | the same split for the indexer's query (64 + 64 per head), the 12 sparse layers |
| `blk.l.ffn_gate_inp.{weight,correction}` | `blk.l.glm_router.{weight,correction}` | the router is read by the tree's GLM router block; f32 |
| (unchanged) | `attn_kv_b` | stays bf16 plain: the plugin projects through it as stored, a recipe naming it is refused by name |

Recipe patterns must use the new names: `blk.*.attn_q_nope.weight`, not `attn_q_b`. The routed experts,
Engram tables, DFly2 (`dfly.*`) and the tower keep their names. Router, iHC, sinks, indexer pooling and the
position parameters are not quantisable (a recipe naming one is refused at declare).

## What changed since "ready" (6dcaa5c8)

- **Update the engine.** A decode or verify step that crossed the 2048 dense limit was refused as issued
  differently from its recording (the pass key now carries whether the step selects; 2528faf5, with the
  WP13.0 rollback gate in b72b7994). Anything older fails the second step of a request that crosses 2048.
- **Re-convert for the vision tower only.** The tower is served now, with its weights re-declared: fused
  `v.blk.N.attn_qkv.{weight,bias}`, `v.position.weight` without the cls row (rows 1..side^2), and
  `v.merger.newline` as `[1, E]`. The text trunk's names are unchanged, so a container made for text needs
  the tower weights re-converted, nothing else. It is served only when the config has the M-RoPE sections,
  the vision heads and the media ids, and `--mm-max-patches` is given.
- **Media decoding needs ffmpeg at build time**: configure with `-DRAD_WITH_FFMPEG=ON` and
  `PKG_CONFIG_PATH=/opt/ffmpeg/lib/pkgconfig`. The default AUTO configure finds none and the server refuses
  every picture or video part.
- **Streamed sparse KV** (`--kv-resident N`, [KV-STREAMING.md](KV-STREAMING.md)): the sparse latent group lives
  in host memory and a ring of N tokens x max_seqs stays on the card; only worth it past the ring floor
  (32768 tokens a sequence). Bit-identical to resident on gpu-mini.
- **Optional quantised indexer**: `RADIANCE_HY4_INDEXER=fp8` or `int8` (default `exact`; other values
  refused). `fp8` is E4M3 at a power-of-two scale, `int8` symmetric int8 at an f32 scale per key / query head
  row; both use the same packed 136-byte page. Opt-in and not part of "ready": on dummies the selection
  differs from the exact path (see notes/idxint8.md for fp8 / int8 Jaccard and KL), so your KL with real
  weights decides which, if any, to use.
- **Chained launches are on** for Hy4-flash decode (about 8 % faster on gpu-mini); nothing to set.
- **DFly2 is served** (the release's block drafter; `--num-speculative-tokens 1..5`, off unless stated; TP1/2/4;
  [ARCHITECTURES.md, "DFly2, the Hy4-flash drafter"](ARCHITECTURES.md#dfly2-the-hy4-flash-drafter)). Nothing to
  re-convert: the `dfly.*` names are unchanged. **Its linears are read in whatever the container stores**: bf16
  (as before), rotated or plain int8, IQ8R, in any mix, sliced at TP like the bf16 ones. The decision is the
  recipe's: `data/recipes/hy4-flash-dfly-i8.recipe` and `hy4-flash-dfly-iq8r.recipe` (append their lines to a
  trunk recipe, or use one alone for a bf16 trunk with an int8 drafter) are the two forms the tree tests.
  The draft's bf16 linears were 11 % of a decode step in the friend's measurement (about 2 GB a draft pass at the
  real size, half of it in int8; on gpu-mini, whose draft is launch-bound, the int8 form is not faster: draft pass
  0.39 ms bf16, 0.45 rotated int8, 0.37 IQ8R at TP1). Not quantisable, as in the trunk: `attn_kv_b`
  and the token embedding (a recipe naming one declines the drafter, which then falls back to prompt lookup, and says
  so at startup). `kv_a` is left bf16 by the recipes (576 rows; above 64 draft rows an int8 one runs on the host).
- **TP4 and `tcq_rows`**: unchanged, already in the section on tensor parallelism below (128 rows on the
  gate_ups, 256 on the attention projections); a trellis container with other runs is refused at TP4.

## The commands

```sh
# 0. the declared set and sizes, no weights read (about 10 s; needs only the safetensors headers and config.json)
rad-convert <ckpt> --plan-only --recipe data/recipes/hy4-flash-w4nl64-rtn.recipe

# 1. bf16 container (the Engram hash constants first)
python3 data/hy4flash_engram_tables.py <ckpt> engram-tables.safetensors
rad-convert <ckpt> --add engram-tables.safetensors --recipe data/recipes/hy4-flash-bf16.recipe -o hy4-bf16.rad
#    a checkpoint larger than your disk: --defer / --in-place (TOOLS.md)

# 2. calibrate: ARCHITECTURES.md "Calibrating Hy4-flash" has the whole flow (rad-yaml, --calib-corpus,
#    --calib-dir, --calib-layers windows, tools/calib_st2gram.py, TP, the corpus). Short form:
radiance --config calib.yaml          # at --tp 4, from the bf16 container; then
python3 tools/calib_st2gram.py grams

# 3. quantise against the Grams (a recipe line with calib=$CALIB)
CALIB=$PWD/grams rad-convert <ckpt> --add engram-tables.safetensors \
    --recipe data/recipes/hy4-flash-w4nl64-gptq.recipe -o hy4-q.rad
```

Recipes in `data/recipes/`: `hy4-flash-bf16.recipe` (Engram E4M3 only), `hy4-flash-bf16-f32.recipe`
(the same at an f32 scale a row, code for code the port's f32 round trip), `hy4-flash-w4nl64-rtn.recipe`
(uncalibrated baseline, also the boot container a calibration can run from), `hy4-flash-w4nl64-gptq.recipe`,
`hy4-flash-w5nl64-rtn.recipe`, `hy4-flash-w6nl64-rtn.recipe`, `hy4-flash-i8.recipe` (trunk, experts and
head int8), `hy4-flash-iqr-gptq.recipe` (u4 trellis gate_up, w6nl64 down, u5 trellis trunk, int8 rest),
`hy4-flash-dfly-i8.recipe` / `hy4-flash-dfly-iq8r.recipe` (the DFly2 drafter's linears only, rotated int8 / IQ8R).
Write your own from them; ARCHITECTURES.md "Quantised Hy4-flash" lists every form the graph reads, per weight.

What is served (all of it measured only on random-weight dummies):

| weights | forms | note |
|---|---|---|
| routed experts | w4nl64, w5nl64, w6nl64 (rotated E4M3 activations); rotated int8, the IQR trellis, w4nl64 (rotated int8 activations); a w5nl64 / w6nl64 down behind a trellis gate_up; bf16 | SwiGLU clamp (gate <= 10, \|up\| <= 10) in every layer, every form: clamp in bf16, then quantise. TP2 / TP4: a slice of every expert (768 / 384 columns) |
| trunk linears | bf16; int8 rotated (`/fwht128`) or not; IQ8R; the i8a8h trellis (u5 ...); rotated u4/u5 with a zero point | each int8 linear quantises its own input; kv_a stays bf16 in the recipes: 576 rows are no whole 128-row tile of libr4d's trellis GEMM nor of its int8 prefill tile (`gemm_i8a8_tiled`), so above 64 tokens an int8 kv_a falls to the host |
| lm_head | bf16, int8 | |
| Engram table | bf16, E4M3 rows at a bf16 or f32 scale | |
| DFly2 draft linears (`dfly.blk.i.attn_q_a`, `attn_q_b`, `attn_kv_a`, `attn_gate`, `attn_output`, `ffn_gate_up`, `ffn_down`, `dfly.fusion`, `dfly.correct_gate` / `_up` / `_down`) | bf16; rotated or plain int8; IQ8R (not the trellis or zero-point forms) | the same `Lin`; sliced at TP; K in whole 128 (IQ8R 256) groups a rank |
| kv_b (also the draft's), router, iHC, sinks, norms, indexer pooling, Engram projections, token embedding (also the draft's) | as shipped | refused by name when a recipe touches them (a draft one declines the drafter) |

The IQR trellis (experts or trunk) is decoded by libr4d only: such a container serves on the cards, not
on the host. libavx has no w6nl64 kernel; on the host a w6nl64 container runs on libref.

A trellis is tail-biting over runs of `tcq_rows` rows down each column, and a tensor-parallel rank
takes only whole runs: the loader refuses a row shard that cuts one (`rows [a, b) ... cut a N-row
trellis run`). For a row-sharded trellis linear, `tcq_rows` must divide the rows a rank takes of each
stacked part at every TP you mean to serve: Hy4-flash at TP4 takes 3456 rows a part of the dense
`ffn_gate_up`, 384 of `ffn_gate_up_shexp`, 384 of an expert's gate_up and 1536 / 512 / 2048 of
`attn_q_nope` / `attn_q_rope` / `attn_gate` (8 sparse heads), so the recipe runs 128 rows on the
gate_ups and 256 on the attention projections. A container converted with other runs serves at the
TPs its runs divide, and is refused at the others.

## KL against your HF reference

`radiance --kld-ref DIR` scores a container against a directory of recorded log-probabilities
(TOOLS.md "The KL mode"; format in `core/kld.h`). radiance records its own with `--kld-record DIR
--kld-corpus corpus.jsonl`; yours comes from HF logits. Run the transformers port (c2ec8505) in bf16
over your token corpus, save each document's logits `[n_tokens, 120832]` as `.npy` (f32), and convert:

```sh
# corpus-ids.jsonl: {"ids": [...], "source": "code", "score_from": 0} a line; the ids HF saw (BOS, template tokens)
python3 tools/hf_logits_to_kld.py ref/ --docs corpus-ids.jsonl --logits-dir hf-logits/    # doc0.npy, doc1.npy ...
python3 tools/hf_logits_to_kld.py ref/ --docs corpus-ids.jsonl --logits-dir hf-logits/ --top 512   # sparse form for big corpora
rad-yaml --model hy4-q.rad --tp 4 ... --kld-ref ref/ --kld-out kl.json > cand.yaml && radiance --config cand.yaml
```

- Row `p` of a document's logits predicts token `p + 1`; positions `[score_from, n - 1)` are scored.
  radiance scores exactly the ids stored in `ref/tokens.i32`, so its tokeniser need not match yours.
- The width must equal radiance's logits width (120832; a wider HF logit matrix needs `--crop-vocab`,
  a narrower one is refused). The tool can also run the HF model itself (`--hf-model DIR`, CPU, slow).
- The full form is 240 KB a position (f16 over the vocabulary); `--top N` keeps the N most likely tokens and the tail
  mass (8 bytes a token) and scores a coarsened KL; numbers of the two forms do not compare.
- Run every candidate with the same flags. Reference and candidate need the same `--tp` only for the speed; the
  KL does not depend on it beyond rounding. Held-out text, not the calibration corpus.
- Tested on the dummy models: HF f32 logits of `mini` against radiance's bf16 container give KL about 1e-4
  (the parity harness's own noise) and 100 % top-1; shifted rows give a KL two orders larger.

## `tools/hy4flash_ready.py`

Checks the readiness list (A1..A11 of the plan) and prints a pass/fail table: build tests, plan-only over the
release structure (fetches the 148 headers once, ~10 MB), the host parity harness, calibration on `mini4`
at four fake ranks, the KL round trip, the chat tests. Items that need a card or a package not yet merged are
listed as OPEN with the command that decides them.

## TP4 safety (junction twin, kernel arguments)

- The fused `ar_rmsnorm_ihc_junction` (the all-reduce, sandwich norm, iHC post/pre and the next
  input norm in one launch) fuses the reduction into the norm's own kernel, so it holds the CUs
  while it waits on the link: it pays only where the step is launch-bound, which is one sequence.
  At world 2 the plugin uses it at every height; at world 4 up to eight rows (a depth-5 verify of
  one sequence), and the chain (add, standalone exact `all_reduce`, plain `rmsnorm_ihc_junction`)
  carries everything above. Measured on 4x R9700 behind Gen4 switches at 210 W (Hy4-flash, DFly2
  depth 5, greedy, bit-identical output): six rows 132.8 tok/s fused against 131.0 chained (+1.4%),
  twelve rows -3.8%, twenty-four -6.2%, forty-eight -9%. `RADIANCE_HY4_JUNCTION_TP4=1` widens the
  fuse to the whole verify band (the pre-split behaviour, and it loses at two sequences and up),
  `=0` keeps the chain at four ranks at every height. `RADIANCE_HY4_JUNCTION=0` keeps the whole
  unfused chain at every world.
- Kernel arguments passed by value (peers, row and gather structs of the all-reduce family) carry
  no padding (`static_assert`s in libr4d), since a recorded pass's audit compares argument bytes.

## Known limits

- The quantised forms above serve on the host (libref, libavx) at one, two and four (fake) ranks; on the
  cards only kernel by kernel so far (`r4d_selftest`): `gpu-mini` does not run end to end on a card yet.
  Calibrate from the bf16 container (the boot container with w4nl64 experts works too; its expert Grams
  are then in the rotated domain the GEMMs read).
- The Grams match `X^T X` of the plugin's own taps to 1e-7, of HF activations to 0.3 % to 1.6 % relative
  (bf16 activation rounding), TP4 against TP1 to 2 %.
- The indexer's selection ties (relu scores of 0) and the router's near-ties break differently from HF's `topk`;
  a parity comparison on random weights needs the conditioned reference of `ref/` (README there).
- DFly2 is served (see above) but its acceptance on the real weights is not measured here: `--num-speculative-tokens`
  stays off by default and the draft depth is not chosen automatically until it is (the friend's container
  measured 54 % when rewriting code, 93 % when copying, ~1 % on the distilled dummies). `--max-spec 0` is a
  `rad-convert` flag that leaves the drafter out. Pictures and video are served on the
  dummies only (no real tower weight has run).
- `gpu-mini` does not start on a card in some builds (Engram round trip kernels); the real-geometry card
  numbers are not measured.
- Engram E4M3 rows are not exact against the port in bf16 (2.9 % of codes differ by one step: the port divides in
  bf16 before the cast). `hy4-flash-bf16-f32.recipe` is exact against the port in f32.
