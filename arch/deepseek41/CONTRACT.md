# DeepSeek V4.1 Flash text contract

The input is the installed bakon3 `DeepSeek-V4.1-Flash-1017-uncensored-rad/deepseek-v4.1-flash-1017-uncensored.rad`, a 269,650,269,832-byte little-endian RAD2 file with architecture `deepseek41`, 32,701 entries and 11 encodings. The behavioral reference is the same quantized file on the PR #266 text tree at `68ef3618`. Use BF16 KV, exact collectives, and zero draft depth and prompt lookup. Compare fixed token IDs and chunk positions before judging generated text.

## Container

The full directory inventory and per-encoding SHA-256 were taken with the bounded inventory below and are kept with the port's working notes, outside the source tree. The header SHA-256 is `0cd353f6b102a741f7ebb831e960e7f9ff8a455ea36a633d1db994f4bde19` and the encoding table SHA-256 is `8491593ca4f44b89759ab362d0153a60e906f68dc040459edb05f75f22b82e57`. The bounded inventory reads only the header, directory, encoding, metadata, string and vocab tables.

- Global text weights are `token_embd.weight`, `output.weight`, `output_norm.weight`, `deepseek41.engram.token_map` and `per_layer_token_embd.weight`. The embeddings and head are 129,280 by 5,120. Vocab is baked BPE with 127,741 merges, BOS 0, EOS/EOT 1 and a baked Jinja chat template. Engram's hash pad ID is 2.
- The 40-layer trunk names weights under `blk.N`. Its attention families are `attn_q_a`, `attn_q_b`, `attn_kv`, `attn_sinks`, `attn_output_a.gG`, `attn_output_b` and the attention and low-rank norms. Source layers own the compressor and indexer tensors. FFN families are `glm_router`, `ffn_gate_up_shexp`, `ffn_down_shexp`, `ffn_gate_up_exps.E` and `ffn_down_exps.E`. The mHC weights are the `hc_attn_*` and `hc_ffn_*` families.
- Each routed movement unit has slot 0 gate/up (4,608 by 5,120) and slot 1 down (5,120 by 2,304). Canonical IQ1_S has u11 codes, signed vector grid, u4 scale, f16 scale and BF16 zero. IQ2_XXS has u8 codes, unsigned vector grid, signs, u5 scale, f16 scale and tensor f32 scale. Their native block pitches are 50 and 66 bytes per 256 elements. Other trunk rows use Q6_K, Q5_K, Q8_0, BF16, FP32 and integer metadata weights. IQ4_NL Engram uses u4 indices into a signed 16-value codebook and one f16 scale per 32 values.
- The text graph excludes the five global `dspark.*` tensors and layers 40–42. These total 854 entries and 8,500,993,928 bytes of directory extents, including all MXFP4 experts.

## Layer and operation map

- Trunk width 5,120 with four mHC streams, 64 attention heads of 512, Q rank 1,280, eight output groups of rank 1,024, one shared 512-wide latent KV head and a raw window of 128 tokens. The model has 384 whole experts per layer, top six scoring by sqrt(softplus), correction bias for selection, unbiased normalized weights scaled by 1.5, and one shared 2,304-column expert.
- Layers 0–1 and 37–39 use window attention only. Layers 2–19 use compression-ratio code 2, and 20–36 use code 1. Latent sources are 2, 8, 14 and 20. Index sources are 2, 8, 14, 20, 24, 28, 32 and 36. Candidate source 20 uses blocks of eight and up to 2,048 candidate blocks. Index top-k is 512 over 128-wide keys with 32 query heads. The ratio codes select graph geometry, so cache boundaries must come from the declared block size.
- Existing Windows generic rows supply norms, ordinary GEMM, mHC, token IDs, copy, all-reduce and all-gather. DS4.1 needs pitched projections, sparse entry-store/score/candidates/top-k/sink attention, compressor tail and round-trip operations, grouped GGML linear and expert rows, and grouped output projections. Reference implementations live in PR #266's `libref`; native rows in its `r4d_ds41_elem`, `r4d_ds41_sparse` and `r4d_ds41_ggml` families. Keep Windows' exact mailbox transport.
- At layers 1 and 14 map the token IDs, hash orders 2–4 across eight heads, gather 24 IQ4_NL rows of width 256 from the mapped table, then activate the gate and apply the residual. CPU gather and activation transfer are explicit placement boundaries. The two combined logical embedding ranges contain 768,022,850 rows.

## Capacity sketch

These are admission inputs for the short profile (one sequence, 4,096 context, at most 512 prefill rows). Live per-rank capacity, deferred allocations and peak host memory decide whether a real run can start.

| Input | TP4 first | TP2 second |
|---|---:|---:|
| PCI buses | 3, 9, 6, 32 | 3, 9 |
| Attention heads / output groups / experts per rank | 16 / 2 / 96 | 32 / 4 / 192 |
| Shared FFN rank widths | 768 / 512 / 512 / 512 | 1,280 / 1,024 |
| Compact routed experts per rank | 26.829 GiB | 53.657 GiB |
| Initial VRAM expert allowance per rank | 25 GiB | 25 GiB |
| Configured pinned pool, model total | 16 GiB | 40 GiB |
| Charged pins including late allocations, target maximum | 20 GiB | 44 GiB |
| Default process pin cap at historical 99.67 GiB non-local budget | 22.92 GiB | 47.83 GiB |
| Minimum beyond the VRAM allowance and configured pool | 0 | 17.314 GiB |
| Ordinary packed host expert store | 0 | up to 24 GiB |
| Illustrative additional physical peak, including 8 GiB miscellaneous | 28 GiB | 76 GiB |

The measured compact expert payload is 115,228,016,640 bytes across all ranks. Canonical planes occupy 143,333,563,392 bytes in their directory extents. Other text weights occupy 7,150,909,696 bytes before sharding, repack and replication. Account for cache duplication, workspace, per-class slack, staging, mailbox pins and allocator rounding before admission. Use the default pin limit computed from the live DXGI budget and the number of visible devices. Preserve at least 24 GiB of available physical RAM throughout admission, startup, serve and teardown, and check available commit independently.

Map the 110,595,290,656-byte Engram extent read-only on D:. Charge actual page residency and activation transfers to the host budget. Two lookups carry about 6.75 KiB of useful codes and scales per token before page amplification and duplicate rank gathers. TP4 admits a resident expert plan only if every rank fits its VRAM and pinned classes without expert file reads after startup. TP2 reads overflow once from D: into ordinary host RAM, repacks there, and stages selections through bounded pinned buffers. Keep D: as the fallback for omitted units, with the same layout and byte accounting. Refuse the plan before allocations if its projected peak crosses either guard.
