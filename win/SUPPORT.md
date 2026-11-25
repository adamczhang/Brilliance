# What Brilliance 0.3 does on Windows

The support record for the Windows build of Radiance 2.2: which models run on how many cards,
which of upstream's features work, the known issues, and what changes for a Brilliance 0.2.2
user. It is kept with the source and ships beside `PREVIEW.md`.

Measured on the project's test machine: four Radeon AI PRO R9700 (gfx1201) at -20% board power
(240 W), -50 mV and the stock VRAM clock (2518 MHz), Windows 11, Adrenalin 32.0.32015.12012.

## Models and card counts

| Model (container) | 1 card | 2 cards | 4 cards |
|---|---|---|---|
| Qwen3.8-27B IQ4R-L-V4 (`StillDeadcode/qwen3.8-27b-iqr-v4`) | **Qualified**: 8K and 32K x 4, DFlash2 or plain, bf16 or i8 KV; text, tools, pictures, video. Profiles: `qwen27-preview`, `qwen27-8k-plain` | Not run | Not run |
| Qwen3.8-Flash-Next IQR-V4 (`StillDeadcode/qwen3.8-next-flash-iqr-v4`, 113 GiB) | Not run. Upstream lists a one-card layout (experts tiered to a 43,000 MiB host pool, `deploy/compose/tp1.radiance.yaml`); whether it fits Windows' one-card pin limit is unverified | **Supported**: upstream's two-card settings, tuned, on the exact wire (8 sequences, 256K, int8 KV, MTP, prefix cache, experts tiered to host memory, 6,144-token prefill steps). The eight requests share a KV cache of about 262K tokens: one full-length context fills it ("Context, slots and the KV cache"). A lost write is re-sent, at about 4% of single-stream decode (known issue 1); the re-send and stop were checked at 6,144-token steps. Long-run stability is unproven on the current profile. Profile: `flashnext-2card` | **Supported**: every expert on the cards (`placement: all_vram`); soaked at 32K x 8, int8 KV, MTP on its earlier settings (2026-11-17), before the four-card re-send and stop (known issue 1). Its tuned settings, 256K x 8 (a shared KV cache of about 919K tokens, 3.5 full-length contexts), a bf16 KV cache and 6,144-token prefill steps, passed fault-injection, serving and limited stop checks. MTP drafting with an adaptive depth up to 5, as on two cards, was added after those checks: it measured about 13% faster single-stream decode on four R9700s, with prefill unchanged and greedy output identical to the earlier fixed depth of 3 on the tested prompts. When the cards' PCIe links differ, the large four-card exchanges give their reduce work to the fastest links (`BRILLIANCE_AR4_SHARD_WEIGHTS`, `auto` by default), with the same output bits as an even split. Long-run stability is unproven on the current profile; its stops are bounded without CPU load beside them, and a stop beside a large build is untested. Profile: `flashnext-4card` (known issues 8 and 12) |
| Qwen3.8-Flash-Next IQR-V4 (fast configuration) | - | **Runs**: serving and clean stop checked. Profile: `flashnext-2card-fast` | - |
| Qwen3.8-Flash-Next IQR-V4 (512K context, static YaRN) | - | **Runs**: loads and retrieves at 500K and 520K tokens; 512K x 2, int8 KV, MTP ([Context, slots and the KV cache](#context-slots-and-the-kv-cache)). Profile: `flashnext-2card-512k` | - |
| MiniCPM5-2B IQ8R (`StillDeadcode/minicpm5-2b-iq8r`) | **Qualified** for the local first run and tested text and tool APIs. Cancelling an active decode is unverified. On the CPU kernels with drafting off (`-Cpu`), it is **Preview** (known issue 10). Profile: `minicpm5-2b-1card`. [RC4 evidence](#alpha5-rc4-qualification) | - | - |
| Inkling 2 Mini IQ8R (`StillDeadcode/inkling-2-mini-iq8r`, `inkling-2-mini-iq8r.rad`, 26.14 GiB) | **Supported**: loads on one card in about 12 s (28.13 GiB of it); chat, Responses and Messages, streaming, tool calls and a picture checked on one card. 32K x 4, int8 KV, no drafting (the container's int8 drafter needs a later upstream release, and a start with drafting on is refused). No soak or concurrency qualification yet; long-run stability is unproven. Profile: `inkling2-mini-1card` | Not run | - |
| Inkling 2 Mini with DSpark (`gostar77/inkling2-mini-i8-dspark-radiance`, `inkling2-mini-i8-vl-dspark.rad`, 17.83 GiB) | **Supported**: loads on one card in 10 to 23 s (29.41 GiB of it); chat, Responses and Messages, streaming, tool calls and a picture checked on one card, and the drafter proposes and is verified. 128K x 4, int8 KV, DSpark seven deep on an adaptive window. No soak or concurrency qualification yet; long-run stability is unproven. Profile: `inkling2-mini-dspark-1card` | Not run | - |
| Qwen3.8-27B IQ8R-V4 (`StillDeadcode/qwen3.8-27b-iqr-v4`, `qwen3.8-27b-iq8r-v4.rad`, 27.58 GiB) | **Soak-clean**: 16K x 4, int8 KV, DFlash2 (26.67 GiB of weights leave about 3 GiB of cache: 55,616 tokens, 3.4 full-length contexts). Profile: `qwen27-iq8r-1card` | **Soak-clean**: 32K x 4, DFlash2. Profile: `qwen27-iq8r-2card` | Not run |
| EmbeddingGemma 2 bf16 (`google/embeddinggemma-2`, converted with `rad-convert`, 1.40 GiB) | **Qualified**: `/v1/embeddings` for text, pictures, video and audio, up to 8,192 tokens an input, 64 sequences, on one card. Also checked on the CPU kernels: across 17 comparison vectors the CPU and the card agreed to a cosine similarity of at least 0.999866 (close, not identical) | - | - |
| Qwen3-ASR 1.7B bf16 (`Qwen/Qwen3-ASR-1.7B-hf` as it ships, 4.09 GB) | **Qualified**: `/v1/audio/transcriptions` (5 formats, 4 response forms, past 30 s cut at pauses) and `/v1/realtime`, 16 sequences | - | - |
| Nemotron 3.5 Rerank 8B BF16 (`nvidia/Nemotron-3.5-Rerank-8B-BF16` @ f88f8d91 as it ships, 16.78 GB) | **Qualified**: `/rerank` and `/v1/rerank`, text and picture documents, 32,768 tokens a pair | - | - |
| SDXL base 1.0 bf16 (`stabilityai/stable-diffusion-xl-base-1.0`, the single-file checkpoint converted with `rad-convert`, 6.39 GiB) | **Qualified**: `/v1/images/generations`; a seed repeats its picture byte for byte; `/v1/images/edits` answers 501 | - | - |
| LTX-2.5 distilled IQ8R (`StillDeadcode/radiance-ltx-2.5-distilled-iq8r`, `ltx-2.5-distilled-iq8r.rad`, 36.86 GiB; gated, LTX-2.x Community License) | **Qualified** for standalone `/v1/videos` text and image-reference jobs and in-progress cancellation with Windows Media Foundation H.264 (`h264_mf`). AMD AMF encoding is not yet qualified. Its Gemma text encoder (14.00 GiB) and token embedding (1.88 GiB) are pinned in host memory: the DiT and the activation arena fill the card. A profile of your own may leave out `vision-tower-placement: ram`: the engine then moves the encoder to host memory itself and says so in the log. Profile: `ltx-2.5-1card`. [RC4 evidence](#alpha5-rc4-qualification) | - | - |
| Gemma 4 31B a4 (`keptn/gemma4-31b-a4`, `gemma4-31b-a4.rad`, 19.95 GiB) | **Qualified**: chat, Responses and Anthropic Messages, whole and streamed; tools; pictures and video; 32K x 2, bf16 KV, MTP; thinking off by the profile, as for the 12B (a request turns it on). Audio is refused with a 501. Speed not yet measured. Profile: `gemma4-31b-1card` | Not run | Not run |
| Gemma 4 12B i8g (`keptn/gemma4-12b-i8g`, `gemma4-12b-i8g.rad`, 13.79 GiB) | **Qualified**: chat, Responses and Anthropic Messages, whole and streamed; tools; pictures, audio and video; 32K x 4, bf16 KV, MTP; thinking off by the profile, as the container's card sets it (a request turns it on). Speed not yet measured. Profile: `gemma4-12b-1card` | - | - |
| Qwen-Image 2.1 IQ8R (`StillDeadcode/radiance-qwen-image-2.1-iq8r`, `qwen-image-2.1-iq8r.rad`, 16.13 GiB; Qwen's research licence, non-commercial) | **Qualified** for generation: `/v1/images/generations`, PNG, JPEG and WebP; a seed repeats its picture byte for byte; `n` gives seeds s, s + 1, ...; sizes from 512x512 to the default 2048x2048 on one card, a size off the model's grid rounded down; `response_format: url` is refused with a 400. Edits (`/v1/images/edits`) work for most seeds; see known issue 13. Profile: `qwen-image-1card` | Not run | - |
| Qwen3.6-35B-A3B IQ8R (`StillDeadcode/qwen3.6-35b-a3b-iq8r`, `qwen3.6-35b-a3b-iq8r.rad`, 35.04 GiB) | - | **Qualified**: chat, Responses and Anthropic Messages, whole and streamed; tools; pictures and video; 32K x 4, int8 KV, MTP; the weights fit the two cards, so no expert is read from host memory. It thinks before answering by default: a client's token limit counts the reasoning too. With `tool_choice: "required"` it can repeat its call; send `parallel_tool_calls: false` for exactly one. Audio is refused with a 501. Speed not yet measured. Profile: `qwen36-35b-2card` | Not run |
| DeepSeek-V4.1-Flash-1017, bakon3's uncensored container (`bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad`, `deepseek-v4.1-flash-1017-uncensored.rad`, 251.13 GiB with a 103 GiB Engram table; IQ1_S and IQ2_XXS routed experts; MIT) | - | **Runs, not yet qualified** (preview): text; plain completions and chat with thinking on and off; 4K x 1, bf16 KV, no drafting, the exact wire. The routed experts do not fit two cards: each card holds what fits, then a 20 GiB pinned pool a card, and the rest (about 9 GiB a card) is read once from the file at load, repacked and kept in ordinary RAM, with the file as the fallback; nothing was read from the disk while it served. About 61 GiB less host memory is available while it serves; the start refuses before allocating the placement pools if its peak would leave less than 24 GiB available or pin more than the process may. The `deepseek41-2card` instructions in `PREVIEW.md` show how to print the plan and its verdict. It matches four cards in distribution within a measured numeric tolerance (median KL 0.017 at confident positions); send `temperature` with each request, and expect greedy text to differ token by token between layouts (known issue 14). Profile: `deepseek41-2card` | **Runs, not yet qualified** (preview): text; plain completions and chat with thinking on and off; 4K x 1, bf16 KV, no drafting, the exact wire; every routed expert on the cards (about 29.7 GiB a card), the Engram table read from the file as each step needs it, about 21 GiB less host memory available while it serves. The checkpoint sets no sampling of its own: send `temperature` with each request. No soak, stop or concurrency checks yet; not run in the Sandbox. Windows HIP determines device order. The engine startup report shows the rank assignments. Profile: `deepseek41-4card` |
| Hy4-flash IQR-V4 u3 (`keptn/hy4-flash-iqr-v4`, `hy4-flash-iqr-v4-u3.rad`, 128.61 GiB; trellis u3 and w4nl64 routed experts, a u5 trellis trunk, 60.24 GiB of E4M3 Engram tables, the IQ8R DFly2 drafter; Tencent Hy License 1.0) | Not run | **Preview**: text and tools; 8K x 1, bf16 KV, no drafting, the exact wire. Experts that do not fit a card sit in a 17,408 MiB pinned pool (25.5 GiB of experts on each card and 3.8 GiB in the pool on the test machine); the Engram tables are read from the file. Loads in about 16 s from a warm file cache, 30.5 GiB of each card. Smoke checks pass; greedy text agrees with four cards except at near-ties. Decode about 48 tok/s. With the DFly2 drafter (32K x 2, coupled, adaptive depth): greedy text with and without it agrees except at near-ties, BetterBench decodes 52 tok/s combined against 47 without it, faster on JSON and summaries and slower on prose (43 against 48) and reasoning, because the verify reads more of the pooled experts. Not soaked; stop under load unchecked. Profiles: `hy4-2card`, `hy4-2card-dfly` | **Preview**: text and tools; 64K x 8, bf16 KV, the DFly2 block drafter (five positions, coupled, adaptive depth, SuffixDecoding), every expert on the cards, the exact wire. Loads in about 18 s from a warm file cache, 28.6 GiB of each card. Smoke checks pass; greedy text with and without the drafter agrees except at near-ties; three needles found at 16K and 56K. Not soaked; stop under load unchecked. Profile: `hy4-4card-dfly` |
| Other upstream models | Not qualified on Windows. The package carries many of upstream's architecture plugins, but no other model and modality has been run here ("Find your model file", below). Speech-only and music-only models cannot serve in this build (speech synthesis and music are refused, "Upstream features") | | |

**Preview** for the Sandbox fleet. Profile: `sandbox`.
On one card, switching to LTX produced a VRAM-admission 503. LTX served a video when it was
the first model loaded in a fresh Sandbox. [RC4 evidence](#alpha5-rc4-qualification)

For Qwen3.8-Flash-Next IQR-V4 on two cards, **safe `flashnext-2card`** is the recommended
**Supported** profile, with the exact wire and copy engines (SDMA) off by default.
**Fastest `flashnext-2card-fast`** **runs** (model listing, chat, streaming, four concurrent
requests and clean stop). Start it with `.\serve.ps1 flashnext-2card-fast -Bus 3,9` to enable
wht6 and SDMA. On the test machine's R9700s on PCI buses 3 and 9, wht6 gave about 10% faster prefill for
32K–128K prompts. The compressed wire is lossy and can change greedy text. Its measured effect
on next-token probabilities was comparable to changing the prefill chunk size or KV cache
precision, and all 124 greedy test answers matched. With the exact wire, SDMA preserved greedy
output in sequential drafter tests. Prefill was unchanged with SDMA, and its decode gain was
inconclusive.

**A faster prefill for long prompts, opt-in:** `.\serve.ps1 flashnext-2card -Bus 3,9 -Env
BRILLIANCE_TP2_PIPELINE=1` (or `flashnext-2card-fast`). A long prompt prefilled alone in a step
wider than one expert pass (the profiles' 6,144-token steps) sends each layer's two-card exchange in
pieces on a second stream while the next piece is computed; the attention and recurrent layers
still run the whole step at once. Other steps keep the usual schedule: several requests in one
step, short steps, draft and verify passes, pictures and audio, active control vectors, a
checkpoint inside the step, and `prefill-span-pass: step`. It is off by default in both profiles.
On the exact wire it gives the same greedy tokens as without it, at the same batch shape and
drafting settings, with the copy engines on or off. Its exchange has the same protection as the
others (known issue 1): a lost write is re-sent, and one past the limit stops the server with
status 70. This was checked on the exact wire and, for lost writes and a stuck block, with
`flashnext-2card-fast`'s wht6.

Containers of the 0.2.2 era (whole-model FP8, EXL3, the old Flash-Next FP8/IQ4R, GLM-5.3-Flash
MXFP4, Brilliance 0.2.2's DeepSeek-V4.1 containers, Gemma 4 FP8) are not read by Radiance 2.2.
Use the current containers above; the DeepSeek-V4.1 container this build serves is the one in the
table.

## Alpha.5 RC4 qualification

These labels use the alpha.5 RC4 package (`Brilliance-0.3.0-alpha.5-88bd8fc-win64`), tested on
2026-11-22.

- **MiniCPM5-2B IQ8R on one card is qualified** for the first run from a local container and the
  tested Chat Completions, Responses and Messages text and tool APIs. The checks covered whole
  and streamed replies, typed and multiple tool calls, tool round trips, malformed inputs,
  client-close recovery and tested context limits. The stream ended naturally before the planned
  decode interruption, so cancelling an active decode is unverified. Fetching the model was
  outside this run.
- **LTX-2.5 distilled IQ8R on one card is qualified** for standalone text and image-reference video
  jobs with Windows Media Foundation H.264 (`h264_mf`). The checks covered job polling, MP4 and
  thumbnail retrieval, in-progress cancellation and a subsequent job. AMD AMF encoding is not yet
  qualified. Set `BRILLIANCE_H264_ENCODER=h264_mf` to select the tested encoder.
- **MiniCPM5-2B IQ8R on the CPU kernels is preview.** A local first-run request passed with
  `-Cpu`, which selects `libavx,libref` and turns drafting off.
- **The Sandbox fleet is preview.** Seven entries served in sequence on one card. Switching to LTX
  then returned a VRAM-admission 503 with the other entries cold. LTX served a video when loaded
  first in a fresh Sandbox. Browser workflows and the multi-card entries were outside this run.

## Find your model file

What `model:` can name, and what has been run on Windows, by the file you have. Status words are
the ones in "Models and card counts"; "not run" means no Windows evidence either way.

**Served directly.** A `.rad` container, or a checkpoint folder upstream serves at its own
precision (`model:` names the folder).

| File or folder | From | Run on Windows | APIs checked |
|---|---|---|---|
| `qwen3.8-27b-iq4r-l-v4.rad` | `StillDeadcode/qwen3.8-27b-iqr-v4` | 1 card: **Qualified**. Profile `qwen27-preview` | Chat Completions, Responses, Messages, streaming, tools, reasoning, pictures and video in, grammars, `/v1/completions`, `/completion`, `/infill` |
| `qwen3.8-27b-iq8r-v4.rad` | `StillDeadcode/qwen3.8-27b-iqr-v4` | 1 and 2 cards: **Soak-clean**. Profiles `qwen27-iq8r-1card`, `qwen27-iq8r-2card` | Chat, streaming |
| The repository's other sizes (`qwen3.8-27b-iq2r-v4.rad` and the rest) | `StillDeadcode/qwen3.8-27b-iqr-v4` | Not run | - |
| `qwen3.8-next-flash-iqr-v4.rad` | `StillDeadcode/qwen3.8-next-flash-iqr-v4` | **Supported** on `flashnext-2card` (safe) and `flashnext-4card`. **Runs** on `flashnext-2card-fast` (fastest) and `flashnext-2card-512k` (512K, static YaRN). 1 card: not run | Chat, streaming; upstream 2.2's drafting options and control vectors |
| `minicpm5-2b-iq8r.rad` | `StillDeadcode/minicpm5-2b-iq8r` | 1 card **Qualified** for the local first run and tested text and tool APIs. CPU kernels **Preview** with drafting off (`-Cpu`). Profile `minicpm5-2b-1card`. [RC4 evidence](#alpha5-rc4-qualification) | Chat Completions, Responses, Messages, streaming and tools. Cancelling an active decode is unverified |
| `ltx-2.5-distilled-iq8r.rad` | `StillDeadcode/radiance-ltx-2.5-distilled-iq8r` | 1 card **Qualified** with Windows Media Foundation H.264 (`h264_mf`). AMD AMF encoding is not yet qualified. Profile `ltx-2.5-1card`. [RC4 evidence](#alpha5-rc4-qualification) | Standalone `/v1/videos` text and image-reference jobs, including in-progress cancellation |
| `deepseek-v4.1-flash-1017-uncensored.rad` | `bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad` (reassembled from its six parts) | 2 and 4 cards: **Runs, not yet qualified**. Profiles `deepseek41-2card`, `deepseek41-4card` | `/completion`, Chat Completions, thinking on and off |
| `inkling-2-mini-iq8r.rad` | `StillDeadcode/inkling-2-mini-iq8r` | 1 card: **Supported**, drafting off. Profile `inkling2-mini-1card` | Chat Completions, Responses, Messages, streaming, tools, pictures in |
| `inkling2-mini-i8-vl-dspark.rad` | `gostar77/inkling2-mini-i8-dspark-radiance` | 1 card: **Supported**, with its DSpark drafter. Profile `inkling2-mini-dspark-1card` | Chat Completions, Responses, Messages, streaming, tools, pictures in |
| `hy4-flash-iqr-v4-u3.rad` | `keptn/hy4-flash-iqr-v4` | 2 and 4 cards: **Preview**. Profiles `hy4-2card`, `hy4-2card-dfly`, `hy4-4card-dfly`. 1 card: not run | Chat Completions, Messages, streaming, tools |
| `gemma4-31b-a4.rad`, `gemma4-12b-i8g.rad`, `qwen3.6-35b-a3b-iq8r.rad`, `qwen-image-2.1-iq8r.rad` | the repositories in "Models and card counts" | See each model's row there | See each model's row there |
| The folder `Qwen3-ASR-1.7B-hf` | `Qwen/Qwen3-ASR-1.7B-hf`, as it ships | 1 card: **Qualified** | `/v1/audio/transcriptions`, `/v1/realtime` |
| The folder `Nemotron-3.5-Rerank-8B-BF16` | `nvidia/Nemotron-3.5-Rerank-8B-BF16` @ f88f8d91, as it ships | 1 card: **Qualified** | `/rerank`, `/v1/rerank` |
| Any other upstream container or checkpoint | upstream's model list | Not run | - |

**Converted first.** These inputs are not served as they are: `bin\rad-convert.exe` (CPU only on
Windows) writes a `.rad` from them, and the `.rad` is what `model:` names.

| Input | Converted to | Run on Windows | APIs checked |
|---|---|---|---|
| The folder `google/embeddinggemma-2` (its `model.safetensors`), no recipe | `embeddinggemma2-bf16.rad`, 1.40 GiB | 1 card and the CPU kernels: **Qualified** | `/v1/embeddings` |
| `sd_xl_base_1.0.safetensors` (`stabilityai/stable-diffusion-xl-base-1.0`), with the recipe `share\radiance\recipes\sdxl-bf16.recipe` and CLIP's `tokenizer.json` (`--tokenizer`) | a bf16 `.rad`, 6.39 GiB | 1 card: **Qualified** | `/v1/images/generations` |

**Not served, on any platform.** Upstream 2.2 refuses at startup a container whose weights are
block FP8, MXFP4, M2XFP4 or ParoQuant, naming the first such weight; an int8 or bf16 container of
the same model is the alternative where one exists. The 0.2.2-era containers above are not read
either.

**Look inside a container before loading it.** `rad-info` reads a container's header without a
card and without loading the model:

    .\bin\rad-info.exe D:\models\hf\StillDeadcode\minicpm5-2b-iq8r\minicpm5-2b-iq8r.rad

It prints the architecture, the model it was converted from (`model`, `created by`) and the
metadata; `--recipe` prints the recipe it was converted with, and `-v` lists every weight with its
encoding. It does not say whether this build serves the container: match the file against the
tables above, then let `.\serve.ps1 <profile> -Check` validate the YAML that names it.

## Context, slots and the KV cache

A model's `max-model-len` is the longest context one request may reach, and `max-num-seqs` is how
many requests it runs at once. Together they do not reserve a full-length context for every slot:
the running requests share one KV cache, sized at load from the memory left on the cards. Its size
is in the log, in the line that starts `kv: the cache can grow to`. The shipped profiles, as the
0.3.0-alpha.3 package started them on the test machine (32 GiB cards):

| Profile | Requests x context | KV cache, a card | Full-length contexts it holds |
|---|---|---|---|
| `qwen27-preview` (1 card) | 4 x 32K | 131,072 tokens | 4.0 |
| `qwen27-iq8r-1card` | 4 x 16K | 55,616 tokens | 3.4 |
| `flashnext-2card` | 8 x 256K | 262,120 tokens | 1.0 |
| `flashnext-2card-512k` | 2 x 512K | 524,384 tokens | 1.0 |
| `flashnext-4card` | 8 x 256K | 919,452 tokens | 3.5 |

So `flashnext-2card` runs eight requests at once only while their contexts together fit about 262K
tokens; one full 256K context fills its cache. When the running requests need more than the cache
holds, the newest wait for room, and upstream's default lets a long prompt that fills the cache
partway be set back and computed again from its start. A machine with other cards, other memory or
other settings gets a different cache: read the `kv:` line from its own start.

Since 0.3.1 the cache never holds fewer tokens than one `max-model-len` request: the 0.3.0 table's
262,120 tokens for `flashnext-2card`, 24 short of 262,144, was a rounding loss, and the same start
now holds 262,260.

**A 512K context on two cards: `flashnext-2card-512k`** (**Runs**). Static YaRN (`rope-scaling:
yarn:2`, [docs/YARN.md](https://github.com/adamczhang/Brilliance/blob/main/docs/YARN.md)) over the 262,144 trained positions, `max-model-len:
524288`, two requests at once, a 9 GiB host pool a rank and an 8 GiB prefix host tier. On the test
machine (two R9700s at -20% board power, -50 mV and 2518 MHz VRAM; 125.5 GiB
of RAM), on the profile's container:

- **Startup.** Loaded in 17.8 s, holding 30.32 GiB a card. The cache holds 524,384 tokens (3.90
  GiB a card), one full 512K conversation. About 1.8 GiB of routed experts a card moved to the
  host pool (7.74 GiB a rank there, against 5.93 at 256K); 30.00 GiB pinned, against 24.10.
- **Retrieval.** Three six-digit codes hidden at 10%, 50% and 90% of seeded prose, asked for in
  order with thinking off: all three, in order, at 8K, 32K, 250K, 500K and 520K tokens.
- **Speed.** Prefill 5,611 tok/s at 250K (5,673 on `flashnext-2card`), 5,147 tok/s at 500K (97 s)
  and 5,181 tok/s at 520K (100 s). A second turn on the same 500K prompt hit the prefix cache for
  499,712 tokens and answered in 0.24 s. Decode of 1,024 tokens behind 500K: 227.8 tok/s with the
  MTP drafting stack (a continuation of the same prose, where drafts are accepted often).
- **Host memory.** At least 56.7 GiB stayed available through the 500K and 520K requests. The
  server committed 87.8 GiB (WDDM charges its VRAM to commit); the n-gram table's 47.68 GiB is page
  cache that memory pressure can reclaim.

YaRN is static, so it changes the output slightly at every length; `flashnext-2card` stays the
everyday profile. The streamed KV cache (`kv-resident`) is not needed for this and has not been run
on Windows. A card that also drives a display loses VRAM while the server loads: give that card
alone a larger reserve with `memory.card-headroom-mib: {CARD: MiB}`, so the other card keeps its
room for experts ([GUIDE section 7.2](https://github.com/adamczhang/Brilliance/blob/main/docs/GUIDE.md#72-messages-and-what-they-mean)).

## Upstream features

| Feature | On Windows |
|---|---|
| Chat Completions, Responses, Messages; streaming; tools; reasoning | Qualified on Qwen3.8-27B. MiniCPM5-2B IQ8R is qualified on one card for the tested text and tool APIs, including streaming. Cancelling an active decode is unverified. [RC4 evidence](#alpha5-rc4-qualification) |
| Grammars and structured output (`response_format`); `/v1/completions` and the llama-server routes (`/completion`, `/infill`) | Yes, checked on Qwen3.8-27B IQ4R-L-V4, one card (2026-11-19); not qualified. Grammars and structured output (`response_format` json_object and json_schema, llama-server's `json_schema`, GBNF on `/v1/completions` and `/completion`), tool calls under their parameter schemas, `/v1/completions`, `/completion`, `/completions` and `/infill`. A GBNF `grammar` on chat is refused on Qwen3.8 with a 400 (its chat template produces no grammar for it): use `response_format` or `/v1/completions`. There is no regex constraint (upstream has none). `/infill` needs a model with fill-in-the-middle tokens (Qwen3.8 and MiniCPM5 have them). Its `content` can end with the separator token's text (`<\|file_sep\|>`), which upstream does not treat as an end of generation as llama-server does: a client should cut the completion there. The host grammar tests pass (`gbnf_test`) |
| Pictures and video in requests (`image_url`, `video_url`) | Yes, through the bundled FFmpeg 8.1 (LGPL); checked on Qwen3.8-27B. H.264 input is decoded by FFmpeg's built-in decoder; H.264 decoding is patent-encumbered, and no H.264 patent licence is supplied (`share/doc/radiance/third_party/THIRD-PARTY.md`, "H.264"). JPEG XL pictures are refused with a 400 ("JPEG XL images are not supported by this build; send PNG, JPEG or WebP"), by their bytes or an `image/jxl` data URL: the bundled FFmpeg is built without libjxl |
| Audio in requests, `/v1/audio/transcriptions`, realtime | Audio in an embeddings request: yes (FFmpeg decodes it; checked on EmbeddingGemma 2). Transcription and realtime: yes on Qwen3-ASR 1.7B, one card (2026-11-18). `/v1/audio/transcriptions` takes wav, mp3, flac, ogg and m4a uploads, each word for word the WAV's transcript, and answers as json, text, verbose_json or a stream alike; `/v1/realtime` (OpenAI's protocol over a WebSocket) settles words while the audio streams and gives the transcript about 0.075 s after the commit, equal to the upload route's; 24 kHz audio is resampled. A model without the endpoint answers 501 with the reason (checked) |
| Image generation (`/v1/images/generations`) | Yes on SDXL base 1.0 bf16, one card (2026-11-18), in these tests: converted on the CPU with `rad-convert` from Stability's single-file checkpoint and CLIP's `tokenizer.json` (9.6 s, 6.39 GiB written). PNG, JPEG and WebP (FFmpeg) as `b64_json`; `response_format: url` is refused with a 400; the same seed repeated its picture byte for byte; `n` 1-8 gave seeds s, s + 1, ...; a size off the model's grid was rounded down (1000x700 gave 992x672); about 5.4 s for one 1024x1024 picture at 28 steps. `/v1/images/edits` answers 501 on SDXL, which makes pictures from text only. Qwen-Image 2.1 IQ8R is also **Qualified** on one card, for generation ("Models and card counts"); its edits work for most seeds (known issue 13) |
| Video generation | Standalone text and image-reference jobs (`input_reference`) and in-progress cancellation are **qualified** on one card with LTX-2.5 distilled IQ8R and Windows Media Foundation H.264 (`h264_mf`). AMD AMF encoding (`h264_amf`) is not yet qualified. Videos are MP4 (H.264 and AAC), at most 161 frames (6.7 s) on Windows: upstream allows 321 on Linux, but the larger activation arena it reserves does not fit beside LTX-2.5 on a 32 GB card. The default order is `h264_amf`, then `h264_mf`. `BRILLIANCE_H264_ENCODER` accepts a comma-separated list of those names and can restrict them or change their order. `h264_amf` needs the AMD driver's AMF runtime (`amfrt64.dll`, installed with the driver) and opens its device on Windows' default display adapter. On the test machine it encoded on the AMD integrated GPU. `h264_mf` needs Windows Media Foundation's H.264 encoder, which Windows "N" editions lack until Microsoft's Media Feature Pack is installed. With neither, a video fails with the reason. Every MP4 says it is AI-generated in its metadata (`comment`: `ai_generated=true; model=...`, and `description`), without a visible watermark. No H.264 encoder implementation ships in 0.3.1 (`share/doc/radiance/third_party/THIRD-PARTY.md`, "H.264"). [RC4 evidence](#alpha5-rc4-qualification) |
| Speech synthesis (TTS) | Not supported in 0.3.1. The Windows build refuses `/v1/audio/speech`, `/audio/speech` and `/v1/audio/speech/stream` for every model: the POST routes answer 501 (`not_implemented_error`, "not supported in this build"), the WebSocket an error event and close 1008 |
| Music generation | Not served: it shares `/v1/audio/speech`, which the Windows build refuses (above). It needs MiniMax-Music3, which is not run on Windows |
| Embeddings | Yes: `/v1/embeddings` on EmbeddingGemma 2, on one card and on the CPU kernels. Text, pictures, audio and video inputs; batches; `dimensions`; base64; concurrent requests each get their own vector; an input past 8,192 tokens is refused with a 400 |
| Rerank | Yes: `/rerank` and `/v1/rerank` on Nemotron 3.5 Rerank 8B, one card (2026-11-18). Text and picture documents; `top_n`, `return_documents`; scores bit-identical however the documents are batched or ordered; a pair past 32,768 tokens is refused with a 400, not truncated. Other models answer 501 with the reason (checked) |
| Multi-model fleet, routing, reload | Yes. The Sandbox fleet is **preview**. Switching to LTX on one card produced a VRAM-admission 503. Reload by `POST /admin/reload` or `tools\reload-server.ps1` (Windows' SIGHUP). The server rereads the profile used by `serve.ps1` or the service's copy under `%ProgramData%\Brilliance`, with `-Listen`, `-Port` and `-ApiKey` still applied. [RC4 evidence](#alpha5-rc4-qualification) |
| Model switching | Yes. **One card visible: by parking** (HIP virtual memory on, below): the leaving model's memory is unmapped at its addresses and the weights come back from the container; nothing is kept in host memory. **Several cards visible: by unloading and reloading**, no parking. Since 2026-11-18 PAL's resource cache is off on every start (`GPU_RESOURCE_CACHE_SIZE=0`; an operator's value wins), so a model reloaded in the same process plans against the same VRAM as its first load (before, up to about 4 GiB a card less) and a park takes 0.06-0.33 s. Repeated switching on two cards: checked 2026-11-19 with Qwen3.8-Flash-Next and Qwen3.8-27B IQ8R, both at tp 2; not qualified. 40 requests alternating, each switch an unload and a reload (Flash-Next about 17 s, Qwen3.8-27B about 6 s), with no drift in free VRAM or pinned memory; requests queued on a load are served by one load; a model that cannot get room beside a pinned one is refused with a 503 and the pinned one serves on (the client's message reads "out of memory"; the log names the reason). In a fleet of several models on more than one card, give each model `gpu-memory-mib` (known issue 2) |
| Prefix cache, checkpoint snapshots, agent replay | Yes |
| Speculation: DFlash2, MTP | Yes |
| Upstream 2.2's drafting options: `draft-coupling`, `draft-verify` (token, block, traversal), `draft-depth: adaptive` with `draft-depth-load`, `draft-suffix` | Yes on Qwen3.8-Flash-Next, two cards (2026-11-19); not qualified. Each serves; `draft-suffix` can change greedy text at near-ties (known issue 5, rechecked 2026-11-22), the others matched the uncoupled base on the probe. Coupling and the traversal verify also work on Qwen3.8-27B's DFlash2. Where a model cannot take an option, the start is refused, naming why |
| Control vectors (upstream 2.2) | Yes on Qwen3.8-Flash-Next with the shipped vector, two cards (2026-11-19); not qualified. The per-request field, the `MODEL:NAME` variant and `/admin/control-vector`. Only Flash-Next's architecture applies control vectors; Qwen3.8-27B refuses the key at start, naming why |
| Tensor parallel | 2 cards: soak-clean when the weights fit the cards, supported when experts are read from host memory (known issue 1), on the exact wire (the default). Flash-Next's fastest profile runs with the same protection on lossy wht6. 4 cards: Flash-Next supported with every expert on the cards, soaked at 32K on its earlier settings before the same re-send and stop were added (known issue 1). Long-run stability is unproven on the current Flash-Next TP2 and TP4 profiles. TP4 stops are bounded without CPU load beside them; a stop beside a large build is untested (known issue 12). 3 and 8 cards have not been run |
| Peer-to-peer between cards | No on Windows: collectives go through pinned host memory |
| Two-card all-reduce register form, coded wire and staged coded form (upstream since `3deabdae`) | No: they run only over peer-to-peer, so two-card collectives on Windows use the plain kernels through host memory |
| Upstream 2.2's launch-path speed-ups in the AQL backend: chained launches without the packet processor's acquire, the host data path flushed before each doorbell, the scalar cache emptied behind a chained wait, kept queues handed out oldest first, the expert prefetch stream's argument ring | No: Windows runs the native HIP backend, which has no AQL queues and does not chain launches. Upstream's two-card decode times for 2.2 (Flash-Next at tp 2: 11.2 ms a step at one stream, 12.4 in 2.1) come partly from these and from the peer-to-peer forms above, so they do not carry over to Windows as they are |
| Runtime LoRA on a rotated int8 trunk (`plugin/lora_plain`, upstream 2.2) | No: upstream builds the library by hand, the Windows build does not make it, and its source loads `libr4d.so` by name |
| `rad-convert`'s GPTQ step through OpenBLAS (upstream 2.2) | No: the Windows build has no OpenBLAS, so `libquant`'s GPTQ factorisations take upstream's single-threaded reference loops (upstream's own path without OpenBLAS; slower conversions, same method) |
| Dashboard | Yes; per card it shows load (GFX activity) and board power from the AMD driver's ADL (`atiadlxx.dll`, no admin needed). It shows the ADL junction temperature when available and falls back to the adapter temperature when Windows supplies it. Without ADL, load and power show as not available on Windows. Each model's page has a Monitoring tab (upstream's views) and a Brilliance **Sandbox** tab for a quick check: Start and Stop load and unload the model through `/admin/models/NAME/{load,unload}` (loopback callers only, as those routes are, unless the server has an API key), and a plain streamed chat over `/v1/chat/completions` kept in the page only. Start loads with `?hold=1`, so until Stop a plain API request for another model that needs its cards is refused (503, naming the hold) instead of evicting it. From another machine without a key, the page lists the models from the public `/v1/models` and opens each one's pages and Sandbox; the fleet overview and Start and Stop stay with the server's own machine. Checked against a stand-in server and on one card; not qualified |
| Terms of use (`GET /terms`) | Brilliance's own, not upstream's. The page the server shows the people it serves, linked from the dashboard and open without the API key: the shipped template `TERMS-OF-USE.md`, which carries the use restrictions some model licences (LTX-2.5's, SDXL's) require operators to pass on, or the operator's own file named by `terms-file:` under `server:`. Operators who serve other people must adopt it as their own terms (`PREVIEW.md`, "Serving on the network"). `/terms/licenses` shows each listed model's full licence text, from the files the package carries in `share/doc/radiance/model-licenses`, by the same rules: open without the key, no model woken |
| Diagnose mode | Runs; the busy-host warning reads the CPUs' load over a quarter second |
| `rad-convert` | CPU only, by decision; `--gpus` refused. A real model converted: SDXL, 9.6 s on the CPU |
| `rad-export` (finetune to container) | Yes, with Python 3 (`bin\rad-export.cmd`, which runs the `py` launcher or `python` from PATH). It runs the package's `rad-convert` beside it and finds the recipes in `share\radiance\recipes`; `--rad-convert PATH` chooses another converter. From the unpacked folder: `.\bin\rad-export.cmd --list-recipes` lists them, and `.\bin\rad-export.cmd <checkpoint folder> -o <name>.rad` exports. Its exit status is rad-export's. Folding a LoRA adapter also needs torch and safetensors in that Python |
| Model fetch by repo id | Yes (`hfget.ps1`, curl.exe and PowerShell). A repository of several containers is refused before the download, not after it. `serve.ps1 -Fetch` fetches the one file a profile names and serves the verified copy without an edit to the profile |
| Windows service (LocalSystem) | **Preview in 0.3.1.** Host and dry-run setup checks cover the YAML, environment, key handling and recovery settings. Host tests also cover simulated service controls and console-mode health, API-key and stop checks. Elevated Service Control Manager (SCM) install, key ACL, graceful stop and detachment, boot and preshutdown, reinstall, and one-restart recovery remain unverified. The launcher's no-card guard blocks the no-GPU recovery drill when HIP enumeration succeeds. Review `-Install -WhatIf` before installing and check readiness after starting. See `SERVICE.md`. |
| Hang detection (`/health`'s `"stuck"`, `scheduler.stuck-exit`) | Partly, and differently from upstream. Upstream's guide describes Linux, where the backend watches the device's queues and `scheduler.stuck-exit` ends a stuck process so a supervisor restarts it. Windows' HIP shows no queue progress, so the Windows stall watch times the host instead: when a thread is blocked waiting on a card and nothing has completed for 5 s (no wait returned, and no stream's progress beacon moved: every 64th op a stream writes one into host memory, so a long healthy prefill is not a stall), the log names the stall; once that has lasted 60 s, `/health` answers 503 with `"status": "stuck"`. A stall that blocks no host wait is not seen. This `stuck` answer has not been seen on a real hang on Windows. **The process is never ended for a stall on Windows:** `scheduler.stuck-exit` is not acted on, so a hung server stays up and a service's recovery actions never restart it. Someone has to act: save its stacks (`tools/STACKS.md`), then stop it with Ctrl-Break (`tools\stop-server.ps1`); a drain that cannot finish keeps waiting rather than ending the process by force |
| `--ngram-placement ram=DIR` (one table shared by engines) | Yes: the table share on the CPU engine, and the native HIP backend maps the shared file for the cards (device tests). Not yet run end to end: the one tested model with an n-gram table, Flash-Next, needs two cards |
| `ngram-warm: true` (an n-gram table read into the file cache at load) | Brilliance's own. Since 0.3.1 it is off in `flashnext-2card` and `flashnext-4card`: the Windows gather now prefetches the rows it needs and reads ahead (`BRILLIANCE_NGRAM_PREFETCH`), and from a cold file cache decode, 8-stream throughput and prefill measured within about 1-2% with and without the warm on two and four cards (2026-11-22), while the warm added about 9 s to each load. It stays available, and on in `flashnext-2card-fast`. Under `ngram-placement: disk` the Windows gather reads the table through the container's mapping, so a row the file cache lacks is read from the drive mid-step; the warm reads the table once before serving (47.68 GiB for Flash-Next, into the Windows file cache, nothing pinned). It runs only when the available memory covers the table plus the larger of 24 GiB and the configured host headroom; otherwise the start goes on without it and the log says why. The cache is Windows' to reclaim: under memory pressure rows are read from the drive again. Measured on the test machine (2026-11-20): the read took 18.9-21.0 s on two cards starting from a cold file cache, 6.0 s on two cards when much of the table was already cached, and 11.0 s in one four-card start; the load grows by that. On two cards from a cold file cache, prefill of fresh prompts (6K, 24K and 47K tokens, text the server had not seen) was 5.0-5.6x faster with the warm than without it; single-request outputs with the same seed were identical. Four cards' speed with and without it was not measured |
| Plugin SDK, out-of-tree plugins | From a source build's install; not in the package |
| Linux `scripts/` (measurement, traces, Docker) | Not ported; `win/tools` has the Windows tools (serve, stop, reload, soak, package check, stacks) |
| Cards other than gfx1201 (RDNA3, Strix Halo, gfx1200) | No: the toolchain is pinned to gfx1201 |

## Known issues

This table tracks Windows support limits and the checks that could change them. **Open** means a
known issue or pending verification. **Preview** means the feature runs within the documented
scope and wider qualification is pending. **Not supported** marks a current boundary. **Explained**
means a measured characteristic or numeric tolerance, not an open defect.

| Issue or limit | Owner | Evidence | Status | Next check |
|---|---|---|---|---|
| 1. Flash-Next host-memory writes can be lost. Keep resend enabled. | GPU collectives | [Issue 1 notes](#evidence-notes-for-numbered-issues) | open | Trace the origin of a lost write and verify resend and bounded failure on the exact wire and wht6. |
| 2. HIP virtual memory is on by default with exactly one visible device and off with more. Multi-card fleets need explicit per-model GPU budgets. | Windows HIP and fleet | [Issue 2 notes](#evidence-notes-for-numbered-issues) | open | Check the next HIP runtime for sparse commits, budget enforcement and repeated fleet switching with several visible devices. |
| 3. Windows caps pinned memory by visible device count. DeepSeek TP2 also requires at least 24 GiB of available host memory at its projected peak. | Windows memory placement | [Issue 3 notes](#evidence-notes-for-numbered-issues), [DeepSeek measurements](#models-and-card-counts) | open | Verify preallocation refusals at both guards and the layout chosen within them. |
| 4. The GPU package requires explicit `-Cpu` to serve without a usable card. MiniCPM5-2B IQ8R passed a CPU first-run request with drafting off. | Windows launcher and packaging | [Issue 4 notes](#evidence-notes-for-numbered-issues), [RC4 evidence](#alpha5-rc4-qualification) | preview | Check no-card refusal on a CPU-only Windows host, then repeat the local first run and exercise the text and tool APIs with `-Cpu`. |
| 5. Speculative and plain decoding can differ near a tie. Exact verify covers only selected layers. | Speculative decoding | [Issue 5 notes](#evidence-notes-for-numbered-issues) | open | Compare greedy tokens and logprobs with drafting on and off at fixed shapes, including long prompts and exact verify. |
| 6. Dashboard load and power require AMD ADL. | Windows dashboard | [Issue 6 notes](#evidence-notes-for-numbered-issues), [Dashboard behavior](#upstream-features) | open | Check packaged browser display with ADL available and unavailable. |
| 7. Some archive and console tools cannot handle paths outside the active code page. | Windows packaging | [Issue 7 notes](#evidence-notes-for-numbered-issues) | open | Repeat unpack, configure and serve from a Unicode path with the documented ZIP tool. |
| 8. Flash-Next output can vary with shared steps and prefix-cache state. | Model numerics | [Issue 8 notes](#evidence-notes-for-numbered-issues) | open | Retain fixed batch and cache comparisons and replay a client workflow with background requests. |
| 9. Fleet startup warm-up can leave both members of an exclusive group resident. | Fleet scheduler | [Issue 9 notes](#evidence-notes-for-numbered-issues) | open | Reproduce request admission during a partner's warm-up and verify single residency if the scheduler changes. |
| 10. The MiniCPM5-2B IQ8R DSpark drafter is unavailable on CPU kernels. | CPU kernels | [Issue 10 notes](#evidence-notes-for-numbered-issues), [RC4 evidence](#alpha5-rc4-qualification) | not supported | Verify the missing int8 norm and a CPU launch with drafting enabled once the kernel is implemented. |
| 11. Saturating the host's logical processors can sharply slow decoding. | Runtime performance | [Issue 11 notes](#evidence-notes-for-numbered-issues) | open | Repeat a controlled decode comparison with and without CPU headroom after scheduler changes. |
| 12. Flash-Next TP4 stops are bounded without CPU load beside them: idle, mid-decode, and mid-prefill, both when the request finishes inside the drain and when the drain runs out and ends it. A stop while a large build runs beside the server is untested. | Windows shutdown and HIP | [Observed hard lock and the four-card stop checks](#evidence-notes-for-numbered-issues); [the stop contract](#stopping-a-multi-card-server) and its stop journal | preview | Repeat the mid-prefill stops, including one whose drain runs out, with a large build running beside the server and the stop journal on. Until then, don't stop a four-card server mid-prefill while a large build runs. |
| 13. Some Qwen-Image 2.1 edits produce colour noise. | Image pipeline | [Issue 13 notes](#evidence-notes-for-numbered-issues) | open | Compare the affected seeds, sizes and prompts with the same container upstream on Linux and retain a regression fixture. |
| 14. DeepSeek-V4.1 TP2 matches TP4 in distribution within a measured numeric tolerance (median KL 0.017 at confident positions); greedy text can differ token by token across layouts. | DeepSeek graph and collectives | [Issue 14 notes](#evidence-notes-for-numbered-issues) | explained | Keep the five-layer TP2/TP4 collective and routing regression fixture in the host test suite. |
| Flash-Next long-run stability is unproven on the current profiles. No multi-hour soak covers the current TP2 or TP4 settings. | Windows runtime qualification | [Flash-Next profile coverage and earlier TP4 soak](#models-and-card-counts) | open | Run multi-hour checks for the current settings before claiming long-run stability. |
| Desktop apps that use a serving card can stop `flashnext-4card` from starting: its plan leaves about 445 MiB of VRAM a card, and a browser or editor on an R9700 can take more. Set such apps to Power saving in Windows Settings > System > Display > Graphics so they use another GPU, and restart them before starting four cards. The startup log lists the VRAM other programs hold on each serving card (program, process id, MiB); if they take more while the server loads and that leaves a card short, startup stops with an error naming each one and the reserve that would leave it room, instead of failing later on an allocation. With several cards the error names that card's own `memory.card-headroom-mib` entry (`{CARD: MiB}`), so the busy card keeps more free while the others keep their reserve and, under `expert_tiered`, spend the difference on resident experts. | Windows memory placement | [Model and card coverage](#models-and-card-counts), [Other programs on a serving card](PREVIEW.md#known-limits) | open | Check the report and the refusal on a card with a desktop app holding VRAM (a small model, one card) |
| In 0.3.0, `ngram-placement: auto` on one card could pin Flash-Next's 47.7 GiB n-gram table on top of its expert pool, leaving about 4 GiB of host memory spare; when other processes then took memory, the next GPU call could hang instead of failing (issue 3). Now `auto` keeps at least 24 GiB free beside a RAM copy and checks again just before pinning, reads a table larger than a quarter of RAM from the file (prefetched) on Windows, and the backend refuses any pin larger than the RAM available at that moment. The one-card Flash-Next profiles keep `ngram-placement: disk`. | Windows n-gram placement | [Issue 3 notes](#evidence-notes-for-numbered-issues) | explained | Checked 2026-11-25 on one card: auto chose the file, and 55.9 GiB stayed available while serving |
| Windows service runs as LocalSystem and is a preview in 0.3.1. | Windows service lifecycle | [Preview verification and gaps](SERVICE.md#qualification-in-an-elevated-powershell) | preview | Verify elevated SCM install, key ACLs, readiness, stop and detachment, boot and preshutdown, reinstall and one-restart recovery. |
| Distribution basis for the complete AMD Windows HIP runtime DLL remains unresolved. | Windows runtime packaging | [Runtime inventory and terms](https://github.com/adamczhang/Brilliance/blob/main/win/licenses/THIRD-PARTY.md#amds-windows-hip-distribution-terms), [AMD's Windows build record](https://github.com/ROCm/TheRock/blob/b324869fb3116353ac7eec5e6926aa3d452ac987/docs/development/windows_support.md#building-clr-from-partial-sources) | open | Document terms covering the exact binary and its linked archives, or qualify a runtime with documented distribution terms. |
| LAN policy | HTTP authentication and dashboard | [The network policy](PREVIEW.md#serving-on-the-network-as-a-service-and-reloading); `win/tools/lan-matrix.ps1` (316 cells, key off and on, on the package) | open | Decided and fixed 2026-11-22 (`lan-f3-f4`): without the key, `/health` and `/ping` answer the status alone on a keyed server, a keyless server shows a load failure's reason only to the operator on the machine, and a bare `/ping` on a multi-model profile answers as `/health`. `/ping?model=NAME` reports that model's state as `/health?model=NAME` does and never loads it (`ping-state`). Rerun the matrix on the release package. |
| The Sandbox fleet is preview. Switching to LTX on one card produced a VRAM-admission 503. LTX worked when loaded first in a fresh Sandbox. | Fleet scheduler and dashboard | [RC4 evidence](#alpha5-rc4-qualification), [Sandbox profile](PREVIEW.md#the-profiles) | preview | Repeat the one-card sequence with VRAM readings before LTX admission. Exercise the browser workflows, multi-card entries and reload from the package. |
| Cancelling an active MiniCPM5-2B IQ8R decode on one card is unverified. The local first run and tested text and tool APIs are qualified. | Model qualification | [RC4 evidence](#alpha5-rc4-qualification) | open | Disconnect a streaming client while generation is still active, confirm the engine records an abort and verify that the next request succeeds. |
| LTX-2.5 videos are at most 161 frames (6.7 s) on Windows, as in 0.3.0. Upstream Radiance now allows 321 (13.4 s), but the activation arena is sized from that ceiling, and at 321 frames it does not fit beside LTX-2.5 on one 32 GB card, so the Windows build keeps the lower ceiling and `ltx-2.5-1card` starts as before. | Video generation | [Video generation](#upstream-features) | explained | Raise it when the arena is sized per request, or for a card with more memory. |
| AMD AMF encoding, encoder fallback and refusal with neither encoder available are unverified in the RC4 LTX qualification. Standalone one-card jobs are qualified with Windows Media Foundation H.264. | Video pipeline and packaging | [RC4 evidence](#alpha5-rc4-qualification), [Video feature checks](#upstream-features) | open | Run the one-card jobs with `h264_amf` selected, then verify the default fallback to `h264_mf` and clean refusal when both encoders are unavailable. |
| DeepSeek-V4.1 TP2 and TP4 are text previews limited to one 4K request, with drafting off. | DeepSeek model qualification | [DeepSeek profile checks](#models-and-card-counts) | preview | Verify same-container parity, cancellation, stop, admission and queue behavior within the one-request limit. |
| DeepSeek TP2 dry planning has a separate verification gap. | Windows launcher and DeepSeek placement | [DeepSeek memory placement contract](https://github.com/adamczhang/Brilliance/blob/main/arch/deepseek41/CONTRACT.md) | open | Check that `--plan-only` prints the complete plan and guard verdict or gives a specific refusal without loading the model. |
| Flash-Next TP2 fast profile uses lossy wht6 and opt-in pipelining is scoped to eligible prefill steps. | GPU collectives and model qualification | [Fast profile and pipeline checks](#models-and-card-counts) | preview | Repeat the profile's parity and fault-recovery probes on eligible steps and qualify the full serving profile. |
| Flash-Next TP1 and layouts at TP3 and TP8 have no Windows run evidence. | Tensor parallel qualification | [Model and card coverage](#models-and-card-counts), [Tensor parallel coverage](#upstream-features) | open | Check admission within the applicable pin and VRAM budgets, then verify serving and stopping at each claimed card count. |
| Context and concurrency limits share a single KV budget. Flash-Next TP2 has room for about one full-length context. | KV cache and admission | [Context and slot measurements](#context-slots-and-the-kv-cache) | open | Verify startup cache totals, one maximum-length request and admission under mixed prompt lengths. |
| GPU package targets gfx1201 only. | Windows build and HIP | [Windows GPU coverage](PREVIEW.md#before-you-start-your-hardware) | not supported | Qualify a package built for any additional advertised GPU target, including device kernels and model serving. |
| The packaged workflow requires Windows 11. | Windows packaging and qualification | [Runtime requirements](PREVIEW.md#before-you-start-your-hardware) | not supported | Verify install, launch and shutdown on any additional Windows version before listing it. |
| Radeon RX 9070 and driver versions beyond the tested Adrenalin version are unverified. | Windows hardware qualification | [Tested hardware and driver](PREVIEW.md#before-you-start-your-hardware) | open | Repeat first run and stop on an RX 9070 with a model that fits and on each newly claimed driver. |
| Peer-to-peer and the upstream P2P-only two-card collectives are unavailable on Windows. | Windows HIP and collectives | [Transport and collective support](#upstream-features) | not supported | Validate peer-access capability, direct transfers and collective parity before enabling those paths. |
| Linux AQL launch optimizations are unavailable on the native HIP backend. | Windows HIP | [Backend capability comparison](#upstream-features) | not supported | Verify equivalent native HIP paths and compare output and decode performance before claiming parity. |
| Automatic process exit on a stall (`scheduler.stuck-exit`) is unavailable on Windows. Stall status depends on a blocked host wait. | Windows stall watch and lifecycle | [Hang detection behavior](#upstream-features) | not supported | Verify real-hang detection and bounded shutdown or recovery before enabling automatic exit. |
| Shared n-gram RAM placement has device tests. Windows disk placement uses mapped reads, and warming skips when headroom is low. | Windows n-gram placement | [Placement, headroom and warm measurements](#upstream-features) | open | Check a shared-table run end to end and verify skip behavior and output parity across warm and cold starts. |
| Speech synthesis and music generation are unavailable. | Windows audio generation | [Endpoint refusals](#upstream-features) | not supported | Qualify the audio route and its streaming behavior with supported speech and music models. |
| JPEG XL input is unavailable in the bundled FFmpeg build. H.264 decoding carries no patent licence from the package. | Media packaging | [Input coverage](#upstream-features), [H.264 component record](https://github.com/adamczhang/Brilliance/blob/main/win/licenses/THIRD-PARTY.md#h264) | not supported | Qualify a JPEG XL-enabled build before advertising it and verify that package notices match the H.264 components. |
| Image generation returns base64 rather than URLs. SDXL edits are unavailable. | Image API | [Image format and endpoint checks](#upstream-features) | not supported | Exercise URL responses or SDXL editing end to end before advertising either feature. |
| Media inputs and APIs depend on the loaded model. Gemma 4 31B and Qwen3.6-35B refuse audio. | Model capability checks | [Per-model coverage](#models-and-card-counts), [Endpoint behavior](#upstream-features) | not supported | Verify the declared input modalities, successful requests and clean refusals for each profile offering media. |
| Qwen3.6 can repeat a required tool call unless parallel calls are disabled. | Tool calling and model templates | [Qwen3.6 profile checks](#models-and-card-counts) | open | Retain required-tool regression coverage for both parallel calls and single-call requests. |
| Grammar and infill coverage is limited. Qwen3.8 chat refuses GBNF, regex is unavailable, and infill needs model tokens and can expose a separator. | Structured output and completion APIs | [Grammar and infill checks](#upstream-features) | open | Validate supported formats, chat-template refusal and fill-in-the-middle end markers on the claimed models. |
| EmbeddingGemma refuses inputs over 8,192 tokens. Nemotron rerank refuses pairs over 32,768 tokens. | Embeddings and rerank | [Endpoint limits](#upstream-features) | not supported | Verify boundary acceptance and over-limit refusal for each advertised model and input type. |
| Drafting options vary by model. Only Flash-Next applies runtime control vectors. | Architecture plugins | [Drafting and control-vector checks](#upstream-features) | preview | Qualify each advertised option on the supported profiles and exercise startup refusals on other models. |
| A standalone runtime `draft-model` is refused. | Speculative decoding and configuration | [Settings reference](https://github.com/adamczhang/Brilliance/blob/main/REFERENCE.md#speculative-decoding-drafting) | not supported | Validate startup refusal and a converted container with its drafter included. |
| Runtime LoRA on rotated int8 and OpenBLAS-backed GPTQ conversion are unavailable. | Windows plugin and quantization builds | [LoRA and GPTQ build limits](#upstream-features) | not supported | Port and test the respective Windows builds, including adapter application and GPTQ parity. |
| Windows `rad-convert` is CPU-only and refuses `--gpus`. Export requires Python, with extra packages for LoRA folding. | Conversion and export tools | [Conversion and export coverage](#upstream-features) | not supported | Revalidate the packaged CPU conversion and export on a clean host. Qualify GPU conversion before making a GPU claim. |
| A repository ID with several containers is refused; choose one file. Failed transfers stop loading unless a complete existing copy can be used. | Model fetch and launcher | [Repository fetch and fallback rules](PREVIEW.md#known-limits) | open | Check cold fetch, interrupted resume, multi-container refusal, revision pinning and the DeepSeek split-file checksum flow. |
| Legacy and incompatible containers need replacement or conversion. | Model loading and conversion | [Container and checkpoint compatibility](#find-your-model-file) | not supported | Exercise startup refusals and successful conversion or load for each newly advertised format. |
| Reload applies model changes. Changes to `server:` or `cards:` require a restart. | Fleet configuration | [Windows reload behavior](PREVIEW.md#serving-on-the-network-as-a-service-and-reloading) | not supported | Verify that model changes apply and that a rejected server or card change leaves the active configuration intact. |
| Plugin SDK requires a source build. Linux scripts and Docker tooling are outside the Windows package. | Windows packaging and SDK | [Package tooling coverage](#upstream-features) | not supported | Validate any newly shipped tools or SDK from a clean unpack before listing them as package features. |
| Streaming KV, sidecar LoRA, decision models and 3-D shape models have no Windows run evidence. | Windows model qualification | [Upstream feature coverage on Windows](https://github.com/adamczhang/Brilliance/blob/main/README.md#feature-parity) | open | Run end-to-end fixtures for each feature before claiming Windows support. |
| Other upstream models and unlisted profile layouts or modalities are unqualified. | Windows model qualification | [Model coverage](#models-and-card-counts), [Container coverage](#find-your-model-file) | open | Check the exact container, profile, card count and modalities before extending support. |
| Fresh ZIP checks and named client round trips cover a limited host and client set. | Windows first-run and client integration | [Client setup record](PREVIEW.md#connecting-a-client), [Tested Windows configuration](PREVIEW.md#before-you-start-your-hardware) | open | Repeat first run, browser rendering and tool round trips with the release ZIP on a clean Windows host. |
| Published throughput describes the measured model, profile and hardware settings. | Runtime performance | [One-card measurements](PREVIEW.md#what-was-measured-one-r9700-at--20-board-power--50-mv-vram-at-its-stock-2518-mhz), [Multi-card profile evidence](#models-and-card-counts) | open | Repeat the fixed protocol on the release package with matched inputs, tuning and card links before extending any performance claim. |
| Startup reports PCIe speeds at card links; upstream bridge speeds can show `?`. | Windows hardware reporting | [PCIe reporting limit](PREVIEW.md#known-limits) | open | Check bridge readings against a known topology if Windows exposes that data. |
| Gated downloads and model usage or redistribution conditions depend on the individual model licence. | Model packaging and licence notices | [Model licence records and access conditions](PREVIEW.md#what-is-in-this-package) | open | Check the upstream licence and the package's notices and terms for each shipped profile. |

### Evidence notes for numbered issues

1. **Flash-Next on two cards, with experts read from host memory, loses a write now and then.** In
   an all-reduce, a 64-byte write from one GPU to host memory is lost or misdirected. Where the
   write goes wrong is still open (below Brilliance's code).
   - **It is recovered.** The two-card host exchange of the exact wire acknowledges each block,
     and the writer re-sends a wrong one, so the step completes.
   - **`tp-wire: wht6`**, upstream's own two-card setting, is covered the same way: its two-card
     host collectives re-send a wrong block and stop loudly past the limit. The recommended safe
     profile uses the exact wire. The fastest profile opts into wht6.
   - **What it costs:** about 4% of single-stream decode and about 2% at 4 streams; prefill is
     unchanged. `BRILLIANCE_AR_RESEND=0` turns the resend off. On the exact wire the stop below
     stays; with `tp-wire: wht6` nothing is left to catch a lost write, and a request can hang.
     Keep the default.
   - **A block that cannot be recovered stops the server** within about 2 s (30 s for a missing
     flag). The process exits with status 70, its last line `radiance: fatal:` naming the fault.
     The preview configures SCM to restart the service after a first failure. Recovery from this
     exit remains unverified under SCM. A hang that leaves the process running never triggers
     recovery (see "Hang detection" above). `/stats` (`host_mailbox`) counts faults and resends
     per collective.
     During a long prompt's prefill the fault is noticed at the step that hit it, not only at the
     prompt's last step: the request fails with no token sent. A fault that only a cancelled
     request could have seen stops the server before the next request runs.
   - **The opt-in faster prefill** (`BRILLIANCE_TP2_PIPELINE=1`, "Models and card counts") sends
     its exchange through the same checked host collectives: a lost write is re-sent and one past
     the limit stops the server (status 70), on the exact wire and on wht6.
   - **The data slots use non-coherent memory** by default on two and four cards, for every
     model, which makes the fault rarer; the flags stay coherent. `BRILLIANCE_MAILBOX_MEM`
     overrides (`default` or `coherent` gives coherent slots).
   - **Four cards have the same protection:** the four-card host collectives (one-shot, gather
     and two-shot) check each block, re-send a wrong one and stop loudly past the limit;
     `BRILLIANCE_AR_RESEND=0` leaves only the stop. With every expert on the cards no real fault
     has been seen there. On Flash-Next at four cards the checks cost about 4% of single-stream
     decode, 4-5.5% at 4 and 8 streams, and 7-11% of prefill, against a build without them.
   - **The minute log line counts flag/ack re-stores,** control words re-sent when an
     acknowledgement is slow; they carry no data and are not resends. With the Flash-Next
     profiles' 6,144-token prefill steps the counts run higher: on four cards about six times
     the figure at 2,048-token steps (tens of thousands every five minutes under long prompts),
     on two cards about the same. This is harmless: a re-store never counts toward the stop.
2. **HIP virtual memory only with one card visible.** With exactly one device visible to the
   process it is on since 2026-11-17: the KV cache is elastic and a model that makes room is
   parked. Parking keeps no host copies (the backend has no host-backed mappings), so a parked
   model's weights are read from its container again and other memory goes to the disk tier
   where one is set. With more than one device visible (several cards, or a card and an
   integrated GPU, which `serve.ps1` hides unless told otherwise) it is off: the runtime commits a reservation's whole size at its first mapping
   (rocm-systems#13232). The KV cache is then sized whole at load, nothing is parked, and a
   switch reloads from the file. `BRILLIANCE_VMM=0` turns it off; `=1` forces it on and is not
   qualified with several devices.
   - **In a fleet of several models on more than one card, give each model `gpu-memory-mib`.**
     Without HIP virtual memory, the fleet's first claim for a model is final, and it is the
     model's floor, not its whole need. Qwen3.8-Flash-Next, its experts tiered to host memory,
     then does not fit them and fails to load with "out of memory"; alone it gets the whole
     card. Upstream documents the same: without virtual memory a card hosts one model with static
     pools. A two-model fleet on two cards, checked 2026-11-19 (repeated switching, queued
     requests, a pinned model's refusal):

     ```yaml
     models:
       fn:
         model: D:/models/hf/StillDeadcode/qwen3.8-next-flash-iqr-v4/qwen3.8-next-flash-iqr-v4.rad
         tp: 2
         max-num-seqs: 4
         max-model-len: 32768
         placement: expert_tiered
         host-pool-mib: 16384
         gpu-memory-mib: 30720     # 30 GiB a card
       q27:
         model: D:/models/hf/StillDeadcode/qwen3.8-27b-iqr-v4/qwen3.8-27b-iq8r-v4.rad
         tp: 2
         max-num-seqs: 4
         max-model-len: 32768
         gpu-memory-mib: 28672     # 28 GiB a card
     ```

     The two claims together exceed a card, so the fleet holds one model at a time and a switch
     unloads one and loads the other.
3. **The pin limit.** A process can pin about the shared-memory budget (half of RAM or more) divided
   by the cards it sees, less 2 GiB. The planner accounts for it since 2026-11-17 and places what
   does not fit elsewhere. `BRILLIANCE_PIN_LIMIT_GIB` can only lower it: a value above that
   figure is capped at it, with a warning naming both, because past it a pin does not fail and
   the next GPU operation hangs instead (since 2026-11-19).
4. **The GPU package without a card** serves on the CPU with `serve.ps1 <profile> -Cpu`. That
   applies what the CPU needs over the profile: the host backend with the HIP runtime never asked
   for a device (`BRILLIANCE_DEVICE=host`, with every card hidden from HIP as well), the CPU kernels
   (`kernels: libavx,libref`; the gfx1201 libraries cannot run on the CPU, and without the setting
   the model fails to declare: `rad_arch_declare returned invalid argument`) and no speculative
   decoding (known issue 10). Without `-Cpu`, `serve.ps1` refuses a start that would leave the
   server no card, so a failed card never turns into a CPU server unasked. Started without
   `serve.ps1`, the GPU package needs `kernels: libavx,libref` under `server:` and
   `HIP_VISIBLE_DEVICES=-1`: with a card visible and only the `kernels` setting, the GPU build
   refuses to start ("no device kernel"). The CPU package needs no setting.
5. **Speculative decoding (DFlash2, MTP) and plain decoding** can differ in greedy text where two
   tokens are almost tied, as upstream documents: the verify pass and the single-token pass round
   differently. Both are the model's answer. `BRILLIANCE_EXACT_VERIFY=1` (off by default) makes the
   IQR 5-bit row-scaled layers (and short MoE gate/up batches) verify exactly as single-token
   decoding does. It costs about 36-41% of DFlash2 single-stream decode on Qwen3.8-27B IQ4R-L-V4
   and does not cover every layer, so it narrows such differences without removing them. A `draft-suffix` step is one such case: with the switch on, some short Flash-Next probes still parted from the head-only run.
   On Qwen3.8-Flash-Next with two cards (`flashnext-2card`, 2026-11-21), the shipped MTP drafting
   and decoding without drafts gave the same greedy text on 2 of 8 prompts. Every token was the
   top choice of the pass that computed it, and where the texts parted, the two leading tokens
   were within 0.02-0.13 nat of each other. Decoding without drafts changes as much when other
   requests share its steps (0 of 8 the same as alone; known issue 8), so neither text is the
   reference. With `BRILLIANCE_EXACT_VERIFY=1`, 6 of 8 were identical, logprobs included; the
   prompts of about 28K and 57K tokens still parted.
6. **The dashboard** reads card load and power only through the AMD driver's ADL. Windows' own GPU
   counters do not see HIP's work, so a card without ADL shows "not available on Windows" for them.
7. **Folder names outside the system code page.** `radiance.exe` and the package's scripts use
   UTF-8 throughout: the package was checked unpacked into, and serving from, a folder whose name
   has spaces and non-ASCII characters. Other tools (some archive and console tools) can still
   depend on the Windows version and the active code page. If one cannot unpack or reach such a
   folder, unpack with Explorer or another Unicode-capable ZIP tool.
8. **Flash-Next's output can change when requests share a step.** A request computed together with
   others in one step can get a slightly different next-token distribution than the same request
   sent alone, and identical requests in the same step can differ from each other, so sampled output
   varies more than the temperature alone explains. This is rounding, not wrong data: the expert
   kernel takes a shorter path for an expert's last one or two rows, and other kernels choose their
   shape from the step's size. Upstream Radiance documents the same for batched requests. Flash-Next
   magnifies these last-bit differences more than most models; Qwen3.8-27B is barely affected.
   Sending one request at a time removes the shared step, but not every difference: a long prompt
   sent a second time and resumed from the prefix cache is computed in different chunks, and on two
   cards two long greedy prompts (about 28K and 57K tokens) came back with different reasoning text
   and the same final answer. To compare outputs, compare requests sent alone from the same cache
   state. Clients that send side requests of their own (titles, summaries) change what shares a
   step.
9. **An `exclusive` group can end up with both models resident** when a request loads one while
   the start-up warm-up is still loading the other: a partner still loading is not yet counted.
   This is upstream's fleet, on every platform. For strict turns, set `scheduler: warm-up: false`.
10. **MiniCPM5-2B IQ8R on the CPU kernels needs its drafter off.** The container carries a DSpark
    drafter, and speculation is on by default. The CPU kernels have no form of the drafter's int8
    norm, so the model does not start: `no loaded kernel plugin declares op
    'rmsnorm_had_quant_i8g'`. `serve.ps1 -Cpu` turns speculation off for every model. Started
    without it, set `num-speculative-tokens: 0` in the model's block of the YAML file (the engine
    takes model settings from the YAML, not from command-line flags); the model then serves on the
    CPU with plain decoding,
    without the drafter's speed-up. Upstream's CPU kernels have the same gap. On a card the drafter
    runs.
11. **Leave CPUs free for the server.** Decoding needs only a few CPUs, but it needs them on time.
    With at least 8 of the machine's logical processors free, other work costs decode a few percent
    at most. When every logical processor is busy (a parallel build or test run with `-j` at the
    processor count or more), decode becomes 15-40 times slower, even though the server runs at
    Above Normal priority (`BRILLIANCE_PRIORITY`, the default). On a 32-thread machine, run builds
    and tests with `-j 24` or lower beside a serving card, or confine them to some cores (for
    example `start /affinity`, or a job object). `BRILLIANCE_PRIORITY=high` is available but was
    not measured against this. Never use realtime.
12. **Stopping `flashnext-4card`: bounded without CPU load beside it, untested beside a large
    build.** The server once hard-locked the four-card test machine: it was stopped with Ctrl-Break
    during the prefill of a prompt of about 120K tokens while a large build and its tests ran on the
    CPU; the display went black with the fans on, and a reset was needed. Windows left no crash
    dump, driver reset (TDR) or hardware error event. It was seen once and not reproduced. A stop
    could then leave a step issued by some ranks and not others; since 2026-11-22 every rank issues
    a released step or none does ("The rules the order keeps", below).
    - **Checked on four cards since that fix,** with the stop journal on and nothing else running on
      the machine: stops while idle, mid-decode, mid-prefill on prompts up to 200K tokens whose
      request finished inside the 30 s drain, and mid-prefill on a 250K-token prompt whose drain ran
      out, so the open request was ended while its prefill chunks were still being released. Every
      stop was bounded: each step within its bound, every rank leaving at the same step, every
      stream drained, memory returned, exit status 0, about 35 s at the longest, and no driver,
      hardware or error event.
    - **A request ended by a stop** whose drain ran out gets HTTP 429 `server_overloaded`: "this
      model is stopping or leaving its cards; send the request again". A request that finishes
      inside the drain gets its whole answer.
    - **Untested: a stop with a large build running beside the server,** the hard lock's
      condition. Until it is checked, don't stop a four-card server during a long prefill while a
      large build runs beside it: let the prefill finish first, or keep the CPU light (known issue
      11). To record where a stop gets to, set `BRILLIANCE_STOP_LOG` ("Stopping a multi-card
      server", below).
13. **Some Qwen-Image 2.1 edits come out burnt and noisy.** An edit (`/v1/images/edits`) at 1024 pixels
    wide can come out over-saturated and covered in colour noise for some seeds: at 1024x1024, seed 7
    of seeds 1-11 did, with two different prompts; the other ten were clean, and so was seed 7 at the
    default size (2048x2048) and at other sizes (2026-11-20, one card). The decode, colour, VAE,
    cached conditioning and attention were each checked and are not the cause; whether upstream on
    Linux gives the same picture is not yet known. If an edit comes out burnt, retry with another
    seed.
14. **DeepSeek-V4.1 on two cards matches four cards in distribution, within a measured numeric
    tolerance (explained).** `deepseek41-2card` and `deepseek41-4card` serve the same container.
    Across 96 teacher-forced positions, the median KL divergence at confident positions is 0.017
    and the top token agreed at 86, with two flips where default TP4 led by more than one nat
    (1.50 and 1.19 nats). Per-layer tracing of those two positions showed that the difference is
    expected bf16 rounding amplification across 40 layers, not a graph or collective defect:
    - Each rank rounds its attention output projection's partial sum to bf16 before the exact
      all-reduce (TP2 sums two rounded partials, TP4 four), producing a 0.31% RMS difference at
      layer 0 from bit-identical inputs. DeepSeek's fp8 (E4M3) round trips and near-tied expert
      choices (6th-versus-7th router score gaps down to 2e-4, with the first expert swap at layer
      3 or 4) amplify that one-ulp difference across the trunk.
    - Every all-reduce is within 0.5 bf16 ulp of the exact sum of the ranks' partials on every
      layer in both layouts, every router selection matches an exact f64 reference, and device
      kernels agree with the reference kernels through layer 2.
    - Regrouping two experts between ranks within the same card count changes only how the bf16
      partials are grouped, yet diverges just as much: TP4 with two experts regrouped flips both
      confident positions to TP2's tokens, and the TP2-versus-TP4 KL at the two worst flips (0.91
      and 0.35) stays within the spread of same-layout regroupings (up to 1.05 and 0.45).
    - **Guidance:** the checkpoint sets no sampling defaults of its own. Send `temperature` with
      each request, and expect greedy text to differ token by token between TP2 and TP4 just as it
      can between two TP4 layouts.

## Stopping a multi-card server

A stop is Ctrl-C or Ctrl-Break at the server's console, `tools\stop-server.ps1`, or the service's
stop. It runs the same steps in the same order whatever the server was doing, and each step has a
time bound. The bounds below are what a healthy stop keeps; the server does not end itself when one
is passed (only its own five-minute limits do, below), but the stop journal records it.

**What a stop does in each phase.**

- **Idle:** nothing is in flight. The drain finds no open response, the step loop is waiting for
  work and returns at once, and the teardown below follows. A few seconds in all on four cards.
- **Decoding:** the server takes no new request, and the responses already being sent run on, for
  up to `drain-timeout` (30 s by default). What is still open after that is ended. The step in
  flight then completes, and the step loop returns.
- **In the middle of a prompt's prefill:** the same drain. The prefill goes on, one chunk a step,
  until it ends or the drain time is up; then the request is ended, the chunk in flight completes
  on every card (the stop is taken between steps, never inside one), and the step loop returns.
  So a long prompt delays a stop by up to the drain time plus one chunk. A request ended this way
  gets HTTP 429 `server_overloaded` ("this model is stopping or leaving its cards; send the
  request again"); send it again once the server is back.

**What has been checked on four cards** (`flashnext-4card`, with nothing else running on the
machine): stops while idle, mid-decode, mid-prefill with the request finishing inside the drain
(prompts up to 200K tokens), and mid-prefill with the drain running out and ending the request
(a 250K-token prompt). Each was bounded, with every rank leaving at the same step and exit status
0, in about 35 s at the longest. A stop with a large build or test run beside the server is
untested (known issue 12).

**The steps, in order, with their bounds.**

| # | Step | Bound | What it waits for |
|---|---|---|---|
| 1 | The fleet sees the stop | 1 s | its loop's 100 ms tick |
| 2 | History thread joined | 2 s | its 200 ms tick |
| 3 | A model load or restore in progress gives up | 60 s | the load's next check of the stop |
| 4 | Each model's drain | `drain-timeout` (30 s) | the open responses; past it they are ended |
| 5 | The step loop returns | 10 s | the step in flight, at most one prefill chunk or one decode step, issued and finished by every rank |
| 6 | Every rank thread leaves its loop | 10 s | each card's thread finishing what it issued |
| 7 | The listener closes | 5 s | open connections |
| 8 | Each model's teardown, in reverse load order | 60 s each | steps 8a-8h |
| 8a | Each rank's streams checked for drained work | 2 s a rank | nothing: an observation, never a longer wait |
| 8b | Transfer streams synced, expert tiering quiesced | 10 s | copies into the activation arena |
| 8c | Each rank's context shut down, rank by rank | 10 s each | the rank's compute streams, then its device and pinned host arenas are freed |
| 8d | The cards unclaimed | 10 s | |
| 8e | KV stream mirrors and step state freed (pinned host memory) | 10 s | |
| 8f | The scheduler's staging freed | 10 s | |
| 8g | Each rank's program, pools and kernel instances freed | 10 s | the collectives' host mailboxes are freed here, after every rank's streams were synced in 8c |
| 8h | Host-mapped tables freed, plugins released | 10 s | |
| 9 | Remaining buffers swept | 10 s | |
| 10 | The process returns; the runtime libraries unload | | the HIP runtime's own teardown |

**The rules the order keeps.**

- A step is issued by every rank or by none, because a collective issued by some ranks waits on
  the others. The decision is made once for all of them, at the release: once a stop begins, no new
  step is released, and a step already released when it began is issued by every rank, which then
  waits for the others at the step's end before leaving. So a stop that lands just after a release
  delays the stop by that one step. Before 2026-11-22 each rank checked for the stop on its own
  after the release, so a rank that woke late could leave a step its peers had issued; host tests
  at two and four ranks now stop at that moment. The stop journal still flags a rank that left its
  loop after a release without issuing (`SPLIT`), which should never appear.
- No memory a card may still read or write is freed before that card's streams are synced: the
  arenas after their own rank's sync, the collectives' host mailboxes only after every rank's.
- The syncs in 8b and 8c wait as long as the work they wait on. A kernel that never finishes (a
  collective whose peers never came) holds the stop there; the journal names the step and the card.
- The server's own outer limits: a step loop not back five minutes after the stop, or responses
  still held five minutes after they were ended, end the process with `radiance: fatal:` so a
  restart can bring the server back.
- **Send one stop, not two.** A second Ctrl-C or Ctrl-Break ends the process at once, with work
  still on the cards. Never end a working server from Task Manager.

**The stop journal.** Set `BRILLIANCE_STOP_LOG` to a file, or to an existing folder (the journal is
then `radiance-stop-<pid>.log` in it), before the server starts. The server opens the file at start
and writes nothing more while it serves. During a stop it writes one line a step: when the stop
began and why, what each model was doing (idle, mid-decode, mid-prefill, with each request's
prompt length and how much of it was in), each step's begin and end with its time against its bound,
how each rank thread left its loop, whether each rank's streams had drained, and each release. Every
line is written through to the disk before the stop goes on, so if the machine stops responding the
journal still holds every step up to the last one that completed. A step that runs past its bound
is recorded while it still runs, at 1x, 2x, 4x ... its bound, so a step that never ends is named.
`win/tools/stop_log_check.py JOURNAL` reads a journal and answers BOUNDED, UNBOUNDED (a step over its
bound, a rank that skipped a step, a stream not drained) or TRUNCATED (no closing line: the process
or the machine stopped, and the last line and the open steps say where). A drain that timed out and
ended the open responses (the retire) is the designed path, not a failure: the checker lists it as
expected and it does not by itself make a stop UNBOUNDED.

## From Brilliance 0.2.2

- **Models:** 0.2.2's containers do not load; download the current ones (table above).
- **Configuration:** a YAML file (`--config`), as upstream's; `serve.ps1 <profile>` replaces 0.2.2's
  launcher flags. `--check` validates a file without loading.
- **Windows service (preview):** `serve.ps1 -Install` now uses a YAML profile. The installer
  is designed to put the API key in a protected file and pass the file path on the command line.
  See the verification status in `SERVICE.md`.
- **Stopping:** Ctrl-C / Ctrl-Break drain, as in 0.2.2; never end a working server from Task
  Manager.
