# What Brilliance 0.3 does on Windows

The support record for the Windows build of Radiance 2.2: which models run on how many cards,
which of upstream's features work, the known issues, and what changes for a Brilliance 0.2.2
user. It is kept with the source and ships beside `PREVIEW.md`.

Measured on the project's test machine: four Radeon AI PRO R9700 (gfx1201) at -20% board power
(240 W), -50 mV and the stock VRAM clock (2518 MHz), Windows 11, Adrenalin 32.0.32015.12012.

## Models and card counts

| Model (container) | 1 card | 2 cards | 4 cards |
|---|---|---|---|
| Qwen3.8-27B IQ4R-L-V4 (`StillDeadcode/qwen3.8-27b-iqr-v4`) | **Qualified**: 8K and 32K x 4, DFlash2 or plain, bf16 or i8 KV; text, tools, pictures, video | Not run | Not run |
| Qwen3.8-Flash-Next IQR-V4 (`StillDeadcode/qwen3.8-next-flash-iqr-v4`, 113 GiB) | Not run. Upstream lists a one-card layout (experts tiered to a 43,000 MiB host pool, `deploy/compose/tp1.radiance.yaml`); whether it fits Windows' one-card pin limit is unverified | **Supported**: upstream's two-card settings, tuned, on the exact wire (8 sequences, 256K, int8 KV, MTP, prefix cache, experts tiered to host memory, 6,144-token prefill steps). The eight requests share a KV cache of about 262K tokens: one full-length context fills it ("Context, slots and the KV cache"). A lost write is re-sent, at about 4% of single-stream decode (known issue 1); the re-send and stop were checked at 6,144-token steps. An 8-hour soak has not been run on it yet. Profile: `flashnext-2card` | **Supported**: every expert on the cards (`placement: all_vram`); soaked at 32K x 8, int8 KV, MTP on its earlier settings (2026-11-17), before the four-card re-send and stop (known issue 1). Its tuned settings, 256K x 8 (a shared KV cache of about 919K tokens, 3.5 full-length contexts), a bf16 KV cache and 6,144-token prefill steps, pass the injection, stop and serving checks with them. A soak on them has not run yet. Profile: `flashnext-4card` (known issues 8 and 12) |
| MiniCPM5-2B IQ8R (`StillDeadcode/minicpm5-2b-iq8r`) | Runs (bring-up model); also on the CPU kernels without its drafter: `serve.ps1 minicpm5-2b-1card -Cpu` turns it off (known issue 10) | - | - |
| Qwen3.8-27B IQ8R-V4 (`StillDeadcode/qwen3.8-27b-iqr-v4`, `qwen3.8-27b-iq8r-v4.rad`, 27.58 GiB) | **Soak-clean**: 16K x 4, int8 KV, DFlash2 (26.67 GiB of weights leave about 3 GiB of cache: 55,616 tokens, 3.4 full-length contexts). Profile: `qwen27-iq8r-1card` | **Soak-clean**: 32K x 4, DFlash2. Profile: `qwen27-iq8r-2card` | Not run |
| EmbeddingGemma 2 bf16 (`google/embeddinggemma-2`, converted with `rad-convert`, 1.40 GiB) | **Qualified**: `/v1/embeddings` for text, pictures, video and audio, up to 8,192 tokens an input, 64 sequences, on one card. Also checked on the CPU kernels: across 17 comparison vectors the CPU and the card agreed to a cosine similarity of at least 0.999866 (close, not identical) | - | - |
| Qwen3-ASR 1.7B bf16 (`Qwen/Qwen3-ASR-1.7B-hf` as it ships, 4.09 GB) | **Qualified**: `/v1/audio/transcriptions` (5 formats, 4 response forms, past 30 s cut at pauses) and `/v1/realtime`, 16 sequences | - | - |
| Nemotron 3.5 Rerank 8B BF16 (`nvidia/Nemotron-3.5-Rerank-8B-BF16` @ f88f8d91 as it ships, 16.78 GB) | **Qualified**: `/rerank` and `/v1/rerank`, text and picture documents, 32,768 tokens a pair | - | - |
| SDXL base 1.0 bf16 (`stabilityai/stable-diffusion-xl-base-1.0`, the single-file checkpoint converted with `rad-convert`, 6.39 GiB) | **Qualified**: `/v1/images/generations`; a seed repeats its picture byte for byte; `/v1/images/edits` answers 501 | - | - |
| LTX-2.5 distilled IQ8R (`StillDeadcode/radiance-ltx-2.5-distilled-iq8r`, `ltx-2.5-distilled-iq8r.rad`, 36.86 GiB; gated, LTX-2.x Community License) | **Runs, not yet qualified**: `/v1/videos`, text to video and image to video (`input_reference`), MP4. Its Gemma text encoder (14.00 GiB) and token embedding (1.88 GiB) are pinned in host memory: the DiT and the activation arena fill the card. A profile of your own may leave out `vision-tower-placement: ram`: the engine then moves the encoder to host memory itself and says so in the log. Profile: `ltx-2.5-1card` | - | - |
| Gemma 4 31B a4 (`keptn/gemma4-31b-a4`, `gemma4-31b-a4.rad`, 19.95 GiB) | **Qualified**: chat, Responses and Anthropic Messages, whole and streamed; tools; pictures and video; 32K x 2, bf16 KV, MTP; thinking off by the profile, as for the 12B (a request turns it on). Audio is refused with a 501. Speed not yet measured. Profile: `gemma4-31b-1card` | Not run | Not run |
| Gemma 4 12B i8g (`keptn/gemma4-12b-i8g`, `gemma4-12b-i8g.rad`, 13.79 GiB) | **Qualified**: chat, Responses and Anthropic Messages, whole and streamed; tools; pictures, audio and video; 32K x 4, bf16 KV, MTP; thinking off by the profile, as the container's card sets it (a request turns it on). Speed not yet measured. Profile: `gemma4-12b-1card` | - | - |
| Qwen-Image 2.1 IQ8R (`StillDeadcode/radiance-qwen-image-2.1-iq8r`, `qwen-image-2.1-iq8r.rad`, 16.13 GiB; Qwen's research licence, non-commercial) | **Qualified** for generation: `/v1/images/generations`, PNG, JPEG and WebP; a seed repeats its picture byte for byte; `n` gives seeds s, s + 1, ...; sizes from 512x512 to the default 2048x2048 on one card, a size off the model's grid rounded down; `response_format: url` is refused with a 400. Edits (`/v1/images/edits`) work for most seeds; see known issue 13. Profile: `qwen-image-1card` | Not run | - |
| Qwen3.6-35B-A3B IQ8R (`StillDeadcode/qwen3.6-35b-a3b-iq8r`, `qwen3.6-35b-a3b-iq8r.rad`, 35.04 GiB) | - | **Qualified**: chat, Responses and Anthropic Messages, whole and streamed; tools; pictures and video; 32K x 4, int8 KV, MTP; the weights fit the two cards, so no expert is read from host memory. It thinks before answering by default: a client's token limit counts the reasoning too. With `tool_choice: "required"` it can repeat its call; send `parallel_tool_calls: false` for exactly one. Audio is refused with a 501. Speed not yet measured. Profile: `qwen36-35b-2card` | Not run |
| DeepSeek-V4.1-Flash-1017, bakon3's uncensored container (`bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad`, `deepseek-v4.1-flash-1017-uncensored.rad`, 251.13 GiB with a 103 GiB Engram table; IQ1_S and IQ2_XXS routed experts; MIT) | - | **Runs, not yet qualified** (preview): text; plain completions and chat with thinking on and off; 4K x 1, bf16 KV, no drafting, the exact wire. The routed experts do not fit two cards: each card holds what fits, then a 20 GiB pinned pool a card, and the rest (about 9 GiB a card) is read once from the file at load, repacked and kept in ordinary RAM, with the file as the fallback; nothing was read from the disk while it served. About 61 GiB less host memory is available while it serves; the start refuses, before it allocates, when its peak would leave less than 24 GiB available or pin more than the process may (`--plan-only` prints the plan and the check). Its greedy text can differ from four cards' (known issue 14). Profile: `deepseek41-2card` | **Runs, not yet qualified** (preview): text; plain completions and chat with thinking on and off; 4K x 1, bf16 KV, no drafting, the exact wire; every routed expert on the cards (about 29.7 GiB a card), the Engram table read from the file as each step needs it, about 21 GiB less host memory available while it serves. The checkpoint sets no sampling of its own: send `temperature` with each request. No soak, stop or concurrency checks yet; not run in the Sandbox. Rank 0 follows the runtime's device order. Profile: `deepseek41-4card` |
| Other upstream models | Not qualified on Windows. The package carries many of upstream's architecture plugins, but no other model and modality has been run here ("Find your model file", below). Speech-only and music-only models cannot serve in this build (speech synthesis and music are refused, "Upstream features") | | |

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
| `qwen3.8-next-flash-iqr-v4.rad` | `StillDeadcode/qwen3.8-next-flash-iqr-v4` | **Supported** on `flashnext-2card` (safe) and `flashnext-4card`. **Runs** on `flashnext-2card-fast` (fastest). 1 card: not run | Chat, streaming; upstream 2.2's drafting options and control vectors |
| `minicpm5-2b-iq8r.rad` | `StillDeadcode/minicpm5-2b-iq8r` | 1 card and the CPU kernels: **Runs** (bring-up model). Profile `minicpm5-2b-1card` (on the CPU: `-Cpu`) | Chat, streaming |
| `ltx-2.5-distilled-iq8r.rad` | `StillDeadcode/radiance-ltx-2.5-distilled-iq8r` | 1 card: **Runs, not yet qualified**. Profile `ltx-2.5-1card` | `/v1/videos` |
| `deepseek-v4.1-flash-1017-uncensored.rad` | `bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad` (reassembled from its six parts) | 2 and 4 cards: **Runs, not yet qualified**. Profiles `deepseek41-2card`, `deepseek41-4card` | `/completion`, Chat Completions, thinking on and off |
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
| `flashnext-4card` | 8 x 256K | 919,452 tokens | 3.5 |

So `flashnext-2card` runs eight requests at once only while their contexts together fit about 262K
tokens; one full 256K context fills its cache. When the running requests need more than the cache
holds, the newest wait for room, and upstream's default lets a long prompt that fills the cache
partway be set back and computed again from its start. A machine with other cards, other memory or
other settings gets a different cache: read the `kv:` line from its own start.

## Upstream features

| Feature | On Windows |
|---|---|
| Chat Completions, Responses, Messages; streaming; tools; reasoning | Yes (qualified on Qwen3.8-27B) |
| Grammars and structured output (`response_format`); `/v1/completions` and the llama-server routes (`/completion`, `/infill`) | Yes, checked on Qwen3.8-27B IQ4R-L-V4, one card (2026-11-19); not qualified. Grammars and structured output (`response_format` json_object and json_schema, llama-server's `json_schema`, GBNF on `/v1/completions` and `/completion`), tool calls under their parameter schemas, `/v1/completions`, `/completion`, `/completions` and `/infill`. A GBNF `grammar` on chat is refused on Qwen3.8 with a 400 (its chat template produces no grammar for it): use `response_format` or `/v1/completions`. There is no regex constraint (upstream has none). `/infill` needs a model with fill-in-the-middle tokens (Qwen3.8 and MiniCPM5 have them). Its `content` can end with the separator token's text (`<\|file_sep\|>`), which upstream does not treat as an end of generation as llama-server does: a client should cut the completion there. The host grammar tests pass (`gbnf_test`) |
| Pictures and video in requests (`image_url`, `video_url`) | Yes, through the bundled FFmpeg 8.1 (LGPL); checked on Qwen3.8-27B. H.264 input is decoded by FFmpeg's built-in decoder; H.264 decoding is patent-encumbered, and no H.264 patent licence is supplied (`share/doc/radiance/third_party/THIRD-PARTY.md`, "H.264"). JPEG XL pictures are refused with a 400 ("JPEG XL images are not supported by this build; send PNG, JPEG or WebP"), by their bytes or an `image/jxl` data URL: the bundled FFmpeg is built without libjxl |
| Audio in requests, `/v1/audio/transcriptions`, realtime | Audio in an embeddings request: yes (FFmpeg decodes it; checked on EmbeddingGemma 2). Transcription and realtime: yes on Qwen3-ASR 1.7B, one card (2026-11-18). `/v1/audio/transcriptions` takes wav, mp3, flac, ogg and m4a uploads, each word for word the WAV's transcript, and answers as json, text, verbose_json or a stream alike; `/v1/realtime` (OpenAI's protocol over a WebSocket) settles words while the audio streams and gives the transcript about 0.075 s after the commit, equal to the upload route's; 24 kHz audio is resampled. A model without the endpoint answers 501 with the reason (checked) |
| Image generation (`/v1/images/generations`) | Yes on SDXL base 1.0 bf16, one card (2026-11-18), in these tests: converted on the CPU with `rad-convert` from Stability's single-file checkpoint and CLIP's `tokenizer.json` (9.6 s, 6.39 GiB written). PNG, JPEG and WebP (FFmpeg) as `b64_json`; `response_format: url` is refused with a 400; the same seed repeated its picture byte for byte; `n` 1-8 gave seeds s, s + 1, ...; a size off the model's grid was rounded down (1000x700 gave 992x672); about 5.4 s for one 1024x1024 picture at 28 steps. `/v1/images/edits` answers 501 on SDXL, which makes pictures from text only. Qwen-Image 2.1 IQ8R is also **Qualified** on one card, for generation ("Models and card counts"); its edits work for most seeds (known issue 13) |
| Video generation | Works on one card with LTX-2.5 distilled IQ8R, the model for 0.3.0; not yet qualified. Videos are MP4 (H.264 and AAC). By default they are written through AMD's encoder (`h264_amf`), else Windows' (`h264_mf`); image to video (`input_reference`) uses the same two. `BRILLIANCE_H264_ENCODER` (a comma-separated list of those two names) can restrict them or change their order; no other H.264 encoder is ever used. `h264_amf` needs the AMD driver's AMF runtime (`amfrt64.dll`, installed with the driver) and opens its device on Windows' default display adapter: on the test machine, which has an AMD integrated GPU, it encoded on the integrated GPU, not on the card that ran the model. `h264_mf` needs Windows Media Foundation's H.264 encoder, which Windows "N" editions lack until Microsoft's Media Feature Pack is installed. With neither, a video fails with the reason. Every MP4 says it is AI-generated in its metadata (`comment`: `ai_generated=true; model=...`, and `description`); there is no visible watermark. No H.264 encoder implementation ships in 0.3.0 (`share/doc/radiance/third_party/THIRD-PARTY.md`, "H.264") |
| Speech synthesis (TTS) | Not supported in 0.3.0. The Windows build refuses `/v1/audio/speech`, `/audio/speech` and `/v1/audio/speech/stream` for every model: the POST routes answer 501 (`not_implemented_error`, "not supported in this build"), the WebSocket an error event and close 1008 |
| Music generation | Not served: it shares `/v1/audio/speech`, which the Windows build refuses (above). It needs MiniMax-Music3, which is not run on Windows |
| Embeddings | Yes: `/v1/embeddings` on EmbeddingGemma 2, on one card and on the CPU kernels. Text, pictures, audio and video inputs; batches; `dimensions`; base64; concurrent requests each get their own vector; an input past 8,192 tokens is refused with a 400 |
| Rerank | Yes: `/rerank` and `/v1/rerank` on Nemotron 3.5 Rerank 8B, one card (2026-11-18). Text and picture documents; `top_n`, `return_documents`; scores bit-identical however the documents are batched or ordered; a pair past 32,768 tokens is refused with a 400, not truncated. Other models answer 501 with the reason (checked) |
| Multi-model fleet, routing, reload | Yes, reload by `POST /admin/reload` or `tools\reload-server.ps1` (Windows' SIGHUP). The server rereads the file it was started on: the profile itself under `serve.ps1`, a service's copy under `%ProgramData%\Brilliance`; `-Listen`, `-Port` and `-ApiKey` stay applied over it |
| Model switching | Yes. **One card visible: by parking** (HIP virtual memory on, below): the leaving model's memory is unmapped at its addresses and the weights come back from the container; nothing is kept in host memory. **Several cards visible: by unloading and reloading**, no parking. Since 2026-11-18 PAL's resource cache is off on every start (`GPU_RESOURCE_CACHE_SIZE=0`; an operator's value wins), so a model reloaded in the same process plans against the same VRAM as its first load (before, up to about 4 GiB a card less) and a park takes 0.06-0.33 s. Repeated switching on two cards: checked 2026-11-19 with Qwen3.8-Flash-Next and Qwen3.8-27B IQ8R, both at tp 2; not qualified. 40 requests alternating, each switch an unload and a reload (Flash-Next about 17 s, Qwen3.8-27B about 6 s), with no drift in free VRAM or pinned memory; requests queued on a load are served by one load; a model that cannot get room beside a pinned one is refused with a 503 and the pinned one serves on (the client's message reads "out of memory"; the log names the reason). In a fleet of several models on more than one card, give each model `gpu-memory-mib` (known issue 2) |
| Prefix cache, checkpoint snapshots, agent replay | Yes |
| Speculation: DFlash2, MTP | Yes |
| Upstream 2.2's drafting options: `draft-coupling`, `draft-verify` (token, block, traversal), `draft-depth: adaptive` with `draft-depth-load`, `draft-suffix` | Yes on Qwen3.8-Flash-Next, two cards (2026-11-19); not qualified. Each serves with greedy text identical to the uncoupled base. Coupling and the traversal verify also work on Qwen3.8-27B's DFlash2. Where a model cannot take an option, the start is refused, naming why |
| Control vectors (upstream 2.2) | Yes on Qwen3.8-Flash-Next with the shipped vector, two cards (2026-11-19); not qualified. The per-request field, the `MODEL:NAME` variant and `/admin/control-vector`. Only Flash-Next's architecture applies control vectors; Qwen3.8-27B refuses the key at start, naming why |
| Tensor parallel | 2 cards: soak-clean when the weights fit the cards, supported when experts are read from host memory (known issue 1), on the exact wire (the default). Flash-Next's fastest profile runs with the same protection on lossy wht6. 4 cards: Flash-Next supported with every expert on the cards, soaked at 32K on its earlier settings before the same re-send and stop were added (known issue 1); with them its tuned settings pass the injection, stop and serving checks, and a soak on them has not run yet; 3 and 8 not run |
| Peer-to-peer between cards | No on Windows: collectives go through pinned host memory |
| Two-card all-reduce register form, coded wire and staged coded form (upstream since `3deabdae`) | No: they run only over peer-to-peer, so two-card collectives on Windows use the plain kernels through host memory |
| Upstream 2.2's launch-path speed-ups in the AQL backend: chained launches without the packet processor's acquire, the host data path flushed before each doorbell, the scalar cache emptied behind a chained wait, kept queues handed out oldest first, the expert prefetch stream's argument ring | No: Windows runs the native HIP backend, which has no AQL queues and does not chain launches. Upstream's two-card decode times for 2.2 (Flash-Next at tp 2: 11.2 ms a step at one stream, 12.4 in 2.1) come partly from these and from the peer-to-peer forms above, so they do not carry over to Windows as they are |
| Runtime LoRA on a rotated int8 trunk (`plugin/lora_plain`, upstream 2.2) | No: upstream builds the library by hand, the Windows build does not make it, and its source loads `libr4d.so` by name |
| `rad-convert`'s GPTQ step through OpenBLAS (upstream 2.2) | No: the Windows build has no OpenBLAS, so `libquant`'s GPTQ factorisations take upstream's single-threaded reference loops (upstream's own path without OpenBLAS; slower conversions, same method) |
| Dashboard | Yes; per card it shows load (GFX activity), board power and junction temperature, read from the AMD driver's ADL (`atiadlxx.dll`, no admin needed). A card without ADL shows its temperature, and the page says load and power are not available on Windows. Each model's page has a Monitoring tab (upstream's views) and a Brilliance **Sandbox** tab for a quick check: Start and Stop load and unload the model through `/admin/models/NAME/{load,unload}` (loopback callers only, as those routes are, unless the server has an API key), and a plain streamed chat over `/v1/chat/completions` kept in the page only. Start loads with `?hold=1`, so until Stop a plain API request for another model that needs its cards is refused (503, naming the hold) instead of evicting it. From another machine without a key, the page lists the models from the public `/v1/models` and opens each one's pages and Sandbox; the fleet overview and Start and Stop stay with the server's own machine. Checked against a stand-in server and on one card; not qualified |
| Terms of use (`GET /terms`) | Brilliance's own, not upstream's. The page the server shows the people it serves, linked from the dashboard and open without the API key: the shipped template `TERMS-OF-USE.md`, which carries the use restrictions some model licences (LTX-2.5's, SDXL's) require operators to pass on, or the operator's own file named by `terms-file:` under `server:`. Operators who serve other people must adopt it as their own terms (`PREVIEW.md`, "Serving on the network"). `/terms/licenses` shows each listed model's full licence text, from the files the package carries in `share/doc/radiance/model-licenses`, by the same rules: open without the key, no model woken |
| Diagnose mode | Runs; the busy-host warning reads the CPUs' load over a quarter second |
| `rad-convert` | CPU only, by decision; `--gpus` refused. A real model converted: SDXL, 9.6 s on the CPU |
| `rad-export` (finetune to container) | Yes, with Python 3 (`bin\rad-export.cmd`, which runs the `py` launcher or `python` from PATH). It runs the package's `rad-convert` beside it and finds the recipes in `share\radiance\recipes`; `--rad-convert PATH` chooses another converter. From the unpacked folder: `.\bin\rad-export.cmd --list-recipes` lists them, and `.\bin\rad-export.cmd <checkpoint folder> -o <name>.rad` exports. Its exit status is rad-export's. Folding a LoRA adapter also needs torch and safetensors in that Python |
| Model fetch by repo id | Yes (`hfget.ps1`, curl.exe and PowerShell). A repository of several containers is refused before the download, not after it. `serve.ps1 -Fetch` fetches the one file a profile names and serves the verified copy without an edit to the profile |
| Running as a service | `radiance --service`, installed by `serve.ps1 -Install`; to be qualified in an elevated shell (`SERVICE.md`) |
| Hang detection (`/health`'s `"stuck"`, `scheduler.stuck-exit`) | Partly, and differently from upstream. Upstream's guide describes Linux, where the backend watches the device's queues and `scheduler.stuck-exit` ends a stuck process so a supervisor restarts it. Windows' HIP shows no queue progress, so the Windows stall watch times the host instead: when a thread is blocked waiting on a card and nothing has completed for 5 s (no wait returned, and no stream's progress beacon moved: every 64th op a stream writes one into host memory, so a long healthy prefill is not a stall), the log names the stall; once that has lasted 60 s, `/health` answers 503 with `"status": "stuck"`. A stall that blocks no host wait is not seen. This `stuck` answer has not been seen on a real hang on Windows. **The process is never ended for a stall on Windows:** `scheduler.stuck-exit` is not acted on, so a hung server stays up and a service's recovery actions never restart it. Someone has to act: save its stacks (`tools/STACKS.md`), then stop it with Ctrl-Break (`tools\stop-server.ps1`); a drain that cannot finish keeps waiting rather than ending the process by force |
| `--ngram-placement ram=DIR` (one table shared by engines) | Yes: the table share on the CPU engine, and the native HIP backend maps the shared file for the cards (device tests). Not yet run end to end: the one tested model with an n-gram table, Flash-Next, needs two cards |
| `ngram-warm: true` (an n-gram table read into the file cache at load) | Brilliance's own; on in both Flash-Next profiles. Under `ngram-placement: disk` the Windows gather reads the table through the container's mapping, so a row the file cache lacks is read from the drive mid-step; the warm reads the table once before serving (47.68 GiB for Flash-Next, into the Windows file cache, nothing pinned). It runs only when the available memory covers the table plus the larger of 24 GiB and the configured host headroom; otherwise the start goes on without it and the log says why. The cache is Windows' to reclaim: under memory pressure rows are read from the drive again. Measured on the test machine (2026-11-20): the read took 18.9-21.0 s on two cards starting from a cold file cache, 6.0 s on two cards when much of the table was already cached, and 11.0 s in one four-card start; the load grows by that. On two cards from a cold file cache, prefill of fresh prompts (6K, 24K and 47K tokens, text the server had not seen) was 5.0-5.6x faster with the warm than without it; single-request outputs with the same seed were identical. Four cards' speed with and without it was not measured |
| Plugin SDK, out-of-tree plugins | From a source build's install; not in the package |
| Linux `scripts/` (measurement, traces, Docker) | Not ported; `win/tools` has the Windows tools (serve, stop, reload, soak, package check, stacks) |
| Cards other than gfx1201 (RDNA3, Strix Halo, gfx1200) | No: the toolchain is pinned to gfx1201 |

## Known issues

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
     flag). The process exits with status 70, its last line `radiance: fatal:` naming the fault,
     so a service's recovery should restart it (not yet run under the Service Control Manager:
     `SERVICE.md`). A hang that does not end in this stop is not restarted ("Hang detection",
     under "Upstream features"). `/stats` (`host_mailbox`) counts faults and resends per collective.
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
   and does not cover every layer, so it narrows such differences without removing them.
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
12. **`flashnext-4card` hard-locked the four-card test machine once.** The server was stopped with
    Ctrl-Break during the prefill of a prompt of about 120K tokens while a large build and its tests
    ran on the CPU; the display went black with the fans on, and a reset was needed. Windows left no
    crash dump, driver reset (TDR) or hardware error event. It was seen once and not reproduced, and
    its cause is unknown; starts, stops during the model load and stops under load pass on four
    cards. **Don't stop a four-card server during a long prefill while a large build runs beside
    it:** let the prefill finish first, or keep the CPU light (known issue 11).
13. **Some Qwen-Image 2.1 edits come out burnt and noisy.** An edit (`/v1/images/edits`) at 1024 pixels
    wide can come out over-saturated and covered in colour noise for some seeds: at 1024x1024, seed 7
    of seeds 1-11 did, with two different prompts; the other ten were clean, and so was seed 7 at the
    default size (2048x2048) and at other sizes (2026-11-20, one card). The decode, colour, VAE,
    cached conditioning and attention were each checked and are not the cause; whether upstream on
    Linux gives the same picture is not yet known. If an edit comes out burnt, retry with another
    seed.
14. **DeepSeek-V4.1 on two cards can give different greedy text from four cards.** `deepseek41-2card`
    and `deepseek41-4card` serve the same file, but the two-card profile's next-token choices differ
    from the four-card one's at some positions: in a check of 96 positions, the top token agreed at
    86. Most differences are where the model is unsure between two tokens, but a few are where the
    four-card profile is confident. The cause is not yet known. Take `deepseek41-4card` as the
    reference, and expect `deepseek41-2card`'s answers to part from it now and then.

## From Brilliance 0.2.2

- **Models:** 0.2.2's containers do not load; download the current ones (table above).
- **Configuration:** a YAML file (`--config`), as upstream's; `serve.ps1 <profile>` replaces 0.2.2's
  launcher flags. `--check` validates a file without loading.
- **Service:** `serve.ps1 -Install` as in 0.2.2, now on the YAML file. The API key is read from a
  protected file, never stored on the service's command line.
- **Stopping:** Ctrl-C / Ctrl-Break drain, as in 0.2.2; never end a working server from Task
  Manager.
