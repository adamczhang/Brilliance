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
| Qwen3.8-Flash-Next IQR-V4 (`StillDeadcode/qwen3.8-next-flash-iqr-v4`, 113 GiB) | Not run. Upstream lists a one-card layout (experts tiered to a 43,000 MiB host pool, `deploy/compose/tp1.radiance.yaml`); whether it fits Windows' one-card pin limit is unverified | **Supported**: upstream's two-card settings, tuned, on the exact wire (8 sequences, 256K, int8 KV, MTP, prefix cache, experts tiered to host memory, 6,144-token prefill steps). A lost write is re-sent, at about 4% of single-stream decode (known issue 1); the re-send and stop were checked at 6,144-token steps. An 8-hour soak has not been run on it yet. Profile: `flashnext-2card` | **Supported**: every expert on the cards (`placement: all_vram`); soaked at 32K x 8, int8 KV, MTP on its earlier settings (2026-11-17), before the four-card re-send and stop (known issue 1). Its tuned settings, 256K x 8, a bf16 KV cache and 6,144-token prefill steps, pass the injection, stop and serving checks with them. A soak on them has not run yet. Profile: `flashnext-4card` (known issues 8 and 12) |
| MiniCPM5-2B IQ8R (`StillDeadcode/minicpm5-2b-iq8r`) | Runs (bring-up model); also on the CPU package, without its drafter: set `num-speculative-tokens: 0` (known issue 10) | - | - |
| Qwen3.8-27B IQ8R-V4 (`StillDeadcode/qwen3.8-27b-iqr-v4`, `qwen3.8-27b-iq8r-v4.rad`, 27.58 GiB) | **Soak-clean**: 16K x 4, int8 KV, DFlash2 (26.67 GiB of weights leave about 3 GiB of cache). Profile: `qwen27-iq8r-1card` | **Soak-clean**: 32K x 4, DFlash2. Profile: `qwen27-iq8r-2card` | Not run |
| EmbeddingGemma 2 bf16 (`google/embeddinggemma-2`, converted with `rad-convert`, 1.40 GiB) | **Qualified**: `/v1/embeddings` for text, pictures, video and audio, 8,192 tokens an input, 64 sequences; also on the CPU kernels, with the same vectors | - | - |
| Qwen3-ASR 1.7B bf16 (`Qwen/Qwen3-ASR-1.7B-hf` as it ships, 4.09 GB) | **Qualified**: `/v1/audio/transcriptions` (5 formats, 4 response forms, past 30 s cut at pauses) and `/v1/realtime`, 16 sequences | - | - |
| Nemotron 3.5 Rerank 8B BF16 (`nvidia/Nemotron-3.5-Rerank-8B-BF16` @ f88f8d91 as it ships, 16.78 GB) | **Qualified**: `/rerank` and `/v1/rerank`, text and picture documents, 32,768 tokens a pair | - | - |
| SDXL base 1.0 bf16 (`stabilityai/stable-diffusion-xl-base-1.0`, the single-file checkpoint converted with `rad-convert`, 6.39 GiB) | **Qualified**: `/v1/images/generations`; a seed repeats its picture byte for byte; `/v1/images/edits` answers 501 | - | - |
| LTX-2.5 distilled IQ8R (`StillDeadcode/radiance-ltx-2.5-distilled-iq8r`) | **Runs, not yet qualified**: `/v1/videos`, text to video and image to video (`input_reference`), MP4 | - | - |
| Every other upstream model | Source builds; not run on Windows | | |

Containers of the 0.2.2 era (whole-model FP8, EXL3, the old Flash-Next FP8/IQ4R, GLM-5.3-Flash
MXFP4, DeepSeek-V4.1, Gemma 4 FP8) are not read by Radiance 2.2. Use the current containers above.

## Upstream features

| Feature | On Windows |
|---|---|
| Chat Completions, Responses, Messages; streaming; tools; reasoning | Yes (qualified on Qwen3.8-27B) |
| Grammars and structured output (`response_format`); `/v1/completions` and the llama-server routes (`/completion`, `/infill`) | Yes, checked on Qwen3.8-27B IQ4R-L-V4, one card (2026-11-19); not qualified. Grammars and structured output (`response_format` json_object and json_schema, llama-server's `json_schema`, GBNF on `/v1/completions` and `/completion`), tool calls under their parameter schemas, `/v1/completions`, `/completion`, `/completions` and `/infill`. A GBNF `grammar` on chat is refused on Qwen3.8 with a 400 (its chat template produces no grammar for it): use `response_format` or `/v1/completions`. There is no regex constraint (upstream has none). `/infill` needs a model with fill-in-the-middle tokens (Qwen3.8 and MiniCPM5 have them). Its `content` can end with the separator token's text (`<\|file_sep\|>`), which upstream does not treat as an end of generation as llama-server does: a client should cut the completion there. The host grammar tests pass (`gbnf_test`) |
| Pictures and video in requests (`image_url`, `video_url`) | Yes, through the bundled FFmpeg 8.1 (LGPL); checked on Qwen3.8-27B. H.264 input is decoded by FFmpeg's built-in decoder; H.264 decoding is patent-encumbered, and no H.264 patent licence is supplied (`licenses\THIRD-PARTY.md`, "H.264"). JPEG XL pictures are refused with a 400 ("JPEG XL images are not supported by this build; send PNG, JPEG or WebP"), by their bytes or an `image/jxl` data URL: the bundled FFmpeg is built without libjxl |
| Audio in requests, `/v1/audio/transcriptions`, realtime | Audio in an embeddings request: yes (FFmpeg decodes it; checked on EmbeddingGemma 2). Transcription and realtime: yes on Qwen3-ASR 1.7B, one card (2026-11-18). `/v1/audio/transcriptions` takes wav, mp3, flac, ogg and m4a uploads, each word for word the WAV's transcript, and answers as json, text, verbose_json or a stream alike; `/v1/realtime` (OpenAI's protocol over a WebSocket) settles words while the audio streams and gives the transcript about 0.075 s after the commit, equal to the upload route's; 24 kHz audio is resampled. A model without the endpoint answers 501 with the reason (checked) |
| Image generation (`/v1/images/generations`) | Yes on SDXL base 1.0 bf16, one card (2026-11-18): converted on the CPU with `rad-convert` from Stability's single-file checkpoint and CLIP's `tokenizer.json` (9.6 s, 6.39 GiB). PNG, JPEG and WebP (FFmpeg) as `b64_json`; `response_format: url` is refused with a 400; a seed repeats its picture byte for byte; `n` 1-8, seeds s, s + 1, ...; a size off the model's grid is rounded down (1000x700 gives 992x672); about 5.4 s at 1024x1024, 28 steps. `/v1/images/edits` answers 501 on SDXL (a text-only model) |
| Video generation | Works on one card with LTX-2.5 distilled IQ8R, the model for 0.3.0; not yet qualified. Videos are MP4 (H.264 and AAC) written through AMD's encoder (`h264_amf`), else Windows' (`h264_mf`), never another H.264 encoder; image to video (`input_reference`) uses the same two. `h264_amf` needs the AMD driver's AMF runtime (`amfrt64.dll`, installed with the driver), and encodes on the first display adapter Windows lists: on a machine with an AMD integrated GPU, the integrated GPU, not the card that runs the model. `h264_mf` needs Windows Media Foundation's H.264 encoder, which Windows "N" editions lack until Microsoft's Media Feature Pack is installed. With neither, a video fails with the reason. Every MP4 says it is AI-generated in its metadata (`comment`: `ai_generated=true; model=...`, and `description`); there is no visible watermark. No H.264 encoder implementation ships in 0.3.0 (`licenses\THIRD-PARTY.md`, "H.264") |
| Speech synthesis (TTS) | Not supported in 0.3.0. The Windows build refuses `/v1/audio/speech`, `/audio/speech` and `/v1/audio/speech/stream` for every model: the POST routes answer 501 (`not_implemented_error`, "not supported in this build"), the WebSocket an error event and close 1008 |
| Music generation | Not served: it shares `/v1/audio/speech`, which the Windows build refuses (above). It needs MiniMax-Music3, which is not run on Windows |
| Embeddings | Yes: `/v1/embeddings` on EmbeddingGemma 2, on one card and on the CPU kernels. Text, pictures, audio and video inputs; batches; `dimensions`; base64; concurrent requests each get their own vector; an input past 8,192 tokens is refused with a 400 |
| Rerank | Yes: `/rerank` and `/v1/rerank` on Nemotron 3.5 Rerank 8B, one card (2026-11-18). Text and picture documents; `top_n`, `return_documents`; scores bit-identical however the documents are batched or ordered; a pair past 32,768 tokens is refused with a 400, not truncated. Other models answer 501 with the reason (checked) |
| Multi-model fleet, routing, reload | Yes, reload by `POST /admin/reload` or `tools\reload-server.ps1` (Windows' SIGHUP) |
| Model switching | Yes. **One card visible: by parking** (HIP virtual memory on, below): the leaving model's memory is unmapped at its addresses and the weights come back from the container; nothing is kept in host memory. **Several cards visible: by unloading and reloading**, no parking. Since 2026-11-18 PAL's resource cache is off on every start (`GPU_RESOURCE_CACHE_SIZE=0`; an operator's value wins), so a model reloaded in the same process plans against the same VRAM as its first load (before, up to about 4 GiB a card less) and a park takes 0.06-0.33 s. Repeated switching on two cards: checked 2026-11-19 with Qwen3.8-Flash-Next and Qwen3.8-27B IQ8R, both at tp 2; not qualified. 40 requests alternating, each switch an unload and a reload (Flash-Next about 17 s, Qwen3.8-27B about 6 s), with no drift in free VRAM or pinned memory; requests queued on a load are served by one load; a model that cannot get room beside a pinned one is refused with a 503 and the pinned one serves on (the client's message reads "out of memory"; the log names the reason). In a fleet of several models on more than one card, give each model `gpu-memory-mib` (known issue 2) |
| Prefix cache, checkpoint snapshots, agent replay | Yes |
| Speculation: DFlash2, MTP | Yes |
| Upstream 2.2's drafting options: `draft-coupling`, `draft-verify` (token, block, traversal), `draft-depth: adaptive` with `draft-depth-load`, `draft-suffix` | Yes on Qwen3.8-Flash-Next, two cards (2026-11-19); not qualified. Each serves with greedy text identical to the uncoupled base. Coupling and the traversal verify also work on Qwen3.8-27B's DFlash2. Where a model cannot take an option, the start is refused, naming why |
| Control vectors (upstream 2.2) | Yes on Qwen3.8-Flash-Next with the shipped vector, two cards (2026-11-19); not qualified. The per-request field, the `MODEL:NAME` variant and `/admin/control-vector`. Only Flash-Next's architecture applies control vectors; Qwen3.8-27B refuses the key at start, naming why |
| Tensor parallel | 2 cards: soak-clean when the weights fit the cards, supported when experts are read from host memory (known issue 1), on the exact wire (the default); `tp-wire: wht6` runs with the same protection, not qualified; 4 cards: Flash-Next supported with every expert on the cards, soaked at 32K on its earlier settings before the same re-send and stop were added (known issue 1); with them its tuned settings pass the injection, stop and serving checks, and a soak on them has not run yet; 3 and 8 not run |
| Peer-to-peer between cards | No on Windows: collectives go through pinned host memory |
| Two-card all-reduce register form, coded wire and staged coded form (upstream since `3deabdae`) | No: they run only over peer-to-peer, so two-card collectives on Windows use the plain kernels through host memory |
| Upstream 2.2's launch-path speed-ups in the AQL backend: chained launches without the packet processor's acquire, the host data path flushed before each doorbell, the scalar cache emptied behind a chained wait, kept queues handed out oldest first, the expert prefetch stream's argument ring | No: Windows runs the native HIP backend, which has no AQL queues and does not chain launches. Upstream's two-card decode times for 2.2 (Flash-Next at tp 2: 11.2 ms a step at one stream, 12.4 in 2.1) come partly from these and from the peer-to-peer forms above, so they do not carry over to Windows as they are |
| Runtime LoRA on a rotated int8 trunk (`plugin/lora_plain`, upstream 2.2) | No: upstream builds the library by hand, the Windows build does not make it, and its source loads `libr4d.so` by name |
| `rad-convert`'s GPTQ step through OpenBLAS (upstream 2.2) | No: the Windows build has no OpenBLAS, so `libquant`'s GPTQ factorisations take upstream's single-threaded reference loops (upstream's own path without OpenBLAS; slower conversions, same method) |
| Dashboard | Yes; per card it shows temperature, not load or power |
| Terms of use (`GET /terms`) | Brilliance's own, not upstream's. The page the server shows the people it serves, linked from the dashboard and open without the API key: the shipped template `TERMS-OF-USE.md`, which carries the use restrictions some model licences (LTX-2.5's, SDXL's) require operators to pass on, or the operator's own file named by `terms-file:` under `server:`. Operators who serve other people must adopt it as their own terms (`PREVIEW.md`, "Serving on the network"). `/terms/licenses` shows each listed model's full licence text, from the files the package carries in `share/doc/radiance/model-licenses`, by the same rules: open without the key, no model woken |
| Diagnose mode | Runs; the busy-host warning reads the CPUs' load over a quarter second |
| `rad-convert` | CPU only, by decision; `--gpus` refused. A real model converted: SDXL, 9.6 s on the CPU |
| `rad-export` (finetune to container) | Yes, with Python 3 (`bin\rad-export.cmd`); recipes in `share\radiance\recipes` |
| Model fetch by repo id | Yes (`hfget.ps1`, curl.exe and PowerShell) |
| Running as a service | `radiance --service`, installed by `serve.ps1 -Install`; to be qualified in an elevated shell (`SERVICE.md`) |
| `--ngram-placement ram=DIR` (one table shared by engines) | Yes: the table share on the CPU engine, and the native HIP backend maps the shared file for the cards (device tests). Not yet run end to end: the one tested model with an n-gram table, Flash-Next, needs two cards |
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
     host collectives re-send a wrong block and stop loudly past the limit. The exact wire stays
     the default and the one the profiles use.
   - **What it costs:** about 4% of single-stream decode and about 2% at 4 streams; prefill is
     unchanged. `BRILLIANCE_AR_RESEND=0` turns the resend off. On the exact wire the stop below
     stays; with `tp-wire: wht6` nothing is left to catch a lost write, and a request can hang.
     Keep the default.
   - **A block that cannot be recovered stops the server** within about 2 s (30 s for a missing
     flag). The process exits with status 70, its last line `radiance: fatal:` naming the fault,
     so a service restarts it. `/stats` (`host_mailbox`) counts faults and resends per collective.
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
   where one is set. With more than one device visible (several cards, or a card and the
   integrated GPU) it is off: the runtime commits a reservation's whole size at its first mapping
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
4. **The GPU package without a card** serves on the CPU only once its gfx1201 kernel libraries
   are left out: add `kernels: libavx,libref` under `server:`. Without it the model fails to
   declare (`rad_arch_declare returned invalid argument`). Upstream's startup warning gives the same
   advice ("build it for host, or leave it out with --kernels"). The CPU package needs no setting.
   **With a card visible, hide it as well:** set `HIP_VISIBLE_DEVICES=-1` beside
   `kernels: libavx,libref`. With a card visible and only the `kernels` setting, the GPU build
   refuses to start ("no device kernel").
5. **Speculative decoding (DFlash2, MTP) and plain decoding** can differ in greedy text where two
   tokens are almost tied, as upstream documents: the verify pass and the single-token pass round
   differently. Both are the model's answer. `BRILLIANCE_EXACT_VERIFY=1` (off by default) makes the
   IQR 5-bit row-scaled layers (and short MoE gate/up batches) verify exactly as single-token
   decoding does. It costs about 36-41% of DFlash2 single-stream decode on Qwen3.8-27B IQ4R-L-V4
   and does not cover every layer, so it narrows such differences without removing them.
6. **The dashboard** shows no card load or power on Windows.
7. **`tar.exe` and other ANSI-code-page tools** cannot reach folders named outside the system code
   page; `radiance.exe` and the package's scripts can (UTF-8 throughout).
8. **Flash-Next's output can change when requests share a step.** A request computed together with
   others in one step can get a slightly different next-token distribution than the same request
   sent alone, and identical requests in the same step can differ from each other, so sampled output
   varies more than the temperature alone explains. This is rounding, not wrong data: the expert
   kernel takes a shorter path for an expert's last one or two rows, and other kernels choose their
   shape from the step's size. Upstream Radiance documents the same for batched requests. Flash-Next
   magnifies these last-bit differences more than most models; one request at a time is repeatable,
   and Qwen3.8-27B is barely affected.
9. **An `exclusive` group can end up with both models resident** when a request loads one while
   the start-up warm-up is still loading the other: a partner still loading is not yet counted.
   This is upstream's fleet, on every platform. For strict turns, set `scheduler: warm-up: false`.
10. **MiniCPM5-2B IQ8R on the CPU kernels needs its drafter off.** The container carries a DSpark
    drafter, and speculation is on by default. The CPU kernels have no form of the drafter's int8
    norm, so the model does not start: `no loaded kernel plugin declares op
    'rmsnorm_had_quant_i8g'`. Set `num-speculative-tokens: 0` in the model's block (or pass
    `--num-speculative-tokens 0`); the model then serves on the CPU with plain decoding, without
    the drafter's speed-up. Upstream's CPU kernels have the same gap. On a card the drafter
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

## From Brilliance 0.2.2

- **Models:** 0.2.2's containers do not load; download the current ones (table above).
- **Configuration:** a YAML file (`--config`), as upstream's; `serve.ps1 <profile>` replaces 0.2.2's
  launcher flags. `--check` validates a file without loading.
- **Service:** `serve.ps1 -Install` as in 0.2.2, now on the YAML file. The API key is read from a
  protected file, never stored on the service's command line.
- **Stopping:** Ctrl-C / Ctrl-Break drain, as in 0.2.2; never end a working server from Task
  Manager.
