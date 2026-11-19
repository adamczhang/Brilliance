# Brilliance 0.3.0-alpha.1 for Windows

A Windows build of [Radiance](https://codeberg.org/StillDeadcode/radiance) 2.1 on its native HIP
backend, qualified on **one Radeon AI PRO R9700** with **Qwen3.8-27B IQ4R-L-V4**. This is an alpha
preview of Brilliance 0.3.0: one qualified card count and model, more that run soak-clean or
supported (`SUPPORT.md`), and the limits below. Expect rough edges.

## Quick start

1. Windows 11 with the AMD Adrenalin driver 32.0.32015.12012 or later (the one it was tested on).
   Nothing else to install: the HIP runtime, FFmpeg and the VC++ runtime are in `bin\`.
2. Download the model from Hugging Face `StillDeadcode/qwen3.8-27b-iqr-v4`, file
   `qwen3.8-27b-iq4r-l-v4.rad` (17.42 GiB, SHA-256
   `d57d1428218e20a4cd578c3e286f30a2e566e589c832bf5004f49e164cb48ae4`). The package's fetcher
   gets that one file, resuming and checking its SHA-256:
   `.\libexec\brilliance\hfget.ps1 StillDeadcode/qwen3.8-27b-iqr-v4 <dest-dir> main qwen3.8-27b-iq4r-l-v4.rad`.
   Naming this repository as `model:` does not work: it holds several containers, and the server
   refuses it (below).
3. Edit `model:` in `profiles\qwen27-preview.yaml` to where the file is.
4. Start it: `.\serve.ps1 qwen27-preview -Bus 3`. `-Bus` picks the card by its PCI bus number
   (Device Manager or GPU-Z); without it every discrete card HIP shows is used, and the
   integrated GPU is left out. It is ready when it prints
   `server: listening on http://127.0.0.1:8000`.
   - `bin\radiance.exe --config profiles\qwen27-preview.yaml` works too. On a machine with an
     integrated GPU or several cards, set `$env:HIP_VISIBLE_DEVICES` to the card's HIP index first.
5. Use any OpenAI- or Anthropic-compatible client against `http://127.0.0.1:8000/v1`:
   - Chat Completions, Responses and Messages, streamed or not, with tools;
   - pictures and video clips as `image_url` / `video_url` parts.
6. The **dashboard** is at `http://127.0.0.1:8000/`: models, memory, performance, requests and sessions.
   The **terms of use** are at `http://127.0.0.1:8000/terms`, linked from the dashboard.
7. Stop it with **Ctrl-C** in its window: it finishes what is in flight and exits.
   - Do not end it from Task Manager while it is working: a GPU process killed mid-work has taken the
     driver down.
   - From a script, `tools\stop-server.ps1 -ProcessId <pid>` sends Ctrl-Break (the same drain).
   - `.\serve.ps1 <profile> -Background -Log FILE` starts a server in a console of its own.

`bin\radiance.exe --config profiles\qwen27-preview.yaml --check` validates a profile without
loading.

## Serving on the network, as a service, and reloading

- **On the network:** `.\serve.ps1 qwen27-preview -Listen 0.0.0.0 -ApiKey env:MY_KEY` listens on
  every interface and requires the key that `MY_KEY` holds (`-ApiKey off` for none). A firewall rule
  for the port is the administrator's to add.
- **As a Windows service:** `.\serve.ps1 <profile> -Install` (in an elevated PowerShell; `-WhatIf`
  shows every step first). Unpack the package under `C:\Program Files` before installing, since the
  service runs as LocalSystem from that folder. `SERVICE.md` has the details and the qualification
  steps.
- **Terms of use, when you serve other people:** some model licences require whoever serves the
  model to other people to pass its use restrictions on to them as enforceable terms (LTX-2.5's and
  SDXL's do). The server shows a terms-of-use page at `/terms`, linked from the dashboard and open
  without the API key. Out of the box it is `TERMS-OF-USE.md`, a template that carries those
  restrictions for the models listed in `SUPPORT.md`. Before others use your server, adopt it:
  - copy `TERMS-OF-USE.md`, name yourself as the operator, remove the models you do not serve and
    add your own terms;
  - set `terms-file:` under `server:` in your profile to your copy. A `.html` or `.htm` file is
    served as HTML (no scripts), anything else as UTF-8 plain text;
  - restart the server: the file is read when it starts, and a file that is missing, empty, not
    UTF-8 or over 1 MiB stops the start with the reason.
  The template is not legal advice; the terms are yours, as the operator.
- **Reloading the YAML** without a restart: `tools\reload-server.ps1 -ProcessId <pid>`, the Windows
  counterpart of `kill -HUP`, or `POST /admin/reload`. Changed models reload, removed ones unload,
  new ones are added.

## The profiles

| Profile | Model | Cards | Status | Context | Requests at once | Speculation | KV cache |
|---|---|---|---|---|---|---|---|
| `qwen27-preview.yaml` | Qwen3.8-27B IQ4R-L-V4 | 1 | Qualified | 32K | 4 | DFlash2, depth 7 | bf16 |
| `qwen27-8k-plain.yaml` | Qwen3.8-27B IQ4R-L-V4 | 1 | Qualified | 8K | 1 | off | bf16 |
| `qwen27-iq8r-1card.yaml` | Qwen3.8-27B IQ8R-V4 | 1 | Soak-clean | 16K | 4 | DFlash2 | int8 |
| `qwen27-iq8r-2card.yaml` | Qwen3.8-27B IQ8R-V4 | 2 (TP2) | Soak-clean | 32K | 4 | DFlash2 | bf16 |
| `flashnext-4card.yaml` | Qwen3.8-Flash-Next IQR-V4 | 4 (TP4), every expert on the cards | Soak-clean, before the four-card re-send; with it, injection and serving checks pass | 32K | 8 | MTP, depth 3 | int8 |
| `flashnext-2card.yaml` | Qwen3.8-Flash-Next IQR-V4 | 2 (TP2), experts tiered to host memory | Supported; upstream's two-card settings | 200K | 8 | MTP, depth 3 | int8 |

- The IQ8R-V4 file is `qwen3.8-27b-iq8r-v4.rad` in `StillDeadcode/qwen3.8-27b-iqr-v4` (27.58 GiB); the
  Flash-Next file is `qwen3.8-next-flash-iqr-v4.rad` in `StillDeadcode/qwen3.8-next-flash-iqr-v4`
  (113.32 GiB). Each profile's comment has the file's SHA-256; edit its `model:` to where the file is.
- Name the cards with `-Bus`: `.\serve.ps1 qwen27-iq8r-1card -Bus 6`, `.\serve.ps1 qwen27-iq8r-2card -Bus 3,9`.
  Without it every discrete card is visible and the profile takes the first ones in HIP's order. On a
  machine with exactly four cards, `.\serve.ps1 flashnext-4card` needs no `-Bus`.
- `flashnext-4card` has a known issue: Flash-Next's output changes when requests share a step, so
  sampling over requests sent together is looser than the temperature asked (`SUPPORT.md`, known
  issue 8). One request at a time behaves normally.
- `flashnext-2card` (`.\serve.ps1 flashnext-2card -Bus 3,9`) carries upstream's two-card settings on
  the exact wire and sets no runtime variable. The server re-sends a lost write (`SUPPORT.md`,
  known issue 1).
- A profile's `# brilliance-env: NAME=VALUE` lines, and `serve.ps1 -Env NAME=VALUE[,...]`, set
  `GPU_*`, `HIP_*`, `AMD_*`, `BRILLIANCE_*` and `RADIANCE_*` variables for the server only, and for
  `-Install` in the service's environment. `.\serve.ps1 <profile> -WhatIf` prints what a start would
  set and run.
- `kv-cache-dtype: i8` halves the cache (5.3 GiB instead of 9.4 GiB at 32K x 4).

## What was measured (one R9700 at -20% board power, -50 mV, VRAM at its stock 2518 MHz)

- Plain decoding: about 30 tok/s for one request. With DFlash2: about 86 tok/s for one request.
- 32K x 4 with DFlash2: 100 short requests, 4 at a time, all completed; median latency 0.87 s.
- The API qualification (`win\tools\qualify_api.py` in the source):
  - Chat Completions, Responses and Messages, streamed and whole;
  - typed, multiline, empty and repeated tool calls, and their results;
  - clients that disconnect during prefill or decode;
  - malformed requests;
  - concurrent conversations that never cross;
  - the tool loop that went off topic in Codeberg issue #4, fresh and after unrelated history;
  - a picture and an H.264 clip, described correctly (2026-11-17). H.264 input is decoded by
    FFmpeg's built-in decoder; H.264 decoding is patent-encumbered, and no H.264 patent licence is
    supplied.

## Known limits

- **One card is qualified.** Qwen3.8-Flash-Next IQR-V4 runs on two cards and passes the same API
  checks, but with its production settings (8 sequences, 200K context, speculation, prefix cache)
  a write to host memory is lost now and then (`SUPPORT.md`, known issue 1). The server re-sends
  it, at about 4% of single-stream decode, so Flash-Next on two cards is Supported; the 8-hour soak
  is still to run. A block that cannot be recovered stops the server, which exits with status 70.
  A server that stops making progress for any reason can be read with `bin\brl-stacks.exe`
  (`tools\STACKS.md`). On four cards, with every expert on the cards, it was soak-clean, as was
  Qwen3.8-27B IQ8R-V4 on one and on two. Four cards now carry the same re-send and stop: Flash-Next
  was soaked there before they were added, and with them it passes the injection and serving checks.
  `SUPPORT.md` has the table of models and card counts.
- **Cards:** the build carries GPU code for gfx1201 (R9700, RX 9070 family) only.
- **Generation endpoints:**
  - **Video generation** works on one card with LTX-2.5 distilled IQ8R and is not yet qualified.
    It writes H.264 only through AMD's (`h264_amf`), else Windows' (`h264_mf`) licensed encoders.
    `h264_amf` encodes on the first display adapter Windows lists: on a machine with an AMD
    integrated GPU, the integrated GPU, not the card that runs the model.
  - **Speech synthesis (TTS) is not supported** in 0.3.0: `/v1/audio/speech` answers 501 ("not
    supported in this build"). Music generation, on the same endpoint, is not served either.
  - **Image generation** works (`SUPPORT.md`).
- **No peer-to-peer access between cards on Windows.** Multi-card collectives pass through pinned
  host memory, and a process can pin about the shared-memory budget divided by the cards it sees. A
  table too large to pin is read from the file instead (`--ngram-placement disk`).
- **HIP virtual memory is on with one card visible, and off with more.**
  - With exactly one device visible to the process (`-Bus` or `-Device` naming one card, or
    `HIP_VISIBLE_DEVICES` naming one) it is on, since 2026-11-17. The KV cache is elastic: it
    grows as sessions fill it. A model that makes room for another is parked: its weights come
    back from the container, and nothing is kept in host memory.
  - With more than one device visible (several cards, or a card and the integrated GPU) it is
    off: the runtime commits a reservation's whole size at its first mapping
    (rocm-systems#13232). The KV cache then holds its whole size from the start, nothing is
    parked, and switching models unloads one and loads the other from its file.
  - The startup log says which, in a line that begins `device: HIP virtual memory is on` or
    `is off`. `BRILLIANCE_VMM=0` turns it off; `=1` forces it on and is not qualified with
    several devices.
- **The dashboard shows each card's temperature, not its load or power.** Windows' own counters do
  not see HIP's work, and AMD's telemetry library cannot ship with an open-source engine.
- **DFlash2 and plain decoding can differ in greedy text** where two tokens are almost tied: the
  verify pass and the single-token pass round differently. Both are the model's answer; neither is
  a fault.
- **A model by repository id** (`model: owner/name`) is fetched by
  `libexec\brilliance\hfget.ps1` into `<dir>\<owner>\<name>`. `<dir>` is `download-dir:` under
  `server:` if set, else `BRILLIANCE_HF_CACHE`, else `%LOCALAPPDATA%\radiance\hf`. This serves the
  one `.rad` at the repository's top level. A repository with several containers, such as
  `StillDeadcode/qwen3.8-27b-iqr-v4`, is refused, naming them; fetch the file and point `model:` at
  it instead. Every start checks the repository for changes; set `model-revision:` to a tag or a
  commit to keep the weights you reviewed. `serve.ps1` copies no model: each is served from where
  its `model:` points.
- **`rad-convert` runs on the CPU only**: on Windows it never uses a GPU, and `--gpus` is refused.
  The recipes are in `share\radiance\recipes`. `bin\rad-export.cmd` turns a finetune into a
  container with the stock model's recipe; it needs Python 3, plus torch and safetensors for a
  LoRA adapter.
- **Serving without a card:** the GPU package's kernel libraries for gfx1201 cannot run on the
  CPU. Add `kernels: libavx,libref` under `server:`, as radiance's startup warning advises, or use
  the CPU package (`-win64-cpu`), which needs nothing.
- **The startup report** cannot see PCIe link speeds above each card's own link (Windows does not
  report them for bridges); those hops print "?".

## What is in this package

- `MANIFEST.json`: the source commit, toolchain, HIP runtime and driver this package was built and
  tested with, and the SHA-256 of every file.
- `SUPPORT.md`: models and card counts, upstream's features on Windows, the known issues, and what
  changes from Brilliance 0.2.2. `SERVICE.md`: running it as a Windows service.
- `TERMS-OF-USE.md`: the terms-of-use template the server shows at `/terms` (above).
- `share\doc\radiance`: `LICENSE` and `NOTICE`. `share\doc\radiance\third_party`: the licences of
  what the package carries, FFmpeg's (LGPL, linked dynamically) among them. `THIRD-PARTY.md` there
  is the index: each DLL in `bin\`, the libraries built into it, their licence files and where
  their source is.
- **Model licences** (the models are not in the package; each model's own licence governs, and
  this is not legal advice from the project). `TERMS-OF-USE.md` carries the use restrictions below
  that must be passed on to the people you serve. The server shows each licence's full text at
  `/terms/licenses`, linked from `/terms`; the same files are in `share\doc\radiance\model-licenses`.
  - **Qwen3.8-27B** (`StillDeadcode/qwen3.8-27b-iqr-v4`, both containers): Apache 2.0, as its base
    model Qwen/Qwen3.8-27B. No use restrictions to pass on.
  - **Qwen3.8-Flash-Next IQR-V4** (`StillDeadcode/qwen3.8-next-flash-iqr-v4`): the **Qwen
    Community License 1.0** (`LICENSE` in that repository). Free to use, change, host and sell,
    keeping its copyright notice, with two conditions and no use restrictions to pass on:
    - **Model as a Service.** If you or a company you are affiliated with runs a business that
      gives other people access to language-model inference or fine-tuning, for example through
      an API or a hosted endpoint where they control the inputs or settings, you need a separate
      licence from Qwen before any commercial use of the model. The same holds for an "AI Work
      Assistant" business (an AI product mainly for coding or office work). Use inside your own
      organisation that makes nothing available to anyone outside it needs no such licence.
      Qwen's contact for it is model-business@notice.qwencloud.com.
    - **Large products.** A commercial product or service with more than 100 million monthly
      active users or US$20 million monthly revenue must show the model's name prominently in
      its user interface.
  - **MiniCPM5-2B IQ8R** (`StillDeadcode/minicpm5-2b-iq8r`): Apache 2.0. No use restrictions to
    pass on.
  - **EmbeddingGemma 2** (`google/embeddinggemma-2`): Apache 2.0. Its model card says deployments
    must adhere to the Gemma Prohibited Use Policy, which `TERMS-OF-USE.md` passes on.
  - **Qwen3-ASR 1.7B** (`Qwen/Qwen3-ASR-1.7B-hf`): Apache 2.0. No use restrictions to pass on.
  - **Nemotron 3.5 Rerank 8B** (`nvidia/Nemotron-3.5-Rerank-8B-BF16`): OpenMDW 1.1 (`LICENSE`
    and `NOTICE` in that repository), a permissive licence: a copy you pass on keeps the licence
    and the notices. It sets nothing on outputs and no use restrictions to pass on.
  - **SDXL 1.0:** CreativeML Open RAIL++-M; hosting is allowed; the use-based restrictions of
    paragraph 5 and Attachment A must be passed on to users as enforceable terms. The CLIP
    tokenizer comes from openai/clip-vit-large-patch14, whose vocabulary OpenAI publishes under
    MIT in openai/CLIP.
  - **LTX-2.5 distilled IQ8R** (`StillDeadcode/radiance-ltx-2.5-distilled-iq8r`, video): the
    **LTX-2.x Community License Agreement** of Lightricks Ltd. (`LICENSE` in that repository).
    - **The repository is gated.** Each user accepts the licence on its Hugging Face page with
      their own account and downloads with their own token (`HF_TOKEN` for `hfget.ps1`).
      Brilliance accepts nothing on a user's behalf.
    - **Revenue.** Free of charge for any purpose to an entity (with its affiliates) under
      US$10M a year in revenue. At or above it, any use other than a Non-Commercial Purpose
      (section 2.2: testing, evaluation or research outside production, with no payment from it)
      needs a paid licence from Lightricks (licensing@ltx.io).
    - **Use restrictions.** Section 4 and Attachment A (a list of forbidden uses, plus
      Lightricks' Acceptable Use Policy, which it can update) apply to the model and to the
      videos it makes. Whoever lets others use the model, for example by serving it on a
      network, must require those users to follow them. Whoever hosts it for others or passes
      the file on must pass the restrictions on as enforceable terms, with a copy of the licence.
    - **Marking.** Videos must not be published without saying plainly that they are machine
      generated (Attachment A, item 5), and disclosure, metadata, watermark or provenance
      features must not be removed (section 6). Brilliance marks every MP4 as AI-generated in
      its metadata, with no visible watermark; keep that marking. The operator is responsible
      for AI-disclosure laws, such as the EU AI Act and the California AI Transparency Act,
      which the licence names.
    - **The container** is a modified copy (re-encoded to IQ8R; its `NOTICE` says how), not an
      official Lightricks release, and not endorsed by Lightricks.
- **H.264** (`THIRD-PARTY.md`, "H.264", has the whole of it):
  - This package ships no H.264 encoder implementation (no OpenH264, no x264). Its FFmpeg,
    Brilliance's own built with vcpkg, keeps only two thin wrappers over encoders their makers
    license: `h264_amf` (AMD's
    hardware encoder, through AMF, which comes with the AMD driver) and `h264_mf` (Windows' Media
    Foundation encoder, missing from Windows "N" editions without Microsoft's Media Feature Pack).
    The encoder vendors' licences are what cover H.264 output through them; this is not legal
    advice from the project.
  - H.264 input is decoded by FFmpeg's built-in decoder; H.264 decoding is patent-encumbered, and
    no H.264 patent licence is supplied.
