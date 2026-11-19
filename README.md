![Brilliance](docs/img/banner.png)

# Brilliance

A native Windows 11 distribution of [Radiance](https://codeberg.org/StillDeadcode/radiance), the C++/HIP
LLM inference engine, for AMD Radeon AI PRO R9700 cards (RDNA4, `gfx1201`): no WSL, no Linux, no Docker.
It speaks Radiance's HTTP API (OpenAI's Chat Completions and Responses, Anthropic's Messages), so existing
clients work unchanged.

**This is a new fork, based on Radiance 2.1:** upstream's `v2.1.0` tag and its main branch up to `3deabdae`
(2026-11-18), with the Windows port on top. It is not a continuation of Brilliance 0.2.2's code. Upstream's source layout,
YAML configuration, model implementations and plugin ABI are kept; the Windows work is a reviewable layer
over them.

- **The new home** is [github.com/adamczhang/Brilliance](https://github.com/adamczhang/Brilliance).
- **Brilliance 0.2.2 is archived** at its original address,
  [codeberg.org/Dad_Vibes/Brilliance](https://codeberg.org/Dad_Vibes/Brilliance). Its `.rad` containers do
  not load in this version ([from 0.2.2](#from-brilliance-022)).
- **Brilliance 0.3.0-alpha.1 is this fork's first public release: an alpha preview of 0.3.0.** Its
  package, `Brilliance-0.3.0-alpha.1-<sha>-win64.zip`, is on the
  [v0.3.0-alpha.1 release page](https://github.com/adamczhang/Brilliance/releases/tag/v0.3.0-alpha.1).
  Expect rough edges: what is qualified and what is not is listed below and in
  [win/SUPPORT.md](win/SUPPORT.md).

Upstream's own README is kept unchanged as [RADIANCE-README.md](RADIANCE-README.md).

## Status

Alpha preview (0.3.0-alpha.1). What is qualified on Windows today, from [win/SUPPORT.md](win/SUPPORT.md):

- **One card:** Qwen3.8-27B (IQ4R-L-V4 qualified, IQ8R-V4 soak-clean).
- **Two cards:** soak-clean when the weights fit the cards (Qwen3.8-27B IQ8R-V4).
- **Four cards:** Qwen3.8-Flash-Next IQR-V4, soak-clean with every expert on the cards (soaked before the
  four-card re-send and stop were added; with them it passes the injection and serving checks).
- **Model switching on one card** by parking, with HIP virtual memory.
- **Embeddings** (`/v1/embeddings`) with EmbeddingGemma 2.
- **Transcription and realtime** (`/v1/audio/transcriptions`, `/v1/realtime`) with Qwen3-ASR 1.7B, and
  **rerank** (`/rerank`, `/v1/rerank`) with Nemotron 3.5 Rerank 8B, each on one card.
- **Image generation** (`/v1/images/generations`) with SDXL base 1.0, on one card.
- **Clients:** Open WebUI and OpenCode pass their tool round trips.

The main known issues ([all of them](win/SUPPORT.md#known-issues)):

- **Flash-Next on two cards loses a write now and then** with its production settings, where experts are
  read from host memory. A lost block is re-sent and the step completes, at about 4% of single-stream
  decode; one that cannot be recovered stops the server, which exits with status 70. Flash-Next on two
  cards is Supported; the 8-hour soak is still to run. Profile: `flashnext-2card`.
- **Flash-Next's output changes when requests share a step:** identical requests sent together get
  different next-token distributions. Under investigation.
- **Video generation works on one card and is not yet qualified** (LTX-2.5 distilled IQ8R). It writes H.264
  only through AMD's or Windows' licensed encoders, on Brilliance's own FFmpeg ([H.264](#h264)).
- **Speech synthesis (TTS) is not supported** in 0.3.0, and **music generation is not served**:
  `/v1/audio/speech`, which both use, answers 501, "not supported in this build", for every model.
  A model without an endpoint answers 501 with the reason.

## Installation

### 1. Requirements

- **Windows 11.**
- **One or more AMD Radeon AI PRO R9700** (`gfx1201`, 32 GB each). The package carries GPU code for
  `gfx1201` (R9700, RX 9070 family) only; other cards are not supported.
- **The AMD Adrenalin driver 32.0.32015.12012 or later**, the version it was tested on.
- **Memory.**
  - One card holds Qwen3.8-27B IQ4R-L-V4 (a 17.42 GiB file). IQ8R-V4 on one card (26.67 GiB of weights)
    leaves about 3 GiB of KV cache.
  - Qwen3.8-Flash-Next IQR-V4 (113 GiB) does not fit one card's memory; four cards hold every expert.
    Upstream also serves it on one card with experts in host memory; that layout is not run on Windows.
  - A process can pin about the shared-memory budget (half of RAM or more), divided by the cards it sees,
    less 2 GiB. Multi-card exchanges go through that pinned memory.
- **Disk** for the models you serve.
- **Recommended:** an integrated GPU or another display card for the desktop. `serve.ps1` leaves the
  integrated GPU out on its own.

### 2. Get the package

Download `Brilliance-0.3.0-alpha.1-<sha>-win64.zip` from the
[v0.3.0-alpha.1 release](https://github.com/adamczhang/Brilliance/releases/tag/v0.3.0-alpha.1), check it
against the `.sha256` beside it (`Get-FileHash <zip>` in PowerShell), and unzip it anywhere; a path with spaces
works. You can also [build it from source](#building-from-source). A CPU-only package
(`Brilliance-<version>-<sha>-win64-cpu.zip`), which serves on the CPU kernels and needs no card, can be built
from source; it is not published with this release.

| In the package | What it is |
|---|---|
| `bin\` | `radiance.exe`, the `rad-*` tools, the HIP runtime, FFmpeg, the VC++ runtime; `brl-stacks.exe` for reading a hung server |
| `share\radiance\` | the plugins (kernels, architectures, quantizers) and the conversion recipes |
| `share\doc\radiance\` | `LICENSE`, `NOTICE`, and the third-party licences under `third_party\` |
| `profiles\` | the qualified YAML profiles |
| `libexec\brilliance\` | `hfget.ps1`, the Hugging Face fetcher |
| `tools\` | `serve-bg.ps1`, `stop-server.ps1`, `reload-server.ps1`, `STACKS.md` |
| `serve.ps1` | starts a profile, choosing cards by PCI bus |
| `PREVIEW.md`, `SUPPORT.md`, `SERVICE.md` | quick start and limits; models, features and known issues; running as a service |
| `MANIFEST.json` | the source commit, toolchain, HIP runtime and driver it was built and tested with, and every file's SHA-256 |

Nothing else is installed: no ROCm, no VC++ redistributable.

### 3. Get a model

The package's profiles serve three containers (the profiles are in [step 4](#4-start-a-server)):

- **Qwen3.8-27B IQ4R-L-V4** (`qwen27-preview`, `qwen27-8k-plain`): file `qwen3.8-27b-iq4r-l-v4.rad` (17.42 GiB,
  SHA-256 `d57d1428218e20a4cd578c3e286f30a2e566e589c832bf5004f49e164cb48ae4`) from the Hugging Face repository
  [`StillDeadcode/qwen3.8-27b-iqr-v4`](https://huggingface.co/StillDeadcode/qwen3.8-27b-iqr-v4).
- **Qwen3.8-27B IQ8R-V4** (`qwen27-iq8r-1card`, `qwen27-iq8r-2card`): file `qwen3.8-27b-iq8r-v4.rad`
  (27.58 GiB, SHA-256 `474cb9fc49714a56455a26c0ab3049bd029dc1abae31e4eb9cc66026f0edf04d`) from the same
  repository.
- **Qwen3.8-Flash-Next IQR-V4** (`flashnext-4card`, `flashnext-2card`): file `qwen3.8-next-flash-iqr-v4.rad` (113.32 GiB,
  SHA-256 `25aff8c950f4b823bc8b77c8184ed48827cb47058ffce08ae2c83b12e021fea0`) from
  [`StillDeadcode/qwen3.8-next-flash-iqr-v4`](https://huggingface.co/StillDeadcode/qwen3.8-next-flash-iqr-v4).

- **Download it** from that page, or fetch the one file with the package's fetcher, which resumes and checks
  each file's SHA-256:

  ```powershell
  .\libexec\brilliance\hfget.ps1 <repo-id> [dest-dir] [revision] [file]
  .\libexec\brilliance\hfget.ps1 StillDeadcode/qwen3.8-27b-iqr-v4 D:\models\hf\StillDeadcode\qwen3.8-27b-iqr-v4 main qwen3.8-27b-iq4r-l-v4.rad
  ```

  `HF_TOKEN` authenticates a gated or private repository.
- **Point the profile at it:** edit `model:` in the profile (`profiles\qwen27-preview.yaml`, say) to the file's
  path. The profiles name `D:\models\hf\<owner>\<repo>\<file>`.
- **Or name a repository** as `model: owner/name`, and the server fetches it with
  `libexec\brilliance\hfget.ps1` before it starts, into `<dir>\<owner>\<name>`. `<dir>` is the
  `download-dir:` under `server:` if set, else `BRILLIANCE_HF_CACHE`, else `%LOCALAPPDATA%\radiance\hf`.
  This serves the one `.rad` at the repository's top level. A repository with several containers, such as
  `qwen3.8-27b-iqr-v4`, is refused; fetch the file instead.
  - Every start checks the repository and fetches what changed. A start that cannot reach it serves the copy
    fetched before.
  - Without `model-revision:` it follows the repository's `main` branch. Name a tag or a commit so a restart
    does not bring weights nobody reviewed.
- **Where a model is served from:** where its `model:` points. `serve.ps1` copies nothing; keep the files on
  the fastest drive you have.

The other containers are in [Supported models](#supported-models).

### 4. Start a server

From the unzipped folder, in PowerShell:

```powershell
.\serve.ps1 qwen27-preview -Check     # validate the profile and exit
.\serve.ps1 qwen27-preview -Bus 3     # serve on the card at PCI bus 3
```

- **`-Bus`** picks cards by PCI bus number (Device Manager or GPU-Z show it). Without it, every discrete card
  HIP shows is used, and the integrated GPU is left out. `-Device` takes HIP indices instead.
- It is ready when it prints `server: listening on http://127.0.0.1:8000`.
- **The API** is at `http://127.0.0.1:8000/v1`: Chat Completions, Responses and Messages, streamed or not,
  with tools, and pictures and video clips as `image_url` / `video_url` parts.
- **The dashboard** is at `http://127.0.0.1:8000/`: models, memory, performance, requests and sessions.
- **The terms of use** are at `http://127.0.0.1:8000/terms`, linked from the dashboard (section 6), and the
  models' full licence texts at `/terms/licenses`.
- **`-Background -Log FILE`** starts the server in a console of its own.
- **`-WhatIf`** prints what a start would set (the cards, the environment) and the command line, and starts
  nothing.
- **The environment:** a profile can name variables its server needs in a comment line,
  `# brilliance-env: NAME=VALUE`, and `-Env NAME=VALUE[,NAME=VALUE]` adds or overrides them for one start.
  Only `GPU_*`, `HIP_*`, `AMD_*`, `BRILLIANCE_*` and `RADIANCE_*` are taken; the cards stay with `-Bus`. They
  are set for the server only, and the startup output names each one.
- `bin\radiance.exe --config profiles\qwen27-preview.yaml` works too. With an integrated GPU or several cards,
  set `$env:HIP_VISIBLE_DEVICES` to the card's HIP index first. It does not read `# brilliance-env:` lines:
  set those variables yourself.

| Profile | Model | Cards | Status | Context | Requests at once | Speculation | KV cache |
|---|---|---|---|---|---|---|---|
| `qwen27-preview` | Qwen3.8-27B IQ4R-L-V4 | 1 | **Qualified** | 32K | 4 | DFlash2, depth 7 | bf16 |
| `qwen27-8k-plain` | Qwen3.8-27B IQ4R-L-V4 | 1 | **Qualified** | 8K | 1 | off | bf16 |
| `qwen27-iq8r-1card` | Qwen3.8-27B IQ8R-V4 | 1 | **Soak-clean** | 16K | 4 | DFlash2 | int8 |
| `qwen27-iq8r-2card` | Qwen3.8-27B IQ8R-V4 | 2 (TP2) | **Soak-clean** | 32K | 4 | DFlash2 | bf16 |
| `flashnext-4card` | Qwen3.8-Flash-Next IQR-V4 | 4 (TP4), every expert on the cards | **Soak-clean**, before the four-card re-send; with it, injection and serving checks pass | 32K | 8 | MTP, depth 3 | int8 |
| `flashnext-2card` | Qwen3.8-Flash-Next IQR-V4 | 2 (TP2), experts tiered to host memory | **Supported**: upstream's two-card settings | 200K | 8 | MTP, depth 3 | int8 |

- **Name the cards a profile needs** with `-Bus` (or `-Device`): `-Bus 6` for one card, `-Bus 3,9` for two.
  Without it every discrete card is visible and the profile takes the first one, two or four in HIP's order.
  A card left visible but unused still costs: HIP virtual memory is on only with exactly one card visible,
  and the pin limit is divided by every card the process sees. On a machine with exactly four cards,
  `.\serve.ps1 flashnext-4card` needs no `-Bus`.
- **`flashnext-4card` has a known issue:** Flash-Next's output changes when requests share a step. Identical
  requests sent together get different next-token distributions, and sampling over them is looser than the
  temperature asked; one request at a time behaves normally. Under investigation
  ([known issue 8](win/SUPPORT.md#known-issues)).
- **`flashnext-2card`** carries upstream's two-card settings on the exact wire (`tp-wire` left at its
  default), and with them a 64-byte write to host memory is lost now and then
  ([known issue 1](win/SUPPORT.md#known-issues)). The server re-sends the block, at about 4% of
  single-stream decode. It sets no runtime variable.
- `kv-cache-dtype: i8` halves the cache (5.3 GiB instead of 9.4 GiB at 32K x 4).

### 5. Stop it

- **Ctrl-C** in its window: it finishes what is in flight, releases the card and exits.
- From a script: `tools\stop-server.ps1 -ProcessId <pid>` sends Ctrl-Break, the same drain.
- **Do not end it from Task Manager** while it is working: a GPU process killed mid-work has taken the driver
  down. A server that does not stop can be read with `bin\brl-stacks.exe` (`tools\STACKS.md`).
- **Reload the YAML** without a restart with `tools\reload-server.ps1 -ProcessId <pid>` (Windows' counterpart
  of `kill -HUP`) or `POST /admin/reload`.

### 6. Serve the local network

The profiles listen on loopback with no key. `serve.ps1` changes that from the command line:

```powershell
.\serve.ps1 qwen27-preview -Listen 0.0.0.0 -ApiKey env:MY_KEY   # every interface; requires the key MY_KEY holds
.\serve.ps1 qwen27-preview -Listen 0.0.0.0 -ApiKey off          # every interface, no key
```

- `-ApiKey` takes `off` or `env:NAME`. A key itself is refused on the command line, where other processes can
  read it. Without `-ApiKey`, the profile's own key setting stands.
- **The firewall rule is yours to add.** Allow the port for the private network only.
- **Adopt the terms of use before serving other people.** Some model licences (LTX-2.5's, SDXL's) require
  whoever serves the model to other people to pass its use restrictions on to them as enforceable terms. The
  server shows a terms-of-use page at `/terms`, open without the key. Out of the box it is
  [win/TERMS-OF-USE.md](win/TERMS-OF-USE.md) (`TERMS-OF-USE.md` in the package), a template that carries
  those restrictions. Copy it, make it yours (name yourself, remove the models you do not serve, add your own
  terms), and set `terms-file:` under `server:` in the profile to your copy; `.html` is served as HTML,
  anything else as plain text. The file is read at startup, and one that cannot be read stops the start. The
  template is not legal advice.

### 7. Optional: run as a Windows service

`.\serve.ps1 <profile> -Install` registers the profile as a service, started at boot and stopped by the same
drain as Ctrl-C. **It is not advertised yet:** it passes its host tests, and its qualification in an elevated
shell is open. [SERVICE.md](win/SERVICE.md) has the steps.

- Unpack the package under `C:\Program Files` first: the service runs as LocalSystem from that folder.
- Run it from an elevated PowerShell. `-WhatIf` prints every step first and changes nothing.
- With `-ApiKey env:NAME`, the key goes into a file only SYSTEM and Administrators can read, never onto a
  command line or into the registry.
- The cards, `BRILLIANCE_HF_CACHE`, and the profile's `# brilliance-env:` and `-Env` variables go into the
  service's own environment.
- A failure nobody asked for restarts it once, after 2 minutes, and not again that day (`-NoRestart`: never).
- `-Uninstall` stops it with the drain and removes it.

## Dependencies

### What the package bundles

The [third-party index](win/licenses/THIRD-PARTY.md) lists each DLL, the libraries built into it, its licence
and its source.

| In `bin\` | What it is | Licence |
|---|---|---|
| `amdhip64_7.dll`, `amd_comgr.dll`, `rocm_kpack.dll` | AMD's HIP runtime (HIP 7.16.26385), code-object manager and kernel-pack reader, from AMD's ROCm SDK 10.1.0 wheel | MIT, NCSA and Apache-2.0 WITH LLVM-exception; AMD's EULA for part of the HIP runtime ([below](#licence-and-credits)) |
| `avcodec-62.dll`, `avformat-62.dll`, `avutil-60.dll`, `swresample-6.dll`, `swscale-9.dll` | FFmpeg 8.1.3, Brilliance's own LGPL shared build (vcpkg, [win/ffmpeg](win/ffmpeg/)): picture, video and audio decoding, picture encoding, and H.264 only through the `h264_amf` and `h264_mf` wrappers over AMD's and Windows' encoders; no H.264 encoder implementation ships ([H.264](#h264)) | LGPL-2.1-or-later, linked dynamically |
| `libomp.dll` | LLVM's OpenMP runtime | Apache-2.0 WITH LLVM-exception AND NCSA |
| `vcruntime140*.dll`, `msvcp140*.dll`, `concrt140.dll`, `vccorlib140.dll` | Microsoft's Visual C++ runtime, app-local | Visual Studio Distributable Code |

The engine's own third-party code (cpp-httplib, yyjson, llama.cpp and the rest) is named in [NOTICE](NOTICE).

### What you need to run it

- **The AMD Adrenalin driver**, 32.0.32015.12012 or later.
- **Python 3, only for `rad-export`** (`bin\rad-export.cmd`, turning a finetune into a container), plus
  torch and safetensors for a LoRA adapter, and for `resolve-stacks.py`, which names a hung server's frames
  (`tools\STACKS.md`). The source tree's test and qualification tools (`win\tools\*.py`) use Python too.
  Serving needs no Python.
- `hfget.ps1` uses the `curl.exe` and PowerShell that Windows ships.

### What you need to build from source

- **Git.**
- **Visual Studio Build Tools with the C++ workload:** the MSVC 14.51 toolset and the Windows SDK
  10.0.26100.0. ROCm's clang compiles against the MSVC STL, the UCRT and the Windows SDK, and links with
  them. `setup.ps1` checks for them and does not install them.
- **Everything else is pinned and fetched by [`win\setup.ps1`](win/setup.ps1)** into `.toolchain\` in the
  checkout, each archive checked against the SHA-256 written in the script:

| `.toolchain\` | Version |
|---|---|
| `rocm-venv` | ROCm 10.1.0 (TheRock wheels: ROCm's clang, the SDK, and the `gfx1201` device libraries) |
| `cmake` | CMake 4.4.5 |
| `ninja` | Ninja 1.13.2 |
| `ffmpeg` | FFmpeg 8.1.3, Brilliance's LGPL shared build ([win/ffmpeg](win/ffmpeg/)), pinned by the zip's SHA-256 |
| `python` | CPython 3.13.18, used only to hold the ROCm wheels |
| `downloads` | the archives themselves, kept so a reinstall needs no network |

The ROCm wheels are installed from `downloads` with `pip --no-index` after their hashes are checked, so a
configure never sees a version the script does not name. The result is recorded in
`.toolchain\toolchain.json`, which `win\env.ps1` reads.

## Supported models

Status words as [win/SUPPORT.md](win/SUPPORT.md) records them, measured on four R9700s at -20% board power
(240 W), -50 mV and the stock VRAM clock (2518 MHz).

| Model (container) | 1 card | 2 cards | 4 cards |
|---|---|---|---|
| Qwen3.8-27B IQ4R-L-V4 ([`StillDeadcode/qwen3.8-27b-iqr-v4`](https://huggingface.co/StillDeadcode/qwen3.8-27b-iqr-v4)) | **Qualified**: 8K and 32K x 4, DFlash2 or plain, bf16 or i8 KV; text, tools, pictures, video | Not run | Not run |
| Qwen3.8-27B IQ8R-V4 (same repository, `qwen3.8-27b-iq8r-v4.rad`, 27.58 GiB) | **Soak-clean**: 16K x 4, int8 KV, DFlash2 | **Soak-clean**: 32K x 4, DFlash2 | Not run |
| Qwen3.8-Flash-Next IQR-V4 ([`StillDeadcode/qwen3.8-next-flash-iqr-v4`](https://huggingface.co/StillDeadcode/qwen3.8-next-flash-iqr-v4), 113 GiB) | Not run (upstream lists a one-card layout with experts in host memory) | **Supported**: with production settings a lost write is re-sent, at about 4% of single-stream decode. The 8-hour soak is still to run. Profile: `flashnext-2card` | **Soak-clean**: every expert on the cards (`placement: all_vram`), 32K x 8, int8 KV, MTP; soaked before the four-card re-send and stop were added; with them it passes the injection and serving checks |
| MiniCPM5-2B IQ8R ([`StillDeadcode/minicpm5-2b-iq8r`](https://huggingface.co/StillDeadcode/minicpm5-2b-iq8r)) | Runs (the bring-up model); also on the CPU package, without its drafter: set `num-speculative-tokens: 0` | - | - |
| EmbeddingGemma 2 bf16 (`google/embeddinggemma-2`, converted with `rad-convert`, 1.40 GiB) | **Qualified**: `/v1/embeddings` for text, pictures, video and audio; also on the CPU kernels | - | - |
| Qwen3-ASR 1.7B bf16 (`Qwen/Qwen3-ASR-1.7B-hf` as it ships, 4.09 GB) | **Qualified**: `/v1/audio/transcriptions` (5 formats, 4 response forms) and `/v1/realtime`, 16 sequences | - | - |
| Nemotron 3.5 Rerank 8B BF16 (`nvidia/Nemotron-3.5-Rerank-8B-BF16` as it ships, 16.78 GB) | **Qualified**: `/rerank` and `/v1/rerank`, text and picture documents, 32,768 tokens a pair | - | - |
| SDXL base 1.0 bf16 (`stabilityai/stable-diffusion-xl-base-1.0`, the single-file checkpoint converted with `rad-convert`, 6.39 GiB) | **Qualified**: `/v1/images/generations`; PNG, JPEG and WebP; about 5.4 s at 1024x1024, 28 steps | - | - |
| LTX-2.5 distilled IQ8R (`StillDeadcode/radiance-ltx-2.5-distilled-iq8r`) | **Runs, not yet qualified**: `/v1/videos`, text and image to video, MP4 | - | - |
| Every other upstream model | Source builds; not run on Windows | | |

**The package's profiles** cover the Qwen3.8 layouts above that are **Qualified** or **Soak-clean**, and one
that is **Supported** ([step 4](#4-start-a-server) has their settings):

- Qwen3.8-27B IQ4R-L-V4, 1 card, **Qualified**: `qwen27-preview` (32K x 4) and `qwen27-8k-plain` (8K, one request).
- Qwen3.8-27B IQ8R-V4, 1 card, **Soak-clean**: `qwen27-iq8r-1card`.
- Qwen3.8-27B IQ8R-V4, 2 cards, **Soak-clean**: `qwen27-iq8r-2card`.
- Qwen3.8-Flash-Next IQR-V4, 4 cards, **Soak-clean**: `flashnext-4card`, soaked before the four-card re-send
  and stop were added; with them it passes the injection and serving checks.
  Its known issue: the output changes when requests share a step, so sampling is looser than asked (under
  investigation).
- Qwen3.8-Flash-Next IQR-V4, 2 cards, **Supported**: `flashnext-2card`, upstream's two-card settings on the
  exact wire; a lost write is re-sent. The 8-hour soak is still to run.

**Brilliance 0.2.2's containers do not load.** Radiance 2.1 does not read the 0.2.2-era formats: whole-model
FP8, EXL3, the old Flash-Next FP8/IQ4R, GLM-5.3-Flash MXFP4, DeepSeek-V4.1 and Gemma 4 FP8. Use the current
containers above.

## Feature parity

Radiance 2.1's features, and where each stands on Windows, from [win/SUPPORT.md](win/SUPPORT.md).

| Radiance 2.1 feature | On Windows | Note |
|---|---|---|
| Chat Completions, Responses, Messages; streaming | **Yes** | Qualified on Qwen3.8-27B |
| Tools, reasoning | **Yes** | Qualified on Qwen3.8-27B; Open WebUI and OpenCode tool round trips pass |
| Grammars and structured output (`response_format`) | **Unverified** | Built; the host grammar tests pass; not checked against a served model |
| Pictures and video in requests (`image_url`, `video_url`) | **Yes** | Through the bundled FFmpeg 8.1; checked on Qwen3.8-27B. H.264 input is decoded by FFmpeg's built-in decoder, with no H.264 patent licence supplied ([H.264](#h264)) |
| Audio in requests | **Yes** | In an embeddings request (EmbeddingGemma 2) and for transcription (Qwen3-ASR 1.7B, one card) |
| Embeddings (`/v1/embeddings`) | **Yes** | EmbeddingGemma 2, on one card and on the CPU kernels; batches, `dimensions`, base64 |
| Rerank | **Yes** | Nemotron 3.5 Rerank 8B, one card; `/rerank` and `/v1/rerank`, text and picture documents |
| Transcription (`/v1/audio/transcriptions`), realtime | **Yes** | Qwen3-ASR 1.7B, one card; five upload formats, `/v1/realtime` over a WebSocket |
| Speech synthesis (TTS) | **Not supported in 0.3.0** | `/v1/audio/speech`, `/audio/speech` and `/v1/audio/speech/stream` are refused (501; the socket closes 1008) |
| Image generation | **Yes** | SDXL base 1.0 bf16, converted on the CPU from the single-file checkpoint plus CLIP's `tokenizer.json` (9.6 s, 6.39 GiB), on one card; PNG, JPEG and WebP as `b64_json`; `url` refused; seeds repeat; `n` 1-8; about 5.4 s at 1024x1024, 28 steps |
| Video generation | **Works, not yet qualified** | LTX-2.5 distilled (IQ8R) on one card, the model for 0.3.0. MP4s through AMD's (`h264_amf`), else Windows' (`h264_mf`) licensed H.264 encoders, each marked AI-generated in its metadata ([H.264](#h264)) |
| Music generation | **Not yet** | Built; host tests pass; no model on hand. Writes WAV or raw PCM itself. Its endpoint, `/v1/audio/speech`, is refused (speech, above) |
| Multi-model fleet, routing, reload | **Yes** | Reload by `POST /admin/reload` or `tools\reload-server.ps1` (Windows' SIGHUP) |
| Model switching | **Yes** | One card visible: by parking. Several cards visible: by unloading and reloading; since 2026-11-18 a reload keeps its VRAM (PAL's resource cache off); switching repeats at TP2 not run |
| Parking (HIP virtual memory) | **Partial** | On with exactly one device visible: an elastic KV cache, and parking with no host copies. Off with more than one (rocm-systems#13232) |
| Prefix cache, checkpoint snapshots, agent replay | **Yes** | |
| Speculation: DFlash2, MTP | **Yes** | |
| Tensor parallel, 2 cards | **Partial** | Soak-clean when the weights fit the cards; supported when experts are read from host memory (a lost write is re-sent; the 8-hour soak still to run), on the exact wire; `tp-wire: wht6` runs with the same protection, not qualified |
| Tensor parallel, 4 cards | **Partial** | Flash-Next soak-clean with every expert on the cards, before the four-card re-send; with it, a lost write is re-sent as on two cards and the injection and serving checks pass; 3 and 8 cards not run |
| Peer-to-peer between cards | **Not on Windows** | Collectives go through pinned host memory |
| Dashboard | **Partial** | Per card it shows temperature, not load or power |
| Diagnose mode | **Yes** | The busy-host warning reads the CPUs' load over a quarter second |
| `rad-convert` | **Partial** | CPU only, by decision; `--gpus` is refused. A real model converted: SDXL, 9.6 s on the CPU |
| `rad-export` (finetune to container) | **Yes** | With Python 3 (`bin\rad-export.cmd`) |
| Model fetch by repository id | **Yes** | `hfget.ps1`, with curl.exe and PowerShell |
| Running as a service | **Not yet** | `radiance --service`, installed by `serve.ps1 -Install`; host-tested, its elevated-shell qualification open ([SERVICE.md](win/SERVICE.md)) |
| `--ngram-placement ram=DIR` | **Partial** | Device tests pass; not run end to end |
| Plugin SDK, out-of-tree plugins | **Partial** | From a source build's install; not in the package |
| Linux `scripts/` (measurement, traces, Docker) | **Not on Windows** | `win/tools` has the Windows tools |
| Cards other than `gfx1201` | **Not on Windows** | The toolchain is pinned to `gfx1201` |

## From Brilliance 0.2.2

- **Models:** 0.2.2's containers do not load; download the current ones ([Supported models](#supported-models)).
- **Configuration:** a YAML file (`--config`), as upstream's; `serve.ps1 <profile>` replaces 0.2.2's launcher
  flags. `--check` validates a file without loading.
- **Service:** `serve.ps1 -Install` as in 0.2.2, now on the YAML file. The API key is read from a protected
  file, never stored on the service's command line.
- **Stopping:** Ctrl-C / Ctrl-Break drain, as in 0.2.2; never end a working server from Task Manager.

## Building from source

From PowerShell, with the build prerequisites [above](#what-you-need-to-build-from-source):

```powershell
git clone https://github.com/adamczhang/Brilliance.git
cd Brilliance
powershell -NoProfile -ExecutionPolicy Bypass -File win\setup.ps1                     # check the machine; fetch the pinned toolchain
powershell -NoProfile -ExecutionPolicy Bypass -File win\build.ps1 -Flavor hip -Test   # the GPU build and its tests
powershell -NoProfile -ExecutionPolicy Bypass -File win\package.ps1                   # build-win\hip -> dist\Brilliance-<version>-<sha>-win64.zip
```

- `setup.ps1 -CheckOnly` reports and changes nothing. It is safe to re-run.
- `build.ps1` makes its own process a build shell (`win\env.ps1`). `-Flavor host` (the default) builds the
  host backend, every tool, plugin and test that needs no card; `-Flavor hip` the GPU build. `-Test` runs the
  test gate.
- `package.ps1` packages a build that is current; it does not build. Run on `build-win\host`
  (`-BuildDir build-win\host`), it makes the CPU package.

## Documentation

| | |
|---|---|
| [win/SUPPORT.md](win/SUPPORT.md) | models and card counts, upstream's features on Windows, the known issues, and what changes from 0.2.2 |
| [win/PREVIEW.md](win/PREVIEW.md) | the package's quick start and known limits |
| [win/SERVICE.md](win/SERVICE.md) | running the server as a Windows service, and its qualification steps |
| [win/TERMS-OF-USE.md](win/TERMS-OF-USE.md) | the terms-of-use template the server shows at `/terms`, to adopt before serving other people |
| [win/licenses/THIRD-PARTY.md](win/licenses/THIRD-PARTY.md) | every third-party DLL and library in the package, its licence and source |
| [tools/STACKS.md](win/tools/stacks/README.md) | reading a server that stopped making progress (`brl-stacks`) |
| [RADIANCE-README.md](RADIANCE-README.md), [docs/](docs/) | upstream Radiance's README and documentation, unchanged; [docs/GUIDE.md](docs/GUIDE.md) covers configuration and the HTTP API |

## Licence and credits

Brilliance is a fork of [Radiance](https://codeberg.org/StillDeadcode/radiance), the C++/HIP inference engine
by **Deadcode and the radiance contributors**. The engine, its kernels, its model plugins and the `.rad` format
are theirs; Brilliance adds the Windows port. To support the development of Radiance:
[Patreon](https://patreon.com/StillDeadCode).

- **Licence:** [Apache-2.0](LICENSE), as upstream's. Keep [NOTICE](NOTICE) with any copy or derivative: it
  credits Radiance's authors and lists the third-party code the engine includes.
- **The package's third-party components** carry their own licences. Their texts are in
  [win/licenses](win/licenses/) (in the package, `share\doc\radiance\third_party`), indexed by
  [THIRD-PARTY.md](win/licenses/THIRD-PARTY.md). FFmpeg is under the LGPL (2.1 or later) and linked
  dynamically.
- **AMD's EULA.** On Windows, AMD's HIP runtime (`amdhip64_7.dll`) is built with code from the directory
  `shared/amdgpu-windows-interop` of ROCm's rocm-systems, whose root licence is the AMD Software End User
  License Agreement, not an open-source licence. The package keeps the DLL and ships the EULA text unedited
  beside the MIT and NCSA texts. The package is for AMD GPUs. The details are in
  [THIRD-PARTY.md, "AMD's EULA in amdhip64_7.dll"](win/licenses/THIRD-PARTY.md#amds-eula-in-amdhip64_7dll).
- **H.264:** what ships and what does not is in [H.264](#h264) below.
- **Model weights** are not covered by these licences: each keeps the licence of the model it was converted
  from. [PREVIEW.md](win/PREVIEW.md), "What is in this package", lists the served models' licences.
  - **LTX-2.5 (video)** is under Lightricks' LTX-2.x Community License Agreement. Its Hugging Face repository
    is gated: each user accepts the licence there with their own account.
  - Entities with US$10M or more a year in revenue need a paid licence from Lightricks for any use other than
    a Non-Commercial Purpose.
  - Its use restrictions (section 4 and Attachment A) cover the videos too, and must be passed on to anyone
    you serve.
  - Generated videos must be disclosed as machine generated. Brilliance marks each MP4 so in its metadata;
    keep that marking.
  - [win/TERMS-OF-USE.md](win/TERMS-OF-USE.md), the template the server shows at `/terms`, carries the use
    restrictions to pass on ([Serve the local network](#6-serve-the-local-network)).

Thanks to **Roberto Franchetti** for one-to-one testing of Brilliance 0.2 on a four-R9700 machine of their
own. Those runs traced the earlier port's pinned-memory limit to the integrated GPU left visible to the
runtime.

### H.264

What ships for H.264 ([THIRD-PARTY.md, "H.264"](win/licenses/THIRD-PARTY.md#h264), has the whole of it):

- **No H.264 encoder implementation ships in Brilliance 0.3.0.** OpenH264 and x264 are not included.
- **The bundled FFmpeg keeps exactly two H.264 encoders**, both thin wrappers over encoders their makers
  license:
  - `h264_amf`: AMD's hardware encoder, through AMF, whose runtime comes with the AMD driver;
  - `h264_mf`: Windows' Media Foundation encoder. Windows "N" editions lack it unless Microsoft's Media
    Feature Pack is installed.

  Any H.264 output goes only through these two, AMD's first. The encoder vendors' licences are what cover
  H.264 output through their encoders; this is not legal advice from the project.
- **Video generation** writes H.264 only through AMD's (`h264_amf`), else Windows' (`h264_mf`) licensed
  encoders, and so does image to video's first-frame round trip. It works on one card and is not yet
  qualified. `h264_amf` encodes on the first display adapter Windows lists: on a machine with an AMD
  integrated GPU, the integrated GPU, not the card that runs the model.
- **Speech synthesis (TTS) is not supported** in 0.3.0; its endpoint answers 501. Music
  generation, which shares that endpoint, is not served either (no model yet); it writes WAV
  or raw PCM and needs no encoder.
- **Decoding H.264 input** (video sent in requests) uses FFmpeg's built-in decoder. H.264 decoding is
  patent-encumbered, and no H.264 patent licence is supplied.
- **The LGPL source** of the bundled FFmpeg is the recipe in [win/ffmpeg](win/ffmpeg/) plus two assets of the
  [v0.3.0-alpha.1 release](https://github.com/adamczhang/Brilliance/releases/tag/v0.3.0-alpha.1): the FFmpeg
  zip, pinned by its SHA-256, whose `source\` folder holds the recipe and patches, and FFmpeg's own `n8.1.3`
  source archive.
