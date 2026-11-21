![Brilliance](docs/img/banner.png)

# Brilliance

A native Windows 11 distribution of [Radiance](https://codeberg.org/StillDeadcode/radiance), the C++/HIP
LLM inference engine, for AMD Radeon AI PRO R9700 cards (RDNA4, `gfx1201`): no WSL, no Linux, no Docker.
It serves Radiance's HTTP API: OpenAI-style Chat Completions and Responses, and Anthropic-style Messages.
A client that lets you set its base URL and model name can use it ([Connect a client](#connect-a-client));
Open WebUI and OpenCode tool round trips were checked on Qwen3.8-27B.

**This is a new fork, based on Radiance 2.2:** upstream's `v2.2.0` tag (`b6b8751c`, 2026-11-19), with the
Windows port on top. It is not a continuation of Brilliance 0.2.2's code. It keeps upstream's YAML
configuration and model implementations; the port adds a Windows device backend and platform layer, and
some of upstream's features work differently on Windows or not at all ([Feature parity](#feature-parity)).
Plugins are built for Windows from a source build's SDK.

- **The new home** is [github.com/adamczhang/Brilliance](https://github.com/adamczhang/Brilliance).
- **Brilliance 0.2.2 is archived** at its original address,
  [codeberg.org/Dad_Vibes/Brilliance](https://codeberg.org/Dad_Vibes/Brilliance). Its `.rad` containers do
  not load in this version ([from 0.2.2](#from-brilliance-022)).
- **Brilliance 0.3.0-alpha.5 is this fork's current release: the fifth alpha preview of 0.3.0**
  (0.3.0-alpha.1 was the first public release). Its package, `Brilliance-0.3.0-alpha.5-<sha>-win64.zip`,
  is on the [v0.3.0-alpha.5 release page](https://github.com/adamczhang/Brilliance/releases/tag/v0.3.0-alpha.5).
  Expect rough edges: what is qualified and what is not is listed below and in
  [win/SUPPORT.md](win/SUPPORT.md).

Upstream's own README is kept unchanged as [RADIANCE-README.md](RADIANCE-README.md).

## How you use it

- **Brilliance is an API server.** You start it on a model, then connect an app or agent to it: its
  OpenAI-style base URL is `http://<host>:<port>/v1` and its Anthropic-style base URL is
  `http://<host>:<port>` ([Connect a client](#connect-a-client)).
- **Its dashboard, at `http://<host>:<port>/` in a browser, is for watching it:** each model's
  Monitoring tab has its speed, requests, sessions and details. A model's **Sandbox** tab is for a
  quick check that the model works: Start and Stop load and unload it (from a browser on the
  server's own machine), and a plain chat box streams its replies. It keeps nothing. Start holds the
  model until Stop: a request for another model that needs its cards is refused meanwhile, rather
  than moving it. From another
  machine, a server without an API key shows every model's own pages and Sandbox (not the
  overview), and its chat works with a model that is loaded.
- **For everyday chat in a browser, use Open WebUI; for coding, OpenCode.** Both connect as in
  [Connect a client](#connect-a-client).

## Status

Alpha preview (0.3.0-alpha.5). What is qualified on Windows today, from [win/SUPPORT.md](win/SUPPORT.md):

- **One card:** Qwen3.8-27B (IQ4R-L-V4 qualified, IQ8R-V4 soak-clean); Gemma 4 31B a4 and Gemma 4
  12B i8g, qualified (chat, tools, pictures and video; audio too on the 12B).
- **Two cards:** soak-clean when the weights fit the cards (Qwen3.8-27B IQ8R-V4); Qwen3.6-35B-A3B
  IQ8R qualified, every expert on the cards. DeepSeek-V4.1-Flash-1017 runs as a preview with the experts
  that do not fit held in host memory; its greedy text can differ from four cards'.
- **Four cards:** Qwen3.8-Flash-Next IQR-V4, supported with every expert on the cards (soaked at 32K on
  its earlier settings, 2026-11-17; its profile's tuned 256K settings pass the injection, stop and
  serving checks; a soak on them has not run yet). DeepSeek-V4.1-Flash-1017 (bakon3's uncensored
  container) runs as a preview, every routed expert on the cards, not yet qualified.
- **Model switching on one card** by parking, with HIP virtual memory.
- **Embeddings** (`/v1/embeddings`) with EmbeddingGemma 2.
- **Transcription and realtime** (`/v1/audio/transcriptions`, `/v1/realtime`) with Qwen3-ASR 1.7B, and
  **rerank** (`/rerank`, `/v1/rerank`) with Nemotron 3.5 Rerank 8B, each on one card.
- **Image generation** (`/v1/images/generations`) with SDXL base 1.0 and Qwen-Image 2.1 IQ8R, each
  on one card. Qwen-Image's edits work for most seeds ([known issue 13](win/SUPPORT.md#known-issues)),
  and its licence allows non-commercial use only.
- **Clients:** Open WebUI and OpenCode pass their tool round trips.

The main known issues ([all of them](win/SUPPORT.md#known-issues)):

- **Flash-Next on two cards loses a write now and then** with its production settings, where experts are
  read from host memory. A lost block is re-sent and the step completes, at about 4% of single-stream
  decode; one that cannot be recovered stops the server, which exits with status 70. Flash-Next on two
  cards is Supported; an 8-hour soak has not been run on it yet. Profile: `flashnext-2card`.
- **Flash-Next's output can change when requests share a step:** identical requests sent together can get
  different next-token distributions. On the tested prompts this came from rounding, not wrong data, as
  upstream documents for batched requests. Compare repeats of one request from the same cache state: in
  the two-card checks, a long prompt sent again and resumed from the prefix cache changed its
  reasoning text, with the same final answer.
- **Video generation works on one card and is not yet qualified** (LTX-2.5 distilled IQ8R, profile
  `ltx-2.5-1card`). It writes H.264 only through AMD's or Windows' licensed encoders, on Brilliance's own
  FFmpeg ([H.264](#h264)).
- **Speech synthesis (TTS) is not supported** in 0.3.0, and **music generation is not served**:
  `/v1/audio/speech`, which both use, answers 501, "not supported in this build", for every model.
  A request for something the loaded model does not do (chat on an image model, say) answers 501 with the
  reason. A path the server does not have answers 404, and a missing key or a malformed request is refused
  before any model sees it.

## Installation

### 1. Requirements

Check your hardware before you download anything:

| Your hardware | What this package does |
|---|---|
| AMD Radeon AI PRO R9700 (`gfx1201`, 32 GB) | The card it is built for and tested on. What each model does on one, two or four of them is in [Supported models](#supported-models). |
| AMD Radeon RX 9070 family (`gfx1201`) | The same GPU code target, so the package can start on these cards, but **Not run**: no model has been checked on one. They have 16 GB or less, which the 17.42 GiB starter model does not fit on one card. |
| Other AMD GPUs (RDNA4 `gfx1200`, RDNA3 `gfx11xx`, Strix Halo `gfx1151`, older), NVIDIA or Intel | **Not supported** by this package: it carries GPU code for `gfx1201` only. Upstream Radiance serves several of these AMD targets on Linux. |
| No usable GPU | The same package serves a small model on the CPU: `.\serve.ps1 <profile> -Cpu` ([Serving on the CPU](#serving-on-the-cpu)). Checked with MiniCPM5-2B IQ8R and EmbeddingGemma 2, not qualified; expect a fraction of a card's speed. A CPU-only package can be built from source. |

- **Windows 11.**
- **The AMD Adrenalin driver 32.0.32015.12012**, the version it was tested on. Later drivers have not been
  checked; if one misbehaves, report it with its version ([Getting help](#getting-help)).
- **Memory.**
  - One card holds Qwen3.8-27B IQ4R-L-V4 (a 17.42 GiB file). IQ8R-V4 on one card (26.67 GiB of weights)
    leaves about 3 GiB of KV cache.
  - Qwen3.8-Flash-Next IQR-V4 (113 GiB) does not fit one card's memory; four cards hold every expert.
    Upstream also serves it on one card with experts in host memory; that layout is not run on Windows.
  - DeepSeek-V4.1-Flash-1017 (a 251 GiB file, the largest) runs on four cards with every routed expert on
    them; its 103 GiB Engram table is read from the file, so keep the file on a fast local drive. On two
    cards the experts that do not fit are held in host memory: about 61 GiB less is available while it
    serves, and it refuses to start when it would leave less than 24 GiB.
  - A process can pin about the shared-memory budget (half of RAM or more), divided by the cards it sees,
    less 2 GiB. Multi-card exchanges go through that pinned memory.
- **Disk** for the models you serve.
- **Recommended:** an integrated GPU or another display card for the desktop. The engine never runs a model
  on an integrated GPU beside a discrete card, but while HIP can still see it, it divides the pinned-memory
  limit and turns HIP virtual memory off. `serve.ps1` hides it from HIP unless told otherwise, and `-Bus`
  chooses the cards that serve ([step 4](#4-start-a-server)).

### 2. Get the package

Download `Brilliance-0.3.0-alpha.5-<sha>-win64.zip` from the
[v0.3.0-alpha.5 release](https://github.com/adamczhang/Brilliance/releases/tag/v0.3.0-alpha.5), check it
against the `.sha256` beside it (`Get-FileHash <zip>` in PowerShell), and unzip it anywhere; a path with spaces
works. You can also [build it from source](#building-from-source). With no usable GPU, the same package serves
on the CPU: `serve.ps1 -Cpu` ([Serving on the CPU](#serving-on-the-cpu)). A CPU-only package
(`Brilliance-<version>-<sha>-win64-cpu.zip`), which serves on the CPU kernels and needs no card, can be built
from source; it is not published with this release.

| In the package | What it is |
|---|---|
| `bin\` | `radiance.exe`, the `rad-*` tools, the HIP runtime, FFmpeg, the VC++ runtime; `brl-stacks.exe` for reading a hung server |
| `share\radiance\` | the plugins (kernels, architectures, quantizers) and the conversion recipes |
| `share\doc\radiance\` | `LICENSE`, `NOTICE`, and the third-party licences under `third_party\` |
| `profiles\` | YAML profiles; status and card count for each in [step 4](#4-start-a-server) |
| `libexec\brilliance\` | `hfget.ps1`, the Hugging Face fetcher |
| `tools\` | `serve-bg.ps1`, `stop-server.ps1`, `reload-server.ps1`, `STACKS.md` |
| `serve.ps1` | starts a profile, choosing cards by PCI bus |
| `PREVIEW.md`, `SUPPORT.md`, `SERVICE.md` | quick start and limits; models, features and known issues; running as a service |
| `MANIFEST.json` | the source commit, toolchain, HIP runtime and the driver on the build machine, and the SHA-256 of every other file in the package |

Nothing else is installed: no ROCm, no VC++ redistributable.

**Open PowerShell in the unzipped folder:** in File Explorer, type `powershell` in the folder's address bar
and press Enter. Windows PowerShell runs no scripts on many machines until told to, so allow the package's
scripts for this window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force   # this PowerShell window only; nothing is changed for good
```

Run that again in each new PowerShell window you use for the commands below. If it reports that a policy set
by your organisation overrides it, the scripts cannot run on that machine until its administrator allows
them; check the ZIP's hash and follow your organisation's script policy.

### 3. Get a model

The profiles serve eight model containers. There are two ways in, each on one card:

- **A small first run:** MiniCPM5-2B IQ8R, `minicpm5-2b-1card`, a 3.04 GiB text-only model that loads in
  seconds ([below](#a-small-first-run-minicpm5-2b)). Good on a slow link or a small disk.
- **The full example:** Qwen3.8-27B IQ4R-L-V4, `qwen27-preview`, the qualified model, 17.42 GiB, with tools,
  pictures and video. The numbered steps below and steps 4 to 6 use it.

Every container is on Hugging Face:

| Container file | Repository | Size | Profiles |
|---|---|---|---|
| `minicpm5-2b-iq8r.rad` | [`StillDeadcode/minicpm5-2b-iq8r`](https://huggingface.co/StillDeadcode/minicpm5-2b-iq8r) | 3.04 GiB | `minicpm5-2b-1card` |
| `qwen3.8-27b-iq4r-l-v4.rad` | [`StillDeadcode/qwen3.8-27b-iqr-v4`](https://huggingface.co/StillDeadcode/qwen3.8-27b-iqr-v4) | 17.42 GiB | `qwen27-preview`, `qwen27-8k-plain` |
| `qwen3.8-27b-iq8r-v4.rad` | the same repository | 27.58 GiB | `qwen27-iq8r-1card`, `qwen27-iq8r-2card` |
| `qwen3.8-next-flash-iqr-v4.rad` | [`StillDeadcode/qwen3.8-next-flash-iqr-v4`](https://huggingface.co/StillDeadcode/qwen3.8-next-flash-iqr-v4) | 113.32 GiB | `flashnext-4card`, `flashnext-2card`, `flashnext-2card-fast` |
| `gemma4-31b-a4.rad` | [`keptn/gemma4-31b-a4`](https://huggingface.co/keptn/gemma4-31b-a4) | 19.95 GiB | `gemma4-31b-1card` |
| `gemma4-12b-i8g.rad` | [`keptn/gemma4-12b-i8g`](https://huggingface.co/keptn/gemma4-12b-i8g) | 13.79 GiB | `gemma4-12b-1card` |
| `qwen3.6-35b-a3b-iq8r.rad` | [`StillDeadcode/qwen3.6-35b-a3b-iq8r`](https://huggingface.co/StillDeadcode/qwen3.6-35b-a3b-iq8r) | 35.04 GiB | `qwen36-35b-2card` |
| `qwen-image-2.1-iq8r.rad` | [`StillDeadcode/radiance-qwen-image-2.1-iq8r`](https://huggingface.co/StillDeadcode/radiance-qwen-image-2.1-iq8r) | 16.13 GiB | `qwen-image-1card` |
| `ltx-2.5-distilled-iq8r.rad` | [`StillDeadcode/radiance-ltx-2.5-distilled-iq8r`](https://huggingface.co/StillDeadcode/radiance-ltx-2.5-distilled-iq8r) (gated: accept its licence with your own account) | 36.86 GiB | `ltx-2.5-1card` |
| `deepseek-v4.1-flash-1017-uncensored.rad` | [`bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad`](https://huggingface.co/bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad) (six parts to join, as its `RUNNING.md` says; `-Fetch` does not join them) | 251.13 GiB | `deepseek41-4card`, `deepseek41-2card` |

Each profile's comment has its file's SHA-256 (`qwen3.8-27b-iq4r-l-v4.rad`:
`d57d1428218e20a4cd578c3e286f30a2e566e589c832bf5004f49e164cb48ae4`). The other models in
[Supported models](#supported-models) have no profile in the package.

1. **Choose where the file goes.** Every profile expects `D:\models\hf\<owner>\<repo>\<file>`. Put it there
   and the profile needs no edit; any other folder works once you edit the profile (step 3 below).
2. **Download the one file**, from the repository's "Files" page or with the package's fetcher. The fetcher
   resumes a partial download and checks each large file it downloads against the SHA-256 the repository
   lists (`HF_TOKEN` authenticates a gated or private repository):

   ```powershell
   # .\libexec\brilliance\hfget.ps1 <repo-id> <dest-dir> <revision> <file>
   .\libexec\brilliance\hfget.ps1 StillDeadcode/qwen3.8-27b-iqr-v4 D:\models\hf\StillDeadcode\qwen3.8-27b-iqr-v4 main qwen3.8-27b-iq4r-l-v4.rad
   Get-FileHash D:\models\hf\StillDeadcode\qwen3.8-27b-iqr-v4\qwen3.8-27b-iq4r-l-v4.rad   # compare with the SHA-256 above
   ```

3. **Elsewhere, point the profile at it:** open `profiles\qwen27-preview.yaml` in a text editor and change the
   `model:` line to the file's full path. Use forward slashes, and put a path with spaces in double quotes:

   ```yaml
   models:
     qwen3.8-27b:
       model: "C:/Users/me/My Models/qwen3.8-27b-iq4r-l-v4.rad"
   ```

   `qwen3.8-27b`, the name above `model:`, is the model's name in the API.
4. **Check the path:** `Test-Path` on it must print `True`. The profile check in step 4 reads the YAML only and
   does not open the model file.

**A model is served from where its `model:` points.** `serve.ps1` copies nothing; keep the files on the
fastest drive you have.

#### Fetching with serve.ps1

Steps 1 to 4 can be one command. `-Fetch` fetches the file a profile names in its
`# brilliance-fetch: <owner/repo> <revision> <file>` line, then serves it, with no edit to the profile:

```powershell
.\serve.ps1 qwen27-preview -Fetch -Bus 3
.\serve.ps1 qwen27-preview -Fetch -Bus 3 -WhatIf   # the plan only: nothing is fetched or started
```

- Before a byte moves it prints the repository, the revision and the commit it resolves to, the destination
  and each file's size in bytes.
- The destination is `<dir>\<owner>\<repo>`. `<dir>` is `-ModelDir` if given, else `BRILLIANCE_HF_CACHE`,
  else `%LOCALAPPDATA%\radiance\hf`: the same store as a repository named as `model:` (below).
- A whole copy is kept with no transfer. A cut transfer is tried again from where it stopped, up to six
  times, and resumes when the command runs again. Each file is checked against its SHA-256 before it is
  kept, and the verified file is what the server is given.
- With no network, a copy a complete fetch left is served, and the output says so.
- The profile's `model:` line is not read and the file is not changed. `-Model <path>` does the same for a
  file you fetched yourself. In a profile of several models, `-ModelName` says which one.

`hfget.ps1` alone prints the same plan, and `-Plan` stops after it. Without a file it refuses a repository of
several containers, listing each with its size, before anything is downloaded.

**Or name a repository** as `model: owner/name`. When the model loads, the server fetches the whole
repository with `hfget.ps1` into `<dir>\<owner>\<name>`. `<dir>` is `download-dir:` under `server:` if set,
else `BRILLIANCE_HF_CACHE`, else `XDG_CACHE_HOME\radiance\hf`, else `%LOCALAPPDATA%\radiance\hf`.

- It serves the repository's one top-level `.rad`; with none, a directory of supported checkpoint files.
- A repository with several containers, such as `qwen3.8-27b-iqr-v4`, is refused before anything is
  downloaded, naming each with its size. Fetch the one file instead, as above.
- At each load it checks the repository for changes. If the check changes nothing on disk (the repository
  cannot be listed, say) and a complete copy was fetched before, that copy is served; any other failed
  transfer stops the load.
- Without `model-revision:` it follows the repository's `main` branch. Name a tag or a commit so a restart
  does not bring weights nobody reviewed.

#### A small first run: MiniCPM5-2B

MiniCPM5-2B IQ8R answers text only, loads in about 4 s and holds about 9 GiB of the card. Its profile has
thinking off, so the answer arrives in `content` at once. In the PowerShell window from
[step 2](#2-get-the-package):

1. **Fetch it and start it** on the card's PCI bus ([step 4](#4-start-a-server) shows how to find it):

   ```powershell
   .\serve.ps1 minicpm5-2b-1card -Fetch -Bus 3
   ```

   `-Fetch` gets `minicpm5-2b-iq8r.rad` (3.04 GiB, 3,267,162,112 bytes, SHA-256
   `6048448b1ffbb29df9ed44d5277cc00fc71db686959cc67f678c203e4d1d80ac`) from
   [`StillDeadcode/minicpm5-2b-iq8r`](https://huggingface.co/StillDeadcode/minicpm5-2b-iq8r), as
   [Fetching with serve.ps1](#fetching-with-serveps1) describes, then serves it. No edit to the profile. Without
   a card, use `-Cpu` in place of `-Bus 3` ([Serving on the CPU](#serving-on-the-cpu)). It is ready when it
   prints `server: listening on http://127.0.0.1:8000`.
2. **Next time**, the same command transfers nothing: the copy is whole. A copy you already have is served
   with `-Model <its path>` in place of `-Fetch`.
3. **Ask it something**, from a second PowerShell window:

   ```powershell
   $body = '{"model":"minicpm5-2b","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
   Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/completions -ContentType application/json -Body $body |
     ForEach-Object { $_.choices[0].message.content }
   ```

   Or **in cmd** (not PowerShell, which strips the `\"` quotes and gets a 400), with the `curl.exe` Windows
   ships:

   ```
   curl.exe http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"minicpm5-2b\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one sentence.\"}],\"max_tokens\":64}"
   ```

4. **Stop it** with Ctrl-C in its window ([step 6](#6-stop-it)).

To let it reason first, a request sends `"chat_template_kwargs": {"enable_thinking": true}`. For tools,
pictures and video, use `qwen27-preview`.

### 4. Start a server

In the PowerShell window from [step 2](#2-get-the-package), in the unzipped folder.

**Find the card's PCI bus number.** In Device Manager, expand "Display adapters", double-click the Radeon
AI PRO R9700 and read "Location" on the General tab: `PCI bus 3, device 0, function 0` is bus 3. Or list
every display adapter at once:

```powershell
Get-PnpDevice -Class Display -PresentOnly | ForEach-Object { '{0}: {1}' -f $_.FriendlyName, (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_LocationInfo).Data }
```

`-Bus` takes these decimal numbers. Some tools, and the server's own log, write the bus in hexadecimal
(`0000:20:00.0` is bus 32); `-Bus` takes that form too, or `0x20`. Put your bus number where the commands
below say `3`. A machine with one discrete card can leave `-Bus` out.

**Check the profile, then start the server in this window:**

```powershell
.\serve.ps1 qwen27-preview -Check     # read the profile and exit; success ends "model qwen3.8-27b = <path>, tp 1"
.\serve.ps1 qwen27-preview -Bus 3     # serve on the card at PCI bus 3
```

- Lines starting `I ` are the server's log, written to standard error; they are not errors. `W ` is a
  warning, `E ` an error.
- It is ready when it prints `server: listening on http://127.0.0.1:8000`, once the model has loaded. Leave
  this window open and send requests from a second PowerShell window ([step 5](#5-send-a-first-request)).
- **`-Bus`** picks cards by PCI bus number, one or several (`-Bus 3,9`). `-Device` takes HIP indices
  instead. With neither, the server sees the cards `$env:HIP_VISIBLE_DEVICES` names if this window set it,
  and otherwise every discrete card, with an integrated GPU hidden from HIP: the engine runs no model on
  one beside a discrete card, and one left visible would divide the pinned-memory limit and turn HIP
  virtual memory off. A machine with one discrete card needs no `-Bus`, integrated GPU or not.
- **Every start lists the cards the server will see**, as its card 0, 1, ..., each with its PCI bus and
  HIP index, and what is hidden (`serve: card 0 = PCI bus 3 (0x03), HIP device 2`). A profile's `cards:`
  and `memory.card-headroom-mib` count these cards from 0.
- **The dashboard** is at `http://127.0.0.1:8000/`: models, memory, performance, requests and sessions.
- **The terms of use** are at `http://127.0.0.1:8000/terms`, linked from the dashboard
  ([step 7](#7-serve-the-local-network)), and the models' full licence texts at `/terms/licenses`.
- **`-WhatIf`** prints what a start would set (the cards, the environment) and the command line, and the
  host, port and key the server reads from the profile with them; it starts nothing.
- **The environment:** a profile can set runtime variables in comment lines, `# brilliance-env: NAME=VALUE`,
  and `-Env NAME=VALUE[,NAME=VALUE]` adds or overrides them for one start. Only names starting `GPU_`,
  `HIP_`, `AMD_`, `BRILLIANCE_` and `RADIANCE_` are taken. `HIP_VISIBLE_DEVICES` is refused (choose cards
  with `-Bus` or `-Device`), and so is any name containing `KEY`, `TOKEN`, `SECRET` or `PASSWORD`. They are
  set for the server only, and the startup output names each one.
- `bin\radiance.exe --config profiles\qwen27-preview.yaml` works too. With an integrated GPU or several cards,
  set `$env:HIP_VISIBLE_DEVICES` to the card's HIP index first. It does not read `# brilliance-env:` lines:
  set those variables yourself.

**Or start it in the background**, with no window of its own, and keep using this one:

```powershell
$server = .\serve.ps1 qwen27-preview -Bus 3 -Background                  # the server's process ID
Select-String -Path .\radiance-qwen27-preview.log -Pattern 'listening on'   # prints the line once it is ready
```

- The log goes to `radiance-<profile>.log` in the current folder. `-Log` names another: a file name
  (`-Log srv.log`, in the current folder) or a full path (`-Log "$env:USERPROFILE\qwen27.log"`). It starts
  with `serve.ps1`'s own lines (the cards, any `-Listen`, `-Port` or `-ApiKey`, and the file a reload
  rereads), then the server's.
- `Get-Content .\radiance-qwen27-preview.log -Tail 20` shows the end of the log; a start that fails says why
  there.
- Stop it with `tools\stop-server.ps1` ([step 6](#6-stop-it)), never from Task Manager.

| Profile | Model | Cards | Status | Context | Requests at once | Speculation | KV cache |
|---|---|---|---|---|---|---|---|
| `minicpm5-2b-1card` | MiniCPM5-2B IQ8R (text) | 1 | **Runs, not yet qualified**: the small first run | 32K | 4 | DSpark, depth 7 | bf16 |
| `qwen27-preview` | Qwen3.8-27B IQ4R-L-V4 | 1 | **Qualified** | 32K | 4 | DFlash2, depth 7 | bf16 |
| `qwen27-8k-plain` | Qwen3.8-27B IQ4R-L-V4 | 1 | **Qualified** | 8K | 1 | off | bf16 |
| `qwen27-iq8r-1card` | Qwen3.8-27B IQ8R-V4 | 1 | **Soak-clean** | 16K (about 55.6K tokens of KV cache in all) | 4 | DFlash2 | int8 |
| `qwen27-iq8r-2card` | Qwen3.8-27B IQ8R-V4 | 2 (TP2) | **Soak-clean** | 32K | 4 | DFlash2 | bf16 |
| `flashnext-4card` | Qwen3.8-Flash-Next IQR-V4 | 4 (TP4), every expert on the cards | **Supported**: soaked at 32K on its earlier settings (2026-11-17); these tuned settings pass the injection, stop and serving checks; a soak on them has not run yet | 256K | 8 | MTP, depth 3 | bf16 |
| `flashnext-2card` | Qwen3.8-Flash-Next IQR-V4 | 2 (TP2), experts tiered to host memory | **Supported**, recommended safe profile | 256K | 8 | MTP, adaptive depth to 5 | int8 |
| `flashnext-2card-fast` | Qwen3.8-Flash-Next IQR-V4 | 2 (TP2), experts tiered to host memory | **Runs** (functional smoke), fastest option | 256K | 8 | MTP, adaptive depth to 5 | int8 |
| `gemma4-31b-1card` | Gemma 4 31B a4 | 1 | **Qualified** | 32K | 2 | MTP, depth 3 | bf16 |
| `gemma4-12b-1card` | Gemma 4 12B i8g | 1 | **Qualified** | 32K | 4 | MTP, depth 3 | bf16 |
| `qwen36-35b-2card` | Qwen3.6-35B-A3B IQ8R | 2 (TP2), every expert on the cards | **Qualified** | 32K | 4 | MTP, depth 3 | int8 |
| `qwen-image-1card` | Qwen-Image 2.1 IQ8R (pictures) | 1 | **Qualified** for generation; edits: known issue 13 | Pictures to 2048x2048, the default | - | off | - |
| `ltx-2.5-1card` | LTX-2.5 distilled IQ8R (video with sound) | 1 | **Runs, not yet qualified** | Videos to 6.7 s, 960x544 by default | 1 | off | - |
| `deepseek41-4card` | DeepSeek-V4.1-Flash-1017 (bakon3's uncensored container; text) | 4 (TP4), every routed expert on the cards | **Runs, not yet qualified** (preview) | 4K | 1 | off | bf16 |
| `deepseek41-2card` | DeepSeek-V4.1-Flash-1017 (bakon3's uncensored container; text) | 2 (TP2), experts that do not fit held in host memory | **Runs, not yet qualified** (preview); greedy text can differ from four cards' ([known issue 14](win/SUPPORT.md#known-issues)) | 4K | 1 | off | bf16 |

- **Context and requests at once are limits, not a promise that all of them fit at full length.** Context is
  the longest one request may be; requests at once is how many the scheduler runs together. All of them
  share one KV cache, which can hold fewer tokens than the two multiplied. On the test machine: about 131K
  tokens for `qwen27-preview` and `minicpm5-2b-1card` (4 x 32K fit), about 55.6K for `qwen27-iq8r-1card`, about 262K for
  `flashnext-2card` and about 919K for `flashnext-4card`. The startup log gives the figure on your machine
  (`kv cache ... tokens`).
- **Name the cards a profile needs** with `-Bus` (or `-Device`): `-Bus 6` for one card, `-Bus 3,9` for two.
  With neither, the server sees `$env:HIP_VISIBLE_DEVICES`'s cards if it is set, else every discrete card
  (an integrated GPU hidden), and the profile takes the first one, two or four in HIP's order. A card left
  visible but unused still costs: HIP virtual memory is on only with exactly one device visible, and the pin
  limit is divided by every device the process sees. Give `-Bus` for `flashnext-4card` too, unless the
  machine has exactly four discrete cards and no `HIP_VISIBLE_DEVICES` set.
- **`flashnext-4card` has two known issues.** When requests share a step, rounding can change Flash-Next's
  next-token distributions, so sampling over requests sent together is looser than the temperature asked
  ([known issue 8](win/SUPPORT.md#known-issues)). Compare repeats of one request from the same cache state.
  And it hard-locked the test machine once, when stopped during a long prefill beside a large build: let a
  prefill finish before stopping, or keep the CPU light ([known issue 12](win/SUPPORT.md#known-issues)).
- **Safe `flashnext-2card`** is the recommended two-card profile. It uses the exact wire with copy
  engines (SDMA) off by default. The server re-sends a lost host-memory write at about 4% of
  single-stream decode. [Known issue 1](win/SUPPORT.md#known-issues)
- **Fastest `flashnext-2card-fast`** uses lossy wht6 and turns SDMA on at launch with
  `.\serve.ps1 flashnext-2card-fast -Bus 3,9`. On two R9700s, wht6 gave about 10% faster
  prefill for 32K–128K prompts. It can change greedy text, with a small measured effect on next-token
  probabilities. SDMA prefill was unchanged and decode gains were inconclusive.
  [Flash-Next profile evidence](win/SUPPORT.md#models-and-card-counts)
- **The Flash-Next profiles are tuned:** the model's full 256K context, prefill steps of up to 6,144
  tokens for a prompt alone in its step (`max-num-batched-tokens: 6144`, `prefill-span: 3`), eight
  linear-state snapshots and the picture and audio encoders in host memory. Each profile's comment explains
  its settings.
- **Leave `p2p:` at `auto`** (the default) in profiles of your own. Windows' driver gives these cards no
  peer-to-peer access, so `auto` exchanges between cards through pinned host memory, and `p2p: on`, which
  some upstream recipes set, refuses to start. The lossy compressed exchange `tp-wire: wht6`
  has the same two-card resend and stop protection. [Known issue 1](win/SUPPORT.md#known-issues)
- `kv-cache-dtype: i8` shrinks the cache: about 5.3 GiB instead of 9.4 GiB for bf16 at 32K x 4 on
  Qwen3.8-27B, about 44% less.

### 5. Send a first request

From a second PowerShell window, once the server prints `server: listening`. These use PowerShell's own
`Invoke-RestMethod` and need no execution-policy change. They are for `qwen27-preview`; for another profile,
use the model name `/v1/models` lists.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health                 # status ok, the model resident
(Invoke-RestMethod http://127.0.0.1:8000/v1/models).data.id    # the model's name in the API: qwen3.8-27b
$body = @{
  model = 'qwen3.8-27b'
  messages = @(@{ role = 'user'; content = 'Say hello in five words.' })
  enable_thinking = $false
  max_tokens = 64
} | ConvertTo-Json -Depth 10
(Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body).choices[0].message.content
```

- **Qwen3.8 thinks before it answers** unless a request says otherwise, and its reasoning counts against
  `max_tokens`. With a small limit the whole budget can go on reasoning: the answer comes back empty, with
  `finish_reason` `length`. `enable_thinking = $false` turns thinking off for the request; or give it more
  tokens.
- **Streamed:** `curl.exe`, not `curl`, which in Windows PowerShell is `Invoke-WebRequest` and cannot stream.
  Each `data:` line is one chunk, and the stream ends with `data: [DONE]`:

  ```powershell
  $body = @{ model = 'qwen3.8-27b'; messages = @(@{ role = 'user'; content = 'Count from one to ten.' }); stream = $true; enable_thinking = $false; max_tokens = 100 } | ConvertTo-Json -Depth 10
  $body | curl.exe -sN http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" --data-binary "@-"
  ```

- **A tool call, round trip:** the model asks for a tool, you run it, and you send the result back with the
  call's `id`. Here the "tool" is a fixed answer:

  ```powershell
  $tools = @(@{ type = 'function'; function = @{
    name = 'get_time'; description = 'The current time in a city'
    parameters = @{ type = 'object'; properties = @{ city = @{ type = 'string' } }; required = @('city') } } })
  $req = @{ model = 'qwen3.8-27b'; tools = $tools; enable_thinking = $false; max_tokens = 300
            messages = @(@{ role = 'user'; content = 'What time is it in Paris? Use get_time.' }) }
  $r = Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json' -Body ($req | ConvertTo-Json -Depth 10)
  $call = $r.choices[0].message.tool_calls[0]
  $call.function.name; $call.function.arguments                              # get_time {"city":"Paris"}
  $req.messages += $r.choices[0].message                                     # the assistant's tool call
  $req.messages += @{ role = 'tool'; tool_call_id = $call.id; content = '14:05' }   # the tool's result
  (Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json' -Body ($req | ConvertTo-Json -Depth 10)).choices[0].message.content
  ```

  Streamed, the call arrives in pieces: `delta.tool_calls` chunks carry the `id` and name first, then the
  arguments, and the stream's last chunk says `finish_reason` `tool_calls`.
- **Pictures and video clips** go in Chat Completions as `image_url` and `video_url` content parts (checked on
  Qwen3.8-27B). The Responses and Messages APIs take their own content formats.

#### Connect a client

| Setting | Value |
|---|---|
| OpenAI-style base URL (OpenAI SDKs, Open WebUI, OpenCode and most agents) | `http://127.0.0.1:8000/v1` |
| Anthropic-style base URL (the Anthropic SDK, and clients that add `/v1/messages` themselves) | `http://127.0.0.1:8000` |
| Model | the name `/v1/models` lists: `qwen3.8-27b` for the Qwen3.8-27B profiles |
| API key | none, unless the server was started with one ([step 7](#7-serve-the-local-network)); a client that insists on a key can be given any text, such as `none` |

The routes are `/v1/chat/completions`, `/v1/responses` and `/v1/messages`. A Messages client given the
OpenAI-style base sends to `/v1/v1/messages`, which answers 404.

**OpenCode** was checked with this `opencode.json` in the project folder, against `qwen27-preview` on one
card:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "brilliance/qwen3.8-27b",
  "provider": {
    "brilliance": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Brilliance",
      "options": { "baseURL": "http://127.0.0.1:8000/v1", "apiKey": "none" },
      "models": {
        "qwen3.8-27b": { "name": "Qwen3.8-27B IQ4R-L-V4", "tool_call": true, "reasoning": true,
                         "limit": { "context": 32768, "output": 4096 } }
      }
    }
  }
}
```

**Open WebUI** connects as an OpenAI API connection with the OpenAI-style base URL above. Other clients and
agents have not been checked; any that lets you set the base URL and model name may work.

### 6. Stop it

- **Ctrl-C** in its window: the server stops taking requests, lets those in flight finish within the drain
  timeout (30 s by default, `drain-timeout:` under `scheduler:`), releases the card and exits.
- **A background server**, or one in another window: `tools\stop-server.ps1` sends Ctrl-Break, the same drain,
  and waits for the exit:

  ```powershell
  .\tools\stop-server.ps1 -ProcessId $server            # "stop: <id> exited in 1.6 s"
  Get-Content .\radiance-qwen27-preview.log -Tail 1     # "radiance exit status 0"
  ```

- **Let a long prefill finish before stopping.** A stop waits for the GPU step in hand, and a stop during a
  long prefill once hard-locked the test machine ([known issue 12](win/SUPPORT.md#known-issues)).
- **Do not end it from Task Manager** while it is working: a GPU process killed mid-work has taken the driver
  down. A server that does not stop can be read with `bin\brl-stacks.exe` (`tools\STACKS.md`).
- **Reload the YAML** without a restart with `tools\reload-server.ps1 -ProcessId <pid>` (Windows' counterpart
  of `kill -HUP`) or `POST /admin/reload`. The server rereads the file it was started on: for `serve.ps1`,
  the profile itself, so edit it and reload (`reload-server.ps1` prints which file that is). `-Listen`,
  `-Port` and `-ApiKey` stay applied over it. Changed models reload; a change to `server:` or `cards:` is
  refused until a restart.

### 7. Serve the local network

The profiles listen on loopback with no key. `serve.ps1` changes that from the command line:

```powershell
$env:MY_KEY = Read-Host 'API key'                                 # set the key in this window first
.\serve.ps1 qwen27-preview -Listen 0.0.0.0 -ApiKey env:MY_KEY   # every IPv4 interface; requires the key MY_KEY holds
.\serve.ps1 qwen27-preview -Listen 0.0.0.0 -ApiKey off          # every IPv4 interface, no key
```

- `-ApiKey` takes `off` or `env:NAME`. A key itself is refused on the command line, where other processes can
  read it. Without `-ApiKey`, the profile's own key setting stands.
- With a key, every inference request must carry it, as `Authorization: Bearer <key>` or `X-Api-Key: <key>`.
  The health probes, the dashboard page and the terms pages stay readable without it.
- **The firewall rule is yours to add.** Allow the port for the private network only.
- **Adopt the terms of use before serving other people.** Some model licences (LTX-2.5's, SDXL's) require
  whoever serves the model to other people to pass its use restrictions on to them as enforceable terms. The
  server shows a terms-of-use page at `/terms`, open without the key. Out of the box it is
  [win/TERMS-OF-USE.md](win/TERMS-OF-USE.md) (`TERMS-OF-USE.md` in the package), a template that carries
  those restrictions. Copy it, make it yours (name yourself, remove the models you do not serve, add your own
  terms), and set `terms-file:` under `server:` in the profile to your copy's full path. A `.html` or `.htm`
  file is served as HTML with scripts blocked, anything else as UTF-8 plain text. The file is read at
  startup, so restart to change it; one that is missing, empty, not UTF-8 or over 1 MiB stops the start. The
  template is not legal advice.

### 8. Optional: run as a Windows service

`.\serve.ps1 <profile> -Install` registers the profile as a service, started at boot and stopped by the same
drain as Ctrl-C. **It is not advertised yet:** it passes its host tests, and its qualification in an elevated
shell is open. [SERVICE.md](win/SERVICE.md) has the steps.

- Unpack the package into a folder only administrators can change, such as `C:\Program Files\Brilliance`,
  first: the service runs `radiance.exe` from there as LocalSystem, and anyone who can write to that folder
  could run code as SYSTEM. `serve.ps1` warns when the package is elsewhere. The service's working folder is
  its YAML's, under `%ProgramData%\Brilliance`, so give paths in the profile in full.
- Run it from an elevated PowerShell. `-WhatIf` prints every step first and changes nothing.
- With `-ApiKey env:NAME`, the key goes into a file only SYSTEM and Administrators can read, never onto a
  command line or into the registry.
- The cards, `BRILLIANCE_HF_CACHE`, and the profile's `# brilliance-env:` and `-Env` variables go into the
  service's own environment.
- A failure nobody asked for restarts it once, after 2 minutes. A later failure restarts nothing until a day
  has passed with no failure, when Windows resets the count (`-NoRestart`: never restart).
- `-Uninstall` stops it with the drain and removes it.
- A service keeps running the release it was installed from: remove it and install it again from
  this package, as [SERVICE.md](win/SERVICE.md) "Moving a service to a new release" describes (not yet
  run in an elevated shell).

### Serving on the CPU

Add `-Cpu` to `serve.ps1`, from the GPU package or the CPU package:

```powershell
.\serve.ps1 minicpm5-2b-1card -Cpu
```

`-Cpu` starts the server on the CPU with no GPU at all, and changes nothing in the profile. It applies three
settings over the profile, which the server's startup report names:

- the host backend only: the HIP runtime is never asked for a device, and every card is hidden from it;
- the CPU kernels, `libavx,libref` (the package's `gfx1201` kernels cannot run on the CPU);
- no speculative decoding for any model (the CPU kernels cannot run every drafter; MiniCPM5-2B's cannot).

`-Cpu` takes neither `-Bus` nor `-Device`. A profile of several models keeps only the one-card ones. Without
`-Cpu`, a start that would leave the server no card (HIP shows none, or `HIP_VISIBLE_DEVICES` hides them
all) is refused, so a card that has failed never turns into a CPU server unasked. Expect a fraction of a
card's speed. This was checked with MiniCPM5-2B IQ8R and EmbeddingGemma 2, and is not qualified
([known issues 4 and 10](win/SUPPORT.md#known-issues)). The CPU-only package (`-win64-cpu`) can be
[built from source](#building-from-source).

### Getting help

- **Bugs and questions about the Windows build:** open an issue at
  [github.com/adamczhang/Brilliance/issues](https://github.com/adamczhang/Brilliance/issues). Check
  [win/SUPPORT.md](win/SUPPORT.md)'s known issues first.
- **Include:** the release (`0.3.0-alpha.5`, or the `MANIFEST.json` commit), the Adrenalin driver version, the
  cards and the `-Bus` you gave, the profile and anything you changed in it, the exact command (with any key
  removed), and the server's log from its start to the failure.
- **A server that stopped making progress:** read it with `bin\brl-stacks.exe` before stopping it
  (`tools\STACKS.md`), and attach what it prints.
- **Upstream Radiance** ([codeberg.org/StillDeadcode/radiance](https://codeberg.org/StillDeadcode/radiance))
  is for problems that also happen on its Linux build. Report a Windows problem here first.

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

- **The AMD Adrenalin driver** 32.0.32015.12012, the version it was tested on; later ones have not been
  checked.
- **Serving needs no Python.** Python 3 is needed for `rad-export` (`bin\rad-export.cmd`, turning a finetune
  into a container), with torch and safetensors to fold in a LoRA adapter, and for `resolve-stacks.py`,
  which names a hung server's frames (`tools\STACKS.md`). The source tree's test and qualification tools
  (`win\tools\*.py`) use Python too.
- **`rad-export` runs the package's own converter** (`bin\rad-convert.exe`, beside it) and lists its
  recipes (`share\radiance\recipes`). From the unzipped folder:

  ```powershell
  .\bin\rad-export.cmd --list-recipes
  ```

  Export with the finetune's folder and `-o OUTPUT.rad`. `--rad-convert PATH` chooses another converter.
  Its exit status is rad-export's, so a script can tell a failed export.
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
| Qwen3.8-Flash-Next IQR-V4 ([`StillDeadcode/qwen3.8-next-flash-iqr-v4`](https://huggingface.co/StillDeadcode/qwen3.8-next-flash-iqr-v4), 113 GiB) | Not run (upstream lists a one-card layout with experts in host memory) | **Supported**: with production settings a lost write is re-sent, at about 4% of single-stream decode. An 8-hour soak has not been run on it yet. Profile: `flashnext-2card` | **Supported**: every expert on the cards (`placement: all_vram`); soaked at 32K x 8, int8 KV, MTP on its earlier settings (2026-11-17), before the four-card re-send and stop were added; with them the profile's tuned 256K x 8, bf16 KV settings pass the injection, stop and serving checks; a soak on them has not run yet. Profile: `flashnext-4card` |
| MiniCPM5-2B IQ8R ([`StillDeadcode/minicpm5-2b-iq8r`](https://huggingface.co/StillDeadcode/minicpm5-2b-iq8r), 3.04 GiB) | **Runs, not yet qualified**: chat and streaming, 32K x 4, bf16 KV, DSpark; the small first run. Also on the CPU kernels, without its drafter: `.\serve.ps1 minicpm5-2b-1card -Cpu` turns it off. Profile: `minicpm5-2b-1card` | - | - |
| EmbeddingGemma 2 bf16 (`google/embeddinggemma-2`, converted with `rad-convert`, 1.40 GiB) | **Qualified**: `/v1/embeddings` for text, pictures, video and audio; also on the CPU kernels | - | - |
| Qwen3-ASR 1.7B bf16 (`Qwen/Qwen3-ASR-1.7B-hf` as it ships, 4.09 GB) | **Qualified**: `/v1/audio/transcriptions` (5 formats, 4 response forms) and `/v1/realtime`, 16 sequences | - | - |
| Nemotron 3.5 Rerank 8B BF16 (`nvidia/Nemotron-3.5-Rerank-8B-BF16` as it ships, 16.78 GB) | **Qualified**: `/rerank` and `/v1/rerank`, text and picture documents, 32,768 tokens a pair | - | - |
| SDXL base 1.0 bf16 (`stabilityai/stable-diffusion-xl-base-1.0`, the single-file checkpoint converted with `rad-convert`, 6.39 GiB) | **Qualified**: `/v1/images/generations`; PNG, JPEG and WebP; about 5.4 s at 1024x1024, 28 steps | - | - |
| LTX-2.5 distilled IQ8R ([`StillDeadcode/radiance-ltx-2.5-distilled-iq8r`](https://huggingface.co/StillDeadcode/radiance-ltx-2.5-distilled-iq8r), 36.86 GiB; gated, LTX-2.x Community License) | **Runs, not yet qualified**: `/v1/videos`, text and image to video, MP4 (H.264 and AAC). Its text encoder is pinned in host memory (about 15.9 GiB pinned in all). Profile: `ltx-2.5-1card` | - | - |
| Gemma 4 31B a4 ([`keptn/gemma4-31b-a4`](https://huggingface.co/keptn/gemma4-31b-a4), 19.95 GiB) | **Qualified**: chat, Responses and Anthropic Messages, whole and streamed; tools; pictures and video; 32K x 2, bf16 KV, MTP. Audio is refused with a 501 | Not run | Not run |
| Gemma 4 12B i8g ([`keptn/gemma4-12b-i8g`](https://huggingface.co/keptn/gemma4-12b-i8g), 13.79 GiB) | **Qualified**: chat, Responses and Anthropic Messages, whole and streamed; tools; pictures, audio and video; 32K x 4, bf16 KV, MTP | - | - |
| Qwen3.6-35B-A3B IQ8R ([`StillDeadcode/qwen3.6-35b-a3b-iq8r`](https://huggingface.co/StillDeadcode/qwen3.6-35b-a3b-iq8r), 35.04 GiB) | - | **Qualified**: chat, Responses and Anthropic Messages, whole and streamed; tools; pictures and video; 32K x 4, int8 KV, MTP; every expert on the cards. It thinks before answering by default. Audio is refused with a 501 | Not run |
| Qwen-Image 2.1 IQ8R ([`StillDeadcode/radiance-qwen-image-2.1-iq8r`](https://huggingface.co/StillDeadcode/radiance-qwen-image-2.1-iq8r), 16.13 GiB; non-commercial licence) | **Qualified** for generation: `/v1/images/generations`, PNG, JPEG and WebP; a seed repeats its picture byte for byte; sizes from 512x512 to the default 2048x2048. Edits work for most seeds ([known issue 13](win/SUPPORT.md#known-issues)) | Not run | - |
| DeepSeek-V4.1-Flash-1017, bakon3's uncensored container ([`bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad`](https://huggingface.co/bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad), 251.13 GiB; MIT) | - | **Runs, not yet qualified** (preview): experts that do not fit the cards held in host memory (about 61 GiB less available while it serves); its greedy text can differ from four cards' ([known issue 14](win/SUPPORT.md#known-issues)). Profile: `deepseek41-2card` | **Runs, not yet qualified** (preview): text, chat with thinking on and off; 4K x 1, bf16 KV, no drafting; every routed expert on the cards, the 103 GiB Engram table read from the file. Send `temperature` with each request. Profile: `deepseek41-4card` |
| Other upstream models | Not run on Windows. The package carries many of upstream's architecture plugins, but no other model has been checked. Speech-synthesis and music models cannot be served by this build | | |

**The profiles** cover the Qwen3.8 layouts above that are **Qualified** or **Soak-clean**, two that are
**Supported**, four more **Qualified** models, and four that run, not yet qualified ([step 4](#4-start-a-server) has their settings):

- MiniCPM5-2B IQ8R, 1 card, **Runs, not yet qualified**: `minicpm5-2b-1card` (32K x 4), the small first run.
- Qwen3.8-27B IQ4R-L-V4, 1 card, **Qualified**: `qwen27-preview` (32K x 4) and `qwen27-8k-plain` (8K, one request).
- Qwen3.8-27B IQ8R-V4, 1 card, **Soak-clean**: `qwen27-iq8r-1card`.
- Qwen3.8-27B IQ8R-V4, 2 cards, **Soak-clean**: `qwen27-iq8r-2card`.
- Qwen3.8-Flash-Next IQR-V4, 4 cards, **Supported**: `flashnext-4card`, soaked at 32K on its earlier settings
  (2026-11-17), before the four-card re-send and stop were added; with them its tuned 256K settings pass the
  injection, stop and serving checks. A soak on them has not run yet.
  Its known issues: when requests share a step, rounding can change the output, so sampling is looser than
  asked; and one hard lock, seen once, when it was stopped during a long prefill beside a large build.
- Qwen3.8-Flash-Next IQR-V4, 2 cards, **Supported**: `flashnext-2card`, upstream's two-card settings, tuned,
  on the exact wire; a lost write is re-sent. An 8-hour soak has not been run on it yet. A faster prefill
  for long prompts is opt-in (`-Env BRILLIANCE_TP2_PIPELINE=1`), exact on the exact wire.
- Gemma 4 31B a4, 1 card, **Qualified**: `gemma4-31b-1card` (32K x 2).
- Gemma 4 12B i8g, 1 card, **Qualified**: `gemma4-12b-1card` (32K x 4).
- Qwen3.6-35B-A3B IQ8R, 2 cards, **Qualified**: `qwen36-35b-2card` (32K x 4). It thinks
  before answering by default, and a client's token limit counts the reasoning too.
- Qwen-Image 2.1 IQ8R, 1 card, **Qualified** for generation: `qwen-image-1card`. Some of its edits
  come out with colour noise ([known issue 13](win/SUPPORT.md#known-issues)), and its licence allows
  non-commercial use only ([Licence and credits](#licence-and-credits)).
- LTX-2.5 distilled IQ8R, 1 card, **Runs, not yet qualified**: `ltx-2.5-1card`, video with sound from text
  or a first frame. Its text encoder and token embedding are pinned in host memory (about 15.9 GiB), and
  its licence carries use restrictions you must pass on ([Licence and credits](#licence-and-credits)).
- DeepSeek-V4.1-Flash-1017, 4 cards, **Runs, not yet qualified** (preview): `deepseek41-4card` (4K, one
  request), every routed expert on the cards. Rank 0 follows the runtime's device order.
- DeepSeek-V4.1-Flash-1017, 2 cards, **Runs, not yet qualified** (preview): `deepseek41-2card`, the experts
  that do not fit the cards held in host memory. Its greedy text can differ from the four-card profile's
  ([known issue 14](win/SUPPORT.md#known-issues)).

**Brilliance 0.2.2's containers do not load.** Radiance 2.2 does not read the 0.2.2-era formats: whole-model
FP8, EXL3, the old Flash-Next FP8/IQ4R, GLM-5.3-Flash MXFP4, Brilliance 0.2.2's DeepSeek-V4.1 containers and
Gemma 4 FP8. Use the current containers above.

## Feature parity

Radiance 2.2's features, and where each stands on Windows, from [win/SUPPORT.md](win/SUPPORT.md).

| Radiance 2.2 feature | On Windows | Note |
|---|---|---|
| Chat Completions, Responses, Messages; streaming | **Yes** | Qualified on Qwen3.8-27B |
| Tools, reasoning | **Yes** | Qualified on Qwen3.8-27B; Open WebUI and OpenCode tool round trips pass |
| Grammars and structured output (`response_format`); `/v1/completions`, `/completion`, `/infill` | **Yes** | Checked on Qwen3.8-27B IQ4R-L-V4, one card (2026-11-19), not qualified: json_object, json_schema, GBNF on the completion routes, strict tool schemas. GBNF on chat is refused on Qwen3.8 (use `response_format`); no regex constraint (upstream has none); `/infill` needs fill-in-the-middle tokens, and its text can end in `<\|file_sep\|>` |
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
| Model switching | **Yes** | One card visible: by parking. Several cards visible: by unloading and reloading; since 2026-11-18 a reload keeps its VRAM (PAL's resource cache off). Repeated switching at TP2 checked 2026-11-19 (Qwen3.8-Flash-Next and Qwen3.8-27B IQ8R, 40 alternating requests, no memory drift; queued requests and a pinned model's refusal as documented), not qualified. A fleet of several models on more than one card needs `gpu-memory-mib` on each model ([SUPPORT](win/SUPPORT.md), known issue 2) |
| Parking (HIP virtual memory) | **Partial** | On with exactly one device visible: an elastic KV cache, and parking with no host copies. Off with more than one (rocm-systems#13232) |
| Prefix cache, checkpoint snapshots, agent replay | **Yes** | |
| Speculation: DFlash2, MTP | **Yes** | |
| Drafting options (`draft-coupling`, `draft-verify`, `draft-depth: adaptive`, `draft-suffix`) | **Yes** | Checked on Qwen3.8-Flash-Next, two cards (2026-11-19), not qualified: greedy text identical to the uncoupled base. Coupling and the traversal verify also on Qwen3.8-27B's DFlash2. A model that cannot take an option is refused at start, naming why |
| Control vectors | **Yes** | Checked on Qwen3.8-Flash-Next with the shipped vector, two cards (2026-11-19), not qualified: the per-request field, the `MODEL:NAME` variant, `/admin/control-vector`. Only Flash-Next's architecture applies them |
| Tensor parallel, 2 cards | **Partial** | Soak-clean when the weights fit the cards; supported when experts are read from host memory on the exact wire (a lost write is re-sent; no 8-hour soak run yet). The Flash-Next fastest profile runs with the same protection on lossy wht6. Flash-Next's faster prefill is opt-in (`BRILLIANCE_TP2_PIPELINE=1`). DeepSeek-V4.1 runs, not yet qualified, with experts in host memory (known issue 14) |
| Tensor parallel, 4 cards | **Partial** | Flash-Next supported with every expert on the cards: soaked at 32K on its earlier settings, before the four-card re-send and stop; with them, a lost write is re-sent as on two cards, and its tuned settings pass the injection, stop and serving checks (a soak on them has not run yet). DeepSeek-V4.1 runs, not yet qualified, every routed expert on the cards; 3 and 8 cards not run |
| Peer-to-peer between cards | **Not on Windows** | The driver gives no peer access, so collectives go through pinned host memory with `p2p: auto` (the default); `p2p: on` refuses to start ([step 4](#4-start-a-server)) |
| Dashboard | **Yes** | Per card: load (GFX activity), board power and junction temperature, read from the AMD driver's ADL; a card whose driver has no ADL shows its temperature and says the rest is not available on Windows |
| Diagnose mode | **Yes** | The busy-host warning reads the CPUs' load over a quarter second |
| `rad-convert` | **Partial** | CPU only, by decision; `--gpus` is refused. A real model converted: SDXL, 9.6 s on the CPU |
| `rad-export` (finetune to container) | **Yes** | With Python 3 (`bin\rad-export.cmd`), which runs the package's converter and recipes ([What you need to run it](#what-you-need-to-run-it)) |
| Model fetch by repository id | **Yes** | `hfget.ps1`, with curl.exe and PowerShell |
| Running as a service | **Not yet** | `radiance --service`, installed by `serve.ps1 -Install`; host-tested, its elevated-shell qualification open ([SERVICE.md](win/SERVICE.md)) |
| `--ngram-placement ram=DIR` | **Partial** | Device tests pass; not run end to end |
| Plugin SDK, out-of-tree plugins | **Partial** | From a source build's install; not in the package |
| Linux `scripts/` (measurement, traces, Docker) | **Not on Windows** | `win/tools` has the Windows tools |
| Cards other than `gfx1201` | **Not on Windows** | The toolchain is pinned to `gfx1201` |
| Tensor parallel, 3 and 8 cards ([docs/TP3.md](docs/TP3.md); eight cards on unequal links, [GUIDE](docs/GUIDE.md) section 4.7) | **Not run on Windows** | The test machine has four cards. Flash-Next on three would need its experts tiered to host memory, a layout not run here |
| Streaming KV (`kv-resident`, [GUIDE](docs/GUIDE.md) section 4.4.1, [docs/KV-STREAMING.md](docs/KV-STREAMING.md)) | **Not run on Windows** | It needs a model with indexer-selected attention; none has been run here |
| LoRA adapters beside the base model ([GUIDE](docs/GUIDE.md) section 4.9.1) | **Not run on Windows** | No adapter has been served here. Folding an adapter into a container with `rad-export` is a separate step |
| Decisions (`/v1/systemone`, `/v1/decisions`; [GUIDE](docs/GUIDE.md) section 5.13) | **Not run on Windows** | No decision model has been run here |
| 3-D shapes (Hunyuan3D-2's `/generate`, `/send`, `/status/:uid`; [GUIDE](docs/GUIDE.md) section 5.17) | **Not run on Windows** | No shape model has been run here |

[RADIANCE-README.md](RADIANCE-README.md) and upstream's guide ([docs/GUIDE.md](docs/GUIDE.md)) are upstream's
own: their model lists and timings are upstream's, measured on Linux. A feature they describe that this table
does not list has not been run on Windows.

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
| [win/PREVIEW.md](win/PREVIEW.md) | the package's quick start, first requests, client settings and known limits |
| [win/SERVICE.md](win/SERVICE.md) | running the server as a Windows service, and its qualification steps |
| [win/TERMS-OF-USE.md](win/TERMS-OF-USE.md) | the terms-of-use template the server shows at `/terms`, to adopt before serving other people |
| [win/licenses/THIRD-PARTY.md](win/licenses/THIRD-PARTY.md) | every third-party DLL and library in the package, its licence and source |
| [tools/STACKS.md](win/tools/stacks/README.md) | reading a server that stopped making progress (`brl-stacks`) |
| [RADIANCE-README.md](RADIANCE-README.md), [docs/](docs/) | upstream Radiance's README and documentation, unchanged. They describe the Linux build: its install, service, plugin and stop commands do not apply here, and on Windows a stuck card neither turns `/health` to 503 nor ends the server (`stuck-exit`), so watch the log and use `bin\brl-stacks.exe`. Use [docs/GUIDE.md](docs/GUIDE.md) for the YAML keys and the HTTP API's request fields, within the Windows limits in [win/SUPPORT.md](win/SUPPORT.md); its `curl` examples are written for a Linux shell and the model name `m`: in Windows PowerShell, `curl` is not curl and the JSON's inner quotes are stripped on the way to `curl.exe`, so send the body as [step 5](#5-send-a-first-request) does, with your model's name |

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
    restrictions to pass on ([Serve the local network](#7-serve-the-local-network)).

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
