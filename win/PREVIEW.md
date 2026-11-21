# Brilliance 0.3.0-alpha.5 for Windows

A Windows build of [Radiance](https://codeberg.org/StillDeadcode/radiance) 2.2 on its native HIP
backend, for **Radeon AI PRO R9700** cards. This is an alpha preview of Brilliance 0.3.0: some
models are qualified on one card and one on two, more run soak-clean, supported or not yet
qualified (`SUPPORT.md` has the table), and the limits below apply. Expect rough edges.

## How you use it

- **Brilliance is an API server.** You start it on a model, then connect an app or agent to it: its
  OpenAI-style base URL is `http://<host>:<port>/v1` and its Anthropic-style base URL is
  `http://<host>:<port>` ("Connecting a client", below).
- **Its dashboard, at `http://<host>:<port>/` in a browser, is for watching it:** each model's
  Monitoring tab has its speed, requests, sessions and details. A model's **Sandbox** tab is for a
  quick check that the model works: Start and Stop load and unload it (from a browser on the
  server's own machine), and a plain chat box streams its replies. It keeps nothing. Start holds the
  model until Stop: a request for another model that needs its cards is refused meanwhile, rather
  than moving it. From another
  machine, a server without an API key shows every model's own pages and Sandbox (not the
  overview), and its chat works with a model that is loaded.
- **For everyday chat in a browser, use Open WebUI; for coding, OpenCode.** Both connect as in
  "Connecting a client", below.

## Before you start: your hardware

| Your hardware | What this package does |
|---|---|
| AMD Radeon AI PRO R9700 (gfx1201, 32 GB) | The card it is built for and tested on (`SUPPORT.md` has each model and card count). |
| AMD Radeon RX 9070 family (gfx1201) | The same GPU code target, so it can start on these cards, but **Not run**: no model has been checked on one. They have 16 GB or less, which the 17.42 GiB starter model does not fit on one card. |
| Other AMD GPUs (RDNA4 gfx1200, RDNA3 gfx11xx, Strix Halo gfx1151, older), NVIDIA or Intel | **Not supported** by this package: it carries GPU code for gfx1201 only. Upstream Radiance serves several of these AMD targets on Linux. |
| No usable GPU | `.\serve.ps1 <profile> -Cpu` serves on the CPU kernels, at a fraction of a card's speed, from this same package ("Serving without a card", under Known limits). Not qualified. |

You also need Windows 11 and the AMD Adrenalin driver 32.0.32015.12012, the version it was tested
on; later drivers have not been checked. Nothing else to install: the HIP runtime, FFmpeg and the
VC++ runtime are in `bin\`. Serving needs no Python.

## Two ways in

- **A small first run:** MiniCPM5-2B IQ8R with `minicpm5-2b-1card`, a 3.04 GiB text-only model that
  loads in seconds (next section). Good on a slow link or a small disk.
- **The full example:** Qwen3.8-27B IQ4R-L-V4 with `qwen27-preview`, the qualified model, 17.42 GiB,
  with tools, pictures and video ("Quick start", below).

Both run on one card, and both start the same way: steps 1, 4 and 5 of the quick start (the
execution policy, the card's bus, the check and the start) apply to each.

## A small first run

MiniCPM5-2B IQ8R answers text only, loads in about 4 s and holds about 9 GiB of the card. Its
profile has thinking off, so the answer arrives in `content` at once.

1. Open PowerShell in the unzipped folder and allow the package's scripts for the window, as in
   step 1 of the quick start.
2. **Fetch it and start it** on the card's PCI bus (step 4 of the quick start shows how to find it):

   ```powershell
   .\serve.ps1 minicpm5-2b-1card -Fetch -Bus 3
   ```

   `-Fetch` gets `minicpm5-2b-iq8r.rad` (3.04 GiB, 3,267,162,112 bytes, SHA-256
   `6048448b1ffbb29df9ed44d5277cc00fc71db686959cc67f678c203e4d1d80ac`) from the Hugging Face
   repository `StillDeadcode/minicpm5-2b-iq8r` and serves it, with no edit to the profile
   ("Fetching with serve.ps1", after the quick start). Without a card, use `-Cpu` in place of
   `-Bus 3`. It is ready when it prints `server: listening on http://127.0.0.1:8000`.
3. **Next time**, the same command transfers nothing. A copy you already have is served with
   `-Model <its path>` in place of `-Fetch`.
4. **Ask it something**, from a second PowerShell window:

   ```powershell
   $body = '{"model":"minicpm5-2b","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
   Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/completions -ContentType application/json -Body $body |
     ForEach-Object { $_.choices[0].message.content }
   ```

   Or **in cmd** (not PowerShell, which strips the `\"` quotes and gets a 400), with the
   `curl.exe` Windows ships:

   ```
   curl.exe http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"minicpm5-2b\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one sentence.\"}],\"max_tokens\":64}"
   ```

5. **Stop it** with Ctrl-C in its window (step 8 of the quick start).

To let it reason first, a request sends `"chat_template_kwargs": {"enable_thinking": true}`. For
tools, pictures and video, use `qwen27-preview`.

## Quick start

These steps serve Qwen3.8-27B IQ4R-L-V4, the qualified model, on one R9700 with the profile
`qwen27-preview`, whose model is named `qwen3.8-27b` in the API.

1. **Open PowerShell in the unzipped folder** (in File Explorer, type `powershell` in the folder's
   address bar and press Enter), and allow the package's scripts for this window only:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force   # this window only; nothing is changed for good
   ```

   Run it again in each new PowerShell window that runs the package's scripts. If it reports that a
   policy set by your organisation overrides it, the scripts cannot run until its administrator
   allows them; check the ZIP's published SHA-256 and follow your organisation's script policy.

2. **Download the model**, file `qwen3.8-27b-iq4r-l-v4.rad` (17.42 GiB, SHA-256
   `d57d1428218e20a4cd578c3e286f30a2e566e589c832bf5004f49e164cb48ae4`) from the Hugging Face
   repository `StillDeadcode/qwen3.8-27b-iqr-v4`. Every profile expects its model under
   `D:\models\hf\<owner>\<repo>\`; put it there and the profile needs no edit. The package's
   fetcher gets the one file, resumes a partial download and checks it against the SHA-256 the
   repository lists:

   ```powershell
   .\libexec\brilliance\hfget.ps1 StillDeadcode/qwen3.8-27b-iqr-v4 D:\models\hf\StillDeadcode\qwen3.8-27b-iqr-v4 main qwen3.8-27b-iq4r-l-v4.rad
   Get-FileHash D:\models\hf\StillDeadcode\qwen3.8-27b-iqr-v4\qwen3.8-27b-iq4r-l-v4.rad
   ```

   Do not name this repository as `model:` instead: it holds several containers, and the server
   would download all of them and then refuse it ("A model by repository id", below).

3. **Elsewhere, point the profile at it.** Any folder works, such as one on C:. Open
   `profiles\qwen27-preview.yaml` in a text editor and change the `model:` line to the file's full
   path, with forward slashes, in double quotes if it has spaces:

   ```yaml
       model: "C:/Users/me/My Models/qwen3.8-27b-iq4r-l-v4.rad"
   ```

   `Test-Path` on that path must print `True`: the profile check in step 5 reads the YAML only and
   does not open the model file.

4. **Find the card's PCI bus number.** In Device Manager, expand "Display adapters", double-click
   the Radeon AI PRO R9700 and read "Location" on the General tab: `PCI bus 3, device 0, function 0`
   is bus 3. Or list every display adapter:

   ```powershell
   Get-PnpDevice -Class Display -PresentOnly | ForEach-Object { '{0}: {1}' -f $_.FriendlyName, (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_LocationInfo).Data }
   ```

   `-Bus` takes these decimal numbers, or the hexadecimal form the server's own log writes
   (`0000:20:00.0`, or `0x20`, is bus 32). Put yours where the commands below say `3`. Give `-Bus`
   when the machine has more than one discrete card. Without `-Bus` or `-Device`, the server sees
   every discrete card (or those `$env:HIP_VISIBLE_DEVICES` names, if this window set it), and an
   integrated GPU is hidden from HIP: the engine runs no model on one beside a discrete card, and
   one left visible would divide the pinned-memory limit and turn HIP virtual memory off. Every
   start, and `-WhatIf`, lists the cards the server will see as its card 0, 1, ..., each with its
   PCI bus and HIP index; a profile's `cards:` and `memory.card-headroom-mib` count these.

5. **Check the profile, then start the server:**

   ```powershell
   .\serve.ps1 qwen27-preview -Check     # read the profile and exit; success ends "model qwen3.8-27b = <path>, tp 1"
   .\serve.ps1 qwen27-preview -Bus 3     # serve on the card at PCI bus 3, in this window
   ```

   Lines starting `I ` are the server's log on standard error, not errors (`W ` warns, `E ` is an
   error). It is ready when it prints `server: listening on http://127.0.0.1:8000`, once the model
   has loaded. Leave the window open and use a second one for requests.

   **Or in the background**, with no window, keeping this one:

   ```powershell
   $server = .\serve.ps1 qwen27-preview -Bus 3 -Background                  # the server's process ID
   Select-String -Path .\radiance-qwen27-preview.log -Pattern 'listening on'   # prints the line once it is ready
   ```

   The log is `radiance-<profile>.log` in the current folder; `Get-Content` it with `-Tail 20` to
   see why a start failed. `-Log` names another, a file name in the current folder (`-Log srv.log`)
   or a full path (`-Log "$env:USERPROFILE\qwen27.log"`). The log starts with `serve.ps1`'s own
   lines: the cards, any `-Listen`, `-Port` or `-ApiKey`, and the file a reload rereads.

6. **Send a first request** (see "First requests" below):

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8000/health                 # status ok, the model resident
   (Invoke-RestMethod http://127.0.0.1:8000/v1/models).data.id    # qwen3.8-27b
   ```

7. The **dashboard** is at `http://127.0.0.1:8000/`: models, memory, performance, requests and
   sessions. The **terms of use** are at `http://127.0.0.1:8000/terms`, linked from the dashboard.

8. **Stop it.** In its window, press **Ctrl-C**. The server stops taking requests, lets those in
   flight finish within the drain timeout (30 s by default, `drain-timeout:` under `scheduler:`),
   releases the card and exits. A background server, or one in another window, is stopped with
   Ctrl-Break, the same drain, by:

   ```powershell
   .\tools\stop-server.ps1 -ProcessId $server            # "stop: <id> exited in 1.6 s"
   Get-Content .\radiance-qwen27-preview.log -Tail 1     # "radiance exit status 0"
   ```

   - Let a long prefill finish before stopping: a stop during one once hard-locked the test
     machine (`SUPPORT.md`, known issue 12).
   - Do not end it from Task Manager while it is working: a GPU process killed mid-work has taken
     the driver down. A server that does not stop can be read with `bin\brl-stacks.exe`
     (`tools\STACKS.md`).

`bin\radiance.exe --config profiles\qwen27-preview.yaml` serves without the launcher. On a machine
with an integrated GPU or several cards, set `$env:HIP_VISIBLE_DEVICES` to the card's HIP index
first; it does not read `# brilliance-env:` lines. With `--check` it reads the profile and exits.

**Fetching with serve.ps1.** Steps 2 and 3 can be one command: `-Fetch` fetches the file a
profile names in its `# brilliance-fetch: <owner/repo> <revision> <file>` line, then serves it.
The profile is not edited, and its `model:` line is not read.

```powershell
.\serve.ps1 qwen27-preview -Fetch -Bus 3 -WhatIf   # the plan only: nothing is fetched or started
.\serve.ps1 qwen27-preview -Fetch -Bus 3
```

- Before a byte moves it prints the repository, the revision and the commit it resolves to, the
  destination and each file's size in bytes.
- The destination is `<dir>\<owner>\<repo>`. `<dir>` is `-ModelDir` if given, else
  `BRILLIANCE_HF_CACHE`, else `%LOCALAPPDATA%\radiance\hf`.
- A whole copy is kept with no transfer. A cut transfer is tried again from where it stopped, up to
  six times, and resumes when the command runs again. Each file is checked against its SHA-256 before it is kept, and the server is given the verified file.
- With no network, a copy a complete fetch left is served, and the output says so.
- `-Model <path>` serves a file you fetched yourself the same way. In a profile of several models,
  `-ModelName` says which one.

## First requests

From a second PowerShell window, with the server listening. `Invoke-RestMethod` is PowerShell's own
and needs no execution-policy change. For a profile other than the Qwen3.8-27B ones, use the model
name `/v1/models` lists.

```powershell
$body = @{
  model = 'qwen3.8-27b'
  messages = @(@{ role = 'user'; content = 'Say hello in five words.' })
  enable_thinking = $false
  max_tokens = 64
} | ConvertTo-Json -Depth 10
(Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body).choices[0].message.content
```

- **Qwen3.8 thinks before it answers** unless a request says otherwise, and its reasoning counts
  against `max_tokens`. With a small limit the whole budget can go on reasoning, and the answer
  comes back empty with `finish_reason` `length`. `enable_thinking = $false` turns thinking off for
  the request; or give it more tokens.
- **Streamed:** use `curl.exe`, not `curl`, which in Windows PowerShell is `Invoke-WebRequest` and
  cannot stream. Each `data:` line is one chunk; the stream ends with `data: [DONE]`.

  ```powershell
  $body = @{ model = 'qwen3.8-27b'; messages = @(@{ role = 'user'; content = 'Count from one to ten.' }); stream = $true; enable_thinking = $false; max_tokens = 100 } | ConvertTo-Json -Depth 10
  $body | curl.exe -sN http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" --data-binary "@-"
  ```

- **A tool call, round trip:** the model asks for a tool, you run it, and you send the result back
  with the call's `id`. Here the "tool" is a fixed answer:

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

  Streamed, the call arrives in pieces: `delta.tool_calls` chunks carry the `id` and name first,
  then the arguments, and the last chunk says `finish_reason` `tool_calls`.
- **Pictures and video clips** go in Chat Completions as `image_url` and `video_url` content parts
  (checked on Qwen3.8-27B). The Responses and Messages APIs take their own content formats.

## Connecting a client

| Setting | Value |
|---|---|
| OpenAI-style base URL (OpenAI SDKs, Open WebUI, OpenCode and most agents) | `http://127.0.0.1:8000/v1` |
| Anthropic-style base URL (the Anthropic SDK, and clients that add `/v1/messages` themselves) | `http://127.0.0.1:8000` |
| Model | the name `/v1/models` lists: `qwen3.8-27b` for the Qwen3.8-27B profiles |
| API key | none, unless the server was started with one; a client that insists can be given any text, such as `none` |

The routes are `/v1/chat/completions`, `/v1/responses` and `/v1/messages`. A Messages client given
the OpenAI-style base sends to `/v1/v1/messages`, which answers 404. Streaming and tools were
checked on Qwen3.8-27B through all three.

**OpenCode** was checked with this `opencode.json` in the project folder, against `qwen27-preview`
on one card:

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

**Open WebUI** connects as an OpenAI API connection with the OpenAI-style base URL. Other clients
and agents have not been checked; one that lets you set the base URL and model name may work.

## Getting help

Report bugs and ask questions about this Windows build in the issue tracker of
github.com/adamczhang/Brilliance, after checking `SUPPORT.md`'s known issues. Include the release
(`0.3.0-alpha.5`, or the commit in `MANIFEST.json`), the Adrenalin driver version, the cards and
the `-Bus` you gave, the profile and anything you changed in it, the exact command (with any key
removed) and the server's log from its start to the failure. For a server that stopped making
progress, read it with `bin\brl-stacks.exe` first (`tools\STACKS.md`) and attach what it prints.
Upstream Radiance (codeberg.org/StillDeadcode/radiance) is for problems its Linux build has too.

## Serving on the network, as a service, and reloading

- **On the network:** set the key in the window first (`$env:MY_KEY = Read-Host 'API key'`), then
  `.\serve.ps1 qwen27-preview -Listen 0.0.0.0 -ApiKey env:MY_KEY` listens on every IPv4 interface
  and requires that key on every inference request, as `Authorization: Bearer <key>` or
  `X-Api-Key: <key>`. The health probes, the dashboard page and the terms pages stay readable
  without it. `-ApiKey off` serves with no key; without `-ApiKey` the profile's own setting stands.
  A firewall rule for the port is the administrator's to add.
- **As a Windows service** (not yet qualified: `SERVICE.md` has the remaining steps): unpack the
  package into a folder only administrators can change, such as `C:\Program Files\Brilliance`,
  since the service runs `radiance.exe` from there as LocalSystem, then run
  `.\serve.ps1 <profile> -Install` in an elevated PowerShell (`-WhatIf` shows every step first and
  changes nothing). The service runs a copy of the profile,
  `%ProgramData%\Brilliance\<Name>\<Name>.yaml`; to change a model later, edit that copy and
  reload. Its working folder is that copy's, so give paths in the profile in full.
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
  counterpart of `kill -HUP`, or `POST /admin/reload`. The server rereads the file it was started
  on: for `serve.ps1`, the profile itself, so edit it and reload; for a service, its copy under
  `%ProgramData%\Brilliance` (reload from an elevated shell). `reload-server.ps1` prints which file
  that is. `-Listen`, `-Port` and `-ApiKey` stay applied over it. Changed models reload, removed
  ones unload, new ones are added; a change to `server:` or `cards:` is refused until a restart.

## The profiles

| Profile | Model | Cards | Status | Context | Requests at once | Speculation | KV cache |
|---|---|---|---|---|---|---|---|
| `minicpm5-2b-1card.yaml` | MiniCPM5-2B IQ8R (text) | 1 | Runs, not yet qualified: the small first run | 32K | 4 | DSpark, depth 7 | bf16 |
| `qwen27-preview.yaml` | Qwen3.8-27B IQ4R-L-V4 | 1 | Qualified | 32K | 4 | DFlash2, depth 7 | bf16 |
| `qwen27-8k-plain.yaml` | Qwen3.8-27B IQ4R-L-V4 | 1 | Qualified | 8K | 1 | off | bf16 |
| `qwen27-iq8r-1card.yaml` | Qwen3.8-27B IQ8R-V4 | 1 | Soak-clean | 16K (about 55.6K tokens of KV cache in all) | 4 | DFlash2 | int8 |
| `qwen27-iq8r-2card.yaml` | Qwen3.8-27B IQ8R-V4 | 2 (TP2) | Soak-clean | 32K | 4 | DFlash2 | bf16 |
| `flashnext-4card.yaml` | Qwen3.8-Flash-Next IQR-V4 | 4 (TP4), every expert on the cards | Supported: soaked at 32K on its earlier settings (2026-11-17); these tuned settings pass the injection, stop and serving checks; a soak on them has not run yet | 256K | 8 | MTP, depth 3 | bf16 |
| `flashnext-2card.yaml` | Qwen3.8-Flash-Next IQR-V4 | 2 (TP2), experts tiered to host memory | Supported, recommended safe profile | 256K | 8 | MTP, adaptive depth to 5 | int8 |
| `flashnext-2card-fast.yaml` | Qwen3.8-Flash-Next IQR-V4 | 2 (TP2), experts tiered to host memory | Runs (functional smoke), fastest option | 256K | 8 | MTP, adaptive depth to 5 | int8 |
| `gemma4-31b-1card.yaml` | Gemma 4 31B a4 | 1 | Qualified | 32K | 2 | MTP, depth 3 | bf16 |
| `gemma4-12b-1card.yaml` | Gemma 4 12B i8g | 1 | Qualified | 32K | 4 | MTP, depth 3 | bf16 |
| `qwen36-35b-2card.yaml` | Qwen3.6-35B-A3B IQ8R | 2 (TP2), every expert on the cards | Qualified | 32K | 4 | MTP, depth 3 | int8 |
| `qwen-image-1card.yaml` | Qwen-Image 2.1 IQ8R (pictures) | 1 | Qualified for generation; edits: `SUPPORT.md`, known issue 13 | Pictures to 2048x2048, the default | - | off | - |
| `ltx-2.5-1card.yaml` | LTX-2.5 distilled IQ8R (video with sound) | 1 | Runs, not yet qualified | Videos to 6.7 s, 960x544 by default | 1 | off | - |
| `deepseek41-4card.yaml` | DeepSeek-V4.1-Flash-1017 (bakon3's uncensored container; text) | 4 (TP4), every routed expert on the cards | Runs, not yet qualified (preview) | 4K | 1 | off | bf16 |
| `deepseek41-2card.yaml` | DeepSeek-V4.1-Flash-1017 (bakon3's uncensored container; text) | 2 (TP2), experts that do not fit held in host memory | Runs, not yet qualified (preview); greedy text can differ from four cards' (`SUPPORT.md`, known issue 14) | 4K | 1 | off | bf16 |
| `sandbox.yaml` | Every profile above, one model each, none loaded at start | What `-Bus` names | For the dashboard's Sandbox tab only: pick a model there and Start loads it | As each profile | As each profile | As each profile | As each profile |

- **Context and requests at once are limits, not a promise that all of them fit at full length.**
  Context is the longest one request may be; requests at once is how many run together. All of
  them share one KV cache, which can hold fewer tokens than the two multiplied. On the test
  machine: about 131K tokens for `qwen27-preview` and `minicpm5-2b-1card` (4 x 32K fit), about 55.6K for
  `qwen27-iq8r-1card`, about 262K for `flashnext-2card` and about 919K for `flashnext-4card`. The
  startup log gives the figure on your machine (`kv cache ... tokens`).
- The IQ8R-V4 file is `qwen3.8-27b-iq8r-v4.rad` in `StillDeadcode/qwen3.8-27b-iqr-v4` (27.58 GiB); the
  Flash-Next file is `qwen3.8-next-flash-iqr-v4.rad` in `StillDeadcode/qwen3.8-next-flash-iqr-v4`
  (113.32 GiB). Each profile's comment has the file's SHA-256; edit its `model:` to where the file is.
- The four models qualified on 2026-11-20: `gemma4-31b-a4.rad` in `keptn/gemma4-31b-a4` (19.95 GiB),
  `gemma4-12b-i8g.rad` in `keptn/gemma4-12b-i8g` (13.79 GiB), `qwen3.6-35b-a3b-iq8r.rad` in
  `StillDeadcode/qwen3.6-35b-a3b-iq8r` (35.04 GiB, two cards: `-Bus 3,9`) and
  `qwen-image-2.1-iq8r.rad` in `StillDeadcode/radiance-qwen-image-2.1-iq8r` (16.13 GiB). Qwen3.6
  thinks before answering by default, and a client's token limit counts the reasoning too. Qwen-Image
  is qualified for `/v1/images/generations`, from 512x512 to its default 2048x2048 on one card; its
  edits (`/v1/images/edits`) work for most seeds, but some come out with colour noise: retry those
  with another seed (`SUPPORT.md`, known issue 13). Its licence allows research and evaluation only;
  commercial use needs a separate licence from Qwen ("Model licences", below).
- Name the cards with `-Bus`: `.\serve.ps1 qwen27-iq8r-1card -Bus 6`, `.\serve.ps1 qwen27-iq8r-2card -Bus 3,9`.
  With neither `-Bus` nor `-Device`, the server sees `$env:HIP_VISIBLE_DEVICES`'s cards if it is
  set, else every discrete card (an integrated GPU hidden), and the profile takes the first ones in
  HIP's order. An unused card left visible divides the pinned-memory limit and turns HIP virtual
  memory off. Give `-Bus` for `flashnext-4card` too, unless the machine has exactly four discrete
  cards and no `HIP_VISIBLE_DEVICES` set.
- `sandbox.yaml` runs every profile above in one server for the dashboard's Sandbox tab, none loaded
  until Start: `.\serve.ps1 sandbox -Bus 3,9`. The launcher drops, and names, each model that needs
  more cards than it names. Speeds and memory there are not the same as running that profile on its
  own: with several cards visible, HIP virtual memory is off and the pin limit is lower.
- `flashnext-4card` has two known issues. When requests share a step, rounding can change
  Flash-Next's next-token distributions, so sampling over requests sent together is looser than the
  temperature asked (`SUPPORT.md`, known issue 8). Compare repeats of one request from the same
  cache state: a long prompt sent again and resumed from the prefix cache changed its reasoning text
  in the two-card checks, with the same final answer. And it hard-locked the test machine once, when
  stopped during a long prefill beside a large build: let a prefill finish before stopping, or keep
  the CPU light (`SUPPORT.md`, known issue 12).
- **Safe `flashnext-2card`** is recommended. It uses the exact wire with copy engines (SDMA) off by
  default. The server re-sends a lost write (`SUPPORT.md`, known issue 1).
- **Fastest `flashnext-2card-fast`** uses lossy wht6 and turns SDMA on when started with
  `.\serve.ps1 flashnext-2card-fast -Bus 3,9`. On two R9700s, wht6 gave about 10% faster
  prefill for 32K–128K prompts, with a small measured effect on next-token probabilities. Greedy text
  can change. SDMA prefill was unchanged and decode gains were inconclusive. Sandbox applies the fast
  entry's wht6 setting and uses the server's shared SDMA setting.
- **A faster prefill for long prompts, opt-in:** add `-Env BRILLIANCE_TP2_PIPELINE=1` to either
  Flash-Next two-card start. It overlaps a long prompt's two-card exchanges with its computation;
  on the exact wire it gives the same greedy tokens as without it, and a lost write on its exchange
  is re-sent as on the others (`SUPPORT.md`, "Models and card counts" and known issue 1).
- **`deepseek41-4card`** serves DeepSeek-V4.1-Flash-1017 (bakon3's uncensored container,
  `deepseek-v4.1-flash-1017-uncensored.rad` in `bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad`,
  251.13 GiB, reassembled from the repository's six parts) on four cards, every routed expert on
  the cards; its 103 GiB Engram table is read from the file as each step needs it. A preview: text
  only, 4,096 tokens, one request at a time, no drafting; it runs and is not yet qualified. The
  checkpoint sets no sampling of its own, so send `temperature` with each request. Rank 0 follows
  the runtime's device order, not the order `-Bus` names the cards in.
- **`deepseek41-2card`** serves the same file on two cards (`-Bus 3,9`): each card holds what fits,
  then a 20 GiB pinned pool a card, and the rest of the experts (about 9 GiB a card) are read once
  at load into ordinary RAM. About 61 GiB less host memory is available while it serves, and the
  start refuses, before it allocates anything, when it would leave less than 24 GiB. Its greedy
  text can differ from `deepseek41-4card`'s, which is the reference (`SUPPORT.md`, known issue 14).
- The Flash-Next profiles are tuned: the model's full 256K context, prefill steps of up to 6,144
  tokens for a prompt alone in its step (`max-num-batched-tokens: 6144`, `prefill-span: 3`), eight
  linear-state snapshots and the picture and audio encoders in host memory. Each profile's comment
  explains its settings.
- A profile's `# brilliance-env: NAME=VALUE` lines, and `serve.ps1 -Env NAME=VALUE[,...]`, set
  variables whose names start `GPU_`, `HIP_`, `AMD_`, `BRILLIANCE_` or `RADIANCE_`, for the server
  only, and for `-Install` in the service's environment. `HIP_VISIBLE_DEVICES` is refused (choose
  cards with `-Bus` or `-Device`), and so is any name containing `KEY`, `TOKEN`, `SECRET` or
  `PASSWORD`. `.\serve.ps1 <profile> -WhatIf` prints what a start would set and run, and the host,
  port and key the server reads from the profile with them.
- `kv-cache-dtype: i8` shrinks the cache: about 5.3 GiB instead of 9.4 GiB for bf16 at 32K x 4 on
  Qwen3.8-27B, about 44% less.

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

- **Qualified on one card, and Qwen3.6-35B-A3B on two.** Qwen3.8-Flash-Next IQR-V4 runs on two cards and passes the same API
  checks, but with its production settings (experts in host memory, 8 sequences, speculation, prefix cache)
  a write to host memory is lost now and then (`SUPPORT.md`, known issue 1). The server re-sends
  it, at about 4% of single-stream decode, so Flash-Next on two cards is Supported; an 8-hour soak
  has not been run on it yet. A block that cannot be recovered stops the server, which exits with status 70.
  A server that stops making progress for any reason can be read with `bin\brl-stacks.exe`
  (`tools\STACKS.md`). Qwen3.8-27B IQ8R-V4 is soak-clean on one card and on two. On four cards,
  with every expert on the cards, Flash-Next is Supported: it was soaked at 32K on its earlier
  settings (2026-11-17), before four cards carried the same re-send and stop; with them its
  profile's tuned 256K settings pass the injection, stop and serving checks, and a soak on them has
  not run yet.
  `SUPPORT.md` has the table of models and card counts.
- **Cards:** the build carries GPU code for gfx1201 only. The R9700 is the card it is tested on; the
  RX 9070 family shares the target but has not been run ("Before you start: your hardware", above).
- **Generation endpoints:**
  - **Video generation** works on one card with LTX-2.5 distilled IQ8R (`ltx-2.5-1card.yaml`) and
    is not yet qualified. Its file is `ltx-2.5-distilled-iq8r.rad` in the gated repository
    `StillDeadcode/radiance-ltx-2.5-distilled-iq8r` (36.86 GiB). Its text encoder and token
    embedding are pinned in host memory, about 15.9 GiB, beside 30.3 GiB of the card.
    It writes H.264 only through AMD's (`h264_amf`), else Windows' (`h264_mf`) licensed encoders.
    `h264_amf` encodes on the first display adapter Windows lists: on a machine with an AMD
    integrated GPU, the integrated GPU, not the card that runs the model.
  - **Speech synthesis (TTS) is not supported** in 0.3.0: `/v1/audio/speech` answers 501 ("not
    supported in this build"). Music generation, on the same endpoint, is not served either.
  - **Image generation** works (`SUPPORT.md`).
- **No peer-to-peer access between cards on Windows.** Multi-card collectives pass through pinned
  host memory, and a process can pin about the shared-memory budget divided by the cards it sees. A
  table too large to pin is read from the file instead (`--ngram-placement disk`). Leave `p2p:` at
  `auto`, the default, in profiles of your own: `auto` exchanges through host memory, while
  `p2p: on`, which some upstream recipes set, refuses to start here. `tp-wire: wht6` is a separate
  setting, a lossy compressed exchange with the same two-card resend and stop protection as the
  exact wire.
- **HIP virtual memory is on with one card visible, and off with more.**
  - With exactly one device visible to the process (`-Bus` or `-Device` naming one card, or
    `HIP_VISIBLE_DEVICES` naming one) it is on, since 2026-11-17. The KV cache is elastic: it
    grows as sessions fill it. A model that makes room for another is parked: its weights come
    back from the container, and nothing is kept in host memory.
  - With more than one device visible (several cards, or a card and an integrated GPU, which
    `serve.ps1` hides unless told otherwise) it is off: the runtime commits a reservation's whole size at its first mapping
    (rocm-systems#13232). The KV cache then holds its whole size from the start, nothing is
    parked, and switching models unloads one and loads the other from its file.
  - The startup log says which, in a line that begins `device: HIP virtual memory is on` or
    `is off`. `BRILLIANCE_VMM=0` turns it off; `=1` forces it on and is not qualified with
    several devices.
- **The dashboard reads each card's load, power and temperature from the AMD driver's ADL**, the
  same firmware figures Linux reads. Windows' own counters do not see HIP's work, so where ADL is
  missing the page shows the temperature and says load and power are not available on Windows.
- **Drafting (DFlash2, MTP) and plain decoding can differ in greedy text** where two tokens are
  almost tied: the verify pass and the single-token pass round differently. Both are the model's
  answer; neither is a fault. On Flash-Next, plain decoding's own text also changes when other
  requests share its steps (SUPPORT.md, known issues 5 and 8).
- **A model by repository id** (`model: owner/name`): when the model loads, the server fetches the
  whole repository with `libexec\brilliance\hfget.ps1` into `<dir>\<owner>\<name>`. `<dir>` is
  `download-dir:` under `server:` if set, else `BRILLIANCE_HF_CACHE`, else
  `XDG_CACHE_HOME\radiance\hf`, else `%LOCALAPPDATA%\radiance\hf`. It serves the repository's one
  top-level `.rad`; with none, a directory of supported checkpoint files. A repository with several
  containers, such as `StillDeadcode/qwen3.8-27b-iqr-v4`, is refused before anything is
  downloaded, naming each with its size; fetch the one file instead (`serve.ps1 -Fetch`). Each load checks the repository
  for changes: if the check changes nothing on disk (the repository cannot be listed, say) and a
  complete copy was fetched before, that copy is served, and any other failed transfer stops the
  load. Set `model-revision:` to a tag or a commit to keep the weights you reviewed. `serve.ps1`
  copies no model: each is served from where its `model:` points.
- **`rad-convert` runs on the CPU only**: on Windows it never uses a GPU, and `--gpus` is refused.
  The recipes are in `share\radiance\recipes`. `bin\rad-export.cmd` turns a finetune into a
  container with the stock model's recipe. It needs Python 3 (plus torch and safetensors to fold in
  a LoRA adapter). It runs the converter beside it, `bin\rad-convert.exe`, and lists that package's
  recipes: `.\bin\rad-export.cmd --list-recipes` from the unzipped folder. Export with the
  finetune's folder and `-o OUTPUT.rad`; `--rad-convert PATH` chooses another converter. Its exit
  status is rad-export's, so a script can tell a failed export.
- **Serving without a card:** add `-Cpu`, as in `.\serve.ps1 minicpm5-2b-1card -Cpu`. The server
  then uses no GPU: the HIP runtime is never asked for a device, the CPU kernels `libavx,libref`
  replace the package's gfx1201 ones, and no model uses speculative decoding (the CPU kernels
  cannot run every drafter). These apply over the profile, which is not changed, and the startup
  report names them. `-Cpu` takes neither `-Bus` nor `-Device`; a profile of several models keeps
  its one-card models. Without `-Cpu`, a start that would leave the server no card is refused, so
  a failed card never becomes a CPU server unasked. This was checked with MiniCPM5-2B IQ8R and
  EmbeddingGemma 2, and is not qualified (`SUPPORT.md`, known issues 4 and 10). A CPU-only package
  (`-win64-cpu`) can be built from source.
- **The startup report** cannot see PCIe link speeds above each card's own link (Windows does not
  report them for bridges); those hops print "?".

## What is in this package

- `MANIFEST.json`: the source commit, toolchain, HIP runtime and the driver on the build machine,
  and the SHA-256 of every other file in the package.
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
  - **Gemma 4 31B a4 and 12B i8g** (`keptn/gemma4-31b-a4`, `keptn/gemma4-12b-i8g`): Apache 2.0, as
    Google's Gemma 4; the repositories carry the licence text unchanged. No use restrictions to
    pass on.
  - **Qwen3.6-35B-A3B IQ8R** (`StillDeadcode/qwen3.6-35b-a3b-iq8r`): Apache 2.0, as Qwen's
    Qwen3.6-35B-A3B. No use restrictions to pass on.
  - **Qwen-Image 2.1 IQ8R** (`StillDeadcode/radiance-qwen-image-2.1-iq8r`, pictures): the **Qwen
    RESEARCH LICENSE AGREEMENT** of Hangzhou Tongyi Laboratory Technology Co., Ltd.
    (`/terms/licenses/qwen-research`).
    - **Non-commercial use only.** It allows use for research or evaluation; any commercial use
      needs a separate licence from Qwen (model-business@notice.qwencloud.com).
    - **Rules of use** (section 4): export-control laws apply; a model made or improved with it or
      its outputs and then made available must say "Built with Qwen" or "Improved using Qwen" in
      its documentation; "Qwen" may not be the main name of a derivative product.
    - **Passing it on.** A copy passed on carries the licence and the notice "Qwen is licensed
      under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory
      Technology Co., Ltd. All Rights Reserved." `TERMS-OF-USE.md` passes the non-commercial limit
      on to the people you serve.
    - **The container** is a modified copy: its linear weights are re-encoded as rotated int8.
  - **DeepSeek-V4.1-Flash-1017, bakon3's uncensored container**
    (`bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad`, text): the **MIT License**, as DeepSeek's
    `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`LICENSE` there: "Copyright (c) 2023 DeepSeek";
    `/terms/licenses/deepseek-mit`). The container's README states MIT for the original and for its
    modifications, and neither repository is gated. No use restrictions to pass on; a copy passed
    on keeps the copyright notice and the licence. The
    container is a modified copy, not a DeepSeek release: its weights are re-encoded to GGML
    types, its Engram table is generated from the tokenizer, and 18 attention-output projections
    are edited to remove the model's refusal direction ("abliterated"), so it refuses less than
    DeepSeek's model.
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
