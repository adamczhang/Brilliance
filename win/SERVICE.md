# Brilliance as a Windows service

radiance can run as a Windows service: started by the system at boot, stopped by the same graceful
drain as Ctrl-C in a console. **This is not advertised until the qualification below has passed on
the machine.** Everything that does not need the Service Control Manager is covered by the host tests
`service_win_test` and `serve_service_test`. The steps here are the part that needs an elevated shell.

## What it does

The Service Control Manager (SCM) starts

    radiance.exe --service NAME --config FILE [--log FILE] [--env-from-file VAR=FILE]...
                 [--stop-timeout SECONDS]

That is radiance's own `main` with `--config FILE`: the same engine, the same YAML, the same fleet.
The service adds only what a service needs and a console already has:

| | |
|---|---|
| **Log** | stdout and stderr go to `--log FILE`, else `<config folder>\<config name>.log`. At each start the service tries to keep the previous log as `.log.1`; if that rename fails, it goes on without saying so. The service's own lines start `service:` with a time stamp. |
| **Secrets** | `--env-from-file VAR=FILE` sets the environment variable `VAR` to the one line in `FILE` before the engine reads its YAML. A YAML that says `api-key-env: VAR` (radiance's own key), or `BRILLIANCE_SERVER_API_KEY=env:VAR` in the service's environment (what `serve.ps1 -Install -ApiKey env:VAR` sets), then finds the key, and the key is on no command line and in no registry value. The log says that VAR was set, never the value. |
| **Working directory** | the config's folder. The SCM starts services in System32. |
| **Status** | `START_PENDING`, then `RUNNING` just before the engine starts, so `RUNNING` does not mean a model is ready. The model then loads (seconds to minutes). With a one-model profile, wait until `/health` answers and a test request gets an answer. |

**A stop is the drain.** `STOP` (`Stop-Service`, `sc.exe stop`), `PRESHUTDOWN` (Windows shutting
down) and `SHUTDOWN` all reach the engine as SIGTERM, which is what Ctrl-Break does in a console.
The fleet stops admitting work, lets running requests finish for up to `scheduler.drain-timeout`
(30 s by default) and then ends them, releases the cards, and exits 0. A stop that arrives while a
model is loading waits for that load to finish first. A stop that arrives before the engine has
registered its handler (while it reads its configuration or probes the cards) is kept, and the
engine takes it as soon as it registers, before it loads anything.

While it drains, the service reports `STOP_PENDING` with a new checkpoint every 2 s (wait hint
10 s), so the SCM sees progress.

**Nothing ends the process by force.** Under Brilliance 0.2.2, killing a process with work on a GPU
took the display driver down. If the drain has not finished after `--stop-timeout` (180 s by
default), the log says so, with the process id, and the checkpoint stops moving. The service stays
`STOP_PENDING` until the engine returns. Do not `taskkill /f` it. Run `brl-stacks <pid>` (in `bin\`)
to see where it waits, and decide from there.

**A hang is not ended either.** Upstream's guide describes Linux, where a device that stops making
progress turns `/health` to `"stuck"` and `scheduler.stuck-exit` ends the process, so a supervisor
restarts it. On Windows neither ends the process: the stall watch logs a stall and, after 60 s of a
blocked wait, `/health` answers 503 with `"status": "stuck"`, but `scheduler.stuck-exit` is never
acted on. The process stays up, the service stays `RUNNING`, and the recovery below never fires for
a hang. A person has to act (`SUPPORT.md`, "Hang detection").

**Exit status.** `sc.exe query NAME` shows `WIN32_EXIT_CODE` and `SERVICE_EXIT_CODE`:

| | Meaning |
|---|---|
| `0` | the engine returned normally, or returned an error after a stop was asked for (the log has the engine's status). A process that ends abruptly, such as the `fatal()` exit below, reports nothing here |
| `1066` / `1` | the engine failed without being asked to stop (no card, no model started, ...) |
| `1066` / `2` | the command line or the YAML was refused |
| `1066` / `3` | the log file could not be opened |
| `1066` / `4` | an `--env-from-file` file could not be read, was empty or had several lines |

A device error that stops a serving model's engine ends the process through upstream's `fatal()`: a
`radiance: fatal:` line in the log, then `_exit(70)`, with no stop reported to the Service Control
Manager. That covers a two-card host-mailbox fault the engine cannot recover (since 2026-11-18;
`SUPPORT.md`, known issue 1) and, since 2026-11-19, any other device error
(`RAD_E_DEVICE`) the step loop stops on; other stops (out of memory, an unload or a swap) do not
exit. The manager should then see the process end unexpectedly, which the recovery below counts
as a failure. Unverified: no such fault has been run under the service, so the codes `sc.exe query`
shows for it are not recorded.

**Recovery** (`serve.ps1 -Install` sets it): the first failure restarts the service after 2
minutes, and later failures do nothing until a full day (24 hours) has passed without a failure,
when the count starts again. So a card in trouble gets one more load at most. `-NoRestart` sets no
action at all. A non-zero exit nobody asked for counts as a failure, as does a crash. A stop that
was asked for and completes is reported as a success. A hang is not a failure the Service Control
Manager sees (above).

**Run from a console**, the same command line runs there as the service would. It prints
`not started by the Service Control Manager` and the log's path, writes its output to the log,
and stops on Ctrl-C or Ctrl-Break (`tools\stop-server.ps1 -ProcessId N`). Setting the event
`Local\brilliance-service-stop-<pid>` takes the service's own STOP path. In a console the status
the SCM would have been told goes to the log as `(console) status ...` lines. This is how the host
test runs it, and the first step below.

## Installing: `serve.ps1 -Install`

    .\win\serve.ps1 <profile> [-Bus N | -Device N] [-Listen ADDR] [-Port N] [-ApiKey off|env:NAME]
                    -Install [-Name NAME] [-Manual] [-NoRestart] [-StopTimeoutS S] [-Log FILE] [-Exe PATH] [-WhatIf]
    .\win\serve.ps1 <profile> -Uninstall [-Name NAME] [-WaitS S] [-WhatIf]

(In the package it is `.\serve.ps1`, beside `bin\`.)

- The service is named `Brilliance-<profile>` unless `-Name` says otherwise. It runs as LocalSystem,
  in its own process, with a delayed automatic start (about 2 minutes after boot). `-Manual` starts
  it only on demand.
- The YAML is a copy of the profile as it is, at `%ProgramData%\Brilliance\<Name>\<Name>.yaml`,
  and the log goes beside it. That folder is SYSTEM and Administrators full control, local users
  read. To change a model later, edit that copy and reload (`tools\reload-server.ps1 -ProcessId
  <pid>` from an elevated shell, or `POST /admin/reload`); the profile is not read again.
- `-Listen`, `-Port` and `-ApiKey` are not written into the YAML. They go into the service's
  environment as `BRILLIANCE_SERVER_HOST`, `BRILLIANCE_SERVER_PORT` and `BRILLIANCE_SERVER_API_KEY`
  (`off` or `env:NAME`), which radiance applies over the YAML's `server:` block at the start and at
  every reload. `-WhatIf` and `-Install` print the host, port and key radiance reads that way.
- `-ApiKey env:NAME` copies the value of `NAME` in the installing shell into
  `%ProgramData%\Brilliance\keys\<Name>.key`, readable by SYSTEM and Administrators only, and the
  service gets `--env-from-file NAME=<that file>` and `BRILLIANCE_SERVER_API_KEY=env:NAME`.
  `-ApiKey off` serves without a key. A profile with a literal `api-key:` is refused, with or without
  `-ApiKey`: its copy would hold the key, readable by every local user. A profile that already says `api-key-env: NAME`, installed with no
  `-ApiKey`, needs `NAME` twice: in the installing PowerShell, for the install's `--check`, and in
  the *system* environment, which is what the service (LocalSystem) reads when it starts. Use
  `-ApiKey env:NAME` instead to store the installing shell's value in the protected key file.
- The service's own environment (the registry value `Environment`) carries, in this order: the cards
  (`HIP_VISIBLE_DEVICES`, from `-Bus`, `-Device`, the shell's, or every discrete card with an
  integrated GPU hidden), `BRILLIANCE_HF_CACHE` from the installing shell, the profile's
  `# brilliance-env: NAME=VALUE` lines with any `-Env NAME=VALUE[,...]` given to `-Install` (`-Env`
  overrides the profile, and either overrides the shell's `BRILLIANCE_HF_CACHE`), then the
  `BRILLIANCE_SERVER_*` values above. Only `GPU_*`, `HIP_*`, `AMD_*`, `BRILLIANCE_*` and `RADIANCE_*`
  names are taken, not `HIP_VISIBLE_DEVICES` (use `-Bus`), not `BRILLIANCE_SERVER_*` (use `-Listen`,
  `-Port`, `-ApiKey`) and no name that looks like a secret; the value
  is readable by local users. A LocalSystem service has none of the user's variables. HIP indices
  are resolved at install, so after a card is added or moved, uninstall and install again; after a
  profile's environment changes, likewise.
- At shutdown the SCM waits up to 180 s (PRESHUTDOWN), or `-StopTimeoutS` if that is longer, for
  the drain.
- `-WhatIf` prints every step in order, exactly as it would run, and changes nothing. It needs no
  elevation. `-Install` and `-Uninstall` themselves need an elevated PowerShell.
- `-Uninstall` stops the service (the drain) and waits up to `-WaitS` (300 s) for it to stop. Only
  then does it delete the service and its key file. A service that is still stopping is reported and
  left alone: it is not deleted and not killed. The YAML and the logs stay.
- A service runs `radiance.exe` from where it was installed, as LocalSystem. Any account that can
  write to that folder can run code as SYSTEM. On a machine others use, install from a package
  unpacked under `C:\Program Files`; `-Install` says so when it is not.

## Moving an alpha.3 service to alpha.4

**Not yet run in an elevated shell.** These steps follow how `-Install` and `-Uninstall` work in
`serve.ps1`; they have not been run end to end on an installed service.

A service runs the `radiance.exe` it was installed from, so an alpha.3 service keeps running
alpha.3 until it is installed again. There is no upgrade in place: `-Install` refuses a name that is
already installed. So record what the service has, remove it, and install it again from alpha.4.
Three things changed in `-Install`:

- alpha.3 wrote the service's YAML as the profile with its `server:` block rewritten (`host`,
  `port`, `api-key-env` from `-Listen`, `-Port`, `-ApiKey`). alpha.4 copies the profile as it is and
  puts those three into the service's environment (`BRILLIANCE_SERVER_*`, above).
- alpha.3 dropped a profile's `api-key:` line when `-ApiKey` was given. alpha.4 refuses a profile
  that holds `api-key:`, with or without `-ApiKey`.
- Without `-Bus`, `-Device` or the shell's `HIP_VISIBLE_DEVICES`, alpha.4 gives the service every
  discrete card with an integrated GPU hidden. Give the same `-Bus` as before.

In an elevated PowerShell:

0. **Unpack alpha.4 into a folder of its own** under `C:\Program Files` (for example
   `C:\Program Files\Brilliance-0.3.0-alpha.4`), check its hash, and run the rest from there. Do not
   unpack over the alpha.3 folder: the service is running its `radiance.exe`.

       Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
       $svc = 'Brilliance-qwen27-preview'      # the service's name
       $dir = "$env:ProgramData\Brilliance\$svc"

1. **Record what the service has now.**

       sc.exe qc $svc         # BINARY_PATH_NAME (the alpha.3 radiance.exe, any --log, --env-from-file, --stop-timeout), START_TYPE
       sc.exe qfailure $svc   # a RESTART action, or none (installed with -NoRestart)
       (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Services\$svc").Environment
       Copy-Item "$dir\$svc.yaml" "$dir\$svc.alpha3.yaml"

   They map onto alpha.4's options: `DEMAND_START` is `-Manual`; `--log FILE` is `-Log FILE`;
   `--stop-timeout S` is `-StopTimeoutS S`; `--env-from-file NAME=...` is `-ApiKey env:NAME`; no
   failure action is `-NoRestart`; the YAML copy's `server:` `host` and `port` are `-Listen` and
   `-Port`; the environment's `HIP_VISIBLE_DEVICES` is the HIP indices of the cards you named (give
   the same `-Bus`), and its other names (besides `BRILLIANCE_HF_CACHE`) are `-Env` values.

2. **Bring your changes into the profile.** `-Install` replaces `$svc.yaml` with the profile as it
   is. If you edited the service's copy after installing it (a model path, a setting), make the
   same change in the profile you install from, best the alpha.4 package's own
   `profiles\<profile>.yaml`. Leave out the `server:` lines alpha.3 wrote: `-Listen`, `-Port` and
   `-ApiKey` give them now. Remove any `api-key:` line and use `-ApiKey env:NAME`.

3. **With a key, put it in this shell first.** `-Uninstall` deletes the key file, and `-Install
   -ApiKey env:NAME` reads the key from the installing shell:

       $env:NAME = ([IO.File]::ReadAllText("$env:ProgramData\Brilliance\keys\$svc.key")).Trim()

   (`NAME` is the variable from `--env-from-file`.) `BRILLIANCE_HF_CACHE` comes from the installing
   shell too; check that this one has it if the old environment did.

4. **Remove the alpha.3 service.** Its running requests end with the drain, so pick a quiet moment.
   Either package's `serve.ps1` removes it by name (add `-Name` if it was installed with one):

       .\serve.ps1 qwen27-preview -Uninstall -WhatIf
       .\serve.ps1 qwen27-preview -Uninstall

   A service that has not stopped within `-WaitS` (300 s) is left alone, not deleted and not
   killed; run `-Uninstall` again once it has stopped. The YAML and the logs stay.

5. **Install it from alpha.4**, with the options from step 1. Preview first:

       .\serve.ps1 qwen27-preview -Bus 3 -Install -WhatIf
       .\serve.ps1 qwen27-preview -Bus 3 -Install

   In the preview, the `sc.exe create` line names alpha.4's `bin\radiance.exe`, the YAML is the
   profile, and the environment holds the same `HIP_VISIBLE_DEVICES` as in step 1 and the
   `BRILLIANCE_SERVER_*` values; `-Install` prints the host, port and key the service will use. The
   alpha.3 log stays, and the first start keeps it as `.log.1`.

6. **Start it and check it serves** (steps 3 and 4 of the qualification below), and that it is
   alpha.4. With a key, add `-Headers @{ Authorization = "Bearer $env:NAME" }` to the `/version`
   request.

       Start-Service $svc
       Invoke-RestMethod http://127.0.0.1:8000/health
       (Invoke-RestMethod http://127.0.0.1:8000/version).brilliance.version   # 0.3.0-alpha.4

7. **A firewall rule that names the program** (the one alpha.3's `-Install` printed for the local
   network names alpha.3's `radiance.exe`) must name alpha.4's now:

       Get-NetFirewallRule -DisplayName $svc | Set-NetFirewallApplicationFilter -Program "$PWD\bin\radiance.exe"

8. **Once it serves,** delete `$dir\$svc.alpha3.yaml` and the alpha.3 folder, and close this shell
   (it holds the key).

## Qualification, in an elevated PowerShell

The service in these steps is `Brilliance-qwen27-preview`, the preview profile on the card on
PCI bus 3, loopback, no key. Adjust `-Bus`, `-Listen` and `-ApiKey` to the machine's policy; a
port opened to the network needs a firewall rule, which is the administrator's to add. Use the HIP
build (`build-win\hip\bin`) or a package. Note the GPU tuning in force when you record results.

0. **Open an elevated PowerShell** (Run as administrator) in the source tree.

       $svc = 'Brilliance-qwen27-preview'
       $dir = "$env:ProgramData\Brilliance\$svc"

1. **Preview the install.** Read the steps: the folders, the YAML (check `model:`), the `sc.exe
   create` line, the recovery actions and the environment (`HIP_VISIBLE_DEVICES` should be the
   card on bus 3).

       .\win\serve.ps1 qwen27-preview -Bus 3 -Install -WhatIf

2. **Install.**

       .\win\serve.ps1 qwen27-preview -Bus 3 -Install

   Expect `radiance ... --check` to print the model line, then `sc.exe` `[SC] CreateService SUCCESS`,
   and the `qfailure` printout: `RESTART -- Delay = 120000 milliseconds` then `-- Delay = 0`. Check:

       sc.exe qc $svc                  # BINARY_PATH_NAME: radiance.exe --service ... --config ...yaml
                                       # START_TYPE: 2 AUTO_START (DELAYED); SERVICE_START_NAME: LocalSystem
       sc.exe qfailureflag $svc        # FAILURE_ACTIONS_ON_NONCRASH_FAILURES: TRUE
       Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Services\$svc" | Select-Object PreshutdownTimeout, Environment
                                       # 180000; HIP_VISIBLE_DEVICES=<index>, BRILLIANCE_HF_CACHE=<the shell's>
       icacls "$env:ProgramData\Brilliance"   # SYSTEM and Administrators F, Users RX, nothing else

   With `-ApiKey env:NAME`, also: `icacls "$env:ProgramData\Brilliance\keys"` shows SYSTEM and
   Administrators only, `Select-String -Path "$dir\*.yaml" -Pattern api-key` shows no key, and the
   service's `Environment` holds `BRILLIANCE_SERVER_API_KEY=env:NAME`.

3. **Start it and watch the log.**

       Start-Service $svc; Get-Service $svc       # Running at once; the model loads in the background
       Get-Content "$dir\$svc.log" -Wait          # Ctrl-C ends the watching, not the service

   The log starts with `service: 'Brilliance-qwen27-preview' (pid N): ...radiance.exe --config ...`.
   The engine's own lines follow, through `server: listening on http://127.0.0.1:8000`.

4. **Check that it serves.**

       Invoke-RestMethod http://127.0.0.1:8000/health          # status ok, the model ready
       $body = '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hello in five words."}],"enable_thinking":false,"max_tokens":32}'
       $r = Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
       $r.choices[0].message.content                            # a short greeting

   `"enable_thinking":false` matters here. Qwen3.8-27B thinks before it answers by default, and
   `max_tokens` counts the thinking too: without it, these 32 tokens can all go to reasoning, and the
   answer comes back empty with `finish_reason` `length`. With thinking on, give `max_tokens` room
   for the reasoning as well as the answer.

   With a key, add `-Headers @{ Authorization = "Bearer $env:NAME" }` to the chat request; the same
   request without it is refused with 401. `/health` answers with or without the key.

5. **Stop it under load, and check the drain.** In a second PowerShell window, start a long
   streamed request. The JSON goes in a file, because Windows PowerShell 5.1 strips the inner
   quotes from a JSON body typed on `curl.exe`'s command line (the server then answers 400):

       $f = "$env:TEMP\brl-stream.json"
       [IO.File]::WriteAllText($f, '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Tell a long story about a lighthouse."}],"enable_thinking":false,"max_tokens":2000,"stream":true}')
       curl.exe -N http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" --data-binary "@$f"

   With a key, add `-H "Authorization: Bearer $env:NAME"` before `--data-binary`. While it streams,
   in the first window:

       $t = Get-Date; Stop-Service $svc; '{0:N1} s' -f ((Get-Date) - $t).TotalSeconds
       sc.exe query $svc                           # STOPPED, WIN32_EXIT_CODE 0, SERVICE_EXIT_CODE 0

   Expect the stream to end, either finished or cut off at `drain-timeout`, and the stop to take
   seconds, not minutes. The log ends with
   `service: STOP: draining ...`, `stopping: draining in-flight requests`, `service: stopped`.
   Also check that the display driver did not reset during the stop:

       Get-WinEvent -FilterHashtable @{ LogName = 'System'; Id = 4101; StartTime = $t } -ErrorAction SilentlyContinue
                                                   # nothing: no "display driver stopped responding"

6. **Restart it, three times.** Each start begins a new log and should keep the last one as
   `.log.1`. Each must serve again. Serving again does not by itself show that a stop leaked
   nothing: also note each card's free VRAM and the host's free memory before the first start and
   after each stop, and compare them.

       1..3 | ForEach-Object {
         Start-Service $svc
         do { Start-Sleep 5; try { $h = Invoke-RestMethod http://127.0.0.1:8000/health } catch { $h = $null } } until ($h -and $h.status -eq 'ok')
         $r = Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
         "start $_ healthy, answered: $($r.choices[0].message.content)"; Stop-Service $svc; (Get-Service $svc).Status
       }
       Get-Item "$dir\$svc.log", "$dir\$svc.log.1"

   (A one-model profile opens its port only once the model has loaded, so the first `/health` that
   answers is a model that serves.)

7. **Boot and shutdown (optional, needs a restart of the machine).** Leave the service running and
   restart Windows. About 2 minutes after logging in, it should be `Running` and healthy again.
   The previous log (`$svc.log.1`) should end with `service: PRESHUTDOWN: draining ...` and
   `service: stopped`: the drain ran at shutdown.

8. **Recovery without a card (optional).** This installs a second service that cannot start, to see
   the "restart once, then not again" policy. `-Device -1` hides every card, so no GPU is used.

       $p = "$env:TEMP\failtest.yaml"
       "server:`n  port: 18399`nmodels:`n  x:`n    model: C:/no/such/model.rad`n" | Set-Content -Encoding ascii $p
       .\win\serve.ps1 $p -Name Brilliance-failtest -Device -1 -Manual -Install
       Start-Service Brilliance-failtest; Start-Sleep 5
       sc.exe query Brilliance-failtest     # STOPPED, WIN32_EXIT_CODE 1066, SERVICE_EXIT_CODE 1
       Start-Sleep 150; Get-WinEvent -FilterHashtable @{ LogName = 'System'; ProviderName = 'Service Control Manager' } -MaxEvents 10 |
         Where-Object Message -match 'failtest' | Format-List TimeCreated, Message
       # one restart, about 120 s after the first failure, then no further one
       .\win\serve.ps1 $p -Name Brilliance-failtest -Uninstall

9. **Uninstall, while it runs.**

       Start-Service $svc; Start-Sleep 60
       .\win\serve.ps1 qwen27-preview -Uninstall   # stops (the drain), waits, then deletes
       Get-Service $svc -ErrorAction SilentlyContinue   # nothing
       Get-ChildItem $dir                          # the YAML and the logs are still there

Record for each step what it showed (times, exit codes, the log's last lines). Whether the service
is advertised is decided from that record.

## In the package

- **`serve.ps1`** is at the package root. Its `-Install` finds `bin\radiance.exe` beside it, so a
  package needs no `-Exe`, and the examples above read `.\serve.ps1`.
- **Nothing else is needed in `bin\`.** The service is `radiance.exe --service`, and
  `brl-stacks.exe`, which the overdue-drain message names, is there.
- **Not in the package:** the host tests (`win\tests\`, `service_win_test`); they run from a
  source build.
