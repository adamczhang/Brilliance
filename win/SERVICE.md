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
| **Log** | stdout and stderr go to `--log FILE`, else `<config folder>\<config name>.log`. The previous log is kept as `.log.1`. The service's own lines start `service:` with a time stamp. |
| **Secrets** | `--env-from-file VAR=FILE` sets the environment variable `VAR` to the one line in `FILE` before the engine reads its YAML. A YAML that says `api-key-env: VAR` (radiance's own key) then finds the key, and the key is on no command line and in no registry value. The log says that VAR was set, never the value. |
| **Working directory** | the config's folder. The SCM starts services in System32. |
| **Status** | `START_PENDING`, then `RUNNING` once the engine starts. A model takes minutes to load: `/health` says when it serves. |

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
to see where it waits, and decide from there. (On Windows `scheduler.stuck-exit` is reported, never
acted on, so a hung GPU leaves the process up rather than ending it.)

**Exit status.** `sc.exe query NAME` shows `WIN32_EXIT_CODE` and `SERVICE_EXIT_CODE`:

| | Meaning |
|---|---|
| `0` | stopped cleanly, or ended in any way after a stop was asked for (the log has the engine's status) |
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

**Recovery** (`serve.ps1 -Install` sets it): the first failure in a day restarts the service after
2 minutes, and any later failure does nothing, so a card in trouble gets one more load at most.
`-NoRestart` sets no action at all. A non-zero exit nobody asked for counts as a failure, as does
a crash. A requested stop never does.

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
- The YAML is the profile with its server block as asked, the same file `serve.ps1` runs in a
  console. It goes to `%ProgramData%\Brilliance\<Name>\<Name>.yaml`, and the log goes beside it.
  That folder is SYSTEM and Administrators full control, local users read.
- `-ApiKey env:NAME` copies the value of `NAME` in the installing shell into
  `%ProgramData%\Brilliance\keys\<Name>.key`, readable by SYSTEM and Administrators only. The YAML
  says `api-key-env: NAME`, and the service gets `--env-from-file NAME=<that file>`. `-ApiKey off`
  serves without a key. A profile with a literal `api-key:` is refused: the service's YAML is
  readable by every local user. A profile's own `api-key-env:` with no `-ApiKey` is read from the
  *system* environment, which LocalSystem has, not from the installing user's.
- The service's own environment (the registry value `Environment`) carries, in this order: the cards
  (`HIP_VISIBLE_DEVICES`, from `-Bus`, `-Device`, or the shell's), `BRILLIANCE_HF_CACHE` from the
  installing shell, and the profile's `# brilliance-env: NAME=VALUE` lines with any `-Env
  NAME=VALUE[,...]` given to `-Install` (`-Env` overrides the profile, and either overrides the
  shell's `BRILLIANCE_HF_CACHE`). Only `GPU_*`, `HIP_*`, `AMD_*`, `BRILLIANCE_*` and `RADIANCE_*`
  names are taken, not `HIP_VISIBLE_DEVICES` (use `-Bus`) and no name that looks like a secret; the value
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
   Administrators only, and `Select-String -Path "$dir\*.yaml" -Pattern api-key` shows
   `api-key-env: NAME` and no key.

3. **Start it and watch the log.**

       Start-Service $svc; Get-Service $svc       # Running at once; the model loads in the background
       Get-Content "$dir\$svc.log" -Wait          # Ctrl-C ends the watching, not the service

   The log starts with `service: 'Brilliance-qwen27-preview' (pid N): ...radiance.exe --config ...`.
   The engine's own lines follow, through `server: listening on http://127.0.0.1:8000`.

4. **Check that it serves.**

       Invoke-RestMethod http://127.0.0.1:8000/health          # status ok, the model ready
       $body = '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32}'
       Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body

   With a key, add `-Headers @{ Authorization = "Bearer $env:NAME" }`. Without it the answer is 401.

5. **Stop it under load, and check the drain.** In a second window, start a long streamed request
   (for example the request above with `"max_tokens":2000,"stream":true` through `curl.exe -N`).
   While it streams:

       $t = Get-Date; Stop-Service $svc; '{0:N1} s' -f ((Get-Date) - $t).TotalSeconds
       sc.exe query $svc                           # STOPPED, WIN32_EXIT_CODE 0, SERVICE_EXIT_CODE 0

   Expect the stream to end, either finished or cut off at `drain-timeout`, and the stop to take
   seconds, not minutes. The log ends with
   `service: STOP: draining ...`, `stopping: draining in-flight requests`, `service: stopped`.
   Also check that the display driver did not reset during the stop:

       Get-WinEvent -FilterHashtable @{ LogName = 'System'; Id = 4101; StartTime = $t } -ErrorAction SilentlyContinue
                                                   # nothing: no "display driver stopped responding"

6. **Restart it, three times.** Each start begins a new log and keeps the last one as `.log.1`. Each
   must serve again, which shows that a stop leaks no pinned memory or queues.

       1..3 | ForEach-Object {
         Start-Service $svc
         do { Start-Sleep 5; try { $h = Invoke-RestMethod http://127.0.0.1:8000/health } catch { $h = $null } } until ($h -and $h.status -eq 'ok')
         "start $_ healthy"; Stop-Service $svc; (Get-Service $svc).Status
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
