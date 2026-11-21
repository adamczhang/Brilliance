#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: check a package the way a user meets it -- unzipped to a path with spaces and
  non-ASCII characters, run from a shell with a minimal environment, serving a model on the CPU
  (every GPU hidden) and listening on the local network. No card is used.

.DESCRIPTION
    powershell -File win\tools\package-check.ps1 -Zip dist\Brilliance-...-win64.zip -Model D:\models\...\minicpm5-2b-iq8r.rad

  Steps, each printed as PASS or FAIL:
    1. unzip into <Work>\pkg tëst 日本 (.NET's zip reader); the package carries its tools, docs,
       FFmpeg and recipes, and every file matches MANIFEST.json's hash, with none unlisted
    2. the model reached through a hard link with a non-ASCII name in that folder (no copy; the
       link is removed at the end, the model is untouched)
    3. bin\radiance.exe --help from a minimal environment (SystemRoot, a PATH of System32 only,
       TEMP): every DLL comes from the package or Windows
    4. --check on a CPU profile written into the folder
    5. serve.ps1 <profile> -Cpu -Listen 0.0.0.0 -Port <Port> -ApiKey off -Background: no GPU
       (-Cpu: the host backend, HIP not asked, the CPU kernels, no drafting): /health, a chat
       request and the dashboard page over loopback AND over this machine's LAN address
    6. stopped with tools\stop-server.ps1 (Ctrl-Break), exit status 0 in the log

  The GPU package serves on the CPU only without its gfx1201 kernel libraries. Step 5's -Cpu sets
  the CPU kernels (libavx,libref) itself, so -Kernels is not needed; if given, it is written into
  the test profile (server.kernels). -NoServe stops after step 4.

  The LAN step proves the bind and the address; a request from another machine is what proves the
  firewall rule, and is left to the user (printed at the end).

  Windows PowerShell 5.1; the script file is UTF-8 with a BOM so its non-ASCII names survive.
#>
param(
  [Parameter(Mandatory)][string]$Zip,
  [Parameter(Mandatory)][string]$Model,
  [string]$Work = '',
  [int]$Port = 8000,
  # Optional server.kernels for the CPU profile. Not needed: serve.ps1 -Cpu (step 5) sets
  # 'libavx,libref' itself, since the package's gfx1201 kernel libraries cannot run on the host.
  [string]$Kernels = '',
  # Steps 1-4 only
  [switch]$NoServe
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Work) { $Work = Join-Path $repo 'artifacts\m6' }
$Zip = (Resolve-Path $Zip).Path
$Model = (Resolve-Path $Model).Path
$fails = 0
function Result([bool]$ok, [string]$what) {
  if ($ok) { Write-Host "PASS $what" } else { Write-Host "FAIL $what"; $script:fails++ }
}

# A child process with a minimal environment: what a fresh shell on a machine without the
# toolchain has. Returns @{ code; out }.
function Invoke-Minimal([string]$exe, [string[]]$argv, [hashtable]$extra = @{}) {
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $exe
  $psi.Arguments = ($argv | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join ' '
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.StandardOutputEncoding = [Text.Encoding]::UTF8
  $psi.EnvironmentVariables.Clear()
  # What every Windows session has even when nothing else is set: without SystemDrive and ProgramData,
  # Windows' own shell components write their caches under a literal '%SystemDrive%' folder.
  foreach ($k in 'SystemRoot', 'windir', 'SystemDrive', 'ProgramData', 'ALLUSERSPROFILE') {
    $psi.EnvironmentVariables[$k] = [Environment]::GetEnvironmentVariable($k)
  }
  $psi.EnvironmentVariables['PATH'] = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
  $psi.EnvironmentVariables['PATHEXT'] = '.COM;.EXE;.BAT;.CMD'
  $psi.EnvironmentVariables['TEMP'] = $env:TEMP
  $psi.EnvironmentVariables['TMP'] = $env:TEMP
  $psi.EnvironmentVariables['USERPROFILE'] = $env:USERPROFILE
  $psi.EnvironmentVariables['LOCALAPPDATA'] = $env:LOCALAPPDATA
  foreach ($k in $extra.Keys) { $psi.EnvironmentVariables[$k] = $extra[$k] }
  $p = [Diagnostics.Process]::Start($psi)
  $out = $p.StandardOutput.ReadToEnd() + $p.StandardError.ReadToEnd()
  $p.WaitForExit()
  return @{ code = $p.ExitCode; out = $out }
}

# 1. unzip
$dir = Join-Path $Work 'pkg tëst 日本'
if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
New-Item -ItemType Directory -Force $dir | Out-Null
# .NET's zip reader, not tar.exe: bsdtar takes its arguments in the ANSI code page, which has no
# '日本', so it cannot even reach the folder. (radiance.exe declares UTF-8 as its code page in its
# manifest; reaching this folder through it is what the steps below check.)
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::ExtractToDirectory($Zip, $dir)
$pkg = Get-ChildItem $dir -Directory | Select-Object -First 1
if (-not $pkg) { $pkg = Get-Item $dir }
$root = if (Test-Path (Join-Path $dir 'bin\radiance.exe')) { $dir } else { $pkg.FullName }
Result (Test-Path (Join-Path $root 'bin\radiance.exe')) "unzip into '$dir'"
$exe = Join-Path $root 'bin\radiance.exe'

# 1b. what it must carry, and every file against the manifest
$want = 'bin\brl-stacks.exe', 'bin\radiance.map', 'bin\rad-convert.exe', 'bin\rad-export.cmd',
        'libexec\brilliance\hfget.ps1', 'libexec\brilliance\resolve-stacks.py', 'serve.ps1',
        'tools\serve-bg.ps1', 'tools\stop-server.ps1', 'tools\reload-server.ps1', 'tools\STACKS.md',
        'PREVIEW.md', 'SERVICE.md', 'SUPPORT.md', 'MANIFEST.json'
$lack = @($want | Where-Object { -not (Test-Path -LiteralPath (Join-Path $root $_)) })
$ffdll = @(Get-ChildItem (Join-Path $root 'bin') -Filter 'avcodec-*.dll' -ErrorAction SilentlyContinue)
$recipes = @(Get-ChildItem (Join-Path $root 'share\radiance\recipes') -Filter *.recipe -ErrorAction SilentlyContinue)
if (-not $ffdll) { $lack += 'bin\avcodec-*.dll (FFmpeg)' }
if (-not $recipes) { $lack += 'share\radiance\recipes\*.recipe' }
Result ($lack.Count -eq 0) ("it carries its tools, docs, FFmpeg ($(($ffdll | ForEach-Object Name) -join ', ')) and $($recipes.Count) recipes" +
                            $(if ($lack.Count) { "; missing: $($lack -join ', ')" } else { '' }))
$manifest = Join-Path $root 'MANIFEST.json'
if (Test-Path -LiteralPath $manifest) {
  $files = (Get-Content -LiteralPath $manifest -Raw -Encoding UTF8 | ConvertFrom-Json).files
  $listed = @{}
  $bad = New-Object System.Collections.Generic.List[string]
  foreach ($p in $files.PSObject.Properties) {
    $listed[$p.Name] = $true
    $f = Join-Path $root ($p.Name -replace '/', '\')
    if (-not (Test-Path -LiteralPath $f)) { $bad.Add("missing $($p.Name)"); continue }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $f).Hash.ToLowerInvariant() -ne $p.Value) { $bad.Add("changed $($p.Name)") }
  }
  foreach ($f in (Get-ChildItem -LiteralPath $root -Recurse -File)) {
    $rel = $f.FullName.Substring($root.Length + 1).Replace('\', '/')
    if ($rel -ne 'MANIFEST.json' -and -not $listed.ContainsKey($rel)) { $bad.Add("unlisted $rel") }
  }
  Result ($listed.Count -gt 0 -and $bad.Count -eq 0) ("MANIFEST.json: $($listed.Count) files, each hash matches" +
                                                      $(if ($bad.Count) { "; " + (($bad | Select-Object -First 5) -join ', ') } else { '' }))
}

# 2. the model under a non-ASCII name (a hard link: same volume, no copy)
$link = Join-Path $dir 'modèle 模型.rad'
try {
  New-Item -ItemType HardLink -Path $link -Target $Model -ErrorAction Stop | Out-Null
  Result $true "model reached as '$link' (hard link)"
} catch {
  Result $false "hard link to the model: $($_.Exception.Message) (the model and the work folder must share a volume)"
  $link = $Model
}

# 3. --help from a minimal environment (radiance takes no other flags: its settings are YAML)
$r = Invoke-Minimal $exe @('--help')
Result ($r.code -eq 0 -and $r.out.Length -gt 200) "radiance --help in a minimal environment ($($r.out.Length) chars)"

# 4. a CPU profile beside the model, and --check
$yaml = Join-Path $dir 'cpu ü.yaml'
$m = $link -replace '\\', '/'
$kline = if ($Kernels) { "`n  kernels: $Kernels" } else { '' }
@"
# package-check: MiniCPM5-2B on the CPU, every GPU hidden; the server block is serve.ps1's.
server:
  host: 127.0.0.1
  port: $Port$kline
models:
  minicpm5-2b:
    model: "$m"
    tp: 1
    max-num-seqs: 1
    max-model-len: 2048
    num-speculative-tokens: 0
"@ | Set-Content -Encoding UTF8 $yaml
$r = Invoke-Minimal $exe @('--config', $yaml, '--check') @{ HIP_VISIBLE_DEVICES = '-1' }
Result ($r.code -eq 0) "radiance --config '$yaml' --check (exit $($r.code))"

if ($NoServe) {
  if ($link -ne $Model -and (Test-Path $link)) { Remove-Item -Force $link }
  Write-Host $(if ($fails) { "$fails check(s) FAILED" } else { 'every check passed (-NoServe: steps 1-4)' })
  exit $(if ($fails) { 1 } else { 0 })
}

# 5. serve on the CPU, on every interface, no key
$log = Join-Path $dir 'serve ü.log'
$serve = Join-Path $root 'serve.ps1'
$ps = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$r = Invoke-Minimal $ps @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $serve, $yaml, '-Cpu', '-Listen', '0.0.0.0',
                          '-Port', "$Port", '-ApiKey', 'off', '-Background', '-Log', $log) @{ HIP_VISIBLE_DEVICES = '-1' }
$serverPid = 0
$pidLine = ($r.out -split "`r?`n" | Where-Object { $_ -match '^\s*\d+\s*$' } | Select-Object -Last 1)
if ($pidLine) { $serverPid = [int]$pidLine.Trim() }
Result ($serverPid -gt 0) "serve.ps1 -Cpu -Listen 0.0.0.0 -Port $Port -ApiKey off -Background (pid $serverPid)"
if (-not $serverPid) { Write-Host $r.out }
$deadline = (Get-Date).AddSeconds(180)
$up = $false
while ((Get-Date) -lt $deadline) {
  if ((Test-Path $log) -and (Select-String -Path $log -Pattern 'listening on' -Quiet)) { $up = $true; break }
  if ($serverPid -and -not (Get-Process -Id $serverPid -ErrorAction SilentlyContinue)) { break }
  Start-Sleep -Milliseconds 500
}
Result $up "the server is listening (log '$log')"
$lan = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and $_.PrefixOrigin -ne 'WellKnown' } | Select-Object -First 1).IPAddress
$bound = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
Result ($bound -and $bound.LocalAddress -eq '0.0.0.0') "listening on $($bound.LocalAddress):$Port"
# A reasoning model spends its first tokens thinking (reasoning_content); either text is an answer.
$body = '{"model":"minicpm5-2b","messages":[{"role":"user","content":"Name a primary colour. One word."}],"max_tokens":256,"temperature":0}'
foreach ($h in @('127.0.0.1', $lan)) {
  if (-not $h) { continue }
  try {
    $hr = Invoke-WebRequest -UseBasicParsing -Uri "http://${h}:$Port/health" -TimeoutSec 10
    Result ($hr.StatusCode -eq 200) "GET http://${h}:$Port/health -> $($hr.StatusCode)"
    $cr = Invoke-WebRequest -UseBasicParsing -Method Post -Uri "http://${h}:$Port/v1/chat/completions" -ContentType 'application/json' -Body $body -TimeoutSec 300
    $msg = ($cr.Content | ConvertFrom-Json).choices[0].message
    $txt = if ($msg.content) { $msg.content } else { $msg.reasoning_content }
    $short = if ($txt) { $txt.Trim() -replace '\s+', ' ' } else { '' }
    if ($short.Length -gt 80) { $short = $short.Substring(0, 80) + '...' }
    Result ($cr.StatusCode -eq 200 -and $txt) "POST http://${h}:$Port/v1/chat/completions -> $($cr.StatusCode): '$short'"
    $dr = Invoke-WebRequest -UseBasicParsing -Uri "http://${h}:$Port/" -TimeoutSec 10
    Result ($dr.StatusCode -eq 200 -and $dr.Content -match '<html') "GET http://${h}:$Port/ (the dashboard page) -> $($dr.StatusCode), $($dr.Content.Length) bytes"
    $sr = Invoke-WebRequest -UseBasicParsing -Uri "http://${h}:$Port/stats" -TimeoutSec 10
    Result ($sr.StatusCode -eq 200) "GET http://${h}:$Port/stats (the dashboard's data) -> $($sr.StatusCode)"
  } catch {
    Result $false "requests to ${h}:${Port}: $($_.Exception.Message)"
  }
}

# 6. stop with Ctrl-Break
if ($serverPid) {
  & (Join-Path $root 'tools\stop-server.ps1') -ProcessId $serverPid -WaitS 60 | Out-Host
  Start-Sleep -Seconds 1
  Result ((Test-Path $log) -and (Select-String -Path $log -Pattern 'radiance exit status 0' -Quiet)) 'stopped with Ctrl-Break, exit status 0'
}
if ($link -ne $Model -and (Test-Path $link)) { Remove-Item -Force $link }

Write-Host ""
Write-Host "From another machine on the network: curl http://${lan}:$Port/health (proves the firewall rule)."
Write-Host $(if ($fails) { "$fails check(s) FAILED" } else { 'every check passed' })
exit $(if ($fails) { 1 } else { 0 })
