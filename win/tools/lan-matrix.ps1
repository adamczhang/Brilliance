#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: the LAN access matrix -- key off and on, inference, telemetry, engine and fleet admin,
  Dashboard and Sandbox -- against a packaged radiance on the CPU, with no GPU and no LAN port.
.DESCRIPTION
  Starts the package's serve.ps1 with -Cpu (HIP hidden, HIP_VISIBLE_DEVICES=-1) on 127.0.0.1 only,
  twice: once with -ApiKey off and once with -ApiKey env:BRILLIANCE_LANM_KEY (a random key made
  here). Each run serves a two-entry fleet written into -Out: the given small model, cold at start,
  and `broken-entry`, whose model file does not exist, so its load fails with a reason that names a
  path.

  Nothing listens beyond loopback. A LAN caller is simulated the way the guards tell callers apart
  (core/server/http.cpp, local_operator): the socket is always loopback here, so a "LAN" request is
  one whose Host names a non-loopback address (203.0.113.50), which is also exactly what a remote
  client behind a localhost reverse proxy looks like. A request from a real non-loopback socket is
  not made; the guard refuses such a socket before it reads the Host at all, so the simulated cell
  is the weaker of the two remote cases. The callers:
    local    Host 127.0.0.1:<port>                      the operator on the box
    xff      Host 127.0.0.1, X-Forwarded-For 203.0.113.50   attribution is not authority
    lan      Host 203.0.113.50:<port>                    a LAN client's request (simulated)
    xorigin  Host 127.0.0.1, Origin http://evil.example  a foreign page in the operator's browser
  With a key, each of none / wrong / Bearer / X-Api-Key is crossed with local and lan.

  Each cell records the request, the expected status (and, where it matters, what the body may or
  may not carry) and the actual one, into <Out>\matrix.csv and <Out>\matrix.md. Exit code 0 when
  every cell matched its expectation, 1 otherwise (a mismatch is a finding, not necessarily a bug:
  matrix.md names the policy each cell checks).

  The Dashboard and Sandbox are checked at the HTTP level always, and, unless -NoBrowser, with a
  headless Edge (or Chrome) already installed on the machine: --dump-dom of the Sandbox page from
  127.0.0.1 and from a name mapped to 127.0.0.1 (lan.test) that the guards read as remote.

  Class C (CPU): the model runs on the CPU kernels. Short generations only (8 tokens).
.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File win\tools\lan-matrix.ps1 -Package <unpacked package> -Out artifacts\lan-matrix\run1
#>
param(
  # An unpacked package: bin\radiance.exe, serve.ps1, tools\stop-server.ps1.
  [Parameter(Mandatory)][string]$Package,
  [Parameter(Mandatory)][string]$Out,
  [string]$Model = 'D:\models\hf\StillDeadcode\minicpm5-2b-iq8r\minicpm5-2b-iq8r.rad',
  [string]$ModelName = 'minicpm5-2b',
  [int]$Port = 8033,
  # off, on, both, or none: only the config cells (no server is started).
  [ValidateSet('off', 'on', 'both', 'none')][string]$Key = 'both',
  [switch]$NoBrowser,
  [int]$ReadyS = 180
)
$ErrorActionPreference = 'Stop'
$Package = (Resolve-Path $Package).Path
$Out = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)
New-Item -ItemType Directory -Force $Out | Out-Null
if (-not (Test-Path $Model)) { throw "no model file at $Model" }
$serve = Join-Path $Package 'serve.ps1'
$stop = Join-Path $Package 'tools\stop-server.ps1'
foreach ($f in @($serve, $stop, (Join-Path $Package 'bin\radiance.exe'))) { if (-not (Test-Path $f)) { throw "not a package: $f is missing" } }
$curl = Join-Path $env:SystemRoot 'System32\curl.exe'
if (-not (Test-Path $curl)) { throw "curl.exe not found at $curl" }
$utf8 = New-Object Text.UTF8Encoding($false)
$base = "http://127.0.0.1:$Port"
$lanHost = '203.0.113.50'
$bodies = Join-Path $Out 'bodies'
New-Item -ItemType Directory -Force $bodies | Out-Null
$tmp = Join-Path $Out 'tmp'
New-Item -ItemType Directory -Force $tmp | Out-Null

# ---------------------------------------------------------------- the profile
$missing = (Join-Path $Out 'no-such-dir\no-such-model.rad') -replace '\\', '/'
$yaml = Join-Path $Out 'lan-matrix.yaml'
$profileText = @"
# lan-matrix: written by win/tools/lan-matrix.ps1. Loopback only; serve.ps1 -Cpu supplies the CPU
# kernels and turns drafting off; -ApiKey supplies the key.
server:
  host: 127.0.0.1
  port: $Port
scheduler:
  warm-up: false
models:
  ${ModelName}:
    model: $($Model -replace '\\', '/')
    tp: 1
    max-num-seqs: 2
    max-model-len: 4096
    chat-template-kwargs: {"enable_thinking": false}
  broken-entry:
    model: $missing
    tp: 1
    max-model-len: 4096
"@
[IO.File]::WriteAllText($yaml, $profileText.Replace("`r`n", "`n"), $utf8)

function Write-Json([string]$name, $obj) {
  $p = Join-Path $bodies $name
  [IO.File]::WriteAllText($p, ($obj | ConvertTo-Json -Depth 6 -Compress), $utf8)
  $p
}
$chatBody   = Write-Json 'req-chat.json'   @{ model = $ModelName; max_tokens = 8; temperature = 0; messages = @(@{ role = 'user'; content = 'Say hi.' }) }
$brokenBody = Write-Json 'req-broken.json' @{ model = 'broken-entry'; max_tokens = 8; messages = @(@{ role = 'user'; content = 'Say hi.' }) }
$complBody  = Write-Json 'req-compl.json'  @{ model = $ModelName; max_tokens = 8; temperature = 0; prompt = 'The capital of France is' }
$msgBody    = Write-Json 'req-msg.json'    @{ model = $ModelName; max_tokens = 8; messages = @(@{ role = 'user'; content = 'Say hi.' }) }
$tokBody    = Write-Json 'req-tok.json'    @{ model = $ModelName; content = 'hello world' }
$modelBody  = Write-Json 'req-model.json'  @{ model = $ModelName }
$cvBody     = Write-Json 'req-cv.json'     @{ model = $ModelName; enabled = $true }

# ---------------------------------------------------------------- callers
$script:secret = ''
function Headers([string]$caller) {
  $h = @()
  switch -Regex ($caller) {
    '^lan'     { $h += "Host: ${lanHost}:$Port" }
    '^xorigin' { $h += "Host: 127.0.0.1:$Port"; $h += 'Origin: http://evil.example' }
    '^xff'     { $h += "Host: 127.0.0.1:$Port"; $h += "X-Forwarded-For: $lanHost" }
    '^local'   { $h += "Host: 127.0.0.1:$Port" }
  }
  switch -Regex ($caller) {
    '\+bearer$' { $h += "Authorization: Bearer $script:secret" }
    '\+xkey$'   { $h += "X-Api-Key: $script:secret" }
    '\+wrong$'  { $h += 'Authorization: Bearer not-the-key' }
  }
  $h
}

# ---------------------------------------------------------------- one request
$script:rows = New-Object System.Collections.Generic.List[object]
$script:n = 0
function Cell {
  param([string]$Mode, [string]$Group, [string]$Caller, [string]$Method, [string]$Path,
        [string]$Body = '', [int[]]$Expect, [string]$Policy = '',
        [string]$MustHave = '', [string]$MustNotHave = '', [int]$TimeoutS = 600)
  $script:n++
  $id = '{0}{1:000}' -f ($(if ($Mode -eq 'off') { 'A' } else { 'B' })), $script:n
  $bf = Join-Path $bodies "$id.txt"
  $a = @('-s', '-S', '-o', $bf, '-w', '%{http_code}', '--max-time', "$TimeoutS", '-X', $Method)
  foreach ($h in (Headers $Caller)) { $a += @('-H', $h) }
  if ($Body) { $a += @('-H', 'Content-Type: application/json', '--data-binary', "@$Body") }
  $a += "$base$Path"
  $ErrorActionPreference = 'Continue'
  $code = & $curl @a 2>$null
  $ErrorActionPreference = 'Stop'
  $st = 0; [void][int]::TryParse(("$code").Trim(), [ref]$st)
  $text = if (Test-Path $bf) { [IO.File]::ReadAllText($bf, [Text.Encoding]::UTF8) } else { '' }
  $ok = $Expect -contains $st
  $why = @()
  if ($ok -and $MustHave -and $text -notmatch $MustHave) { $ok = $false; $why += "body lacks /$MustHave/" }
  if ($ok -and $MustNotHave -and $text -match $MustNotHave) { $ok = $false; $why += "body carries /$MustNotHave/" }
  $snip = ($text -replace '\s+', ' ')
  if ($snip.Length -gt 140) { $snip = $snip.Substring(0, 140) + '...' }
  $row = [pscustomobject]@{
    id = $id; key = $Mode; group = $Group; caller = $Caller; request = "$Method $Path" + $(if ($Body) { ' ' + [IO.Path]::GetFileName($Body) } else { '' })
    expected = ($Expect -join '|') + $(if ($MustHave) { " +/$MustHave/" } else { '' }) + $(if ($MustNotHave) { " -/$MustNotHave/" } else { '' })
    actual = $st; bytes = $text.Length; verdict = $(if ($ok) { 'ok' } else { 'MISMATCH' }); note = ($why -join '; '); policy = $Policy; body = $snip
  }
  $script:rows.Add($row)
  "{0} {1,-8} {2,-14} {3,-58} exp {4,-10} got {5} {6}" -f $id, $Mode, $Caller, $row.request, ($Expect -join '|'), $st, $(if ($ok) { 'ok' } else { "MISMATCH $($row.note)" })
}

# ---------------------------------------------------------------- server lifecycle
function Start-Server([string]$mode) {
  $log = Join-Path $Out "server-key-$mode.log"
  $saved = @{ HIP_VISIBLE_DEVICES = $env:HIP_VISIBLE_DEVICES; TEMP = $env:TEMP; TMP = $env:TMP }
  $env:HIP_VISIBLE_DEVICES = '-1'   # no GPU at all: the HIP runtime sees none
  $env:TEMP = $tmp; $env:TMP = $tmp  # anything the launcher derives stays under -Out
  $keyArg = 'off'
  if ($mode -eq 'on') {
    $b = New-Object byte[] 24; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
    $script:secret = ([BitConverter]::ToString($b) -replace '-', '').ToLower()
    $env:BRILLIANCE_LANM_KEY = $script:secret
    $keyArg = 'env:BRILLIANCE_LANM_KEY'
  }
  try {
    $o = & powershell -NoProfile -ExecutionPolicy Bypass -File $serve $yaml -Cpu -Listen 127.0.0.1 -Port $Port -ApiKey $keyArg -Background -Log $log 2>&1
  } finally {
    foreach ($k in $saved.Keys) { Set-Item "env:$k" $saved[$k] -ErrorAction SilentlyContinue; if ($null -eq $saved[$k]) { Remove-Item "env:$k" -ErrorAction SilentlyContinue } }
    Remove-Item env:BRILLIANCE_LANM_KEY -ErrorAction SilentlyContinue
  }
  ($o | ForEach-Object { "  serve: $_" }) | Write-Host
  $procId = 0
  foreach ($l in $o) { if ("$l".Trim() -match '^\d+$') { $procId = [int]"$l".Trim() } }
  if (-not $procId) { throw "serve.ps1 did not report a process id (see $log)" }
  $t0 = Get-Date
  while (((Get-Date) - $t0).TotalSeconds -lt $ReadyS) {
    $ErrorActionPreference = 'Continue'
    $c = & $curl -s -o NUL -w '%{http_code}' --max-time 5 "$base/health" 2>$null
    $ErrorActionPreference = 'Stop'
    if ("$c" -eq '200') { Write-Host "  server pid $procId ready after $([int]((Get-Date) - $t0).TotalSeconds) s"; return $procId }
    if (-not (Get-Process -Id $procId -ErrorAction SilentlyContinue)) { throw "server $procId exited before /health answered (see $log)" }
    Start-Sleep -Milliseconds 500
  }
  throw "server $procId not ready after $ReadyS s (see $log)"
}
function Stop-Server([int]$procId) {
  $r = & powershell -NoProfile -ExecutionPolicy Bypass -File $stop -ProcessId $procId -WaitS 120 2>&1
  ($r | ForEach-Object { "  stop: $_" }) | Write-Host
}

# ---------------------------------------------------------------- the matrix
$M = $ModelName
# A bare /ping on a fleet of several models is the process's liveness, as /health is (finding F4 in the
# lan-matrix report, fixed in lan-f3-f4; it answered 400 before).
$public = @(
  @('GET', '/health'), @('GET', '/v1/health'), @('GET', "/ping?model=$M"), @('POST', "/ping?model=$M"),
  @('GET', '/'), @('GET', '/dashboard'), @('GET', '/terms'), @('GET', '/terms/licenses'))
# /ping naming a model reads its state and never loads it, as /health?model= (ping-state): 503 and the
# state while the model is cold, 200 once something else has loaded it. The cells that run while it is
# cold take these two apart from $public.
$pingModel = @(@('GET', "/ping?model=$M"), @('POST', "/ping?model=$M"))
$publicNoPing = @($public | Where-Object { $_[1] -notlike '/ping*' })
$inference = @(
  @('GET', '/v1/models', ''), @('POST', '/v1/chat/completions', $chatBody), @('POST', '/v1/completions', $complBody),
  @('POST', '/v1/messages', $msgBody), @('POST', '/tokenize', $tokBody))
$telemetry = @(
  '/metrics', "/stats?model=$M", "/sessions?model=$M", "/server_info?model=$M", "/props?model=$M",
  "/load?model=$M", "/version?model=$M", "/graph?model=$M", "/history?model=$M")
# Engine admin routes: [method, path, body, statuses that mean "admitted"].
$engineAdmin = @(
  @('GET', "/diagnostics?model=$M", '', @(200)),
  @('GET', "/admin/control-vector?model=$M", '', @(404)),          # admitted: no vector in this model
  @('POST', "/admin/control-vector?model=$M", $cvBody, @(404)),
  @('POST', '/reset_prefix_cache', $modelBody, @(200, 501)),
  @('DELETE', "/sessions/lan-matrix-none?model=$M", '', @(200, 404)))
# Fleet admin and history, read-only ones: [method, path, admitted].
$fleetRead = @(@('GET', '/admin/fleet', @(200)), @('GET', '/history', @(200)))

function Run-KeyOff {
  $mode = 'off'
  # Cold start: a guarded route must not wake the model for a stranger; inference may.
  Cell $mode 'cold' 'lan' 'GET' "/diagnostics?model=$M" -Expect 403, 409 -Policy 'a guarded route never wakes a cold model for a LAN caller (the front answers 409 model_not_open before the engine guard; 403 once loaded)'
  Cell $mode 'cold' 'lan' 'GET' '/v1/models' -Expect 200 -MustHave "`"id`":`"$M`"[^}]*`"state`":`"cold`"" -Policy 'the refused guarded GET left the model cold'
  foreach ($c in @('lan', 'local')) { foreach ($p in $pingModel) { Cell $mode 'cold' $c $p[0] $p[1] -Expect 503 -MustHave '"status":"cold"' -Policy '/ping naming a cold model reports its state and does not load it, as /health?model= (ping-state)' } }
  Cell $mode 'cold' 'lan' 'GET' '/v1/models' -Expect 200 -MustHave "`"id`":`"$M`"[^}]*`"state`":`"cold`"" -Policy 'the /ping probes left the model cold (ping-state)'
  Cell $mode 'cold' 'lan' 'POST' '/v1/chat/completions' -Body $chatBody -Expect 200 -Policy 'keyless LAN inference may demand (load) a cold model'
  foreach ($c in @('lan', 'local')) { foreach ($p in $public) { Cell $mode 'public' $c $p[0] $p[1] -Expect 200 -Policy 'public page or probe' } }
  Cell $mode 'public' 'lan' 'GET' '/ping' -Expect 200 -MustHave '"models"' -Policy 'bare /ping on a fleet of several models: the process liveness, as /health (F4, fixed)'
  foreach ($c in @('lan', 'xorigin', 'local')) { foreach ($p in $inference) { Cell $mode 'inference' $c $p[0] $p[1] -Body $p[2] -Expect 200 -Policy 'keyless: inference open to anyone who reaches the port' } }
  foreach ($c in @('lan', 'local')) { foreach ($p in $telemetry) { Cell $mode 'telemetry' $c 'GET' $p -Expect 200 -Policy 'keyless: operational telemetry open to anyone who reaches the port' } }
  # What strangers see of this machine's paths.
  Cell $mode 'exposure' 'lan' 'GET' '/v1/models' -Expect 200 -MustNotHave '"root":"[A-Za-z]:' -Policy 'model card root (a path) is the operator''s only'
  Cell $mode 'exposure' 'local' 'GET' '/v1/models' -Expect 200 -MustHave '"root":"[A-Za-z]:' -Policy 'the operator sees the model path'
  Cell $mode 'exposure' 'lan' 'GET' "/server_info?model=$M" -Expect 200 -MustHave '"container":"[^"/:]+"' -MustNotHave '"container":"[A-Za-z]:' -Policy 'engine.container cut to its file name for strangers, no path (finding F1; fixed on pm/lan-matrix)'
  Cell $mode 'exposure' 'local' 'GET' "/server_info?model=$M" -Expect 200 -MustHave '"container":"[A-Za-z]:/' -Policy 'the operator sees the container path'
  Cell $mode 'exposure' 'lan' 'GET' "/props?model=$M" -Expect 200 -MustHave '"model_path":""' -Policy '/props model_path empty for strangers'
  Cell $mode 'exposure' 'local' 'GET' "/props?model=$M" -Expect 200 -MustHave '"model_path":"[A-Za-z]:/' -Policy 'the operator sees model_path'
  Cell $mode 'exposure' 'lan' 'GET' "/stats?model=$M" -Expect 200 -Policy 'live request and engine figures open (operational telemetry)'
  Cell $mode 'exposure' 'lan' 'GET' '/metrics' -Expect 200 -MustHave 'radiance_' -Policy 'Prometheus metrics open'
  # Engine admin: operator and XFF admitted, LAN and foreign Origin refused.
  foreach ($c in @('lan', 'xorigin', 'local', 'xff')) {
    foreach ($p in $engineAdmin) {
      $exp = if ($c -in 'lan', 'xorigin') { @(403) } else { $p[3] }
      Cell $mode 'engine-admin' $c $p[0] $p[1] -Body $p[2] -Expect $exp -Policy 'keyless: local operator only (loopback socket, no foreign Origin, loopback Host)'
    }
  }
  # Fleet admin and fleet history.
  foreach ($c in @('lan', 'xorigin', 'local', 'xff')) {
    foreach ($p in $fleetRead) {
      $exp = if ($c -in 'lan', 'xorigin') { @(403) } else { $p[2] }
      Cell $mode 'fleet-admin' $c $p[0] $p[1] -Expect $exp -Policy 'keyless: fleet overview and fleet history local operator only'
    }
  }
  foreach ($c in @('lan', 'xorigin')) {
    Cell $mode 'sandbox' $c 'POST' "/admin/models/$M/load?wait=0&hold=1" -Expect 403 -Policy 'Sandbox Start refused off the box'
    Cell $mode 'sandbox' $c 'POST' "/admin/models/$M/unload" -Expect 403 -Policy 'Sandbox Stop refused off the box'
    Cell $mode 'fleet-admin' $c 'POST' '/admin/reload' -Expect 403 -Policy 'reload refused off the box'
  }
  Cell $mode 'sandbox' 'local' 'POST' "/admin/models/$M/unload" -Expect 200 -Policy 'Sandbox Stop from the box'
  Cell $mode 'sandbox' 'local' 'POST' "/admin/models/$M/load?wait=0&hold=1" -Expect 200, 202 -Policy 'Sandbox Start from the box (hold)'
  Cell $mode 'sandbox' 'local' 'POST' "/admin/models/$M/load" -Expect 200 -TimeoutS 300 -Policy 'load waits for the model'
  Cell $mode 'sandbox' 'xff' 'POST' "/admin/models/$M/unload" -Expect 200 -Policy 'X-Forwarded-For does not change the socket verdict'
  Cell $mode 'fleet-admin' 'local' 'POST' '/admin/reload' -Expect 200 -Policy 'reload from the box (unchanged file)'
  # A model whose load failed: the reason names a path; strangers should learn only that it failed.
  Cell $mode 'failed-model' 'local' 'POST' '/v1/chat/completions' -Body $brokenBody -Expect 503 -MustHave 'could not be loaded: ' -Policy 'the operator is told why the load failed'
  Cell $mode 'failed-model' 'lan' 'POST' '/v1/chat/completions' -Body $brokenBody -Expect 503 -MustNotHave 'io error' -Policy 'a stranger learns the model failed, never why (front.cpp lease)'
  Cell $mode 'failed-model' 'lan' 'GET' '/health?model=broken-entry' -Expect 503 -MustHave '"status":"failed"' -MustNotHave 'io error|"error"' -Policy 'the per-model health probe tells a stranger the state, never why (F3, fixed)'
  Cell $mode 'failed-model' 'lan' 'GET' '/v1/models' -Expect 200 -MustHave 'broken-entry' -MustNotHave '"error":' -Policy 'nor does the public model list (F3, fixed)'
  Cell $mode 'failed-model' 'local' 'GET' '/health?model=broken-entry' -Expect 503 -MustHave 'io error' -Policy 'the operator reads the reason on the probe'
  # The model history keeps working for strangers after all that (it repeats /stats).
  Cell $mode 'history' 'lan' 'GET' "/history?model=$M" -Expect 200 -MustHave '"live"' -Policy 'model history open as /stats'
  Cell $mode 'history' 'lan' 'GET' '/history' -Expect 403 -Policy 'fleet history guarded as /admin/fleet'
}

function Run-KeyOn {
  $mode = 'on'
  # No key and a wrong key: public pages only.
  foreach ($c in @('lan', 'local', 'lan+wrong', 'local+wrong')) {
    foreach ($p in $publicNoPing) { Cell $mode 'public' $c $p[0] $p[1] -Expect 200 -Policy 'exempt from the key: probes, dashboard page, terms' }
    # The model is still cold here: no key holder has asked for it yet.
    foreach ($p in $pingModel) { Cell $mode 'cold' $c $p[0] $p[1] -Expect 503 -MustHave '^\{"status":"cold"\}\s*$' -Policy 'no key: /ping naming a cold model gives its state alone and does not load it (ping-state)' }
    foreach ($p in $inference) { Cell $mode 'inference' $c $p[0] $p[1] -Body $p[2] -Expect 401 -Policy 'key required' }
    foreach ($p in $telemetry) { Cell $mode 'telemetry' $c 'GET' $p -Expect 401 -Policy 'key required' }
    foreach ($p in $engineAdmin) { Cell $mode 'engine-admin' $c $p[0] $p[1] -Body $p[2] -Expect 401 -Policy 'key required, even from the box' }
    foreach ($p in $fleetRead) { Cell $mode 'fleet-admin' $c $p[0] $p[1] -Expect 401 -Policy 'key required, even from the box' }
    Cell $mode 'sandbox' $c 'POST' "/admin/models/$M/load?wait=0&hold=1" -Expect 401 -Policy 'key required'
  }
  Cell $mode 'cold' 'local+bearer' 'GET' '/v1/models' -Expect 200 -MustHave "`"id`":`"$M`"[^}]*`"state`":`"cold`"" -Policy 'the keyless /ping probes left the model cold (ping-state)'
  Cell $mode 'cold' 'lan+bearer' 'GET' "/ping?model=$M" -Expect 503 -MustHave "`"status`":`"cold`"" -Policy 'key holder: /ping naming a cold model reports it, with the model name, and does not load it (ping-state)'
  # A valid key, either header, from the box or the LAN: everything, admin included.
  foreach ($c in @('lan+bearer', 'lan+xkey', 'local+bearer', 'xorigin+bearer')) {
    Cell $mode 'inference' $c 'POST' '/v1/chat/completions' -Body $chatBody -Expect 200 -TimeoutS 300 -Policy 'key holder: inference'
    Cell $mode 'inference' $c 'POST' '/tokenize' -Body $tokBody -Expect 200 -Policy 'key holder: inference'
    foreach ($p in $telemetry) { Cell $mode 'telemetry' $c 'GET' $p -Expect 200 -Policy 'key holder: telemetry' }
    foreach ($p in $engineAdmin) { Cell $mode 'engine-admin' $c $p[0] $p[1] -Body $p[2] -Expect $p[3] -Policy 'a valid inference key also authorizes admin, from any Host' }
    foreach ($p in $fleetRead) { Cell $mode 'fleet-admin' $c $p[0] $p[1] -Expect $p[2] -Policy 'a valid inference key also authorizes the fleet admin' }
  }
  Cell $mode 'exposure' 'lan+bearer' 'GET' '/v1/models' -Expect 200 -MustHave '"root":"[A-Za-z]:' -Policy 'with a key, the model path rides with the key'
  # The key holder's inference above loaded the model: the probe now gives its answer.
  foreach ($c in @('lan', 'lan+wrong')) { foreach ($p in $pingModel) { Cell $mode 'public' $c $p[0] $p[1] -Expect 200 -MustHave '^\{"status":"ok"\}\s*$' -Policy 'no key: /ping naming a loaded model gives the liveness alone' } }
  Cell $mode 'sandbox' 'lan+bearer' 'POST' "/admin/models/$M/unload" -Expect 200 -Policy 'key holder off the box: Sandbox Stop'
  Cell $mode 'sandbox' 'lan+bearer' 'POST' "/admin/models/$M/load?wait=0&hold=1" -Expect 200, 202 -Policy 'key holder off the box: Sandbox Start'
  Cell $mode 'sandbox' 'lan+bearer' 'POST' "/admin/models/$M/load" -Expect 200 -TimeoutS 300 -Policy 'load waits'
  Cell $mode 'fleet-admin' 'lan+bearer' 'POST' '/admin/reload' -Expect 200 -Policy 'key holder off the box: reload'
  Cell $mode 'failed-model' 'lan+bearer' 'POST' '/v1/chat/completions' -Body $brokenBody -Expect 503 -MustHave 'could not be loaded: ' -Policy 'a key holder is an operator: told why'
  Cell $mode 'failed-model' 'lan' 'GET' '/health?model=broken-entry' -Expect 503 -MustHave '^\{"status":"failed"\}\s*$' -Policy 'no key: the key-exempt probe gives the state alone (F3, fixed)'
  Cell $mode 'failed-model' 'lan' 'GET' '/health' -Expect 200 -MustHave '^\{"status":"ok"\}\s*$' -Policy 'no key: /health names no models (F3, fixed)'
  Cell $mode 'failed-model' 'lan' 'GET' '/ping' -Expect 200 -MustHave '^\{"status":"ok"\}\s*$' -Policy 'no key: bare /ping on several models is the liveness alone (F3, F4, fixed)'
  Cell $mode 'failed-model' 'lan+bearer' 'GET' '/health' -Expect 200 -MustHave '"broken-entry":"failed"' -Policy 'key holder: /health names every model and state'
  Cell $mode 'failed-model' 'lan+bearer' 'GET' '/health?model=broken-entry' -Expect 503 -MustHave 'io error' -Policy 'key holder: the probe carries the reason'
}

# ---------------------------------------------------------------- browser
function Find-Browser {
  foreach ($p in @("${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
                   "$env:ProgramFiles\Google\Chrome\Application\chrome.exe")) { if ($p -and (Test-Path $p)) { return $p } }
  return $null
}
function Dump-Dom([string]$mode, [string]$label, [string]$url, [string]$browser) {
  $prof = Join-Path $Out "browser-profile-$mode-$label"
  $dom = Join-Path $Out "dom-key-$mode-$label.html"
  $err = Join-Path $Out "dom-key-$mode-$label.err.txt"
  $bargs = @('--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--disable-extensions',
            '--disable-background-networking', '--disable-sync', '--disable-component-update',
            "--user-data-dir=$prof", "--host-resolver-rules=MAP lan.test 127.0.0.1", '--virtual-time-budget=8000', '--dump-dom', $url)
  $p = Start-Process -FilePath $browser -ArgumentList ($bargs | ForEach-Object { if ($_ -match ' ') { "`"$_`"" } else { $_ } }) `
       -RedirectStandardOutput $dom -RedirectStandardError $err -WindowStyle Hidden -PassThru
  if (-not $p.WaitForExit(60000)) { Write-Host "  browser did not finish in 60 s for $url"; return '' }
  # The launcher can exit before the process that writes the DOM lets go of the file: retry the read.
  for ($i = 0; $i -lt 60; $i++) {
    # The page's own inline script carries every string it can draw: drop scripts and styles, so a
    # check reads what was rendered.
    try { if (Test-Path $dom) { return [regex]::Replace([IO.File]::ReadAllText($dom, [Text.Encoding]::UTF8), '(?is)<script\b.*?</script>|<style\b.*?</style>', '') } else { return '' } }
    catch [IO.IOException] { Start-Sleep -Milliseconds 500 }
  }
  Write-Host "  $dom stayed locked"; return ''
}
function Browser-Cells([string]$mode) {
  $b = if ($NoBrowser) { $null } else { Find-Browser }
  if (-not $b) {
    $script:rows.Add([pscustomobject]@{ id = '-'; key = $mode; group = 'browser'; caller = '-'; request = 'headless browser'; expected = '-'; actual = '-'; bytes = 0; verdict = 'not run'; note = 'no browser or -NoBrowser'; policy = ''; body = '' })
    return
  }
  $local = Dump-Dom $mode 'local' "http://127.0.0.1:$Port/#/m/$M/sandbox" $b
  $lan = Dump-Dom $mode 'lan' "http://lan.test:$Port/#/m/$M/sandbox" $b
  $refusedNote = 'Start and Stop work only from a browser'
  # The Start button, enabled (no disabled="" attribute) or disabled. The model is cold when the
  # browser runs (the matrix unloaded it), so only the guard can disable Start.
  $startOn  = '(?s)<button(?![^>]*\sdisabled="")[^>]*>(?:(?!</button>).)*Start</button>'
  $startOff = '(?s)<button(?=[^>]*\sdisabled="")[^>]*>(?:(?!</button>).)*Start</button>'
  $checks = if ($mode -eq 'off') {
    @(@('local', $local, $startOn, $refusedNote, 'Sandbox from the box: Start enabled, no local-only note'),
      @('lan.test', $lan, $startOff, '', 'Sandbox from a LAN name: Start disabled'),
      @('lan.test', $lan, $refusedNote, '', 'Sandbox from a LAN name: the local-only note shown'),
      @('lan.test', $lan, '(?s)Send.*Clear', '', 'Sandbox from a LAN name: the chat is still drawn'),
      @('lan.test', $lan, "$M", '', 'the remote page still lists the model (public /v1/models)'))
  } else {
    @(@('local', $local, '(?i)api key', '', 'keyed: the page asks for the key'),
      @('lan.test', $lan, '(?i)api key', '', 'keyed: the page asks for the key from a LAN name too'))
  }
  foreach ($c in $checks) {
    $script:n++
    $id = '{0}{1:000}' -f ($(if ($mode -eq 'off') { 'A' } else { 'B' })), $script:n
    $ok = $c[1].Length -gt 0 -and $c[1] -match $c[2] -and (-not $c[3] -or $c[1] -notmatch [regex]::Escape($c[3]))
    $script:rows.Add([pscustomobject]@{ id = $id; key = $mode; group = 'browser'; caller = $c[0]; request = 'headless --dump-dom #/m/' + $M + '/sandbox'
      expected = "/$($c[2])/" + $(if ($c[3]) { " -/$($c[3])/" } else { '' }); actual = "dom $($c[1].Length) chars"; bytes = $c[1].Length
      verdict = $(if ($ok) { 'ok' } else { 'MISMATCH' }); note = ''; policy = $c[4]; body = '' })
    "{0} browser {1,-10} {2} {3}" -f $id, $c[0], $c[4], $(if ($ok) { 'ok' } else { 'MISMATCH' })
  }
}

# ---------------------------------------------------------------- config: where a profile listens
# Read with radiance --check (it reads the file and serves nothing) and serve.ps1 -Check, never by
# binding: what host, port and key a profile resolves to. Omitting host uses the engine's loopback
# default. An explicit -Listen still overrides the profile.
function Config-Cells {
  $exe = Join-Path $Package 'bin\radiance.exe'
  $noHost = Join-Path $Out 'nohost.yaml'
  [IO.File]::WriteAllText($noHost, ("# lan-matrix: a hand-written profile with no server: block`nmodels:`n  ${ModelName}:`n    model: $($Model -replace '\\', '/')`n    max-model-len: 4096`n"), $utf8)
  $cases = @(
    @('radiance --check', $yaml, 'server host 127\.0\.0\.1 \(server\.host\), port \d+ \(server\.port\), api key none', 'the matrix profile (host: 127.0.0.1, as the shipped ones): loopback'),
    @('radiance --check', $noHost, 'server host 127\.0\.0\.1 \(default\), port 8000 \(default\), api key none', 'no host: loopback, port 8000, no key'),
    @('serve.ps1 -Cpu -Check', $noHost, 'server host 127\.0\.0\.1 \(default\), port 8000 \(default\), api key none', 'the launcher reports the loopback default'),
    @('serve.ps1 -Cpu -Check -Listen 0.0.0.0', $yaml, 'server host 0\.0\.0\.0 \(environment \(BRILLIANCE_SERVER_HOST\)\), port \d+ \(server\.port\), api key none', 'an explicit -Listen overrides the profile', '0.0.0.0'))
  $saved = @{}
  foreach ($k in 'HIP_VISIBLE_DEVICES', 'BRILLIANCE_DEVICE', 'TEMP', 'TMP',
                'BRILLIANCE_SERVER_HOST', 'BRILLIANCE_SERVER_PORT', 'BRILLIANCE_SERVER_API_KEY') {
    $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process')
  }
  $env:HIP_VISIBLE_DEVICES = '-1'; $env:BRILLIANCE_DEVICE = 'host'; $env:TEMP = $tmp; $env:TMP = $tmp
  $env:BRILLIANCE_SERVER_HOST = $null; $env:BRILLIANCE_SERVER_PORT = $null; $env:BRILLIANCE_SERVER_API_KEY = $null
  try {
    foreach ($c in $cases) {
      $ErrorActionPreference = 'Continue'
      $o = if ($c[0] -like 'radiance*') { & $exe --config $c[1] --check 2>&1 | ForEach-Object { "$_" } }
           else {
             $launcherArgs = @($c[1], '-Cpu', '-Check')
             if ($c.Count -gt 4) { $launcherArgs += @('-Listen', $c[4]) }
             & powershell -NoProfile -ExecutionPolicy Bypass -File $serve @launcherArgs 2>&1 | ForEach-Object { "$_" }
           }
      $rc = $LASTEXITCODE
      $ErrorActionPreference = 'Stop'
      $line = @($o | Where-Object { $_ -match 'server host ' }) | Select-Object -First 1
      $ok = $rc -eq 0 -and "$line" -match $c[2]
      $script:n++
      $id = 'C{0:000}' -f $script:n
      $script:rows.Add([pscustomobject]@{ id = $id; key = '-'; group = 'config'; caller = '-'; request = "$($c[0]) $([IO.Path]::GetFileName($c[1]))"
        expected = "/$($c[2])/"; actual = (("$line" -replace '^.*?: server host', 'server host')); bytes = 0
        verdict = $(if ($ok) { 'ok' } else { 'MISMATCH' }); note = $(if ($rc -ne 0) { "exit $rc" }); policy = $c[3]; body = '' })
      "{0} config {1,-28} {2}" -f $id, $c[0], $(if ($ok) { 'ok' } else { "MISMATCH (exit ${rc}): $line" })
    }
  } finally {
    foreach ($k in $saved.Keys) { if ($null -eq $saved[$k]) { Remove-Item "env:$k" -ErrorAction SilentlyContinue } else { Set-Item "env:$k" $saved[$k] } }
  }
}

# ---------------------------------------------------------------- run
Config-Cells
$modes = if ($Key -eq 'both') { @('off', 'on') } elseif ($Key -eq 'none') { @() } else { @($Key) }
foreach ($mode in $modes) {
  Write-Host "== key $mode =="
  $procId = Start-Server $mode
  try {
    if ($mode -eq 'off') { Run-KeyOff } else { Run-KeyOn }
    Browser-Cells $mode
  } finally {
    Stop-Server $procId
  }
}

# ---------------------------------------------------------------- report
$csv = Join-Path $Out 'matrix.csv'
$script:rows | Export-Csv -NoTypeInformation -Encoding UTF8 $csv
$md = New-Object System.Collections.Generic.List[string]
$md.Add('| id | key | group | caller | request | expected | actual | verdict | policy |')
$md.Add('|---|---|---|---|---|---|---|---|---|')
foreach ($r in $script:rows) {
  $md.Add(('| {0} | {1} | {2} | {3} | `{4}` | {5} | {6} | {7}{8} | {9} |' -f $r.id, $r.key, $r.group, $r.caller, $r.request, ($r.expected -replace '\|', '\|'), $r.actual, $r.verdict, $(if ($r.note) { ' (' + $r.note + ')' } else { '' }), $r.policy))
}
[IO.File]::WriteAllText((Join-Path $Out 'matrix.md'), (($md -join "`n") + "`n"), $utf8)
$bad = @($script:rows | Where-Object { $_.verdict -eq 'MISMATCH' })
Write-Host "lan-matrix: $($script:rows.Count) cells, $($bad.Count) mismatches; $Out\matrix.md"
if ($bad.Count) { exit 1 } else { exit 0 }
