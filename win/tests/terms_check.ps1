#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: the terms-of-use page (GET /terms) on a running server: the template, an operator's
  file, the dashboard's link, and the startup refusal of a file that cannot be served.
.DESCRIPTION
  1. Refusal, no card: a profile whose `terms-file` names a missing file stops the start (exit 2)
     with the reason, before any card or model is touched (HIP_VISIBLE_DEVICES=-1).
  2. Template: one model on one card (-Devices), behind an API key. GET /terms answers the
     template without the key, POST /terms needs it, and GET /dashboard carries the link.
  3. Operator file (-Operator): the same with `terms-file` naming a UTF-8 file; /terms answers it.
  Each server starts through win\tools\serve-bg.ps1 (a hidden console of its own) and is stopped
  with win\tools\stop-server.ps1 (Ctrl-Break, the drain); nothing is force-killed. Exit 0 when
  every check holds, 1 otherwise, 3 when a server did not stop (left for a person).
.EXAMPLE
  powershell -File win\tests\terms_check.ps1 -Bin build-win\hip\bin -Devices 0 `
    -Model D:\models\hf\google\embeddinggemma-2\embeddinggemma2-bf16.rad -Work $env:TEMP\terms-check
.EXAMPLE
  # No card: a host build on the CPU kernels.
  powershell -File win\tests\terms_check.ps1 -Bin build-win\host\bin -Devices -1 -Kernels libavx,libref `
    -Model D:\models\hf\google\embeddinggemma-2\embeddinggemma2-bf16.rad -Work $env:TEMP\terms-check
#>
param(
  [Parameter(Mandatory)][string]$Bin,
  [Parameter(Mandatory)][string]$Model,
  [Parameter(Mandatory)][string]$Work,
  [string]$Devices = '0',
  [string]$Kernels = '',
  [int]$Port = 18391,
  [int]$ReadyS = 600,
  [switch]$Operator
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Bin = (Resolve-Path -LiteralPath $Bin).Path
$exe = Join-Path $Bin 'radiance.exe'
New-Item -ItemType Directory -Force $Work | Out-Null
$Work = (Resolve-Path -LiteralPath $Work).Path
$utf8 = New-Object System.Text.UTF8Encoding($false)
$script:failed = 0
$script:checks = 0
function Check([bool]$ok, [string]$what) {
  $script:checks++
  if ($ok) { Write-Host "  ok   $what" } else { $script:failed++; Write-Host "  FAIL $what" }
}
function Profile([string]$name, [string]$termsLine) {
  $p = Join-Path $Work "$name.yaml"
  $m = $Model -replace '\\', '/'
  $y = "server:`n  host: 127.0.0.1`n  port: $Port`n  api-key: terms-check-key`n"
  if ($Kernels) { $y += "  kernels: $Kernels`n" }
  if ($termsLine) { $y += "  $termsLine`n" }
  $y += "models:`n  terms-check:`n    model: $m`n    max-num-seqs: 1`n    max-model-len: 4096`n"
  [IO.File]::WriteAllText($p, $y, $utf8)
  return $p
}
# One HTTP exchange; status, headers and body, with no exception for a 4xx.
function Http([string]$method, [string]$path, [string]$key = '') {
  $req = [Net.HttpWebRequest]::Create("http://127.0.0.1:$Port$path")
  $req.Method = $method
  $req.Timeout = 30000
  if ($key) { $req.Headers.Add('Authorization', "Bearer $key") }
  if ($method -eq 'POST') { $req.ContentLength = 0 }
  try { $res = $req.GetResponse() } catch [Net.WebException] { $res = $_.Exception.Response; if (-not $res) { throw } }
  $sr = New-Object IO.StreamReader($res.GetResponseStream(), [Text.Encoding]::UTF8)
  $body = $sr.ReadToEnd(); $sr.Close()
  $o = [pscustomobject]@{ Status = [int]$res.StatusCode; Type = $res.ContentType; Body = $body }
  $res.Close()
  return $o
}
# The server's console holds its log open for writing; read it shared.
function ReadLog([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) { return '' }
  $fs = [IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
  $sr = New-Object IO.StreamReader($fs, [Text.Encoding]::UTF8)
  try { return $sr.ReadToEnd() } finally { $sr.Close() }
}
function Serve([string]$name, [string]$cfg) {
  $log = Join-Path $Work "$name.log"
  $savedDev = $env:HIP_VISIBLE_DEVICES
  $env:HIP_VISIBLE_DEVICES = $Devices
  try {
    $id = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo 'win\tools\serve-bg.ps1') -Bin $Bin -Config $cfg -Log $log
  } finally { $env:HIP_VISIBLE_DEVICES = $savedDev }
  $id = [int]("$id".Trim())
  $deadline = (Get-Date).AddSeconds($ReadyS)
  $ready = $false
  while ((Get-Date) -lt $deadline) {
    $t = ReadLog $log
    if ($t -match 'radiance exit status') { break }
    # /health answers 200 once the model loaded at startup is open (it needs no key).
    if ($t -match 'listening on http://127\.0\.0\.1:') {
      try { if ((Http 'GET' '/health').Status -eq 200) { $ready = $true; break } } catch { }
    }
    Start-Sleep -Milliseconds 500
  }
  return [pscustomobject]@{ Id = $id; Log = $log; Ready = $ready }
}
function Stop([object]$s) {
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo 'win\tools\stop-server.ps1') -ProcessId $s.Id -WaitS 180 | Out-Host
  if ($LASTEXITCODE -eq 3) { Write-Host "  server $($s.Id) did not stop; left running for a person"; exit 3 }
}
$template = [IO.File]::ReadAllText((Join-Path $repo 'win\TERMS-OF-USE.md'), [Text.Encoding]::UTF8)

Write-Host "1. A missing terms-file stops the start (no card)"
$missing = Join-Path $Work 'no-such-terms.md'
$cfg = Profile 'refused' ("terms-file: " + ($missing -replace '\\', '/'))
$savedDev = $env:HIP_VISIBLE_DEVICES
$env:HIP_VISIBLE_DEVICES = '-1'
$ErrorActionPreference = 'Continue'
$out = (& $exe --config $cfg 2>&1 | ForEach-Object { "$_" }) -join "`n"
$rc = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
$env:HIP_VISIBLE_DEVICES = $savedDev
Check ($rc -eq 2) "exit status 2 (was $rc)"
Check ($out.Contains('server.terms-file') -and $out.Contains('no such file')) "the reason names the key and the file: $(($out -split "`n" | Select-String 'terms-file' | Select-Object -First 1))"
Check (-not $out.Contains('listening on')) 'nothing listened'

Write-Host "2. The template, with one model loaded (HIP_VISIBLE_DEVICES=$Devices)"
$s = Serve 'template' (Profile 'template' '')
Check $s.Ready "the server listened and the model loaded ($($s.Log))"
if ($s.Ready) {
  $log = ReadLog $s.Log
  Check ($log.Contains('terms of use at /terms: the built-in template')) 'the startup log names the page'
  $r = Http 'GET' '/terms'
  Check ($r.Status -eq 200) "GET /terms without the key: $($r.Status)"
  Check ($r.Type -like 'text/plain*utf-8*') "as plain text: $($r.Type)"
  Check ($r.Body -eq $template) 'the body is win\TERMS-OF-USE.md, byte for byte'
  Check ((Http 'POST' '/terms').Status -eq 401) 'POST /terms needs the key'
  $idx = Http 'GET' '/terms/licenses'
  $index = [IO.File]::ReadAllText((Join-Path $repo 'win\model-licenses\index.md'), [Text.Encoding]::UTF8)
  Check ($idx.Status -eq 200 -and $idx.Body -eq $index) "GET /terms/licenses without the key: $($idx.Status), the shipped index"
  foreach ($id in 'ltx-2.x-community-license', 'creativeml-openrail-pp-m', 'apache-2.0') {
    $want = [IO.File]::ReadAllText((Join-Path $repo "win\model-licenses\$id.txt"), [Text.Encoding]::UTF8)
    $l = Http 'GET' "/terms/licenses/$id"
    Check ($l.Status -eq 200 -and $l.Body -eq $want) "GET /terms/licenses/$id without the key: $($l.Status), the shipped text"
  }
  Check ((Http 'GET' '/terms/licenses/nope').Status -eq 401) 'an unknown licence id needs the key'
  Check ((Http 'GET' '/stats').Status -eq 401) 'GET /stats still needs the key'
  $d = Http 'GET' '/dashboard'
  Check ($d.Status -eq 200 -and $d.Body.Contains('<a href="/terms"')) "GET /dashboard links /terms ($($d.Status))"
  $v = Http 'GET' '/v1/models' 'terms-check-key'
  Check ($v.Status -eq 200 -and $v.Body.Contains('terms-check')) "the model answers /v1/models ($($v.Status))"
}
Stop $s

if ($Operator) {
  Write-Host "3. An operator's file"
  $own = Join-Path $Work 'our terms.html'
  [IO.File]::WriteAllText($own, "<h1>Our terms</h1><p>Caf$([char]0xE9) rules.</p>", $utf8)
  $s = Serve 'operator' (Profile 'operator' ("terms-file: '" + ($own -replace '\\', '/') + "'"))
  Check $s.Ready "the server listened and the model loaded ($($s.Log))"
  if ($s.Ready) {
    $r = Http 'GET' '/terms'
    Check ($r.Status -eq 200 -and $r.Type -like 'text/html*') "GET /terms: $($r.Status) $($r.Type)"
    Check ($r.Body -eq "<h1>Our terms</h1><p>Caf$([char]0xE9) rules.</p>") 'the body is the operator file'
  }
  Stop $s
}

Write-Host "$($script:checks) checks, $($script:failed) failed"
exit ([int]($script:failed -gt 0))
