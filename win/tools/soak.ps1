#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: run win\tools\qualify_api.py against a live server N times in a row, stopping at the
  first failing pass. A hang shows as a pass that does not end; the server's stall watch logs it.
.EXAMPLE
  powershell -File win\tools\soak.ps1 -N 12 -Out out\soak.txt -Model flashnext
  powershell -File win\tools\soak.ps1 -N 6 -Out soak.txt -Model flashnext -Extra '--only','tools_typed,issue4'
  powershell -File win\tools\soak.ps1 -Hours 8 -KeepGoing -Out out\soak\soak.txt -Model qwen3.8-27b -Port 18500
.NOTES
  Output: one "=== iteration" line before and after each pass, the harness's lines between, and
  every check as JSON lines in <Out>.jsonl. cmd's redirection keeps the harness's UTF-8 as it is
  (Windows PowerShell 5.1's own would re-encode it as UTF-16). Stop a soak by ending this script and
  its python child; neither touches a GPU.

  -Hours runs passes until that much time has gone by (N is ignored); a pass is never cut short.
  -KeepGoing goes on after a failed pass as long as the server's /health answers, so a long soak
  records every failure rather than its first; a server that stops answering ends the soak either
  way. win\tools\soak-sample.ps1 records memory and driver events beside it.
#>
param(
  [int]$N = 6,
  [Parameter(Mandatory)][string]$Out,
  [string]$Model = 'flashnext',
  [int]$Concurrency = 4,
  [int]$Port = 8100,
  [int]$Stress = 40,
  [int]$Issue4Rounds = 10,
  [string[]]$Extra = @(),
  [double]$Hours = 0,
  [switch]$KeepGoing
)
$r = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$py = Join-Path $r '.toolchain\python\python.exe'
$Out = [IO.Path]::GetFullPath($Out)
$more = ($Extra | ForEach-Object { "`"$_`"" }) -join ' '
$until = if ($Hours -gt 0) { (Get-Date).AddHours($Hours) } else { $null }
$failed = 0
for ($i = 1; ; $i++) {
  if ($until) { if ((Get-Date) -ge $until) { break } } elseif ($i -gt $N) { break }
  $of = if ($until) { "until $($until.ToString('h:mm tt'))" } else { "of $N" }
  "=== iteration $i $of, $(Get-Date -Format 'h:mm:ss tt')" | Out-File -Append -Encoding ascii $Out
  cmd /c "`"$py`" -I `"$r\win\tools\qualify_api.py`" --url http://127.0.0.1:$Port --model $Model --out `"$Out.jsonl`" --stress $Stress --concurrency $Concurrency --issue4-rounds $Issue4Rounds $more >> `"$Out`" 2>&1"
  $rc = $LASTEXITCODE
  "=== iteration $i exit $rc, $(Get-Date -Format 'h:mm:ss tt')" | Out-File -Append -Encoding ascii $Out
  if ($rc -ne 0) {
    $failed++
    $up = $false
    try { $up = (Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 15).StatusCode -eq 200 } catch { }
    if (-not $KeepGoing -or -not $up) {
      "=== stopped after iteration $i (server healthy: $up)" | Out-File -Append -Encoding ascii $Out
      break
    }
  }
}
"=== done: $($i - 1) pass(es), $failed failed, $(Get-Date -Format 'h:mm:ss tt')" | Out-File -Append -Encoding ascii $Out
