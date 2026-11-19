#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: sample a soaking server's memory once a minute, and at the end list the display
  driver's warnings and errors from the System log (recording failures, memory growth
  and driver events).
.EXAMPLE
  powershell -File win\tools\soak-sample.ps1 -ProcessId 1234 -Port 18500 -Out out\soak\samples.csv -Hours 8.2
.NOTES
  One CSV row a sample: time; the process's private bytes, working set, handles and threads; the
  cards' free VRAM and the host memory the process pins, from /admin/fleet; /health's status. It
  ends when the process does or the hours run out, then writes <Out>.events.txt: every System-log
  event since its start from the display driver (amdkmdag, amduw23g) or Windows' display stack
  (Display, a TDR's 4101), warnings and worse. Reads only; touches no GPU.
#>
param(
  [Parameter(Mandatory)][int]$ProcessId,
  [int]$Port = 8000,
  [Parameter(Mandatory)][string]$Out,
  [double]$Hours = 8,
  [int]$EveryS = 60
)
$Out = [IO.Path]::GetFullPath($Out)
$start = Get-Date
$until = $start.AddHours($Hours)
if (-not (Test-Path $Out)) {
  'time,private_mib,working_set_mib,handles,threads,cards_free_mib,pinned_mib,health' | Out-File -Encoding ascii $Out
}
while ((Get-Date) -lt $until) {
  $p = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
  if (-not $p) { break }
  $free = ''; $held = ''; $health = ''
  try {
    $f = (Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/admin/fleet" -TimeoutSec 15).Content | ConvertFrom-Json
    $free = [math]::Round((($f.cards | Measure-Object -Property free_bytes -Sum).Sum) / 1MB)
    $held = [math]::Round($f.host.held_bytes / 1MB)
  } catch { }
  try { $health = (Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 15).StatusCode } catch { $health = 'down' }
  '{0},{1},{2},{3},{4},{5},{6},{7}' -f (Get-Date -Format 's'), [math]::Round($p.PrivateMemorySize64 / 1MB),
    [math]::Round($p.WorkingSet64 / 1MB), $p.HandleCount, $p.Threads.Count, $free, $held, $health |
    Out-File -Append -Encoding ascii $Out
  Start-Sleep -Seconds $EveryS
}
$ev = @(Get-WinEvent -FilterHashtable @{ LogName = 'System'; StartTime = $start; Level = 1, 2, 3 } -ErrorAction SilentlyContinue |
        Where-Object { $_.ProviderName -match 'amdkmdag|amduw23g|amdwddmg|Display' -or $_.Id -eq 4101 })
$lines = @("display driver events, warnings and worse, $($start.ToString('s')) to $((Get-Date).ToString('s')): $($ev.Count)")
foreach ($e in $ev) { $lines += "$($e.TimeCreated.ToString('s'))  $($e.ProviderName)  id $($e.Id)  $($e.LevelDisplayName)  $(($e.Message -split "`n")[0])" }
$lines | Out-File -Encoding utf8 "$Out.events.txt"
