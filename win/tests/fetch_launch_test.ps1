#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: libexec\hfget.ps1 and serve.ps1 -Fetch against a stand-in hub on loopback -- the plan
  printed before a transfer, a fresh destination, a warm cache that transfers nothing, a transfer
  cut short and resumed (by the next run, and by each try within a run, from the byte it reached),
  a repo of several containers that needs a choice, and the verified file served without an edit to
  the profile. No card, no network beyond 127.0.0.1, fixture bytes only.
.DESCRIPTION
  ctest runs it (win\tests\CMakeLists.txt) with the radiance.exe of the build tree:
    powershell -File win\tests\fetch_launch_test.ps1 -Exe build-win\hip\bin\radiance.exe
  The hub is win\tests\hub_stub.py (Python's standard library); with no Python the test is skipped
  (exit 77). radiance runs only as --check. Exit 0 when every check holds, 1 otherwise.
#>
param([Parameter(Mandatory)][string]$Exe)
$ErrorActionPreference = 'Stop'
$win = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$serve = Join-Path $win 'serve.ps1'
$hfget = Join-Path $win 'libexec\hfget.ps1'
$Exe = (Resolve-Path -LiteralPath $Exe).Path
$py = Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $py) { Write-Host 'fetch_launch_test: no python.exe for the stand-in hub; skipped'; exit 77 }
$script:failed = 0
$script:checks = 0
function Check([bool]$ok, [string]$what) {
  $script:checks++
  if (-not $ok) { $script:failed++; Write-Host "  FAIL $what" }
}
function Has([string]$text, [string]$part) { return $text.Contains($part) }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("brl-fetch-launch-test-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory $tmp | Out-Null
$hubProc = $null
$saved = @{}
foreach ($k in @('HF_ENDPOINT', 'HF_TOKEN', 'BRILLIANCE_HF_CACHE', 'HIP_VISIBLE_DEVICES', 'BRILLIANCE_SERVE_TEST_DEVICES')) { $saved[$k] = [Environment]::GetEnvironmentVariable($k) }
function Start-Hub([int]$drop = 0) {
  foreach ($f in @('port', 'log')) { Remove-Item -LiteralPath (Join-Path $tmp $f) -ErrorAction SilentlyContinue }
  $script:hubProc = Start-Process -FilePath $py.Source -WindowStyle Hidden -PassThru -ArgumentList @(
    '-I', "`"$(Join-Path $PSScriptRoot 'hub_stub.py')`"", '--root', "`"$(Join-Path $tmp 'hub')`"",
    '--port-file', "`"$(Join-Path $tmp 'port')`"", '--log', "`"$(Join-Path $tmp 'log')`"", '--drop', "$drop")
  for ($i = 0; $i -lt 100 -and -not (Test-Path -LiteralPath (Join-Path $tmp 'port')); $i++) { Start-Sleep -Milliseconds 100 }
  $port = [IO.File]::ReadAllText((Join-Path $tmp 'port')).Trim()
  $env:HF_ENDPOINT = "http://127.0.0.1:$port"
}
function Stop-Hub { if ($script:hubProc -and -not $script:hubProc.HasExited) { Stop-Process -Id $script:hubProc.Id -Force; $script:hubProc.WaitForExit(5000) | Out-Null } }
function HubLog { $p = Join-Path $tmp 'log'; if (Test-Path -LiteralPath $p) { [IO.File]::ReadAllText($p) } else { '' } }
function Transfers([string]$file) { @((HubLog) -split "`n" | Where-Object { $_ -match "^GET /\S+/resolve/\S+/$([regex]::Escape($file)) " }).Count }
# The Range of each transfer of a file since the hub started, in order ('-' for none), comma-joined.
function Ranges([string]$file) { @((HubLog) -split "`n" | Where-Object { $_ -match "^GET /\S+/resolve/\S+/$([regex]::Escape($file)) " } | ForEach-Object { ($_.Trim() -split ' ')[2] }) -join ',' }
function Run([string]$script, [string[]]$a) {
  $ErrorActionPreference = 'Continue'
  $o = & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $script @a 2>&1 | ForEach-Object { "$_" }
  $rc = $LASTEXITCODE
  $ErrorActionPreference = 'Stop'
  return [pscustomobject]@{ Out = ($o -join "`n"); Rc = $rc }
}
function Sha([string]$p) { (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLowerInvariant() }

try {
  # The hub's repos: one container (1 MiB), and a ladder of two beside a README.
  $rng = New-Object Random 11
  foreach ($f in @(@('brl-test\one\m.rad', 1048576), @('brl-test\multi\small.rad', 65536), @('brl-test\multi\big.rad', 131072), @('brl-test\multi\README.md', 100))) {
    $p = Join-Path (Join-Path $tmp 'hub') $f[0]
    [void][IO.Directory]::CreateDirectory((Split-Path -Parent $p))
    $b = New-Object byte[] $f[1]; $rng.NextBytes($b); [IO.File]::WriteAllBytes($p, $b)
  }
  $src = Join-Path $tmp 'hub\brl-test\one\m.rad'
  $commit = '0123456789abcdef0123456789abcdef01234567'
  Start-Hub

  # ======================================================================= the plan, and -Plan
  $dest = Join-Path $tmp 'dest one'
  $r = Run $hfget @('brl-test/one', $dest, 'main', 'm.rad', '-Plan')
  Check ($r.Rc -eq 0) "-Plan exits 0:`n$($r.Out)"
  Check ((Has $r.Out "revision main, commit $commit") -and (Has $r.Out "hfget: to      $dest") -and (Has $r.Out 'm.rad  1,048,576 bytes, to fetch')) "the plan names the commit, the destination and the bytes:`n$($r.Out)"
  Check (-not (Test-Path -LiteralPath $dest) -and (Transfers 'm.rad') -eq 0) "-Plan creates and transfers nothing"

  # ======================================================================= a fresh destination
  $r = Run $hfget @('brl-test/one', $dest, 'main', 'm.rad')
  $o = $r.Out
  Check ($r.Rc -eq 0) "a fresh fetch exits 0:`n$o"
  Check ($o.IndexOf('1,048,576 bytes, to fetch') -ge 0 -and $o.IndexOf('1,048,576 bytes, to fetch') -lt $o.IndexOf('  > m.rad')) "the plan comes before the transfer:`n$o"
  Check ((Sha (Join-Path $dest 'm.rad')) -eq (Sha $src)) "the file arrived whole"
  Check ((Test-Path -LiteralPath (Join-Path $dest '.hfget-manifest-m.rad')) -and (Has $o "hfget: verified $(Join-Path $dest 'm.rad')")) "it is verified and recorded:`n$o"
  Check ((HubLog) -match "GET /brl-test/one/resolve/$commit/m.rad") "the file is read at the commit the plan named"

  # ======================================================================= a warm cache
  $before = Transfers 'm.rad'
  $r = Run $hfget @('brl-test/one', $dest, 'main', 'm.rad')
  Check ($r.Rc -eq 0 -and (Has $r.Out 'kept (already whole)') -and (Has $r.Out '0 bytes (0.00 GiB) to fetch')) "a warm cache keeps the file:`n$($r.Out)"
  Check ((Transfers 'm.rad') -eq $before) "a warm cache transfers nothing"

  # ======================================================================= a transfer cut short
  # Every response for the first six (one try and five more) stops half way; the run ends incomplete
  # with a partial file, and the next run resumes from where it stopped. Each try within the run
  # resumes too: it asks for the bytes after those it has, never byte 0 again (curl's own --retry
  # did, 2026-11-20). Half of what was asked arrives each time: 512 KiB, then 256, 128, 64, 32, 16.
  Stop-Hub; Start-Hub 6
  $cut = Join-Path $tmp 'dest cut'
  $r = Run $hfget @('brl-test/one', $cut, 'main', 'm.rad')
  $part = Join-Path $cut 'm.rad'
  $len = if (Test-Path -LiteralPath $part) { (Get-Item -LiteralPath $part).Length } else { -1 }
  Check ($r.Rc -ne 0 -and $r.Rc -ne 3 -and (Has $r.Out 'incomplete -- re-run to resume')) "a cut transfer fails as incomplete:`n$($r.Out)"
  Check ($len -gt 0 -and $len -lt 1048576 -and -not (Test-Path -LiteralPath (Join-Path $cut '.hfget-manifest-m.rad'))) "a partial file and no manifest ($len bytes)"
  $rs = Ranges 'm.rad'
  Check ($rs -eq '-,bytes=524288-,bytes=786432-,bytes=917504-,bytes=983040-,bytes=1015808-' -and $len -eq 1032192) "each try in the run resumed where the one before stopped ($len bytes):`n$rs"
  Check ((Has $r.Out 'm.rad: curl exit 18 at byte 524288 of 1048576; try 2 of 6 resumes there') -and (Has $r.Out 'try 6 of 6')) "each new try is announced with its byte:`n$($r.Out)"
  $r = Run $hfget @('brl-test/one', $cut, 'main', 'm.rad')
  Check ($r.Rc -eq 0 -and (Has $r.Out "to resume at byte $len") -and (Has $r.Out "resuming at $len of 1048576")) "the next run resumes:`n$($r.Out)"
  Check ((HubLog) -match "resolve/$commit/m.rad bytes=$len-") "it asked for the rest only (Range bytes=$len-)"
  Check ((Sha $part) -eq (Sha $src) -and (Test-Path -LiteralPath (Join-Path $cut '.hfget-manifest-m.rad'))) "the resumed file is whole and verified"

  # ======================================================================= cut twice, one run
  # Two cuts and tries left: the same run finishes the file, each try from where the last stopped,
  # so the hub sends the file's bytes once in all (512 KiB + 256 KiB + 256 KiB).
  Stop-Hub; Start-Hub 2
  $twice = Join-Path $tmp 'dest twice'
  $r = Run $hfget @('brl-test/one', $twice, 'main', 'm.rad')
  Check ($r.Rc -eq 0 -and (Has $r.Out "hfget: verified $(Join-Path $twice 'm.rad')")) "two cuts within tries: one run completes:`n$($r.Out)"
  $rs = Ranges 'm.rad'
  Check ($rs -eq '-,bytes=524288-,bytes=786432-') "the second and third tries resumed (no byte 0 again):`n$rs"
  Check ((Sha (Join-Path $twice 'm.rad')) -eq (Sha $src)) "the file resumed within the run is whole"
  Stop-Hub; Start-Hub

  # ======================================================================= several containers
  $md = Join-Path $tmp 'dest multi'
  $r = Run $hfget @('brl-test/multi', $md, 'main')
  Check ($r.Rc -eq 4 -and (Has $r.Out 'holds 2 containers; name the one to fetch') -and (Has $r.Out 'small.rad') -and (Has $r.Out '131,072 bytes')) "a repo of two containers needs a choice:`n$($r.Out)"
  Check (-not (Test-Path -LiteralPath $md) -and (Transfers 'small.rad') -eq 0 -and (Transfers 'big.rad') -eq 0) "nothing was created or transferred"
  $r = Run $hfget @('brl-test/multi', $md, 'main', 'small.rad')
  Check ($r.Rc -eq 0 -and (Test-Path -LiteralPath (Join-Path $md 'small.rad')) -and -not (Test-Path -LiteralPath (Join-Path $md 'big.rad'))) "the chosen one alone is fetched:`n$($r.Out)"

  # ======================================================================= serve.ps1 -Fetch
  $env:BRILLIANCE_SERVE_TEST_DEVICES = '0:33:1,1:6:0,2:3:0,3:9:0,4:32:0'
  $env:HIP_VISIBLE_DEVICES = $null
  $profile = Join-Path $tmp 'one.yaml'
  [IO.File]::WriteAllText($profile, "# a test profile`n# brilliance-fetch: brl-test/one main m.rad`nserver:`n  port: 18391`nmodels:`n  a:`n    model: C:/nowhere/m.rad`n")
  $store = Join-Path $tmp 'store'
  $served = Join-Path $store 'brl-test\one\m.rad'
  $r = Run $serve @($profile, '-Fetch', '-ModelDir', $store, '-Cpu', '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out '-Plan: nothing was fetched or changed') -and (Has $r.Out 'model a: model from environment (BRILLIANCE_MODEL_PATH)')) "-Fetch -WhatIf shows the plan and the path it would serve:`n$($r.Out)"
  Check (-not (Test-Path -LiteralPath $store)) "-Fetch -WhatIf creates nothing"
  $r = Run $serve @($profile, '-Fetch', '-ModelDir', $store, '-Cpu', '-Check', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out "model a = $served, tp 1")) "-Fetch fetches, and radiance reads the verified file:`n$($r.Out)"
  Check ((Sha $served) -eq (Sha $src)) "the served file is the hub's"
  Check (([IO.File]::ReadAllText($profile)).Contains('model: C:/nowhere/m.rad')) "the profile is not edited"
  $before = Transfers 'm.rad'
  $r = Run $serve @($profile, '-Fetch', '-ModelDir', $store, '-Cpu', '-Check', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out 'kept (already whole)') -and (Transfers 'm.rad') -eq $before) "a second -Fetch transfers nothing:`n$($r.Out)"
  # BRILLIANCE_HF_CACHE is the store without -ModelDir, as for radiance's own fetch
  $env:BRILLIANCE_HF_CACHE = $store
  $r = Run $serve @($profile, '-Fetch', '-Cpu', '-Check', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out "model a = $served, tp 1")) "the store is BRILLIANCE_HF_CACHE by default:`n$($r.Out)"
  # no hub: the copy a complete fetch left is served, and said so
  Stop-Hub
  $r = Run $serve @($profile, '-Fetch', '-Cpu', '-Check', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out 'serving the copy a complete fetch left') -and (Has $r.Out "model a = $served, tp 1")) "no hub: the verified copy is served:`n$($r.Out)"
  $r = Run $serve @($profile, '-Fetch', '-ModelDir', (Join-Path $tmp 'empty store'), '-Cpu', '-Check', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out 'did not complete')) "no hub and no copy: refused:`n$($r.Out)"
  $env:BRILLIANCE_HF_CACHE = $null
  Start-Hub
  # a profile naming a repo of several containers and no file
  $multi = Join-Path $tmp 'multi.yaml'
  [IO.File]::WriteAllText($multi, "# brilliance-fetch: brl-test/multi main`nmodels:`n  a:`n    model: C:/nowhere/m.rad`n")
  $r = Run $serve @($multi, '-Fetch', '-ModelDir', $store, '-Cpu', '-Check', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out 'holds several containers') -and -not (Test-Path -LiteralPath (Join-Path $store 'brl-test\multi'))) "-Fetch of an ambiguous repo is refused before a transfer:`n$($r.Out)"
  $r = Run $serve @($profile, '-Fetch', '-Model', $served, '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out '-Model and -Fetch: one at a time')) "-Fetch with -Model:`n$($r.Out)"
} finally {
  Stop-Hub
  foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
Write-Host "fetch_launch_test: $($script:checks - $script:failed) of $($script:checks) checks hold"
if ($script:failed) { exit 1 }
exit 0
