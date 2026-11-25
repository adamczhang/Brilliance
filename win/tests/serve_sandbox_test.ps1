#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: win\serve.ps1 on a profile of several models -- the entries that need more cards than
  radiance will see are dropped and named, and radiance reads a copy of the profile less them. No
  card is used.
.DESCRIPTION
  ctest runs it (win\tests\CMakeLists.txt) with the radiance.exe of the build tree:
    powershell -File win\tests\serve_sandbox_test.ps1 -Exe build-win\host\bin\radiance.exe
  BRILLIANCE_SERVE_TEST_DEVICES hands serve.ps1 the test machine's device table (the iGPU, then buses 6, 3, 9,
  32), so the HIP runtime is never asked; radiance runs only as --check. Exit 0 when every check
  holds, 1 otherwise (each failure printed).
#>
param([Parameter(Mandatory)][string]$Exe)
$ErrorActionPreference = 'Stop'
$serve = (Resolve-Path (Join-Path $PSScriptRoot '..\serve.ps1')).Path
$Exe = (Resolve-Path -LiteralPath $Exe).Path
$script:failed = 0
$script:checks = 0
function Check([bool]$ok, [string]$what) {
  $script:checks++
  if (-not $ok) { $script:failed++; Write-Host "  FAIL $what" }
}
$red = '0:33:1,1:6:0,2:3:0,3:9:0,4:32:0'
function Serve([string[]]$a) {
  $all = @{ BRILLIANCE_SERVE_TEST_DEVICES = $red; HIP_VISIBLE_DEVICES = $null
            BRILLIANCE_SERVER_HOST = $null; BRILLIANCE_SERVER_PORT = $null; BRILLIANCE_SERVER_API_KEY = $null }
  $saved = @{}
  foreach ($k in $all.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $all[$k]) }
  try {
    $ErrorActionPreference = 'Continue'
    $o = & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $serve @a 2>&1 | ForEach-Object { "$_" }
    return [pscustomobject]@{ Out = ($o -join "`n"); Rc = $LASTEXITCODE }
  } finally {
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
  }
}
function Has([string]$text, [string]$part) { return $text.Contains($part) }
# The copy radiance reads, from the launcher's "radiance reads <file>, the profile less those" line.
# The path may hold spaces (a TEMP folder whose name has a space): everything up to the comma on that line.
function DerivedOf([string]$out) {
  if ($out -match '(?m)radiance reads (.+?), the profile less those') { return $Matches[1] }
  return ''
}
# A path as serve.ps1's -WhatIf prints it on radiance's command line (its Quote-Arg): in double
# quotes when it holds a space.
function Quoted([string]$p) { if ($p -match '[\s"]') { return "`"$p`"" } return $p }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("brl-serve-sandbox-test-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory $tmp | Out-Null
try {
  $m = 'C:/nonexistent/a.rad'
  $fleet = Join-Path $tmp 'fleet.yaml'
  [IO.File]::WriteAllText($fleet, "server:`n  host: 127.0.0.1`n  port: 18381`nscheduler:`n  warm-up: false`nmodels:`n  one:       # 1 card`n    model: $m`n    tp: 1`n  two:`n    model: $m`n    # a comment inside an entry`n    tp: 2`n    max-num-seqs: 2`n  four:`n    model: $m`n    tp: 4`n  plain:`n    model: $m`n")

  # Two cards: the four-card entry is dropped and named; radiance reads a copy without it.
  $r = Serve @($fleet, '-Bus', '3,9', '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -eq 0) "two cards: -WhatIf exits 0:`n$($r.Out)"
  Check (Has $r.Out 'dropped four: it needs 4 cards and radiance will see 2') "two cards: four is dropped and named:`n$($r.Out)"
  Check (-not (Has $r.Out 'dropped one') -and -not (Has $r.Out 'dropped two') -and -not (Has $r.Out 'dropped plain')) "two cards: nothing else is dropped:`n$($r.Out)"
  $d = DerivedOf $r.Out
  Check ($d -and (Test-Path -LiteralPath $d)) "two cards: the copy exists ($d)"
  if ($d -and (Test-Path -LiteralPath $d)) {
    $t = [IO.File]::ReadAllText($d)
    Check (-not ($t -match '(?m)^  four:')) "two cards: the copy has no four:`n$t"
    Check (($t -match '(?m)^  one:') -and ($t -match '(?m)^  two:') -and ($t -match '(?m)^  plain:') -and (Has $t 'max-num-seqs: 2') -and (Has $t '# a comment inside an entry')) "two cards: the copy keeps the rest whole:`n$t"
    Check (Has $r.Out "--config $(Quoted $d)")"two cards: radiance is run on the copy:`n$($r.Out)"
  }
  Check (Has $r.Out 'radiance reads the profile as: host 127.0.0.1') "two cards: radiance --check accepts the copy:`n$($r.Out)"

  # One card: the two- and four-card entries go.
  $r = Serve @($fleet, '-Bus', '3', '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out 'dropped two: it needs 2 cards and radiance will see 1') -and (Has $r.Out 'dropped four: it needs 4 cards')) "one card: two and four are dropped:`n$($r.Out)"

  # Every discrete card (the iGPU hidden): nothing dropped, the profile itself is run.
  $r = Serve @($fleet, '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and -not (Has $r.Out 'dropped') -and (Has $r.Out "--config $(Quoted $fleet)")) "four cards: nothing dropped, the profile itself:`n$($r.Out)"

  # The iGPU named beside a card is not a card the engine runs on.
  $r = Serve @($fleet, '-Bus', '33,3', '-WhatIf', '-Exe', $Exe)
  Check ((Has $r.Out 'dropped two: it needs 2 cards and radiance will see 1')) "the iGPU is not counted:`n$($r.Out)"

  # -Check with -Bus drops the same, and radiance checks the copy.
  $r = Serve @($fleet, '-Bus', '3,9', '-Check', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out 'dropped four')) "-Check: drops and exits 0:`n$($r.Out)"

  # Every entry too wide: refused with the reason.
  $wide = Join-Path $tmp 'wide.yaml'
  [IO.File]::WriteAllText($wide, "models:`n  a:`n    model: $m`n    tp: 2`n  b:`n    model: $m`n    tp: 4`n")
  $r = Serve @($wide, '-Bus', '3', '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out 'needs more than the 1 card(s) radiance will see')) "all too wide: refused:`n$($r.Out)"

  # Left to radiance: a one-model profile, a file with groups:, a file with per-model cards:.
  $one = Join-Path $tmp 'one.yaml'
  [IO.File]::WriteAllText($one, "models:`n  a:`n    model: $m`n    tp: 2`n")
  $r = Serve @($one, '-Bus', '3', '-WhatIf', '-Exe', $Exe)
  Check (-not (Has $r.Out 'dropped')) "one model: nothing dropped:`n$($r.Out)"
  $grp = Join-Path $tmp 'groups.yaml'
  [IO.File]::WriteAllText($grp, "models:`n  a:`n    model: $m`n  b:`n    model: $m`n    tp: 2`ngroups:`n  g: { together: [a, b] }`n")
  $r = Serve @($grp, '-Bus', '3', '-WhatIf', '-Exe', $Exe)
  Check (-not (Has $r.Out 'dropped')) "groups: nothing dropped:`n$($r.Out)"
  $own = Join-Path $tmp 'own.yaml'
  [IO.File]::WriteAllText($own, "models:`n  a:`n    model: $m`n    cards: [0]`n  b:`n    model: $m`n    tp: 2`n")
  $r = Serve @($own, '-Bus', '3', '-WhatIf', '-Exe', $Exe)
  Check (-not (Has $r.Out 'dropped')) "per-model cards: nothing dropped:`n$($r.Out)"

  # -Cpu: the host backend is one device, so every entry wider than one goes, and radiance reads the
  # copy on the CPU (no card asked about).
  $r = Serve @($fleet, '-Cpu', '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out 'dropped two: it needs 2 cards and -Cpu serves on one device') -and (Has $r.Out 'dropped four: it needs 4 cards and -Cpu serves on one device') -and
         -not (Has $r.Out 'dropped one') -and (Has $r.Out 'device host: the CPU only')) "-Cpu: two and four are dropped:`n$($r.Out)"
  $r = Serve @($wide, '-Cpu', '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out '-Cpu serves on one device')) "-Cpu, all too wide: refused:`n$($r.Out)"
  $r = Serve @('sandbox', '-Cpu', '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out 'dropped flashnext-2card') -and (Has $r.Out 'dropped flashnext-4card') -and -not (Has $r.Out 'dropped minicpm5-2b-1card') -and
         (Has $r.Out 'model minicpm5-2b-1card: model from models.minicpm5-2b-1card.model, num-speculative-tokens 0 (environment (BRILLIANCE_MODEL_SPECULATION))')) "sandbox -Cpu: the one-card entries stay, drafting off:`n$($r.Out)"

  # The shipped sandbox profile on the two-card pair: only the entries wider than two go, read from
  # the profile's own tp lines so a new four-card entry needs no edit here.
  $wider = @(); $entry = $null
  foreach ($line in [IO.File]::ReadAllLines((Join-Path $PSScriptRoot '..\profiles\sandbox.yaml'))) {
    if ($line -match '^  ([\w.-]+):') { $entry = $Matches[1] }
    elseif ($entry -and $line -match '^    tp: (\d+)' -and [int]$Matches[1] -gt 2) { $wider += $entry }
  }
  $want = (@($wider | Sort-Object) -join ',')
  $r = Serve @('sandbox', '-Bus', '3,9', '-WhatIf', '-Exe', $Exe)
  $dropped = @([regex]::Matches($r.Out, 'dropped (\S+):') | ForEach-Object { $_.Groups[1].Value } | Sort-Object)
  Check ($r.Rc -eq 0 -and $wider.Count -gt 0 -and ($dropped -join ',') -eq $want) "sandbox on buses 3 and 9: only the entries wider than two dropped (want '$want', got '$($dropped -join ',')'):`n$($r.Out)"
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
Write-Host "serve_sandbox_test: $($script:checks - $script:failed) of $($script:checks) checks hold"
if ($script:failed) { exit 1 }
exit 0
