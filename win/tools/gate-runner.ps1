#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: run the release gates against one candidate and write one evidence record
  (win\tools\gate_runner.py, from PowerShell). How it works: win\tools\GATE-RUNNER.md.

.DESCRIPTION
    powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File win\tools\gate-runner.ps1 `
      -Plan <plan.json> -Zip <Brilliance-...-win64.zip> -Out <evidence folder> `
      [-Inputs <inputs.json>] [-Pkg <unpacked>] [-Source <build clone>] [-Criteria <criteria.json>] `
      [-AllowCards 3,9] [-SoloWindow '<note>'] [-BuildLease <folder>] [-LeaseWait 600] `
      [-Only G1.docs_check,G2.package_check] [-KeepGoing] [-Resume] [-List]

  Writes <Out>\evidence.json, <Out>\EVIDENCE.md and <Out>\logs\. Exit 0 ready, 1 fail, 2 incomplete,
  3 a usage error: read the verdict from EVIDENCE.md all the same.

  Python: the repository's .toolchain\python\python.exe when there is one, else python3 or python on
  PATH, else -Python.
#>
param(
  [Parameter(Mandatory)][string]$Plan,
  [Parameter(Mandatory)][string]$Zip,
  [Parameter(Mandatory)][string]$Out,
  [string]$Inputs = '',
  [string]$Pkg = '',
  [string]$Source = '',
  [string]$Criteria = '',
  # Typed -AllowCards 3,9 arrives as an array, under -File as one string: both are joined.
  [string[]]$AllowCards = @(),
  [string]$SoloWindow = '',
  [string]$BuildLease = '',
  [int]$LeaseWait = 0,
  [string[]]$Only = @(),
  [switch]$KeepGoing,
  [switch]$Resume,
  [switch]$List,
  [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Python) {
  $tc = Join-Path $repo '.toolchain\python\python.exe'
  if (Test-Path $tc) { $Python = $tc }
  else {
    foreach ($n in 'python3', 'python') {
      $c = Get-Command $n -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
      if ($c) { $Python = $c.Source; break }
    }
  }
}
if (-not $Python) { Write-Error 'no Python found: give -Python'; exit 3 }

$argv = @('-I', (Join-Path $PSScriptRoot 'gate_runner.py'), '--plan', $Plan, '--zip', $Zip, '--out', $Out)
if ($Inputs) { $argv += @('--inputs', $Inputs) }
if ($Pkg) { $argv += @('--pkg', $Pkg) }
if ($Source) { $argv += @('--source', $Source) }
if ($Criteria) { $argv += @('--criteria', $Criteria) }
if ($AllowCards.Count) { $argv += @('--allow-cards', ($AllowCards -join ',')) }
if ($SoloWindow) { $argv += @('--solo-window', $SoloWindow) }
if ($BuildLease) { $argv += @('--build-lease', $BuildLease, '--lease-wait', "$LeaseWait") }
if ($Only.Count) { $argv += @('--only', ($Only -join ',')) }
if ($KeepGoing) { $argv += '--keep-going' }
if ($Resume) { $argv += '--resume' }
if ($List) { $argv += '--list' }

# A native command's stderr line is not a failure: the code comes from $LASTEXITCODE, never from $?.
$ErrorActionPreference = 'Continue'
& $Python @argv
exit $LASTEXITCODE
