#Requires -Version 5.1
<#
  win\tools\msvc-toolset.ps1 (Select-MsvcToolset, which win\setup.ps1 uses to choose the MSVC
  toolset) against fake VC\Tools\MSVC folder layouts: empty folders named like toolsets, under
  RAD_TEST_TMPDIR (else the system temp folder). No Visual Studio, no build. Prints ok/FAIL per
  check; exit 0 when every check passed, 1 otherwise. ctest runs it as msvc_toolset_test
  (win\tests\CMakeLists.txt) under Windows PowerShell 5.1, which is what runs win\setup.ps1.
#>
param([string]$Lib = '')
$ErrorActionPreference = 'Continue'
Set-StrictMode -Version 2.0   # as win\setup.ps1 runs it
if (-not $Lib) { $Lib = Join-Path (Split-Path -Parent $PSScriptRoot) 'tools\msvc-toolset.ps1' }
. $Lib

$script:fails = 0
function Check([string]$what, [bool]$cond, [string]$detail = '') {
  if ($cond) { Write-Output "  ok    $what" }
  else { $script:fails++; Write-Output "  FAIL  $what$(if ($detail) { ": $detail" })" }
}

$base = $env:RAD_TEST_TMPDIR
if (-not $base) { $base = [IO.Path]::GetTempPath() }
$dir = Join-Path $base ("brl-msvc-toolset-test-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $dir -Force | Out-Null
Write-Output "PowerShell $($PSVersionTable.PSVersion); fixtures in $dir"

# One fake <VS>\VC\Tools\MSVC per case, holding a folder per listed toolset.
$n = 0
function Layout([string[]]$names) {
  $script:n++
  $root = Join-Path $dir "case$script:n\VC\Tools\MSVC"
  New-Item -ItemType Directory -Path $root -Force | Out-Null
  foreach ($x in $names) { New-Item -ItemType Directory -Path (Join-Path $root $x) -Force | Out-Null }
  $root
}
function Pick([string[]]$names) { Select-MsvcToolset -MsvcRoot (Layout $names) -Tested '14.51' }
function Show($r) { if ($r) { "$($r.Name) fallback=$($r.Fallback)" } else { 'none' } }

# The four cases of the PR: 14.52 only, 14.51 beside 14.52, 14.50 only, nothing.
$r = Pick @('14.52.37025')
Check '14.52 only: 14.52.37025, as a fallback' ($r -and $r.Name -eq '14.52.37025' -and $r.Fallback) (Show $r)
$r = Pick @('14.51.36231', '14.52.37025')
Check '14.51 beside 14.52: the tested 14.51.36231, no fallback' ($r -and $r.Name -eq '14.51.36231' -and -not $r.Fallback) (Show $r)
$r = Pick @('14.50.35717')
Check '14.50 only: none (older than tested)' ($null -eq $r) (Show $r)
$r = Pick @()
Check 'an empty MSVC folder: none' ($null -eq $r) (Show $r)
$r = Select-MsvcToolset -MsvcRoot (Join-Path $dir 'no-such-vs\VC\Tools\MSVC') -Tested '14.51'
Check 'no MSVC folder at all: none' ($null -eq $r) (Show $r)

# Versions compare as versions, not strings.
$r = Pick @('14.51.36231', '14.51.9999')
Check 'two 14.51 builds: the newer by version (36231 over 9999)' ($r -and $r.Name -eq '14.51.36231' -and -not $r.Fallback) (Show $r)
$r = Pick @('14.52.37025', '14.100.10000')
Check '14.52 and 14.100: 14.100 is newer' ($r -and $r.Name -eq '14.100.10000' -and $r.Fallback) (Show $r)
$r = Pick @('14.9.1000')
Check '14.9 only: none (14.9 is older than 14.51)' ($null -eq $r) (Show $r)
$r = Pick @('14.50.35717', '14.53.38000', '14.52.37025')
Check '14.50, 14.52 and 14.53: the newest, 14.53' ($r -and $r.Name -eq '14.53.38000' -and $r.Fallback) (Show $r)

# Another major version, and folders that are not versions, never qualify.
$r = Pick @('15.0.10000')
Check '15.0 only: none (another major version)' ($null -eq $r) (Show $r)
$r = Pick @('readme', '14.52.37025-pre', '14.52.37025')
Check 'non-version folders are ignored' ($r -and $r.Name -eq '14.52.37025' -and $r.Fallback) (Show $r)
$r = Pick @('14.51')
Check 'a bare 14.51 folder counts as the tested toolset' ($r -and $r.Name -eq '14.51' -and -not $r.Fallback) (Show $r)

# A file named like a toolset is not a toolset.
$root = Layout @()
[IO.File]::WriteAllText((Join-Path $root '14.52.37025'), '')
$r = Select-MsvcToolset -MsvcRoot $root -Tested '14.51'
Check 'a file named 14.52.37025: none' ($null -eq $r) (Show $r)

Remove-Item -Recurse -Force $dir
if ($script:fails) { Write-Output "msvc_toolset_test: $script:fails failed"; exit 1 }
Write-Output 'msvc_toolset_test: all passed'
exit 0
