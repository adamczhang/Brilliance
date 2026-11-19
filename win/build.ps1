#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: configure and build in a build shell, and optionally run the test gate.

.DESCRIPTION
  Makes this process a build shell (win\env.ps1), then configures build-win\<flavor> with Ninja and
  the project's toolchain file and builds it.

    host  RAD_WITH_HIP=OFF: the host backend, every tool, plugin and test that needs no card (M1)
    hip   RAD_WITH_HIP=ON for the pinned GPU target (M2)

.PARAMETER Flavor
  host (default) or hip.
.PARAMETER Test
  Run ctest afterwards (-j from -Jobs), output on failure.
.PARAMETER Jobs
  Parallel jobs for the build and the tests (default: the logical processor count).
.PARAMETER Reconfigure
  Delete the CMake cache first.
.PARAMETER Target
  Build only these targets.
.PARAMETER KeepGoing
  Keep building past failures (ninja -k 0), to collect every error in one pass.
#>
[CmdletBinding()]
param(
  [ValidateSet('host', 'hip')][string]$Flavor = 'host',
  [switch]$Test,
  [int]$Jobs = [Environment]::ProcessorCount,
  [switch]$Reconfigure,
  [string[]]$Target,
  [switch]$KeepGoing
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'env.ps1')

$build = Join-Path $repo "build-win\$Flavor"
if ($Reconfigure -and (Test-Path (Join-Path $build 'CMakeCache.txt'))) { Remove-Item -Force (Join-Path $build 'CMakeCache.txt') }
$hip = $(if ($Flavor -eq 'hip') { 'ON' } else { 'OFF' })
$cfg = @('-S', $repo, '-B', $build, '-G', 'Ninja',
         "-DCMAKE_TOOLCHAIN_FILE=$repo\win\cmake\windows-rocm.cmake",
         '-DCMAKE_BUILD_TYPE=RelWithDebInfo', "-DRAD_WITH_HIP=$hip")
if ($Flavor -eq 'hip') { $cfg += "-DRAD_GPU_TARGETS=$env:BRILLIANCE_GPU_TARGETS" }
& cmake @cfg
if ($LASTEXITCODE -ne 0) { Write-Host 'configure failed' -ForegroundColor Red; exit 1 }

$b = @('--build', $build, '-j', $Jobs)
if ($Target) { $b += @('--target') + $Target }
if ($KeepGoing) { $b += @('--', '-k', '0') }
& cmake @b
if ($LASTEXITCODE -ne 0) { Write-Host 'build failed' -ForegroundColor Red; exit 2 }

if ($Test) {
  # A few tests are POSIX shell scripts (`sh` in their add_test). Git for Windows' sh and the tools
  # they call (awk, sha256sum, find) go at the END of PATH, for the tests only: its usr\bin also has
  # a link.exe (coreutils ln) that would shadow MSVC's linker if it came first.
  $gitUsr = Join-Path $env:ProgramFiles 'Git\usr\bin'
  if ((Test-Path (Join-Path $gitUsr 'sh.exe')) -and (($env:PATH -split ';') -notcontains $gitUsr)) { $env:PATH = "$env:PATH;$gitUsr" }
  & ctest --test-dir $build -j $Jobs --output-on-failure
  if ($LASTEXITCODE -ne 0) { Write-Host 'tests failed' -ForegroundColor Red; exit 3 }
}
exit 0
