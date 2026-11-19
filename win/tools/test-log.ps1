#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: run a build tree's ctest suite in a build shell and keep the whole log.
.DESCRIPTION
  Git for Windows' sh and the tools the shell-script tests call go at the END of PATH, for the tests
  only (its usr\bin has a link.exe that would shadow MSVC's linker if it came first). The log is
  written through cmd's redirection so it can be read while the suite runs.
.EXAMPLE
  powershell -File win\tools\test-log.ps1 -Build build-win\host -Log out\ctest-1.log
#>
param(
  [Parameter(Mandatory)][string]$Build,
  [Parameter(Mandatory)][string]$Log,
  [int]$Jobs = [Math]::Max(1, [Environment]::ProcessorCount / 2),
  [string]$Regex = '',
  [int]$Timeout = 0
)
$ErrorActionPreference = 'Continue'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'env.ps1') | Out-Null
# Git for Windows' sh and its tools FIRST on PATH while the tests run: the shell-script tests call
# find, sort and sha256sum, and System32's find.exe and sort.exe answer to those names otherwise.
# Nothing a test links goes through PATH (the toolchain is named by full path), so its link.exe
# (coreutils ln) shadowing MSVC's is harmless here, unlike in a build shell.
$gitUsr = Join-Path $env:ProgramFiles 'Git\usr\bin'
if (Test-Path (Join-Path $gitUsr 'sh.exe')) { $env:PATH = "$gitUsr;" + (($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $gitUsr }) -join ';') }
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null
$sel = $(if ($Regex) { "-R `"$Regex`"" } else { '' })
$to = $(if ($Timeout -gt 0) { "--timeout $Timeout" } else { '' })
$t = Get-Date
cmd.exe /c "ctest --test-dir `"$Build`" -j $Jobs --output-on-failure $sel $to > `"$Log`" 2>&1"
$rc = $LASTEXITCODE
"ctest exit $rc; {0:N0} s" -f ((Get-Date) - $t).TotalSeconds
Get-Content $Log -Tail 3
