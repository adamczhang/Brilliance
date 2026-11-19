#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: run named ctest tests one at a time, each into its own log, for triage.
.DESCRIPTION
  A parallel ctest run interleaves outputs; this runs each test alone (-R ^name$) so its log holds
  only its own lines. Prints one summary line per test: the result, then its first FAIL lines and
  the file:line of their checks.
.EXAMPLE
  powershell -File win\tools\test-each.ps1 -Build build-win\host -Out out\each -Names server_test,util_test
#>
param(
  [Parameter(Mandatory)][string]$Build,
  [Parameter(Mandatory)][string]$Out,
  [Parameter(Mandatory)][string[]]$Names
)
$ErrorActionPreference = 'Continue'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'env.ps1') | Out-Null
$gitUsr = Join-Path $env:ProgramFiles 'Git\usr\bin'
# First on PATH while tests run, as win\tools\test-log.ps1 explains.
if (Test-Path (Join-Path $gitUsr 'sh.exe')) { $env:PATH = "$gitUsr;" + (($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $gitUsr }) -join ';') }
New-Item -ItemType Directory -Force $Out | Out-Null
foreach ($n in ($Names -split ',')) {
  $n = $n.Trim(); if (-not $n) { continue }
  $log = Join-Path $Out "$n.log"
  cmd.exe /c "ctest --test-dir `"$Build`" -R `"^$n`$`" --output-on-failure > `"$log`" 2>&1"
  $c = Get-Content $log
  $res = ($c | Select-String "Test +#\d+: $n " | Select-Object -First 1).Line
  $res = $(if ($res) { ($res -replace '^.*Test +#\d+: ', '').Trim() } else { 'not found' })
  "== $res"
  $c | Select-String '^\s*FAIL |: (check|CHECK|REQUIRE)|\.(cpp|h):\d+:' | Select-Object -First 6 | ForEach-Object { '   ' + $_.Line.Trim() }
}
