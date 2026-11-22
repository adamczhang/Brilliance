#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: print the identity a build tree's binaries carry, as one log line.

.DESCRIPTION
  Reads brl\brl_identity.h, the header win\cmake\brl_identity.cmake writes at every build, and prints

    build identity: Brilliance <version> commit <12 hex> clean, upstream <tag sha>

  ("dirty" in place of "clean" when the build was made from uncommitted changes; the commit then
  ends "-dirty", as the header has it). win\build.ps1 prints it after a successful build, so a build
  log names the commit it built: the release gate runner's G1.build gate binds the log to the
  candidate by that line (win\investigations\release\gate-plan.json).

  Exit 0 when the line was printed; 1 when the header is missing or does not name a commit, after
  printing "build identity: none (...)".

.PARAMETER BuildDir
  The build tree (build-win\hip, build-win\host).
.PARAMETER Header
  The header itself, in place of -BuildDir.
#>
[CmdletBinding()]
param(
  [string]$BuildDir,
  [string]$Header
)
if (-not $Header) { $Header = Join-Path $BuildDir 'brl\brl_identity.h' }
if (-not (Test-Path -LiteralPath $Header)) {
  Write-Output "build identity: none (no $Header)"
  exit 1
}
$text = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $Header).Path, [Text.Encoding]::UTF8)
function Get-Define([string]$name) {
  $m = [regex]::Match($text, '(?m)^#define ' + $name + ' "([^"]*)"')
  if ($m.Success) { return $m.Groups[1].Value }
  return ''
}
$version = Get-Define 'BRILLIANCE_VERSION'
$commit = Get-Define 'BRILLIANCE_COMMIT'
$upstream = Get-Define 'BRILLIANCE_UPSTREAM'
if ($commit -notmatch '^[0-9a-f]{12}(-dirty)?$') {
  Write-Output "build identity: none ($Header names no commit: '$commit')"
  exit 1
}
$state = $(if ($commit.EndsWith('-dirty')) { 'dirty' } else { 'clean' })
Write-Output "build identity: Brilliance $version commit $commit $state, upstream $upstream"
exit 0
