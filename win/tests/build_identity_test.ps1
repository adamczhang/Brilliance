#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: win\tools\build-identity.ps1 on fixture headers: the line win\build.ps1 prints after a
  build, which the release's G1.build gate binds to the candidate.

.DESCRIPTION
  No build, no card. Writes fixture brl_identity.h files under RAD_TEST_TMPDIR (else the system temp
  folder) and checks the printed line and the exit status: a clean commit, a dirty one, a missing
  header, a header with no commit. Prints ok/FAIL per check; exit 0 when every check passed.
#>
$ErrorActionPreference = 'Stop'
$script = Join-Path (Split-Path -Parent $PSScriptRoot) 'tools\build-identity.ps1'
$base = $env:RAD_TEST_TMPDIR
if (-not $base) { $base = [IO.Path]::GetTempPath() }
$dir = Join-Path $base ('build-identity-test-' + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force (Join-Path $dir 'clean\brl') | Out-Null
New-Item -ItemType Directory -Force (Join-Path $dir 'dirty\brl') | Out-Null
New-Item -ItemType Directory -Force (Join-Path $dir 'nocommit\brl') | Out-Null
$utf8 = New-Object Text.UTF8Encoding($false)
function Header([string]$commit) {
  "/* Generated at build time by win/cmake/brl_identity.cmake, not in the source tree. */`n#pragma once`n" +
  "#define BRILLIANCE_VERSION `"0.3.0`"`n#define BRILLIANCE_COMMIT `"$commit`"`n" +
  "#define BRILLIANCE_UPSTREAM `"v2.2.0 b6b8751cccb0e834a13ae9e5afca824ee9a487aa`"`n"
}
[IO.File]::WriteAllText((Join-Path $dir 'clean\brl\brl_identity.h'), (Header '0123456789ab'), $utf8)
[IO.File]::WriteAllText((Join-Path $dir 'dirty\brl\brl_identity.h'), (Header '0123456789ab-dirty'), $utf8)
[IO.File]::WriteAllText((Join-Path $dir 'nocommit\brl\brl_identity.h'), (Header 'unknown'), $utf8)

$fails = 0
function Check([string]$what, [bool]$cond, [string]$detail) {
  if ($cond) { Write-Output "  ok    $what" } else { $script:fails++; Write-Output "  FAIL  $what : $detail" }
}
function Run([string[]]$argv) {
  $out = & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $script @argv
  return @{ out = (($out | Out-String).Trim()); code = $LASTEXITCODE }
}

$r = Run @('-BuildDir', (Join-Path $dir 'clean'))
Check 'a clean build names its version, commit, state and upstream' `
  ($r.out -eq 'build identity: Brilliance 0.3.0 commit 0123456789ab clean, upstream v2.2.0 b6b8751cccb0e834a13ae9e5afca824ee9a487aa') $r.out
Check 'a clean build exits 0' ($r.code -eq 0) "exit $($r.code)"

$r = Run @('-Header', (Join-Path $dir 'dirty\brl\brl_identity.h'))
Check 'a dirty build says dirty, and its commit keeps -dirty' `
  ($r.out -eq 'build identity: Brilliance 0.3.0 commit 0123456789ab-dirty dirty, upstream v2.2.0 b6b8751cccb0e834a13ae9e5afca824ee9a487aa') $r.out

$r = Run @('-BuildDir', (Join-Path $dir 'missing'))
Check 'no header: "build identity: none", exit 1' (($r.out -like 'build identity: none (no *') -and $r.code -eq 1) "$($r.out) / exit $($r.code)"

$r = Run @('-BuildDir', (Join-Path $dir 'nocommit'))
Check 'a header with no commit: "none", exit 1' (($r.out -like 'build identity: none (*names no commit*') -and $r.code -eq 1) "$($r.out) / exit $($r.code)"

Remove-Item -Recurse -Force $dir
if ($fails) { Write-Output "build_identity_test: $fails failed"; exit 1 }
Write-Output 'build_identity_test: 0 failed'
exit 0
