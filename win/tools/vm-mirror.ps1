#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: keep a local mirror of this checkout for building inside the development VM.

.DESCRIPTION
  In the VM the checkout is reached over the guest's SMB view of the host's drive. Sixteen parallel
  compiles reading it starve each other (measured 2026-11-16: reads fell to ~12 KB/s with the CPUs
  13% busy), and links writing debug files there took about 50 minutes. So the VM builds from a
  copy on its own disk: the sources (no .git, no build trees, no evidence) and the toolchain (no
  downloads), refreshed by robocopy, which copies only what changed. The checkout stays the one
  place edits and commits happen; the mirror is a cache and can be deleted at any time.

  The mirror's .toolchain\toolchain.json is rewritten to its own paths, so its win\env.ps1 makes a
  build shell on the local copy. On the physical Windows boot none of this is needed: build in the
  checkout's build-win\ with win\build.ps1.

.PARAMETER Dest
  The mirror (default %LOCALAPPDATA%\Brilliance2\mirror).
.PARAMETER Toolchain
  Also mirror .toolchain (the first time, or after win\setup.ps1 changed it).
#>
param(
  [string]$Dest = (Join-Path $env:LOCALAPPDATA 'Brilliance2\mirror'),
  [switch]$Toolchain
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force $Dest | Out-Null

function Mirror([string]$from, [string]$to, [string[]]$xd) {
  # /MIR copies new and changed files and removes what the source no longer has; /XD keeps the
  # named directories out of both. Exit codes below 8 are success (1 = files copied).
  $a = @($from, $to, '/MIR', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/R:2', '/W:2', '/MT:16')
  if ($xd) { $a += '/XD'; $a += $xd }
  & robocopy.exe @a | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "robocopy $from -> $to failed ($LASTEXITCODE)" }
}

$t = Get-Date
Mirror $repo $Dest @((Join-Path $repo '.git'), (Join-Path $repo 'build-win'), (Join-Path $repo 'artifacts'),
                     (Join-Path $repo '.toolchain'), (Join-Path $Dest 'build-win'), (Join-Path $Dest '.toolchain'))
"sources mirrored in {0:N1} s" -f ((Get-Date) - $t).TotalSeconds

if ($Toolchain -or -not (Test-Path (Join-Path $Dest '.toolchain\toolchain.json'))) {
  $t = Get-Date
  Mirror (Join-Path $repo '.toolchain') (Join-Path $Dest '.toolchain') @((Join-Path $repo '.toolchain\downloads'))
  $m = Get-Content (Join-Path $repo '.toolchain\toolchain.json') -Raw
  $m = $m.Replace(($repo -replace '\\', '\\'), ($Dest -replace '\\', '\\'))
  Set-Content -Path (Join-Path $Dest '.toolchain\toolchain.json') -Value $m -Encoding ASCII
  "toolchain mirrored in {0:N1} s" -f ((Get-Date) - $t).TotalSeconds
}
$Dest
