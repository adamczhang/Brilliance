#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: turn the current PowerShell into a build shell. Dot-source it:  . .\win\env.ps1

.DESCRIPTION
  Imports the MSVC x64 environment of the pinned toolset (the STL, UCRT and Windows SDK that
  ROCm's clang compiles against) and puts the project's own toolchain first on PATH, all from the
  manifest win\setup.ps1 writes (.toolchain\toolchain.json).

  ROCM_PATH IS LEFT UNSET ON PURPOSE. With ROCM_PATH or ROCM_HOME set, clang drops its own
  device-library heuristics and can pick up a different ROCm's bitcode; HIP_DEVICE_LIB_PATH names
  the right one explicitly instead. A stale machine-wide HIP_PATH from the legacy HIP SDK is the
  same hazard, so HIP_PATH is overridden for this shell.
#>

# A CHILD SCOPE, so dot-sourcing changes the shell's environment and nothing else: the locals
# below and the Stop preference stay in here. Variables set through env: belong to the process
# and outlive the scope, which is the point.
& {
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $repo '.toolchain\toolchain.json'
if (-not (Test-Path $manifestPath)) { throw "no $manifestPath -- run .\win\setup.ps1 first" }
$tc = Get-Content $manifestPath -Raw | ConvertFrom-Json
foreach ($k in 'clang', 'cmake', 'ninja', 'vcvars') {
  if (-not $tc.$k -or -not (Test-Path $tc.$k)) { throw "the manifest's $k ($($tc.$k)) does not exist -- run .\win\setup.ps1" }
}

if ($env:BRILLIANCE_VCVARS_DONE -ne $tc.msvc_toolset) {
  # -vcvars_ver selects the pinned toolset even when Visual Studio has a newer one installed.
  cmd.exe /c "`"$($tc.vcvars)`" -vcvars_ver=$($tc.msvc_toolset) >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
  }
  # Marked done only when it worked: a failed vcvars would otherwise leave this shell without the
  # STL and SDK for good, every later dot-source skipping the import.
  if (-not $env:VCToolsInstallDir -or $env:VCToolsVersion -ne $tc.msvc_toolset) {
    throw "$($tc.vcvars) -vcvars_ver=$($tc.msvc_toolset) did not set up that toolset (got '$env:VCToolsVersion')"
  }
  $env:BRILLIANCE_VCVARS_DONE = $tc.msvc_toolset
}

$root = $tc.rocm_root
Remove-Item Env:ROCM_PATH -ErrorAction SilentlyContinue
Remove-Item Env:ROCM_HOME -ErrorAction SilentlyContinue
$env:HIP_PATH            = $root
$env:HIP_PLATFORM        = 'amd'
$env:HIP_DEVICE_LIB_PATH = Join-Path $root 'lib\llvm\amdgcn\bitcode'
$env:HIP_CLANG_PATH      = Join-Path $root 'lib\llvm\bin'
$env:LLVM_PATH           = Join-Path $root 'lib\llvm'
$env:BRILLIANCE_ROCM     = $root
$env:BRILLIANCE_GPU_TARGETS = $tc.gpu_target
if ($tc.libomp_dir) { $env:BRILLIANCE_LIBOMP_DIR = $tc.libomp_dir }
# The pinned FFmpeg (win\setup.ps1), which win\cmake\RadWindows.cmake makes the media target from.
if ($tc.PSObject.Properties['ffmpeg_root'] -and $tc.ffmpeg_root) { $env:BRILLIANCE_FFMPEG_ROOT = $tc.ffmpeg_root }

$prepend = @((Join-Path $root 'bin'), (Join-Path $root 'lib\llvm\bin'), (Split-Path $tc.cmake), (Split-Path $tc.ninja))
$rest = @($env:PATH -split ';' | Where-Object { $_ -and ($prepend -notcontains $_) })
$env:PATH = ($prepend + $rest) -join ';'

Write-Host ("Brilliance build shell: ROCm {0} ({1}), {2}" -f $tc.rocm_version, $tc.gpu_target, (& $tc.clang --version | Select-Object -First 1)) -ForegroundColor Green
Write-Host ("  MSVC {0}, Windows SDK {1}, {2}, ninja {3}" -f $env:VCToolsVersion, $env:WindowsSDKVersion, (& $tc.cmake --version | Select-Object -First 1), (& $tc.ninja --version))
}
