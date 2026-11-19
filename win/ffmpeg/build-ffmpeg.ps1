<#
.SYNOPSIS
  Builds Brilliance's FFmpeg 8.1.3 (LGPL, shared, win64, decode-first) with vcpkg and MSVC 14.51,
  and packs it as brilliance-ffmpeg-8.1.3-win64-lgpl-<date>.zip with its SHA-256.

.DESCRIPTION
  The recipe is win\ffmpeg (README.md there): vcpkg.json (features), ports\ffmpeg (the overlay port:
  FFmpeg n8.1.3 and the Brilliance configure block), triplets\x64-windows-brl.cmake.

  What it fetches (once; later runs reuse it):
    - the vcpkg registry at tag 2026.07.29 (commit 9e593bb1...), a depth-1 git fetch, into
      .toolchain\vcpkg;
    - through vcpkg, into .toolchain\vcpkg-downloads: the FFmpeg n8.1.3 source, the dav1d, libwebp
      and zlib sources, AMD's AMF headers, and the tools vcpkg acquires (msys2 packages, nasm,
      meson, Python, pkgconf, CMake, 7-Zip, PowerShell 7).
  The vcpkg program is the one bundled with the pinned Visual Studio Build Tools 18 (2026-07-27,
  the version the registry tag names), copied into the registry clone. Nothing is written to C:
  apart from what MSVC itself does.

  FFmpeg's vcpkg port refuses a source path with a space in it, and the repository lives in
  "Brilliance 2". So the repository is mapped to a drive letter with subst (B: by default) for
  the length of the run, and removed again at the end.

  Outputs (git-ignored):
    .toolchain\ffmpeg-vcpkg\{installed,bt,pk}   vcpkg's install, build and package trees
    .toolchain\vcpkg-archives                    vcpkg's binary cache (a rerun restores from it)
    artifacts\ffmpeg\<name>\                     the staged package
    artifacts\ffmpeg\<name>.zip, .zip.sha256     the release asset
    artifacts\ffmpeg\tools\ffprobe.exe           for checks only; not in the zip

  No GPU, no server. Runs in Windows PowerShell 5.1.

.PARAMETER Date
  The date in the package name, yyyyMMdd. Default: today.

.PARAMETER Drive
  The drive letter subst maps the repository to. Default: B.

.PARAMETER Clean
  Remove .toolchain\ffmpeg-vcpkg and vcpkg's binary cache first, for a build from source.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File win\ffmpeg\build-ffmpeg.ps1 -Clean
#>
[CmdletBinding()]
param(
  [string]$Date = (Get-Date -Format 'yyyyMMdd'),
  [string]$Drive = 'B',
  [switch]$Clean
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$RegistryTag = '2026.07.29'
$RegistryCommit = '9e593bb18ea69cc5095e012465dcd675a822ed0d'
$VcpkgTool = '2026-07-27'
$Triplet = 'x64-windows-brl'
$FfVersion = '8.1.3'
$Name = "brilliance-ffmpeg-$FfVersion-win64-lgpl-$Date"
$Libs = @('avcodec', 'avformat', 'avutil', 'swscale', 'swresample')

function Say([string]$m) { Write-Host ("[{0}] {1}" -f (Get-Date -Format 'h:mm tt'), $m) }
function Need([bool]$ok, [string]$m) { if (-not $ok) { throw $m } }
function Run([string]$exe, [string[]]$a) {
  & $exe @a
  if ($LASTEXITCODE -ne 0) { throw "$exe failed ($LASTEXITCODE): $($a -join ' ')" }
}

# ---------------------------------------------------------------- the pinned Visual Studio
$tcj = Join-Path $Repo '.toolchain\toolchain.json'
$VsPath = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools'
if (Test-Path $tcj) {
  $tc = Get-Content -Raw $tcj | ConvertFrom-Json
  if ($tc.vs_path) { $VsPath = $tc.vs_path }
}
Need (Test-Path (Join-Path $VsPath 'VC\Tools\MSVC\14.51.36231')) "MSVC 14.51.36231 not found under $VsPath (win\setup.ps1 installs it)"
$BundledVcpkg = Join-Path $VsPath 'VC\vcpkg\vcpkg.exe'
Need (Test-Path $BundledVcpkg) "no vcpkg in $VsPath\VC\vcpkg"

# ---------------------------------------------------------------- the registry, pinned
$Registry = Join-Path $Repo '.toolchain\vcpkg'
if (-not (Test-Path (Join-Path $Registry '.git'))) {
  Say "fetching the vcpkg registry at $RegistryTag (depth 1)"
  New-Item -ItemType Directory -Force $Registry | Out-Null
  Run git @('-C', $Registry, 'init', '-q')
  Run git @('-C', $Registry, 'remote', 'add', 'origin', 'https://github.com/microsoft/vcpkg.git')
  Run git @('-C', $Registry, 'fetch', '-q', '--depth', '1', 'origin', "refs/tags/${RegistryTag}:refs/tags/$RegistryTag")
  Run git @('-C', $Registry, 'checkout', '-q', $RegistryTag)
}
$head = (& git -C $Registry rev-parse HEAD).Trim()
Need ($head -eq $RegistryCommit) "the registry is at $head, not $RegistryCommit"
$Vcpkg = Join-Path $Registry 'vcpkg.exe'
if (-not (Test-Path $Vcpkg)) { Copy-Item $BundledVcpkg $Vcpkg }
$ver = (& $Vcpkg version | Select-Object -First 1)
Need ($ver -match [regex]::Escape($VcpkgTool)) "vcpkg is '$ver', expected $VcpkgTool"

# ---------------------------------------------------------------- subst: a path with no space
$D = "${Drive}:"
$existing = (& subst) | Where-Object { $_ -like "${D}\*" }
if ($existing) { throw "$D is already a subst drive ($existing); pick another with -Drive" }
Need (-not (Test-Path "$D\")) "$D is in use; pick another with -Drive"
Run subst @($D, $Repo)
try {
  $W = "$D\.toolchain\ffmpeg-vcpkg"
  $Cache = "$D\.toolchain\vcpkg-archives"
  if ($Clean) {
    Say 'clean: removing the vcpkg trees and the binary cache'
    foreach ($p in @($W, $Cache)) { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }
  }
  foreach ($p in @($W, $Cache, "$D\.toolchain\vcpkg-downloads", "$D\.toolchain\vcpkg-registries")) {
    New-Item -ItemType Directory -Force $p | Out-Null
  }

  $env:VCPKG_ROOT = "$D\.toolchain\vcpkg"
  $env:VCPKG_DOWNLOADS = "$D\.toolchain\vcpkg-downloads"
  $env:VCPKG_DEFAULT_BINARY_CACHE = $Cache
  $env:X_VCPKG_REGISTRIES_CACHE = "$D\.toolchain\vcpkg-registries"
  $env:VCPKG_DISABLE_METRICS = '1'
  $env:BRL_VS_PATH = $VsPath

  Say "vcpkg install (triplet $Triplet; logs under $W\bt)"
  $t0 = Get-Date
  Run "$D\.toolchain\vcpkg\vcpkg.exe" @(
    'install',
    "--vcpkg-root=$D\.toolchain\vcpkg",
    "--x-manifest-root=$D\win\ffmpeg",
    "--x-install-root=$W\installed",
    "--x-buildtrees-root=$W\bt",
    "--x-packages-root=$W\pk",
    "--overlay-ports=$D\win\ffmpeg\ports",
    "--overlay-triplets=$D\win\ffmpeg\triplets",
    "--triplet=$Triplet",
    "--host-triplet=$Triplet",
    "--binarysource=clear;files,$Cache,readwrite",
    '--disable-metrics')
  Say ("vcpkg install done in {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)
} finally {
  & subst $D /D | Out-Null
  foreach ($v in @('VCPKG_ROOT', 'VCPKG_DOWNLOADS', 'VCPKG_DEFAULT_BINARY_CACHE', 'X_VCPKG_REGISTRIES_CACHE')) {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
  }
}

# ---------------------------------------------------------------- stage the package
$Inst = Join-Path $Repo ".toolchain\ffmpeg-vcpkg\installed\$Triplet"
$Bt = Join-Path $Repo '.toolchain\ffmpeg-vcpkg\bt'
$Out = Join-Path $Repo 'artifacts\ffmpeg'
$Stage = Join-Path $Out $Name
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
foreach ($d in @('bin', 'lib', 'include', 'licenses')) { New-Item -ItemType Directory -Force (Join-Path $Stage $d) | Out-Null }
New-Item -ItemType Directory -Force (Join-Path $Out 'tools') | Out-Null

foreach ($l in $Libs) {
  $dll = @(Get-ChildItem (Join-Path $Inst 'bin') -Filter "$l-*.dll")
  Need ($dll.Count -eq 1) "expected one $l-*.dll in $Inst\bin, found $($dll.Count)"
  Copy-Item $dll[0].FullName (Join-Path $Stage 'bin')
  Copy-Item (Join-Path $Inst "lib\$l.lib") (Join-Path $Stage 'lib')
  Copy-Item -Recurse (Join-Path $Inst "include\lib$l") (Join-Path $Stage 'include')
}
$extra = @(Get-ChildItem (Join-Path $Inst 'bin') -Filter '*.dll' | Where-Object { -not (Test-Path (Join-Path $Stage "bin\$($_.Name)")) })
if ($extra.Count -gt 0) { throw "unexpected DLLs in the install: $(($extra | ForEach-Object { $_.Name }) -join ', ') (a dependency was built shared)" }
Copy-Item (Join-Path $Inst 'tools\ffmpeg\ffprobe.exe') (Join-Path $Out 'tools') -Force
foreach ($l in $Libs) { Copy-Item (Join-Path $Stage "bin\$l-*.dll") (Join-Path $Out 'tools') -Force }

# Licences: upstream's files, unedited, from the source archives vcpkg downloaded, under the names
# win\licenses uses (THIRD-PARTY-ffmpeg.md lists them). FFmpeg's is also LICENSE.txt, the name
# win\cmake\RadWindows.cmake installs as ffmpeg.LICENSE.txt.
$Dl = Join-Path $Repo '.toolchain\vcpkg-downloads'
$Texts = @(
  @{ tar = 'ffmpeg-ffmpeg-n8.1.3.tar.gz';         path = 'FFmpeg-n8.1.3/COPYING.LGPLv2.1'; name = 'ffmpeg.LICENSE' },
  @{ tar = 'videolan-dav1d-1.5.3.tar.gz';         path = 'dav1d-1.5.3/COPYING';             name = 'dav1d.LICENSE' },
  @{ tar = 'videolan-dav1d-1.5.3.tar.gz';         path = 'dav1d-1.5.3/doc/PATENTS';         name = 'dav1d.PATENTS' },
  @{ tar = 'webmproject-libwebp-v1.6.0.tar.gz';   path = 'libwebp-1.6.0/COPYING';           name = 'libwebp.LICENSE' },
  @{ tar = 'webmproject-libwebp-v1.6.0.tar.gz';   path = 'libwebp-1.6.0/PATENTS';           name = 'libwebp.PATENTS' },
  @{ tar = 'madler-zlib-v1.3.2.tar.gz';           path = 'zlib-1.3.2/LICENSE';              name = 'zlib.LICENSE' }
)
$Tmp = Join-Path $Out 'tmp-licences'
if (Test-Path $Tmp) { Remove-Item -Recurse -Force $Tmp }
New-Item -ItemType Directory -Force $Tmp | Out-Null
foreach ($t in $Texts) {
  $a = Join-Path $Dl $t.tar
  Need (Test-Path $a) "missing $a (a build from source fetches it: run with -Clean)"
  # Windows' own tar (bsdtar): a GNU tar earlier on PATH would read "D:" as a remote host.
  Run (Join-Path $env:SystemRoot 'System32\tar.exe') @('-xzf', $a, '-C', $Tmp, $t.path)
  Copy-Item (Join-Path $Tmp $t.path) (Join-Path $Stage "licenses\$($t.name)")
}
Remove-Item -Recurse -Force $Tmp
# The AMF headers' licence: the amd-amf port downloads it by itself, from the v1.5.2 tag
# (https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/AMF/v1.5.2/LICENSE.txt).
$amfLicence = Join-Path $Dl 'LICENSE.txt'
Need (Test-Path $amfLicence) "missing $amfLicence (the amd-amf port fetches it: run with -Clean)"
Copy-Item $amfLicence (Join-Path $Stage 'licenses\amf.LICENSE')
Copy-Item (Join-Path $Stage 'licenses\ffmpeg.LICENSE') (Join-Path $Stage 'LICENSE.txt')

# The recipe itself, so the zip carries what the LGPL calls the scripts used to build the library
# (FFmpeg's source is the n8.1.3 archive plus the port's patches in source\ports\ffmpeg).
$Src = Join-Path $Stage 'source'
New-Item -ItemType Directory -Force $Src | Out-Null
foreach ($f in @('README.md', 'build-ffmpeg.ps1', 'vcpkg.json', 'check-ffmpeg.c', 'THIRD-PARTY-ffmpeg.md')) {
  $p = Join-Path $PSScriptRoot $f
  if (Test-Path $p) { Copy-Item $p $Src }
}
Copy-Item -Recurse (Join-Path $PSScriptRoot 'ports') $Src
Copy-Item -Recurse (Join-Path $PSScriptRoot 'triplets') $Src

# BUILDINFO.txt: what was built, from what, and how.
$probe = Join-Path $Out 'tools\ffprobe.exe'
# Windows PowerShell turns a native program's stderr into errors; these two only report.
$ErrorActionPreference = 'Continue'
$buildconf = (& $probe -hide_banner -buildconf 2>&1 | ForEach-Object { "$_" } | Out-String).Trim()
$ports = (& "$Registry\vcpkg.exe" list "--vcpkg-root=$Registry" "--x-install-root=$(Join-Path $Repo '.toolchain\ffmpeg-vcpkg\installed')" | Out-String).Trim()
$ErrorActionPreference = 'Stop'
$cfgline = ''
$cfglog = Join-Path $Bt "ffmpeg\$Triplet-rel\ffbuild\config.log"
if (Test-Path $cfglog) { $cfgline = (Get-Content $cfglog -TotalCount 1) }
$info = @(
  "Brilliance FFmpeg $FfVersion, win64, LGPL-2.1-or-later, shared",
  "package: $Name",
  "recipe: win\ffmpeg in the Brilliance repository (build-ffmpeg.ps1)",
  "vcpkg registry: microsoft/vcpkg tag $RegistryTag, commit $RegistryCommit",
  "vcpkg tool: $ver",
  "compiler: MSVC 14.51.36231 (Visual Studio Build Tools 18), triplet $Triplet",
  "FFmpeg source: https://github.com/FFmpeg/FFmpeg/archive/n$FfVersion.tar.gz",
  '',
  'configure (as run, from ffbuild\config.log):',
  $cfgline,
  '',
  'ffprobe -buildconf:',
  $buildconf,
  '',
  'vcpkg list:',
  $ports
)
[IO.File]::WriteAllLines((Join-Path $Stage 'BUILDINFO.txt'), [string[]]$info, (New-Object Text.UTF8Encoding($false)))

# ---------------------------------------------------------------- the zip
# Entries sorted, every timestamp set to the package date, so the zip depends only on the bytes.
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Zip = Join-Path $Out "$Name.zip"
if (Test-Path $Zip) { Remove-Item -Force $Zip }
$stamp = [DateTimeOffset]::new([datetime]::ParseExact($Date, 'yyyyMMdd', $null))
$fs = [IO.File]::Open($Zip, 'CreateNew')
$za = New-Object IO.Compression.ZipArchive($fs, [IO.Compression.ZipArchiveMode]::Create)
try {
  $files = Get-ChildItem -Recurse -File $Stage | Sort-Object { $_.FullName.Substring($Stage.Length).Replace('\', '/') } -CaseSensitive
  foreach ($f in $files) {
    $rel = $Name + '/' + $f.FullName.Substring($Stage.Length + 1).Replace('\', '/')
    $e = $za.CreateEntry($rel, [IO.Compression.CompressionLevel]::Optimal)
    $e.LastWriteTime = $stamp
    $s = $e.Open()
    try { $b = [IO.File]::ReadAllBytes($f.FullName); $s.Write($b, 0, $b.Length) } finally { $s.Dispose() }
  }
} finally { $za.Dispose(); $fs.Dispose() }
$sha = (Get-FileHash -Algorithm SHA256 $Zip).Hash.ToLowerInvariant()
[IO.File]::WriteAllText("$Zip.sha256", "$sha  $Name.zip`n", (New-Object Text.UTF8Encoding($false)))
Say ("{0} ({1:N1} MB)" -f $Zip, ((Get-Item $Zip).Length / 1MB))
Say "SHA-256 $sha"
