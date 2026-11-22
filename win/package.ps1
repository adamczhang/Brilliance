#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: a self-contained zip of a built tree -- unzip anywhere, even a path with spaces, and run
  bin\radiance.exe --config profiles\<profile>.yaml.

.DESCRIPTION
    .\win\package.ps1                         build-win\hip -> dist\Brilliance-<version>-<sha>-win64.zip
    .\win\package.ps1 -BuildDir build-win\hip -Out D:\releases

  What is in it:
    bin\              radiance.exe, the rad-* tools, the HIP runtime the tree was built against
                      (amdhip64_7, amd_comgr, rocm_kpack), libomp, and the VC++ runtime DLLs copied
                      app-locally, so a machine without the redistributable runs it; brl-stacks.exe
                      and radiance.map, radiance's link map, which name a hung process's frames
                      (tools\STACKS.md)
    share\radiance\   the plugin home radiance finds beside its bin\ (kernels, architectures,
                      quantizers, vectors), and recipes\, the conversion recipes rad-convert and
                      rad-export use (bin\rad-export.cmd runs rad-export with the machine's Python)
    share\doc\radiance\  LICENSE, NOTICE and the third-party licenses NOTICE names
    profiles\         the qualified YAML profiles (win\profiles)
    libexec\brilliance\  hfget.ps1, the Hugging Face fetcher a repo-id --model runs, and
                      resolve-stacks.py, which names brl-stacks' frames from radiance.map
    tools\            serve-bg.ps1 and stop-server.ps1: start a server in a console of its own and
                      stop it with Ctrl-Break, never a kill; STACKS.md: reading a hung server
    serve.ps1         start a profile, choosing cards by PCI bus (win\serve.ps1)
    PREVIEW.md        quick start and known limits (win\PREVIEW.md)
    REFERENCE.md      every setting: profile keys, the scripts' parameters, the tools' flags and the
                      environment variables, with defaults (REFERENCE.md at the repository root)
    SERVICE.md        running radiance as a Windows service (win\SERVICE.md; serve.ps1 -Install)
    SUPPORT.md        models and card counts, upstream features on Windows, known issues, and what
                      changes from Brilliance 0.2.2 (win\SUPPORT.md)
    TERMS-OF-USE.md   the terms-of-use template the server shows at /terms, for operators to adopt
                      (win\TERMS-OF-USE.md); the models' full licence texts it points to, which the
                      server shows at /terms/licenses, are in share\doc\radiance\model-licenses\
                      (win\model-licenses)
    MANIFEST.json     the source commit, toolchain, FFmpeg archive, HIP runtime and driver it was
                      built and tested with, and the SHA-256 of every file in the package

  Left out: the plugin SDK (include\, lib\; out-of-tree plugin work uses the build tree's install),
  the Linux-only fetcher (libexec\radiance\hfget.sh), and debug symbols. libexec\brilliance\hfget.ps1
  stays: radiance --model <owner/repo> fetches with it. rad-convert.exe ships; on Windows it uses no
  GPU. rad-export ships with its .cmd and the recipes.

  The build must be current (win\build.ps1 first): this installs it, it does not build it, and a
  tree whose binaries report another commit than the checkout's HEAD is refused. Every DLL
  a packaged module imports must be in the package or in Windows, or the package is refused. So is
  one whose FFmpeg DLLs are not the archive win\setup.ps1 pins (a build tree configured before
  setup replaced .toolchain\ffmpeg: reconfigure it), and one win\tools\package_audit.py finds
  errors in (licence index, OpenH264 or x264 code, release documents, internal files, hashes).

  Windows PowerShell 5.1 compatible; ASCII only.
#>
[CmdletBinding()]
param(
  [string]$BuildDir = '',
  [string]$Out = ''
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $BuildDir) { $BuildDir = Join-Path $repo 'build-win\hip' }
if (-not $Out) { $Out = Join-Path $repo 'dist' }
$BuildDir = (Resolve-Path $BuildDir).Path
$cache = Join-Path $BuildDir 'CMakeCache.txt'
if (-not (Test-Path $cache)) { throw "$BuildDir is not a configured build tree" }
if (-not (Test-Path (Join-Path $BuildDir 'bin\radiance.exe'))) { throw "no bin\radiance.exe in $BuildDir -- run win\build.ps1 first" }

$version = 'dev'
$m = Select-String -Path $cache -Pattern '^BRILLIANCE_VERSION:INTERNAL=(.+)$' | Select-Object -First 1
if ($m) { $version = $m.Matches[0].Groups[1].Value }
# The upstream pin, as the binaries report it (win/cmake/RadWindows.cmake, BRILLIANCE_UPSTREAM).
$upstreamPin = 'v2.2.0 b6b8751cccb0e834a13ae9e5afca824ee9a487aa'
$m = Select-String -Path $cache -Pattern '^BRILLIANCE_UPSTREAM:INTERNAL=(.+)$' | Select-Object -First 1
if ($m) { $upstreamPin = $m.Matches[0].Groups[1].Value }
$sha = (& git -C $repo rev-parse HEAD).Trim()
$short = $sha.Substring(0, 7)
$dirty = [bool](& git -C $repo status --porcelain --untracked-files=no)
$hip = [bool](Select-String -Path $cache -Pattern '^RAD_HIP_FOUND:INTERNAL=(1|TRUE|ON)$' -Quiet) -or
       (Test-Path (Join-Path $BuildDir 'radiance_home\kernels\libr4d.dll'))
$flavor = if ($hip) { 'win64' } else { 'win64-cpu' }
# THE BINARIES ARE FROM THIS COMMIT. The package is named after HEAD, but this installs what the
# tree last built; the commit the binaries report (/version, the startup report, a video's tag)
# is the one win/cmake/brl_identity.cmake wrote at that build. A tree built at another commit, or
# from uncommitted changes, is refused.
$idh = Join-Path $BuildDir 'brl\brl_identity.h'
if (Test-Path $idh) {
  $idm = Select-String -Path $idh -Pattern '^#define BRILLIANCE_COMMIT "([^"]+)"' | Select-Object -First 1
  $built = $(if ($idm) { $idm.Matches[0].Groups[1].Value } else { 'unknown' })
  $want = $sha.Substring(0, 12) + $(if ($dirty) { '-dirty' } else { '' })
  if ($built -ne $want) {
    throw "the binaries in $BuildDir report commit ${built}, and the checkout is at ${want}: run win\build.ps1 first"
  }
}
$name = "Brilliance-$version-$short$(if ($dirty) { '-dirty' })-$flavor"
New-Item -ItemType Directory -Force $Out | Out-Null
# Absolute, so the manifest's paths (cut from the stage's full names below) are right for a
# relative -Out too.
$Out = (Resolve-Path $Out).Path
$stage = Join-Path $Out $name
$zip = Join-Path $Out "$name.zip"

. (Join-Path $PSScriptRoot 'env.ps1') | Out-Null
$ErrorActionPreference = 'Stop'

if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null
Write-Host "== install $BuildDir -> $stage"
& cmake --install $BuildDir --prefix $stage | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'cmake --install failed' }

foreach ($p in 'include', 'lib', 'libexec\radiance') {
  $x = Join-Path $stage $p
  if (Test-Path $x) { Remove-Item -Recurse -Force $x }
}

# VC++ runtime, app-local. The whole CRT folder: the STL's satellites (msvcp140_atomic_wait, ...)
# are imported by whichever module uses that part of the library; the import check holds the
# package to it.
$bin = Join-Path $stage 'bin'
$redist = Get-ChildItem -Path (Join-Path $env:VCToolsRedistDir 'x64') -Directory -Filter 'Microsoft.VC*.CRT' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $redist) { throw 'the VC++ runtime DLLs were not found under VCToolsRedistDir' }
Get-ChildItem -Path $redist.FullName -Filter *.dll | Copy-Item -Destination $bin

# EVERY DLL A PACKAGED MODULE IMPORTS IS IN THE PACKAGE OR IN WINDOWS. Delay-loaded imports are left
# out: a plugin's delay-load of brilliance_host.dll is bound to the running exe by the hook.
$readobj = Join-Path (Split-Path (Get-Command clang++).Source) 'llvm-readobj.exe'
$mods = @(Get-ChildItem -Path $stage -Recurse -Include *.exe, *.dll)
$have = @{}
foreach ($x in $mods) { $have[$x.Name.ToLowerInvariant()] = $true }
$sys = Join-Path $env:SystemRoot 'System32'
$missing = New-Object System.Collections.Generic.List[string]
foreach ($x in $mods) {
  $inImport = $false
  foreach ($line in (& $readobj --coff-imports $x.FullName 2>$null)) {
    if ($line -match '^\s*(Delay)?Import \{') { $inImport = -not $Matches[1]; continue }
    if ($inImport -and $line -match '^\s*Name: (\S+)$') {
      $n = $Matches[1].ToLowerInvariant()
      $inImport = $false
      if ($have.ContainsKey($n) -or $n.StartsWith('api-ms-') -or $n.StartsWith('ext-ms-')) { continue }
      if (Test-Path (Join-Path $sys $n)) { continue }
      $missing.Add("$($x.Name) -> $n")
    }
  }
}
if ($missing.Count) {
  throw ("modules import DLLs that are neither in the package nor in Windows:`n  " + (($missing | Sort-Object -Unique) -join "`n  "))
}
Write-Host "== imports: every DLL the $($mods.Count) modules import is in the package or in Windows"

# THE FFMPEG DLLS ARE THE PINNED BUILD. The install rules name the FFmpeg tree the build was
# configured against, so a tree configured before win\setup.ps1 replaced .toolchain\ffmpeg (or
# against another BRILLIANCE_FFMPEG_ROOT) would ship whatever that was. The staged DLLs must be
# byte-identical to the ones in the toolchain's FFmpeg, and that tree's stamp must be the archive
# win\setup.ps1 pins.
$tc = Get-Content (Join-Path $repo '.toolchain\toolchain.json') -Raw | ConvertFrom-Json
$ffRoot = $(if ($tc.PSObject.Properties['ffmpeg_root'] -and $tc.ffmpeg_root) { $tc.ffmpeg_root } else { Join-Path $repo '.toolchain\ffmpeg' })
$setupText = Get-Content (Join-Path $PSScriptRoot 'setup.ps1') -Raw
$pin = [regex]::Match($setupText, "key='ffmpeg'; file='([^']+)'[\s\S]*?sha='([0-9a-f]{64})'")
if (-not $pin.Success) { throw 'the FFmpeg pin was not found in win\setup.ps1' }
$ffFile = $pin.Groups[1].Value
$ffSha = $pin.Groups[2].Value
$stampPath = Join-Path $ffRoot '.archive-sha256'
$stamp = $(if (Test-Path $stampPath) { ([string](Get-Content -Path $stampPath -TotalCount 1)).Trim() } else { '' })
if ($stamp -ne $ffSha) {
  throw "$ffRoot is not the pinned FFmpeg ($ffFile, $ffSha; its stamp says '$stamp'): run win\setup.ps1, then win\build.ps1 -Flavor <flavor> to reconfigure"
}
$ffDlls = @(Get-ChildItem -Path $bin -File | Where-Object { $_.Name -match '^(avcodec|avformat|avutil|swresample|swscale)-\d+\.dll$' })
if ($ffDlls.Count -ne 5) { throw "expected the five FFmpeg DLLs in bin\, found $($ffDlls.Count)" }
foreach ($d in $ffDlls) {
  $ref = Join-Path $ffRoot "bin\$($d.Name)"
  if (-not (Test-Path $ref) -or (Get-FileHash $d.FullName).Hash -ne (Get-FileHash $ref).Hash) {
    throw "bin\$($d.Name) is not the pinned FFmpeg's ($ref): the build tree was configured against another FFmpeg; reconfigure it (win\build.ps1)"
  }
}
Write-Host "== ffmpeg: the five DLLs are $ffFile's"

Copy-Item -Recurse (Join-Path $PSScriptRoot 'profiles') (Join-Path $stage 'profiles')
New-Item -ItemType Directory -Force (Join-Path $stage 'tools') | Out-Null
Copy-Item (Join-Path $PSScriptRoot 'tools\serve-bg.ps1'), (Join-Path $PSScriptRoot 'tools\stop-server.ps1') (Join-Path $stage 'tools')
Copy-Item (Join-Path $PSScriptRoot 'tools\stacks\README.md') (Join-Path $stage 'tools\STACKS.md')

# A hung server's stacks are read from outside (bin\brl-stacks.exe) and named from radiance's link
# map; the map must come from the same link as the radiance.exe beside it, so it is taken from the
# build tree with it, and both are checked present.
$map = Join-Path $BuildDir 'bin\radiance.map'
if (-not (Test-Path $map)) { throw "no bin\radiance.map in $BuildDir -- rebuild radiance (win\build.ps1)" }
if ((Get-Item $map).LastWriteTimeUtc -lt (Get-Item (Join-Path $BuildDir 'bin\radiance.exe')).LastWriteTimeUtc.AddSeconds(-5)) {
  throw 'bin\radiance.map is older than bin\radiance.exe: they are not from the same link'
}
Copy-Item $map $bin
foreach ($need in 'bin\brl-stacks.exe', 'libexec\brilliance\resolve-stacks.py', 'libexec\brilliance\hfget.ps1',
                  'bin\rad-export', 'bin\rad-export.cmd', 'share\radiance\recipes') {
  if (-not (Test-Path (Join-Path $stage $need))) { throw "the install did not produce $need" }
}
Copy-Item (Join-Path $PSScriptRoot 'PREVIEW.md') $stage
Copy-Item (Join-Path $PSScriptRoot '..\REFERENCE.md') $stage
Copy-Item (Join-Path $PSScriptRoot 'SERVICE.md') $stage
Copy-Item (Join-Path $PSScriptRoot 'SUPPORT.md') $stage
Copy-Item (Join-Path $PSScriptRoot 'TERMS-OF-USE.md') $stage
Copy-Item -Recurse (Join-Path $PSScriptRoot 'model-licenses') (Join-Path $stage 'share\doc\radiance\model-licenses')
Copy-Item (Join-Path $PSScriptRoot 'tools\reload-server.ps1') (Join-Path $stage 'tools')
Copy-Item (Join-Path $PSScriptRoot 'serve.ps1') $stage
Get-ChildItem -Path $stage -Recurse -Filter *.pdb | Remove-Item -Force

# The manifest: what this was built from and with, and every file's hash.
$clangv = ((& clang++ --version) | Select-Object -First 1)
$hipv = (Get-Item (Join-Path $bin 'amdhip64_7.dll') -ErrorAction SilentlyContinue).VersionInfo.FileVersion
$drv = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Radeon' -and $_.Name -notmatch 'Graphics$' } | Select-Object -First 1).DriverVersion
$files = [ordered]@{}
foreach ($f in (Get-ChildItem -Path $stage -Recurse -File | Sort-Object FullName)) {
  $rel = $f.FullName.Substring($stage.Length + 1).Replace('\', '/')
  $files[$rel] = (Get-FileHash -Algorithm SHA256 $f.FullName).Hash.ToLowerInvariant()
}
$manifest = [ordered]@{
  package        = $name
  brilliance     = $version
  source_commit  = $sha
  source_dirty   = $dirty
  upstream       = "radiance $upstreamPin"
  built          = (Get-Date).ToString('s')
  build_tree     = $BuildDir
  toolchain      = [ordered]@{ rocm = $tc.rocm_version; gpu_target = $tc.gpu_target; clang = $clangv;
                               msvc = $tc.msvc_toolset; windows_sdk = $tc.windows_sdk; archives = $tc.archives }
  ffmpeg         = [ordered]@{ archive = $ffFile; sha256 = $ffSha }
  hip_runtime    = $hipv
  driver_on_build_machine = $drv
  files          = $files
}
# UTF-8 without a byte-order mark (Windows PowerShell 5.1's Set-Content -Encoding UTF8 writes one,
# which strict JSON readers refuse).
[IO.File]::WriteAllText((Join-Path $stage 'MANIFEST.json'), ($manifest | ConvertTo-Json -Depth 6), (New-Object Text.UTF8Encoding $false))

# THE AUDIT (win\tools\package_audit.py), on the staged folder before it is zipped: every DLL in the
# licence index, no OpenH264 or x264 code, the release documents docs_check.py covers and no
# others, nothing internal, the manifest's hashes. Any error refuses the package.
$py = $(if ($tc.python -and (Test-Path $tc.python)) { $tc.python } else { 'python' })
& $py -I (Join-Path $PSScriptRoot 'tools\package_audit.py') $stage
if ($LASTEXITCODE -ne 0) { throw 'the package audit found errors (above); the package was not zipped' }

if (Test-Path $zip) { Remove-Item -Force $zip }
Write-Host "== zip -> $zip"
# tar.exe (bsdtar, in Windows since 1803), not Compress-Archive: Windows PowerShell 5.1's writes entry
# names with backslashes, which other unzippers take as one long file name. Windows' own, by full
# path: from Git Bash, PATH finds Git's GNU tar first, which reads "C:" as a remote host.
$tar = Join-Path $env:SystemRoot 'System32\tar.exe'
if (-not (Test-Path -LiteralPath $tar)) { throw "Windows' tar.exe is missing ($tar; in Windows since 1803); the package was not zipped" }
$top = @(Get-ChildItem -Path $stage | ForEach-Object { $_.Name })
& $tar -a -c -f $zip -C $stage @top
if ($LASTEXITCODE -ne 0) { throw "$tar failed to write the zip" }
$mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "done: $zip ($mb MiB, $($files.Count) files)"
