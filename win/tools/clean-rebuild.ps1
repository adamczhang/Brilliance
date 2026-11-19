#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: rebuild the package from a clean checkout with the recorded toolchain and nothing else
  (the release candidate's rebuild check), then compare it with a
  package of the same commit.

.DESCRIPTION
    powershell -File win\tools\clean-rebuild.ps1 -Dest out\clean [-Jobs 8] [-Compare dist\X.zip]

  1. git clone of this checkout's HEAD into -Dest (a path of its own; it must not exist yet)
  2. the pinned archives hard-linked from this checkout's .toolchain\downloads (same volume; no
     network: setup.ps1 then installs only from them, each checked against its SHA-256)
  3. win\setup.ps1, win\build.ps1 -Flavor hip, win\package.ps1, all inside the clone
  4. with -Compare: the new package's MANIFEST.json against that package's, file by file

  Each step's output goes to <Dest>.log; the last line says how it ended. Touches no GPU: the
  build compiles device code but runs nothing on a card.
#>
param(
  [Parameter(Mandatory)][string]$Dest,
  [int]$Jobs = 8,
  [string]$Compare = ''
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Dest = [IO.Path]::GetFullPath($Dest)
$log = "$Dest.log"
function Log([string]$s) { "$(Get-Date -Format 'h:mm:ss tt')  $s" | Out-File -Append -Encoding ascii $log }
if (Test-Path $Dest) { throw "$Dest exists; give a new path" }
$ps = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
function Step([string]$what, [string]$script, [string[]]$argv) {
  Log "== $what"
  $t = Get-Date
  $line = ($argv | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
  cmd /c "`"$ps`" -NoProfile -ExecutionPolicy Bypass -File `"$Dest\$script`" $line >> `"$log`" 2>&1"
  $rc = $LASTEXITCODE
  Log "== ${what}: exit $rc in $([int]((Get-Date) - $t).TotalSeconds) s"
  if ($rc -ne 0) { Log "STOPPED at $what"; exit 1 }
}

$sha = (& git -C $repo rev-parse HEAD).Trim()
Log "clean rebuild of $sha into $Dest"
& git clone --quiet --no-hardlinks $repo $Dest 2>&1 | Out-File -Append -Encoding ascii $log
& git -C $Dest checkout --quiet $sha 2>&1 | Out-File -Append -Encoding ascii $log
New-Item -ItemType Directory -Force (Join-Path $Dest '.toolchain\downloads') | Out-Null
foreach ($f in Get-ChildItem (Join-Path $repo '.toolchain\downloads') -File) {
  New-Item -ItemType HardLink -Path (Join-Path $Dest ".toolchain\downloads\$($f.Name)") -Target $f.FullName | Out-Null
}
Log "linked $((Get-ChildItem (Join-Path $Dest '.toolchain\downloads') -File).Count) pinned archive(s)"

Step 'setup' 'win\setup.ps1' @()
Step 'build (hip)' 'win\build.ps1' @('-Flavor', 'hip', '-Jobs', "$Jobs")
Step 'package' 'win\package.ps1' @()

$zip = Get-ChildItem (Join-Path $Dest 'dist') -Filter '*-win64.zip' | Sort-Object LastWriteTime | Select-Object -Last 1
Log "package: $($zip.FullName)"
if ($Compare) {
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  function Manifest([string]$z) {
    $a = [IO.Compression.ZipFile]::OpenRead($z)
    try {
      $e = $a.Entries | Where-Object { $_.FullName -match '(^|/)MANIFEST\.json$' } | Select-Object -First 1
      $r = New-Object IO.StreamReader($e.Open(), [Text.Encoding]::UTF8)
      try { return ($r.ReadToEnd() | ConvertFrom-Json) } finally { $r.Close() }
    } finally { $a.Dispose() }
  }
  $a = Manifest ([IO.Path]::GetFullPath($Compare)); $b = Manifest $zip.FullName
  $fa = @{}; foreach ($p in $a.files.PSObject.Properties) { $fa[$p.Name] = $p.Value }
  $fb = @{}; foreach ($p in $b.files.PSObject.Properties) { $fb[$p.Name] = $p.Value }
  $only_a = @($fa.Keys | Where-Object { -not $fb.ContainsKey($_) })
  $only_b = @($fb.Keys | Where-Object { -not $fa.ContainsKey($_) })
  $same = @($fa.Keys | Where-Object { $fb.ContainsKey($_) -and $fa[$_] -eq $fb[$_] })
  $diff = @($fa.Keys | Where-Object { $fb.ContainsKey($_) -and $fa[$_] -ne $fb[$_] } | Sort-Object)
  Log "compare with ${Compare}: $($fa.Count) vs $($fb.Count) files; identical $($same.Count); differ $($diff.Count); only in the first $($only_a.Count); only in the clean build $($only_b.Count)"
  Log "source commits: $($a.source_commit) / $($b.source_commit); toolchain equal: $((($a.toolchain | ConvertTo-Json -Depth 5) -eq ($b.toolchain | ConvertTo-Json -Depth 5)))"
  $kinds = $diff | Group-Object { [IO.Path]::GetExtension($_).ToLowerInvariant() } | ForEach-Object { "$($_.Name) $($_.Count)" }
  Log "differing files by extension: $($kinds -join ', ')"
  foreach ($d in ($diff | Where-Object { $_ -notmatch '\.(exe|dll|map)$' })) { Log "  differs: $d" }
  foreach ($d in ($only_a + $only_b)) { Log "  only one side: $d" }
}
Log 'DONE'
