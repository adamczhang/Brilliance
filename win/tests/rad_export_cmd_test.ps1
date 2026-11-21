#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: bin\rad-export.cmd in an unpacked package finds the rad-convert beside it, with that
  folder not on PATH (win\libexec\rad-export.cmd).
.DESCRIPTION
  ctest runs it (win\tests\CMakeLists.txt). It lays out a package's bin\ and share\radiance\recipes
  in a temporary folder -- rad-export.cmd and rad-export from the tree, a stub rad-convert.cmd that
  only says it ran, the tree's recipes -- and runs `rad-export.cmd --list-recipes --json` with a PATH
  that holds no rad-convert. Exit 0 when every check holds, 1 otherwise, 77 (skipped) with no Python.
#>
param()
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$script:failed = 0
$script:checks = 0
function Check([bool]$ok, [string]$what) {
  $script:checks++
  if (-not $ok) { $script:failed++; Write-Host "  FAIL $what" }
}
if (-not (Get-Command py -ErrorAction SilentlyContinue) -and -not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Host 'rad_export_cmd_test: no Python 3 (py or python) on PATH; skipped'
  exit 77
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("brl-rad-export-test-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
$pkg = Join-Path $tmp 'Brilliance pkg'   # a space in the path, as an unpacked ZIP may have
$bin = Join-Path $pkg 'bin'
$recipes = Join-Path $pkg 'share\radiance\recipes'
New-Item -ItemType Directory $bin, $recipes | Out-Null
try {
  Copy-Item (Join-Path $repo 'win\libexec\rad-export.cmd') $bin
  Copy-Item (Join-Path $repo 'tools\rad-export') $bin
  [IO.File]::WriteAllText((Join-Path $bin 'rad-convert.cmd'), "@echo stub rad-convert %*`r`n@exit /b 0`r`n")
  $src = @(Get-ChildItem (Join-Path $repo 'data\recipes') -Filter '*.recipe' | Select-Object -First 3)
  $src | Copy-Item -Destination $recipes
  # PATH without any folder that holds a rad-convert (a build tree's bin\ may be on it)
  $path = (($env:PATH -split ';') | Where-Object { $_ -and -not (Test-Path -LiteralPath (Join-Path $_ 'rad-convert.exe')) -and
                                                   -not (Test-Path -LiteralPath (Join-Path $_ 'rad-convert.cmd')) }) -join ';'
  $saved = $env:PATH
  $env:PATH = $path
  try {
    $ErrorActionPreference = 'Continue'
    $o = & (Join-Path $bin 'rad-export.cmd') --list-recipes --json 2>&1 | ForEach-Object { "$_" }
    $rc = $LASTEXITCODE
  } finally { $env:PATH = $saved; $ErrorActionPreference = 'Stop' }
  $text = $o -join "`n"
  Check ($rc -eq 0) "rad-export.cmd --list-recipes exits 0 (got $rc):`n$text"
  Check (-not ($text -match 'no rad-convert on PATH')) "it finds the rad-convert beside it:`n$text"
  foreach ($r in $src) { Check ($text.Contains($r.BaseName)) "it lists the recipe beside it, $($r.BaseName):`n$text" }
  # --rad-convert still chooses another
  $other = Join-Path $tmp 'other'
  New-Item -ItemType Directory $other | Out-Null
  $env:PATH = $path
  try {
    $ErrorActionPreference = 'Continue'
    $o = & (Join-Path $bin 'rad-export.cmd') --list-recipes --rad-convert (Join-Path $other 'rad-convert.exe') 2>&1 | ForEach-Object { "$_" }
    $rc = $LASTEXITCODE
  } finally { $env:PATH = $saved; $ErrorActionPreference = 'Stop' }
  Check (($o -join "`n") -match 'no recipes next to') "--rad-convert PATH is still honoured:`n$($o -join "`n")"
  # ...and that refusal is rad-export's exit status (die: 1), not the wrapper's 0
  Check ($rc -eq 1) "a failed rad-export fails rad-export.cmd (exit $rc, expected 1)"
  # a Python exit status passes through as it is: an option argparse refuses (2)
  $env:PATH = $path
  try {
    $ErrorActionPreference = 'Continue'
    $o = & (Join-Path $bin 'rad-export.cmd') --no-such-option 2>&1 | ForEach-Object { "$_" }
    $rc = $LASTEXITCODE
  } finally { $env:PATH = $saved; $ErrorActionPreference = 'Stop' }
  Check ($rc -eq 2) "argparse's refusal is rad-export.cmd's exit status (exit $rc, expected 2):`n$($o -join "`n")"
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
Write-Host "rad_export_cmd_test: $($script:checks - $script:failed) of $($script:checks) checks hold"
if ($script:failed) { exit 1 }
exit 0
