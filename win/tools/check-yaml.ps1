#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: run `radiance --config FILE --check` over YAML files and record each result (M1).
.EXAMPLE
  powershell -File win\tools\check-yaml.ps1 -Bin build-win\host\bin -Dir out\yaml -Log out\check.log
#>
param(
  [Parameter(Mandatory)][string]$Bin,
  [Parameter(Mandatory)][string]$Dir,
  [Parameter(Mandatory)][string]$Log
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$env:RADIANCE_HOME = Join-Path (Split-Path -Parent $Bin) 'radiance_home'
Remove-Item $Log -ErrorAction SilentlyContinue
foreach ($y in Get-ChildItem $Dir -Filter *.yaml | Sort-Object Name) {
  $tmp = Join-Path $env:TEMP "brl-check-$($y.BaseName).txt"
  cmd.exe /c "`"$Bin\radiance.exe`" --config `"$($y.FullName)`" --check > `"$tmp`" 2>&1"
  $code = $LASTEXITCODE
  $txt = Get-Content $tmp -Raw
  Add-Content -Path $Log -Value "===== $($y.Name) -> exit $code`n$txt"
  $last = @($txt -split "`n" | Where-Object { $_ -match '\S' } | Select-Object -Last 1)
  "{0,-22} exit {1,-3} {2}" -f $y.Name, $code, $(if ($last) { $last[0].Trim() } else { '' })
}
