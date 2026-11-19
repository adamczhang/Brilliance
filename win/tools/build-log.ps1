#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: build an already configured tree in a build shell, keep going past failures, and
  write the whole log -- the porting loop's way of collecting every error in one pass.
.EXAMPLE
  powershell -File win\tools\build-log.ps1 -Build build-win\host -Log out\build.log
#>
param(
  [Parameter(Mandatory)][string]$Build,
  [Parameter(Mandatory)][string]$Log,
  [int]$Jobs = [Environment]::ProcessorCount,
  [string]$Target = ''
)
$ErrorActionPreference = 'Continue'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'env.ps1') | Out-Null
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null
$t = Get-Date
# Through cmd's redirection, which leaves the log readable while it is written (PowerShell's
# Set-Content holds it exclusively, so progress could not be watched).
cmd.exe /c "ninja -C `"$Build`" -k 0 -j $Jobs $Target > `"$Log`" 2>&1"
$rc = $LASTEXITCODE
$errs = @(Select-String -Path $Log -Pattern ': error: ' ).Count
$failed = @(Select-String -Path $Log -Pattern '^FAILED: ').Count
"ninja exit $rc; {0} FAILED steps; {1} 'error:' lines; {2:N0} s" -f $failed, $errs, ((Get-Date) - $t).TotalSeconds
