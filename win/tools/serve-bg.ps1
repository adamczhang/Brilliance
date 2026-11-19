#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: start radiance in a console of its own, logging to a file, for a test or a measurement.
.DESCRIPTION
  The server gets its own (minimised) console so that win\tools\stop-server.ps1 can stop it with
  Ctrl-Break, which the engine takes as SIGTERM: it drains, frees its pools and exits. A process
  holding a GPU is never force-killed (a killed one took the driver down under Brilliance 0.2.2).
  The log ends with "radiance exit status N".
  Prints the server's process id. HIP_VISIBLE_DEVICES, RADIANCE_HOME and the rest come from the
  caller's environment, or -Devices / -RadianceHome.
.EXAMPLE
  powershell -File tools\serve-bg.ps1 -Bin bin -Config profiles\qwen27-preview.yaml -Log logs\serve.log -Devices 2
#>
param(
  [Parameter(Mandatory)][string]$Bin,
  [Parameter(Mandatory)][string]$Config,
  [Parameter(Mandatory)][string]$Log,
  [string]$Devices = '',
  [string]$RadianceHome = ''
)
$ErrorActionPreference = 'Stop'
$Bin = (Resolve-Path $Bin).Path
$Config = (Resolve-Path $Config).Path
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null
$Log = [IO.Path]::GetFullPath($Log)
# Without -RadianceHome the engine finds its plugin home beside itself: share\radiance in a
# package, radiance_home in a build tree.
if ($RadianceHome) { $env:RADIANCE_HOME = $RadianceHome }
if ($Devices) { $env:HIP_VISIBLE_DEVICES = $Devices }
# cmd owns the new console and the redirection; radiance shares that console, so the Ctrl-Break
# stop-server.ps1 sends to it reaches the engine. cmd appends radiance's exit status to the log
# (delayed expansion, /v:on: %errorlevel% would be read before radiance ran). The console window is
# hidden (the owner, 2026-11-18: no windows on the desktop); the console still exists, so the
# Ctrl-Break that stop-server.ps1 sends through AttachConsole still reaches the engine. Closing a
# visible server console had ended a hung test server once (why-prod-hwq2-1).
$cmd = "`"`"$Bin\radiance.exe`" --config `"$Config`" > `"$Log`" 2>&1 & echo radiance exit status !errorlevel! >> `"$Log`"`""
$p = Start-Process -FilePath "$env:SystemRoot\System32\cmd.exe" -ArgumentList '/v:on', '/c', $cmd -WindowStyle Hidden -PassThru
# The engine is cmd's child; wait until it exists and report its id.
for ($i = 0; $i -lt 100; $i++) {
  $child = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id) AND Name='radiance.exe'" -ErrorAction SilentlyContinue
  if ($child) { break }
  Start-Sleep -Milliseconds 100
}
if (-not $child) { throw "radiance did not start (see $Log)" }
$child.ProcessId
