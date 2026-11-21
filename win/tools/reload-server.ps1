#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: ask a running radiance to re-read its YAML file -- what `kill -HUP <pid>` does on Linux.
.DESCRIPTION
  Windows has no SIGHUP. radiance watches a named event of its own (os::on_reload,
  win/os/os_win32.cpp), Global\Brilliance-Reload-<pid> when it could create one (a service) or
  Local\Brilliance-Reload-<pid> (a console server); setting it makes the fleet reload the file, as
  POST /admin/reload does over HTTP. Changed models reload, removed ones unload, new ones are added;
  what the server refuses (server: and cards: changes need a restart) is said in its log.

  The file reread is the one radiance was started on, its --config, which this prints first: the
  profile itself for a server serve.ps1 started, %ProgramData%\Brilliance\<Name>\<Name>.yaml for a
  service. serve.ps1's -Listen, -Port and -ApiKey (BRILLIANCE_SERVER_*) apply again over it. A
  server started by an older serve.ps1 runs on a temporary copy of its profile, which an edit to the
  profile does not reach; this says so, and a restart with the current serve.ps1 is the remedy.
.EXAMPLE
  powershell -File win\tools\reload-server.ps1 -ProcessId 1234
.NOTES
  Exit status: 0 the event was set; 2 no such event (not radiance, or a build without on_reload).
  A service's Global\ event is opened, and its command line read, from an elevated shell.
#>
param([Parameter(Mandatory)][int]$ProcessId)
$ErrorActionPreference = 'Stop'

# The file the server will reread: --config on its command line.
$cl = $null
try { $cl = (Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction Stop).CommandLine } catch { }
if ($cl -and $cl -match '--config\s+("([^"]*)"|(\S+))') {
  $file = if ($Matches[2]) { $Matches[2] } else { $Matches[3] }
  Write-Host "reload: process $ProcessId rereads $file"
  $tmp = [IO.Path]::GetFullPath($env:TEMP).TrimEnd('\')
  $leaf = [IO.Path]::GetFileName($file)
  $dir = [IO.Path]::GetDirectoryName($file)
  if ($leaf -match '^brilliance-.*-\d+\.yaml$' -and $dir -and [string]::Equals([IO.Path]::GetFullPath($dir).TrimEnd('\'), $tmp, [StringComparison]::OrdinalIgnoreCase)) {
    Write-Host "reload: note: that is a temporary copy an older serve.ps1 made of the profile; edits to the profile itself do not reach it. Stop the server (tools\stop-server.ps1) and start it again with the current serve.ps1, which runs radiance on the profile itself"
  }
} elseif (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
  Write-Host "reload: process $ProcessId's command line could not be read here (a service's needs an elevated shell); it rereads the file it was started on"
}

foreach ($scope in 'Global', 'Local') {
  $name = "$scope\Brilliance-Reload-$ProcessId"
  $ev = $null
  if ([System.Threading.EventWaitHandle]::TryOpenExisting($name, [ref]$ev)) {
    [void]$ev.Set()
    $ev.Dispose()
    Write-Host "reload: set $name; the server's log says what the reload did"
    exit 0
  }
}
Write-Host "reload: process $ProcessId has no Brilliance-Reload event (is it radiance?)"
exit 2
