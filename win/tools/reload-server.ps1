#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: ask a running radiance to re-read its YAML file -- what `kill -HUP <pid>` does on Linux.
.DESCRIPTION
  Windows has no SIGHUP. radiance watches a named event of its own (os::on_reload,
  win/os/os_win32.cpp), Global\Brilliance-Reload-<pid> when it could create one (a service) or
  Local\Brilliance-Reload-<pid> (a console server); setting it makes the fleet reload the file, as
  POST /admin/reload does over HTTP. Changed models reload, removed ones unload, new ones are added;
  what the server refuses is said in its log.
.EXAMPLE
  powershell -File win\tools\reload-server.ps1 -ProcessId 1234
.NOTES
  Exit status: 0 the event was set; 2 no such event (not radiance, or a build without on_reload).
  A service's Global\ event is opened from an elevated shell.
#>
param([Parameter(Mandatory)][int]$ProcessId)
$ErrorActionPreference = 'Stop'
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
