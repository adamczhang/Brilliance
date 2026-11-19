#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: stop a radiance server gracefully -- Ctrl-Break to its console -- and wait for it.
.DESCRIPTION
  A helper process detaches from this console, attaches to the server's, shields itself and sends
  CTRL_BREAK_EVENT to every process on that console. The engine's handler (os::on_terminate) takes
  it as SIGTERM: it drains in-flight requests, frees its pools and exits.

  Ctrl-Break, not Ctrl-C: a process inherits "Ctrl-C ignored" from whatever started it (a shell
  started in a process group of its own passes it to every descendant), and such a server never
  sees a Ctrl-C. Ctrl-Break cannot be ignored that way. Seen 2026-11-16: a server started from an
  agent's shell ignored the Ctrl-C and kept serving.

  Nothing is force-killed: a server still alive after -WaitS is reported (exit code 3) and left
  for a person to decide about.
.EXAMPLE
  powershell -File win\tools\stop-server.ps1 -ProcessId 1234
#>
param([Parameter(Mandatory)][int]$ProcessId, [int]$WaitS = 120)
$ErrorActionPreference = 'Continue'
if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { "stop: $ProcessId is not running"; exit 0 }
$helper = @"
`$cs = @'
using System; using System.Runtime.InteropServices;
public static class BrlCtl {
  public delegate bool Handler(uint type);
  [DllImport("kernel32.dll")] public static extern bool FreeConsole();
  [DllImport("kernel32.dll")] public static extern bool AttachConsole(uint pid);
  [DllImport("kernel32.dll")] static extern bool SetConsoleCtrlHandler(Handler h, bool add);
  [DllImport("kernel32.dll")] public static extern bool GenerateConsoleCtrlEvent(uint ev, uint group);
  static readonly Handler keep = delegate(uint type) { return true; };
  public static bool Shield() { return SetConsoleCtrlHandler(keep, true); }
}
'@
Add-Type -TypeDefinition `$cs -ErrorAction Stop
[void][BrlCtl]::FreeConsole()
if (-not [BrlCtl]::AttachConsole($ProcessId)) { exit 2 }
if (-not [BrlCtl]::Shield()) { exit 5 }
if (-not [BrlCtl]::GenerateConsoleCtrlEvent(1, 0)) { exit 4 }
Start-Sleep -Milliseconds 500
exit 0
"@
# Encoded: Start-Process joins its arguments with spaces and does not quote them, so a -Command
# script loses its double quotes on the way (the C# above would not compile and nothing is sent).
$enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($helper))
$p = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') -ArgumentList '-NoProfile', '-NonInteractive', '-EncodedCommand', $enc -WindowStyle Hidden -PassThru -Wait
if ($p.ExitCode -ne 0) { "stop: the helper could not reach $ProcessId's console (exit $($p.ExitCode): 2 attach, 4 send, 5 shield)"; exit 1 }
"stop: Ctrl-Break to $ProcessId's console (helper exit $($p.ExitCode))"
$t0 = Get-Date
while (((Get-Date) - $t0).TotalSeconds -lt $WaitS -and (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { Start-Sleep -Milliseconds 250 }
if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
  "stop: $ProcessId still running after $WaitS s. Not killed; ask the user."
  exit 3
}
"stop: $ProcessId exited in {0:N1} s" -f ((Get-Date) - $t0).TotalSeconds
exit 0
