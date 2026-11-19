#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: start radiance on a named profile, choosing the cards by PCI bus and the listening
  address and key policy from the command line -- in this console, in one of its own, or as a
  Windows service.

.DESCRIPTION
    .\serve.ps1 qwen27-preview                          loopback, the profile's port, its card choice
    .\serve.ps1 qwen27-preview -Bus 3                   the card on PCI bus 3
    .\serve.ps1 qwen27-preview -Listen 0.0.0.0 -ApiKey off     the local network, no key
    .\serve.ps1 qwen27-preview -ApiKey env:RADIANCE_API_KEY    a key read from that variable
    .\serve.ps1 qwen27-preview -Check                   validate the YAML and exit
    .\serve.ps1 qwen27-preview -Env GPU_X=1,HIP_Y=2     also set these for the server (ENVIRONMENT below)
    .\serve.ps1 qwen27-preview -WhatIf                  print what a start would set and run; start nothing
    .\serve.ps1 qwen27-preview -Background -Log srv.log start it in a console of its own (Ctrl-Break
                                                        stops it: tools\stop-server.ps1 -ProcessId N)
    .\serve.ps1 qwen27-preview -Bus 3 -Install          register it as a service (elevated)
    .\serve.ps1 qwen27-preview -Install -WhatIf         print what -Install would run; change nothing
    .\serve.ps1 qwen27-preview -Uninstall               stop it (the drain) and remove it (elevated)

  A thin selector over radiance's own YAML: the profile (profiles\<name>.yaml, or a path) is copied
  to a temporary file with the server block's host, port and key lines replaced by what was asked,
  and radiance runs on that. Everything else is the profile's. Nothing here is a second
  configuration language.

  CARDS BY PCI BUS: HIP numbers devices in its own order (on the test machine: the iGPU, then buses 6, 3, 9, 32),
  so -Bus is translated to HIP indices by asking the HIP runtime beside radiance for each device's
  PCI bus, and -Device takes HIP indices as they are. Neither given: HIP_VISIBLE_DEVICES as the
  caller set it, else every discrete card HIP shows, the integrated GPU left out.

  Runs in this console by default, so Ctrl-C stops it the way the engine expects: it drains and
  exits. Nothing here kills a server.

  ENVIRONMENT: radiance's YAML has no environment key, so a profile names the variables its server
  needs in a marked comment line, which radiance itself skips as a comment:
      # brilliance-env: BRILLIANCE_MAILBOX_MEM=noncoherent-slots
  (NAME=VALUE, several on one line separated by commas, as many lines as needed; the rest of the
  line is the value list, so no trailing comment.) -Env NAME=VALUE[,NAME=VALUE] adds to them or
  overrides them for this start. Only the HIP runtime's and the engine's own names are taken: GPU_*,
  HIP_*, AMD_*, BRILLIANCE_*, RADIANCE_*. Not HIP_VISIBLE_DEVICES (-Bus or -Device choose the
  cards), and no name that looks like a secret (KEY, TOKEN, SECRET, PASSWORD: -ApiKey env:NAME keeps
  a key out of files and the registry). A value has no comma. The startup output names each one set.
  They are set for radiance only (this shell's own values come back when it exits) and, with
  -Install, go into the service's environment beside the cards.

  AS A SERVICE (win\SERVICE.md has the whole story and the qualification steps). -Install writes the
  derived YAML to %ProgramData%\Brilliance\<Name>\<Name>.yaml and registers the service <Name>
  (default Brilliance-<profile>) to run, as LocalSystem,
      radiance.exe --service <Name> --config <that YAML> [--env-from-file VAR=<key file>]
  with a delayed automatic start (-Manual: on demand), and its log beside the YAML. -ApiKey env:NAME
  copies this shell's NAME into %ProgramData%\Brilliance\keys\<Name>.key, which only SYSTEM and
  Administrators can read; the YAML says `api-key-env: NAME`, and the service sets NAME from the
  file. The key is never on a command line or in the registry. The cards (-Bus, -Device, or this
  shell's HIP_VISIBLE_DEVICES), BRILLIANCE_HF_CACHE and the profile's and -Env's variables
  (ENVIRONMENT above) go into the service's own environment.
  On a failure nobody asked for, the service is restarted once, after 2 minutes, and not again
  within a day (-NoRestart: never), so a card that is in trouble is not loaded again and again. A stop
  is the engine's drain; one that has not finished in -StopTimeoutS (180) is reported in the log and
  never ended by force. -Uninstall stops the service, waits up to -WaitS (300) for the drain, and
  only then deletes it; the YAML and the logs stay. -WhatIf prints every step either would take, in
  order, exactly as it would run, and changes nothing; it needs no elevation. Without -Install or
  -Uninstall, -WhatIf prints what a start would set (the cards, the ENVIRONMENT) and the command
  line it would run, and starts nothing.

  Windows PowerShell 5.1 compatible; ASCII only.
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0, Mandatory)][string]$Profile,
  [string]$Bus = '',      # comma-separated PCI bus numbers (decimal, as Windows shows them)
  [string]$Device = '',   # comma-separated HIP indices
  [string]$Listen = '',
  [int]$Port = 0,
  [string]$ApiKey = '',
  [switch]$Check,
  [switch]$Background,
  [string]$Log = '',
  [string]$Exe = '',      # the radiance.exe to run (default: bin\ beside this script, else build-win\hip\bin)
  # NAME=VALUE pairs for the server's environment. An array (-Env A=1,B=2 from PowerShell) or one
  # comma-joined string (the same words through powershell -File) both arrive here: split on commas.
  [string[]]$Env = @(),
  # the Windows service (win\SERVICE.md)
  [switch]$Install,
  [switch]$Uninstall,
  [switch]$WhatIf,
  [string]$Name = '',     # the service's name (default Brilliance-<profile>)
  [switch]$Manual,        # start on demand rather than delayed-automatic at boot
  [switch]$NoRestart,     # no recovery action at all
  [int]$StopTimeoutS = 180,
  [int]$WaitS = 300
)
$ErrorActionPreference = 'Stop'

$service = $Install -or $Uninstall
if ($Install -and $Uninstall) { throw '-Install and -Uninstall: one at a time' }
if ($WhatIf -and $Check) { throw '-WhatIf and -Check: one at a time (-Check starts no server either)' }
if ($Uninstall -and $Env.Count) { throw '-Env goes with a start or -Install, not -Uninstall' }
if ($service -and ($Check -or $Background)) { throw '-Install and -Uninstall take neither -Check nor -Background' }
if (-not $service -and ($Name -or $Manual -or $NoRestart)) { throw '-Name, -Manual and -NoRestart go with -Install or -Uninstall' }
if ($StopTimeoutS -lt 1 -or $StopTimeoutS -gt 86400) { throw '-StopTimeoutS takes 1 to 86400 seconds' }

# The service's name: the profile's, unless one is given. Letters, digits, '.', '_' and '-' only, so
# it is the same word in sc.exe, the registry and the file names below.
$stem = [IO.Path]::GetFileNameWithoutExtension($Profile)
if (-not $Name) { $Name = 'Brilliance-' + ($stem -replace '[^A-Za-z0-9._-]', '-') }
if ($service -and $Name -notmatch '^[A-Za-z0-9._-]{1,200}$') { throw "-Name '$Name': a service name here is letters, digits, '.', '_' and '-'" }

# ===================================================================================== service steps
# -Install and -Uninstall first decide every step from what is there now (reading only), then either
# print the list (-WhatIf) or run it, the same list either way: what -WhatIf prints is what runs.
# A step is a native command (run exactly as printed, its arguments one string with no PowerShell
# requoting) or a PowerShell statement (printed as written).
$SidSystem = 'S-1-5-18'; $SidAdmins = 'S-1-5-32-544'; $SidUsers = 'S-1-5-32-545'
$sys32 = Join-Path $env:SystemRoot 'System32'
$scExe = Join-Path $sys32 'sc.exe'          # not `sc`: that is Set-Content in PowerShell
$icaclsExe = Join-Path $sys32 'icacls.exe'
$dataDir = Join-Path $env:ProgramData 'Brilliance'
$svcDir = Join-Path $dataDir $Name
$keyDir = Join-Path $dataDir 'keys'
$keyFile = Join-Path $keyDir "$Name.key"
$regKey = "HKLM:\SYSTEM\CurrentControlSet\Services\$Name"

$steps = New-Object System.Collections.Generic.List[object]
function Add-Note([string]$text) { $steps.Add([pscustomobject]@{ Kind = 'note'; Text = $text }) }
function Add-Native([string]$exe, [string]$argLine, [int[]]$ok = @(0)) {
  $steps.Add([pscustomobject]@{ Kind = 'native'; Exe = $exe; Args = $argLine; Ok = $ok })
}
function Add-Ps([string]$shown, [scriptblock]$do, [hashtable]$data = @{}, [string]$body = '') {
  $steps.Add([pscustomobject]@{ Kind = 'ps'; Shown = $shown; Do = $do; Data = $data; Body = $body })
}

# One argument quoted so CommandLineToArgvW -- the CRT's split, and the SCM's of a service's command
# line -- gives it back unchanged (os::quote_arg in win\os).
function Quote-Arg([string]$a) {
  if ($a.Length -gt 0 -and $a -notmatch '[\s"]') { return $a }
  $sb = New-Object System.Text.StringBuilder
  [void]$sb.Append('"')
  $bs = 0
  foreach ($c in $a.ToCharArray()) {
    if ($c -eq '\') { $bs++; continue }
    if ($c -eq '"') { [void]$sb.Append('\' * (2 * $bs + 1)).Append('"'); $bs = 0; continue }
    if ($bs) { [void]$sb.Append('\' * $bs); $bs = 0 }
    [void]$sb.Append($c)
  }
  [void]$sb.Append('\' * (2 * $bs)).Append('"')
  return $sb.ToString()
}
function Quote-Ps([string]$s) { "'" + $s.Replace("'", "''") + "'" }

function Invoke-Native([string]$exe, [string]$argLine, [int[]]$ok) {
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $exe; $psi.Arguments = $argLine; $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true
  $p = [System.Diagnostics.Process]::Start($psi)
  $errTask = $p.StandardError.ReadToEndAsync()
  $out = $p.StandardOutput.ReadToEnd()
  $p.WaitForExit()
  $text = ($out + $errTask.Result).Trim()
  if ($text) { $text -split "`r?`n" | ForEach-Object { Write-Host "     $_" } }
  if ($ok -notcontains $p.ExitCode) { throw "$([IO.Path]::GetFileName($exe)) exited with $($p.ExitCode)" }
}

function Show-Steps([string]$what) {
  Write-Host "serve: $what -WhatIf: these steps would run, in order, from an elevated PowerShell. Nothing was changed."
  $i = 0
  foreach ($s in $steps) {
    if ($s.Kind -eq 'note') { Write-Host "     # $($s.Text)"; continue }
    $i++
    $line = if ($s.Kind -eq 'native') { "$(Quote-Arg $s.Exe) $($s.Args)" } else { $s.Shown }
    Write-Host ("{0,3}  {1}" -f $i, $line)
    if ($s.Body) { $s.Body.TrimEnd("`n") -split "`n" | ForEach-Object { Write-Host "       | $_" } }
  }
}

function Invoke-Steps {
  $i = 0
  foreach ($s in $steps) {
    if ($s.Kind -eq 'note') { continue }
    $i++
    $line = if ($s.Kind -eq 'native') { "$(Quote-Arg $s.Exe) $($s.Args)" } else { $s.Shown }
    Write-Host ("{0,3}  {1}" -f $i, $line)
    if ($s.Kind -eq 'native') { Invoke-Native $s.Exe $s.Args $s.Ok }
    else { & $s.Do $s.Data }
  }
}

function Assert-Admin([string]$what) {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  if (-not (New-Object Security.Principal.WindowsPrincipal $id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "$what needs an elevated PowerShell (Run as administrator). -WhatIf prints what it would do without one."
  }
}

# %PROGRAMDATA% LETS ANY USER MAKE A FOLDER, WHICH ITS MAKER THEN OWNS, and its ACL lets Users add
# files and folders inside the folders below it. A Brilliance or keys folder made first by an
# ordinary user, or a junction planted in its place, would hand that user the API key or the YAML a
# LocalSystem service runs (a plugin path in it is code running as SYSTEM). So each folder is
# checked -- not a reparse point, owned by Administrators or SYSTEM -- and given a protected ACL:
# SYSTEM and Administrators full control, Users read where -UsersRead says (the YAML and the logs),
# nothing on keys. (Brilliance 0.2.2's -Install did the same to the same folders.)
function Get-Attributes([string]$path) {
  try { return [System.IO.File]::GetAttributes($path) }
  catch [System.IO.FileNotFoundException], [System.IO.DirectoryNotFoundException] { return $null }
}
function Assert-SafeDir([string]$dir) {
  $at = Get-Attributes $dir
  if ($null -eq $at) { throw "refusing: $dir is not there" }
  if ($at -band [IO.FileAttributes]::ReparsePoint) { throw "refusing: $dir is a junction or symbolic link. Remove it (rmdir, not del) and run again" }
  if (-not ($at -band [IO.FileAttributes]::Directory)) { throw "refusing: $dir is a file, not a folder. Remove it and run again" }
  $owner = (Get-Acl -LiteralPath $dir).GetOwner([Security.Principal.SecurityIdentifier])
  if ($owner.Value -notin @($SidSystem, $SidAdmins)) {
    $who = $owner.Value
    try { $who = $owner.Translate([Security.Principal.NTAccount]).Value } catch { }
    throw "refusing: $dir is owned by $who, not Administrators or SYSTEM, so someone else may have made it. Check what is in it, remove it, and run again"
  }
}
function Assert-ProtectedDir([string]$dir, [bool]$usersRead) {
  Assert-SafeDir $dir
  $acl = Get-Acl -LiteralPath $dir
  if (-not $acl.AreAccessRulesProtected) { throw "refusing: $dir still inherits its permissions" }
  $rx = [int]([Security.AccessControl.FileSystemRights]::ReadAndExecute -bor [Security.AccessControl.FileSystemRights]::Synchronize)
  foreach ($r in $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
    $sid = $r.IdentityReference.Value
    if ($r.AccessControlType -ne 'Allow' -or $sid -in @($SidSystem, $SidAdmins)) { continue }
    if ($usersRead -and $sid -eq $SidUsers -and ([int]$r.FileSystemRights -band -bnot $rx) -eq 0) { continue }
    throw "refusing: $dir also grants $sid $($r.FileSystemRights); remove that entry (icacls $dir /remove *$sid) and run again"
  }
}
function Add-ProtectDir([string]$dir, [bool]$usersRead) {
  if ($null -eq (Get-Attributes $dir)) {
    # No -Force: it fails if something appeared there in the meantime.
    Add-Ps "New-Item -ItemType Directory -Path $(Quote-Ps $dir)" { param($d) New-Item -ItemType Directory -Path $d.Dir | Out-Null } @{ Dir = $dir }
    Add-Native $icaclsExe "$(Quote-Arg $dir) /setowner *$SidAdmins"
  } else {
    Add-Ps "Assert-SafeDir $(Quote-Ps $dir)" { param($d) Assert-SafeDir $d.Dir } @{ Dir = $dir }
  }
  $grants = "*${SidSystem}:(OI)(CI)F *${SidAdmins}:(OI)(CI)F"
  if ($usersRead) { $grants += " *${SidUsers}:(OI)(CI)RX" }
  Add-Native $icaclsExe "$(Quote-Arg $dir) /inheritance:r /grant:r $grants"
  # Read back, not assumed: an explicit entry icacls leaves alone could still let someone write.
  Add-Ps "Assert-ProtectedDir $(Quote-Ps $dir) `$$usersRead" { param($d) Assert-ProtectedDir $d.Dir $d.UsersRead } @{ Dir = $dir; UsersRead = $usersRead }
}
# A file is replaced, not overwritten: one already there may predate the protected folder and carry
# an ACL of its own. The new one has only the folder's.
function Add-ReplaceFile([string]$path, [string]$text, [string]$shownText, [string]$body) {
  $at = Get-Attributes $path
  if ($null -ne $at) {
    if ($at -band ([IO.FileAttributes]::ReparsePoint -bor [IO.FileAttributes]::Directory)) { throw "refusing: $path is a link or a folder. Remove it and run again" }
    Add-Ps "Remove-Item -LiteralPath $(Quote-Ps $path) -Force" { param($d) Remove-Item -LiteralPath $d.Path -Force } @{ Path = $path }
  }
  # UTF-8 without a byte-order mark, as the YAML reader and the key reader expect.
  Add-Ps "[IO.File]::WriteAllText($(Quote-Ps $path), $shownText)" { param($d) [IO.File]::WriteAllText($d.Path, $d.Text) } @{ Path = $path; Text = $text } $body
}

# The SCM settings sc.exe cannot make: failure actions with an explicit "none" (sc.exe's own
# parser drops an empty action, and the SCM repeats the LAST action for every later failure, so a
# list ending in restart restarts forever), the flag that counts a non-zero exit as a failure, and
# the time the SCM waits at shutdown (PRESHUTDOWN) for the drain. ChangeServiceConfig2W, as
# Brilliance 0.2.2's installer called it.
function Add-BrlSvcType {
  if ('BrlSvc' -as [type]) { return }
  Add-Type -TypeDefinition @'
using System; using System.ComponentModel; using System.Runtime.InteropServices;
public static class BrlSvc {
  [StructLayout(LayoutKind.Sequential)] struct FailureActions {
    public uint dwResetPeriod; public IntPtr lpRebootMsg; public IntPtr lpCommand; public uint cActions; public IntPtr lpsaActions; }
  [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)] static extern IntPtr OpenSCManagerW(string machine, string db, uint access);
  [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)] static extern IntPtr OpenServiceW(IntPtr scm, string name, uint access);
  [DllImport("advapi32.dll", SetLastError = true)] static extern bool CloseServiceHandle(IntPtr h);
  [DllImport("advapi32.dll", SetLastError = true, EntryPoint = "ChangeServiceConfig2W")] static extern bool ChangeFa(IntPtr svc, uint level, ref FailureActions info);
  [DllImport("advapi32.dll", SetLastError = true, EntryPoint = "ChangeServiceConfig2W")] static extern bool ChangeDword(IntPtr svc, uint level, ref uint info);
  const uint SC_MANAGER_CONNECT = 0x1, SERVICE_CHANGE_CONFIG = 0x2, SERVICE_START = 0x10;
  const uint FAILURE_ACTIONS = 2, FAILURE_ACTIONS_FLAG = 4, PRESHUTDOWN_INFO = 7;
  static void With(string name, Action<IntPtr> f) {
    IntPtr scm = OpenSCManagerW(null, null, SC_MANAGER_CONNECT);
    if (scm == IntPtr.Zero) throw new Win32Exception();
    try {
      // SERVICE_START too: a restart action needs it.
      IntPtr s = OpenServiceW(scm, name, SERVICE_CHANGE_CONFIG | SERVICE_START);
      if (s == IntPtr.Zero) throw new Win32Exception();
      try { f(s); } finally { CloseServiceHandle(s); }
    } finally { CloseServiceHandle(scm); }
  }
  // actions: SC_ACTION pairs, type then delay in ms (0 none, 1 restart).
  public static void SetFailureActions(string name, uint resetSeconds, uint[] actions) {
    With(name, s => {
      int n = actions.Length / 2;
      IntPtr a = Marshal.AllocHGlobal(8 * Math.Max(n, 1));
      try {
        for (int i = 0; i < n; ++i) {
          Marshal.WriteInt32(a, 8 * i, (int)actions[2 * i]);
          Marshal.WriteInt32(a, 8 * i + 4, (int)actions[2 * i + 1]);
        }
        FailureActions fa = new FailureActions();
        fa.dwResetPeriod = resetSeconds; fa.cActions = (uint)n; fa.lpsaActions = a;
        if (!ChangeFa(s, FAILURE_ACTIONS, ref fa)) throw new Win32Exception();
      } finally { Marshal.FreeHGlobal(a); }
    });
  }
  public static void SetFailureActionsOnNonCrashFailures(string name, bool on) {
    With(name, s => { uint v = on ? 1u : 0u; if (!ChangeDword(s, FAILURE_ACTIONS_FLAG, ref v)) throw new Win32Exception(); });
  }
  public static void SetPreshutdownTimeout(string name, uint ms) {
    With(name, s => { uint v = ms; if (!ChangeDword(s, PRESHUTDOWN_INFO, ref v)) throw new Win32Exception(); });
  }
}
'@
}

# ===================================================================================== -Uninstall
# STOPPED FIRST, THEN DELETED. sc.exe delete alone only marks a running service for deletion: it
# keeps the cards until it exits, and an -Install meanwhile fails with 1072. The stop is the
# engine's drain; a service that has not stopped in -WaitS is reported and left alone -- not
# deleted, and never killed. The key file goes with the service; the YAML and the logs stay.
function Wait-ServiceStopped([string]$name, [int]$timeoutS) {
  $t0 = Get-Date; $next = 10
  for (;;) {
    $s = Get-Service -Name $name -ErrorAction SilentlyContinue
    if (-not $s -or $s.Status -eq 'Stopped') { Write-Host ("     stopped after {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds); return }
    $el = ((Get-Date) - $t0).TotalSeconds
    if ($el -ge $timeoutS) {
      $svcPid = (Get-CimInstance Win32_Service -Filter "Name='$name'" -ErrorAction SilentlyContinue).ProcessId
      throw ("$name is still $($s.Status) after $timeoutS s. It was NOT deleted and NOT killed: a process with work on a GPU is never ended by force. " +
             "Its log says what the drain waits on ($svcDir); brl-stacks $svcPid shows where it waits. Run -Uninstall again once it has stopped.")
    }
    if ($el -ge $next) { Write-Host ("     {0}, {1:N0} s after the stop" -f $s.Status, $el); $next += 10 }
    Start-Sleep -Milliseconds 500
  }
}

if ($Uninstall) {
  if (-not $WhatIf) { Assert-Admin '-Uninstall' }
  $svc = Get-Service -Name $Name -ErrorAction SilentlyContinue
  if (-not $svc) {
    Write-Host "serve: there is no service $Name$(if ($WhatIf) { '; -Uninstall would do nothing' })"
    if (Test-Path -LiteralPath $keyFile) { Write-Host "  (its key file $keyFile is still there; remove it by hand if the service is gone for good)" }
    exit $(if ($WhatIf) { 0 } else { 1 })
  }
  if ($svc.Status -ne 'Stopped') {
    Add-Note "stop it: the engine drains (ends its requests, releases the cards), then exits"
    Add-Native $scExe "stop $Name" @(0, 1062)
    Add-Ps "Wait-ServiceStopped $(Quote-Ps $Name) $WaitS" { param($d) Wait-ServiceStopped $d.Name $d.Wait } @{ Name = $Name; Wait = $WaitS }
  }
  Add-Native $scExe "delete $Name"
  if (Test-Path -LiteralPath $keyFile) {
    Add-Ps "Remove-Item -LiteralPath $(Quote-Ps $keyFile) -Force" { param($d) Remove-Item -LiteralPath $d.Path -Force } @{ Path = $keyFile }
  }
  Add-Note "kept: $svcDir (the YAML and the logs)"
  if ($WhatIf) { Show-Steps '-Uninstall'; exit 0 }
  Write-Host "serve: removing the service $Name"
  Invoke-Steps
  Write-Host "serve: $Name removed; its YAML and logs are still in $svcDir"
  exit 0
}

# ===================================================================================== the profile
# The package (bin\ beside this script) or the source tree (build-win\hip\bin above win\).
if ($Exe) {
  if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) { throw "-Exe ${Exe}: no such file" }
  $exePath = (Resolve-Path -LiteralPath $Exe).Path
  $bin = Split-Path -Parent $exePath
} else {
  $bin = Join-Path $PSScriptRoot 'bin'
  if (-not (Test-Path (Join-Path $bin 'radiance.exe'))) { $bin = Join-Path $PSScriptRoot '..\build-win\hip\bin' }
  if (-not (Test-Path (Join-Path $bin 'radiance.exe'))) { throw "no radiance.exe beside this script (bin\) or in build-win\hip\bin" }
  $bin = (Resolve-Path $bin).Path
  $exePath = Join-Path $bin 'radiance.exe'
}

$yaml = $Profile
if (-not (Test-Path $yaml)) { $yaml = Join-Path $PSScriptRoot "profiles\$Profile.yaml" }
if (-not (Test-Path $yaml)) { throw "no profile '$Profile' (looked for a file, and for profiles\$Profile.yaml)" }
$yaml = (Resolve-Path $yaml).Path

# ---- the server block, as asked
$lines = [IO.File]::ReadAllLines($yaml)
$out = New-Object System.Collections.Generic.List[string]
$inServer = $false; $sawServer = $false
$replace = @{}
if ($Listen) { $replace['host'] = $Listen }
if ($Port -gt 0) { $replace['port'] = "$Port" }
$keyLines = @()
if ($ApiKey) {
  if ($ApiKey -eq 'off') { $keyLines = @() }
  elseif ($ApiKey -like 'env:*') { $keyLines = @("  api-key-env: $($ApiKey.Substring(4))") }
  else { throw "-ApiKey takes 'off' or 'env:NAME' (a key itself does not belong on a command line)" }
}
foreach ($l in $lines) {
  if ($l -match '^server:\s*$') { $inServer = $true; $sawServer = $true; $out.Add($l); continue }
  if ($inServer -and $l -match '^\S') {
    foreach ($k in $replace.Keys) { $out.Add("  ${k}: $($replace[$k])") }
    if ($ApiKey) { foreach ($x in $keyLines) { $out.Add($x) } }
    $inServer = $false
  }
  if ($inServer -and $l -match '^\s+(host|port|api-key|api-key-env)\s*:') {
    $k = $Matches[1]
    if ($replace.ContainsKey($k)) { continue }
    if ($ApiKey -and ($k -eq 'api-key' -or $k -eq 'api-key-env')) { continue }
  }
  $out.Add($l)
}
if (-not $sawServer -and ($replace.Count -or $keyLines.Count)) {
  $out.Insert(0, 'server:')
  $i = 1
  foreach ($k in $replace.Keys) { $out.Insert($i, "  ${k}: $($replace[$k])"); $i++ }
  foreach ($x in $keyLines) { $out.Insert($i, $x); $i++ }
}
$derivedText = (($out -join "`n") + "`n")
if (-not $Install -and -not $WhatIf) {
  $derived = Join-Path $env:TEMP ("brilliance-" + [IO.Path]::GetFileNameWithoutExtension($yaml) + "-$PID.yaml")
  [IO.File]::WriteAllText($derived, $derivedText)
}

# ---- the cards
function Get-HipBuses {
  $cs = @"
using System; using System.Runtime.InteropServices; using System.Text;
public static class BrlHip {
  [DllImport("amdhip64_7.dll")] public static extern int hipGetDeviceCount(out int n);
  [DllImport("amdhip64_7.dll")] public static extern int hipDeviceGetPCIBusId(StringBuilder id, int len, int dev);
}
"@
  if (-not ('BrlHip' -as [type])) { Add-Type -TypeDefinition $cs }
  $env:PATH = "$bin;$env:PATH"   # the HIP runtime radiance uses, not the driver's System32 copy
  $n = 0
  if ([BrlHip]::hipGetDeviceCount([ref]$n) -ne 0) { throw 'the HIP runtime reports no device' }
  $map = @{}
  for ($d = 0; $d -lt $n; $d++) {
    $sb = New-Object System.Text.StringBuilder 64
    if ([BrlHip]::hipDeviceGetPCIBusId($sb, 64, $d) -eq 0 -and $sb.ToString() -match '^[0-9a-fA-F]+:([0-9a-fA-F]+):') {
      $map[[Convert]::ToInt32($Matches[1], 16)] = $d
    }
  }
  return $map
}
$busList = @($Bus -split '[,\s]+' | Where-Object { $_ } | ForEach-Object { [int]$_ })
$devList = @($Device -split '[,\s]+' | Where-Object { $_ } | ForEach-Object { [int]$_ })
if ($busList.Count -gt 0) {
  # The question loads HIP into this PowerShell for the length of an enumeration (device count and
  # PCI ids; no context, memory or queue on any card), before radiance starts.
  $map = Get-HipBuses
  $idx = foreach ($b in $busList) {
    if (-not $map.ContainsKey($b)) { throw "no HIP device on PCI bus $b (HIP shows buses $(( $map.Keys | Sort-Object) -join ', '))" }
    $map[$b]
  }
  $env:HIP_VISIBLE_DEVICES = ($idx -join ',')
  Write-Host "serve: PCI bus $($busList -join ', ') -> HIP device $env:HIP_VISIBLE_DEVICES"
} elseif ($devList.Count -gt 0) {
  $env:HIP_VISIBLE_DEVICES = ($devList -join ',')
}

# ---- the environment: the profile's `# brilliance-env:` lines, then -Env (ENVIRONMENT above).
# Read and checked here; set for radiance only at the start below, or written into the service.
$envOrder = New-Object System.Collections.Generic.List[string]
$envSet = @{}; $envFrom = @{}   # case-insensitive keys, as Windows' variable names are
function Add-EnvSpec([string]$spec, [string]$from) {
  foreach ($item in ($spec -split ',')) {
    $item = $item.Trim()
    if (-not $item) { continue }
    if ($item -notmatch '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') { throw "${from}: '$item' is not NAME=VALUE" }
    $n = $Matches[1]; $v = $Matches[2].Trim()
    if ($n -notmatch '^(GPU|HIP|AMD|BRILLIANCE|RADIANCE)_[A-Za-z0-9_]+$') {
      throw "${from}: $n is not one serve.ps1 sets. It takes GPU_*, HIP_*, AMD_*, BRILLIANCE_* and RADIANCE_* only (the HIP runtime's and the engine's own); set anything else in the shell yourself"
    }
    if ($n -eq 'HIP_VISIBLE_DEVICES') { throw "${from}: HIP_VISIBLE_DEVICES: choose the cards with -Bus (PCI buses) or -Device (HIP indices)" }
    if ($n -match 'KEY|TOKEN|SECRET|PASSWORD') { throw "${from}: $n looks like a secret, and these values are printed and (with -Install) readable by local users. A key goes through -ApiKey env:NAME" }
    if (-not $v) { throw "${from}: $n has no value" }
    if ($v -match "[`r`n`0]") { throw "${from}: the value of $n is more than one line" }
    if (-not $envSet.ContainsKey($n)) { $envOrder.Add($n) }
    $envSet[$n] = $v; $envFrom[$n] = $from
  }
}
$profileLeaf = Split-Path -Leaf $yaml
foreach ($l in $lines) {
  if ($l -match '^\s*#\s*brilliance-env\s*:(.*)$') { Add-EnvSpec $Matches[1] "the profile $profileLeaf" }
}
foreach ($e in $Env) { Add-EnvSpec $e '-Env' }
foreach ($n in $envOrder) { Write-Host "serve: environment $n=$($envSet[$n]) (from $($envFrom[$n]))" }

# Set for radiance, and this shell's own values put back after it (a value like
# BRILLIANCE_MAILBOX_MEM would otherwise carry over to every later start from this console).
$envSaved = @{}
function Set-RunEnv {
  foreach ($n in $envOrder) {
    $envSaved[$n] = [Environment]::GetEnvironmentVariable($n, 'Process')
    [Environment]::SetEnvironmentVariable($n, $envSet[$n], 'Process')
  }
}
function Restore-RunEnv {
  foreach ($n in $envSaved.Keys) { [Environment]::SetEnvironmentVariable($n, $envSaved[$n], 'Process') }
}

# ===================================================================================== -Install
if ($Install) {
  if (-not $WhatIf) { Assert-Admin '-Install' }
  if (Get-Service -Name $Name -ErrorAction SilentlyContinue) {
    throw "a service $Name is installed already: remove it first (.\serve.ps1 $Profile -Uninstall$(if ($PSBoundParameters.ContainsKey('Name')) { " -Name $Name" }))"
  }
  # What the YAML says about the server, after the replacements above.
  $srv = @{}; $inS = $false
  foreach ($l in $out) {
    if ($l -match '^server:\s*$') { $inS = $true; continue }
    if ($inS -and $l -match '^\S') { $inS = $false }
    if ($inS -and $l -match '^\s+(host|port|api-key|api-key-env)\s*:\s*(.*?)\s*$') { $srv[$Matches[1]] = $Matches[2].Trim('"', "'") }
  }
  # The service's YAML is readable by every local user (so are its logs): a key does not go in it.
  if ($srv.ContainsKey('api-key')) {
    throw "the profile holds an API key itself (api-key:), and the service's copy of it would be readable by every local user. Use -ApiKey env:NAME (the key goes in a file only SYSTEM and Administrators can read) or -ApiKey off"
  }
  $keyVar = ''
  $keyValue = ''
  if ($ApiKey -like 'env:*') {
    $keyVar = $ApiKey.Substring(4)
    if ($keyVar -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') { throw "-ApiKey env:$keyVar is not an environment variable name" }
    $keyValue = [Environment]::GetEnvironmentVariable($keyVar)
    if (-not $keyValue) { throw "-ApiKey env:${keyVar}: $keyVar is not set in this shell, so there is no key to give the service. Set it (`$env:$keyVar = '...') and run again" }
    if ($keyValue -match "[`r`n]") { throw "-ApiKey env:${keyVar}: the key is more than one line" }
  } elseif ($srv.ContainsKey('api-key-env')) {
    Write-Host "serve: the profile names api-key-env: $($srv['api-key-env']). The service (LocalSystem) reads it from the SYSTEM environment, not this shell's; -ApiKey env:$($srv['api-key-env']) would store this shell's value in a key file only SYSTEM and Administrators can read instead."
  }
  if (-not $srv.ContainsKey('api-key-env') -and -not $keyVar -and $srv['host'] -and $srv['host'] -notin @('127.0.0.1', 'localhost', '::1')) {
    Write-Host "serve: the service will listen on $($srv['host']) with no API key: whatever reaches that port can use it"
  }

  # The service's own environment: the cards, and where a repo-id model is fetched to (a LocalSystem
  # service has none of this user's variables, and its own %LOCALAPPDATA% is on C:).
  $svcEnv = New-Object System.Collections.Generic.List[string]
  if ($env:HIP_VISIBLE_DEVICES) { $svcEnv.Add("HIP_VISIBLE_DEVICES=$env:HIP_VISIBLE_DEVICES") }
  if ($env:BRILLIANCE_HF_CACHE -and -not $envSet.ContainsKey('BRILLIANCE_HF_CACHE')) { $svcEnv.Add("BRILLIANCE_HF_CACHE=$env:BRILLIANCE_HF_CACHE") }
  # The profile's `# brilliance-env:` lines and -Env (ENVIRONMENT above).
  foreach ($n in $envOrder) { $svcEnv.Add("$n=$($envSet[$n])") }

  $svcYaml = Join-Path $svcDir "$Name.yaml"
  $svcLog = if ($Log) { [IO.Path]::GetFullPath($Log) } else { Join-Path $svcDir "$Name.log" }
  $cmd = "$(Quote-Arg $exePath) --service $Name --config $(Quote-Arg $svcYaml)"
  if ($Log) { $cmd += " --log $(Quote-Arg $svcLog)" }
  if ($keyVar) { $cmd += " --env-from-file $(Quote-Arg "$keyVar=$keyFile")" }
  if ($StopTimeoutS -ne 180) { $cmd += " --stop-timeout $StopTimeoutS" }
  if ($cmd -match '%') { Write-Host "serve: note: the service's command line has a %; the SCM expands %NAME% in it when it starts the service" }
  $pf = $env:ProgramFiles
  if (-not $exePath.StartsWith("$pf\", [StringComparison]::OrdinalIgnoreCase)) {
    Write-Host "serve: note: the service runs as LocalSystem from $bin; any account that can write to that folder can run code as SYSTEM. On a machine others use, install from a package unpacked under $pf."
  }

  Add-Note "the folders: $dataDir and $svcDir (SYSTEM and Administrators full control, local users read)"
  Add-ProtectDir $dataDir $true
  Add-ProtectDir $svcDir $true
  if ($keyVar) {
    Add-Note "the key folder: $keyDir (SYSTEM and Administrators only)"
    Add-ProtectDir $keyDir $false
  }
  Add-Note "the YAML the service runs: the profile $yaml, its server block as asked"
  Add-ReplaceFile $svcYaml $derivedText '<the YAML below>' $derivedText
  if ($keyVar) {
    Add-Note "the key: this shell's `$env:$keyVar, never printed; the service sets $keyVar from this file"
    Add-ReplaceFile $keyFile $keyValue "`$env:$keyVar <not shown>" ''
  }
  Add-Note "radiance reads the YAML the way the service will (no card is used)"
  Add-Native $exePath "--config $(Quote-Arg $svcYaml) --check"
  Add-Note "the service: LocalSystem, its own process, $(if ($Manual) { 'started on demand' } else { 'a delayed automatic start at boot' })"
  $startType = if ($Manual) { 'demand' } else { 'delayed-auto' }
  Add-Native $scExe "create $Name binPath= $(Quote-Arg $cmd) start= $startType DisplayName= $(Quote-Arg "Brilliance ($stem)")"
  Add-Native $scExe "description $Name $(Quote-Arg "Brilliance (radiance) local inference server: $stem. win\SERVICE.md")"
  if ($NoRestart) {
    Add-Note "recovery: none, ever (-NoRestart)"
    Add-Ps "[BrlSvc]::SetFailureActions('$Name', 86400, @(0, 0))" { param($d) Add-BrlSvcType; [BrlSvc]::SetFailureActions($d.Name, 86400, [uint32[]]@(0, 0)) } @{ Name = $Name }
  } else {
    Add-Note "recovery: the first failure in a day restarts it after 2 minutes, every later one does nothing (not into a card that is in trouble again)"
    Add-Ps "[BrlSvc]::SetFailureActions('$Name', 86400, @(1, 120000, 0, 0))" { param($d) Add-BrlSvcType; [BrlSvc]::SetFailureActions($d.Name, 86400, [uint32[]]@(1, 120000, 0, 0)) } @{ Name = $Name }
  }
  Add-Note "a non-zero exit nobody asked for is a failure too, not only a crash (a requested stop never is)"
  Add-Ps "[BrlSvc]::SetFailureActionsOnNonCrashFailures('$Name', `$true)" { param($d) Add-BrlSvcType; [BrlSvc]::SetFailureActionsOnNonCrashFailures($d.Name, $true) } @{ Name = $Name }
  $preMs = [Math]::Max(180000, $StopTimeoutS * 1000)
  Add-Note "at shutdown the SCM waits up to $($preMs / 1000) s (PRESHUTDOWN) for the drain"
  Add-Ps "[BrlSvc]::SetPreshutdownTimeout('$Name', $preMs)" { param($d) Add-BrlSvcType; [BrlSvc]::SetPreshutdownTimeout($d.Name, [uint32]$d.Ms) } @{ Name = $Name; Ms = $preMs }
  if ($svcEnv.Count) {
    $shownEnv = ($svcEnv | ForEach-Object { Quote-Ps $_ }) -join ', '
    Add-Note "the service's environment (not secret: readable by local users)"
    Add-Ps "Set-ItemProperty -LiteralPath $(Quote-Ps $regKey) -Name Environment -Type MultiString -Value @($shownEnv)" {
      param($d) Set-ItemProperty -LiteralPath $d.Key -Name Environment -Type MultiString -Value $d.Env
    } @{ Key = $regKey; Env = [string[]]$svcEnv.ToArray() }
  }
  Add-Native $scExe "qfailure $Name"

  if ($WhatIf) {
    Add-BrlSvcType   # compiled, not called: a real -Install will not stop at it
    Show-Steps '-Install'
    exit 0
  }
  Write-Host "serve: installing the service $Name"
  Invoke-Steps
  $h = if ($srv['host'] -and $srv['host'] -notin @('0.0.0.0', '::')) { $srv['host'] } else { '127.0.0.1' }
  $p = if ($srv['port']) { $srv['port'] } else { '8000' }
  Write-Host "serve: installed $Name. It is not started yet:"
  Write-Host "  start:   Start-Service $Name      (or sc.exe start $Name)"
  Write-Host "  ready:   Invoke-RestMethod http://${h}:$p/health   (a model takes minutes to load)"
  Write-Host "  log:     $svcLog"
  Write-Host "  stop:    Stop-Service $Name       (the drain; never a kill)"
  Write-Host "  remove:  .\serve.ps1 $Profile -Uninstall$(if ($PSBoundParameters.ContainsKey('Name')) { " -Name $Name" })"
  if ($srv['host'] -and $srv['host'] -notin @('127.0.0.1', 'localhost', '::1')) {
    Write-Host "  the LAN: allow the port for the private network only, if no rule does yet (elevated):"
    Write-Host "    New-NetFirewallRule -DisplayName '$Name' -Direction Inbound -Protocol TCP -LocalPort $p -Program '$exePath' -RemoteAddress LocalSubnet -Profile Private -Action Allow"
  }
  exit 0
}

$exe = $exePath
if ($Check) {
  & $exe --config $derived --check
  $rc = $LASTEXITCODE
  Remove-Item $derived -ErrorAction SilentlyContinue
  exit $rc
}
if ($Background) {
  if (-not $Log) { $Log = Join-Path (Get-Location) ("radiance-" + [IO.Path]::GetFileNameWithoutExtension($yaml) + ".log") }
  $tools = Join-Path $PSScriptRoot 'tools'
  if (-not (Test-Path (Join-Path $tools 'serve-bg.ps1'))) { $tools = $PSScriptRoot + '\tools' }
  # This shell's own executable, not 'powershell' looked up on PATH: a service account's or a
  # trimmed environment may not find it there.
  $self = (Get-Process -Id $PID).Path
  $bgScript = Join-Path $tools 'serve-bg.ps1'
}
if ($WhatIf) {
  # The derived YAML is not written: shown here as it would be.
  $derived = Join-Path $env:TEMP ("brilliance-" + [IO.Path]::GetFileNameWithoutExtension($yaml) + "-<pid>.yaml")
  Write-Host "serve: -WhatIf: nothing was started. A start would set, for radiance only:"
  if ($env:HIP_VISIBLE_DEVICES) { Write-Host "     HIP_VISIBLE_DEVICES=$env:HIP_VISIBLE_DEVICES" }
  else { Write-Host "     (no HIP_VISIBLE_DEVICES: every discrete card HIP shows, the integrated GPU left out)" }
  foreach ($n in $envOrder) { Write-Host "     $n=$($envSet[$n])" }
  Write-Host "  and run:"
  if ($Background) { Write-Host "     $(Quote-Arg $self) -NoProfile -ExecutionPolicy Bypass -File $(Quote-Arg $bgScript) -Bin $(Quote-Arg $bin) -Config $(Quote-Arg $derived) -Log $(Quote-Arg $Log)" }
  else { Write-Host "     $(Quote-Arg $exe) --config $(Quote-Arg $derived)" }
  Write-Host "  on the profile $yaml, its server block as asked:"
  $derivedText.TrimEnd("`n") -split "`n" | ForEach-Object { Write-Host "       | $_" }
  exit 0
}
if ($Background) {
  Set-RunEnv
  try { & $self -NoProfile -ExecutionPolicy Bypass -File $bgScript -Bin $bin -Config $derived -Log $Log; $rc = $LASTEXITCODE }
  finally { Restore-RunEnv }
  exit $rc
}
Write-Host "serve: $exe --config $derived"
Set-RunEnv
try { & $exe --config $derived; $rc = $LASTEXITCODE }
finally {
  Restore-RunEnv
  Remove-Item $derived -ErrorAction SilentlyContinue
}
exit $rc
