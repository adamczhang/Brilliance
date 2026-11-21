#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: start radiance on a named profile, choosing the cards by PCI bus and the listening
  address and key policy from the command line -- in this console, in one of its own, or as a
  Windows service.

.DESCRIPTION
    .\serve.ps1 qwen27-preview                          the profile's listener; every discrete card
    .\serve.ps1 qwen27-preview -Bus 3                   the card on PCI bus 3 (decimal, or 0x03)
    .\serve.ps1 qwen27-preview -Listen 0.0.0.0 -ApiKey off     the local network, no key
    .\serve.ps1 qwen27-preview -ApiKey env:RADIANCE_API_KEY    a key read from that variable
    .\serve.ps1 qwen27-preview -Check                   validate the YAML and exit
    .\serve.ps1 qwen27-preview -Env GPU_X=1,HIP_Y=2     also set these for the server (ENVIRONMENT below)
    .\serve.ps1 qwen27-preview -WhatIf                  print what a start would set and run; start nothing
    .\serve.ps1 qwen27-preview -Background -Log srv.log start it in a console of its own (Ctrl-Break
                                                        stops it: tools\stop-server.ps1 -ProcessId N)
    .\serve.ps1 minicpm5-2b-1card -Cpu                  on the CPU: no GPU touched, the host kernels
    .\serve.ps1 minicpm5-2b-1card -Fetch -Bus 3         fetch the profile's model if it is not whole, serve it
    .\serve.ps1 qwen27-preview -Model D:\m\q.rad -Bus 3 serve this file in place of the profile's model:
    .\serve.ps1 qwen27-preview -Bus 3 -Install          register it as a service (elevated)
    .\serve.ps1 qwen27-preview -Install -WhatIf         print what -Install would run; change nothing
    .\serve.ps1 qwen27-preview -Uninstall               stop it (the drain) and remove it (elevated)

  A thin selector over radiance's own YAML: radiance runs on the profile (profiles\<name>.yaml, or a
  path) itself, so a reload (tools\reload-server.ps1, or POST /admin/reload) rereads the file you
  edit. -Listen, -Port and -ApiKey reach radiance as BRILLIANCE_SERVER_HOST, BRILLIANCE_SERVER_PORT
  and BRILLIANCE_SERVER_API_KEY, which radiance's own YAML reader applies over the profile's server:
  block, at the start and at every reload; its startup report names them as the source. Everything
  else is the profile's. Nothing here is a second configuration language. -WhatIf shows the host,
  port and key radiance reads with them (radiance --check).

  CARDS BY PCI BUS: HIP numbers devices in its own order (on the test machine: the iGPU, then buses
  6, 3, 9, 32), so -Bus is translated to HIP indices by asking the HIP runtime beside radiance for
  each device's PCI bus (in a PowerShell of its own that exits before radiance starts), and -Device
  takes HIP indices as they are. -Bus takes the decimal bus Windows shows (Device Manager, the card's
  Properties > General > Location: "PCI bus 3"), 0x20, or a PCI address as radiance's log prints it
  (0000:20:00.0). Neither given: HIP_VISIBLE_DEVICES as the caller set it, else every discrete card
  HIP shows with the integrated GPU hidden from HIP (visible, it divides the pinned-memory limit and
  turns elastic memory off, though the engine never runs on it). A one-card machine needs no -Bus.
  Every start and -WhatIf prints the cards radiance will see, as its card 0, 1, ... (what the YAML's
  cards: and memory.card-headroom-mib count), each with its PCI bus, and what is hidden; -Background
  writes the same lines at the head of its log.

  Runs in this console by default, so Ctrl-C stops it the way the engine expects: it drains and
  exits. Nothing here kills a server.

  ON THE CPU (-Cpu): radiance serves with no GPU at all, at host speed, on the profile as it is
  with three settings over it: BRILLIANCE_DEVICE=host (the host backend, the HIP runtime never asked
  for a device, and HIP_VISIBLE_DEVICES=-1 besides), server.kernels libavx,libref (the CPU kernels;
  the GPU package's card kernels cannot run there) and every model's num-speculative-tokens 0 (the
  CPU kernels cannot run every drafter; plain decoding). Like -Listen, these reach radiance's own
  YAML reader as variables, at the start and at every reload, and its startup report names them.
  -Cpu takes neither -Bus nor -Device. It works the same from the GPU package and the CPU package.
  NEVER UNASKED: without -Cpu, a start that would leave radiance no card (HIP shows none, or the
  mask hides them all) is refused, so a card that has failed does not quietly become a CPU server.

  THE MODEL FILE (-Model, -Fetch): -Model PATH serves that file (or checkpoint folder) in place of
  the profile's model: line, which stays as it is (BRILLIANCE_MODEL_PATH, applied by radiance's
  reader like -Listen). -Fetch fetches the file the profile's marked comment names,
      # brilliance-fetch: <owner/repo> <revision> [<file>]
  with libexec\brilliance\hfget.ps1 into <store>\<owner>\<repo>: -ModelDir, else
  BRILLIANCE_HF_CACHE, else radiance's own cache (%LOCALAPPDATA%\radiance\hf). The fetcher prints
  the commit, the destination and every file's size in bytes before a byte moves; a copy already
  whole is kept (no transfer), a partial one is resumed, and each file is checked against its
  SHA-256 before it is kept. Then the verified file is served, as -Model would. With no network a
  copy a complete fetch left is served, and said so. -Fetch -WhatIf prints the fetcher's plan and
  transfers nothing. With several models in the profile, -ModelName says which one these replace.

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

  AS A SERVICE (win\SERVICE.md has the whole story and the qualification steps). -Install copies the
  profile as it is to %ProgramData%\Brilliance\<Name>\<Name>.yaml (the file to edit before a
  reload; the profile itself is not read again) and registers the service <Name>
  (default Brilliance-<profile>) to run, as LocalSystem,
      radiance.exe --service <Name> --config <that YAML> [--env-from-file VAR=<key file>]
  with a delayed automatic start (-Manual: on demand), and its log beside the YAML. -ApiKey env:NAME
  copies this shell's NAME into %ProgramData%\Brilliance\keys\<Name>.key, which only SYSTEM and
  Administrators can read; the YAML says `api-key-env: NAME`, and the service sets NAME from the
  file. The key is never on a command line or in the registry, and a profile holding a key itself
  (api-key:) is refused. The cards (as chosen above), BRILLIANCE_HF_CACHE, the BRILLIANCE_SERVER_*
  values and the profile's and -Env's variables (ENVIRONMENT above) go into the service's own
  environment.
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
  # Comma-separated PCI bus numbers (decimal, as Windows shows them) and HIP indices. Typed in a
  # PowerShell window, -Bus 3,9 arrives as an array; through powershell -File, as one string "3,9".
  # Both are taken, and joined below.
  [string[]]$Bus = @(),
  [string[]]$Device = @(),
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
  [int]$WaitS = 300,
  # ON THE CPU and THE MODEL FILE (above)
  [switch]$Cpu,
  [string]$Model = '',
  [string]$ModelName = '',
  [switch]$Fetch,
  [string]$ModelDir = ''
)
$Bus = $Bus -join ','
$Device = $Device -join ','
$ErrorActionPreference = 'Stop'

$service = $Install -or $Uninstall
if ($Install -and $Uninstall) { throw '-Install and -Uninstall: one at a time' }
if ($WhatIf -and $Check) { throw '-WhatIf and -Check: one at a time (-Check starts no server either)' }
if ($Uninstall -and $Env.Count) { throw '-Env goes with a start or -Install, not -Uninstall' }
if ($service -and ($Check -or $Background)) { throw '-Install and -Uninstall take neither -Check nor -Background' }
if (-not $service -and ($Name -or $Manual -or $NoRestart)) { throw '-Name, -Manual and -NoRestart go with -Install or -Uninstall' }
if ($StopTimeoutS -lt 1 -or $StopTimeoutS -gt 86400) { throw '-StopTimeoutS takes 1 to 86400 seconds' }
if ($Uninstall -and ($Cpu -or $Model -or $Fetch)) { throw '-Cpu, -Model and -Fetch go with a start or -Install, not -Uninstall' }
if ($Cpu -and ($Bus -or $Device)) { throw '-Cpu uses no GPU: it takes neither -Bus nor -Device' }
if ($Model -and $Fetch) { throw '-Model and -Fetch: one at a time (-Fetch serves the file it fetched)' }
if ($ModelName -and -not ($Model -or $Fetch)) { throw '-ModelName says which model -Model or -Fetch replaces; give one of them' }
if ($ModelDir -and -not $Fetch) { throw '-ModelDir is where -Fetch stores a model; give -Fetch' }

# The service's name: the profile's, unless one is given. Letters, digits, '.', '_' and '-' only, so
# it is the same word in sc.exe, the registry and the file names below.
# Only a .yaml or .yml extension comes off: a profile named with a dot (ltx-2.5-1card) keeps it.
$stem = [IO.Path]::GetFileName($Profile) -replace '\.ya?ml$', ''
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

$lines = [IO.File]::ReadAllLines($yaml)

# ---- the listener and the key, as asked: for radiance's own YAML reader (core\fleet\config.cpp),
# which applies them over the profile's server: block. The profile itself is what radiance reads,
# and rereads on a reload. Every one is named here, an unasked one as $null (unset for radiance), so
# a value left in this shell from elsewhere does not apply.
$serverEnv = [ordered]@{ BRILLIANCE_SERVER_HOST = $null; BRILLIANCE_SERVER_PORT = $null; BRILLIANCE_SERVER_API_KEY = $null
                         BRILLIANCE_DEVICE = $null; BRILLIANCE_SERVER_KERNELS = $null; BRILLIANCE_MODEL_SPECULATION = $null
                         BRILLIANCE_MODEL_PATH = $null; BRILLIANCE_MODEL_NAME = $null }
$serverFrom = @{ BRILLIANCE_SERVER_HOST = '-Listen'; BRILLIANCE_SERVER_PORT = '-Port'; BRILLIANCE_SERVER_API_KEY = '-ApiKey'
                 BRILLIANCE_DEVICE = '-Cpu'; BRILLIANCE_SERVER_KERNELS = '-Cpu'; BRILLIANCE_MODEL_SPECULATION = '-Cpu'
                 BRILLIANCE_MODEL_PATH = $(if ($Fetch) { '-Fetch' } else { '-Model' }); BRILLIANCE_MODEL_NAME = '-ModelName' }
if ($Listen) { $serverEnv['BRILLIANCE_SERVER_HOST'] = $Listen }
if ($Port -gt 0) { $serverEnv['BRILLIANCE_SERVER_PORT'] = "$Port" }
if ($ApiKey) {
  if ($ApiKey -ne 'off' -and $ApiKey -notmatch '^env:[A-Za-z_][A-Za-z0-9_]*$') {
    throw "-ApiKey takes 'off' or 'env:NAME' (a key itself does not belong on a command line)"
  }
  $serverEnv['BRILLIANCE_SERVER_API_KEY'] = $ApiKey
}
if ($Cpu) {
  $serverEnv['BRILLIANCE_DEVICE'] = 'host'
  $serverEnv['BRILLIANCE_SERVER_KERNELS'] = 'libavx,libref'
  $serverEnv['BRILLIANCE_MODEL_SPECULATION'] = 'off'
  # The CPU kernels, where radiance looks for them: RADIANCE_HOME, the package's share\radiance, or
  # the build tree's radiance_home. Not found at all (a stand-in radiance), radiance says so itself.
  $homes = @($env:RADIANCE_HOME, (Join-Path $bin '..\share\radiance'), (Join-Path $bin '..\radiance_home')) | Where-Object { $_ }
  $kdir = $homes | ForEach-Object { Join-Path $_ 'kernels' } | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
  if ($kdir) {
    $missing = @('libavx.dll', 'libref.dll' | Where-Object { -not (Test-Path -LiteralPath (Join-Path $kdir $_)) })
    if ($missing.Count) { throw "-Cpu serves with the CPU kernels, and $kdir has no $($missing -join ' or ')" }
  }
}
if ($ModelName) { $serverEnv['BRILLIANCE_MODEL_NAME'] = $ModelName }
if ($Model) {
  $m = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Model)
  if (-not (Test-Path -LiteralPath $m)) { throw "-Model ${Model}: no such file or folder" }
  $serverEnv['BRILLIANCE_MODEL_PATH'] = $m
}

# ---- the cards
# HIP's devices, asked of the HIP runtime beside radiance in a PowerShell of its own that exits
# straight after (device count, PCI ids and the integrated flag; no context, memory or queue on any
# card). -Whole clears HIP_VISIBLE_DEVICES there, so the indices are HIP's whole numbering (what -Bus
# and -Device translate to); without it the caller's mask stands, so cards it hides are not asked
# about at all (HIP_VISIBLE_DEVICES=-1: none), and each index is the card's place among those shown.
# BRILLIANCE_SERVE_TEST_DEVICES replaces the question in win\tests (the whole table,
# INDEX:BUS:INTEGRATED,...; 'none' for a machine with no HIP), so those tests touch no card.
function Get-HipDevices([switch]$Whole) {
  $fake = $env:BRILLIANCE_SERVE_TEST_DEVICES
  if ($fake) {
    if ($fake -eq 'none') { throw 'no HIP runtime (BRILLIANCE_SERVE_TEST_DEVICES=none)' }
    $all = @($fake -split ',' | ForEach-Object {
      $f = $_ -split ':'
      [pscustomobject]@{ Index = [int]$f[0]; Bus = [int]$f[1]; Integrated = ($f[2] -eq '1') }
    })
    $m = $env:HIP_VISIBLE_DEVICES
    if ($Whole -or -not $m) { return $all }
    # as HIP does: the listed indices in order, renumbered from 0, up to the first it cannot use
    $shown = @()
    foreach ($t in ($m -split ',')) {
      $i = 0
      if (-not [int]::TryParse($t.Trim(), [ref]$i)) { break }
      $d = @($all | Where-Object { $_.Index -eq $i })
      if (-not $d.Count) { break }
      $shown += [pscustomobject]@{ Index = $shown.Count; Bus = $d[0].Bus; Integrated = $d[0].Integrated }
    }
    return $shown
  }
  if (-not (Test-Path -LiteralPath (Join-Path $bin 'amdhip64_7.dll'))) { throw "no HIP runtime beside radiance ($bin)" }
  # hipDeviceAttributeIntegrated is 16 in HIP 7's hipDeviceAttribute_t (hip_runtime_api.h);
  # hipErrorNoDevice (100) is a mask that leaves nothing.
  $probe = @'
$ErrorActionPreference = 'Stop'
if ($env:BRL_SERVE_HIP_WHOLE) { Remove-Item Env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
$env:PATH = $env:BRL_SERVE_HIP_BIN + ';' + $env:PATH
Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices; using System.Text;
public static class BrlHip {
  [DllImport("amdhip64_7.dll")] public static extern int hipGetDeviceCount(out int n);
  [DllImport("amdhip64_7.dll")] public static extern int hipDeviceGetPCIBusId(StringBuilder id, int len, int dev);
  [DllImport("amdhip64_7.dll")] public static extern int hipDeviceGetAttribute(out int v, int attr, int dev);
}
"@
$n = 0
$rc = [BrlHip]::hipGetDeviceCount([ref]$n)
if ($rc -eq 100) { 'count 0'; exit 0 }
if ($rc -ne 0) { "error hipGetDeviceCount $rc"; exit 0 }
"count $n"
for ($d = 0; $d -lt $n; $d++) {
  $sb = New-Object System.Text.StringBuilder 64
  $v = 0
  if ([BrlHip]::hipDeviceGetAttribute([ref]$v, 16, $d) -ne 0) { $v = -1 }
  if ([BrlHip]::hipDeviceGetPCIBusId($sb, 64, $d) -eq 0) { "device $d $($sb.ToString()) $v" } else { "device $d ? $v" }
}
'@
  $self = (Get-Process -Id $PID).Path
  $saved = @{ BRL_SERVE_HIP_BIN = $env:BRL_SERVE_HIP_BIN; BRL_SERVE_HIP_WHOLE = $env:BRL_SERVE_HIP_WHOLE }
  $env:BRL_SERVE_HIP_BIN = $bin
  $env:BRL_SERVE_HIP_WHOLE = $(if ($Whole) { '1' } else { $null })
  try {
    $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($probe))
    # 'Continue': a stderr line from the probe is text to read, not a terminating error (5.1).
    $ErrorActionPreference = 'Continue'
    $o = & $self -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $enc 2>&1 | ForEach-Object { "$_" }
  } finally {
    $ErrorActionPreference = 'Stop'
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
  }
  $devs = @(); $count = -1
  foreach ($l in $o) {
    if ($l -match '^error (.*)$') { throw "the HIP runtime answered $($Matches[1])" }
    if ($l -match '^count (\d+)$') { $count = [int]$Matches[1] }
    if ($l -match '^device (\d+) [0-9a-fA-F]+:([0-9a-fA-F]+):\S* (-?\d+)$') {
      $devs += [pscustomobject]@{ Index = [int]$Matches[1]; Bus = [Convert]::ToInt32($Matches[2], 16); Integrated = ($Matches[3] -eq '1') }
    }
  }
  if ($count -lt 0 -or $devs.Count -ne $count) { throw "the HIP runtime's devices could not be read ($(($o | Select-Object -First 3) -join ' / '))" }
  return $devs
}
# A bus as Windows shows it (decimal, Device Manager's "PCI bus 3"), as hex with 0x, or a whole PCI
# address as radiance's log prints it (0000:20:00.0, hex).
function ConvertTo-Bus([string]$t) {
  if ($t -match '^\d+$') { return [int]$t }
  if ($t -match '^0x([0-9a-fA-F]+)$') { return [Convert]::ToInt32($Matches[1], 16) }
  if ($t -match '^(?:[0-9a-fA-F]{4}:)?([0-9a-fA-F]{2}):[0-9a-fA-F]{2}\.[0-7]$') { return [Convert]::ToInt32($Matches[1], 16) }
  throw "-Bus '$t': a PCI bus is a decimal number as Windows shows it (3), hex with 0x (0x20), or a PCI address as radiance logs it (0000:20:00.0)"
}
function Format-Bus($d) {
  $s = "PCI bus $($d.Bus) ($('0x{0:x2}' -f $d.Bus))"
  if ($d.Integrated) { $s += ', integrated' }
  return $s
}

$busList = @($Bus -split '[,\s]+' | Where-Object { $_ } | ForEach-Object { ConvertTo-Bus $_ })
$devList = @($Device -split '[,\s]+' | Where-Object { $_ } | ForEach-Object { [int]$_ })
$callerMask = [Environment]::GetEnvironmentVariable('HIP_VISIBLE_DEVICES', 'Process')
# -Bus and -Device need HIP's whole numbering; with neither, the caller's mask (if any) stands and
# only what it shows is asked about. -Check uses no card: the HIP runtime is asked only for -Bus.
$whole = ($busList.Count -gt 0) -or ($devList.Count -gt 0) -or -not $callerMask
$devs = $null; $devsWhy = 'not asked (-Check)'
# -Cpu: the HIP runtime is not asked anything, here or by radiance.
if (-not $Cpu -and (-not $Check -or $busList.Count)) {
  try { $devs = @(Get-HipDevices -Whole:$whole) } catch { $devsWhy = $_.Exception.Message }
}
$mask = $callerMask; $how = ''
if ($Cpu) {
  $mask = '-1'; $how = '-Cpu'
} elseif ($busList.Count -gt 0) {
  if ($null -eq $devs) { throw "-Bus needs the HIP runtime's PCI ids, and there are none: $devsWhy" }
  $idx = foreach ($b in $busList) {
    $d = @($devs | Where-Object { $_.Bus -eq $b })
    if (-not $d.Count) { throw "no HIP device on PCI bus $b (HIP shows $((($devs | ForEach-Object { Format-Bus $_ }) -join '; ')))" }
    if ($d[0].Integrated -and @($devs | Where-Object { -not $_.Integrated }).Count) {
      Write-Host "serve: note: PCI bus $b is the integrated GPU; beside a discrete card the engine does not run on it"
    }
    $d[0].Index
  }
  $mask = ($idx -join ','); $how = "-Bus $($busList -join ',')"
} elseif ($devList.Count -gt 0) {
  $mask = ($devList -join ','); $how = "-Device $($devList -join ',')"
} elseif ($callerMask) {
  $how = "HIP_VISIBLE_DEVICES as this shell set it"
} elseif ($null -ne $devs -and @($devs | Where-Object { $_.Integrated }).Count -and @($devs | Where-Object { -not $_.Integrated }).Count) {
  # The integrated GPU hidden from HIP, not only skipped by the engine: visible, it divides the
  # pinned-memory limit and turns elastic memory off (win\device\hipnative.cpp, warn_visible_igpu).
  $mask = (@($devs | Where-Object { -not $_.Integrated } | ForEach-Object { $_.Index }) -join ',')
  $how = "no -Bus: every discrete card, the integrated GPU hidden"
} else {
  $how = "no -Bus: every card HIP shows"
}
# What radiance will see: its cards 0, 1, ... in this order (the YAML's cards: and
# memory.card-headroom-mib count these).
$seen = $null
if ($null -ne $devs) {
  if (-not $whole) {
    $seen = $devs   # asked through the caller's mask: already what radiance will see
  } elseif ($mask) {
    $seen = @()
    foreach ($t in ($mask -split ',')) {
      $i = 0
      if (-not [int]::TryParse($t.Trim(), [ref]$i) -or $i -lt 0) { break }   # HIP stops at the first index it cannot use
      $d = @($devs | Where-Object { $_.Index -eq $i })
      if (-not $d.Count) { break }
      $seen += $d[0]
    }
  } else { $seen = $devs }
}
$cardLines = New-Object System.Collections.Generic.List[string]
$cardLines.Add("cards ($how)$(if ($mask) { ": HIP_VISIBLE_DEVICES=$mask" } else { ': HIP_VISIBLE_DEVICES not set' })")
if ($Cpu) {
  $cardLines.Add("  no GPU: radiance serves on the CPU with the host backend (BRILLIANCE_DEVICE=host; the HIP runtime is not asked for a device), the CPU kernels libavx,libref and no speculative decoding")
} elseif ($null -eq $seen) {
  $cardLines.Add("  the PCI buses are unknown here: $devsWhy")
} elseif (-not $seen.Count) {
  $cardLines.Add("  radiance sees no card")
} else {
  for ($c = 0; $c -lt $seen.Count; $c++) {
    $cardLines.Add("  card $c = $(Format-Bus $seen[$c])$(if ($whole) { ", HIP device $($seen[$c].Index)" })")
  }
  $hidden = @(if ($whole) { $devs | Where-Object { $seen -notcontains $_ } })
  if ($hidden.Count) { $cardLines.Add("  hidden from HIP: $(($hidden | ForEach-Object { Format-Bus $_ }) -join '; ')") }
  $igpuSeen = @($seen | Where-Object { $_.Integrated }).Count
  if ($igpuSeen -and $igpuSeen -lt $seen.Count) {
    $cardLines.Add("  the integrated GPU is visible: the engine does not run on it, but it divides the pinned-memory limit and turns elastic memory off. -Bus with the discrete cards' buses hides it")
  } elseif ($seen.Count -eq 1) {
    $cardLines.Add("  one visible card: elastic memory (VMM) can be on")
  } else {
    $cardLines.Add("  $($seen.Count) visible cards: elastic memory (VMM) is off by default")
  }
}
if (-not $Check) { foreach ($l in $cardLines) { Write-Host "serve: $l" } }
# NEVER CPU SPEED UNASKED: with no card left for radiance, a HIP build serves on the host backend.
# That is what -Cpu asks for; without it the start is refused, so a card that failed (or a mask that
# hides them all) is not quietly a CPU server.
if (-not $Cpu -and -not $Check -and $null -ne $seen -and -not $seen.Count) {
  throw ("radiance would see no card here ($how), and would serve on the CPU at a fraction of a card's speed. " +
         "That is never done unasked: to serve on the CPU, add -Cpu. Otherwise the cards HIP shows are above; " +
         "check the card in Device Manager, or name it with -Bus.")
}

# ---- a fleet's entries that need more cards than radiance will see
# radiance refuses to start a fleet with any entry whose tp exceeds its cards ("tp N needs N cards").
# A profile of several models (profiles\sandbox.yaml: every shipped profile, none loaded at start)
# is run without those entries: radiance reads a copy of the profile less them, beside the profile's
# name in %TEMP%, and each one dropped is named. A one-model profile is left to radiance's own
# refusal, and so is a file with groups: (a group may name what would be dropped) or per-model
# cards: lists. The count is the cards the engine runs on: the discrete ones, or the integrated GPU
# when it is the only one. Unknown cards (the HIP runtime not asked) drop nothing.
function Get-FleetEntries([string[]]$text) {
  $entries = @(); $inModels = $false; $cur = $null
  foreach ($l in $text) {
    if ($l -match '^([A-Za-z_][\w-]*)\s*:') {
      $inModels = ($Matches[1] -eq 'models'); $cur = $null
      if ($Matches[1] -eq 'groups') { return $null }
      continue
    }
    if (-not $inModels) { continue }
    if ($l -match '^  ([^\s#][^:]*?)\s*:') {
      $cur = [pscustomobject]@{ Name = $Matches[1]; Tp = 1; OwnCards = $false }
      $entries += $cur
    } elseif ($cur -and $l -match '^\s{3,}tp\s*:\s*(\d+)') {
      $cur.Tp = [int]$Matches[1]
    } elseif ($cur -and $l -match '^\s{3,}cards\s*:') {
      $cur.OwnCards = $true
    }
  }
  return $entries
}
# -Cpu: the host backend is one device, so a fleet's tp 2 and tp 4 entries go the same way.
if ($Cpu -or $null -ne $seen) {
  if ($Cpu) { $runOn = 1 }
  else {
    $runOn = @($seen | Where-Object { -not $_.Integrated }).Count
    if (-not $runOn) { $runOn = $seen.Count }
  }
  $entries = Get-FleetEntries $lines
  if ($entries -and $entries.Count -gt 1 -and -not @($entries | Where-Object { $_.OwnCards }).Count) {
    $drop = @($entries | Where-Object { $_.Tp -gt $runOn })
    if ($drop.Count -eq $entries.Count) {
      if ($Cpu) { throw "-Cpu serves on one device (the host backend), and every model in $yaml needs more (tp 2 or above)" }
      throw "every model in $yaml needs more than the $runOn card(s) radiance will see; name more with -Bus"
    }
    if ($drop.Count) {
      $names = @($drop | ForEach-Object { $_.Name })
      $out = New-Object System.Collections.Generic.List[string]
      $out.Add("# $yaml less the models that need more than $runOn card(s), written by serve.ps1: $($names -join ', ')")
      $inModels = $false; $skip = $false
      foreach ($l in $lines) {
        if ($l -match '^([A-Za-z_][\w-]*)\s*:') { $inModels = ($Matches[1] -eq 'models'); $skip = $false }
        elseif ($inModels -and $l -match '^  ([^\s#][^:]*?)\s*:') { $skip = ($names -contains $Matches[1]) }
        if (-not $skip) { $out.Add($l) }
      }
      $derived = Join-Path ([IO.Path]::GetTempPath()) ("brilliance-" + [IO.Path]::GetFileNameWithoutExtension($yaml) + "-$($runOn)card.yaml")
      [IO.File]::WriteAllText($derived, (($out -join "`n") + "`n"), (New-Object Text.UTF8Encoding($false)))
      foreach ($d in $drop) {
        $msg = if ($Cpu) { "dropped $($d.Name): it needs $($d.Tp) cards and -Cpu serves on one device" } else { "dropped $($d.Name): it needs $($d.Tp) cards and radiance will see $runOn" }
        if ($Check) { Write-Host "serve: $msg" } else { $cardLines.Add($msg); Write-Host "serve: $msg" }
      }
      Write-Host "serve: radiance reads $derived, the profile less those"
      $yaml = $derived
      $lines = $out.ToArray()
    }
  }
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
    if ($n -match '^BRILLIANCE_SERVER_') { throw "${from}: ${n}: the listener and key come from the profile's server: block, or -Listen, -Port and -ApiKey" }
    if ($n -match '^BRILLIANCE_(DEVICE|MODEL_PATH|MODEL_NAME|MODEL_SPECULATION)$') { throw "${from}: ${n}: -Cpu, -Model, -Fetch and -ModelName set it" }
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

# -Fetch. The profile's `# brilliance-fetch: <owner/repo> <revision> [<file>]` line names what; the
# store (-ModelDir, BRILLIANCE_HF_CACHE, radiance's own cache) where. hfget.ps1 runs in this console,
# so its plan and its progress are seen and Ctrl-C stops it; -WhatIf runs it with -Plan.
function Invoke-Fetch {
  $marks = @($lines | Where-Object { $_ -match '^\s*#\s*brilliance-fetch\s*:' })
  if (-not $marks.Count) { throw "-Fetch: the profile $profileLeaf has no '# brilliance-fetch: <owner/repo> <revision> [<file>]' line saying what to fetch. Fetch it yourself (libexec\brilliance\hfget.ps1) and pass -Model <the file>" }
  if ($marks.Count -gt 1) { throw "-Fetch: the profile $profileLeaf has $($marks.Count) brilliance-fetch lines; it takes one" }
  $f = @(($marks[0] -replace '^\s*#\s*brilliance-fetch\s*:', '').Trim() -split '\s+' | Where-Object { $_ })
  if ($f.Count -lt 2 -or $f.Count -gt 3 -or $f[0] -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$' -or
      ($f.Count -eq 3 -and ($f[2] -match '(^|[/\\])\.\.([/\\]|$)' -or $f[2] -match '^[/\\]|^[A-Za-z]:|[#?]'))) {
    throw "-Fetch: '$($marks[0].Trim())' is not '# brilliance-fetch: <owner/repo> <revision> [<file>]'"
  }
  $repo = $f[0]; $rev = $f[1]; $file = $(if ($f.Count -eq 3) { $f[2] } else { '' })
  $store = $ModelDir
  if (-not $store) { $store = $env:BRILLIANCE_HF_CACHE }
  if (-not $store -and $env:XDG_CACHE_HOME) { $store = Join-Path $env:XDG_CACHE_HOME 'radiance\hf' }
  if (-not $store -and $env:LOCALAPPDATA) { $store = Join-Path $env:LOCALAPPDATA 'radiance\hf' }
  if (-not $store) { throw '-Fetch: no store to fetch into: give -ModelDir' }
  $store = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($store)
  $dest = Join-Path $store ($repo -replace '/', '\')
  $hfget = @((Join-Path $PSScriptRoot 'libexec\brilliance\hfget.ps1'), (Join-Path $PSScriptRoot 'libexec\hfget.ps1')) |
    Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
  if (-not $hfget) { throw "-Fetch: no libexec\brilliance\hfget.ps1 beside this script" }
  $self = (Get-Process -Id $PID).Path
  $fargs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $hfget, $repo, $dest, $rev)
  if ($file) { $fargs += $file }
  if ($WhatIf) { $fargs += '-Plan' }
  Write-Host "serve: -Fetch: $repo at $rev$(if ($file) { ", the file $file" }) into $dest (the profile's brilliance-fetch line)"
  $ErrorActionPreference = 'Continue'
  & $self @fargs 2>&1 | ForEach-Object { Write-Host "  $_" }
  $rc = $LASTEXITCODE
  $ErrorActionPreference = 'Stop'
  # The manifest hfget writes only once every file arrived whole and matched its SHA-256; for one
  # file it is .hfget-manifest-<file>, for a whole repo .hfget-manifest.
  $manifest = Join-Path $dest $(if ($file) { '.hfget-manifest-' + ($file -replace '[\\/]', '_') } else { '.hfget-manifest' })
  $whole = Test-Path -LiteralPath $manifest
  if ($WhatIf) {
    if ($rc -ne 0) { throw "-Fetch -WhatIf: the fetcher's plan failed (exit $rc, above)" }
  } elseif ($rc -eq 3 -and $whole) {
    Write-Host "serve: -Fetch: the fetcher changed nothing (exit 3, above: the hub could not be asked, or the file is in use); serving the copy a complete fetch left, verified then ($manifest)"
  } elseif ($rc -eq 4) {
    throw "-Fetch: $repo holds several containers and the profile's brilliance-fetch line names none; add the file to that line (one of those above)"
  } elseif ($rc -ne 0) {
    throw "-Fetch: the fetch did not complete (hfget exit $rc, above). Run the same command again: it resumes"
  } elseif (-not $whole) {
    throw "-Fetch: the fetcher exited 0 but left no manifest ($manifest)"
  }
  $path = $dest
  if ($file) { $path = Join-Path $dest ($file -replace '/', '\') }
  elseif (-not $WhatIf) {
    # A whole repo: its lone container, as radiance serves a fetched repo id (core\hf_model.cpp).
    $rads = @(Get-ChildItem -LiteralPath $dest -Filter '*.rad' -File)
    if ($rads.Count -eq 1) { $path = $rads[0].FullName }
    elseif ($rads.Count -gt 1) { throw "-Fetch: $dest holds $($rads.Count) containers; name one in the profile's brilliance-fetch line" }
  }
  if (-not $WhatIf -and -not (Test-Path -LiteralPath $path)) { throw "-Fetch: $path is not there after the fetch" }
  $serverEnv['BRILLIANCE_MODEL_PATH'] = $path
}
# -Fetch: the profile's model, fetched (or found whole) and verified, then served as -Model would.
if ($Fetch) { Invoke-Fetch }
foreach ($n in $serverEnv.Keys) { if ($serverEnv[$n]) { Write-Host "serve: environment $n=$($serverEnv[$n]) (from $($serverFrom[$n]); over the profile)" } }

# Set for radiance, and this shell's own values put back after it (a value like
# BRILLIANCE_MAILBOX_MEM would otherwise carry over to every later start from this console). The
# cards and the server overrides too; an unset one ($null) is removed for radiance.
$envSaved = @{}
function Set-RunEnv([hashtable]$extra = @{}) {
  $all = [ordered]@{}
  foreach ($n in $envOrder) { $all[$n] = $envSet[$n] }
  foreach ($n in $serverEnv.Keys) { $all[$n] = $serverEnv[$n] }
  $all['HIP_VISIBLE_DEVICES'] = $(if ($mask) { $mask } else { $null })
  foreach ($n in $extra.Keys) { $all[$n] = $extra[$n] }
  foreach ($n in $all.Keys) {
    if (-not $envSaved.ContainsKey($n)) { $envSaved[$n] = [Environment]::GetEnvironmentVariable($n, 'Process') }
    [Environment]::SetEnvironmentVariable($n, $all[$n], 'Process')
  }
}

# The listener and key as radiance's own reader takes them from this profile with these variables
# (radiance --check, which reads the file and serves nothing; no card): host, port, key and where
# each came from. Ok is false, with radiance's words in Out, when it refuses the file.
function Get-ServerView([hashtable]$extra = @{}) {
  Set-RunEnv $extra
  try {
    $ErrorActionPreference = 'Continue'
    $o = & $exePath --config $yaml --check 2>&1 | ForEach-Object { "$_" }
    $rc = $LASTEXITCODE
  } finally { Restore-RunEnv; $ErrorActionPreference = 'Stop' }
  $v = [pscustomobject]@{ Ok = ($rc -eq 0); Out = ($o -join "`n"); Host = ''; HostFrom = ''; Port = ''; PortFrom = ''; Key = '' ; KeyEnv = '' }
  foreach ($l in $o) {
    if ($l -match 'server host (\S*) \((.*?)\), port (\d+) \((.*?)\), api key (.*)$') {
      $v.Host = $Matches[1]; $v.HostFrom = $Matches[2]; $v.Port = $Matches[3]; $v.PortFrom = $Matches[4]; $v.Key = $Matches[5]
      if ($v.Key -match '^env (\S+)') { $v.KeyEnv = $Matches[1] }
    }
    if ($l -match '--api-key-env: (\S+) is unset') { $v.KeyEnv = $Matches[1] }
  }
  return $v
}
function Restore-RunEnv {
  foreach ($n in @($envSaved.Keys)) { [Environment]::SetEnvironmentVariable($n, $envSaved[$n], 'Process') }
  $envSaved.Clear()
}

# ===================================================================================== -Install
if ($Install) {
  if (-not $WhatIf) { Assert-Admin '-Install' }
  if (Get-Service -Name $Name -ErrorAction SilentlyContinue) {
    throw "a service $Name is installed already: remove it first (.\serve.ps1 $Profile -Uninstall$(if ($PSBoundParameters.ContainsKey('Name')) { " -Name $Name" }))"
  }
  $keyVar = ''
  $keyValue = ''
  if ($ApiKey -like 'env:*') {
    $keyVar = $ApiKey.Substring(4)
    $keyValue = [Environment]::GetEnvironmentVariable($keyVar)
    if (-not $keyValue) { throw "-ApiKey env:${keyVar}: $keyVar is not set in this shell, so there is no key to give the service. Set it (`$env:$keyVar = '...') and run again" }
    if ($keyValue -match "[`r`n]") { throw "-ApiKey env:${keyVar}: the key is more than one line" }
  }
  # What radiance's reader takes from the profile: first its own key setting (no -ApiKey), then the
  # listener as the service will have it.
  $fileView = Get-ServerView @{ BRILLIANCE_SERVER_API_KEY = $null }
  if (-not $fileView.Ok -and -not $fileView.KeyEnv) { throw "radiance refuses the profile ${yaml}:`n$($fileView.Out)" }
  # The service runs a copy of the profile as it is, readable by every local user (so are its logs):
  # a key does not go in it, whatever -ApiKey says.
  if ($fileView.Ok -and $fileView.Key -like 'in the file*') {
    throw "the profile holds an API key itself (api-key:), and the service's copy of it would be readable by every local user. Remove it from the profile and use -ApiKey env:NAME (the key goes in a file only SYSTEM and Administrators can read), or -ApiKey off"
  }
  $view = Get-ServerView @{ BRILLIANCE_SERVER_API_KEY = $(if ($ApiKey) { $ApiKey } else { 'off' }) }
  if (-not $view.Ok) { throw "radiance refuses the profile ${yaml} with these settings:`n$($view.Out)" }
  $svcKeyEnv = $(if ($keyVar) { $keyVar } elseif ($ApiKey -eq 'off') { '' } else { $fileView.KeyEnv })
  if (-not $ApiKey -and $fileView.KeyEnv) {
    Write-Host "serve: the profile names api-key-env: $($fileView.KeyEnv). The service (LocalSystem) reads it from the SYSTEM environment, not this shell's; -ApiKey env:$($fileView.KeyEnv) would store this shell's value in a key file only SYSTEM and Administrators can read instead."
  }
  Write-Host "serve: the service will listen on $($view.Host) port $($view.Port) (host from $($view.HostFrom), port from $($view.PortFrom)), $(if ($svcKeyEnv) { "API key from $svcKeyEnv" } else { 'no API key' })"
  if (-not $svcKeyEnv -and $view.Host -notin @('127.0.0.1', 'localhost', '::1')) {
    Write-Host "serve: the service will listen on $($view.Host) with no API key: whatever reaches that port can use it"
  }
  $srv = @{ host = $view.Host; port = $view.Port }

  # The service's own environment: the cards, and where a repo-id model is fetched to (a LocalSystem
  # service has none of this user's variables, and its own %LOCALAPPDATA% is on C:).
  $svcEnv = New-Object System.Collections.Generic.List[string]
  if ($mask) { $svcEnv.Add("HIP_VISIBLE_DEVICES=$mask") }
  if ($env:BRILLIANCE_HF_CACHE -and -not $envSet.ContainsKey('BRILLIANCE_HF_CACHE')) { $svcEnv.Add("BRILLIANCE_HF_CACHE=$env:BRILLIANCE_HF_CACHE") }
  # The profile's `# brilliance-env:` lines and -Env (ENVIRONMENT above).
  foreach ($n in $envOrder) { $svcEnv.Add("$n=$($envSet[$n])") }
  # -Listen, -Port and -ApiKey, which radiance's reader applies over the YAML's server: block.
  foreach ($n in $serverEnv.Keys) { if ($serverEnv[$n]) { $svcEnv.Add("$n=$($serverEnv[$n])") } }

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
  $profileText = [IO.File]::ReadAllText($yaml, [Text.Encoding]::UTF8)
  Add-Note "the YAML the service runs: a copy of the profile $yaml as it is. Edit this copy and reload (win\SERVICE.md) to change a model"
  Add-ReplaceFile $svcYaml $profileText '<the profile below>' $profileText
  if ($keyVar) {
    Add-Note "the key: this shell's `$env:$keyVar, never printed; the service sets $keyVar from this file"
    Add-ReplaceFile $keyFile $keyValue "`$env:$keyVar <not shown>" ''
  }
  Add-Note "radiance reads the YAML the way the service will, with its BRILLIANCE_SERVER_* values (no card is used)"
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
  # The check step reads the copy with the overrides the service will have.
  Set-RunEnv
  try { Invoke-Steps } finally { Restore-RunEnv }
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
  # radiance's stderr as plain lines: under Windows PowerShell 5.1 a caller's 2>&1 (serve.ps1 ...
  # -Check 2>&1) turns each stderr line of a native command into an ErrorRecord, which this script's
  # 'Stop' made a terminating NativeCommandError at radiance's first line. The exit code is
  # $LASTEXITCODE's, never $?'s.
  Set-RunEnv
  try {
    $ErrorActionPreference = 'Continue'
    & $exe --config $yaml --check 2>&1 | ForEach-Object { "$_" }
    $rc = $LASTEXITCODE
  } finally { Restore-RunEnv; $ErrorActionPreference = 'Stop' }
  exit $rc
}
# radiance reads the profile itself, and rereads it on a reload.
$reloadHint = "radiance reads $yaml itself: to change a model while it runs, edit that file and run tools\reload-server.ps1 -ProcessId <radiance's pid> (or POST /admin/reload). server: and cards: changes need a restart"
if ($Background) {
  if (-not $Log) { $Log = "radiance-" + [IO.Path]::GetFileNameWithoutExtension($yaml) + ".log" }
  # A full path against this shell's location, so the helper (another PowerShell) and the message
  # below name the same file.
  $Log = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Log)
  $tools = Join-Path $PSScriptRoot 'tools'
  if (-not (Test-Path (Join-Path $tools 'serve-bg.ps1'))) { $tools = $PSScriptRoot + '\tools' }
  # This shell's own executable, not 'powershell' looked up on PATH: a service account's or a
  # trimmed environment may not find it there.
  $self = (Get-Process -Id $PID).Path
  $bgScript = Join-Path $tools 'serve-bg.ps1'
}
if ($WhatIf) {
  Write-Host "serve: -WhatIf: nothing was started. A start would set, for radiance only:"
  if ($mask) { Write-Host "     HIP_VISIBLE_DEVICES=$mask" }
  else { Write-Host "     (no HIP_VISIBLE_DEVICES: radiance sees every card HIP shows)" }
  foreach ($n in $envOrder) { Write-Host "     $n=$($envSet[$n])" }
  foreach ($n in $serverEnv.Keys) { if ($serverEnv[$n]) { Write-Host "     $n=$($serverEnv[$n])" } }
  Write-Host "  and run:"
  if ($Background) { Write-Host "     $(Quote-Arg $self) -NoProfile -ExecutionPolicy Bypass -File $(Quote-Arg $bgScript) -Bin $(Quote-Arg $bin) -Config $(Quote-Arg $yaml) -Log $(Quote-Arg $Log) -Append" }
  else { Write-Host "     $(Quote-Arg $exe) --config $(Quote-Arg $yaml)" }
  # What radiance's own reader makes of the profile with those values (radiance --check).
  $view = Get-ServerView
  if (-not $view.Ok) {
    Write-Host "  radiance refuses the profile as it would start:"
    $view.Out -split "`n" | ForEach-Object { Write-Host "       | $_" }
    exit 1
  }
  Write-Host "  radiance reads the profile as: host $($view.Host) (from $($view.HostFrom)), port $($view.Port) (from $($view.PortFrom)), API key $($view.Key)"
  foreach ($l in ($view.Out -split "`n")) {
    if ($l -match ': (device host: .*|kernels .*|model \S+: model from .*)$') { Write-Host "  radiance reads: $($Matches[1].Trim())" }
  }
  Write-Host "  $reloadHint"
  exit 0
}
Write-Host "serve: $reloadHint"
if ($Background) {
  # The launcher's own lines head the log, so it says which cards and which overrides the server
  # started with; the helper appends radiance's output.
  $head = New-Object System.Collections.Generic.List[string]
  $head.Add("serve: $(Get-Date -Format 'yyyy-MM-dd h:mm:ss tt') $exe --config $yaml")
  foreach ($l in $cardLines) { $head.Add("serve: $l") }
  foreach ($n in $envOrder) { $head.Add("serve: environment $n=$($envSet[$n])") }
  foreach ($n in $serverEnv.Keys) { if ($serverEnv[$n]) { $head.Add("serve: environment $n=$($serverEnv[$n]) (from $($serverFrom[$n]))") } }
  $head.Add("serve: $reloadHint")
  New-Item -ItemType Directory -Force ([IO.Path]::GetDirectoryName($Log)) | Out-Null
  [IO.File]::WriteAllText($Log, (($head -join "`r`n") + "`r`n"))
  Set-RunEnv
  try {
    $ErrorActionPreference = 'Continue'
    & $self -NoProfile -ExecutionPolicy Bypass -File $bgScript -Bin $bin -Config $yaml -Log $Log -Append
    $rc = $LASTEXITCODE
  } finally { Restore-RunEnv; $ErrorActionPreference = 'Stop' }
  exit $rc
}
Write-Host "serve: $exe --config $yaml"
# 'Continue' for the server's run: with a caller's 2>&1 (serve.ps1 ... 2>&1 | Tee-Object) its stderr
# lines arrive as ErrorRecords, and 'Stop' would end this script at the first one while radiance
# runs on. Without a redirect radiance writes to the console directly, as before.
Set-RunEnv
try {
  $ErrorActionPreference = 'Continue'
  & $exe --config $yaml
  $rc = $LASTEXITCODE
} finally { Restore-RunEnv; $ErrorActionPreference = 'Stop' }
exit $rc
