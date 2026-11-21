#Requires -Version 5.1
<#
.SYNOPSIS
  Gate windows that stop at the first failed gate, and exit codes read the one way that is right
  under Windows PowerShell 5.1. Dot-source it: . "$repo\win\tools\gate-lib.ps1"

.DESCRIPTION
  Two defects of the 11-19/11-20 gate scripts (artifacts\pm\sdma-on, sdma-a3), fixed once here:

  - The gate window started the next gate after a failure: on 11-20 fn4 started 13 s after q27's
    FAIL. Invoke-BrlGateWindow runs gates one at a time and stops at the first that fails, naming it,
    its reason and the gates it did not start. A gate that has not exited by its deadline stops the
    window too, and is left running: nothing here ends a process.

  - Exit codes. Under PS 5.1:
      * Start-Process -PassThru without -Wait gives an EMPTY ExitCode after the exit unless the
        process handle was taken while it ran (the 11-19 race wrapper's false stop, "smoke exit ;").
        Start-BrlHidden takes it; Wait-BrlExit reads the code from the process object.
      * A native command's stderr line, once redirected (2>&1), becomes an ErrorRecord: $? is then
        $false whatever the exit code, and under $ErrorActionPreference = 'Stop' the first stderr
        line throws (NativeCommandError). Invoke-BrlNative runs the command with 'Continue' and
        returns its code from $LASTEXITCODE, never from $?.

  Every process is started hidden (headless). No GPU is touched here; what a gate runs is the
  caller's.

  Tested by win\tests\gate_lib_test.ps1 (ctest gate_lib_test) with fixture gates.
#>

function Start-BrlHidden {
  <# Start-Process, hidden, -PassThru, with the handle taken at once so ExitCode is readable after
     the exit. -ArgumentList is passed as Start-Process takes it: an element with a space needs its
     own quotes. #>
  param(
    [Parameter(Mandatory)][string]$FilePath,
    [string[]]$ArgumentList = @(),
    [Parameter(Mandatory)][string]$Out,
    [Parameter(Mandatory)][string]$Err,
    [string]$WorkingDirectory = ''
  )
  $sp = @{ FilePath = $FilePath; WindowStyle = 'Hidden'; PassThru = $true
           RedirectStandardOutput = $Out; RedirectStandardError = $Err }
  if ($ArgumentList.Count) { $sp.ArgumentList = $ArgumentList }
  if ($WorkingDirectory) { $sp.WorkingDirectory = $WorkingDirectory }
  $p = Start-Process @sp
  $null = $p.Handle   # PS 5.1: without the handle held, ExitCode reads empty once the process is gone
  $p
}

function Wait-BrlExit {
  <# The exit code of a process from Start-BrlHidden, as an [int], read from the process object.
     $null when it has not exited within -Seconds (0: wait for ever); the process is left running. #>
  param([Parameter(Mandatory)][System.Diagnostics.Process]$Process, [int]$Seconds = 0)
  if ($Seconds -gt 0) {
    if (-not $Process.WaitForExit($Seconds * 1000)) { return $null }
  }
  $Process.WaitForExit()   # also waits for the redirected output to drain
  [int]$Process.ExitCode
}

function Invoke-BrlNative {
  <# Run a native command; its stdout and stderr (as text) go to -OutFile when given. Returns an
     object: ExitCode (from $LASTEXITCODE) and Output (the lines). A stderr line is not a failure,
     and $? is never consulted. #>
  param([Parameter(Mandatory)][string]$FilePath, [string[]]$ArgumentList = @(), [string]$OutFile = '')
  $eap = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    $lines = @(& $FilePath @ArgumentList 2>&1 | ForEach-Object { "$_" })
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $eap
  }
  if ($OutFile) { [IO.File]::WriteAllLines($OutFile, [string[]]$lines, (New-Object Text.UTF8Encoding($false))) }
  [pscustomobject]@{ ExitCode = [int]$code; Output = $lines }
}

function Invoke-BrlGateWindow {
  <# Run -Gates one at a time, hidden, and stop at the first that fails.

     Each gate is a hashtable:
       Tag          its name in the log
       FilePath     what to start; ArgumentList, WorkingDirectory as Start-BrlHidden takes them
       Out, Err     its redirected stdout and stderr
       Seconds      a deadline from its start (0 or absent: none). Past it the gate is HUNG: the
                    window stops and the process is left for the manager (Ctrl-Break by hand).
       Pass         optional scriptblock, param($gate, [int]$exitCode): $null or '' when the gate
                    passed, else why it failed. Absent: exit code 0 passes, anything else fails.
     -Before        optional scriptblock, param($gate): $null or '' to start the gate, else why not
                    (an idle check); the window stops there, the gate not started.
     -After         optional scriptblock, param($gate, $exitCode): lines to log once a gate exits
                    (its summary), before it is judged.
     -Say           scriptblock, param([string]$line): the window's log. Default: the host.

     Returns one object: Status ('pass', 'fail', 'hung' or 'stopped'), Failed (the tag), Reason,
     Ran (the tags started, in order), NotStarted, and ExitCodes (tag -> code). #>
  param(
    [Parameter(Mandatory)][object[]]$Gates,
    [scriptblock]$Before = $null,
    [scriptblock]$After = $null,
    [scriptblock]$Say = { param([string]$line) Write-Host $line }
  )
  $ran = New-Object System.Collections.Generic.List[string]
  $codes = [ordered]@{}
  $result = { param($status, $failed, $reason, $i)
    $rest = @()
    if ($null -ne $i) { $rest = @($Gates | Select-Object -Skip $i | ForEach-Object { $_.Tag }) }
    [pscustomobject]@{ Status = $status; Failed = $failed; Reason = $reason; Ran = @($ran)
                       NotStarted = $rest; ExitCodes = $codes } }
  for ($i = 0; $i -lt $Gates.Count; $i++) {
    $g = $Gates[$i]
    if ($Before) {
      $why = & $Before $g
      if ($why) {
        $null = & $Say "STOP before $($g.Tag): $why. Not started: $((@($Gates | Select-Object -Skip $i | ForEach-Object { $_.Tag })) -join ', ')"
        return (& $result 'stopped' $g.Tag "$why" $i)
      }
    }
    $null = & $Say "starting $($g.Tag)"
    $args_ = @(); if ($g.ArgumentList) { $args_ = @($g.ArgumentList) }
    $wd = ''; if ($g.WorkingDirectory) { $wd = $g.WorkingDirectory }
    $p = Start-BrlHidden -FilePath $g.FilePath -ArgumentList $args_ -Out $g.Out -Err $g.Err -WorkingDirectory $wd
    $ran.Add($g.Tag)
    $secs = 0; if ($g.Seconds) { $secs = [int]$g.Seconds }
    $code = Wait-BrlExit -Process $p -Seconds $secs
    if ($null -eq $code) {
      $why = "has not exited $secs s after its start (pid $($p.Id)); left running for the manager (Ctrl-Break by hand, never a force-end)"
      $null = & $Say "HUNG: $($g.Tag) $why. Not started: $((@($Gates | Select-Object -Skip ($i + 1) | ForEach-Object { $_.Tag })) -join ', ')"
      return (& $result 'hung' $g.Tag $why ($i + 1))
    }
    $codes[$g.Tag] = $code
    $null = & $Say "$($g.Tag) exit $code"
    if ($After) { foreach ($l in @(& $After $g $code)) { $null = & $Say "  $l" } }
    if ($g.Pass) { $why = & $g.Pass $g $code } else { $why = $(if ($code -ne 0) { "exit $code" } else { '' }) }
    if ($why) {
      $rest = @($Gates | Select-Object -Skip ($i + 1) | ForEach-Object { $_.Tag })
      $null = & $Say "STOP: gate $($g.Tag) failed: $why. Not started: $(if ($rest.Count) { $rest -join ', ' } else { 'none (it was the last)' })"
      return (& $result 'fail' $g.Tag "$why" ($i + 1))
    }
  }
  $null = & $Say 'gate window done: every gate passed'
  & $result 'pass' $null $null $null
}
