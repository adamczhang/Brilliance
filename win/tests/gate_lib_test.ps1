#Requires -Version 5.1
<#
  win\tools\gate-lib.ps1 with fixture gates: small PowerShell and cmd scripts in a temporary folder.
  No GPU, no server, nothing left running. Prints "ok"/"FAIL" per check; exit 0 when every check
  passed, 1 otherwise. ctest runs it as gate_lib_test (win\tests\CMakeLists.txt), under Windows
  PowerShell 5.1, which is where the exit-code traps live.
#>
param([string]$Lib = '')
$ErrorActionPreference = 'Continue'
if (-not $Lib) { $Lib = Join-Path (Split-Path -Parent $PSScriptRoot) 'tools\gate-lib.ps1' }
. $Lib

$script:fails = 0
function Check([string]$what, [bool]$cond, [string]$detail = '') {
  if ($cond) { Write-Output "  ok    $what" }
  else { $script:fails++; Write-Output "  FAIL  $what$(if ($detail) { ": $detail" })" }
}

$ps = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$cmd = "$env:SystemRoot\System32\cmd.exe"
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("brl-gate-lib-test-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $tmp | Out-Null
Write-Output "PowerShell $($PSVersionTable.PSVersion); fixtures in $tmp"

# Fixture gates. Each writes <name>.ran when it starts, so a gate that never started has no .ran.
function Fixture([string]$name, [string]$body) {
  $f = Join-Path $tmp "$name.ps1"
  $text = "'started' | Out-File -Encoding ascii '$tmp\$name.ran'`r`n$body`r`n"
  [IO.File]::WriteAllText($f, $text, (New-Object Text.UTF8Encoding($false)))
  $f
}
$fx = @{
  pass    = Fixture 'pass'    "Write-Output 'VERDICT pass: PASS'; exit 0"
  fail    = Fixture 'fail'    "[Console]::Error.WriteLine('a stderr line'); Write-Output 'VERDICT fail: FAIL'; exit 2"
  quiet   = Fixture 'quiet'   "Write-Output 'VERDICT quiet: FAIL (exit 0 all the same)'; exit 0"
  never   = Fixture 'never'   "exit 0"
  slow    = Fixture 'slow'    "Start-Sleep -Seconds 6; exit 0"
  stderr0 = Fixture 'stderr0' "[Console]::Error.WriteLine('only stderr, and a clean exit'); exit 0"
}
function Gate([string]$tag, [string]$file, [hashtable]$extra = @{}) {
  $g = @{ Tag = $tag; FilePath = $ps; ArgumentList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$file`"")
          Out = Join-Path $tmp "$tag.out.txt"; Err = Join-Path $tmp "$tag.err.txt" }
  foreach ($k in $extra.Keys) { $g[$k] = $extra[$k] }
  $g
}
function Ran([string]$name) { Test-Path (Join-Path $tmp "$name.ran") }
function Reset { Get-ChildItem $tmp -Filter '*.ran' | Remove-Item -Force }
$log = New-Object System.Collections.Generic.List[string]
$say = { param([string]$line) $log.Add($line) }
$verdict = { param($g, [int]$code)
  if ($code -ne 0) { return "exit $code" }
  if (-not (Select-String -Path $g.Out -Pattern '^VERDICT \S+: PASS' -Quiet)) { return "no 'VERDICT ...: PASS' line in $($g.Out)" }
  '' }

Write-Output '1. a failed gate stops the window, named'
Reset; $log.Clear()
$r = Invoke-BrlGateWindow -Gates @((Gate 'pass' $fx.pass), (Gate 'fail' $fx.fail), (Gate 'never' $fx.never)) -Say $say
$log | ForEach-Object { Write-Output "    | $_" }
Check 'status is fail' ($r.Status -eq 'fail') "status '$($r.Status)'"
Check 'the failed gate is named' ($r.Failed -eq 'fail') "failed '$($r.Failed)'"
Check 'its reason is its exit code' ($r.Reason -eq 'exit 2') "reason '$($r.Reason)'"
Check 'the gates before it ran' ((Ran 'pass') -and (Ran 'fail'))
Check 'the gate after it never started' (-not (Ran 'never'))
Check 'NotStarted lists it' ((@($r.NotStarted) -join ',') -eq 'never') "'$(@($r.NotStarted) -join ',')'"
Check 'the log says STOP with the gate and the rest' (@($log | Where-Object { $_ -eq 'STOP: gate fail failed: exit 2. Not started: never' }).Count -eq 1)
Check 'exit codes read as numbers' (($r.ExitCodes['pass'] -eq 0) -and ($r.ExitCodes['fail'] -eq 2))

Write-Output '2. a gate that exits 0 without its PASS verdict stops the window too'
Reset; $log.Clear()
$r = Invoke-BrlGateWindow -Gates @((Gate 'pass' $fx.pass @{ Pass = $verdict }), (Gate 'quiet' $fx.quiet @{ Pass = $verdict }), (Gate 'never' $fx.never)) -Say $say
$log | ForEach-Object { Write-Output "    | $_" }
Check 'status is fail on quiet' (($r.Status -eq 'fail') -and ($r.Failed -eq 'quiet')) "$($r.Status) $($r.Failed)"
Check 'the reason names the missing verdict' ($r.Reason -like "no 'VERDICT ...: PASS' line*") "'$($r.Reason)'"
Check 'never did not start' (-not (Ran 'never'))

Write-Output '3. a refused idle check stops before the gate starts'
Reset; $log.Clear()
$before = { param($g) if ($g.Tag -eq 'fail') { 'a radiance process is running (fixture)' } else { '' } }
$r = Invoke-BrlGateWindow -Gates @((Gate 'pass' $fx.pass), (Gate 'fail' $fx.fail), (Gate 'never' $fx.never)) -Before $before -Say $say
$log | ForEach-Object { Write-Output "    | $_" }
Check 'status is stopped at fail' (($r.Status -eq 'stopped') -and ($r.Failed -eq 'fail')) "$($r.Status) $($r.Failed)"
Check 'fail and never did not start' ((-not (Ran 'fail')) -and (-not (Ran 'never')))

Write-Output '4. a gate past its deadline stops the window and is left running'
Reset; $log.Clear()
$t0 = Get-Date
$r = Invoke-BrlGateWindow -Gates @((Gate 'slow' $fx.slow @{ Seconds = 2 }), (Gate 'never' $fx.never)) -Say $say
$log | ForEach-Object { Write-Output "    | $_" }
Check 'status is hung on slow' (($r.Status -eq 'hung') -and ($r.Failed -eq 'slow')) "$($r.Status) $($r.Failed)"
Check 'it returned at the deadline, not at the exit' (((Get-Date) - $t0).TotalSeconds -lt 5.5)
Check 'never did not start' (-not (Ran 'never'))
$m = [regex]::Match($r.Reason, 'pid (\d+)')
if ($m.Success) {
  $sp = Get-Process -Id ([int]$m.Groups[1].Value) -ErrorAction SilentlyContinue
  Check 'the hung gate is still running' ($null -ne $sp)
  if ($sp) { $sp.WaitForExit(15000) | Out-Null }   # the fixture ends by itself; nothing is ended here
}

Write-Output '5. every gate passing'
Reset; $log.Clear()
$r = Invoke-BrlGateWindow -Gates @((Gate 'pass' $fx.pass @{ Pass = $verdict }), (Gate 'stderr0' $fx.stderr0)) -Say $say
$log | ForEach-Object { Write-Output "    | $_" }
Check 'status is pass, both ran' (($r.Status -eq 'pass') -and ((@($r.Ran) -join ',') -eq 'pass,stderr0')) "$($r.Status) $(@($r.Ran) -join ',')"
Check 'a stderr line with exit 0 is a pass' ($r.ExitCodes['stderr0'] -eq 0)

Write-Output '6. exit codes from a hidden process (Start-BrlHidden, Wait-BrlExit)'
foreach ($want in 0, 3) {
  $p = Start-BrlHidden -FilePath $cmd -ArgumentList @('/c', "echo to-stderr 1>&2 & exit /b $want") -Out "$tmp\x$want.out" -Err "$tmp\x$want.err"
  while (-not $p.HasExited) { Start-Sleep -Milliseconds 50 }   # the 11-19 wrapper's wait
  $code = Wait-BrlExit -Process $p
  Check "cmd exit $want reads $want, not empty" (($null -ne $code) -and ($code -is [int]) -and ($code -eq $want)) "read '$code'"
}

# The 11-19 race wrapper's form, for the record: redirected output, no handle taken, a HasExited
# poll. PS 5.1 then reads ExitCode as empty (the "smoke exit ;" false stop); without the redirects
# it reads 0. Not judged here: it is the bug, and Start-BrlHidden is the fix checked above.
$q = Start-Process -FilePath $cmd -ArgumentList @('/c', 'ping -n 2 127.0.0.1 >nul & exit /b 0') -WindowStyle Hidden -PassThru `
       -RedirectStandardOutput "$tmp\old.out" -RedirectStandardError "$tmp\old.err"
while (-not $q.HasExited) { Start-Sleep -Milliseconds 200 }
Write-Output "    the 11-19 form, for the record: ExitCode '$($q.ExitCode)' (redirected, no handle taken)"

Write-Output '7. a native command writing to stderr and exiting 0 (Invoke-BrlNative)'
$ErrorActionPreference = 'Continue'
& $cmd /c "echo to-stderr 1>&2 & exit /b 0" 2>&1 | Out-Null
$dollarQ = $?
$raw = $LASTEXITCODE
Write-Output "    the 5.1 trap, for the record: `$? = $dollarQ, `$LASTEXITCODE = $raw after 2>&1"
$ErrorActionPreference = 'Stop'
$threw = $false
try { $n = Invoke-BrlNative -FilePath $cmd -ArgumentList @('/c', 'echo to-stderr 1>&2 & exit /b 0') -OutFile "$tmp\native0.txt" }
catch { $threw = $true; Write-Output "    threw: $($_.Exception.Message)" }
$ErrorActionPreference = 'Continue'
Check 'no throw under ErrorActionPreference Stop' (-not $threw)
Check 'exit code 0' ((-not $threw) -and ($n.ExitCode -eq 0)) "read '$($n.ExitCode)'"
Check 'its stderr line is kept as text' ((-not $threw) -and (@($n.Output | Where-Object { $_ -match '^to-stderr\s*$' }).Count -eq 1)) "output '$(@($n.Output) -join '|')'"
Check 'and written to the file' ((Get-Content "$tmp\native0.txt" -Raw) -match 'to-stderr')
$n5 = Invoke-BrlNative -FilePath $cmd -ArgumentList @('/c', 'echo to-stderr 1>&2 & exit /b 5')
Check 'exit code 5 reads 5' ($n5.ExitCode -eq 5) "read '$($n5.ExitCode)'"

Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
if ($script:fails) { Write-Output "gate_lib_test: $script:fails check(s) FAILED"; exit 1 }
Write-Output 'gate_lib_test: every check passed'
exit 0
