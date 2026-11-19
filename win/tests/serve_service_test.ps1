#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: win\serve.ps1 -Install / -Uninstall without an elevated shell -- what -WhatIf prints,
  and what the service options refuse (win\SERVICE.md). Nothing is installed, no card is used.
.DESCRIPTION
  ctest runs it (win\tests\CMakeLists.txt) with the radiance.exe of the build tree:
    powershell -File win\tests\serve_service_test.ps1 -Exe build-win\host\bin\radiance.exe
  Exit 0 when every check holds, 1 otherwise (each failure printed).
#>
param([Parameter(Mandatory)][string]$Exe)
$ErrorActionPreference = 'Stop'
$serve = Join-Path $PSScriptRoot '..\serve.ps1'
$Exe = (Resolve-Path -LiteralPath $Exe).Path
$script:failed = 0
$script:checks = 0
function Check([bool]$ok, [string]$what) {
  $script:checks++
  if (-not $ok) { $script:failed++; Write-Host "  FAIL $what" }
}
# serve.ps1 in a PowerShell of its own, as a person runs it; its output and exit code.
function Serve([string[]]$a, [hashtable]$envSet = @{}) {
  $saved = @{}
  foreach ($k in $envSet.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $envSet[$k]) }
  try {
    $ErrorActionPreference = 'Continue'
    $o = & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $serve @a 2>&1 | ForEach-Object { "$_" }
    return [pscustomobject]@{ Out = ($o -join "`n"); Rc = $LASTEXITCODE }
  } finally {
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
  }
}
function Has([string]$text, [string]$part) { return $text.Contains($part) }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("brl-serve-service-test-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory $tmp | Out-Null
try {
  $profile = Join-Path $tmp 'svc test.yaml'   # a space in the path, on purpose
  [IO.File]::WriteAllText($profile, "server:`n  host: 127.0.0.1`n  port: 18364`nmodels:`n  a:`n    model: C:/nonexistent/a.rad`n")
  $name = 'Brilliance-svc-test'

  # ---- -Install -WhatIf with a key, a card, the LAN and a named service
  $key = 'sk-serve-service-test-7d3c'
  $r = Serve @($profile, '-Install', '-WhatIf', '-Exe', $Exe, '-Name', 'Brl-SvcTest', '-Device', '2',
               '-Listen', '0.0.0.0', '-Port', '18365', '-ApiKey', 'env:BRL_SVC_TEST_KEY') @{ BRL_SVC_TEST_KEY = $key; HIP_VISIBLE_DEVICES = $null; BRILLIANCE_HF_CACHE = 'X:\test-hf' }
  $o = $r.Out
  Check ($r.Rc -eq 0) "-Install -WhatIf exits 0 (got $($r.Rc)):`n$o"
  Check (Has $o 'Nothing was changed') 'says nothing was changed'
  Check (-not (Has $o $key)) 'never prints the key'
  $data = Join-Path $env:ProgramData 'Brilliance'
  $yaml = Join-Path $data 'Brl-SvcTest\Brl-SvcTest.yaml'
  $keyFile = Join-Path $data 'keys\Brl-SvcTest.key'
  $sc = Join-Path $env:SystemRoot 'System32\sc.exe'
  $icacls = Join-Path $env:SystemRoot 'System32\icacls.exe'
  # the service's command line, exactly: the exe quoted (a space in the build path is likely), the
  # key by file, delayed automatic start
  $q = if ($Exe -match '\s') { "\`"$Exe\`"" } else { $Exe }
  $bin = "$q --service Brl-SvcTest --config $yaml --env-from-file BRL_SVC_TEST_KEY=$keyFile"
  Check (Has $o "$sc create Brl-SvcTest binPath= `"$bin`" start= delayed-auto DisplayName= `"Brilliance (svc test)`"") "sc.exe create line:`n$o"
  Check (Has $o "$sc description Brl-SvcTest ") 'a description'
  Check (Has $o "[BrlSvc]::SetFailureActions('Brl-SvcTest', 86400, @(1, 120000, 0, 0))") 'restart once after 2 minutes, then nothing'
  Check (Has $o "[BrlSvc]::SetFailureActionsOnNonCrashFailures('Brl-SvcTest', `$true)") 'non-crash failures count'
  Check (Has $o "[BrlSvc]::SetPreshutdownTimeout('Brl-SvcTest', 180000)") 'a 180 s preshutdown wait'
  Check (Has $o "-Name Environment -Type MultiString -Value @('HIP_VISIBLE_DEVICES=2', 'BRILLIANCE_HF_CACHE=X:\test-hf')") 'the card and the model store in the service environment'
  Check (Has $o "$icacls $(Split-Path $keyFile) /inheritance:r /grant:r *S-1-5-18:(OI)(CI)F *S-1-5-32-544:(OI)(CI)F`n") 'the key folder: SYSTEM and Administrators only'
  Check (Has $o "$icacls $data /inheritance:r /grant:r *S-1-5-18:(OI)(CI)F *S-1-5-32-544:(OI)(CI)F *S-1-5-32-545:(OI)(CI)RX") 'the data folder: users read'
  Check (Has $o "[IO.File]::WriteAllText('$keyFile', `$env:BRL_SVC_TEST_KEY <not shown>)") 'the key written to its file, not shown'
  foreach ($l in '| server:', '|   host: 0.0.0.0', '|   port: 18365', '|   api-key-env: BRL_SVC_TEST_KEY', '|     model: C:/nonexistent/a.rad') {
    Check (Has $o $l) "the derived YAML has '$l':`n$o"
  }
  Check (-not (Has $o '127.0.0.1')) 'the profile''s host is replaced'
  Check (Has $o "--config $yaml --check") 'the YAML checked by radiance'
  # the steps come in the order they must run
  $order = @("WriteAllText('$yaml'", "WriteAllText('$keyFile'", '--check', "$sc create", 'SetFailureActions(', 'Set-ItemProperty', "$sc qfailure")
  $at = -1; $inOrder = $true
  foreach ($x in $order) { $i = $o.IndexOf($x); if ($i -le $at) { $inOrder = $false }; $at = $i }
  Check $inOrder 'the steps are in order'

  # ---- defaults: the profile's name, no key, manual start, no restart, a longer stop timeout
  $r = Serve @($profile, '-Install', '-WhatIf', '-Exe', $Exe, '-Manual', '-NoRestart', '-StopTimeoutS', '600') @{ HIP_VISIBLE_DEVICES = $null; BRILLIANCE_HF_CACHE = $null }
  $o = $r.Out
  Check ($r.Rc -eq 0) "defaults: exit 0 (got $($r.Rc)):`n$o"
  # the profile's name made a service name: 'svc test' -> Brilliance-svc-test; the display name keeps it
  Check (Has $o "$sc create $name binPath= ") "the default name ${name}:`n$o"
  Check (Has $o "--service $name --config $(Join-Path $data "$name\$name.yaml")") 'the service, its YAML in its own folder'
  Check (Has $o 'DisplayName= "Brilliance (svc test)"') 'the display name keeps the profile''s'
  Check (Has $o "start= demand") '-Manual: on demand'
  Check (Has $o "--stop-timeout 600") '-StopTimeoutS reaches the command line'
  Check (Has $o "SetPreshutdownTimeout('$name', 600000)") 'the preshutdown wait covers the stop timeout'
  Check (Has $o "@(0, 0))") '-NoRestart: no action'
  Check (-not (Has $o 'keys')) 'no key, no key folder'
  Check (-not (Has $o 'MultiString')) 'no environment to set'

  # ---- refusals
  $r = Serve @($profile, '-Install', '-WhatIf', '-Exe', $Exe, '-ApiKey', 'env:BRL_SVC_TEST_UNSET') @{ BRL_SVC_TEST_UNSET = $null }
  Check ($r.Rc -ne 0 -and (Has $r.Out 'is not set in this shell')) "an unset key variable is refused:`n$($r.Out)"
  $r = Serve @($profile, '-Install', '-WhatIf', '-Exe', $Exe, '-ApiKey', 'sk-literal')
  Check ($r.Rc -ne 0 -and (Has $r.Out 'a key itself does not belong on a command line')) 'a literal key is refused'
  [IO.File]::WriteAllText((Join-Path $tmp 'withkey.yaml'), "server:`n  api-key: sk-in-the-file`nmodels:`n  a:`n    model: C:/x.rad`n")
  $r = Serve @((Join-Path $tmp 'withkey.yaml'), '-Install', '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out 'holds an API key itself')) "a profile with a key in it is refused:`n$($r.Out)"
  # ---- -WhatIf without -Install: what a start would set and run; nothing started, no file written
  $r = Serve @($profile, '-WhatIf', '-Exe', $Exe, '-Device', '2') @{ HIP_VISIBLE_DEVICES = $null }
  Check ($r.Rc -eq 0 -and (Has $r.Out 'nothing was started') -and (Has $r.Out 'HIP_VISIBLE_DEVICES=2') -and
         (Has $r.Out "--config ")) "-WhatIf alone prints the start:`n$($r.Out)"
  $r = Serve @($profile, '-WhatIf', '-Check', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out 'one at a time')) '-WhatIf with -Check is refused'

  # ---- the environment: a profile's `# brilliance-env:` lines and -Env (ENVIRONMENT in serve.ps1)
  $envProfile = Join-Path $tmp 'env test.yaml'
  [IO.File]::WriteAllText($envProfile, "# brilliance-env: GPU_FLUSH_ON_EXECUTION=1`n# brilliance-env: HIP_BRL_TEST_A=a, AMD_BRL_TEST_B=b=c`nserver:`n  host: 127.0.0.1`n  port: 18366`nmodels:`n  a:`n    model: C:/nonexistent/a.rad`n")
  # one comma-joined string, as powershell -File hands an array over
  $r = Serve @($envProfile, '-WhatIf', '-Exe', $Exe, '-Env', 'GPU_FLUSH_ON_EXECUTION=0,RADIANCE_BRL_TEST=r') @{ HIP_VISIBLE_DEVICES = $null }
  $o = $r.Out
  Check ($r.Rc -eq 0) "-Env with -WhatIf exits 0:`n$o"
  Check (Has $o 'serve: environment GPU_FLUSH_ON_EXECUTION=0 (from -Env)') "-Env overrides the profile's line:`n$o"
  Check (Has $o 'serve: environment HIP_BRL_TEST_A=a (from the profile env test.yaml)') "a profile line with two variables:`n$o"
  Check (Has $o 'serve: environment AMD_BRL_TEST_B=b=c (from the profile env test.yaml)') "a value may hold '=':`n$o"
  Check (Has $o 'serve: environment RADIANCE_BRL_TEST=r (from -Env)') "-Env adds a variable:`n$o"
  $r = Serve @($envProfile, '-Install', '-WhatIf', '-Exe', $Exe, '-Device', '2') @{ HIP_VISIBLE_DEVICES = $null; BRILLIANCE_HF_CACHE = $null }
  Check (Has $r.Out "-Name Environment -Type MultiString -Value @('HIP_VISIBLE_DEVICES=2', 'GPU_FLUSH_ON_EXECUTION=1', 'HIP_BRL_TEST_A=a', 'AMD_BRL_TEST_B=b=c')") "the profile's variables in the service environment:`n$($r.Out)"
  foreach ($bad in @(@('PATH=x', 'is not one serve.ps1 sets'), @('HIP_VISIBLE_DEVICES=1', 'choose the cards with -Bus'),
                     @('RADIANCE_API_KEY=x', 'looks like a secret'), @('GPU_X', 'is not NAME=VALUE'), @('GPU_X=', 'has no value'))) {
    $r = Serve @($profile, '-WhatIf', '-Exe', $Exe, '-Env', $bad[0])
    Check ($r.Rc -ne 0 -and (Has $r.Out $bad[1])) "-Env $($bad[0]) is refused ('$($bad[1])'):`n$($r.Out)"
  }
  $r = Serve @($profile, '-Uninstall', '-WhatIf', '-Env', 'GPU_X=1')
  Check ($r.Rc -ne 0 -and (Has $r.Out '-Env goes with a start or -Install')) '-Env with -Uninstall is refused'
  # a real start: the variables reach the process, and this shell's own value comes back after it
  $stub = Join-Path $tmp 'stub radiance.cmd'
  [IO.File]::WriteAllText($stub, "@set GPU_FLUSH_ON_EXECUTION`r`n@set HIP_BRL_TEST_A`r`n@exit /b 0`r`n")
  $o = & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "`$env:GPU_FLUSH_ON_EXECUTION = '7'; & '$serve' '$envProfile' -Exe '$stub' | Out-Host; 'rc=' + `$LASTEXITCODE; 'after=' + `$env:GPU_FLUSH_ON_EXECUTION + '/' + `$env:HIP_BRL_TEST_A" 2>&1 | ForEach-Object { "$_" }
  $o = $o -join "`n"
  Check ((Has $o 'GPU_FLUSH_ON_EXECUTION=1') -and (Has $o 'HIP_BRL_TEST_A=a') -and (Has $o 'rc=0') -and (Has $o 'after=7/')) "a start sets the variables for radiance only:`n$o"
  $r = Serve @($profile, '-Install', '-Uninstall', '-WhatIf')
  Check ($r.Rc -ne 0 -and (Has $r.Out 'one at a time')) '-Install with -Uninstall is refused'
  $r = Serve @($profile, '-Install', '-Background', '-WhatIf')
  Check ($r.Rc -ne 0 -and (Has $r.Out 'neither -Check nor -Background')) '-Install with -Background is refused'
  $r = Serve @($profile, '-Install', '-WhatIf', '-Name', 'bad/name')
  Check ($r.Rc -ne 0 -and (Has $r.Out 'a service name here is')) 'a bad service name is refused'
  $r = Serve @($profile, '-Manual')
  Check ($r.Rc -ne 0 -and (Has $r.Out 'go with -Install or -Uninstall')) '-Manual without -Install is refused'

  # ---- without elevation, a real -Install or -Uninstall stops before changing anything
  $admin = (New-Object Security.Principal.WindowsPrincipal ([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  if (-not $admin) {
    $r = Serve @($profile, '-Install', '-Exe', $Exe, '-Name', 'Brl-SvcTest')
    Check ($r.Rc -ne 0 -and (Has $r.Out 'needs an elevated PowerShell')) "-Install unelevated is refused:`n$($r.Out)"
    $r = Serve @($profile, '-Uninstall', '-Name', 'Brl-SvcTest')
    Check ($r.Rc -ne 0 -and (Has $r.Out 'needs an elevated PowerShell')) '-Uninstall unelevated is refused'
  }

  # ---- -Uninstall -WhatIf of a service that is not there
  $r = Serve @($profile, '-Uninstall', '-WhatIf', '-Name', 'Brl-SvcTest-not-installed')
  Check ($r.Rc -eq 0 -and (Has $r.Out 'there is no service Brl-SvcTest-not-installed; -Uninstall would do nothing')) "-Uninstall -WhatIf, nothing installed:`n$($r.Out)"

  # ---- the SCM helper compiles, and reaches the SCM (a service that is not there: 1060)
  $r = & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "& '$serve' '$profile' -Install -WhatIf -Exe '$Exe' 6>`$null | Out-Null; try { [BrlSvc]::SetPreshutdownTimeout('Brl-SvcTest-not-installed', 1000); 'no error' } catch { 'native ' + `$_.Exception.InnerException.NativeErrorCode }" 2>&1 |
       ForEach-Object { "$_" } | Select-Object -Last 1
  Check ("$r".Trim() -eq 'native 1060') "[BrlSvc] reaches the SCM (expected 1060, the service does not exist; got '$r')"
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
Write-Host "serve_service_test: $($script:checks - $script:failed) of $($script:checks) checks hold"
if ($script:failed) { exit 1 }
exit 0
