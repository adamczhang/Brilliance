#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: win\serve.ps1 as a launcher -- the profile radiance reads, the listener and key it
  applies, the cards it chooses and reports, and -Background's log. No card is used.
.DESCRIPTION
  ctest runs it (win\tests\CMakeLists.txt) with the radiance.exe of the build tree:
    powershell -File win\tests\serve_launcher_test.ps1 -Exe build-win\hip\bin\radiance.exe
  The HIP runtime is never asked: BRILLIANCE_SERVE_TEST_DEVICES hands serve.ps1 a device table
  (INDEX:BUS:INTEGRATED), and radiance runs only as --check, or with HIP_VISIBLE_DEVICES=-1 on a model
  that is not there. Exit 0 when every check holds, 1 otherwise (each failure printed).

    the profile    radiance is started on the profile itself (not a copy), so a reload rereads it
    the listener   -Listen, -Port and -ApiKey as radiance's own reader applies them, whatever the
                   profile's layout (flow mapping, comments, anchors, a server block at the end)
    the cards      -Bus (decimal, 0x, PCI address), -Device, the caller's mask, and the default that
                   hides an integrated GPU; what radiance will see, by PCI bus
    the log        -Background -Log with a bare file name
    the CPU        -Cpu: no card asked or shown, the host backend, the CPU kernels and no drafting
                   over any profile, in radiance's own reading; no CPU start without it
    the model      -Model over the profile's model: (one model, or -ModelName's of several)
#>
param([Parameter(Mandatory)][string]$Exe)
$ErrorActionPreference = 'Stop'
$serve = (Resolve-Path (Join-Path $PSScriptRoot '..\serve.ps1')).Path
$Exe = (Resolve-Path -LiteralPath $Exe).Path
$script:failed = 0
$script:checks = 0
function Check([bool]$ok, [string]$what) {
  $script:checks++
  if (-not $ok) { $script:failed++; Write-Host "  FAIL $what" }
}
# The machine on which the tests below run: the iGPU first, then four discrete cards, as on the test machine.
$red = '0:33:1,1:6:0,2:3:0,3:9:0,4:32:0'
# serve.ps1 in a PowerShell of its own, as a person runs it; its output and exit code. Every call
# gets a device table (default: the test machine's) and clean server variables unless it says otherwise.
# -InSession runs it as a person types `serve.ps1 ... 2>&1` at a Windows PowerShell 5.1 prompt: the
# redirect in the same session as the script, where a native command's stderr lines become
# ErrorRecords. (A separate powershell.exe -File never shows that: there the redirect is the
# process's.) Each ErrorRecord that reaches the caller is marked 'ERRORRECORD '.
function Serve([string[]]$a, [hashtable]$envSet = @{}, [string]$cwd = '', [switch]$InSession) {
  $all = @{ BRILLIANCE_SERVE_TEST_DEVICES = $red; HIP_VISIBLE_DEVICES = $null
            BRILLIANCE_SERVER_HOST = $null; BRILLIANCE_SERVER_PORT = $null; BRILLIANCE_SERVER_API_KEY = $null
            BRILLIANCE_DEVICE = $null; BRILLIANCE_SERVER_KERNELS = $null; BRILLIANCE_MODEL_SPECULATION = $null
            BRILLIANCE_MODEL_PATH = $null; BRILLIANCE_MODEL_NAME = $null }
  foreach ($k in $envSet.Keys) { $all[$k] = $envSet[$k] }
  $saved = @{}
  foreach ($k in $all.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $all[$k]) }
  $here = Get-Location
  try {
    if ($cwd) { Set-Location -LiteralPath $cwd }
    $ErrorActionPreference = 'Continue'
    if ($InSession) {
      # An option name stays bare, so it binds as the parameter; every value is quoted.
      $q = @($a | ForEach-Object { if ($_ -match '^-[A-Za-z]+$') { $_ } else { "'" + ($_ -replace "'", "''") + "'" } }) -join ' '
      $cmd = "& '$($serve -replace "'", "''")' $q 2>&1 | ForEach-Object { if (`$_ -is [System.Management.Automation.ErrorRecord]) { 'ERRORRECORD ' + `$_ } else { `"`$_`" } }; exit `$LASTEXITCODE"
      $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))
      $o = & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $enc 2>&1 | ForEach-Object { "$_" }
    } else {
      $o = & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $serve @a 2>&1 | ForEach-Object { "$_" }
    }
    return [pscustomobject]@{ Out = ($o -join "`n"); Rc = $LASTEXITCODE }
  } finally {
    Set-Location $here
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
  }
}
function Has([string]$text, [string]$part) { return $text.Contains($part) }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("brl-serve-launcher-test-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory $tmp | Out-Null
try {
  $model = 'C:/nonexistent/a.rad'
  # ======================================================================= the listener and the key
  # The profile layouts the old line rewrite misread; radiance's reader takes each the same way.
  $layouts = [ordered]@{
    'block, commented' = "# a profile`nserver:`n  host: 127.0.0.1   # loopback`n  # port: 1`n  port: 18371`nmodels:`n  a:`n    model: $model`n"
    'flow mapping'     = "server: {host: 127.0.0.1, port: 18371}  # inline`nmodels:`n  a:`n    model: $model`n"
    'server: comment'  = "server:   # the listener`n  host: 127.0.0.1`n  port: 18371`nmodels:`n  a:`n    model: $model`n"
    'anchor'           = "x-net: &net`n  host: 127.0.0.1`n  port: 18371`nserver:`n  <<: *net`nmodels:`n  a:`n    model: $model`n"
    'block at EOF'     = "models:`n  a:`n    model: $model`nserver:`n  host: 127.0.0.1`n  port: 18371"
  }
  $i = 0
  foreach ($name in $layouts.Keys) {
    $i++
    $p = Join-Path $tmp "layout $i.yaml"
    [IO.File]::WriteAllText($p, $layouts[$name])
    # as the file says
    $r = Serve @($p, '-WhatIf', '-Exe', $Exe)
    Check ($r.Rc -eq 0 -and (Has $r.Out 'radiance reads the profile as: host 127.0.0.1 (from server.host), port 18371 (from server.port), API key none')) "${name}: the file's listener:`n$($r.Out)"
    # what was asked, over it: what the launcher says it applied is what radiance reads
    $r = Serve @($p, '-WhatIf', '-Exe', $Exe, '-Listen', '0.0.0.0', '-Port', '18372', '-ApiKey', 'env:BRL_LAUNCHER_TEST_KEY') @{ BRL_LAUNCHER_TEST_KEY = 'k-test' }
    $o = $r.Out
    Check ($r.Rc -eq 0) "${name}: -WhatIf with overrides exits 0:`n$o"
    Check ((Has $o 'BRILLIANCE_SERVER_HOST=0.0.0.0') -and (Has $o 'BRILLIANCE_SERVER_PORT=18372') -and (Has $o 'BRILLIANCE_SERVER_API_KEY=env:BRL_LAUNCHER_TEST_KEY')) "${name}: the launcher names what it applies:`n$o"
    Check (Has $o 'radiance reads the profile as: host 0.0.0.0 (from environment (BRILLIANCE_SERVER_HOST)), port 18372 (from environment (BRILLIANCE_SERVER_PORT)), API key env BRL_LAUNCHER_TEST_KEY (environment (BRILLIANCE_SERVER_API_KEY))') "${name}: radiance reads what the launcher applied:`n$o"
    Check (-not (Has $o 'k-test')) "${name}: the key itself is never printed"
    # -Check reads the same way, through radiance
    $r = Serve @($p, '-Check', '-Exe', $Exe, '-Port', '18373', '-ApiKey', 'off')
    Check ($r.Rc -eq 0 -and (Has $r.Out 'server host 127.0.0.1 (server.host), port 18373 (environment (BRILLIANCE_SERVER_PORT)), api key none')) "${name}: -Check with -Port:`n$($r.Out)"
  }
  # a key in the file, turned off; the file's other keys stay
  $withKey = Join-Path $tmp 'with key.yaml'
  [IO.File]::WriteAllText($withKey, "server: {port: 18374, api-key: sk-in-the-file}`nmodels:`n  a:`n    model: $model`n")
  $r = Serve @($withKey, '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out 'API key in the file (server.api-key)') -and -not (Has $r.Out 'sk-in-the-file')) "the file's key, not shown:`n$($r.Out)"
  $r = Serve @($withKey, '-WhatIf', '-Exe', $Exe, '-ApiKey', 'off')
  Check ($r.Rc -eq 0 -and (Has $r.Out 'port 18374 (from server.port), API key none')) "-ApiKey off over the file's key:`n$($r.Out)"
  # a value left in the caller's shell does not apply when nothing was asked
  $r = Serve @((Join-Path $tmp 'layout 1.yaml'), '-WhatIf', '-Exe', $Exe) @{ BRILLIANCE_SERVER_PORT = '9999' }
  Check ($r.Rc -eq 0 -and (Has $r.Out 'port 18371 (from server.port)') -and -not (Has $r.Out '9999')) "a stray BRILLIANCE_SERVER_PORT is cleared:`n$($r.Out)"
  # refusals
  $r = Serve @((Join-Path $tmp 'layout 1.yaml'), '-WhatIf', '-Exe', $Exe, '-ApiKey', 'sk-literal')
  Check ($r.Rc -ne 0 -and (Has $r.Out 'a key itself does not belong on a command line')) "a literal key is refused:`n$($r.Out)"
  $r = Serve @((Join-Path $tmp 'layout 1.yaml'), '-WhatIf', '-Exe', $Exe, '-Env', 'BRILLIANCE_SERVER_HOST=0.0.0.0')
  Check ($r.Rc -ne 0 -and (Has $r.Out '-Listen, -Port and -ApiKey')) "-Env BRILLIANCE_SERVER_HOST is refused:`n$($r.Out)"
  $bad = Join-Path $tmp 'bad.yaml'
  [IO.File]::WriteAllText($bad, "server:`n  hostt: 0.0.0.0`nmodels:`n  a:`n    model: $model`n")
  $r = Serve @($bad, '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out 'radiance refuses the profile') -and (Has $r.Out 'hostt')) "-WhatIf shows radiance's refusal:`n$($r.Out)"

  # ======================================================================= the profile itself
  # A real start, with a stub in radiance's place that prints what it was given.
  $stubDir = Join-Path $tmp 'stub bin'
  New-Item -ItemType Directory $stubDir | Out-Null
  $stub = Join-Path $stubDir 'radiance.cmd'
  [IO.File]::WriteAllText($stub, "@echo args=%*`r`n@set BRILLIANCE_SERVER_`r`n@echo mask=%HIP_VISIBLE_DEVICES%`r`n@exit /b 0`r`n")
  $p1 = Join-Path $tmp 'layout 1.yaml'
  $r = Serve @($p1, '-Exe', $stub, '-Listen', '0.0.0.0', '-Bus', '3')
  $o = $r.Out
  Check ($r.Rc -eq 0) "a start exits with radiance's status:`n$o"
  Check (Has $o "args=--config `"$p1`"") "radiance is given the profile itself, not a copy:`n$o"
  Check ((Has $o 'BRILLIANCE_SERVER_HOST=0.0.0.0') -and -not (Has $o 'BRILLIANCE_SERVER_PORT=')) "radiance gets the asked override only:`n$o"
  Check (Has $o 'mask=2') "radiance gets the card's HIP index:`n$o"
  Check (Has $o "radiance reads $p1 itself: to change a model while it runs, edit that file and run tools\reload-server.ps1") "the start says which file a reload reads:`n$o"
  $r = Serve @($p1, '-WhatIf', '-Exe', $Exe)
  Check ((Has $r.Out "--config `"$p1`"") -and -not (Has $r.Out 'brilliance-layout')) "-WhatIf runs the profile itself, no Temp copy:`n$($r.Out)"

  # ======================================================================= a caller's 2>&1 (5.1)
  # serve.ps1 ... -Check 2>&1 at a 5.1 prompt: radiance writes its lines to stderr, which used to
  # stop the script at the first one with a NativeCommandError (the alpha.4 walkthrough). Now its
  # lines arrive as text, and the exit code is radiance's.
  $r = Serve @($p1, '-Check', '-Exe', $Exe) -InSession
  Check ($r.Rc -eq 0) "-Check 2>&1 in the caller's session exits 0:`n$($r.Out)"
  Check ((Has $r.Out "I $($p1): model a = $model") -and -not (Has $r.Out 'ERRORRECORD') -and -not (Has $r.Out 'NativeCommandError')) "-Check 2>&1: radiance's lines as text, no error record:`n$($r.Out)"
  $r = Serve @((Join-Path $tmp 'not there.yaml'), '-Check', '-Exe', $Exe) -InSession
  Check ($r.Rc -ne 0 -and -not (Has $r.Out 'NativeCommandError')) "-Check 2>&1 on a refused profile exits nonzero with radiance's words:`n$($r.Out)"
  # A start under the same redirect: a server's stderr line no longer ends the launcher while the
  # server runs on; its later lines and its exit code come through.
  $stub2 = Join-Path $stubDir 'radiance-stderr.cmd'
  [IO.File]::WriteAllText($stub2, "@echo first line, on stderr 1>&2`r`n@echo second line, on stdout`r`n@echo third line, on stderr 1>&2`r`n@exit /b 7`r`n")
  $r = Serve @($p1, '-Exe', $stub2, '-Bus', '3') -InSession
  Check ($r.Rc -eq 7 -and (Has $r.Out 'second line, on stdout') -and (Has $r.Out 'third line, on stderr') -and -not (Has $r.Out 'NativeCommandError')) "a start under 2>&1 runs radiance to its end and returns its status:`n$($r.Out)"
  # -Install: the service runs a copy of the profile as it is, overrides in its environment
  $r = Serve @($p1, '-Install', '-WhatIf', '-Exe', $Exe, '-Name', 'Brl-LaunchTest', '-Bus', '9', '-Listen', '0.0.0.0', '-ApiKey', 'off') @{ BRILLIANCE_HF_CACHE = $null }
  $o = $r.Out
  Check ($r.Rc -eq 0) "-Install -WhatIf exits 0:`n$o"
  foreach ($l in ($layouts['block, commented'].TrimEnd("`n") -split "`n")) { Check (Has $o "| $l") "the service's YAML is the profile ('$l'):`n$o" }
  Check (Has $o "-Name Environment -Type MultiString -Value @('HIP_VISIBLE_DEVICES=3', 'BRILLIANCE_SERVER_HOST=0.0.0.0', 'BRILLIANCE_SERVER_API_KEY=off')") "the card and the overrides in the service environment:`n$o"
  Check (Has $o 'the service will listen on 0.0.0.0 port 18371 (host from environment (BRILLIANCE_SERVER_HOST), port from server.port), no API key') "-Install says what radiance will listen on:`n$o"
  # A profile named with a dot keeps all of its name in the service's (ltx-2.5-1card, not ltx-2).
  # Given bare, as `serve.ps1 ltx-2.5-1card` is; here a file of that name in the current folder.
  $dotted = Join-Path $tmp 'ltx-2.5-1card'
  Copy-Item $p1 $dotted
  $r = Serve @('ltx-2.5-1card', '-Install', '-WhatIf', '-Exe', $Exe) @{ BRILLIANCE_HF_CACHE = $null } $tmp
  Check ($r.Rc -eq 0 -and (Has $r.Out 'create Brilliance-ltx-2.5-1card ') -and -not (Has $r.Out 'Brilliance-ltx-2 ')) "a dotted profile name is the service's whole:`n$($r.Out)"
  $r = Serve @($withKey, '-Install', '-WhatIf', '-Exe', $Exe, '-ApiKey', 'off')
  Check ($r.Rc -ne 0 -and (Has $r.Out 'holds an API key itself')) "a profile with a key is refused for a service even with -ApiKey off (the copy would hold it):`n$($r.Out)"

  # ======================================================================= the cards
  $r = Serve @($p1, '-WhatIf', '-Exe', $Exe)
  $o = $r.Out
  Check (Has $o 'HIP_VISIBLE_DEVICES=1,2,3,4') "no -Bus: the discrete cards, the iGPU hidden:`n$o"
  Check ((Has $o 'card 0 = PCI bus 6 (0x06), HIP device 1') -and (Has $o 'card 3 = PCI bus 32 (0x20), HIP device 4')) "the cards radiance will see, by bus:`n$o"
  Check (Has $o 'hidden from HIP: PCI bus 33 (0x21), integrated') "the hidden iGPU is named:`n$o"
  Check (Has $o '4 visible cards: elastic memory (VMM) is off by default') "the visible-device effect:`n$o"
  foreach ($c in @(@('3', '2', 'card 0 = PCI bus 3 (0x03), HIP device 2'), @('0x20', '4', 'card 0 = PCI bus 32 (0x20)'),
                   @('0000:09:00.0', '3', 'card 0 = PCI bus 9 (0x09)'), @('9,3', '3,2', 'card 1 = PCI bus 3 (0x03), HIP device 2'))) {
    $r = Serve @($p1, '-WhatIf', '-Exe', $Exe, '-Bus', $c[0])
    Check ($r.Rc -eq 0 -and (Has $r.Out "HIP_VISIBLE_DEVICES=$($c[1])") -and (Has $r.Out $c[2])) "-Bus $($c[0]):`n$($r.Out)"
  }
  $r = Serve @($p1, '-WhatIf', '-Exe', $Exe, '-Bus', '3')
  Check ((Has $r.Out 'one visible card: elastic memory (VMM) can be on') -and (Has $r.Out 'hidden from HIP: PCI bus 33 (0x21), integrated; PCI bus 6')) "one card by bus:`n$($r.Out)"
  # the common one-card machine: an iGPU and one card, no -Bus needed
  $r = Serve @($p1, '-WhatIf', '-Exe', $Exe) @{ BRILLIANCE_SERVE_TEST_DEVICES = '0:33:1,1:3:0' }
  Check ((Has $r.Out 'HIP_VISIBLE_DEVICES=1') -and (Has $r.Out 'card 0 = PCI bus 3 (0x03), HIP device 1') -and (Has $r.Out 'one visible card')) "an iGPU and one card:`n$($r.Out)"
  # no iGPU: nothing to hide, no mask
  $r = Serve @($p1, '-WhatIf', '-Exe', $Exe) @{ BRILLIANCE_SERVE_TEST_DEVICES = '0:3:0' }
  Check ((Has $r.Out 'HIP_VISIBLE_DEVICES not set') -and (Has $r.Out 'card 0 = PCI bus 3 (0x03), HIP device 0')) "one card, no iGPU:`n$($r.Out)"
  # the caller's mask stands (HIP is asked through it, so a card it hides is not asked about), and its
  # effect is shown
  $r = Serve @($p1, '-WhatIf', '-Exe', $Exe) @{ HIP_VISIBLE_DEVICES = '0,2' }
  Check ((Has $r.Out 'HIP_VISIBLE_DEVICES as this shell set it') -and (Has $r.Out 'card 0 = PCI bus 33 (0x21), integrated') -and
         (Has $r.Out 'card 1 = PCI bus 3 (0x03)') -and -not (Has $r.Out 'hidden from HIP') -and
         (Has $r.Out 'the integrated GPU is visible')) "the caller's mask, iGPU visible:`n$($r.Out)"
  # every card hidden: shown, and refused -- a CPU start is never made unasked
  $r = Serve @($p1, '-WhatIf', '-Exe', $Exe) @{ HIP_VISIBLE_DEVICES = '-1' }
  Check ((Has $r.Out 'HIP_VISIBLE_DEVICES=-1') -and (Has $r.Out 'radiance sees no card')) "the caller hides every card:`n$($r.Out)"
  Check ($r.Rc -ne 0 -and (Has $r.Out 'never done unasked: to serve on the CPU, add -Cpu')) "no card and no -Cpu is refused:`n$($r.Out)"
  $r = Serve @($p1, '-Exe', $stub, '-Device', '-1')
  Check ($r.Rc -ne 0 -and -not (Has $r.Out 'args=') -and (Has $r.Out 'add -Cpu')) "-Device -1 does not start radiance on the CPU:`n$($r.Out)"
  $r = Serve @($p1, '-WhatIf', '-Exe', $Exe, '-Device', '3')
  Check ((Has $r.Out 'HIP_VISIBLE_DEVICES=3') -and (Has $r.Out 'card 0 = PCI bus 9 (0x09), HIP device 3')) "-Device by HIP index:`n$($r.Out)"
  # refusals, and no HIP at all
  $r = Serve @($p1, '-WhatIf', '-Exe', $Exe, '-Bus', '5')
  Check ($r.Rc -ne 0 -and (Has $r.Out 'no HIP device on PCI bus 5') -and (Has $r.Out 'PCI bus 33 (0x21), integrated')) "a bus with no card:`n$($r.Out)"
  $r = Serve @($p1, '-WhatIf', '-Exe', $Exe, '-Bus', 'three')
  Check ($r.Rc -ne 0 -and (Has $r.Out 'a PCI bus is a decimal number')) "a bus that is not a number:`n$($r.Out)"
  $r = Serve @($p1, '-WhatIf', '-Exe', $Exe) @{ BRILLIANCE_SERVE_TEST_DEVICES = 'none' }
  Check ($r.Rc -eq 0 -and (Has $r.Out 'the PCI buses are unknown here')) "no HIP runtime: a start still goes ahead:`n$($r.Out)"
  $r = Serve @($p1, '-WhatIf', '-Exe', $Exe, '-Bus', '3') @{ BRILLIANCE_SERVE_TEST_DEVICES = 'none' }
  Check ($r.Rc -ne 0 -and (Has $r.Out '-Bus needs the HIP runtime')) "no HIP runtime: -Bus is refused:`n$($r.Out)"

  # ======================================================================= -Cpu
  # A profile that asks for card kernels and a drafter, and n-gram drafting: -Cpu replaces all three
  # in radiance's own reading of it, and names the source of each.
  $cardy = Join-Path $tmp 'cardy.yaml'
  [IO.File]::WriteAllText($cardy, "server:`n  port: 18376`n  kernels: libr4d,libref`ndefaults:`n  num-speculative-tokens: 7`nmodels:`n  a:`n    model: $model`n  b:`n    model: $model`n    num-speculative-tokens: 4`n    spec-ngram: 4`n")
  $r = Serve @($cardy, '-Cpu', '-WhatIf', '-Exe', $Exe)
  $o = $r.Out
  Check ($r.Rc -eq 0) "-Cpu -WhatIf exits 0:`n$o"
  Check ((Has $o 'cards (-Cpu): HIP_VISIBLE_DEVICES=-1') -and (Has $o 'no GPU: radiance serves on the CPU') -and -not (Has $o 'PCI bus')) "-Cpu shows no card and asks HIP about none:`n$o"
  Check ((Has $o 'BRILLIANCE_DEVICE=host') -and (Has $o 'BRILLIANCE_SERVER_KERNELS=libavx,libref') -and (Has $o 'BRILLIANCE_MODEL_SPECULATION=off')) "-Cpu names what it applies:`n$o"
  Check (Has $o 'radiance reads: device host: the CPU only, HIP not asked (environment (BRILLIANCE_DEVICE))') "radiance takes the host backend:`n$o"
  Check (Has $o 'radiance reads: kernels libavx,libref (environment (BRILLIANCE_SERVER_KERNELS))') "the CPU kernels over the profile's own list:`n$o"
  Check ((Has $o 'model a: model from models.a.model, num-speculative-tokens 0 (environment (BRILLIANCE_MODEL_SPECULATION))') -and
         (Has $o 'model b: model from models.b.model, num-speculative-tokens 0 (environment (BRILLIANCE_MODEL_SPECULATION))')) "no drafting for every model, n-gram drafting dropped:`n$o"
  $r = Serve @($cardy, '-WhatIf', '-Exe', $Exe)
  Check ((Has $r.Out 'kernels libr4d,libref (server.kernels)') -and (Has $r.Out 'num-speculative-tokens 7 (defaults.num-speculative-tokens)')) "without -Cpu the profile's own settings stand:`n$($r.Out)"
  # a start: the stand-in radiance gets the mask and the settings; a stray value in the shell does not
  $r = Serve @($p1, '-Exe', $stub, '-Cpu') @{ HIP_VISIBLE_DEVICES = '2' }
  Check ($r.Rc -eq 0 -and (Has $r.Out 'mask=-1') -and (Has $r.Out 'BRILLIANCE_SERVER_KERNELS=libavx,libref')) "a -Cpu start hides every card:`n$($r.Out)"
  $r = Serve @($p1, '-Exe', $stub, '-Bus', '3') @{ BRILLIANCE_SERVER_KERNELS = 'libavx' }
  Check ($r.Rc -eq 0 -and -not (Has $r.Out 'BRILLIANCE_SERVER_KERNELS')) "a stray BRILLIANCE_SERVER_KERNELS is cleared:`n$($r.Out)"
  # -Check through radiance; the service
  $r = Serve @($cardy, '-Cpu', '-Check', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out 'device host: the CPU only') -and (Has $r.Out 'kernels libavx,libref (environment (BRILLIANCE_SERVER_KERNELS))')) "-Cpu -Check:`n$($r.Out)"
  $r = Serve @($p1, '-Install', '-WhatIf', '-Exe', $Exe, '-Name', 'Brl-LaunchTest', '-Cpu') @{ BRILLIANCE_HF_CACHE = $null }
  Check ($r.Rc -eq 0 -and (Has $r.Out "-Value @('HIP_VISIBLE_DEVICES=-1', 'BRILLIANCE_DEVICE=host', 'BRILLIANCE_SERVER_KERNELS=libavx,libref', 'BRILLIANCE_MODEL_SPECULATION=off')")) "-Install -Cpu puts -Cpu in the service environment:`n$($r.Out)"
  # refusals
  $r = Serve @($p1, '-Cpu', '-Bus', '3', '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out 'takes neither -Bus nor -Device')) "-Cpu with -Bus:`n$($r.Out)"
  $r = Serve @($p1, '-WhatIf', '-Exe', $Exe, '-Env', 'BRILLIANCE_DEVICE=host')
  Check ($r.Rc -ne 0 -and (Has $r.Out '-Cpu, -Model, -Fetch and -ModelName set it')) "-Env BRILLIANCE_DEVICE is refused:`n$($r.Out)"

  # ======================================================================= -Model
  $two = Join-Path $tmp 'two.yaml'
  [IO.File]::WriteAllText($two, "server:`n  port: 18377`nmodels:`n  a:`n    model: $model`n  b:`n    model: $model`n    model-revision: main`n")
  $file = Join-Path $tmp 'my model.rad'
  [IO.File]::WriteAllText($file, 'x')
  $r = Serve @($p1, '-Model', $file, '-Bus', '3', '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out "BRILLIANCE_MODEL_PATH=$file (from -Model") -and (Has $r.Out 'model a: model from environment (BRILLIANCE_MODEL_PATH)')) "-Model over the one model:`n$($r.Out)"
  $r = Serve @($p1, '-Model', $file, '-Bus', '3', '-Check', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out "model a = $file, tp 1")) "-Check serves the file -Model names:`n$($r.Out)"
  $r = Serve @($two, '-Model', $file, '-Bus', '3', '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out 'models: names 2 (a, b); name it with BRILLIANCE_MODEL_NAME (serve.ps1 -ModelName)')) "two models and no -ModelName:`n$($r.Out)"
  $r = Serve @($two, '-Model', $file, '-ModelName', 'b', '-Bus', '3', '-Check', '-Exe', $Exe)
  Check ($r.Rc -eq 0 -and (Has $r.Out "model b = $file, tp 1") -and (Has $r.Out "model a = $model, tp 1")) "-ModelName b, its model-revision dropped:`n$($r.Out)"
  $r = Serve @($two, '-Model', $file, '-ModelName', 'c', '-Bus', '3', '-Check', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out 'BRILLIANCE_MODEL_NAME (c) is not a model under models: (a, b)')) "-ModelName of no model:`n$($r.Out)"
  $r = Serve @($p1, '-Model', (Join-Path $tmp 'not there.rad'), '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out 'no such file or folder')) "-Model of nothing:`n$($r.Out)"
  $r = Serve @($p1, '-ModelName', 'a', '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out '-ModelName says which model')) "-ModelName alone:`n$($r.Out)"
  $r = Serve @($p1, '-Fetch', '-WhatIf', '-Exe', $Exe)
  Check ($r.Rc -ne 0 -and (Has $r.Out "has no '# brilliance-fetch:")) "-Fetch on a profile that names nothing to fetch:`n$($r.Out)"

  # ======================================================================= -Background -Log NAME
  # A bare file name, against the caller's location. radiance (-Cpu: no card) fails on the missing
  # model and exits; the log has the launcher's lines, radiance's, and its exit status.
  $work = Join-Path $tmp 'work dir'
  New-Item -ItemType Directory $work | Out-Null
  $bgProfile = Join-Path $tmp 'bg.yaml'
  [IO.File]::WriteAllText($bgProfile, "server:`n  host: 127.0.0.1`n  port: 18375`nmodels:`n  a:`n    model: $model`n")
  $r = Serve @($bgProfile, '-Background', '-Log', 'bare.log', '-Cpu', '-Exe', $Exe) @{} $work
  $log = Join-Path $work 'bare.log'
  Check (-not (Has $r.Out 'Cannot bind argument') -and -not (Has $r.Out 'empty string')) "-Log NAME is accepted:`n$($r.Out)"
  $deadline = (Get-Date).AddSeconds(60); $text = ''
  while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $log) { $text = [IO.File]::ReadAllText($log); if ($text -match 'radiance exit status') { break } }
    Start-Sleep -Milliseconds 500
  }
  Check (Test-Path -LiteralPath $log) "the log is in the caller's folder ($log):`n$($r.Out)"
  Check ((Has $text "--config $bgProfile") -and (Has $text 'serve: cards (-Cpu): HIP_VISIBLE_DEVICES=-1') -and
         (Has $text 'serve: environment BRILLIANCE_DEVICE=host (from -Cpu)')) "the log starts with the launcher's lines:`n$text"
  # A HIP build says it was told to use no GPU (core/device/dispatch.cpp); a host build has no HIP to
  # ask and only names the host backend. Either way no HIP backend and no "built with HIP" fallback
  # line, which is what a HIP build that ignored -Cpu would print.
  $hipLine = Has $text 'device: this process uses no GPU (dev_host_only) -- the host backend, HIP not asked'
  $hostOnly = (Has $text 'device: host backend,') -and -not (Has $text 'built with HIP')
  Check (($hipLine -or $hostOnly) -and -not (Has $text 'device: native HIP backend')) "radiance asks HIP for no device under -Cpu:`n$text"
  Check ($text -match 'radiance exit status [1-9]') "radiance ran, and its exit status is logged:`n$text"
  $r = Serve @($bgProfile, '-WhatIf', '-Background', '-Log', 'bare.log', '-Exe', $Exe) @{} $work
  Check (Has $r.Out "-Log `"$(Join-Path $work 'bare.log')`"") "-WhatIf shows the log's full path:`n$($r.Out)"

  # ======================================================================= tools\reload-server.ps1
  # It names the file the server rereads, from the server's command line, and says when that is an
  # older serve.ps1's temporary copy. A cmd.exe stands in for radiance (it has no reload event: exit 2).
  $reload = (Resolve-Path (Join-Path $PSScriptRoot '..\tools\reload-server.ps1')).Path
  foreach ($c in @(@($p1, $false), @((Join-Path $env:TEMP 'brilliance-layout 1-4242.yaml'), $true))) {
    $fake = Start-Process -FilePath "$env:SystemRoot\System32\cmd.exe" -ArgumentList "/d /c ping -n 8 127.0.0.1 >nul & rem --config `"$($c[0])`"" -WindowStyle Hidden -PassThru
    $ErrorActionPreference = 'Continue'
    $o = (& powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $reload -ProcessId $fake.Id 2>&1 | ForEach-Object { "$_" }) -join "`n"
    $rc = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    Check ((Has $o "reload: process $($fake.Id) rereads $($c[0])") -and $rc -eq 2) "reload-server names the file it rereads:`n$o"
    Check ((Has $o 'temporary copy an older serve.ps1 made') -eq $c[1]) "reload-server flags a temporary copy only when it is one:`n$o"
    $fake.WaitForExit(15000) | Out-Null
  }
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
Write-Host "serve_launcher_test: $($script:checks - $script:failed) of $($script:checks) checks hold"
if ($script:failed) { exit 1 }
exit 0
