#Requires -Version 5.1
<#
.SYNOPSIS
  Brilliance: fetch a Hugging Face repo with the curl.exe Windows ships and nothing else.

.DESCRIPTION
  The Windows twin of scripts/hfget.sh, step for step and exit code for exit code, so
  `radiance --model <org/repo>` behaves the same on both platforms (core/hf_model.cpp runs it):

    hfget.ps1 <repo-id> [dest-dir] [revision] [file]

  ONE FILE OF A REPO: with [file], only that file is fetched. A repo that holds a whole ladder of
  containers (sixteen sizes of one model, 170 GiB) is not what a profile serving one of them wants.
  Its manifest and ledger carry the file's name (.hfget-manifest-<file>), so they never claim the
  directory is a copy of the repo.

  Set HF_TOKEN for a gated or private repo. curl reads it from a config on its stdin, never from its
  command line, which other processes and process-creation auditing can read.

  Re-running resumes: a file whose local size matches the listing is skipped, and a short one of the
  version being fetched is continued with `curl -C -` and, once whole, checked against its LFS sha256.

  A CHANGED FILE IS FETCHED AGAIN WHOLE, at any size: a complete run records each file's content key
  (the LFS sha256, or the size for a small file) in <dest>\.hfget-manifest, written only when every
  file arrived and withdrawn before a run changes anything, so its presence means the directory is a
  complete copy of some revision. A run that stops part way leaves <dest>\.hfget-partial instead:
  the key of every file it started and of every file it knows to be whole, so the next run resumes a
  partial file only into the version it began, and hashes again only what nothing vouches for.

  WEIGHT DUPLICATES ARE SKIPPED: when the listing holds any .safetensors, the other weight formats
  (.bin .msgpack .h5 .onnx .onnx_data .gguf .rad) are left behind. A .pth is a duplicate only when it is
  the model's own weights again (pytorch_model*.pth, consolidated*.pth); any other is a part the
  safetensors do not hold (a codec, say) and is fetched, as upstream's hfget.sh does.

  A LISTED NAME THAT WOULD LAND OUTSIDE THE DESTINATION -- absolute, carrying a .. segment, holding
  a control character, a '#' or a '?' (where curl would end the path and fetch another object), or
  naming hfget's own manifest -- refuses the whole listing (exit 1), as
  hfget.sh does: skipping it would leave a directory that reads as complete and is not.

  EVERY LFS FILE IS HASHED once it has arrived and compared with the sha256 the listing claims; a
  mismatch deletes it and fails the run, so the hub cannot hand back one file for another at the
  same size, and a resumed transfer cannot splice into a plausible length.

  Adapted from Brilliance 0.2.2's scripts/hfget.ps1 to radiance 2.1's hfget.sh.

  Exit status: 0 complete; 3 nothing on disk was changed (the listing could not be read, what is
  still to fetch does not fit, or a file to be replaced is open in another process), so a complete
  earlier copy is intact; anything else, a transfer failed and a file may be partial -- re-run to
  resume.

  HFGET_JOBS concurrent transfers, default 4, 1 to 16. HF_ENDPOINT replaces https://huggingface.co
  (a mirror, or a local stand-in for tests).

  Windows PowerShell 5.1 compatible; ASCII only.
#>
param(
  [Parameter(Position = 0)][string]$Repo,
  [Parameter(Position = 1)][string]$Dest,
  [Parameter(Position = 2)][string]$Rev = 'main',
  [Parameter(Position = 3)][string]$Only = ''
)

$ErrorActionPreference = 'Continue'
# Any error this script did not expect ends it as a failed transfer: the manifest is never written
# past one, so a complete-looking directory is never left behind by a broken run. The transfers in
# flight end with it rather than go on writing behind a run that has already reported failure.
trap {
  [Console]::Error.WriteLine("hfget: $($_.Exception.Message)")
  foreach ($j in @($running)) { if ($j) { try { $j.Proc.Kill() } catch { } } }
  if ($manTmp -and (Test-Path -LiteralPath $manTmp)) { Remove-Item -LiteralPath $manTmp -Force }
  exit 1
}
$ProgressPreference = 'SilentlyContinue'   # the progress bar makes Invoke-WebRequest crawl
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

if (-not $Repo) { [Console]::Error.WriteLine('usage: hfget.ps1 <repo-id> [dest-dir] [revision]'); exit 2 }
if (-not $Dest) { $Dest = Join-Path (Join-Path $env:USERPROFILE 'models') ($Repo -replace '/', '\') }
# 0 would never start a transfer; past a few streams the link is saturated and the disk only seeks.
$jobs = 4
if ($env:HFGET_JOBS -match '^\d+$') { $jobs = [int][Math]::Max(1, [Math]::Min(16, [double]$env:HFGET_JOBS)) }
$curl = Join-Path $env:SystemRoot 'System32\curl.exe'
if (-not (Test-Path -LiteralPath $curl)) { [Console]::Error.WriteLine('hfget: curl.exe is required (Windows 10 1803 or later ships it)'); exit 2 }

$hub = 'https://huggingface.co'
if ($env:HF_ENDPOINT) { $hub = $env:HF_ENDPOINT.TrimEnd('/') }
# A control character in the token would end the curl config line and start another.
$token = ''
if ($env:HF_TOKEN) { $token = $env:HF_TOKEN -replace '[\x00-\x1f]', '' }
$headers = @{}
$curlCfg = ''
if ($token) {
  $headers['Authorization'] = "Bearer $token"
  $curlCfg = 'header = "Authorization: Bearer ' + ($token -replace '\\', '\\' -replace '"', '\"') + '"' + "`n"
  # Windows PowerShell opens a redirected stdin in Console.InputEncoding with its preamble, which
  # under code page 65001 is a UTF-8 BOM, and curl rejects the config whose first option it
  # prefixes. The same code page without a preamble; with no console to set it on, the field
  # behind it. PowerShell 7 drops the preamble itself.
  if ($PSVersionTable.PSEdition -ne 'Core' -and [Console]::InputEncoding.GetPreamble().Length -gt 0) {
    $plain = New-Object System.Text.UTF8Encoding($false)
    try { [Console]::InputEncoding = $plain } catch {
      $fld = [Console].GetField('_inputEncoding', [Reflection.BindingFlags]'NonPublic,Static')
      if ($fld) { $fld.SetValue($null, $plain) }
    }
  }
}

# Free space and the in-use probe below are Win32 calls: DriveInfo takes only a drive root, and .NET
# cannot open a file for DELETE access alone. Without them the script still runs, on DriveInfo and
# on Remove-Item.
$native = $false
try {
  Add-Type -Namespace HfGet -Name Native -ErrorAction Stop -MemberDefinition @'
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern bool GetDiskFreeSpaceEx(string directory, out ulong available, out ulong total, out ulong free);
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern Microsoft.Win32.SafeHandles.SafeFileHandle CreateFile(string name, uint access, uint share, IntPtr security, uint disposition, uint flags, IntPtr template);
'@
  $native = $true
} catch { $native = $false }

Write-Output "hfget: listing $Repo@$Rev"
# THE LISTING IS PAGINATED AND THE PAGE SIZE IS SMALL: follow the Link rel="next" cursor until
# there is none, or a checkpoint arrives missing most of its shards with every size check passing.
$api = "$hub/api/models/$Repo/tree/$Rev`?recursive=true&expand=true"
$listing = New-Object System.Collections.Generic.List[object]
$page = 0
while ($api) {
  try {
    $resp = Invoke-WebRequest -Uri $api -Headers $headers -UseBasicParsing -ErrorAction Stop
  } catch {
    [Console]::Error.WriteLine("hfget: listing $Repo@$Rev failed: $($_.Exception.Message)")
    [Console]::Error.WriteLine('hfget: a gated or private repo needs HF_TOKEN in the environment')
    exit 3
  }
  # -InputObject, not the pipeline: Windows PowerShell 5.1 hands a top-level JSON array down a
  # pipeline as ONE object, and the foreach statement below enumerates it either way.
  $body = $null
  try { $body = ConvertFrom-Json -InputObject $resp.Content } catch { $body = $null }
  if ($null -eq $body) { [Console]::Error.WriteLine("hfget: listing $Repo@$Rev returned no file list"); exit 3 }
  foreach ($e in $body) {
    if (-not $e.PSObject.Properties['type'] -or -not $e.PSObject.Properties['path']) {
      [Console]::Error.WriteLine("hfget: listing $Repo@$Rev returned no file list"); exit 3
    }
    $listing.Add($e)
  }
  $api = $null
  # One string: PowerShell 7 gives each header as an array, and -match on an array filters it
  # without setting $Matches.
  $link = @($resp.Headers['Link']) -join ', '
  if ($link -match '<([^>]*)>;\s*rel="next"') { $api = $Matches[1] }
  $page++
  if ($page -gt 200) { [Console]::Error.WriteLine('hfget: listing did not terminate after 200 pages'); exit 1 }
}
Write-Output "hfget: $page page(s) of listing"

$files = @($listing | Where-Object { $_.type -eq 'file' })
# Names that would not land inside the destination refuse the listing (hfget.sh's rule).
$bad = @($files | Where-Object {
  $p = [string]$_.path
  $p -match '^[/\\]' -or $p -match '(^|[/\\])\.\.([/\\]|$)' -or $p -match '[\x00-\x1f\x7f]' -or
  $p -match '(^|[/\\])\.hfget-manifest' -or $p -match '^[A-Za-z]:' -or $p -match '\\' -or
  # curl ends a URL path at '#' or '?' and fetches the stripped name, a real object a listing can
  # name with its size and oid, so the bytes would land under a name nobody asked for (upstream
  # 444b851, hfget.sh's screen).
  $p -match '[#?]'
} | ForEach-Object { $_.path })
if ($bad.Count -gt 0) {
  [Console]::Error.WriteLine("hfget: the listing names '$($bad -join ', ')', which does not land inside the destination")
  exit 1
}
$haveSt = @($files | Where-Object { $_.path -like '*.safetensors' }).Count
if ($haveSt -gt 0) {
  $files = @($files | Where-Object {
    -not ($_.path -match '\.(bin|msgpack|h5|onnx|onnx_data|gguf|rad)$' -or
          $_.path -match '(^|/)(pytorch_model|consolidated)[^/]*\.pth$')
  })
}
if ($Only) {
  $files = @($files | Where-Object { $_.path -eq $Only })
  if ($files.Count -eq 0) { [Console]::Error.WriteLine("hfget: $Repo@$Rev has no file '$Only'"); exit 3 }
}
if ($files.Count -eq 0) { [Console]::Error.WriteLine('hfget: the listing has no files'); exit 3 }

# Content keys: the LFS sha256, or the size for a file not in LFS. Sha is the key again when it is
# a sha256 this script can check.
$entries = @(foreach ($f in $files) {
  $oid = ''
  if ($f.PSObject.Properties['lfs'] -and $f.lfs -and $f.lfs.oid) { $oid = [string]$f.lfs.oid }
  $sha = ''
  if ($oid -match '^(sha256:)?([0-9a-fA-F]{64})$') { $sha = $Matches[2].ToLowerInvariant() }
  [pscustomobject]@{ Size = [int64]$f.size; Key = $(if ($oid) { $oid } else { [string]$f.size }); Sha = $sha; Path = [string]$f.path }
})

# An absolute provider path: curl and the .NET calls below resolve a relative one against the
# process directory, which is not PowerShell's, and Resolve-Path's .Path gives a UNC one back as
# FileSystem::\\server\share.
$Dest = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Dest)
[void][System.IO.Directory]::CreateDirectory($Dest)
$mark = if ($Only) { '-' + ($Only -replace '[\\/]', '_') } else { '' }
$manifest = Join-Path $Dest ".hfget-manifest$mark"
$manTmp = "$manifest.tmp"
$ledger = Join-Path $Dest ".hfget-partial$mark"
$recorded = @{}
if (Test-Path -LiteralPath $manifest) {
  foreach ($line in Get-Content -LiteralPath $manifest) {
    $t = $line.Split("`t", 2)
    if ($t.Count -eq 2) { $recorded[$t[1]] = $t[0] }
  }
}
# The ledger's lines are <part|done> <key> <path>, appended as the run goes; the last for a path wins.
$progress = @{}
if (Test-Path -LiteralPath $ledger) {
  foreach ($line in Get-Content -LiteralPath $ledger) {
    $t = $line.Split("`t", 3)
    if ($t.Count -eq 3) { $progress[$t[2]] = $t }
  }
}

function Local-Path([string]$p) { Join-Path $Dest ($p -replace '/', '\') }

function Add-Progress([string]$state, $e) {
  Add-Content -LiteralPath $ledger -Value ("{0}`t{1}`t{2}" -f $state, $e.Key, $e.Path) -Encoding ASCII -ErrorAction Stop
  $script:progress[$e.Path] = @($state, $e.Key, $e.Path)
}

# A 1 MiB read and CNG's SHA-256 rather than Get-FileHash, which takes about twice as long, and a
# shard is tens of GB. $null when the file cannot be read, which no key matches.
function Get-Sha256([string]$p) {
  $alg = $null
  try { $alg = New-Object System.Security.Cryptography.SHA256Cng } catch { $alg = [System.Security.Cryptography.SHA256]::Create() }
  try {
    $s = [System.IO.FileStream]::new($p, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read, 1MB, [System.IO.FileOptions]::SequentialScan)
    try { $h = $alg.ComputeHash($s) } finally { $s.Dispose() }
    return [BitConverter]::ToString($h).Replace('-', '').ToLowerInvariant()
  } catch {
    return $null
  } finally { $alg.Dispose() }
}

# GetDiskFreeSpaceEx takes any directory, so a share or a mounted folder reports its own volume and
# the caller's quota. $null when nothing can say, and the run goes ahead unchecked.
function Get-FreeBytes([string]$dir) {
  if ($script:native) {
    $a = [uint64]0; $t = [uint64]0; $f = [uint64]0
    $d = $dir; if (-not $d.EndsWith('\')) { $d += '\' }
    if ([HfGet.Native]::GetDiskFreeSpaceEx($d, [ref]$a, [ref]$t, [ref]$f)) { return [int64]$a }
  }
  $root = [System.IO.Path]::GetPathRoot($dir)
  if ($root -match '^[A-Za-z]:\\?$') {
    try { return [int64](New-Object System.IO.DriveInfo($root)).AvailableFreeSpace } catch { }
  }
  return $null
}

# DELETE access with every sharing mode granted fails exactly when another process holds the file
# without FILE_SHARE_DELETE -- an engine serving this copy -- which is when Remove-Item would.
function Test-InUse([string]$p) {
  if (-not $script:native) { return $false }
  $h = [HfGet.Native]::CreateFile($p, 0x00010000, 7, [IntPtr]::Zero, 3, 0, [IntPtr]::Zero)
  $busy = $h.IsInvalid -and (Test-Path -LiteralPath $p)
  $h.Dispose()
  return $busy
}

# WHAT THE LOCAL BYTES ARE decides what is done with them: the ledger's word when a run touched
# the file since the manifest, else the manifest's, else unknown. Resuming a file of another
# version gives the right length and the wrong bytes (or, when it shrank, curl's 416 forever), so
# only a short file of the version being fetched is resumed; one that is longer, or of another
# version, is fetched whole. A whole file nothing vouches for is hashed once, here.
$plan = New-Object System.Collections.Generic.List[object]
foreach ($e in $entries) {
  $lp = Local-Path $e.Path
  $len = [int64]-1
  if (Test-Path -LiteralPath $lp -PathType Leaf) { $len = (Get-Item -LiteralPath $lp).Length }
  $ver = ''; $whole = $false
  if ($progress.ContainsKey($e.Path)) { $ver = $progress[$e.Path][1]; $whole = $progress[$e.Path][0] -eq 'done' }
  elseif ($recorded.ContainsKey($e.Path)) { $ver = $recorded[$e.Path]; $whole = $true }
  $act = 'keep'; $why = ''
  if ($len -lt 0) { $act = 'new' }
  elseif ($len -gt $e.Size) { $act = 'fresh'; $why = 'longer than the listing' }
  elseif ($ver -and $ver -ne $e.Key) { $act = 'fresh'; $why = 'changed upstream' }
  elseif ($len -lt $e.Size) { $act = 'resume' }
  elseif (-not $whole -and $e.Sha) {
    Write-Output "hfget: checking $($e.Path) against its sha256 (no record of it)"
    if ((Get-Sha256 $lp) -eq $e.Sha) { Add-Progress 'done' $e }
    else { $act = 'fresh'; $why = 'sha256 mismatch' }
  }
  $plan.Add([pscustomobject]@{ Entry = $e; Local = $lp; Len = $len; Act = $act; Why = $why })
}

# WHAT IS STILL TO FETCH, not the repo's size: a warm cache must not read as a download that does
# not fit. A file counts from its local length, and whole when it is fetched again.
$todo = [int64]0
$need = [int64]0
foreach ($p in $plan) {
  $need += $p.Entry.Size
  if ($p.Act -eq 'resume') { $todo += $p.Entry.Size - $p.Len }
  elseif ($p.Act -ne 'keep') { $todo += $p.Entry.Size }
}
$free = Get-FreeBytes $Dest
Write-Output ("hfget: {0} file(s), {1:N1} GiB -> {2}" -f $entries.Count, ($need / 1GB), $Dest)
if ($haveSt -gt 0) { Write-Output 'hfget: safetensors present, pickle weight formats skipped' }
if ($null -eq $free) {
  Write-Output ("hfget: {0:N1} GiB to fetch" -f ($todo / 1GB))
  [Console]::Error.WriteLine('hfget: warning: free space on the destination cannot be measured; going ahead')
} else {
  Write-Output ("hfget: {0:N1} GiB to fetch, {1:N1} GiB free on the destination volume" -f ($todo / 1GB), ($free / 1GB))
  if ($todo -gt $free) { [Console]::Error.WriteLine('hfget: THE DOWNLOAD DOES NOT FIT'); exit 3 }
}

# A FILE TO BE REPLACED THAT ANOTHER PROCESS HOLDS stops the run before anything changes: deleting
# the rest would leave neither revision whole, and exit 3 lets hf_model.cpp serve the copy as is.
$stale = @($plan | Where-Object { $_.Act -eq 'fresh' })
foreach ($p in $stale) {
  if (Test-InUse $p.Local) {
    [Console]::Error.WriteLine("hfget: $($p.Entry.Path) is to be fetched again ($($p.Why)) but another process has it open; nothing was changed")
    exit 3
  }
}

# From the first change on, the directory is no longer the copy the manifest vouches for, and
# hf_model.cpp serves a manifest's directory when a later run exits 3. Its keys move to the ledger
# first, where the next run still finds which files are whole; from here on the exit is never 3.
$changed = $false
function Begin-Change {
  if ($script:changed) { return }
  $carry = @(foreach ($k in $recorded.Keys) { if (-not $progress.ContainsKey($k)) { "done`t$($recorded[$k])`t$k" } })
  if ($carry.Count -gt 0) { Add-Content -LiteralPath $ledger -Value $carry -Encoding ASCII -ErrorAction Stop }
  if (Test-Path -LiteralPath $manifest) { Remove-Item -LiteralPath $manifest -Force -ErrorAction Stop }
  $script:changed = $true
}

$failed = $false
# Before any transfer, so that the space they free is there and a delete that fails costs nothing
# else. The name still being there is the test: a delete Windows leaves pending behind another
# handle keeps it too, and curl would resume into it.
foreach ($p in $stale) {
  Begin-Change
  try { Remove-Item -LiteralPath $p.Local -Force -ErrorAction Stop } catch { }
  if (Test-Path -LiteralPath $p.Local) {
    [Console]::Error.WriteLine("hfget: $($p.Entry.Path) could not be deleted to fetch it again ($($p.Why))")
    $p.Act = 'failed'
    $failed = $true
  }
}

Set-Content -LiteralPath $manTmp -Value $null -Encoding ASCII
$running = New-Object System.Collections.Generic.List[object]

function Finish-One($job) {
  $job.Proc.WaitForExit()
  $e = $job.Entry
  $got = if (Test-Path -LiteralPath $job.Local -PathType Leaf) { (Get-Item -LiteralPath $job.Local).Length } else { -1 }
  if ($job.Proc.ExitCode -ne 0 -or $got -ne $e.Size) {
    [Console]::Error.WriteLine("hfget: $($e.Path) is $got bytes, listing says $($e.Size) (curl exit $($job.Proc.ExitCode))")
    $script:failed = $true
    return
  }
  # Every LFS file is checked against the sha256 the listing claims before it is kept: the size
  # alone would take a swapped file at the same length, and a resumed one at a spliced length. A
  # mismatch deletes it -- a fresh transfer is the only repair -- and nothing reaches the manifest.
  if ($e.Sha) {
    $got = Get-Sha256 $job.Local
    if ($got -ne $e.Sha) {
      [Console]::Error.WriteLine("hfget: $($e.Path) hashes to $got, the listing claims $($e.Sha)")
      try { Remove-Item -LiteralPath $job.Local -Force -ErrorAction Stop } catch { }
      $script:failed = $true
      return
    }
  }
  Add-Content -LiteralPath $manTmp -Value ("{0}`t{1}" -f $e.Key, $e.Path) -Encoding ASCII
  Add-Progress 'done' $e
}

foreach ($p in $plan) {
  $e = $p.Entry
  if ($p.Act -eq 'failed') { continue }
  if ($p.Act -eq 'keep') {
    Write-Output "  = $($e.Path)"
    Add-Content -LiteralPath $manTmp -Value ("{0}`t{1}" -f $e.Key, $e.Path) -Encoding ASCII
    continue
  }
  if ($p.Act -eq 'fresh') { Write-Output "  ~ $($e.Path) ($($p.Why), fetching it whole)" }
  elseif ($p.Act -eq 'resume') { Write-Output "  + $($e.Path) (resuming at $($p.Len) of $($e.Size))" }
  else { Write-Output "  > $($e.Path)" }
  Begin-Change
  Add-Progress 'part' $e
  [void][System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($p.Local))
  $url = "$hub/$Repo/resolve/$Rev/$($e.Path)"
  # One command line, quoted here: ProcessStartInfo takes the arguments as a single string. The
  # token goes on stdin (-K -), never here.
  $cargs = @('-sSL', '--fail', '--retry', '5', '--retry-delay', '5', '--retry-all-errors', '-C', '-')
  if ($curlCfg) { $cargs += @('-K', '-') }
  $cargs += @('-o', "`"$($p.Local)`"", "`"$url`"")
  while ($running.Count -ge $jobs) {
    $done = $null
    foreach ($j in $running) { if ($j.Proc.HasExited) { $done = $j; break } }
    if ($done) { Finish-One $done; [void]$running.Remove($done) } else { Start-Sleep -Milliseconds 200 }
  }
  # Not Start-Process: it cannot hand the child a pipe for stdin. The console and the standard
  # output and error are inherited, as -NoNewWindow did.
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $curl
  $psi.Arguments = $cargs -join ' '
  $psi.UseShellExecute = $false
  $psi.RedirectStandardInput = [bool]$curlCfg
  $proc = [System.Diagnostics.Process]::Start($psi)
  if ($curlCfg) {
    # A curl that has already died is reported by its exit status in Finish-One.
    try { $proc.StandardInput.Write($curlCfg); $proc.StandardInput.Close() } catch { }
  }
  $running.Add([pscustomobject]@{ Proc = $proc; Entry = $e; Local = $p.Local; Resumed = ($p.Act -eq 'resume') })
}
foreach ($j in $running.ToArray()) { Finish-One $j; [void]$running.Remove($j) }

if ($failed) {
  Remove-Item -LiteralPath $manTmp -Force
  [Console]::Error.WriteLine('hfget: incomplete -- re-run to resume')
  exit 1
}
Move-Item -LiteralPath $manTmp -Destination $manifest -Force
if (Test-Path -LiteralPath $ledger) { Remove-Item -LiteralPath $ledger -Force }
$total = if ($Only) { ($entries | Measure-Object Size -Sum).Sum } else { (Get-ChildItem -LiteralPath $Dest -Recurse -File | Measure-Object Length -Sum).Sum }
Write-Output ("hfget: complete -- {0:N1} GiB in {1}" -f ($total / 1GB), $Dest)
exit 0
