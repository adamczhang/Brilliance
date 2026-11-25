#Requires -Version 5.1
<#
  Brilliance: choose the MSVC toolset win\setup.ps1 records in .toolchain\toolchain.json (and
  win\env.ps1 then selects with vcvars64 -vcvars_ver). Dot-sourced by win\setup.ps1 and by
  win\tests\msvc_toolset_test.ps1, which runs it against fake VC\Tools\MSVC folders.

  Select-MsvcToolset -MsvcRoot <VS>\VC\Tools\MSVC -Tested 14.51 returns $null when no toolset fits,
  otherwise an object with:
    Name      the folder name, the full toolset version (14.51.36231)
    Fallback  $false when it is the tested <Tested>.* toolset, $true when it is a newer one

  The tested toolset wins whenever one is installed, the newest build of it if there are several,
  so a Visual Studio update that adds a newer toolset beside it does not change the build. Only
  when it is missing does a newer toolset of the same major version (14.x, the binary-compatible
  v14x line) stand in, the newest. An older one never does. Versions are compared as [version],
  never as strings, so 14.100 would sort above 14.52 and 14.9 below 14.51.

  Written for Windows PowerShell 5.1; ASCII only.
#>

function Select-MsvcToolset {
  param(
    [Parameter(Mandatory = $true)][string]$MsvcRoot,
    [Parameter(Mandatory = $true)][string]$Tested
  )
  $min = [version]"$Tested.0"
  $found = @()
  foreach ($d in @(Get-ChildItem -LiteralPath $MsvcRoot -Directory -ErrorAction SilentlyContinue)) {
    $v = $null
    if ([version]::TryParse($d.Name, [ref]$v)) { $found += [pscustomobject]@{ Name = $d.Name; Version = $v } }
  }
  $same = @($found | Where-Object { $_.Version.Major -eq $min.Major -and $_.Version.Minor -eq $min.Minor } |
              Sort-Object Version -Descending)
  if ($same.Count) { return [pscustomobject]@{ Name = $same[0].Name; Fallback = $false } }
  $newer = @($found | Where-Object { $_.Version.Major -eq $min.Major -and $_.Version -ge $min } |
             Sort-Object Version -Descending)
  if ($newer.Count) { return [pscustomobject]@{ Name = $newer[0].Name; Fallback = $true } }
  return $null
}
