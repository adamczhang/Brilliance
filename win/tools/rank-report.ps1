# Read the engine's startup report rows for serving GPUs. The report is printed in PCI address
# order, so use the recorded rank numbers to present it in rank order. The device number is the
# HIP ordinal inside the server process.
function Get-RadianceRankRows([string[]]$Lines) {
  $pattern = '^I\s+card\s+(?<address>[0-9a-f]{4}:(?<bus>[0-9a-f]{2}):[0-9a-f]{2}\.[0-7])\s+rank\s+(?<rank>\d+)\s+\(device\s+(?<device>\d+)\)\s+\S+(?: "(?<name>[^"]*)")? {2,}'
  $rows = foreach ($line in $Lines) {
    if ($line -match $pattern) {
      [pscustomobject]@{
        Rank = [int]$Matches['rank']
        Device = [int]$Matches['device']
        Bus = [Convert]::ToInt32($Matches['bus'], 16)
        Address = $Matches['address']
        Name = $Matches['name']
        Line = $line
      }
    }
  }
  $rows | Sort-Object -Property Rank, Line -Unique
}

function Compare-RadianceRankBuses([int[]]$RequestedBuses, [object[]]$Ranks) {
  $requested = @($RequestedBuses | Sort-Object -Unique)
  $opened = @($Ranks | ForEach-Object { $_.Bus } | Sort-Object -Unique)
  $missing = @($requested | Where-Object { $opened -notcontains $_ })
  $unexpected = @($opened | Where-Object { $requested -notcontains $_ })
  [pscustomobject]@{
    Ok = $requested.Count -gt 0 -and $missing.Count -eq 0 -and $unexpected.Count -eq 0
    Requested = $requested
    Opened = $opened
    Missing = $missing
    Unexpected = $unexpected
  }
}
