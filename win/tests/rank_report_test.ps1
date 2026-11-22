#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\tools\rank-report.ps1')

function Check([bool]$ok, [string]$label) {
  if (-not $ok) { throw "FAIL $label" }
  Write-Output "PASS $label"
}

# Captured from final-bench's tp4-server.log and alpha.5 RC4's a6-ds2-server.log.
$tp4Lines = @'
I   card 0000:03:00.0  rank 1 (device 2)  0x1002:0x7551 "AMD Radeon AI PRO R9700"  31.86 GiB VRAM  NUMA -1  driver amduw23g-205499-ac7b5433
I   card 0000:06:00.0  rank 0 (device 1)  0x1002:0x7551 "AMD Radeon AI PRO R9700"  31.86 GiB VRAM  NUMA -1  driver amduw23g-205499-ac7b5433
I   card 0000:09:00.0  rank 2 (device 3)  0x1002:0x7551 "AMD Radeon AI PRO R9700"  31.86 GiB VRAM  NUMA -1  driver amduw23g-205499-ac7b5433
I   card 0000:20:00.0  rank 3 (device 4)  0x1002:0x7551 "AMD Radeon AI PRO R9700"  31.86 GiB VRAM  NUMA -1  driver amduw23g-205499-ac7b5433
I   card 0000:21:00.0  device 0, not serving  0x1002:0x13c0 "AMD Radeon(TM) Graphics"  2.00 GiB VRAM  NUMA -1  driver amduw23g-205498-7af7ada9
'@ -split "`r?`n"
$tp2Lines = @'
I   card 0000:03:00.0  rank 0 (device 0)  0x1002:0x7551 "AMD Radeon AI PRO R9700"  31.86 GiB VRAM  NUMA -1  driver amduw23g-205499-ac7b5433
I   card 0000:06:00.0  not serving  0x1002:0x7551  31.86 GiB VRAM  NUMA -1  driver amduw23g-205499-ac7b5433
I   card 0000:09:00.0  rank 1 (device 1)  0x1002:0x7551 "AMD Radeon AI PRO R9700"  31.86 GiB VRAM  NUMA -1  driver amduw23g-205499-ac7b5433
I   card 0000:20:00.0  not serving  0x1002:0x7551  31.86 GiB VRAM  NUMA -1  driver amduw23g-205499-ac7b5433
'@ -split "`r?`n"

$tp4 = @(Get-RadianceRankRows $tp4Lines)
Check (($tp4.Bus -join ',') -eq '6,3,9,32') 'TP4 ranks follow the recorded assignments'
$rankNumbers = $tp4 | ForEach-Object { $_.Rank }
Check (($rankNumbers -join ',') -eq '0,1,2,3' -and ($tp4.Device -join ',') -eq '1,2,3,4') 'TP4 distinguishes ranks from HIP device ordinals'
Check ($tp4[3].Address -eq '0000:20:00.0' -and $tp4[3].Name -eq 'AMD Radeon AI PRO R9700') 'PCI address and device name are parsed'
Check ($tp4[0].Line -ceq $tp4Lines[1]) 'the original engine line is preserved'
Check ((Compare-RadianceRankBuses @(3, 9, 6, 32) $tp4).Ok) 'TP4 request passes in a different order'

$tp2 = @(Get-RadianceRankRows $tp2Lines)
Check (($tp2.Bus -join ',') -eq '3,9' -and ($tp2.Device -join ',') -eq '0,1') 'TP2 includes only serving cards'
Check ((Compare-RadianceRankBuses @(9, 3) $tp2).Ok) 'TP2 request passes in a different order'
$mismatch = Compare-RadianceRankBuses @(3, 6) $tp2
Check (-not $mismatch.Ok -and ($mismatch.Missing -join ',') -eq '6' -and ($mismatch.Unexpected -join ',') -eq '9') 'TP2 request for the wrong bus fails'
Check (-not (Compare-RadianceRankBuses @(3, 9) @()).Ok) 'missing rank lines fail'
