param([Parameter(Mandatory=$true)][ValidateRange(1,4)][int]$Rekan,[string]$Sesi='semalam_20261116')
$ErrorActionPreference='Stop'
if ($Sesi -notmatch '^[A-Za-z0-9_-]{1,60}$') { throw 'Sesi tidak valid' }
$Queue=Join-Path $PSScriptRoot "09_Riset_V4\outputs\queue_rekan_${Rekan}_${Sesi}"
if (-not (Test-Path $Queue)) { throw 'Antrean belum dibuat.' }
New-Item -ItemType File -Path (Join-Path $Queue 'STOP_AFTER_JOB') -Force | Out-Null
Write-Host 'Stop diminta. Fold aktif tetap diselesaikan agar checkpoint dan evaluasi aman.'
