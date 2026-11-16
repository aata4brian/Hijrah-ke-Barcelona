param([Parameter(Mandatory=$true)][ValidateRange(1,4)][int]$Rekan,[string]$Sesi='semalam_20261116',[string]$Python='')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $Python) {
    $Python=@('.venv\Scripts\python.exe','..\09_Riset_V4\.venv\Scripts\python.exe') | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Python) { throw 'Berikan -Python path/python.exe' }
$Python=(Resolve-Path -LiteralPath $Python).Path
if ($Sesi -notmatch '^[A-Za-z0-9_-]{1,60}$') { throw 'Sesi tidak valid' }
$Destination=Join-Path $PSScriptRoot ('HASIL_REKAN_'+$Rekan+'_'+$Sesi+'_'+(Get-Date -Format 'yyyyMMdd_HHmmss')+'.zip')
Push-Location (Join-Path $PSScriptRoot '09_Riset_V4')
try {
    & $Python -m riset_v4.team_results export --member $Rekan --session $Sesi --destination $Destination
    if ($LASTEXITCODE -ne 0) { throw 'Ekspor gagal. Pastikan antrean sudah berhenti, jangan hapus output.' }
    Write-Host "Kirim file lokal ini ke integrator: $Destination"
} finally { Pop-Location }
