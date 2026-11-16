param(
    [Parameter(Mandatory=$true)][ValidateRange(1,4)][int]$Rekan,
    [double]$Jam = 8,
    [string]$Sesi = 'semalam_20261116',
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto',
    [string]$Python = '',
    [switch]$CekSaja
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $Python) {
    $Candidates = @('.venv\Scripts\python.exe','..\09_Riset_V4\.venv\Scripts\python.exe')
    $Python = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $Python) { throw 'Python environment belum ada. Jalankan PASANG_ENV.ps1, atau berikan -Python path/python.exe.' }
$Python = (Resolve-Path -LiteralPath $Python).Path
& $Python (Join-Path $PSScriptRoot 'verify_payload.py')
if ($LASTEXITCODE -ne 0) { throw 'Hash payload gagal; minta copy utuh dari integrator.' }
$Project = Join-Path $PSScriptRoot '09_Riset_V4'
if (-not (Test-Path (Join-Path $Project 'riset_v4\team_queue.py'))) { throw 'Payload belum disiapkan integrator. Jangan copy launcher saja.' }
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='1'
$env:LOKY_MAX_CPU_COUNT='4'
$env:PYTHONNOUSERSITE='1'
Push-Location -LiteralPath $Project
try {
    if ($Rekan -eq 2 -and $Device -eq 'cpu') { throw 'Tugas 2 harus CUDA. Tukar tugas dengan pemilik NVIDIA.' }
    # Windows PowerShell treats native stderr as an error even when tests pass.
    # Redirect inside Python; preserve unittest's real exit status.
    & $Python -c "import sys,unittest; sys.stderr=sys.stdout; unittest.main(module=None, argv=['unittest','riset_v4.test_search_decoder','riset_v4.test_team','riset_v4.test_team_integrity','-q'])"
    if ($LASTEXITCODE -ne 0) { throw 'Preflight tests gagal; jangan mulai semalam.' }
    if ($Rekan -eq 2 -and $Device -ne 'cpu') {
        & $Python -c "import torch; assert torch.cuda.is_available(), 'Rekan 2 butuh NVIDIA CUDA'; print(torch.cuda.get_device_name(0))"
        if ($LASTEXITCODE -ne 0) { throw 'CUDA tidak tersedia. Tukar tugas dengan rekan yang punya GPU.' }
    }
    if ($CekSaja) { Write-Host 'PREFLIGHT_OK: belum ada eksperimen panjang diluncurkan.'; return }
    if ($Rekan -eq 1) {
        & $Python -m riset_v4.team_public_audit --assets '..\audit_publik' --output 'outputs\audit_publik_beku'
        if ($LASTEXITCODE -ne 0) { throw 'Audit kandidat beku gagal. Lihat laporan; jangan tuning memakai GT publik.' }
    }
    & $Python -u -m riset_v4.team_queue --member $Rekan --source 'frozen_v44' --cache 'external_cache' --output 'outputs' --hours $Jam --session $Sesi --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'Antrean gagal. Simpan output dan periksa queue_status.json serta fold_N.log.' }
    Write-Host 'Antrean selesai atau berhenti halus. Jalankan KUMPULKAN.ps1 untuk mengirim hasil.'
} finally { Pop-Location }
