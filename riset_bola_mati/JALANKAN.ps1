param(
    [ValidateSet('probe','audio','flags','all')][string]$Mode='all',
    [string]$Video,
    [string]$Homography,
    [double]$Mulai=450,
    [double]$Durasi=30,
    [double]$SampleHz=2,
    [switch]$SeluruhVideo,
    [switch]$Overlay
)
$ErrorActionPreference='Stop'
$Root=$PSScriptRoot
$Python=Join-Path $Root '..\09_Riset_V4\.venv_repro_v44\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Environment integrator tidak ditemukan. Gunakan Python environment sendiri dan run.py; lihat README.md.'
}
$ConfigPath=Join-Path $Root '..\08_Paket_Tim\.mtah-local.json'
if (-not $Video) {
    $LocalCopy=Join-Path $Root '..\09_Riset_V4\video_cache\BAR-ATH_tactical_buffered.mp4'
    if (Test-Path -LiteralPath $LocalCopy) { $Video=$LocalCopy }
    elseif (Test-Path -LiteralPath $ConfigPath) {
        $Config=Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
        $Video=Join-Path $Config.data_dir 'tactical\BAR-ATH_tactical.mp4'
    } else { throw 'Berikan -Video dengan path video lokal.' }
}
if (-not $Homography -and (Test-Path -LiteralPath $ConfigPath)) {
    $Config=Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
    $Homography=Join-Path $Config.data_dir 'tactical\BAR-ATH_tactical_homography.csv'
}
$Arguments=@((Join-Path $Root 'run.py'),$Mode,'--video',$Video,'--sample-hz',"$SampleHz")
if ($Mode -ne 'probe') {
    $Output=Join-Path $Root ('outputs\'+$Mode+'_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    $Arguments+=@('--out',$Output)
    if ($SeluruhVideo) { $Arguments+=@('--start','0') }
    else { $Arguments+=@('--start',"$Mulai",'--duration',"$Durasi") }
    if ($Homography) { $Arguments+=@('--homography',$Homography) }
    if ($Overlay) { $Arguments+='--overlay' }
}
$env:PYTHONNOUSERSITE='1'
$env:OMP_NUM_THREADS='4'
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw 'Run gagal. Periksa pesan error/manifest; hasil lama tidak ditimpa.' }
