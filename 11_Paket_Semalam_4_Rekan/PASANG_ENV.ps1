param([ValidateSet('cpu','cuda')][string]$Device='cpu')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
if (Test-Path '.venv') { throw '.venv sudah ada; tidak ditimpa. Gunakan environment itu atau rename sendiri setelah diperiksa.' }
Write-Host 'Membuat environment lokal Python 3.12 dan mengunduh dependensi. Tidak mengubah instalasi global.'
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Pasang Python 3.12 x64 terlebih dahulu.' }
$Python='.venv\Scripts\python.exe'
$Index=if ($Device -eq 'cuda') { 'https://download.pytorch.org/whl/cu121' } else { 'https://download.pytorch.org/whl/cpu' }
& $Python -m pip install --timeout 120 --retries 5 'torch==2.5.1' --index-url $Index
if ($LASTEXITCODE -ne 0) { throw 'Unduhan Torch gagal; environment parsial disimpan. Ulangi pip pada environment itu, jangan install global.' }
& $Python -m pip install --timeout 120 --retries 5 -r requirements-team.txt
if ($LASTEXITCODE -ne 0) { throw 'Instalasi dependensi gagal; simpan log.' }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'pip check gagal.' }
Write-Host 'Instalasi selesai. Jalankan JALANKAN.ps1 -Rekan N -CekSaja sebelum malam.'
