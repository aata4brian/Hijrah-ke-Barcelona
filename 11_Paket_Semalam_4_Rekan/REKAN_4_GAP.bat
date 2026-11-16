@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0JALANKAN.ps1" -Rekan 4 %*
if errorlevel 1 echo GAGAL: simpan log, jangan hapus output.
pause
