$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
Get-ChildItem '09_Riset_V4\outputs\queue_rekan_*\queue_status.json' -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host $_.FullName
    Get-Content -LiteralPath $_.FullName
}
