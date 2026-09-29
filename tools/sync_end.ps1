# End-of-session sync (run on any machine AFTER working).
#   1) this machine's Claude memory -> 공용\ai-memory
#   2) local heavy outputs -> Google Drive
#   3) git add / commit / pull --rebase / push main
# Usage:
#   powershell -ExecutionPolicy Bypass -File <repo>\tools\sync_end.ps1 [-Message "what I did"]
# Saved as UTF-8 with BOM (Korean paths) - keep the BOM when editing.
param([string]$Drive = $env:FARMAI_DRIVE, [string]$Message = "")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "sync_common.ps1")

$current = git -C $Root branch --show-current
if ($current -ne "main") {
    throw "This machine is on branch '$current', not main. Run tools\sync_start.ps1 first (it moves you to main)."
}

Write-Host "== [1/3] Claude memory -> 공용\ai-memory ==" -ForegroundColor Cyan
$mem = Get-ClaudeMemoryDir $Root
if (Test-Path $mem) { Copy-Newer $mem (Join-Path $Root "공용\ai-memory") }

Write-Host "== [2/3] local outputs -> Google Drive ==" -ForegroundColor Cyan
$DriveDir = Find-SyncDrive $Drive
if ($DriveDir) {
    foreach ($pair in (Get-ArtifactPairs $Root $DriveDir)) { Copy-Newer $pair.Local $pair.Remote }
} else {
    Write-Warning "Google Drive folder 'farmai_sync' not found - heavy outputs NOT pushed."
}

Write-Host "== [3/3] git commit + push ==" -ForegroundColor Cyan
git -C $Root add -A
$pending = git -C $Root status --porcelain
if ($pending) {
    if (-not $Message) { $Message = "sync: $env:COMPUTERNAME $(Get-Date -Format 'yyyy-MM-dd HH:mm')" }
    git -C $Root commit -m $Message
}
git -C $Root pull --rebase --autostash
if ($LASTEXITCODE -ne 0) { throw "git pull --rebase failed - resolve the conflict (git status), then run this script again." }
git -C $Root push origin main
if ($LASTEXITCODE -ne 0) { throw "git push failed." }
Write-Host "Done. Everything is on GitHub / Drive." -ForegroundColor Green
