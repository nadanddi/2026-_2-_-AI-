# Start-of-session sync (run on any machine BEFORE working).
#   1) git: switch to main (the only branch since 2026-09-29) and pull
#   2) Google Drive  -> local heavy outputs (집\클로드\research\local, ... see sync_common.ps1)
#   3) 공용\ai-memory -> this machine's Claude memory folder
# Usage (PowerShell, from anywhere):
#   powershell -ExecutionPolicy Bypass -File <repo>\tools\sync_start.ps1
# Optional: -Drive "G:\My Drive\farmai_sync"   (or set env var FARMAI_DRIVE once)
# Saved as UTF-8 with BOM (Korean paths) - keep the BOM when editing.
param([string]$Drive = $env:FARMAI_DRIVE)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "sync_common.ps1")

Write-Host "== [1/3] git (main) ==" -ForegroundColor Cyan
git -C $Root fetch --all --prune
$current = git -C $Root branch --show-current
if ($current -ne "main") {
    # Old branches (feature/rmse-improvement, codex-ec, claude-*) were merged into main
    # and deleted on 2026-09-29.  Move this machine to main, keeping any local commits.
    if (git -C $Root status --porcelain) {
        throw "Uncommitted changes on branch '$current'. Commit them (git add -A; git commit -m ...) and run this script again."
    }
    git -C $Root switch main
    if ($LASTEXITCODE -ne 0) { throw "git switch main failed." }
    git -C $Root pull --rebase --autostash
    if ($LASTEXITCODE -ne 0) { throw "git pull failed - resolve the conflict (git status), then run this script again." }
    git -C $Root merge --no-edit $current
    if ($LASTEXITCODE -ne 0) {
        git -C $Root merge --abort
        Write-Warning "Local commits on '$current' could not be merged into main automatically. Ask the AI: '$current 브랜치 main에 병합 충돌 해결해줘'."
    } else { Write-Host "  now on main (local commits of '$current' kept)" }
} else {
    git -C $Root pull --rebase --autostash
    if ($LASTEXITCODE -ne 0) { throw "git pull failed - resolve the conflict (git status), then run this script again." }
}
$global:LASTEXITCODE = 0
Move-OldLayout $Root
Set-LabDataLink $Root

Write-Host "== [2/3] Google Drive -> local outputs ==" -ForegroundColor Cyan
$DriveDir = Find-SyncDrive $Drive
if ($DriveDir) {
    foreach ($pair in (Get-ArtifactPairs $Root $DriveDir)) { Copy-Newer $pair.Remote $pair.Local }
} else {
    Write-Warning "Google Drive folder 'farmai_sync' not found - heavy outputs NOT pulled (see 공용\컴퓨터_설정_안내.md)."
}

Write-Host "== [3/3] AI memory -> Claude ==" -ForegroundColor Cyan
$mem = Get-ClaudeMemoryDir $Root
Copy-Newer (Join-Path $Root "공용\ai-memory") $mem

Write-Host "Done. Read 00_먼저_읽기.md first." -ForegroundColor Green
