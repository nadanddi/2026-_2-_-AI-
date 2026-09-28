# Start-of-session sync (run on any machine BEFORE working).
#   1) git pull (code, docs, catalog, handoff)
#   2) Google Drive  -> local heavy outputs (research/local, analysis/local, blind)
#   3) docs/ai-memory -> this machine's Claude memory folder
# Usage (PowerShell, from anywhere):
#   powershell -ExecutionPolicy Bypass -File <repo>\tools\sync_start.ps1
# Optional: -Drive "G:\My Drive\farmai_sync"   (or set env var FARMAI_DRIVE once)
param([string]$Drive = $env:FARMAI_DRIVE)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "sync_common.ps1")

Write-Host "== [1/3] git pull ==" -ForegroundColor Cyan
git -C $Root fetch --all --prune
git -C $Root pull --rebase --autostash
if ($LASTEXITCODE -ne 0) { throw "git pull failed - resolve the conflict (git status), then run this script again." }

# Bring in Codex's EC branch so HANDOFF.md / reports from Codex are visible here too.
$current = git -C $Root branch --show-current
git -C $Root rev-parse --verify --quiet origin/codex-ec | Out-Null
if ($LASTEXITCODE -eq 0 -and $current -ne "codex-ec") {
    git -C $Root merge-base --is-ancestor origin/codex-ec HEAD
    if ($LASTEXITCODE -ne 0) {
        git -C $Root merge --no-edit origin/codex-ec
        if ($LASTEXITCODE -ne 0) {
            git -C $Root merge --abort
            Write-Warning "codex-ec could not be merged automatically (conflict). Ask the AI: 'codex-ec 브랜치 병합 충돌 해결해줘'."
        } else { Write-Host "  merged origin/codex-ec" }
    }
}
$global:LASTEXITCODE = 0

Write-Host "== [2/3] Google Drive -> local outputs ==" -ForegroundColor Cyan
$DriveDir = Find-SyncDrive $Drive
if ($DriveDir) {
    foreach ($pair in (Get-ArtifactPairs $Root $DriveDir)) { Copy-Newer $pair.Remote $pair.Local }
} else {
    Write-Warning "Google Drive folder 'farmai_sync' not found - heavy outputs NOT pulled (see HANDOFF.md, section Sync)."
}

Write-Host "== [3/3] AI memory -> Claude ==" -ForegroundColor Cyan
$mem = Get-ClaudeMemoryDir $Root
Copy-Newer (Join-Path $Root "docs\ai-memory") $mem

Write-Host "Done. Read HANDOFF.md first." -ForegroundColor Green
