# End-of-session sync (run on any machine AFTER working).
#   1) this machine's Claude memory -> docs/ai-memory
#   2) local heavy outputs -> Google Drive
#   3) git add / commit / pull --rebase / push (all branches, so Codex's branch goes too)
# Usage:
#   powershell -ExecutionPolicy Bypass -File <repo>\tools\sync_end.ps1 [-Message "what I did"]
param([string]$Drive = $env:FARMAI_DRIVE, [string]$Message = "")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "sync_common.ps1")

Write-Host "== [1/3] Claude memory -> docs/ai-memory ==" -ForegroundColor Cyan
$mem = Get-ClaudeMemoryDir $Root
if (Test-Path $mem) { Copy-Newer $mem (Join-Path $Root "docs\ai-memory") }

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
git -C $Root push origin --all
if ($LASTEXITCODE -ne 0) { throw "git push failed." }
Write-Host "Done. Everything is on GitHub / Drive." -ForegroundColor Green
