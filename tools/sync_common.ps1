# Shared helpers for sync_start.ps1 / sync_end.ps1 (dot-sourced).

function Find-SyncDrive([string]$Hint) {
    # A folder named "farmai_sync" at the top of any mounted drive's first level
    # (Google Drive for desktop: G:\<My Drive>\farmai_sync).  No Korean literals here
    # on purpose: Windows PowerShell 5.1 reads BOM-less scripts as ANSI.
    if ($Hint -and (Test-Path $Hint)) { return $Hint }
    foreach ($d in Get-PSDrive -PSProvider FileSystem) {
        $direct = Join-Path $d.Root "farmai_sync"
        if (Test-Path $direct) { return $direct }
        $hit = Get-ChildItem -Path $d.Root -Directory -ErrorAction SilentlyContinue |
               ForEach-Object { Join-Path $_.FullName "farmai_sync" } |
               Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($hit) { return $hit }
    }
    return $null
}

function Get-ArtifactPairs([string]$Root, [string]$DriveDir) {
    # Heavy, git-ignored outputs.  Add a line here to sync another folder.
    @(
        @{ Local = (Join-Path $Root "research\local"); Remote = (Join-Path $DriveDir "research_local") },
        @{ Local = (Join-Path $Root "analysis\local"); Remote = (Join-Path $DriveDir "analysis_local") },
        @{ Local = (Join-Path $Root "blind");          Remote = (Join-Path $DriveDir "blind") }
    )
}

function Copy-Newer([string]$From, [string]$To) {
    # Copies files that are new or newer; never deletes anything on the target.
    if (-not (Test-Path $From)) { Write-Host "  (skip, missing) $From"; return }
    New-Item -ItemType Directory -Force -Path $To | Out-Null
    robocopy $From $To /E /XO /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE): $From -> $To" }
    $global:LASTEXITCODE = 0
    Write-Host "  synced  $From  ->  $To"
}

function Get-ClaudeMemoryDir([string]$Root) {
    # Claude Code keeps per-project memory in ~/.claude/projects/<slug>/memory,
    # slug = project path with every non [A-Za-z0-9] character replaced by '-'.
    $slug = ($Root -replace '[^A-Za-z0-9]', '-')
    $slug = $slug.Substring(0, 1).ToLower() + $slug.Substring(1)
    Join-Path $env:USERPROFILE ".claude\projects\$slug\memory"
}
