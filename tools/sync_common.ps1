# Shared helpers for sync_start.ps1 / sync_end.ps1 (dot-sourced).
# Saved as UTF-8 *with BOM*: Windows PowerShell 5.1 reads BOM-less scripts as ANSI,
# which breaks the Korean folder names below.  Keep the BOM when editing.

# Folder layout since 2026-09-29:  <place>\<ai>\  with place = 집(home) / 연구실(lab),
# ai = 클로드 / 코덱스, plus 공용 (shared) and 제출 (submissions).
$Places = @("집", "연구실")
$Ais = @("클로드", "코덱스")

function Find-SyncDrive([string]$Hint) {
    # A folder named "farmai_sync" at the top of any mounted drive's first level
    # (Google Drive for desktop: G:\<My Drive>\farmai_sync).
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
    # Drive folder names stay as before the reorganisation so old uploads are reused.
    $pairs = @(
        @{ Local = (Join-Path $Root "집\클로드\research\local");  Remote = (Join-Path $DriveDir "research_local") },
        @{ Local = (Join-Path $Root "집\코덱스\analysis\local");  Remote = (Join-Path $DriveDir "analysis_local") },
        @{ Local = (Join-Path $Root "집\클로드\blind");           Remote = (Join-Path $DriveDir "blind") }
    )
    # Every <place>\<ai>\local folder (e.g. 연구실\클로드\local -> lab_claude_local).
    $placeKey = @{ "집" = "home"; "연구실" = "lab" }
    $aiKey = @{ "클로드" = "claude"; "코덱스" = "codex" }
    foreach ($p in $Places) { foreach ($a in $Ais) {
        $pairs += @{ Local = (Join-Path $Root "$p\$a\local"); Remote = (Join-Path $DriveDir "$($placeKey[$p])_$($aiKey[$a])_local") }
    } }
    $pairs
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

function Move-OldLayout([string]$Root) {
    # One-time migration after pulling the 2026-09-29 reorganisation: git moves the
    # tracked files, but git-ignored outputs stay at the old paths.  Move them over.
    # Whole old folders are moved (not just their local/ outputs), so uncommitted scripts
    # or notes left there are not silently dropped: the old paths are git-ignored now.
    # Files that already exist at the new path are never overwritten (/XC /XN /XO);
    # anything left behind is reported so it can be checked by hand.
    $moves = @(
        @("docs\ai-memory",           "공용\ai-memory"),
        @("docs\연구실PC_설정_안내.md", "공용"),
        @("docs\변경이력",             "연구실\클로드\작업일지"),
        @("docs",                     "연구실\클로드\문서"),
        @("research\submissions",     "제출\미분류_옛위치"),
        @("research",                 "집\클로드\research"),
        @("analysis",                 "집\코덱스\analysis"),
        @("blind",                    "집\클로드\blind"),
        @("Claude",                   "연구실\클로드"),
        @("온라인대회자료",            "공용\대회자료")
    )
    foreach ($m in $moves) {
        $old = Join-Path $Root $m[0]; $new = Join-Path $Root $m[1]
        if (-not (Test-Path $old)) { continue }
        New-Item -ItemType Directory -Force -Path $new | Out-Null
        if (Test-Path $old -PathType Leaf) {
            $dest = Join-Path $new (Split-Path $old -Leaf)
            if (-not (Test-Path $dest)) { Move-Item $old $dest; Write-Host "  moved old-layout file  $($m[0])  ->  $($m[1])" }
            continue
        }
        robocopy $old $new /E /MOVE /XC /XN /XO /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE): $old -> $new" }
        $global:LASTEXITCODE = 0
        Write-Host "  moved old-layout folder  $($m[0])  ->  $($m[1])"
    }
    foreach ($o in @("research", "analysis", "docs", "Claude", "blind", "온라인대회자료")) {
        $p = Join-Path $Root $o
        if (-not (Test-Path $p)) { continue }
        $left = @(Get-ChildItem $p -Recurse -File -Force -ErrorAction SilentlyContinue |
                  Where-Object { $_.FullName -notmatch '\\__pycache__\\' })
        if ($left.Count -eq 0) { Remove-Item $p -Recurse -Force; continue }
        Write-Warning "$($left.Count) file(s) left in old folder '$o' (same name already exists at the new path). Ask the AI: '옛 폴더 $o 에 남은 파일 확인해줘'."
        $left | Select-Object -First 10 | ForEach-Object { Write-Host "    $($_.FullName.Substring($Root.Length + 1))" }
    }
}

function Set-LabDataLink([string]$Root) {
    # 연구실\클로드\code\common.py looks for data in 연구실\클로드\정형데이터.  A directory
    # junction to 공용\대회자료 lets those scripts run without a second copy of the CSVs.
    $link = Join-Path $Root "연구실\클로드\정형데이터"
    $target = Join-Path $Root "공용\대회자료\정형데이터\참가자_배포"
    if ((Test-Path $target) -and -not (Test-Path $link)) {
        cmd /c mklink /J "$link" "$target" | Out-Null
        Write-Host "  linked  연구실\클로드\정형데이터  ->  공용\대회자료\정형데이터\참가자_배포"
    }
}

function Get-ClaudeMemoryDir([string]$Root) {
    # Claude Code keeps per-project memory in ~/.claude/projects/<slug>/memory,
    # slug = project path with every non [A-Za-z0-9] character replaced by '-'.
    $slug = ($Root -replace '[^A-Za-z0-9]', '-')
    $slug = $slug.Substring(0, 1).ToLower() + $slug.Substring(1)
    Join-Path $env:USERPROFILE ".claude\projects\$slug\memory"
}
