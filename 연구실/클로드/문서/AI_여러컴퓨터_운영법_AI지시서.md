# AI 지시서 — 여러 컴퓨터 × 여러 AI 운영 구조 만들기

> **이 파일을 받은 AI에게.** 사용자는 한 프로젝트를 여러 장소(예: 집·연구실)의 여러 AI(예: Claude·Codex)로 번갈아 작업하려고 합니다. 아래 지시대로 저장소에 운영 구조(폴더·문서·동기화 스크립트)를 만들고, 이후 모든 세션에서 이 운영 규칙을 따르세요.
> 사람용 설명은 `AI_여러컴퓨터_운영법_설명서.md`에 있습니다. 이 방식은 팜모니 팀이 2026-09-29부터 실제로 쓰고 있는 구조를 일반화한 것입니다.
> 사용자에게는 **한국어로** 답합니다.

---

## 0. 먼저 사용자에게 확인할 것 (모르면 묻고, 답이 오기 전에는 파일을 만들지 않음)

| 항목 | 기본값 | 쓰이는 곳 |
|---|---|---|
| 프로젝트 이름(영문, 공백 없음) | — | 저장소 경로 `C:\work\<이름>`, Drive 폴더 `<이름>_sync` |
| 장소 목록 | 집, 연구실 | 폴더 1단계 |
| 장소별 영문 키 | home, lab | Drive 폴더 이름 |
| AI 목록 | 클로드, 코덱스 | 폴더 2단계 |
| AI별 영문 키 | claude, codex | Drive 폴더 이름 |
| 장소 판별법 | 각 PC의 사용자 폴더 이름(`C:\Users\<이름>`) 또는 컴퓨터 이름(`$env:COMPUTERNAME`) | `00_먼저_읽기.md` 1절 |
| AI별 담당(있으면) | 없음 | `CLAUDE.md`/`AGENTS.md` |
| GitHub 저장소 주소 | — | 설정 안내 |

저장소는 이미 `C:\work\<이름>`에 clone되어 있고 그 폴더가 작업 폴더라고 가정합니다. 아니라면 사용자에게 먼저 그렇게 하라고 안내합니다(경로가 PC마다 같아야 Claude 메모리가 맞게 동기화됨, OneDrive 밖이어야 함).

## 1. 만들 것 전체 목록

```
00_먼저_읽기.md
CLAUDE.md
AGENTS.md                 (CLAUDE.md와 내용 동일, 제목의 "Claude용"만 "Codex용")
README.md                 (없으면 생성, 있으면 "어디부터 보나" 표만 추가)
.gitignore                (아래 줄 추가)
tools/sync_common.ps1     (UTF-8 BOM 필수)
tools/sync_start.ps1      (UTF-8 BOM 필수)
tools/sync_end.ps1        (UTF-8 BOM 필수)
공용/HANDOFF.md
공용/확인기록.md
공용/발견_카탈로그.md
공용/컴퓨터_설정_안내.md
공용/ai-memory/MEMORY.md
<장소>/<AI>/작업일지/.gitkeep    (장소 × AI 모든 조합)
```

`<장소>/<AI>/local/`은 만들 필요 없음(스크립트가 필요 시 생성, git 제외).

## 2. `.gitignore`에 추가

```gitignore
# 무거운 산출물: git이 아니라 Google Drive(<이름>_sync)로 동기화 (tools/sync_*.ps1)
/*/*/local/
__pycache__/
.venv/
```

## 3. 동기화 스크립트

**중요: 세 `.ps1` 파일은 반드시 "UTF-8 with BOM"으로 저장합니다.** Windows PowerShell 5.1은 BOM 없는 스크립트를 ANSI로 읽어 한글 폴더 이름이 깨집니다. 저장 후 아래로 확인하고, 첫 3바이트가 `239 187 191`이 아니면 BOM을 붙여 다시 저장합니다.

```powershell
foreach ($f in "sync_common","sync_start","sync_end") { $b = [IO.File]::ReadAllBytes("tools\$f.ps1")[0..2]; "$f : $b" }
```

BOM 붙이기(필요할 때):
```powershell
foreach ($f in "sync_common","sync_start","sync_end") { $p = (Resolve-Path "tools\$f.ps1").Path; $t = [IO.File]::ReadAllText($p, [Text.Encoding]::UTF8); [IO.File]::WriteAllText($p, $t, (New-Object Text.UTF8Encoding $true)) }
```

`<이름>`, 장소·AI 목록과 키는 0절에서 받은 값으로 바꿉니다.

### 3-1. `tools/sync_common.ps1`

```powershell
# Shared helpers for sync_start.ps1 / sync_end.ps1 (dot-sourced).
# Saved as UTF-8 *with BOM* so Windows PowerShell 5.1 reads Korean folder names correctly.

# ---- project settings -------------------------------------------------------
$Places   = @("집", "연구실")
$Ais      = @("클로드", "코덱스")
$PlaceKey = @{ "집" = "home"; "연구실" = "lab" }
$AiKey    = @{ "클로드" = "claude"; "코덱스" = "codex" }
$SyncFolderName = "<이름>_sync"
# -----------------------------------------------------------------------------

function Find-SyncDrive([string]$Hint) {
    # Looks for <SyncFolderName> at the root or one level below the root of every
    # mounted drive (Google Drive for desktop: G:\내 드라이브\<SyncFolderName>).
    if ($Hint -and (Test-Path $Hint)) { return $Hint }
    foreach ($d in Get-PSDrive -PSProvider FileSystem) {
        $direct = Join-Path $d.Root $SyncFolderName
        if (Test-Path $direct) { return $direct }
        $hit = Get-ChildItem -Path $d.Root -Directory -ErrorAction SilentlyContinue |
               ForEach-Object { Join-Path $_.FullName $SyncFolderName } |
               Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($hit) { return $hit }
    }
    return $null
}

function Get-ArtifactPairs([string]$Root, [string]$DriveDir) {
    # Every <place>\<ai>\local  <->  <Drive>\<placeKey>_<aiKey>_local
    $pairs = @()
    foreach ($p in $Places) { foreach ($a in $Ais) {
        $pairs += @{ Local  = (Join-Path $Root "$p\$a\local");
                     Remote = (Join-Path $DriveDir "$($PlaceKey[$p])_$($AiKey[$a])_local") }
    } }
    return $pairs
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
    # slug = project path with every non [A-Za-z0-9] character replaced by '-'
    # (C:\work\abc -> C--work-abc).  This is why the repo path must be the same on every PC.
    $slug = ($Root -replace '[^A-Za-z0-9]', '-')
    Join-Path $env:USERPROFILE ".claude\projects\$slug\memory"
}
```

### 3-2. `tools/sync_start.ps1`

```powershell
# Start-of-session sync (run BEFORE working, on any machine).
#   1) git pull main   2) Google Drive -> local outputs   3) 공용\ai-memory -> Claude memory
# Usage: powershell -ExecutionPolicy Bypass -File <repo>\tools\sync_start.ps1 [-Drive "G:\내 드라이브\<이름>_sync"]
# Saved as UTF-8 with BOM - keep the BOM when editing.
param([string]$Drive = $env:PROJECT_SYNC_DRIVE)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "sync_common.ps1")

Write-Host "== [1/3] git (main) ==" -ForegroundColor Cyan
$current = git -C $Root branch --show-current
if ($current -ne "main") { throw "This machine is on branch '$current'. Only 'main' is used: commit your work, then 'git switch main'." }
git -C $Root pull --rebase --autostash
if ($LASTEXITCODE -ne 0) { throw "git pull failed - resolve the conflict (git status), then run this script again." }
$global:LASTEXITCODE = 0

Write-Host "== [2/3] Google Drive -> local outputs ==" -ForegroundColor Cyan
$DriveDir = Find-SyncDrive $Drive
if ($DriveDir) {
    foreach ($pair in (Get-ArtifactPairs $Root $DriveDir)) { Copy-Newer $pair.Remote $pair.Local }
} else {
    Write-Warning "Google Drive folder '$SyncFolderName' not found - heavy outputs NOT pulled (see 공용\컴퓨터_설정_안내.md)."
}

Write-Host "== [3/3] AI memory -> Claude ==" -ForegroundColor Cyan
Copy-Newer (Join-Path $Root "공용\ai-memory") (Get-ClaudeMemoryDir $Root)

Write-Host "Done. Tell the AI: '너는 <장소> <AI>야. 00_먼저_읽기.md 읽고 시작해'." -ForegroundColor Green
```

### 3-3. `tools/sync_end.ps1`

```powershell
# End-of-session sync (run AFTER working, on any machine).
#   1) Claude memory -> 공용\ai-memory   2) local outputs -> Google Drive   3) git commit + push main
# Usage: powershell -ExecutionPolicy Bypass -File <repo>\tools\sync_end.ps1 [-Message "what I did"]
# Saved as UTF-8 with BOM - keep the BOM when editing.
param([string]$Drive = $env:PROJECT_SYNC_DRIVE, [string]$Message = "")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "sync_common.ps1")

$current = git -C $Root branch --show-current
if ($current -ne "main") { throw "This machine is on branch '$current', not main." }

Write-Host "== [1/3] Claude memory -> 공용\ai-memory ==" -ForegroundColor Cyan
$mem = Get-ClaudeMemoryDir $Root
if (Test-Path $mem) { Copy-Newer $mem (Join-Path $Root "공용\ai-memory") }

Write-Host "== [2/3] local outputs -> Google Drive ==" -ForegroundColor Cyan
$DriveDir = Find-SyncDrive $Drive
if ($DriveDir) {
    foreach ($pair in (Get-ArtifactPairs $Root $DriveDir)) { Copy-Newer $pair.Local $pair.Remote }
} else {
    Write-Warning "Google Drive folder '$SyncFolderName' not found - heavy outputs NOT pushed."
}

Write-Host "== [3/3] git commit + push ==" -ForegroundColor Cyan
git -C $Root add -A
if (git -C $Root status --porcelain) {
    if (-not $Message) { $Message = "sync: $env:COMPUTERNAME $(Get-Date -Format 'yyyy-MM-dd HH:mm')" }
    git -C $Root commit -m $Message
}
git -C $Root pull --rebase --autostash
if ($LASTEXITCODE -ne 0) { throw "git pull --rebase failed - resolve the conflict (git status), then run this script again." }
git -C $Root push origin main
if ($LASTEXITCODE -ne 0) { throw "git push failed." }
Write-Host "Done. Check that Google Drive finished uploading before shutting down." -ForegroundColor Green
```

## 4. 문서 틀

`<…>`는 0절 값으로 채웁니다. 장소·AI가 다르면 표와 폴더 이름을 그에 맞게 바꿉니다.

### 4-1. `00_먼저_읽기.md`

````markdown
# 00 — 먼저 읽기 (모든 AI 공통, 세션 시작 시 가장 먼저)

이 저장소는 **여러 장소 × 여러 AI**가 번갈아 작업합니다. 이 파일은 "나는 누구이고, 내가 없는 동안 무슨 일이 있었는지"를 빠르게 파악하기 위한 안내입니다.

|  | 클로드 | 코덱스 |
|---|---|---|
| **집** | `집/클로드/` | `집/코덱스/` |
| **연구실** | `연구실/클로드/` | `연구실/코덱스/` |

## 1. 나는 누구인가
| 확인할 것 | 방법 |
|---|---|
| 장소 | <장소 판별법. 예: 사용자 폴더가 C:\Users\AAA면 집, C:\Users\BBB면 연구실>. 모르겠으면 사용자에게 묻는다. |
| AI | 클로드는 `CLAUDE.md`, 코덱스는 `AGENTS.md`를 자동으로 읽는다. |

**내 폴더 = `<장소>/<AI>/`.** 나머지 폴더는 다른 장소·다른 AI의 작업이며 읽기만 한다.

## 2. 세션 시작 순서
1. 사용자가 아직 안 했다면 `tools\sync_start.ps1` 실행을 권한다.
2. 이 파일 → `공용/HANDOFF.md` → `공용/ai-memory/MEMORY.md`(규칙 목록, 반드시 지킴) 순으로 읽는다.
3. 다른 곳 작업 검토(3절). 사용자가 시키면 반드시, 안 시켜도 새 작업일지가 있으면 목록은 먼저 알린다.
4. 필요할 때 `공용/발견_카탈로그.md`를 본다.

## 3. 다른 곳·다른 AI 작업 검토하는 법
1. `공용/확인기록.md`에서 내 줄의 "마지막 검토일"을 찾는다.
2. 다른 폴더들의 `작업일지/`에서 그 날짜 이후(같은 날 포함) 파일을 모두 읽는다.
   일지가 빠졌을 수 있으니 git 기록으로 교차 확인한다: `git log --since=<마지막 검토일> --stat`
3. 일지가 가리키는 결과 파일과 카탈로그 새 번호를 확인한다. 볼 것: 주장한 수치가 로그와 맞는가 / 규칙을 지켰는가 / 내 작업에 영향이 있는가.
4. 사용자에게 **요약 → 문제점·의문 → 내 작업에 반영할 것** 순서로 보고한다.
5. `공용/확인기록.md`의 내 줄만 오늘 날짜로 갱신한다.

## 4. 반드시 지킬 것
- 내 파일은 내 폴더에만 만든다. 공용 문서(HANDOFF·카탈로그·확인기록)는 규칙대로 추가·갱신만 한다.
- 이미 만들었거나 다른 곳에서 쓴 파일은 고치지 않고 새 이름(`_v2`)으로 만든다.
- 브랜치는 `main` 하나. 새 브랜치를 만들지 않는다.
- 같은 PC에서 다른 AI가 동시에 작업 중일 수 있다. AI가 직접 커밋할 때는 `git add -A`를 쓰지 않고 내가 고친 파일만 경로로 `git add` 한다. 모르는 변경은 건드리지 않고 사용자에게 알린다.
- 무거운 결과물(모델, 대용량 예측값 등)은 내 폴더의 `local/`에 둔다(git 제외, Drive로 동기화).
- 파일 삭제는 사용자에게 묻고 한다.

## 5. 작업일지
- 위치: `<장소>/<AI>/작업일지/YYYY-MM-DD.md`, 하루 한 파일. 같은 날 또 하면 `## 세션 2`로 이어 쓴다.
- **세션이 끝나기 전에 반드시 쓴다.** 다른 곳의 AI는 이 폴더만 보고 따라잡는다.
- 기각·실패도 적는다. 수치에는 근거 파일(로그·결과)을 붙인다.

```markdown
# YYYY-MM-DD · <장소> · <AI>

## 한 줄
(오늘 무엇이 달라졌는지 한 문장)

## 한 것
- 실험·변경 (스크립트 이름)

## 결과
| 실험 | 수치 | 판정(채택/기각/보류) | 근거 파일 |
|---|---|---|---|

## 다른 곳에 알릴 것
- (다른 장소·다른 AI가 알아야 할 것. 없으면 "없음")

## 다음
- 이어서 할 일

## 파일
- 추가한 파일
```

## 6. 세션 끝날 때
1. 내 작업일지를 쓴다.
2. `공용/HANDOFF.md` 맨 위에 `> **YYYY-MM-DD HH:MM <장소> <AI>:** 요약` 단락을 추가하고 "현재 상태 / 진행 중 / 다음 할 일"을 갱신한다.
3. 새로 확인하거나 기각한 것은 `공용/발견_카탈로그.md`에 **새 번호로 추가**한다. 기존 항목은 고치지 않고, 정정도 새 항목으로 쓴다.
4. 사용자에게 `tools\sync_end.ps1 -Message "한 일"` 실행을 권한다.

## 7. 폴더 지도
(1절 표의 폴더 + 공용/ + tools/ 설명. 각 폴더에 무엇이 있는지 한 줄씩)
````

### 4-2. `CLAUDE.md` (그리고 같은 내용으로 `AGENTS.md`)

```markdown
# <프로젝트> — AI 작업 안내 (Claude용)

이 저장소는 여러 장소 × 여러 AI가 번갈아 작업합니다. 어느 컴퓨터든 같은 절차를 따릅니다. (Codex용 `AGENTS.md`와 내용 동일)

## 세션 시작할 때
1. 사용자가 아직 안 했다면 `tools\sync_start.ps1` 실행을 권합니다.
2. **`00_먼저_읽기.md`를 가장 먼저 읽습니다.** 거기서 내 장소와 내 폴더를 정합니다.
3. 이어서 `공용/HANDOFF.md` → `공용/ai-memory/MEMORY.md` → `공용/확인기록.md`와 그 이후 다른 폴더의 `작업일지/`를 읽습니다.

## 반드시 지킬 것 (요약)
- 답변은 한국어.
- 내 파일은 내 폴더(`<장소>/<AI>/`)에만. 다른 폴더는 읽기만.
- 이미 만든·쓴 파일은 고치지 말고 새 이름으로.
- 브랜치는 main 하나.
- (역할 분담이 있으면 여기에: 예 "Claude = ○○ 담당, Codex = △△ 담당")
- (프로젝트 고유 규칙이 생기면 여기에 한 줄 요약 + 공용/ai-memory 파일로)

## 세션 끝날 때
1. 내 작업일지 `<장소>/<AI>/작업일지/YYYY-MM-DD.md` 작성.
2. `공용/HANDOFF.md` 갱신(날짜·장소·AI 포함).
3. 새 발견·기각은 `공용/발견_카탈로그.md`에 새 번호로.
4. 사용자에게 `tools\sync_end.ps1` 실행을 권합니다.
```

`AGENTS.md`에는 다음 한 줄을 추가합니다(Codex는 Claude 자동 메모리를 쓰지 않으므로):
> 사용자 규칙은 `공용/ai-memory/MEMORY.md` 목록에 있습니다. 규칙과 관련된 작업을 할 때는 해당 파일을 열어 읽습니다. Codex가 새 규칙을 들으면 `공용/ai-memory/`에 같은 형식의 파일을 추가하고 MEMORY.md에 한 줄을 넣습니다.

### 4-3. `공용/HANDOFF.md`

```markdown
# HANDOFF — 지금 상태 (세션마다 갱신)

> **YYYY-MM-DD HH:MM <장소> <AI>:** (최근 세션 요약 한 단락. 새 단락은 맨 위에 추가)

## 현재 상태
- 현재 최선안 / 주요 수치:
- 진행 중:

## 다음 할 일
- [ ] …

## 각 폴더 최근 작업일지
| 폴더 | 최근 일지 |
|---|---|
```

맨 위 단락이 10개를 넘으면 오래된 것을 `공용/HANDOFF_보관_YYYY-MM.md`(새 파일)로 옮겨 HANDOFF를 짧게 유지합니다.

### 4-4. `공용/확인기록.md`

```markdown
# 확인기록 — 다른 곳·다른 AI 작업을 어디까지 검토했나

각 장소×AI는 다른 폴더의 작업일지를 검토한 뒤 **자기 줄만** 갱신합니다(방법은 `00_먼저_읽기.md` 3절).

| 검토자 | 마지막 검토일 | 무엇까지 봤나 / 메모 |
|---|---|---|
| 집 · 클로드 | (아직 없음) | |
| 집 · 코덱스 | (아직 없음) | |
| 연구실 · 클로드 | (아직 없음) | |
| 연구실 · 코덱스 | (아직 없음) | |
```

### 4-5. `공용/발견_카탈로그.md`

```markdown
# 발견 카탈로그 — 확인된 사실·기각된 방법 (추가만, 수정 금지)

- 번호는 이어서 붙입니다. 다른 문서는 번호로 인용합니다.
- 틀린 항목은 고치지 말고 "N번 정정"이라는 새 항목을 씁니다.

| 번호 | 날짜 · 장소 · AI | 내용 | 판정(확인/기각/보류) | 근거 파일 |
|---|---|---|---|---|
| 1 | | | | |
```

### 4-6. `공용/ai-memory/MEMORY.md`

```markdown
- [Korean only](korean-only.md) — every reply to this user must be in Korean
```

그리고 `공용/ai-memory/korean-only.md`:

```markdown
---
name: korean-only
description: User requires all replies in Korean
metadata:
  type: user
---

Always reply to this user in Korean. Code identifiers, file names and numbers may stay as they are.
```

(MEMORY.md는 목록만, 규칙 하나당 파일 하나. Claude가 메모리에 저장한 규칙은 `sync_end`가 여기로 복사합니다.)

### 4-7. `공용/컴퓨터_설정_안내.md`

사람용 설명서(`AI_여러컴퓨터_운영법_설명서.md`)의 5-2절(각 컴퓨터 설정)과 6절(매일 루틴), 8절(문제 해결)을 이 프로젝트의 실제 이름·경로·저장소 주소로 채워 넣습니다. 프로젝트에 Python 등 실행 환경이 필요하면 설치 방법도 여기에 적습니다.

## 5. 만든 뒤 확인 (사용자에게 결과 보고)

1. `.ps1` 세 개의 BOM 확인(3절 명령). 
2. 메모리 경로 확인: 아래 출력 폴더가 실제 Claude 메모리 폴더와 같은지(Claude 앱에서 한 번이라도 이 저장소를 열었다면 `~\.claude\projects\` 아래에 같은 이름 폴더가 있음).
   ```powershell
   . .\tools\sync_common.ps1; Get-ClaudeMemoryDir (Get-Location).Path
   ```
3. Drive 폴더 찾기 확인: `. .\tools\sync_common.ps1; Find-SyncDrive ""` → 경로가 나와야 함. `$null`이면 사용자에게 Drive 로그인·폴더 위치를 확인하게 함.
4. `git status`로 만든 파일 목록을 보여 주고, 사용자에게 `tools\sync_end.ps1 -Message "운영 구조 생성"` 실행을 권함(AI가 임의로 push하지 않음).
5. 다른 컴퓨터에서는 clone(같은 경로) → `sync_start` 만 하면 된다고 안내.

## 6. 이후 모든 세션에서 AI가 지킬 운영 규칙 (요약)

1. 시작: `00_먼저_읽기.md` → HANDOFF → MEMORY → 확인기록 → 그 이후 다른 폴더 작업일지.
2. 작업: 내 폴더에만 쓴다. 남의 파일은 읽기만. 이미 쓴 파일은 새 이름으로.
3. 끝: 작업일지 → HANDOFF → 카탈로그(새 번호) → 사용자에게 `sync_end` 권유.
4. 커밋을 직접 할 때는 내 파일만 `git add <경로>`. `git add -A` 금지.
5. 동기화 스크립트는 복사만 하고 지우지 않는다. 지울 것은 사용자에게 묻는다.
6. 기록의 수치에는 근거 파일을 붙이고, 일반 지식과 실제로 읽은 자료를 구분해 적는다.
