# 연구실 PC 설정 안내

집 노트북과 같은 상태로 연구실 PC에서 이어서 작업하기 위한 안내입니다.
**처음 1회 설정(약 30분~1시간, 대부분 다운로드 대기)** 후에는 매일 명령 두 줄이면 됩니다.

명령은 모두 **PowerShell**에 복사해 붙여 넣으면 됩니다. (시작 메뉴 → "PowerShell" 검색 → 실행)

---

## A. 처음 1회 설정

### 1. 프로그램 3개 설치
이미 설치된 것은 건너뛰어도 됩니다. 한 줄씩 실행하세요.

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
winget install --id Google.GoogleDrive -e
```

- 설치가 끝나면 **PowerShell 창을 닫았다가 새로 여세요** (새로 설치한 명령을 인식하게 하려면 필요).
- `winget`이 없다고 나오면 각 사이트에서 직접 설치: git-scm.com, python.org(3.12 버전, 설치 화면에서 "Add python.exe to PATH" 체크), google.com/drive/download

### 2. Google Drive 로그인
작업 표시줄의 Google Drive 아이콘 → **집 노트북과 같은 구글 계정**으로 로그인.
탐색기에서 `Google Drive (G:)` → `내 드라이브` → `farmai_sync` 폴더가 보이면 성공입니다. (이 안내서도 그 폴더에 있습니다.)

### 3. 저장소 받기
**OneDrive 밖**(예: `C:\work\farmai`)에 받으세요. OneDrive가 git 폴더를 동시에 동기화하면 충돌이 날 수 있습니다.

```powershell
git clone -b feature/rmse-improvement https://github.com/nadanddi/2026-_2-_-AI-.git C:\work\farmai
```

- 비공개 저장소라 처음에 **GitHub 로그인 창**이 뜹니다. 브라우저로 로그인(아이디 `nadanddi`)하면 이후에는 자동입니다.
- 커밋 작성자 설정(1회):

```powershell
cd C:\work\farmai
git config user.name "nadanddi"
git config user.email "chonadan9@gmail.com"
```

### 4. Python 환경 설치 (약 3GB 다운로드)

```powershell
powershell -ExecutionPolicy Bypass -File C:\work\farmai\tools\setup_env.ps1 -Python "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
```

- 첫 줄에 `Python 3.12.x` 가 찍혀야 합니다. 경로가 없다는 오류가 나면 `py -3.12 -c "import sys; print(sys.executable)"` 로 3.12의 실제 경로를 확인해 `-Python` 뒤에 넣으세요.
- 연구실 PC에 **NVIDIA GPU**가 있으면 끝에 ` -Gpu` 를 붙이세요 (TabPFN 실험이 훨씬 빨라짐).

### 5. 산출물·AI 메모리 받기 (첫 동기화)

```powershell
powershell -ExecutionPolicy Bypass -File C:\work\farmai\tools\sync_start.ps1
```

`Done. Read HANDOFF.md first.` 가 나오면 성공입니다.

### 6. 잘 됐는지 확인 (선택)

```powershell
cd C:\work\farmai\research
$env:PYTHONPATH=""; py -3.12 -c "import env, numpy, pandas, lightgbm, sklearn; print('OK', numpy.__version__, pandas.__version__, lightgbm.__version__)"
```

`OK 2.5.3 3.0.1 4.7.0` 비슷하게 나오면 됩니다.

---

## B. 매일 할 일 (모든 컴퓨터 공통)

| 언제 | 명령 |
|---|---|
| **작업 시작 (출근 후)** | `powershell -ExecutionPolicy Bypass -File C:\work\farmai\tools\sync_start.ps1` |
| **작업 끝 (퇴근 전)** | `powershell -ExecutionPolicy Bypass -File C:\work\farmai\tools\sync_end.ps1 -Message "오늘 한 일"` |

- 집 노트북에서는 경로만 `C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\tools\...` 로 바꾸면 됩니다.
- **퇴근 전 `sync_end`를 꼭 실행하고**, 작업 표시줄 Google Drive 아이콘에서 업로드가 끝났는지 확인한 뒤 끄세요.

## C. AI에게 시킬 때

- Claude Code / Codex를 **저장소 폴더(`C:\work\farmai`)에서** 실행하세요.
- 첫 마디는 이것이면 충분합니다:
  > **"HANDOFF.md 읽고 이어서 해"**
- Claude는 `CLAUDE.md`, Codex는 `AGENTS.md`를 자동으로 읽고, 그 안에 규칙(한국어, 파일 수정 금지, 채택 기준, 역할 분담: Claude=온도 / Codex=EC)과 종료 절차가 적혀 있습니다.
- 세션을 끝낼 때 **"HANDOFF 갱신해줘"** 라고 하면 AI가 상태를 기록하고, 이어서 `sync_end`를 실행하면 다른 컴퓨터로 넘어갑니다.

## D. 문제가 생기면

| 증상 | 해결 |
|---|---|
| `git pull` 충돌 메시지 | 두 컴퓨터에서 같은 파일을 고친 경우. AI에게 "git 충돌 해결해줘"라고 하거나 `git status`로 확인 |
| `farmai_sync not found` 경고 | Google Drive 로그인 확인. 폴더가 `내 드라이브` 바로 아래에 있어야 함 |
| 스크립트 실행이 막힘 | 명령 앞에 `powershell -ExecutionPolicy Bypass -File` 을 꼭 붙이기 |
| `No module named ...` | 4번(환경 설치)을 다시 실행 |
| TabPFN 첫 실행이 느림 | 처음에만 가중치를 인터넷에서 내려받음 (정상) |
