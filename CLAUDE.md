# 농업 AI 경진대회(정형) — AI 작업 안내 (Claude용)

이 저장소는 집 노트북(Claude, Codex)과 학교 연구실 PC에서 번갈아 작업합니다. 어느 컴퓨터든 **같은 절차**를 따릅니다. (Codex용 `AGENTS.md`와 내용 동일)

## 세션 시작할 때
1. 사용자가 아직 안 했다면 `tools/sync_start.ps1` 실행을 권합니다 (git pull + Drive 산출물 + AI 메모리 복원).
2. 이 순서로 읽습니다.
   1. `HANDOFF.md` — 지금 상태, 현재 후보, 진행 중/다음 작업 (가장 먼저)
   2. `docs/ai-memory/MEMORY.md` 와 그 안의 파일들 — 사용자 규칙·선호 (반드시 지킴)
   3. `research/데이터_단서_카탈로그.md` — 검증된 사실·기각된 방법 전체 (항목 번호로 인용)
   4. 필요 시 `research/제출기록.md`, `research/감사결과_2026-09-27.md`, `온라인대회자료/정형데이터/정형데이터_문제설명서.pdf`

## 반드시 지킬 것 (요약, 자세한 건 docs/ai-memory)
- 답변은 **한국어**.
- 역할 분담: **Claude = 온도(sub_temp) 전담, Codex = EC(sub_ec) 전담.** 상대 영역은 결과만 전달.
- 이미 만들었거나 제출에 쓴 파일은 수정하지 말고 새 이름(버전)으로 만든다.
- 채택 기준: 실행 **전에** 규칙을 코드에 고정. 모든 시드 × 모든 검증기에서 같은 방향 + DIAG10 p_worse < 0.025 (여러 안이면 본페로니). 크기 무관.
- 규정: 평가 행 특징은 같은 온실의 현재·이전 입력만. 학습 행 특징은 MASK(test_X 입력 NaN)로.
- 사용자가 "제출하자"고 하기 전에는 제출 파일/"확정 구성"을 만들지 않는다. 실험은 묻지 않고 진행.

## 실행 방법
- Python 3.12. 새 컴퓨터는 `tools/setup_env.ps1` 한 번 실행.
- `cd research` 후 `PYTHONPATH="" python -u <script>.py`. 모든 스크립트 첫 import는 `import env` (TabPFN 등은 그 다음 `import env_extra`, GPU는 `import env_extra_gpu`).
- 무거운 산출물(`research/local`, `analysis/local`, `blind` 결과)은 git이 아니라 Google Drive `farmai_sync`로 동기화됩니다.

## 세션 끝날 때 (중요 — 다음 컴퓨터로 온전히 넘기기)
1. `HANDOFF.md`의 "현재 상태 / 진행 중 / 다음 할 일"을 갱신 (날짜·컴퓨터 이름 포함).
2. 새로 확인·기각한 것은 `research/데이터_단서_카탈로그.md`에 새 번호로 추가 (기존 항목은 고치지 말고 정정은 새 항목으로).
3. 사용자에게 `tools/sync_end.ps1` 실행을 권합니다 (메모리 → docs/ai-memory, 산출물 → Drive, git commit/push).
