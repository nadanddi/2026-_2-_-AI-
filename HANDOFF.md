# HANDOFF — 지금 상태 (세션마다 갱신)

**마지막 갱신:** 2026-09-29, Codex 작업 공간(EC)

## 1. 대회 현황
- 마지막 유효 제출: 5회차 — 온도 0.5456 / EC 0.2134. 목표별 최고: 온도 5회차, EC 3회차 0.2055.
- 남은 기회: 사용자 개인 1회 (공유 7회는 마지막용으로 보류). 마감 2026-10-16 18:00 KST.
- 타 팀 참고 점수: 온도 0.49, EC 0.1279.
- 새 제출 파일은 만들지 않음 (사용자 지시 대기).

## 2. 역할
- **Claude = 온도**, **Codex = EC** (2026-09-28 사용자 결정).
- Codex 요청서: `research/Codex_요청_EC전담_2026-09-28.md`. Codex 보고서: Codex worktree `analysis/codex_independent/종합_진행보고서_2026-09-28.md`.

## 3. 현재 후보 (확정 아님)
- **온도 G_C2** (카탈로그 6.62·6.67·6.69·6.70·6.74·6.75):
  g = clip((현재 in_temp − 8)/2, 0, 1)
  예측 = (0.6 − 0.2(1−g))·MASK 기준 + (0.2 + 0.4(1−g))·Codex 멤버 + 0.2g·TabPFN(Codex 특징, 문맥 2000행 × 8표본 평균)
  → TabPFN 없는 모델 대비 DIAG10 −3.0~−3.3%, EXT8/10/12 −2~−3.7%. 평가 구간(2차) 기대 약 −2%.
- **EC v2**: 0.8·3회차(`day` 포함) + 0.2·TabPFN(4표본). Codex가 독립 확인(잠근 폴드 −3.22%).
- EC 조합안(0.6·3회차 + 0.4·TabPFN[당일 누적 실내 평균, 8표본])은 규칙 통과했지만 EXT12 +1.3~+1.8% 악화 → Codex 판단 대기 (6.77).

## 4. 진행 중
- Codex EC: 백그라운드 실험 없음. EC v2의 출처×계절 잔차 가설을 사전 기준으로 검사했으나 순열 비율 0.423으로 기각. 별도 모델은 만들지 않음.
- Codex 결과·재현 코드는 `codex-ec` 브랜치의 `analysis/codex_independent/`에 보존. EC 제출 판단 정정은 `EC_제출판단_정정_2026-09-29.md` 참고.

## 5. 다음 할 일 후보
- 온도: 조정은 과적합 위험으로 일단 중단. 남은 불확실성은 평가의 ≤ 6℃ 행 9.2%(로컬 검증 불가).
- EC: 개인 제출 기회 1회와 공용 마지막용 7회를 구분한다. EC v2는 공개 잠금 확인에서 −3.22%였지만 평가 날짜 범위 10일에서 효과가 작고 개선이 일부 날짜에 집중됐다. 조합안은 원본 사전 검증 통과에도 EXT12·전이 진단이 악화돼 보류. 공개 데이터의 잔여 오차를 설명할 수 있는 가설만 사전 규칙으로 추가 검증한다.
- 제출 결정 시: 온도 G_C2 + EC 후보로 새 생성기(v8) → 규정 검사 → 재현 ZIP → 설명자료 (CSV·ZIP·설명자료 3종).
- 상세 기록: `research/데이터_단서_카탈로그.md` 6.41~6.78, `research/야간작업_보고_2026-09-28.md`.

## 6. 동기화 방법 (컴퓨터 옮길 때)
- 새 컴퓨터 처음 설정: `docs/연구실PC_설정_안내.md` (Google Drive `farmai_sync` 폴더에도 사본)
- **작업 시작:** `powershell -ExecutionPolicy Bypass -File tools\sync_start.ps1`
- **작업 끝:** `powershell -ExecutionPolicy Bypass -File tools\sync_end.ps1 -Message "한 일 요약"`
- git: 코드·문서·카탈로그·대회 정형데이터·AI 메모리(`docs/ai-memory`). 비공개 저장소 nadanddi/2026-_2-_-AI-, 브랜치 `feature/rmse-improvement` (Codex는 `codex-ec`).
- Google Drive: 내 드라이브 최상위에 `farmai_sync` 폴더 → `research_local`, `analysis_local`, `blind` (무거운 산출물). Google Drive for desktop 설치 필요.
- Python 환경: 컴퓨터마다 `tools\setup_env.ps1` 1회 (`-Gpu` 옵션은 NVIDIA GPU가 있을 때).
