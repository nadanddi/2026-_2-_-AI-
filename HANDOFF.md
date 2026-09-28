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
- Codex EC: 실내 습도 급상승으로 관수를 간접 탐지하는 사전 가설도 기각(공개 EC 급락 89건 중 대리 신호 P1과 겹침 1건). 결과 `analysis/codex_independent/ec_irrigation_proxy/결과보고서.md`.
- Codex EC: 공식 집계 점수의 제곱오차 항등식으로 기존 채점 EC 예측 1·3·5회차의 0.15/0.60/0.25 혼합을 계산하면 RMSE 0.202373~0.202478(반올림 범위). 점수 기반 선택이므로 공개 사전 검증 통과 모델과 구분하고 제출 파일은 만들지 않음. 기존 점수만으로 미제출 EC v2는 0.17814~0.23158의 넓은 수학적 범위여서 판정 불가. 상세 `analysis/codex_independent/ec_leaderboard_blend/결과보고서.md`, `ec_leaderboard_bounds/결과보고서.md`.
- Codex EC: 자정 저온일(≤10℃) 학습 가중치 2배를 EC v2 ExtraTrees에 시험. 탐색 평균 −1.12%지만 12칸 중 1칸 악화, F13 이득 −0.22%뿐이라 사전 조건 미달. 확인 폴드 미사용, 상세 `analysis/codex_independent/ec_cold_weight_et/결과보고서.md`.
- Codex EC: 동일 외부 날씨 날짜군의 F13/F47 OOF 일잔차는 함께 움직이지 않음(상관 0.0045, 44셀 중 후반 2셀). 공유 날짜 라벨을 이용한 보정 가설 기각. `analysis/codex_independent/ec_shared_weather_residual/결과보고서.md`.
- Codex EC: 공개 후반 OOF 36일의 정확한 RMSE 0.38226 중 하루 수준 0.37593, 시간 안 모양 0.06923(제곱오차의 96.7%가 수준). 평가 입력은 밀폐 20/60·저온 27/60으로 후반 OOF 8/36·10/36보다 많다. 첫 진단의 확인 분할 누락 수치 0.099는 무효로 정정. `analysis/codex_independent/ec_validation_gap/결과보고서.md`.
- **2026-09-29 새 사용자 기준:** EC v2를 기준으로 밀폐/비밀폐·저온·출처·구간 오차를 먼저 분해한다. 이후 가설 최대 10개를 실행 전 프로토콜 커밋하고 A/B/DIAG10×2시드, EXT10·EXT12, 새 시드·잠금 폴드, 인과·결정성 검사를 모두 통과해야 후보로 채택한다. 공식 점수 혼합(위 기록)은 새 리더보드 역탐색 금지에 따라 후보가 아니며, 저온 가중치(위 기록)는 새 test_X 통계 학습 금지 기준으로 재사용하지 않는다. `analysis/codex_independent/CAMPAIGN.md`의 새 원장 참고.
- Codex D1 오차 지도: v2 공개 OOF 234일 RMSE 0.213819, 제곱오차 87.6%가 하루 수준. 밀폐 39일이 오차 58.9%, 실제 고EC(일평균≥1.2) 19일이 74.2%; 자정 저온 46일은 21.8%. 다음 D2는 고EC 날의 예측 순위 능력과 수준 과소예측을 구분한다. `analysis/codex_independent/ec_error_atlas/결과보고서.md`.
- Codex D2: v2의 공개 고EC일 순위 AUC 0.9745(0시 예측만 0.9638), 양 농장도 0.95 이상. 실제 고EC 19일의 평균은 1.593, v2는 1.143으로 낮다. 다만 높은 예측 구간 전체의 평균 잔차는 +0.075~+0.116 정도여서 완만한 현재행 예측 보정만 H1로 시험한다. `analysis/codex_independent/ec_high_day_separability/결과보고서.md`.
- Codex 새 최종 확인 40일은 입력 행 ID의 고정 해시만으로 선택해 `ec_final_lock/locked_days.json`에 잠갔다. 이번 H1에서는 이 날짜의 정답·예측을 보지 않았다.
- Codex H1 고정 수준 보정은 공개 OOF 전체 −1.05%였으나 6폴드 중 3개 악화, 하루 재표집 MSE 차이 95% 상한 +0.000714로 사전 선별 실패. 기각하고 계수 재탐색 없이 새 기작 H2로 넘어간다. `analysis/codex_independent/ec_high_level_calibration/결과보고서.md`.
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
