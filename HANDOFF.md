# HANDOFF — 지금 상태 (세션마다 갱신)

## 2026-09-29 Codex EC 새 정보원 캠페인 진행

사용자의 새 목표에 따라 H11~H22(최대 12개, F1/F2/F3 각각 두 구현 이상)를 시작했다. `codex-ec` 작업 사본에서 매 실험 프로토콜을 실행 전 커밋한다. 현재 F1 H11·H12·H17, F2 H13·H14, F3 H15·H16의 서로 다른 구현 일곱 개를 마쳤고 모두 선별 기각이다. H11 고정 d−2 연결은 대리 연결 89/170일(52.4%)이며 양 시드·6폴드 모두 악화했다. H12 입력 인과 연결은 135/170일(79.4%)로 개선했지만 전체 −1.09/−1.17%에 비해 폴드·농장 방향이 갈리고 `p_worse=.2384/.2158`; 채택 불가다. H13 운영 지문+KMeans와 H14 출처별 수준 Ridge도 전칸 탈락했다. H15 고EC 전문가와 H16 하루 수준+모양은 각 +0.6%가량 악화했다. H17 긴 사슬 6열은 전체 −1.78/−1.62%지만 3/6폴드만 개선, p_worse .049/.063; 4단계 대리 연결은 비교 가능한 60일 중 38일, 과거 4일 연속 밀폐는 4일만 있었다. 상세 `analysis/codex_independent/CAMPAIGN.md`, H11~H17 각 `결과보고서.md`. 최종 잠금 40일은 아직 채점하지 않았다. 다음은 F1/F2/F3의 남은 구현을 새 물리·데이터 구조 증거에 맞춰 사전 등록해 H18~H22를 진행한다. 온도·제출 파일은 건드리지 않았다. 이 단락이 아래의 **이전 H1~H10 종료 상태를 갱신**한다.

**마지막 갱신:** 2026-09-29, Codex 작업 공간(EC)

## 1. 대회 현황
- 마지막 유효 제출: 5회차 — 온도 0.5456 / EC 0.2134. 목표별 최고: 온도 5회차, EC 3회차 0.2055.
- 남은 기회: 사용자 개인 1회 (공유 7회는 마지막용으로 보류). 마감 2026-10-16 18:00 KST.
- 타 팀 참고 점수: 온도 0.49, EC 과거 참고 0.1279·사용자가 새 목표에서 제시한 0.1245.
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
- EC 조합안(0.6·3회차 + 0.4·TabPFN[당일 누적 실내 평균, 8표본])은 당시 공개 A/B/DIAG10 규칙은 통과했지만 EXT12 +1.3~+1.8% 악화 → 현 사용자 외삽 무악화 기준에서는 채택하지 않음 (6.77).

## 4. Codex EC 캠페인 종료 및 과거 기록
- **2026-09-29 Codex EC 결론:** 사전 등록 H1~H10을 모두 시험했고 전칸·복수 시드·외삽 기준을 통과한 v2 대비 신규 모델은 없었다. 오차는 공개 OOF 하루 수준 87.6%, 실제 고EC 19/234일에 74.2%, 밀폐 39/234일에 58.9% 집중한다. 새 최종 잠금 40일은 채점하지 않았고 제출 파일은 만들지 않았다. 자세한 확인/기각/불확실·경쟁팀 0.1245 격차 해석은 `analysis/codex_independent/EC_최종연구보고서_2026-09-29.md`.
- 이번 EC 캠페인은 사용자 지정 종료 조건 **사전 등록 10개 가설 소진**으로 마친다. EC v2는 기존 연구 후보로만 유지한다. 새 독립 날짜나 합법적 관리 입력이 확보되기 전에는 H4·H6의 탐색 평균이나 기각된 규칙을 제출 후보로 바꾸지 않는다.
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
- Codex D3: 현재행 팬<10·환기=0 및 v2≥0.6인 965행(56일)이 공개 OOF 제곱오차 68.2%, RMSE 0.4260. 그러나 평균 잔차 +0.075는 비 gate 고예측 행 +0.078과 유사하고 폴드 0·4 음수/8·9 큰 양수로 부호가 뒤집힌다. 사전 H2 진행 기준은 통과하되, 단순 양의 보정은 위험하다. `analysis/codex_independent/ec_causal_seal_diagnostic/결과보고서.md`.
- Codex H2: 0시 입력만으로 하루 평균 EC를 예측하는 Ridge를 고위험 현재행에 20% 결합했으나 공개 OOF 전체 +1.09% 악화, 6폴드 중 5개 악화. 학습마다 검증일·최종 잠금 40일의 ±1일을 제외했다. H2 기각, 잠금일 점수 미열람. `analysis/codex_independent/ec_midnight_setpoint/결과보고서.md`.
- Codex D4: 고위험 행 안에서 현재 CO₂ 공급으로 잔차를 분리하려 했으나 공급 활성 다수 날이 56일 중 1일뿐이라 사전 최소 표본 조건 실패. CO₂ 기반 H3를 만들지 않음. `analysis/codex_independent/ec_co2_regime_diagnostic/결과보고서.md`.
- Codex H3: 현재·과거 입력의 고EC일 분류 확률로 고위험 행을 양방향 보정했으나 공개 OOF 전체 +0.27% 악화, 폴드 0·2·4 악화/6·8·9 개선. 폴드당 고EC 학습일 15~21일. 첫 두 실행 무결성 오류는 점수 전 중단, 새 버전 커밋 후 재실행. H3 기각, 잠금일 미열람. `analysis/codex_independent/ec_high_day_mixture/결과보고서.md`.
- Codex D5: 현재행 3회차-TabPFN 예측 차이 ≥0.15인 646행/57일이 v2 공개 OOF 제곱오차 47.9%. 해당 구간 TabPFN 원시 평균 RMSE 0.367 vs v2 0.436(양 농장·5/6폴드 우세)이나 현재 실내온도 <10℃에서는 0.324 vs v2 0.239로 역전. 사전 H4 진행 기준 통과. `analysis/codex_independent/ec_member_disagreement/결과보고서.md`.
- Codex H4: 현재행 불일치≥0.15·in_temp≥10℃에서만 TabPFN 쪽 25% 이동, 공개 OOF 전체 −3.27%, 양 농장 개선, 하루 재표집 MSE 차이 95% 상한<0. 그러나 폴드 0 +0.26% 악화(다른 5폴드는 개선)로 사용자 사전 전칸 기준 미달. H4 기각, 문턱/혼합률 조정 안 함, 잠금일 미사용. `analysis/codex_independent/ec_disagreement_gate/결과보고서.md`.
- Codex H5: TabPFN 문맥 4개 평균을 중앙값으로 바꾼 효과는 행당 평균 절댓값 0.00872 EC, 공개 OOF 전체 +0.33% 악화·6폴드 중 4개 악화. 극단 문맥 한 개가 H4 불일치 효과를 만들었다는 설명 지지 안 됨. H5 기각. `analysis/codex_independent/ec_context_median/결과보고서.md`.
- Codex H6: 현재행 두 멤버 차이와 기온·저팬·고예측·문맥 산포의 선형 신뢰도 보정. 저장 OOF 선별 전체 −6.21%지만 폴드 0 +2.49%, 하루 재표집 MSE 차이 상한 +0.000324로 기각. 다른 폴드 OOF 기초 모델이 검증 폴드 라벨을 학습한 간접 경로 때문에 전체 수치는 확증 아님. 중첩 재학습·잠금일 미실행. `analysis/codex_independent/ec_member_reliability/결과보고서.md`.
- Codex D6: 폴드 0 H4 변경 7일 중 손해 3일·이득 4일, 큰 손해 2일 몫 75%(사전 80% 미달). 따뜻하고 무환기인 변경행에서 손해·이득이 공존해 단일 현재 체제 구분 실패. 날짜 ID 회피나 H4 재조정 안 함. `analysis/codex_independent/ec_fold0_counterexamples/결과보고서.md`.
- Codex D7: 6시까지 입력으로 같은 온실·2~30일 떨어진 날을 짝지으면 EC 수준차 중앙값 0.0634(무작위 짝 0.1595)로 입력은 유익하다. 다만 가까운 방향 짝 59개 중 7개(서로 다른 4쌍)는 차이≥0.4. 일부는 v2가 이미 구별했고 숨은 관수·급액 원인은 아직 미입증. H7~H10 미실행, 잠금일 미사용. `analysis/codex_independent/ec_observational_aliasing/결과보고서.md`.
- Codex H7: 0~6시 입력 최근접 학습일 다섯 개의 EC 수준을 6시 이후 10% 결합. 전체 0.213819→0.213781(−0.018%)이나 6폴드 중 4개 악화, 하루 재표집 MSE 차이 95% 상한 +0.000333로 기각. 거리·비중 재조정 안 함. H8~H10 남음, 잠금일 미사용. `analysis/codex_independent/ec_early_analog/결과보고서.md`.
- Codex H8: 원본+당일 인과 평활 센서 복제본을 함께 학습한 ET. 시드7 전체 −0.19%지만 2/6폴드만 개선, 시드101 전체 +0.32%·0/6폴드 개선. 사전 기준 탈락. 기존 평활 대체(6.20/6.29)와 달리 원본을 남겼어도 재현 이득 없음. H9~H10 남음, 잠금일 미사용. `analysis/codex_independent/ec_dual_view_et/결과보고서.md`.
- Codex D8: 0~6시 합법 입력 14개와 v2의 6시 예측을 통제한 하루 수준 잔차 상관을 검사. 난방 평균 r=−0.1644(F13/F47 모두 음수)가 최대였으나 최대값 순열 p=0.1569>사전 0.01. 난방·CO₂ 신호 H9 보정으로 연결하지 않음. 잠금일 미사용. `analysis/codex_independent/ec_early_residual_signal/결과보고서.md`.
- Codex H9: 학습일 평균 EC≥1.2인 8~12일/폴드만 ET 가중치 2. 시드7 전체 −0.15%(2/6폴드 개선), 시드101 +0.12%(2/6폴드 개선), 날 재표집 MSE 구간 둘 다 0 포함. 가중치·문턱 재조정 안 함. H10 한 가설 남음, 최종 잠금일 미사용. `analysis/codex_independent/ec_high_label_weight_et/결과보고서.md`.
- Codex H10: 전체 학습 행의 조건부 0.65 분위수 모델을 높은 v2 예측 행에 15% 결합했으나 공개 OOF 시드7 +1.42%, 시드101 +1.34%; 양 시드·6폴드·두 농장 모두 악화. H1~H10 사전 등록 가설 모두 소진, 최종 잠금일 미사용. `analysis/codex_independent/ec_upper_quantile/결과보고서.md`.
- Codex 결과·재현 코드는 `codex-ec` 브랜치의 `analysis/codex_independent/`에 보존. EC 제출 판단 정정은 `EC_제출판단_정정_2026-09-29.md` 참고.

## 5. 다음 할 일 후보
- 온도: 조정은 과적합 위험으로 일단 중단. 남은 불확실성은 평가의 ≤ 6℃ 행 9.2%(로컬 검증 불가).
- EC: 개인 제출 기회 1회와 공용 마지막용 7회를 구분한다. 새 캠페인 H1~H10은 모두 선별 실패하여 종료했고 v2가 기존 연구 후보로 남는다. v2의 공개 잠금 확인 −3.22%는 평가 날짜 범위 10일에서 작고 일부 날짜에 집중됐다. 새 EC 제출 구성은 확정하지 않는다. 새 독립 날짜·합법 관리 입력 등이 확보되면 별도 사전 등록 캠페인으로 검토한다.
- 제출 결정 시: 온도 G_C2 + EC 후보로 새 생성기(v8) → 규정 검사 → 재현 ZIP → 설명자료 (CSV·ZIP·설명자료 3종).
- 상세 기록: `research/데이터_단서_카탈로그.md` 6.41~6.78, `research/야간작업_보고_2026-09-28.md`.

## 6. 동기화 방법 (컴퓨터 옮길 때)
- 새 컴퓨터 처음 설정: `docs/연구실PC_설정_안내.md` (Google Drive `farmai_sync` 폴더에도 사본)
- **작업 시작:** `powershell -ExecutionPolicy Bypass -File tools\sync_start.ps1`
- **작업 끝:** `powershell -ExecutionPolicy Bypass -File tools\sync_end.ps1 -Message "한 일 요약"`
- git: 코드·문서·카탈로그·대회 정형데이터·AI 메모리(`docs/ai-memory`). 비공개 저장소 nadanddi/2026-_2-_-AI-, 브랜치 `feature/rmse-improvement` (Codex는 `codex-ec`).
- Google Drive: 내 드라이브 최상위에 `farmai_sync` 폴더 → `research_local`, `analysis_local`, `blind` (무거운 산출물). Google Drive for desktop 설치 필요.
- Python 환경: 컴퓨터마다 `tools\setup_env.ps1` 1회 (`-Gpu` 옵션은 NVIDIA GPU가 있을 때).
