# 최종 산출물 안내

최종 해석: `실험결과_보고서_v2.md`. 모델10안135비교 전부 REJECT, 새 채택/제출 없음. 기존 실제 제출 W40G·EC 계절v2 유지.

- 최종 수치: `scores_v2.csv`, `decisions_v2.csv`, `segments_v2.csv`, `result_v2.json`.
- 검산: `verification_v2.json` 955체크 PASS, 원 ET 새학습과 이미 평활화된 캐시 최대차2.220446049250313e-16.
- 위험 추가 대조: `risk_simple_comparator_v2.csv`, `risk_comparator_verification_v1.json` 32체크 PASS. 사후 진단이며 높은값/낮은값 두 방향 모두 보존. 새 위험모델의 추가정보를 확인하지 못했다.
- 기준선: `W40G_scores_v2.csv`, `W40G_daily_audit.csv`. 본문의 실패16일·pooled RMSE는 BASE시드7/PFN문맥1–8 기준이며 다른시드·문맥은 CSV에 보존. 실제 제출의3 BASE시드 평균과 동일한 평가 예측을 만든 것은 아니다.
- 실행: 사전 a45ef88, 열순서정정 e9f5480, 후처리정정5e25e6f. 처음부터재현은 `run_v3.py`; 이미학습한두ET안후처리만재현은 `replay_et_v3.py`. 그뒤 `analyze_v2.py → verify_v2.py → risk_comparator.py → report_v2.py`. 재실행 시 OUT 및 보고서 출력 경로를 새 버전으로 변경한다.
- `scores_v1`/`decisions_v1`/`result_v1` 및 `corrected_et_v2`의 두 ET 교체 비교는 중복 평활화 오류가 있어 무효다. 보존은 계산 정정 이력을 위한 것이다. 최종 ET OOF는 `집/코덱스/local/priority_experiments_20261003_v1/corrected_et_v3/`.
- 큰 OOF/phase audit은 `집/코덱스/local/priority_experiments_20261003_v1/`에 있다. 새 PFN 훈련 없이 기존 멤버/inner 문맥 캐시를 재사용했다. Git 외에 Drive 동기화가 필요하다.

보고서의 새 위험분류 평가는 DIAG10 EC360일/온도400일, 시드7, 각각 문맥1–4/1–8이다. 여러 시드에서 위험분류 개선을 증명한 것은 아니다. 모델 채택실험은 모든 사전시드·문맥·검증을 시행했다. 공개기준에 실패했으므로 EL1 추가 guard/새 잠금 채점은 실행하지 않았다.

실행 중 상대 AI가 카탈로그6.202에 별도의 EC 하루한행 TS2를 추가했다. 본 실험과는 예측 시각 구간·입력·트리수·혼합비·비교 기준선이 다른 실험이다. 향후 같은 아이디어를 다시 제안하기 전에 양쪽 기록을 함께 확인한다. 6.204의 상위팀 방식에 관한 오라클 점수 환산은 상대 AI의 추정이며 실제 상위팀 구현을 확인한 근거가 아니다.
