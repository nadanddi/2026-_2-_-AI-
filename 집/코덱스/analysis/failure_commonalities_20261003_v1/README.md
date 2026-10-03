# 실패 공통점 분석 — 최종 파일 안내

사용자 요청에 따라 온도·EC 주요 문제에서 실패한 사례를 모은 사후 분석이다. 온도16일, EC 넓은 실패29일/핵심 과소예측17일. 후보 채택·제출 분석이 아니다.

- 설명: `실패_공통점_보고서_v1.md`와 `추가_검산과_반론_v1.md`.
- 각 특징의 공통 조건을 모두 읽는 목록: `공통점_전체목록_v1.md`. 모든 컷·비공통 결과는 `conditions_with_date_control_v2.csv`.
- 최종 사례: `final_daily_cases_v2.csv`, 원시시간별 공개값 `public_hourly_cases.csv`.
- 전체400 입력 요약·관계·이력: `all_input_features.csv`.
- 명시 조건과 빈도: `fixed_rule_counts_v3.csv`, 두 조건 동시 발생: `pairwise_input_conditions_v2.csv`.
- 실제훈련지원: `actual_training_coverage_v2.csv`, `actual_training_neighbors_v2.csv`, `coverage_summary_v3.csv`.
- 검산: `verification_v4.json`(73264), `training_split_audit.json`(6337), `raw_rule_audit.json`(340). 체크 중복이 있어 합산을 독립 증거 수로 부르지 않는다.

최종 코드 실행 순서는 analyze.py → deeper_v2.py → patterns_v3.py 및 extras_v2.py → verify_v4.py / audit_neighbors_v2.py / audit_rules.py → report_v3.py. env 공용 부트스트랩과 기존 공개OOF·world 캐시가 필요하다. 검산코드는 새 파일 없음을 확인한다. 재현 시 기존 파일을 덮어쓰지 말고 새 버전 디렉터리를 사용한다.

최초 산출물은 기록용으로 보존했다. CSV 기본 읽기 정밀도 문제로 후속 조건 경계가 바뀌어 최종 모든 수치 CSV는 round_trip 읽기를 사용한다. 검산기 v2/v3는 반복 검사 성능 문제로 중단했으며 v4에서 같은 검사를 정상 완료했다. 최초 분할 감사의 ±1일 buffer 누락은 감사 v2에서 수정했다. 본분석의 훈련 분할은 처음부터 원 buffer 규칙이었다. 상세는 보고서7절.

온도1시드/원문맥, EC1시드/원문맥 공개기준선의 탐색이다. 성공 사례가4/2일뿐이고 시기·인접일·동일날씨·사후 임계값 선택이 있어 인과나 새 일반화 성능을 주장하지 않는다. EC 잠금 정답·파일을 읽거나 재채점하지 않았다. 실제 train_y는 이 분석에서 읽지 않았다.
