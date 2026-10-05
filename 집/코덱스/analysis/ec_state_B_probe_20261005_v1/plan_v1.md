# EC B 방향·표현력 선행 관문
2026-10-05 집 코덱스. 사용자 진행 승인. predictive-modeling 적용, 기존 검증 규칙 유지.

새 후보 1개 STATE, 비교 대조 MEAN. 현재 A는 실제계절v2 캐시. SG2/submission14 시험 아님. 가족31 누적 alpha=.025/31. 제출/채택 없음. 검증은 공개 DIAG10/A/B, EXT10/12 참고. 3시드7/101/2024 고정.

STATE: LGBMRegressor squared residual y−A, 100trees lr.03 depth3 leaves7 min_child96 min_split_gain.01 reg_lambda10 deterministic2threads. feature=현재A,prefix A,입력14열의0~h prefix means,h/23(17열). anchor/neighbor/distance/row_id/farm/day/high label 미사용. 전기간 모든 inner b행 학습. B=A+clip(predResidual,±.3), train bounds clip. MEAN은 같은 학습행 잔차 평균(모델선택용 후보가 아닌 대조). 설정/한계 확인 후 튜닝0.

1. 전체22fold×3seed에서 inner b(a-fitted A)로 훈련→outer q(tr-fitted A) 검증, 두 모델66셀씩. 기존 baseline검산 필수.
2. DIAG10 inner b를 기존5기록일3분할±1일 purge로 제외, 두모델×90셀 meta OOF. A/피처/보정법 동일, gate없음. 반복 출현행은 독립표본 아님.
3. 같은 meta valid 행에 학습량 대조: vi를 학습에 넣되 farm별 같은 수의 ti 날짜를 가장 이른 순서로 제거. 각farm 학습행수 같음, 대상vi동일, 원A/피처동일. 이 대조는 vi정답을 학습하므로 재대입 진단일 뿐, CV점수/채택 근거 아님. 학습구성 차이는 남아 순수과적합 크기 추정 아님.
4. 조기 중단 관문: STATE가 outer DIAG/A/B×3seed 전부 개선, meta OOF all/high×3seed 전부 개선, DIAG2차악화<2%를 모두 충족해야 다음 gate 설계 검토. 실패하면 gate학습/하이퍼파라미터 탐색 중단. 통과해도 사전bootstrap alpha와EL1 미확인/완전A-B innercrossfit 미완료 때문에 즉시채택 금지.

누수 점검: A 캐시 provenance 원감사 재확인; 입력 prefix 현재시각까지, 학습MASK기존audit; y는잔차목표만 사용; 모델fit inner/meta train만; test 통계/입력/잠금 읽기0. prefix 결측대체는 기존a훈련자료의 같은시간 중앙값. b 스케일러없음. 날짜통제는 진단분할용만. 미래 입력변조 앞예측불변은 기존prefix audit+새model queryindependence/replay로확인.

기존 가족26은23열/anchor존재2차행/alpha100Ridge/±.06보정, 이번은17열/all행/트리잔차/±.3·meta및동일학습량대조. 기존모든잔차모델무효 주장은하지않음. WT0는구성원가중치로중복아님;로그FAIL/allbetterFalse/P.3626 읽기만. 사전등록 source/plan/verifier hash와 준비정보를 main에 커밋한 후학습.
