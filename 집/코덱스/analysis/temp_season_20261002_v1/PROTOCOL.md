# 온도 W30G 계절 좌표 적용 T-S1 · 사전 고정

- 시작 2026-10-02, 실행 2026-10-03 집 코덱스. 사용자가 이번 온도 개선을 직접 승인하여 역할 예외 적용.
- 목적: 배지 온도 sub_temp RMSE 개선. 평가 현재·이전 같은 온실 입력만, 학습 특징은 test_X 전입력 NaN인 MASK. 제출물 생성 없음.
- 기준: W30G=(.4+.1g) BASE+(.6-.4g) CODEX+.3g PFN, g=clip((in_temp-8)/2,0,1), NaN g=1.
- 단일 후보 T-S1: CODEX 잔차 LightGBM의 day 열을 같은 위치에서 DC4 season으로 교체. Ridge 물리 baseline에는 season을 넣지 않음. BASE/PFN/온도 게이트/하이퍼파라미터/가중치 고정. 계절과 입력 온도를 구별해 저온 외삽 구조를 보존한다.
- season은 각 fold 학습일의 날씨만으로 fit, 검증일은 farm/day 보간. 정답/검증날씨/평가날씨/평가통계 사용 없음. 기존 독립 PAV 구현 재사용.
- 기존 온도 DIAG10(5일 묶음 10fold, +/-1 buffer), EXT10/EXT12 고정. CODEX 시드726/727, BASE 시드7/101와 짝, PFN 문맥1..8 및17..24 평균 각각 확인. 모두 12칸. cache row_id/유한행/기준 CODEX 재학습 일치 최대차1e-8 검증; 불일치면 후보 판정 중단하고 원인 기록.
- 채택: 12칸 전부 delta RMSE<0, 각 DIAG10 칸 온실별5일블록 bootstrap 20,000회 p_worse<.025 및95% delta MSE CI상한<0. 후보1개 본페로니 k=1. 평균/시드분산/fold분산/후반·저온·시간별 결과 서술. CV표준편차보다 작은 개선도 사용자 규칙 통과 시 채택 가능.
- baseline 재학습은 cache provenance 확인용; baseline/candidate 동일 행·동일 학습가중치 비교. 기존 원본 온도 가중치는 input-only 고정 기준선으로 유지하되 전체 학습일 기반 rank를 사용한 기존 방식임을 한계로 명시.
- 누수 점검: 원래 CODEX 피처는 당일 현재까지 집계·reset; 물리식 고정. 학습/검증 원시값이 season fit에 섞이지 않도록 vec를 학습일 key로 제한하고 query 값 변조 불변을 assert. 미래/다른온실 평가 입력은 MASK에서 제거. EC 최종잠금 파일/정답 읽지 않음.
- 후보가 실패하면 T-S1 기각, W30G 유지. 결과를 본 뒤 계절 비중/게이트 조정 금지.
