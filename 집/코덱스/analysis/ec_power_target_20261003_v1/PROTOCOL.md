# EC 제곱 목표 학습 — 실행 전 고정

사용자 목표: 유의미한 EC 개선이 검증될 때까지 분석·모델 실험 지속. 2026-10-03 집 코덱스.

카탈로그/코드 대조: 로그비율(LR1), 고값가중(H9/EC_RARE), 고상태전문가(H15/STATE), 상위분위수(H10)는 시험했으나 모두 규칙 미통과. 이번 후보는 ExtraTrees의 목표를 EC²로 바꿔서 분할과 잎의 학습값을 함께 바꾼다. E[Y²|X]의 제곱근은 조건부 평균과 같지 않으므로 RMSE 최적성이 보장되지 않는다. 고값 과소 감소와 일반 날 과대 증가를 함께 보고하며 해석을 정당화하려고 기준을 낮추지 않는다.

- 단일 후보 POWER2_ET, 기준 현재 계절v2. 기존22fold ×3seed(7/101/2024), DIAG10/A/B/EXT10/EXT12 유지.
- 학습 입력은 기존 MASK세계 FULL38에서 day 대신 기존 동일 season을 끝 열에 사용. 전처리/season은 해당 학습fold만. query 현재이전 같은온실 입력만. 외기·실제계열·과거EC를 새 query특징으로 넣지 않음.
- ET 기존동일600trees/leaf1/max_features1.0. target=EC², raw예측=sqrt(max(ETprediction,0)); 인과 shrink 한 번 적용.
- 후보=clip(기존계절v2 + .48*(shrink(new_ETraw)-기존이미shrink된ET_S캐시),학습EC최소최대). ET 구성원 전체교체의 기존비중이며 사후변경없음. PFN/LGB/MLP 재학습 없음.
- 동일목표 원ET는 첫 DIAGfold 시드7에서 재학습하여 기존캐시(수축 후) 1e-10 이내 재현해야 함. 실패하면 점수 판정 중단/원인교정.
- 모든15칸 개선·DIAG 온실층화5기록일 block bootstrap20,000회 p_worse<.025/2,97.5% CI상한<0. 현재 지속목표에서 지난SOFT_RESID10과이번POWER2 두 시도를 한 family로 보수적 계산. 더 시도하면 전체family에 추가. 공개통과 전EL1/새holdout 없음;이미소모final lock 금지. EL1검증 없이는채택불가.
- 공통원행 기준RMSE/고EC31일·후반46일·농장/편향 보고;fsum 독립검산·부트스트랩재계산·future/다른온실불변성.
- 출력: ownanalysis 코드/로그/요약, ownlocal 폴드별OOF. 새제출/test예측/확정구성 없음. 완료폴드checkpoints는코드해시같을때만재사용.
