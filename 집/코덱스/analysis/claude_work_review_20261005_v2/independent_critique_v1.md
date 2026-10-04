# HK1·AF0·AF0b 독립 코드 비평

코드와 로그만 읽었다. 작성자 코드를 실행하거나 원시 train_y/test/잠금/EL1 산출물을 읽지 않았고, 새 학습·재채점은0이다. 아래 성능 숫자는 로그 인용이며 독립 재계산 수치가 아니다. 이전 SG2 비평의 공식10-05 PDF 원문 미확인 한계도 유지한다.

1. **기존 WV/ref 문제는 그대로다.** 세 코드 모두 SG2 prepare_structure를 가져와 train+test 전체 pass-1 weather로 mean/std를 구한다. ref도 labset−vd이며 모델 tr의validation/lock±1 purge보다 넓다. AF0b의 dong용 z 통계는 X.loc[inref]로 제한되므로 이 부분까지 무차별 test-fit이라고 부르면 부정확하다. 그러나 ref 자체의 purge 불일치와 SG2 weather std는 해결되지 않았다. 실제 데이터 기여량이나 규정 최종 적합성은 여기서 판정하지 않는다.

2. **AF0의 ±3은 직접 anchor/twin 제외이며 전체 pipeline 제외는 아니다.** AF0 anchor_features:82, AF0b add_trust:198은 training query와 같은 farm의 |record day 차이|≤3을 keep에서 제외한다. 직접 자기 day target 및 가까운 anchor를 배제하는 개선은 인정한다. 하지만 cal과 signature z 통계는 keep 전에 전체 ref로 계산하고, fallback full_date는 keep을 받지 않는다. training day의 이전 기록이 cal에 있으면 바로 반환한다. 따라서 “±3 내 기록이 모든 단계에서 제외됐다”는 설명은 맞지 않는다. validation query에는 ±3 keep 자체가 없고 ref=lab−vd다. base 모델의 purge와 anchor learner의 참조 범위가 달라지는 문제도 남는다. 자기 y를 직접 사용했다고 단정하는 것과는 구분한다.

3. **AF0b에는 training/validation의 동 특징 생성 차이가 있다.** dong_model:66은 reference pair 행으로 clf를 fit하고 같은 시간의 모든 행을 predict한다. R3 training query의 af_pq에는 classifier 자신의 training 행에 대한 in-sample 값이 들어갈 수 있다. training anchor의 pa는 알려진 pair이면 정답0/1을 그대로 반환한다(:82). main의 ±3 anchor 제외는 dong classifier 학습 집합이나 z 통계에 적용되지 않는다. query EC 자기정답의 직접 누수 증거는 아니지만, trust 특징의 훈련 중 품질과 validation 품질이 달라질 수 있다. 이를 검증하려면 동 classifier 특징도 training query를 제외한 별도 crossfit 방식과 비교해야 하며, 이번 검토에서는 실행하지 않았다.

4. **로그의 .76은 own-training pair accuracy도 physical 동 accuracy도 아니다.** main:124는 validation day의23h pq를 st_dong_assign의 dong과 비교한다. docstring의 “reference pairs / leave-fold-out accuracy”와 실제 평가 대상이 다르다. 비교 대상 st_dong_assign은 ST6의 구조 기반 분류 산출물이며 물리적 동 정답이 아니다. 같은 weather/순서/운영채널을 공유한 두 추론의 일치율일 수 있다. “동 식별 정확도76%” 대신 “validation 기록에서 기존 inferred dong과 일치율.76(n265)”로 한정해야 한다. classifier를 ref training 행에서 평가하는 in-sample 경로와 이 로그 평가를 혼동하지 않는다.

5. **pair second를 물리 동B로 바꾸는 가정은 미검증이다.** dong_model은 ref의 인접 exact-weather pair를 first0/second1로 지정한다. 세 날 이상 연속 twin이면 가운데 기록의 label이 다음 pair에서0으로 덮어써질 수도 있다(:57). 실제 발생 여부는 데이터를 읽지 않아 확인하지 않았다. pq는 h≥2에서 전날과 현재prefix weather가 같으면1을 강제한다. 이것은 prefix 관측 규칙이지만 동일 날씨가 늘 같은 물리 동 순서를 뜻한다는 보장은 없다. pa의 full-day training 참조는 query의 미래 입력과 구분하되, 물리 동 ground truth를 확보했다는 주장으로 확대하지 않는다.

6. **HK1의 단서는 아직 실패다.** 로그는 gated32일(true23/false9), best d1 AUC.80, farm 내 label permutation/max12특징 family-wise p=.0674, clue=False라고 보고한다. 여러 특징 중 최고 AUC만으로 filter를 정당화하지 않는 것은 적절하다. day 단위 집계는 hour pseudoreplication을 줄이지만 농장 내 시계열·인접 twin의 의존성까지 해결하지 않는다. farm 내 개별 day permutation은 교환가능성을 가정하므로 block 의존성에 대한 민감도 검사가 필요하다. 같은 공개자료로 gate를 설계한 후속 성공은 untouched holdout 확인이 아니다.

7. **AF0/AF0b는 high 개선과 normal 손해가 공존한다.** 로그의 두 clue는 모두False다. AF0b의 DIAG 전체는 세 seed 모두 SG2보다 개선하지만 normal은 +25.2~+43.9%, EL1 normal은 +37.3~+48.6%로 작성자 사전 +2% 허용치를 넘는다. 이 로그만으로 anchor feature나 동 trust 전체의 무효를 증명하지 않으며, physical 동 혼합이 정상일 손해의 확정 원인이라는 AF0b 설명도 가설이다. 실제 학습/참조 분포·target-dependent anchor·표본 불균형 등 다른 설명이 남는다. EL1 숫자는 재계산하지 않았다.

8. **“세 날에 signed SSE 손해 집중”은 별도 정의와 증거가 필요하다.** 읽은 세 스크립트·로그에는 그 비율을 직접 계산하는 코드가 없었다. 다른 진단의 주장이라면 출처를 분리해야 한다. signed ΔSSE를 net 전체 ΔSSE로 나누면 나머지 날의 개선과 상쇄돼 비율이100%를 넘거나 분모가 거의0이 될 수 있다. 이는 총 오차 또는 양의 손해 중 점유율과 다르다. 세 날의 양의 ΔSSE 합/전체 양의 ΔSSE 합, 순변화, 절대변화, seed별 결과와 day 수를 구분해야 한다. 사후 상위3일을 제거한 나머지 성능은 새 채택 근거가 아니며, 그3일의 실패 원인이나 나머지 날의 안전성을 증명하지 않는다.

코드상 위험과 로그 산술 성공/실패를 구분해 전달하는 것이 타당하다. 최신 신호는 고EC 앵커 정보의 유용성을 탐색하는 근거이지만 현재 actualv2·strict15/seed별Bonferroni 채택 PASS나 전체 causal pipeline 재현의 증거는 아니다. cache 존재 시 skip 및 폴더 전체 concat 경로도 그대로여서 source/input/fold/feature manifest 검증 없이 실행 provenance를 확정할 수 없다.
