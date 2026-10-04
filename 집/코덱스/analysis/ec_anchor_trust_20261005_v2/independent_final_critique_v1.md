# family25 GATE / family26 RIDGE 최종 비교 비평 v1

2026-10-05 집 코덱스. 두 mode 완료 verification/crosscheck JSON 및 case_diagnosis_v2.json을 읽고 저장 수치·판정·사례 부호를 대조했다. Decimal30/manual bootstrap3은 각 mode의 기존 독립 검산 receipt를 확인한 것이다. 본 에이전트가 CSV 전수 검산을 새로 수행했다고 주장하지 않는다. 신규 fit/native predict/원시 EC/test/EL1/잠금 조회 및 Claude 코드 실행은 0이다.

두 mode 모두 REJECT가 맞다. 각 66셀/83160행 및 학습 scaler/objective gradient, 저장 출력 replay, Decimal30·farm별 5-observed-day manual bootstrap3은 PASS지만 이는 감사 통과다. 두 후보 모두 strict 개선은 0/15이다. GATE DIAG10 RMSE 변화는 seed7/101/2024 +3.314514%/+3.516610%/+3.078655%; RIDGE는 +0.434774%/+0.416276%/+0.367973%다. RIDGE EXT10도 +0.345169%/+0.342801%/+0.389452% 악화한다. A/B와 EXT12의 동일 결과는 엄격 개선으로 계산하지 않는다. GATE p_worse .98885/.98545/.98960, RIDGE .71035/.70800/.68705이며 두 mode adjusted CI 상한 모두 양수다. 고정 strict15와 alpha=.025/26를 통과하지 못했다. RIDGE는 사전 prior c933 등록의 bounded continuous 대조이므로 GATE 결과를 보고 새로 고른 사후 튜닝이 아니다.

GATE 일반329일은 -1.420175%~-1.945554% 개선했지만 고31일은 +5.664030%~+6.140091%, pass2는 +11.408263%~+13.348377% 악화했다. RIDGE는 고31일 +0.448551%~+0.484545%, 일반329일 +0.216919%~+0.338267%, pass2 +1.412131%~+1.702989%로 모두 악화했다. 작은 연속 교정으로 손해가 작아졌다고 표현할 수 있으나 개선·채택 근거는 아니다. 동일 세 시드는 같은 날짜와 label을 반복 평가하므로 독립 표본 세 배가 아니다.

선택 사례 F13 day231/233, F47 day216은 두 mode에서 각3seed 모두 ΔSSE<0이다. GATE 9개 사례 ΔSSE 범위 -2.928629~-0.399663, RIDGE -0.536237~-0.016948이다. 반면 GATE 고군 변경153 seed-row는 모두 음의 교정이다. 이 수는 seed별 동일 시간 반복을 포함하며 153개 독립 사건이 아니다. 저평가 baseline에 하향교정이 더해져 손해가 생긴 사례와 과대평가 baseline에 도움이 된 사례가 공존한다. 사후 선택된 세 사례 개선으로 전체 실패를 뒤집을 수 없고, 전역 tradeoff를 운영 정보 부재나 특정 물리 원인의 증명으로 확대하면 안 된다.

v1은 eligible0에서 부분21셀 후 중단·score0였고, v2는 빈 학습표본에서 baseline/delta0을 재등록했다. 이는 정의되지 않는 추정량의 실행가능성 보완이며 최초 실행과 완전히 같은 사전등록이라고 표현하지 않는다. empty fallback과 pass2/strict previous-day anchor 조건은 일부 검증기에서 correction 노출이 없다는 설계 제약이다. actualv2 기준선에 우리 독자 anchor/reference/preprocessing/correction 학습을 적용했으므로 Claude SG2 재현 또는 신뢰도 learner만의 순수 인과 효과라고 주장할 수 없다. 이전 near-constant scaler 및 learning-fit receipt 결속 지적은 새 구현으로 보완됐다.

## Claude FZ0/FZ1/HG2 readonly 갱신

FZ0 로그는 gated32일(고23/false9)의 48특징 family-wise p=.1814, leave-one-day-out L2 logistic AUC=.74를 제시한다. 이는 해당 소표본·선택 특징·C=.1 모델의 성능이며 가능한 모든 입력 활용의 정보 상한이 아니다. full-day 정보를 허용한 단일 모델도 비선형 상호작용·다른 regularization·새 표본에서 가능한 성능을 상한으로 제한하지 못한다. 전역 gated 표본의 median imputation 후 LODO를 돌려 imputation이 완전히 fold-local하지 않다는 한계도 있다. 유의하지 않음은 정보 없음의 증명이 아니며 H3의 AUC<.75 판정은 사실상 .74 한 점과 임의 경계에 의존한다.

FZ1 도메인 score는 AUC .77, F13 .65/F47 .95, one-sided p=.0217이나 사전 .80 조건을 못 넘어 clue=False다. log의 false p와 farm 차이를 함께 보고해야 한다. 전체 labelday 통계 및 full-day signature와 inferred dong 기반 earlier/later dip 탐색은 사후 비인과 설명이다. 후일 label을 요구하는 dip는 현재 예측 특징으로 사용할 수 없다. domain 부호의 고정은 통계 weight 튜닝을 줄이지만 이후 threshold와 모델 선택의 사후성을 지우지 않는다.

HG2는 source에 FZ1을 본 뒤 threshold0을 정했다고 명시하며 새 causal prefix와 ref-only S_low table/SIG scaling을 구현했다. 그 개선을 인정한다. 하지만 prepare_structure WV는 여전히 train+test의 전체 p1 통계로 표준화하고 main ref=labset-vd는 ±1 purge train과 다르다. 따라서 모든 통계가 reference-only라는 포괄 표현은 부정확하다. st_dong/role의 inferred provenance 한계도 남는다. HG2는 진행 중이므로 결과를 읽거나 실행·중복 모델을 만들지 않았다. 새로운 검증 분할은 도움이 되지만 반복 사용된 데이터 전체에서 완전히 새로운 독립 holdout이라고 자동 주장하지 않는다. 10-05 공식 PDF 원문을 이번 비평에서 읽지 않았으므로 규정 합법성은 판정하지 않는다.
