# EC 세 종류 soft gate 사전 설계

2026-10-05 집 코덱스. 사용자 승인: 다른 모델들도 실험해보자. 한 번에 게이트 모델 종류만 비교한다. 새 전문가 B 비교는 이 실험에 포함하지 않는다.

- 가족28 LR,29 LGB,30 MLP. 이전 family27의 실제 계절v2 A/인과적 제한 이웃보정 B/23열/모든기간/동일내외분할/캐시를 고정한다. 기존 원코드와 준비/registration SHA를 검사한 뒤 사용한다. SG2/submission14 기준선이 아니다.
- 문서의 B 승리확률 학습을 회귀 손실에 맞게 구현한다. 내부 a 학습 모델의 heldout b에서 cost=(A-y)^2-(B-y)^2, label=(cost>0), weight=abs(cost)/평균abs(cost). 이웃가용이며 abs(cost)>1e-14인 행만 fit. both bad/good인 회귀 행도 포함하며 오차 크기로 비용 가중한다. 동일 cost라벨·가중치·학습행·표준화·출력으로 모델 종류만 변경한다.
- gate 확률을 그대로 g로 사용하고 최종 A+g(B-A). g>=.5 등의 switching/새문턱/비중계수 없음. 이것은 direct-MSE 최적 혼합률과 같다고 주장하지 않는다.
- LR LogisticRegression(C1,max_iter3000,tol1e-8). LGB binary classifier(n_estimators150,learning_rate.03,num_leaves7,max_depth3,min_child_samples30,min_split_gain.01,reg_lambda10,2threads,deterministic/force_col_wise). MLPClassifier(hidden8,tanh,lbfgs,alpha10,max_iter3000,max_fun100000,tol1e-7). 클래스0/1만, 단일class는 그 상수확률, 빈자료는 A유지. seed7/101/2024. 수렴경고는 숨기지 않고 중단한다.
- StandardScaler는 동일 eligible inner b만 fit. 이웃은 inner a만/외부는 outer train만, 모든 입력 prefix는 현재까지. meta에 외부 y/전체test 통계 투입0. 내부 b는 원 matched 한 heldout subset이며 full crossfit 아님.
- 각66셀 총198셀. 판정 DIAG10/A/B 모든seed 개선 + DIAG10 각seed p_worse<.025/30(세후보 및 누적families에 보수적 보정) 및 같은 alpha의 조정CI상한<0. 2차 DIAG >=2% 악화 보류. EXT는 보고만, EL1은 새채점하지 않아 통과하더라도 guard미확인으로 채택보류.
- 동일 기존 농장별 5기록일 블록20000회 RNG20261003+seed. 고EC 공개일평균>=1/일반/농장/pass 및 fold/seed분산은 진단용.
- 첫실제fit 반복 및 전체 직렬화 재생, LR계수·MLP가중치·LGB 트리 JSON으로 독립 forward, 행순서·단독행·다른query독립성, scaler/fsum, 비용라벨/가중치재계산, weighted logloss, 목적함수/gradient 점검(LR/MLP), 모든 RMSE 독립fsum 및 3manualbootstrap 시행 후 보고한다.
- 최초fit 전에 run/verify/prereg/preparation SHA를 main에 등록. 성능을 본 뒤 설정/피처/학습행을 수정하거나 이전 기각 후보를 새규칙으로 재채택하지 않는다. 실패 원인은 제한된 A/B/피처/자료에만 해석한다.

누수 점검: A/B는 원 inner a 또는 outer train으로 구축; scaler는 inner b만; label/비용은 inner b 정답만; 원서명 현재/이전입력; 외부정답 최종채점만. 모든 원 캐시 provenance를 다시 검사하며 기본모델재학습0/잠금0/test값0/제출파일0. 기존 TabDPT나 클로드WT0 작업에는 변경/중복실행 없음.
