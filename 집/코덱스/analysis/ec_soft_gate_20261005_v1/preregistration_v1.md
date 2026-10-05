# EC 연속 soft gate 사전 설계

2026-10-05 집 코덱스. 사용자 승인: 동적 앙상블 전략을 해보자.

- 한 후보 family27. A=실제 계절v2 최종출력. B=clip(A+.5 clip(이웃평균-현재까지 A평균,-.6,.6), 학습정답범위). 동일농장·동일pass·이전 공개 학습일 3이웃만 사용한다. B는 고EC 단독 전문가가 아니라 인과적 이웃 보정 전문가다.
- 전 기간에 적용하며 이웃 부재이면 A 유지. 2차 전용 판정 절차를 사용하지 않는다. 기존 가족25의 2차 전용 적용과 hard gate를 그대로 재시험하는 실험이 아니다.
- 입력 23열: A, prefix A, 이웃평균, 이웃-prefix A, 이웃거리2, 이웃정답표준편차, log1p(이웃가능일수), 현재까지 입력14열평균, 시간/23. 농장별 같은pass 참조 선택 및 거리대체/표준화는 원 코드 그대로 사용한다.
- g=sigmoid(Z theta). 출력 A+g(B-A). 게이트 학습은 내부 heldout b 모든 이웃가용행 혼합 MSE + .01 ||theta-theta0||². theta0의 intercept=logit(.05), slope=0. slope/intercept 모두 정규화. B 승패 라벨, 문턱, 후보 선택, 탐색 없음.
- StandardScaler는 내부 b의 이웃가용행만 fit. 내부 A는 a 학습 모델의 b 예측. 이웃도 a만 참조. 외부 A/B와 게이트 예측에는 외부 평가 정답을 쓰지 않는다. inner b는 전체 crossfit가 아닌 기존 한 heldout subset임을 보고한다.
- L-BFGS-B, 초기 theta0, maxiter3000, ftol1e-13, gtol1e-8, maxls50. 성공 및 gradient infinity norm <=1e-6 요구. 빈자료/수정방향0이면 A 유지. 수치 실패면 중단하며 결과 기반 최적화 설정 변경 금지.
- 기존 matched 캐시의 입력·ID·SHA·문맥·purge를 preparation에서 전수 재확인한다. A/B 원모델 새 적합 없음. test_X 값/전체 test 통계/원시 EC 정답/잠금정답/EL1 새채점 없음.
- seeds7/101/2024, DIAG10 10fold·A5·B5·EXT10/12 각1: 66셀. 채택 필수 DIAG10/A/B 모든seed 개선 및 DIAG10 각seed p_worse<.025/27와 조정CI상한<0. EXT는 보고만. DIAG10 2차 >=2% 악화면 보류. EL1은 미채점이므로 통과해도 채택 보류 후 별도 허용된 guard가 필요하다.
- bootstrap 기존 농장내 시간순 5기록일 블록, 20000회, RNG20261003+seed. 고EC=공개일평균>=1, 일반/농장/pass 세그먼트는 진단용이며 선택에 사용하지 않는다.
- 학습전 synthetic finite-difference gradient 검사 및 내/외부 방향/ID 검사, 학습후 전66 모델 재생·스케일러/목적함수 gradient·독립 RMSE와 bootstrap 검산. 원자료와 원실험 파일 수정 없음. 제출파일 생성 없음.

누수 점검: 현재 및 prefix 입력=현재까지; 이웃정답=허용된 공개학습참조만; 기준예측=inner a 또는 outer train fit; gate scaler=inner b만; 외부 y=최종 채점만. 기존 계절 캐시의 원훈련 재현은 재실행하지 않으며 캐시 provenance 검증의 범위를 명시한다. 반복 공개 검증으로 독립 holdout은 아니다.
