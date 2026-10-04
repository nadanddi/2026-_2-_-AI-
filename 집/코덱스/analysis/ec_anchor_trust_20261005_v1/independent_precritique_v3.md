# family25/26 최종 사전 독립 비평 v3

검토: 2026-10-05 집 코덱스. 대상 run_v6.py / verify_v5.py / verify_learning_v4.py 정적 읽기만. 실제 데이터 조회·모델 학습·예측·점수 계산은 하지 않았다. 기존 파일은 보존했다.

판정: runner v6의 변경은 의도된 설계와 맞는다. 다만 요청된 학습 감사 결속과 scaler 독립 재현에는 아래 두 보완이 남아 있어, 현재 세 파일 전체를 무조건 최종 수용하지 않는다. 모델 조건·채택 문턱을 바꿀 필요는 없다.

## 실행 전 최소 보완

1. verify_learning_v4.py의 scales[scales==0]=1은 StandardScaler의 near-constant 판정과 동일하지 않다. float64에서는 var <= n*eps*var+(n*mean*eps)**2 인 특징도 scale=1이 된다. 평균이 큰 거의 상수 특징을 정확히 0만 검사하면 유효한 원모델을 오염으로 오판할 수 있다. 독립 fsum 분산을 보존하면서 이 명시적 상수 bound를 재현해야 한다. scaler moment는 크기에 따른 상대 오차와 예측의 절대 1e-12를 분리하는 것이 타당하다. 이는 검증기 산술 정확성 보완이며 실제 특징 변경이나 예측 문턱 완화가 아니다.

2. verify_v5.py는 learning receipt의 verifier 문자열을 registration.learning_sha와 비교하지만 실제 verify_learning_v4.py 파일 SHA를 비교하지 않는다. learning mode, 정확한 66개 (validator,fold,seed) 순서, 입력 fit receipt SHA도 결속하지 않는다. learning writer가 실제 사용한 fit receipt SHA를 저장하고 whole이 현재 fit receipt/등록 learning 파일 raw-byte SHA/동일 mode와 66키를 확인해야 한다. 최종 verification JSON에 learning receipt raw-byte SHA도 기록하면 해당 모델의 학습 감사가 점수보다 먼저 완료됐다는 근거가 보존된다. 기존 immutable 출력 운영은 위험을 줄이지만 요구한 명시적 결속을 대체하지 않는다.

## 지지하는 부분

PREV df4ff… immutable 준비물과 모든 이전 cache hash를 재검산하고 a/b/tr/q ID 및 ±1 purge를 확인한다. v6는 NB/NQ/QB/QQ fingerprint를 준비물에 추가했다. top3 선택의 첫 query 24시간 slow/vectorized 일치와 선택 anchor 목록 비교, 원입력 prefix cut/future perturb 6건도 있다. zero-label 직접 자기참조는 발견하지 않았다. whole은 66셀 정확 순서, 17열 CSV schema, ID/좌표/bounds/anchor/d1/d2/공개 target/baseline, coefficients 기반 출력, 83160행 aggregate의 순서와 hash를 score 전에 확인한다. logistic 가중 합 손실의 L2 C=1 gradient, Ridge alpha100 normal-equation gradient의 방향은 일관된다.

## 감사의 정확한 범위와 한계

첫 repeat는 실제 동일 입력 재학습 재현이다. serialized/reverse/single/other-query는 저장된 선형 계수의 NumPy replay 검사이며 각 조건으로 sklearn native predictor를 다시 실행한 감사라고 표현하면 안 된다. prefix 감사는 signature 함수의 여섯 선택 사례이며 모든 query의 종단간 모델 출력에 대한 future perturb 전수 증명은 아니다. REF 함수 재사용 whole은 별도 산술 출력 재현에는 독립적이지만 특징 정의의 독립 구현 검증은 아니다.

inner b의 기존 OOF baseline은 a만으로 만들어졌으나 b 전체에서 correction learner를 학습하므로 b 내부에서 correction까지 완전 crossfit된 설계는 아니다. outer q에 관한 직접 target leakage와 이를 혼동하면 안 된다. ref-only per-hour 통계의 farm/pass pooling, all-NaN median=0, 동일 farm/pass의 이전 day만 anchor 후보로 쓰는 조건도 명시할 한계다. 실제 계절v2 캐시는 검증된 이전 준비물·cache provenance를 재사용하며 새로운 R3/PFN 재학습 재현을 주장하지 않는다.

## 이전 판단 정정

모든 fit 직전 전체 cache SHA를 다시 읽는 것만이 유일한 안전 조건이라는 이전 표현은 지나쳤다. immutable source/등록 receipt, 시작·검증 종료의 fresh input/cache hash와 변경 금지 운영으로도 동일 파일 결속을 달성할 수 있다. v4 PREV 치환 실패를 보존하고 v5/v6 새 파일로 수정한 절차는 적절하다. v1의 1e-10 close 문제는 현재 v6 절대 1e-12로 해결됐다. 현재 남은 위 두 항목은 기존 후보 조건이나 strict15/Bonferroni .025/26 채택 기준의 변경을 요구하지 않는다.
