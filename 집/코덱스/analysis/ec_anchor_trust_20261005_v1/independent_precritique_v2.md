# v3 runner / v2 verifier 갱신 비평

v1 비평에 대한 보완을 확인했다. close는1e-12, prior preparation은df4ff 핀을 검사한다. reference-only NaN 처리·prefix cut/미래변형 감사, 새 fit 반복과 serialized matrix/sigmoid·reverse/single/other query·scalar clip 비교도 코드에 들어갔다. 시작 시 registration의run/prep/prereg SHA와 전체 preparation 재구성 동일성을 확인한다. 직접 자기 target 참조가 발견되지 않았다는 판단은 유지한다. 실제 학습·금지 데이터 읽기·재채점은0이다.

## 실제 판정 전에 남는 필수 gate

- verify_v2는66 manifest의 정확한(v,k,seed) set/order와 CSV schema·key columns를 명시 검사해야 한다. 현재 row_id·정답·baseline·예측을 확인하지만 farm/day/hour/validator/fold/seed/bounds/anchor/d1/d2를 기준 q/signature와 비교하지 않는다. 잘못된 bootstrap grouping이나 다른 seed 표시가 산술 PASS로 남을 수 있다. aggregate 중복 검사는 중복셀을 막지만 이 모든 의미 검사를 대신하지 않는다.
- first receipt는 현재 status만 검사한다. 실제 first 실행 중 감사 assert는 있으나 저장 JSON에는 repeat_error/fit/source만 있어 다른 검사의 수치·전체 key·query/prefix 대상·atol·SHA를 whole가 검증하지 못한다. 최초 감사 수치를 기록하고 source/prep/prereg/firstSHA와 결속해야 한다. verifier 자체도 registration에 고정해야 하며 verify_v2의preregSHA 재확인이 필요하다.
- 별도 내부 검산 코드가 예정돼 있다면 추가 checkpoint metadata는 최소화할 수 있다. 기존 저장mean/scale/coef/intercept/n/positive를 사용하면서 준비된b에서 eligible IDs·feature order/shape/finite, 정규화weight/target, scaler median/population variance/scale/n을 독립 재구성해 비교하면 된다. 최소 추가 항목은 eligible ID hash, feature/target/weight hash, model params, constant probability(또는positive/n exact 검증)와 첫 감사 영수증이다. scaler moment 상대 tolerance와 최종prediction 절대1e-12는 분리해야 한다. 이 검산은 전체66/aggregate gate와 함께 첫 outerscore 전에 완료해야 한다.

## 강제 구현과 단순한계를 구분

이전 보고서의 “매 fit freshness 재확인 필수”는 요구 수준을 필요 이상으로 높였다. immutable main 등록과 입력·cache 수정 금지, 시작/종료의 source/input/runtime/cache pin 일치가 보장되면 매 fit hash가 유일한 수용 방법은 아니다. 실패와 partial output은 보존하고 임의 재시작/선택 채점이 없어야 한다. 이를 증명하는 failure·completion receipt가 필요하며 각 fit마다 같은 전체cache를 읽는 구현까지 강제하지 않는다.

prior immutable SHA와 전체 prior cache hashes·ID·bounds를 재검증해 **기존 검증 증거를 재사용**하는 방법을 수용한다. 이를 새 R3/PFN 학습 또는 독립 provenance regeneration으로 설명하면 안 된다. 현재 original caches가 실제 오염됐다는 주장은 하지 않는다.

single inner b subset, reference-hour pooled farm/pass statistics, samefarm/pass/earlierday anchor 후보의 차이, median0 empty fallback은 명시해야 할 설계 한계이며 그 자체가 실행 차단 사유는 아니다. 입력·target 누수 경로가 없는 한 전체crossfit으로 바꾸거나 모델을 더 복잡하게 만들도록 요구하지 않는다. GATE의supervised cost target과 RIDGE residual target도 의도된 변경이다.

eligible threshold1e-14와 한 클래스 fallback, eligible0의 사전 중단 정책은 코드/사전등록에 명시하고 합성 오염/경계 사례를 확인해야 한다. fit stationarity/gradient 검산은 학습 목표·weight를 확인하는 증거이며 optimizer의 완벽한 글로벌 재현을 약속하는 것은 아니다. query-prefix 계산과 저장모델 replay를 서로 구분해야 한다.

현재v2 whole의 key/schema/first/registry gate는 실제 판정 전에 보강할 문제다. 설계와prior증거 재사용 자체는 지지한다. 보강된 verifier·내부검산·합성 fixture를 새버전으로 고정하면 모델 구조 변경 없이 진행할 수 있다. 실제 결과의 strict15·seed별DIAG/.025/26 및 사전 normal guard를 바꾸지 않는다.
