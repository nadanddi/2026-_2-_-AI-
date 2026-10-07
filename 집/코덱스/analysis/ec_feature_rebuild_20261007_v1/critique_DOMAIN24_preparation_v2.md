# 도메인24 이벤트 v2 독립 재검토

2026-10-07. domain_features_v2.py/prepare_domain_BLK_v2.py/run_domain_BLK_raw_v2.py 소스 읽기만 했다. 준비 감사 session20171의 완료 결과, fit registration, 모델 fit/채점/GPU 실행은 확인·실행하지 않았다. 성능결과 전 이벤트가설을 구현한다는 부모의 전달과 보존된 candidate CSV v4의 선언을 구별하여 검토했다.

## 핵심 판정

v2는 이전 DF01의 미구현 이벤트 설명을 실제 입력 co-change feature로 보완한다. 결과를 보지 않은 현재24가설 범위의 구체화로 볼 수 있으며 추가 가설을 요구하지 않는다. 소스상 새로운 미래입력/다른farm/보류 정답 누수는 발견하지 못했다. 실제 준비 PASS와 fit전 source·exact열 registration 봉인이 남아 있으므로 지금 모델전체 PASS를 뜻하지 않는다.

## DF01 주요 닫힘과 정확한 의미

D05는 vent/fog 값변화 후 humidity_gap 변화, D09는 fog 값변화 후 VPD 변화, D11은 공급 onset/offset 시각의 ΔCO2, D22는 fog 값변화 후 실내온도/습도/VPD 변화를 포함한다. 새8series마다 동일1/2/3/4/6h lag/rate/d2/rolling grammar를 적용하여 한 가족 내 고정묶음으로 둔다. best-window 선택 경로는 없다.

event_response는 마지막 관측 actuator 값변화 시 t−1 환경값을 anchor로 잡고 그 뒤 현재 환경값−anchor를 출력한다. 매시각 단순Δ나 생리적 반응의 순효과가 아니다. 새로운 값변화가 오면 anchor를 덮어쓴다. h0, 비연속시각, actuator/환경 현재 또는 직전 결측이면 anchor를 폐기하며 새 관측 event가 나오기 전까지 NaN이다. 이전 record의 anchor가 넘어가지 않는다.

onset/offset는 >0과<=0 상태전환의 그 한시각 ΔCO2만 출력한다. 연속 공급 중 양의 강도변화를 onset으로 부르지 않으며 모든공급 event의 지속후 반응이라고 표현하면 안 된다. CSV 설명의 '구동기 전환별 분리'에 이 정의를 붙인다. D13의6h 포함 및 same-record origin mapping도 새registration에 명시한다.

## DFV201 — 새series 중복·family 효과의 해석 (P2)

D09 `fog_vpd__event_response`와 D22 `act_fog__vpd_response`는 동일 actuator와 동일 VPD를 event_response에 전달하므로 값과 파생grammar가 동일하다. D22는 이에 온도/습도와 actuator event를 더 포함하므로 두 가족이 독립된 정보 발견이라고 셀 수 없다. D05와D22 습도 관련 proxy도 일부 연관된다.

window1mean/sum/current alias 및 baseline과같은columnname 제거 정책은 이전한계를 유지한다. 문자열상 다른이름이면 같은값이 남을 수 있다. 논리적으로 선언한 전체grammar를 유지하되 actualcolumn/alias map과 effective imputer output열을 기록하고, 결과후 좋은가족만 alias 제거/추가하지 않는다.24family ET조건부효과가 정보추가와split후보중복효과를 함께 포함한다는 한계도 유지한다.

## DF03 보완과 남은 감사범위 (P2)

새3기존series의lag/rate/d2 scalar검산이 rolling검산에 추가되어 이전 누락은 소스상 닫혔다. 합성 event `[0,1,2,3,4,6,7]`의 값변화/h0/gap 및 모든prefix 대조가 마지막 anchor reset 의미와 일치한다. fullfeature prefix/future poison은 새8series도 포함하므로 실데이터에서 인과경계를 검사한다.

합성case에는 환경/actuator NaN이 없고 onset/offset 결과를 별도 scalar로 대조하지 않는다. 새8eventseries dynamics32열의 독립numeric 대조는 기존3series 감사와 같은 함수를 재사용하는 것 이상의 검산이 없다. fit전에 작은 합성 missing/environment/actuator, 0→양수/양수→0 및 강도변화, event직후gap, event현재시각과후속시각에 대한 expected값을 확정하면 감사범위가 더 정확해진다. 핵심미래누수 발견에 따른 중단은 아니며 '모든새열독립수치검산완료'라는 주장만 제한한다.

## rawrunner source 검토

train_base의index를tr.row_id로 지정한 뒤 features[expected]를left join하므로 y의기존tr순서와맞는다. training/query ID disjoint를검사하며 fit은train_base+train행추가열에만적용한다. ET만재fit하고다른구성원/후처리는이runner에서건드리지않는다. 원baselineET 재현3회 후72candidatefit, reverse/scattered32/single8 및 fixed1e-6를저장한다. source변경을fit전후확인하며all_missing_train_columns와imputer statistics SHA를기록한다. 아직전체candidate causal/postprocess감사는다음단계다.

### DFV202 재개·strictreceipt 보완 (P2)

baseline replay resume는status/registration만검사하며현재train/querymatrixSHA와현재원baselinereceiptSHA를saved값과다시대조하지않는다. candidate resume는matrices/row_ids/predSHA/status를검사하지만savedcandidate/seed/finite1440/감사check개수·tolerance·값을완전히대조하지않는다. fittedimputer 실제outputnames/shape도현재record에는없다. 등록되기전runner또는새strictrawreceipt에서이부분을닫는다. 이는새weights/threshold선택이나모델재fit을요구하는변경이아니다.

preparationstatus/sourceSHA/exact24candidateIDs를fitregistration에봉인하고runner가해당receiptSHA를검증해야한다. `candidate_cap==24`만으로selected와registration의exact집합을보장하지않으므로registrymap/실제selected집합을같이검사한다. registration은아직미작성상태로전달되어이연결을본리뷰가확인하지못했다. baseline replay 허용오차와candidate허용오차·resume거부정책은fit전고정한다.

## 후속 허용범위

DF01 주요 구현 불일치는 닫혔다. 새로운 핵심 가설을 추가하지 않고 actual preparation PASS→exact24family/source/runner registration→baselineET replay→72fit 순으로 진행할 수 있다. aliases와P2검산/receipt한계를등록에남기고strict최종gate에서확인한다. BLK48대조+기존6=54는이미노출된탐색진단이며원TM/P2LOO/EL1에서24후보를자동제외하는근거로쓰지않는다. 전체196후보/상호작용단계/최초미사용seed·layout1회/미시험CH2후속은그대로남는다.
