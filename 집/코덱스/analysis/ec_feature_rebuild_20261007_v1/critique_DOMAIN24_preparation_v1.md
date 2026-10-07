# 도메인24 준비 소스 독립 비평 v1

2026-10-07. `domain_features_v1.py`, `prepare_domain_BLK_v1.py`, fast_features_v2/causal_features_v1, candidate CSV v4, 원 model.ops_day, 관련 카탈로그 및 사용자 변수 코멘트를 읽었다. 모델 fit/정답 채점/GPU 실행 0. 본 확인 시 DOMAIN24_BLK_preparation_v1.json은 아직 읽을 완료 산출물로 확인되지 않았다. 코드 예상 검사를 실제 PASS로 보고하지 않는다.

## DF01 — 차이 설명과 실제 feature 집합 일부 불일치 (P1, fit 봉인 전 정정)

24가족 selector는 origin 현재열 또는 `origin+'__'`로 시작하는 열 전부다. 실제 이름·열·operator를 기준으로 새 registry/CSV 설명을 고정해야 한다.

- D05 humidity_gap에는 습도차의 현재/lag/rate/d2/rolling만 있다. CSV의 '구동기 전환과 결합'은 구현되지 않았다.
- D11 co2_dose_response는 act_co2>0일 때 Δin_co2를 남긴 masked series 및 그 dynamics다. '구동기 전환별 분리'나 공급 event별 정렬/조건별 분리 feature는 없다.
- D09 fog_vpd는 act_fog×VPD와 그 시계열 dynamics다. 명시적인 전환시각에 정렬한 전후 환경 반응을 구성한 것은 아니다.
- D22 fog_event는 act_fog 현재/dynamics 및 상태 event5열을 선택한다. CSV의 '전환 후 환경 반응'을 별도 constructed interaction으로 구현하지 않았다. baseline 환경열과 ET의 상호작용 가능성은 이 이름의 정의와 구별한다.
- D13은 CSV에1/2/3/4h라고 되어 있지만 실제1/2/3/4/6h다. 같은 WINDOWS를 모든 가족에 적용한다는 등록 설명으로 맞춘다.

해결은 지금 결과를 보지 않고 설명을 실제 코드로 좁혀 새 이름으로 저장하는 것이다. 미구현 전환·상호작용을 자동 추가하여 가설을 확장하는 것은 별도 등록이 필요하다. 현재 설명 그대로 예측72fit을 봉인하면 기존 기각과 다른 가설을 무엇으로 시험하는지 부정확해진다.

## 이벤트 가족의 실제 범위

D17~23은 `act_*__` prefix이므로 기본31 dynamics 외에도 on_run/off_run/onoff_switch/since_value_change/value_change5열이 포함된다. event5열이 누락된 것은 아니다. switch는 >0 상태변화이고 value_change는 실제 값 변화다. age는 마지막 값 변화 후 시간이며 record 최초 변화가 없으면 NaN이다. 결측값/시간 gap에서 run/event state를 reset한다. 현재 grammar는 event 전후로 VPD/온도/습도 변화량을 붙인 것이 아니라 actuator 자체 이력이다. 두 의미를 합쳐 표현하지 않는다.

D24 sealed_run은 환기==0 AND 팬==0의 연속시간이다. baseline OPS의 seal_run은 환기==0만 보고 결측을0으로 처리하므로 동일 feature라고 볼 수 없다. 새 sealed_run은 결측/gap/day 경계를 더 엄격히 다룬다. 다만 여기서 물리적으로 밀폐된 greenhouse라는 확정은 불가하며 입력조건 대리다.

## DF02 — 전체 grammar와 실효 추가 열을 명확히 기록 (P2, runner 등록 권고)

일반 origin 현재+5lag+5rate+d2+5window×4operation=32열, actuator family는 추가 event5열로37열이 source상 예상된다. maskedCO2 두 가족과 sealed_run은 기존 base current에 dynamics31개씩 더한다. 이는 별도 best-window 탐색이 아니라 한 가족 전체창을 동시에 넣는 안이다. exact selected/additional_ET 열과 shape/dtype/SHA를 runner 전 고정한다. 준비 mapping에 CSV family 외 실제 FAMILIES origin도 기록하면 D13~23의 semantic 이름과 raw origin 연결이 명료하다.

window1의 mean/sum은 current와 정확히 같은 값이고 rate1은1h차분이다. raw origin current가 이미 baseline에 있어 additional에서 빠져도 mean1/sum1 alias가 새 열로 다시 들어갈 수 있다. 이는 정보 추가라기보다 ET의 feature중복·무작위 split 후보 수 변화도 섞인다. '도메인 신규 정보'라는 주장 대신 family conditional effect로 보고하고 logical grammar↔model 실효열 alias를 기록한다. 중복 제거 정책을 선택한다면 지금 전체가족에 일관되게 사전 고정하며 결과 후 좋았던 가족만 제거/추가하지 않는다.

특히 masked response가 희소하므로 imputer가 all-missing 열을 제거하는 경우 실제 fitted width가 달라질 수 있다. 등록 matrix column list와 실제 imputer output names/shape/empty-column 처리를 모델 로그에 연결한다. 복수 family 조합이나 VPD×radiation 등 base에 존재하나 이24 selector에 없는 origin은 이번 시험으로 평가됐다고 볼 수 없다.

## 인과 경계와 DF03 감사 범위 (P2)

day_features는 farm/day를 분리하고 각 rolling은 h−w+1..h, lag는 같은 record의 h−lag만 보므로 full reference+query frame을 한꺼번에 만드는 것 자체가 source상 누수는 아니다. 다른 day/farm 전역 fit은 없고 label도 build_domain으로 들어가지 않는다. BLK는 whole-day query/gap 제거라 train/query가 같은 record에 섞이지 않는 구조다. 원 검증기에서는 MASK 입력 NaN을 동일하게 넣고 삭제/정렬/경계 정책을 따로 확인해야 한다.

준비 소스는 각8block의 양끝 query record×5시점 prefix 및 이후입력 poison을 확인한다(예상160comparisons). 같은 날 미래 poison, order 및 다른farm poison은 좋다. lag/d2가 자정 이전 record로 넘어가지 않는 source도 확인했다. 실제 결과 receipt와 source SHA를 읽어야 PASS가 된다.

새3가족의 independent scalar section은 24시점×5window×mean/std/sum/count만 대조한다. 주석의 'rolling/lag computation'과 달리 lag/rate/d2 독립 대조는 없다. missing-hour 검사는 각새family lag1이 NaN인지3개 확인한다. lag2~6/rate/d2, h0 초기값, 알려진 값 전환·결측·gap 뒤 sealed_run reset을 scalar로 더 대조하면 감사 범위를 정확히 닫을 수 있다. 기존 audit의 base838열 대조는 새93열의 별도 독립대조를 완전히 대신하지 않는다.

## 기존 기각과 이번 차이의 유효성

카탈로그6.116/119는 하루 요약 CO2 대리의 유효일/검증 불안정,6.117/118는 수분수지 및 전날 연결,6.131은0~6h 특정 선형 진단,6.185는 lag3/6의 여러 raw 묶음이다. 이번은 same-record prefix dynamics/상태event와 ET 단일family 추가, 최신cached baseline 및 모든3seed라는 차이가 있다. 재시험 근거는 있지만 이전 단서가 처음 시험된다고 주장하지 않는다. BLK에서 좋아져도 원검증기 방향·본페로니/최초 미사용1회 조건을 통과해야 한다. 기존 source측정 의미가 불명확한 proxy를 CO2 uptake의 인과효과나 실제 수분 flux라고 확정하지 않는다.

## 다음 runner에 대한 판정

DF01 설명 정정과 실제24family exact 추가열 봉인 전 신규fit 진행은 권고하지 않는다. 그 뒤 baseline ET exact CPU replay를 먼저 수행하고 ET만 추가하는72fit, 고정 LGB/MLP/PFN/SG2/clip/shrink 조합을 연결한다. 준비감사 PASS가 모델 fit/imputer/전체pipeline 누수 PASS는 아니다. 새family model의 순서/단일query/future·otherfarm 불변 및 최종mix/후처리 audit가 필요하다.

이미 노출된 BLK는24family×2scope48대조 및 기존6합산54의 탐색가족으로만 기록한다. BLK 실패로24후보를 원TM/P2LOO/EL1에서 자동제외하거나 BLK 좋은 창만 고르면 안 된다.54교정이 전체196후보/조합 탐색의 모든 다중비교를 해결하지 않는다. 이벤트/flow 단일가족24는 모든 시간 변화·상호작용·원자료 조합 전체 완료도 아니다. CH2 source matching은 미시험 후속으로 유지한다.
