# 계획 독립 비평 v1 — 2026-10-07 집 코덱스

판정: **자료 진단·특징 구현은 진행 가능, 정식 개선 비교와 최종 판정은 아직 열 수 없음.** 사용자 변경 목표인 TM111·P2LOO·EL1, 계절 v2+DP1+PFN 0.4+SG2, 24/24/배치256, CPU 전용을 우선했다. 종전 DIAG/A/B 기준을 강제하지 않았다. 신규 모델 학습·채점은 수행하지 않았다.

읽은 근거: PLAN_v1.md, preregistration_v1.json, feature_candidates_v1.csv, fold_registry_v1.json, historical_baseline_recheck_v1.json. 보완 근거로 source_contract_v1.json, PROGRESS.md, bootstrap_v1.py/v2.py, 기존 카탈로그 관련 항목을 읽었다. 카탈로그 전체 독해 완료를 주장하지 않는다. 아래는 결과 효과 검증이 아닌 계획·등록 상태 비평이다.

## C01 · P1 · 후보 단위와 24개 제한이 불일치

현재 등록은 도메인24, 문헌24, 나머지134로 합182개다. 하지만 도메인 한 행의 parameters에 windows5개·lags5개가 있고 현재값·1차/2차차분도 설명에 포함된다. 한 행이 한 묶음 입력인지, 각 window를 별도 실험하는 것인지 정의되지 않았다. 따라서 24후보 상한과 실제 시험 수, 다중비교 분모가 확정되지 않았다.

개선: family_id와 variant_id를 분리하고 입력 열 목록·수식·창·시드·구성원·적용 구간까지 한 variant의 명세를 고정한다. 도메인24가 묶음24인지 독립 대조24인지 실행 전에 명시한다. 묶음이라면 후속 제거시험도 별도 variant로 기록한다. 검증: 생성된 variant 수 및 stage cap 준수 assertion, ID 중복/명세 중복 탐지.

## C02 · P1 · 원열 쌍은 상호작용 전수 등록이 아님

raw_pair91행은 columns 두 개뿐이다. 이는 두 원열을 함께 넣는 실험이며 곱·비율·차·교차 지연·전환 반응을 시험한다는 뜻이 아니다. raw_temporal14행에도 구체적 변환이 없다. 파생군끼리 조합, 원열×파생, 환경×운영×시간, 보조 온도·전체온실 학습·row_id 인코딩 후보가 실제 등록 목록에 없다. source_contract의 허용 자료 목록만으로 해당 활용을 시험한 것이 되지 않는다.

개선: 후속 registry에 기본 상호작용 문법(product/difference/safe-ratio/lagged-product/event-response)과 원열·파생군 조합, 시간/온실/계절 조건부 효과를 별도 등록한다. 처음부터 근거 없는 3중항을 모두 fit하라는 뜻은 아니다. 범위와 미실행 수를 보존해 단계적으로 확장한다. PF1/PF2는 delegated 후보 ID와 수신·감사 상태를 넣어 중복 없이 전체 커버리지에 포함한다. 온도 보조/anchor/교차적합, 49온실 표현 학습, row_id 경로도 대기 후보로 등록한다.

## C03 · P1 · 달력의 순환 검증과 제거 정의 누락

계절을 특징·이웃 선택·일차 연결·블록 단위에 같이 쓰면 복원된 날짜가 자신을 검증하는 순환이 생길 수 있다. 지금 원열 ablation은 '간접 경로 제거를 구분'한다고만 쓰며 season·SG2·표준화·참조검색까지 제거하는 규칙이 없다. 외기 열을 direct 입력에서 제거하면서 계절/SG2에 남기면 그 원자료가 불필요하다고 결론 낼 수 없다.

개선: direct-column ablation과 full-source ablation을 별도 variant로 정의한다. 달력은 fold train만 fit하고 query prefix로만 연결하며, 연결 평가에는 독립 알려진 지표 또는 별도 보류를 둔다. 달력·출처를 정답으로 평가한 탐색은 오라클임을 명시한다. 검증: 각 특징/후처리의 source lineage와 fit/query 시각을 기록하고, 외기 제거시 모든 후손 경로가 제거되는지 감사한다.

## C04 · P1 · fold registry는 query ID만 고정함

66fold의 각 객체에는 validator/fold/query_ids만 있다. training_policy의 ±1 제외 문구만으로 모델 학습·달력 fit·학습 이웃·PFN 문맥·온도 보조·중첩 OOF의 실제 ID가 결정되지 않는다. EC·온도 양 정답 숨김은 올바른 원칙이나 현재 구현 증거는 없다. 같은 외기 복제일/추정출처 누수 감사 역시 문구만으로 해결되지 않는다.

개선: actual fold builder를 먼저 만들고 train/anchor/calendar-fit/context/inner-query/hidden-label ID를 각각 저장한다. baseline과 candidate가 같은 정의를 쓰는지 검증한다. 학습행 anchor 특징은 자기 EC·온도 정답과 내부보류 정답이 빠지도록 교차적합한다. outer query를 제거한 뒤 inner 특징·달력을 다시 구성한다. 실패시 fit 차단 assertion을 둔다.

## C05 · P1 · 기준선 CPU 재현은 수치와 처리 동등성을 분리해야 함

historical_baseline_recheck는 WT2 sg4 저장 출력의 산술 재계산이며 이를 제한적으로 표시한 점은 맞다. 하지만 GPU 산출 PFN을 CPU에서 재학습/추론할 때 'exact'의 허용 오차·문맥·가중치·버전·후처리 순서가 봉인되지 않았다. CPU PFN을 빼거나 다른 근사모델을 쓴 것은 지정 baseline 재현이 아니다.

개선: 각 R3 출력, PFN 문맥별 출력, 혼합 전후, shrink/clip/SG2 단계 출력을 비교하고 허용 오차를 미리 등록한다. CPU 수치차가 있으면 CPU baseline을 별도 version으로 명시하고 기준선과 후보를 동일 CPU 조건으로 재학습한다. stored GPU output을 CPU 재학습이라고 부르지 않는다. 전처리부터 최종 예측까지 future/other-farm/order/single-query 불변성을 검증한다.

## C06 · P1 · 처음 쓰는 배치·시드는 처음 보는 정답을 만들지 않음

최종 배치와 시드는 아직 봉인 전이다. 신규 fold 배치가 기존에 반복 관찰한 400일 정답을 재배치한 것이라면 독립 확인이라고 부를 수 없다. TM 자체도 역사 test 전체일 유사성 선정에 따른 고정 진단이며 그 한계를 명시한 것은 타당하다. 최종 k=finalist 수의 보정은 확인시험의 비교 수 보정일 뿐 탐색의 반복 정답 사용을 지우지 않는다.

개선: 시드·배치 사용 이력과 **행 정답 노출 이력**을 별도 감사한다. 미관찰 정답 holdout이 없다면 최종 1회 판정을 '새 재학습/배치 강건성 확인'으로 제한하고 독립 신규 데이터 검증이라고 표현하지 않는다. 첫 확인 전에 finalists·k·배치·시드·bootstrap block 정의·최종 조합을 봉인하고 탈락 후 같은 판정을 재개하지 않는다. TM/P2LOO/EL1의 배치 변경 방법과 TM111 유지 여부를 먼저 명세한다.

## C07 · P2 · 문헌과 과거 실패의 차이는 현재 표제로만 존재

문헌24개는 4family×6window로 만든 대기 자리이며 출처 URL/논문/근거 문장/기작/측정값 대응이 없다. WAIT_SOURCE 상태인 점은 타당하지만 이미 '문헌 효과 후보24'가 확보됐다고 표현하면 과장이다. 카탈로그 번호와 '짧은창/새기준선'만으로 기존 구현과 실질 차이를 증명하지 못한다. 특히 D06과 L07~12, D02와 L13~18, D14와 문헌 temperature_flow는 겹칠 수 있다.

개선: activation 전에 original code feature names/formula/window/model/validator와 새 명세를 비교한다. 동일 후보면 duplicate alias로 재사용하며 stage source annotation만 추가한다. 논문은 해당 기작이 EC 측정에 직접 연결되는지, 센서 보상/염농도/증산 연관인지 구분하고 효능 주장과 후보 설계 근거를 분리한다. 커튼 0=닫힘 같은 방향·단위는 실제 사용자 자료의 위치를 붙여 검증한다.

## C08 · P2 · 짧은 시간 창의 경계 처리가 미등록

동일 기록일 창을 우선하는 방향은 맞다. 하지만 h=0/1에서 lag/2차차분/6h평균을 어떻게 정의할지, 시간 누락·NaN·첫 전환 이전·지속 상태의 시작점·희소 fog/CO2 이벤트를 어떻게 처리할지 없다. 작은 시각별 효과를 24h 행 RMSE 하나로만 보면 농장/초기시각의 피해가 숨는다.

개선: 과거 관측 수·availability flag, 최소 유효 표본, missing 처리, 0시 reset, run/event-age 정의를 수식으로 고정한다. 0/1/2/3/4/5/6/12/23시 prefix 합성 사례와 실제 행 재생으로 확인한다. stage 보고에는 시간별 효과와 이벤트 coverage, 일반/고EC 손실이 동시에 필요하다.

## C09 · P1 · 중단·재개는 약속이며 실행 장치는 아직 없음

PROGRESS와 bootstrap은 등록 파일을 만든다. checkpoint schema, atomic writer, stale PID 판별, partial artifact 보존/거부, source/env mismatch 검사, prediction 재채점 receipt를 실행하는 runner는 아직 없다. '다시 시작할 수 있게 장치' 요구를 구현 완료로 보고할 수 없다.

개선: 첫 장기 fit 전에 task state machine(REGISTERED/RUNNING/PARTIAL/COMPLETE/VERIFIED/FAILED), manifest/receipt schema, 재개 command를 구현한다. 의도적 중단·잘못된 SHA·누락 출력·PID 재사용·순서/ID 오염을 넣어 거부와 복구를 시험한다. immutable artifact와 PROGRESS 갱신을 분리한다.

## 진행 순서 권고

지금 바로 C01/C02/C04/C08을 등록 v2와 feature generator로 구체화하면서 C07 자료를 읽을 수 있다. 첫 정식 fit 전에 C05/C09 실행 검증을 끝내고, 확인 판정 전 C06 봉인을 끝낸다. stage cap과 CPU 제한을 지키며 C01 variant 정의가 확정된 도메인부터 실행한다. 결함별 issue ID→변경 파일→검사 결과→잔여 한계를 연결해 재비평을 받는다.

최종 해석은 단일 중요도 서열보다 재학습 추가/제거 ΔRMSE·시드×검증기 방향·불확실성·조건부/중복군을 함께 제공해야 한다. 작은 단독 효과를 무의미로 기각하지 말고 조합 실험 대기를 남긴다. 아직 어떤 신규 특징의 EC 개선도 이 비평으로 증명하지 않았다.
