# 원66 혼합 전체 게이트: 독립 비평가 요구 설계

2026-10-07 source-only 계획. 코드 작성/모델 실행/query 추론/보류 정답 숫자/성능채점/GPU/worker 개입은 없다. 이 문서가 실제 게이트 통과 증거는 아니다. 완료 producer·assembly·arithmetic가 아직 없는 동안 새 verifier를 실행하면 안 된다.

## 범위와 이미 확인된 증거

대상은 원 registry 66fold(DIAG10=10, P2LOO=46, EL1=10), R3 seed47/1414/6464, baseline+domain24다. TM111은 별도 새 fold가 아니라 DIAG 모집단의 봉인된111일/2664행 scoring subset다. 같은 ID라도 서로 다른 fold에서 다른 train 문맥을 갖는 실행 cell은 구분한다. 전체 query cell 합은 registry에서 직접 계산하고 exact 기대10848과 대조한다.

현재 assembly3는 `features.load_production_fold`로 fresh24 matrix/학습y를 복원하고 `validation.validate_raw/pfn`으로 raw81+PFN4의 contract·imputer median·context·저장 numeric/cache audit를 묶는다. SG2 baseline 각 row×3seed source audit도 있다. Decimal driver3는 baseline+24×3seed=75개4stage와 최종 저장값 연결을 조건부로 검사한다. 이 결과가 실제 생성되면 동일 계산을 새 gate에서 중복 실행하지 않고 source/current artifact SHA와 exact 분모·허용오차를 재검증해 소비한다.

`original_fold_features_v2.py::load_fold`는 원66 prep의 complete66/396 및 all24 matrix signature를 검증한다. 그러나 prep의396은 fold당3개 sampled row의 domain/baseline 각각1번 검사다. 모델 raw5차이/PFN21차이 역시 전수 입력 prefix 인과성 검사가 아니다. 남은 fullgate는 이 부족한 실제 input row coverage와 SG2 모든 candidate source 범위에 집중한다. 독립 모델 refit을 추가하지 않는다.

## G01 — 전제 및 fresh 문맥·학습 label lineage

실제 실행 전 raw66/5346, PFN66/528, assembled132, arithmetic66/root/result의 정확 명단·source/runtime/policy·전체 transitiveSHA를 검증한다. 모든 import 실제 경로/function identity, Python-O 금지, CPU 환경과 기존 등록271 같은 추정수 대신 actual pin 집합을 고정한다. 현재 arithmetic 등록270을 기반으로 새 source를 추가한 별도 등록을 만든다.

각 fold에서 production loader2로 한 번 fresh 문맥을 구축한다. exact ordered train/query/forbidden IDs, 교집합0, reference_inputs/reference_labels=train exact, query labels/sub_temp와 gap inputs/labels 미제공, 원 registry purge/MASK 정책을 확인한다. full numeric query truths는 열지 않는다. input/y CSV 파일 hash는 읽되 numeric labels는 해당 train ID 필터 이후에만 변환한다.

fresh 학습 y의 finite·digest를 기존 details/validated assembly receipt와 연결한다. 저장 training_label_bounds가 이 y의 정확 min/max인지 새로 비교한다(66번). domain의 train rows가 query 값을 사용하지 않았는지도 G03 query poison 후 train matrix SHA 불변으로 확인한다. imputer/reference 통계·PFN cache를 다시 fit하지 않고 이미 묶인 train-only matrix/median/cache receipt를 소비한다.

## G02 — 모든 query 행의 실제 prefix 특징 재구성

`blk_context_v1.py::query_prefix(rid)`의 exact expected집합은 같은 farm의 모든 query rows 중 day/hour<=현재다. 원66에서는 이전 다른 블록 query 입력도 허용하며 CH2의 별도 same-block 제한을 잘못 이식하지 않는다. training references는 공개 학습 입력/정답으로 별도 제공되며 query forbidden 집합에 섞지 않는다.

모든10848 실행 row에서 새 prefix packet을 만들고 ID/farm/time/gap 경계를 검사한다. baseline `prepare_query(ctx,[rid],calendar)`의 FULL_R3 및 FULL/BASE_R3 subset이 saved fresh fullquery row와 NaN동등/오차0으로 같아야 한다. calendar는 `prepare_reference`가 training weather로 만든 고정 table이며 query numeric 값으로 다시 fit하지 않는다.

domain `build_domain(dataframe(prefix))`의 selected map/1187 columns와 현재 rid의 모든1187 값이 full matrix와 정확히 같아야 한다. train reference를 이 prefix용 domain frame에 넣어서 현재 query 값 차단을 우회하면 안 된다. 각 family 추가열 및 baseline열 조합은 등록map과 exact다.

CPU 병목을 줄이는 안전한 선택: domain2는 farm/day group별 계산이므로 현재 farm/day의0..h만 입력한 재구성으로 바꿀 수 있다. 하지만 이 최적화는 새 코드/source로 봉인하고, full prefix 대비 day-local 결과 동일성 및 그룹 분리 조건을 독립 확인한 뒤 적용한다. baseline도 동일성을 확인하지 않은 채 day-local 축소하지 않는다. 전수 row를 sampled3로 줄이고 full coverage라고 부르면 안 된다. prefix별1187값 체크 수는10848×1187이며 실제 receipt에서 계산해 보고한다.

## G03 — 봉인할 query-input poison 배치

성능값을 보지 않고 fold×farm 각각 queryday first/middle/last와 hour[0,1,5,6,12,23]을 선택해 unique target 집합을 봉인한다(최대66×2×3×6=2376, farm/day 부재·중복은 구조만으로 처리). h0/23, 자정 날짜전환, SG2 h5/twin 전후를 반드시 포함한다. 추가로 각fold earliest pass1/pass2 존재 시 target을 구조만으로 포함해 RAW_PASS 활성/비활성 범위를 보존한다. 실제 명단·분모를 등록한다.

각 target은 독립 세 모드로 검사한다: A 같은farm 미래query만, B 다른farm 모든query만, C A∪B. 다른farm training reference는 변조하지 않는다. future samefarm는 현재 h+1뿐 아니라 이후 날짜도 포함하며 과거 query는 보존한다. fixed extreme/NaN 두 perturbation 패턴을 사전고정하되 현재 prefix의 실제 missingness를 바꾸지 않는다.

각 모드에서 fresh query context를 만들고 raw query value를 바꾼 후 calendar/reference統計·train matrixSHA 불변, current prefix packet/기준선 특징/domain1187 current row exact 불변을 확인한다. reverse query storage 순서도 별도 고정 배치로 검사한다. fullmatrix를 poison해놓고 원prefix를 그대로 반환하는 mock만 검사하면 production path 감사가 아니다.

gap 및 heldout-label 차단은 loader의 ID-before-float 동작을 synthetic file/provider로 확인한다: gap input/양label, query 양label에 변환 시 예외 sentinel을 넣어 numeric 호출0으로 거부/무시되는지 검사한다. 실제 평가 정답을 읽어 sentinel을 만들지 않는다. 여기서는 단순 poison0diff보다 numeric 소비 경계가 중요하다.

**파싱 한계:** 기존 BLKContext는 전체 query input을 float 파싱하여 `_query`에 저장한 후 prefix API로 제한한다. 따라서 gate는 특징/fit/예측/SG2 계산에서 미래query를 사용하지 않았음을 검사하지만 ‘미래query를 어떤 숫자 처리에도 넣지 않았다’는 강한 명제를 현재 producer에 대해 인증할 수 없다. 규정 해석상 원값 숫자 파싱까지 금지라면 이 게이트로 보완했다고 주장할 수 없고 producer 경로 변경·새 baseline 증거가 필요한 별도 사안이다. 이를 limits에 명시한다.

## G04 — SG2 choice·level fresh 기원 및 모든 후보 source

fresh train 문맥의 새 `OriginalSG2Plan`/`RefOnlySG2`를 구축한다. choice cache는 처음 비어 있어야 한다. allquery rows에서 zero prediction prefix로 실제 선택을 새로 만들고 저장 choice의 exact flags/reference_day/level과 대조한다. 원day<179는 identity/참조선택0, day>=179 선택없음 fallback도 분모에 포함한다. 선택일은 같은farm train의 distinct 완전24h record이고 level은 해당 train sub_ec24개 평균과 exact/사전 source tolerance로 대조한다. 참조 scale/calendar/signature가 fresh training inputs만으로 만들어졌는지 구조와 source pin을 확인한다.

`original_choice_snapshot_validation_v1.py`를 소비하고 scope/status/digest/count/max/source5 flags를 확인한다. fresh 실제 선택 전수 대조는 이 metadata 검사와 별도로 필요하다.

각 row,3seed,baseline+24에서 해당 saved `pre_SG2_clip`의 같은day0..h prefix를 가져온다. fresh ref-only SG2 source `predict_one(rid,prefix,BLK_RAW_PASS)`의 보정 전후 및 diag를 저장 stage·choice와 대조한다. candidate24×3 전수 source가 새로 필요한 범위이며 baseline source는 기존 exact source receipt를 재사용하거나 같은 fresh fold source call에서 함께 확인한다. 총 source scalar coverage는10848×75=813600이며 baseline32544와 candidate781056을 따로 보고한다. source 반환 전후clip 차이를 혼동하지 말고 source보정값을 clip해 post_SG2_clip과 비교한다.

G03 각 poison target은 **빈 choice cache의 fresh plan**으로 다시 선택하고 correction을 검사한다. 이미 선택 캐시를 채워 놓은 plan의 ctx만 poison하면 선택 불변 검사가 순환적이다. immutable training 준비를 재사용할 경우 train 구조/tensor/hash와 query choice cache 분리를 명확히 등록한다. 새로운 모델 fit은 하지 않는다. 모든75candidate의 correction을 poison 배치에서 대조하되 계산량 축소는 후보 성능을 보고 선택하지 않는다.

## G05 — 결과 게이트와 재개

각fold strict fresh receipt에는 exact expected input-prefix rows/1187열, poison target×모드×패턴, source row×seed×variant 명단/count, bounds·trainy/선택 level·choice digest, consumed producer/assembly/arithmeticSHA, import/currentsource/runtime, 각오차최대/finite 및 whole 조건을 담는다. source 산술 허용오차는 기존1e-12, saved model receipt1e-6 등 기존 정책을 유지하며 새로운 임의 tolerance를 혼합하지 않는다. 모호threshold는 기존 failclosed를 유지한다.

O_EXCL 단일writer와 fold별 fresh 재대조/바이트보존 resume를 적용한다. root complete는 exact66 fold receipt·예상 분모와 전수 source/output/completeSHA를 끝에 재검사한 뒤 생성한다. truth-read/score 허용 게이트는 이 fresh evidence를 전이 SHA로 고정하고, scorer가 실제 정답 로드 전에 모두 재검증해야 한다. 부분fold PASS를 wholetrue로 만들지 않는다.

## 허용하는 최종 주장과 금지하는 확대

모든 전수/등록 poison 검사를 통과했을 때의 주장은 ‘현재 고정 CPU cached recipe의 저장 원66 예측이 등록된 train-only 문맥/현prefix 특징, 후보별 SG2source 및 독립 산술 증거와 연결되어 있다’이다. 모델 raw/PFN의 행 독립성은 기존 source 구조 및 유한21/5 numeric audit와 함께 평가한다. 모든 가능한 adversarial query에 대한 수학적 증명이나 모든row의 모델 재추론을 실행했다고 주장하지 않는다.

독립 모델 refit, 역사적 GPU/uncached 제출 동일성, BLK의 성능 개선, 원66 성능채점·채택·최초미사용 판정은 이 게이트의 범위가 아니다. 기존 통계/전체 validators와 완료 후 의미 있는 선별정리파일 조건은 그대로 유지한다.
