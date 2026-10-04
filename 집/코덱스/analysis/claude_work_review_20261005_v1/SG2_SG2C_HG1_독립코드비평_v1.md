# SG2·SG2C·HG1 독립 코드 비평

2026-10-05 집 코덱스. 세 스크립트와 로그, 관련 함수·구조 생성 코드만 읽었다. 다른 AI 실행·수정, 모델 학습, 원시 EC/test/EL1/잠금 파일 읽기와 재채점은0이다. 로그 수치는 작성자 출력이며 여기서 새로 검산한 성능 수치가 아니다. HG1 로그는 진행 중이어서 최종 판정을 하지 않는다.

root의 WV 전역 통계와 purge 불일치 지적은 지지한다. SG2C가 실제 계절v2와 같다는 주장에는 추가로 처리 단계 차이가 있다. 반면 “미래의 학습 참조 자체가 금지” 또는 “SIG 전체가 test로 fit됐다”, “동 classifier를 그대로 예측에 쓴다”는 주장은 현재 코드만으로 지지하지 않는다.

## 1. WV 통계는 reference-only가 아니다

SG2 및 HG1 prepare_structure는 train_X와test_X를 concat한 후, 각 farm의 모든 day<179에 대해 weather mean/std를 계산한다(SG2:43–54, HG1:33–44). fold의 ref나 tr을 인자로 받지 않는다. 따라서 validation 입력 및 test 입력이 z 통계의 모집단에 들어갈 수 있는 코드 경로가 명백하다. 어떤 test 행이 실제 얼마만큼 영향을 주는지는 데이터를 읽거나 재실행하지 않아 정량 판정하지 않는다.

weather 거리에서 공통 mean은 상쇄되지만 std는 상쇄되지 않는다. std가 twin threshold .05와 거리·calendar에 영향을 주므로 “label-free 구조 처리라 fit이 아니다”라는 반박만으로 해결되지 않는다. 전역 test perturbation을 고정 reference/query의 prefix 밖에서 가해도 결과가 불변인지 검증할 필요가 있다. 코드만 보고 효과 크기나 개선의 원인을 test 통계로 확정할 수는 없다.

중요한 구분: SIG는 시간 h까지의 raw signature를 만들고 correction 내부의 RS=reference records에서 mean/std를 계산한다(SG2:145). 현재 query의 0..h 입력을 직접 쓰는 것은 hour-causal 관측이고, raw signature 생성 자체를 fitted test 통계라고 부르면 과도하다. 다만 이 RS의 reference 범위 역시 다음 purge 문제를 갖는다.

## 2. 앵커·calendar의 ref는 모델 tr보다 넓다

SG2:181 및HG1:176의 모델 tr은 validation±1 및lock±1을 제외한다. 하지만 SG2:190/HG1:185의 ref는 labset−vd이며 validation의 인접일 및lock 인접일을 제외하지 않는다. ref는 calendar, SIG z 통계, anchor target 일평균에 쓰인다. 따라서 base 모델의 purged fold를 전체 후보가 유지했다고 할 수 없다. SG2C:59도 같은 reference 범위를 쓴다.

이것은 validation일 자기 y를 직접 쓰는 경우와 구분해야 한다. vd의 anchor는 ref에서 제외되며, p3.prepare의 lab은 locked days를 제외한다(코드46–61만 읽음). lab 전체의 ec Series를 만들었다는 사실만으로 자기 정답 누수를 단정하면 안 된다. 실제 label lookup은 ref·ec·lock 조건을 통과한 G에 대해 수행된다.

비판의 정확한 범위는 “기존 ±1 purge를 만족하지 않는 별도의 reference learner”다. 관측 순서 뒤쪽의 **학습** 참조는 새로운 공식 해석에서 허용될 수 있으므로 미래 training target 사용을 자동으로 규정 위반이라 판정하지 않는다. 반대로 허용된 저장 참조라고 해도 독립 검증의 purge를 소급해서 만족시키지는 않는다. 개선이 인접일 공유 상태 때문인지 여부는 여기서 증명하지 않았다.

## 3. SG2C는 actualv2 재현 검사가 아니다

SG2C:38은 전체 공개 lab의 min/max를 bounds로 사용한다. fold train bounds와 다르며 validation target extrema가 포함될 수 있다. clip이 실제로 발동했는지와 점수 영향은 재실행하지 않았다. 미래 운영에서 모든 학습 labels의 bounds가 허용되더라도 OOF에서 heldout labels로 bounds를 정하는 것은 별개다.

더 중요한 처리 단계 차이는 다음 코드 연결로 확인했다.

- DP1_all의 dp_s는 dc5.r3 결과다. dc5.r3:46은 p3.final을 호출하고, p3.final:43은 shrink 후 tr min/max clip을 이미 수행한다.
- v2_integration_oof의 season_pfn은 run_dc4.py:164의 common.finish(bag,tr,va)다. common.finish:87은 역시 shrink 후 tr bounds clip이다.
- SG2C:53–54는 이러한 완성 출력을 .8/.2로 혼합한 후 다시 shrink하고 전체 lab bounds로 clip한다. 변수명이 rawp인 것과 raw prediction인 것은 다르다.

따라서 SG2C는 actualv2의 raw R3/PFN mix→one shrink→fold clip과 동일하지 않다. DP1 특징 추가도 actual 계절v2의 원 R3와 다르다. shrink 자체는 선형이지만 두 번 적용하면 일반적으로 S²≠S이며, 선행 clip은 혼합과 교환되지 않는다. 실제 제출13과의 정확한 동일성을 주장하려면 그 생성 레시피·raw cache·bounds의 일치가 별도로 필요하다. 제출 파일을 읽거나 만들지는 않았다.

SG2C 로그의 “all seeds helps”는 작성자가 고정한 pass-2 비교의 기술적 결과다. 로그에 DIAG seed-mean p=.0633이라고 적혀 있으며 이를 엄격 통계 통과로 설명하면 안 된다. 작성자 스크립트는 모든3seed 개선만 helps 조건으로 쓰고 p는 보고한다. 이를 작성자 기준 실패라고 소급해 바꾸지도 않는다. Codex actualv2 채택 PASS로 옮길 수 없다는 뜻이다.

## 4. role의 출처를 dong classifier와 구분한다

SG2/HG1은 st_dong_assign CSV를 읽지만 correction에서 사용하는 구조 값은 farm/day/role이다. pB와dong은 쓰지 않으며 R.date도 현재 correction/calendar 계산에서 직접 쓰지 않는다. 따라서 st6의 LogisticRegression이 train+test로 fit됐다는 사실을 SG2가 그 classifier output을 사용한다는 주장으로 연결하면 안 된다.

source-only 추적 결과 st6_dong_identity_v1.py는 role을 st_record_roles에서 복사한다. 이 role 생성은 st4_residual_by_pair_role_v1.py:16–30의 train+test 전체 weather와24시간 인접일 twin 판정에 의존한다. label 결과는 role 파일을 쓴 뒤에 읽으므로 role 값의 target-fit 증거는 없다. 다만 role의 입력 범위·시간 가용성과 산출물 source SHA는 별도로 감사해야 한다.

full_date는 현재 query의 바로 이전 기록을 대상으로 full-day weather를 사용한다. “현재 query의 미래 hour를 직접 쓴다”는 주장은 해당 호출 경로만으로 성립하지 않는다. previous record의 full day는 query보다 이전일이다. 그러나 전역 weather std와 외부 role/table 생성이 prefix 인과성을 보장하지 않으므로 전체 pipeline causal PASS라고 할 수도 없다. ref의 full-day 학습 입력 참조와 query 자신의 prefix 입력을 명확히 분리한 신규 감사가 필요하다.

## 5. 작성자 PASS의 의미와 HG1 범위

SG2 로그에는9/9 seed×set 개선과 DIAG10y seed-mean p가0.0000으로 출력돼 PASS가 적혀 있다. 이는 무시할 수 없는 작성자 기준의 성공이다. 다만20k draw 결과를 소수4자리로 표시했으므로 진짜 확률0이나 raw p의 exact0으로 확대하지 않는다. root의 공개 DIAG 독립 검산은 수치 정확성에 대한 추가 증거이며, 위 코드 범위 문제를 해결하는 증거는 아니다.

작성자 bootstrap은 seed별 검사가 아니라 seed-mean prediction의 SSE를 farm_day//5 cluster로 묶어 전체 cluster를 resample한다. 이는 Codex의 farm별 sorted observed5days·seed별 p/adjustedCI와 다른 정의다. DIAG10y/z의 새 fold 배열도 같은 공개 labels의 재분할이며 새로운 untouched holdout은 아니다. 로그에는 SG2의 EL1 high가 세 seed 모두 악화한 것으로 출력돼 있으나, EL1 데이터를 읽거나 점수를 다시 계산하지 않았다. script의 EL1은 lab의 pass-2 공개일을 묶는 이름이며 sealed40을 독립 개봉했다는 뜻으로 해석하면 안 된다.

HG1은 같은 WV/ref/role 문제를 복제한다. pm≥.9와 두 anchor≥1이라는 gate 자체는 현재 prefix prediction·학습 anchor로 정의돼 query 정답을 직접 gate에 쓰지 않는다. high/normal label threshold는 사후 보고용이다. 그러나 HK0 관측에 따라 설계됐고 SG2 baseline과 같은 공개 자료를 사용하므로 최종 성공하더라도 untouched 확인이나 “실제 고EC를 알아낸다”는 인과 주장으로 확대하지 않는다. 현재 live HG1을 재실행하거나 부분 점수를 평가하지 않았다.

## 규정과 재현 한계

2026-10-05 공식 PDF 원문은 이 검토에서 읽지 못했다. C6.309의 출처 요약과 스크립트 설명만 있는 상태이므로 법적·대회규정 최종 적합성 판정은 유보한다. “test에서 fit 금지”와 “같은 온실 현재·이전 query 입력” 해석이 원문 그대로라면 전역 WV 통계·외부 구조 provenance가 우선 해결할 문제다.

현재 source 파일을 읽은 정적 판정이다. 로그가 이 정확한 SHA의 실행인지, checkpoint가 같은 source/input/fold를 재사용했는지는 별도 영수증이 없으면 보장하지 않는다. SG2/HG1은 파일 존재 시 skip하고 checkpoint 폴더 전체를 concat하므로 code 변경·부분 재사용·다른 fold 산출물 혼입도 manifest로 검사해야 한다. root 판단은 “유망한 별도 진단 신호이나 현 actualv2 채택·적법한 causal pipeline 재현은 아직 입증되지 않음”으로 한정하는 것이 타당하다.
