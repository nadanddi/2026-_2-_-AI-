# 원66 독립 산술 driver v1 계획 비평

2026-10-07. `verify_original_domain_arithmetic_v1.py` 소스만 읽었다. 아직 driver 등록·실행은 없고 모델/query/정답 숫자/채점/GPU/worker 개입은 하지 않았다.

## 판단

공급된 모델 예측·bounds·SG2 choice에 조건부인 Decimal 산술 검증 범위는 타당하며 새로운 산술 공식 오류는 찾지 못했다. 다만 실행 전 사용자 재개 요구를 반영하는 checkpoint/single-writer 보완이 필요하다. fullgate로 확대할 수 없다는 whole=false 표시는 적절하다.

## 타당한 연결

양 producer exact66 complete를 먼저 검증하고 assembled3 complete의 정확132 prediction/audit 명단을 해시 검사한다. stable_json은 같은 blob의 SHA와 JSON을 묶고 재읽기 바이트 동일성을 확인한다. driver/library/completion 실제 import 경로와 source SHA 검사도 있다.

원 registry query 명단·fold·assembly 등록 SHA 및 prediction↔audit SHA를 연결한다. 각 seed에서 LGB/MLP와 PFN5..8을 고정하고 baselineET+domain24=75 stage 묶음을 모두 독립 재구성한다. 저장 baseline/candidate 최종값이 post_SG2_clip과 같은지도 검사한다. stage75 개수와 각 예상 tag 접근, exact24 candidate key집합을 결합하므로 현재 코드에서는 누락·대체 tag가 조용히 통과하지 않는다. 총 비교수도 registry query row 합×75×4와 일치해야 한다.

끝에 producer 모든 출력/root·fold complete, assembled132 및 complete, 등록 sources를 재검사한다. 고정1e-12/threshold failclosed가 유지된다. truth 파일 로드나 score 함수는 없고 출력에 heldout_truth_read/model_fit/whole_pipeline_gate_passed=false를 유지한다.

## AD01 — fold 재개·single writer 필요

현재는 모든 결과를 메모리에 모은 뒤 단일 report만 x-write한다. 중간 중단 시 완료 fold 증거가 없고 처음부터 다시 읽고 계산해야 한다. 마지막 report가 이미 존재하면 전수 계산 이후에야 FileExistsError가 발생한다. 두 writer도 처음에는 함께 실행될 수 있다.

새 v2는 O_EXCL writer와 fold별 exact fresh receipt를 두고, source/reg/registry/양 producer·assembly complete/output SHA, 정확75 tag·4stage·비교수·finite 최대오차를 기록해야 한다. resume는 현재 파일에 대해 fresh 재계산하거나 동일한 전체 입력 evidence를 재검증한 후 exact 객체/바이트 보존 정책을 적용한다. 최종 전체 receipt는 exact66 명단과 모든 fold receipt 및 consumed artifacts를 다시 묶는다. 기존 v1은 보존한다.

## AD02 — choice snapshot 내부 증거 검사 미완성

driver는 snapshot의 choices, row_ids, source_checks=3×rows만 사용한다. snapshot status/scope/all_choices_present/whole·heldout flags, choices digest, source SHA 및 source_maximum_difference를 검사하지 않는다. assembly manifest SHA는 해당 저장 바이트의 동일성을 보장하지만 이 내부 주장들의 일관성까지 검사하지 않는다.

새 driver는 정확 snapshot schema·RAW_PASS scope·wholefalse/heldoutfalse, all_choices_present=true, choices canonical digest, source-check 최대오차 finite/nonnegative/등록 허용오차 이내와 exact SG2 source pins를 검사할 것을 권고한다. 이 수정도 reference 선택의 독립 재현을 대신하지 않는다. library 자체는 exact choices coverage를 요구하므로 현재 누락 choice가 산술 PASS로 통과하는 문제는 없다.

## AD03 — 실제 등록과 조건부 한계 명시

등록기 아직 없으므로 source chain은 실행 가능한 상태로 봉인되지 않았다. 새 등록은 assembly3 등록 자체와 전체 전이 pins, raw/PFN 등록·registry, driver/library/completion 실제 currentSHA, 합성 auditor2/result 및 실행3SHA 연결, 정확 counts/시드/24family/4stage/tolerance를 포함해야 한다. 자기 sourceSHA 검사만으로 전체 필수 pin membership이 보장되지는 않는다.

현재 bounds는 assembler 저장값, level와 choice는 공급된 snapshot, producer prediction은 저장된 값이다. bounds=min/max(train labels), level의 training label 기원, 모델 input matrix/y/cache lineage, 참조 선택 및 query 입력 인과성은 여기서 독립 재현하지 않는다. output의 limits가 이를 명시하므로 조건부 산술 report에는 맞는 범위다. 별도 전체 verifier에서는 반드시 확인해야 한다.

실제 stage 경계 모호 행이 나오면 전수 PASS를 만들기 위해 누락시키거나 tolerance를 바꾸면 안 된다. 사전등록된 source replay/명시 실패 정책으로 처리하고 모든 예상 row×stage 분모를 보존한다.

leaf containment 초기 manifest 검사 이후 파일 접근·최종 hash는 연속 snapshot이며 동시 교체에 대한 원자적 잠금 증거가 아니다. 완료 producer/assembled outputs를 불변으로 두는 전제가 유지된다. 새 driver에서 완료 객체나 식별 숫자의 exact 타입/schema를 강화하면 bool/int 동등성 등의 메타데이터 허점도 줄일 수 있다.

이 문서는 source 계획 검토다. 원66 실제 산술 완료, 전체 pipeline gate, 원 검증기 성능 또는 전체 목표 완료를 의미하지 않는다.
