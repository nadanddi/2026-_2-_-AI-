# 원66 산술 driver3 및 choice guard 독립 계획 비평

2026-10-07. driver3/guard1/auditor1 소스, 실제 저장 합성 receipt와 현재2SHA를 읽기 전용 대조했다. driver 등록·실행, 모델/query/정답 숫자/채점/GPU/worker 조작은 하지 않았다.

## 판단

AD01(폴드 재개·single-writer)과 AD02(choice snapshot 내부 증거)는 source 수준에서 수용한다. 새 핵심 산술 실행 차단 결함은 발견하지 않았다. AD03은 신규 registrar 및 실제 등록이 아직 없는 단계이므로 실행 전 등록 증거 검토가 남는다. 조건부 Decimal PASS와 전체 pipeline gate는 계속 분리해야 한다.

## AD01 보완

driver3는 storage.single_writer로 검증 작업을 단일 writer로 제한한다. 모든 fold를 fresh 재계산하여 exact 객체로 기존 arithmetic receipt와 대조하고 기존 바이트를 보존한다. 마지막 result도 persist_complete로 엄격 재개하므로 기존 report가 있는 경우 전수 계산 이후 무조건 FileExistsError가 발생하던 v1 문제는 해소된다.

정확75 tag/fold×4stage와 전체 비교수, fold receipt66SHA, producer 출력/root·fold completion, assembled132/complete 및 sources를 끝에 재검사한다. receipt는 등록SHA와 해당 assembled prediction/auditSHA를 담으며 최종 result는 assembly completionSHA 및66 receiptSHA를 연결한다. 현재 방식은 중단 시 계산을 건너뛰는 최적화가 아니라 완료 fold를 fresh 재검사하며 안전하게 재개하는 설계다.

root singlewriter/부분 lock 쓰기/순차적 파일 snapshot의 기존 저장 helper 한계는 남는다. 이것을 실제 두 process 경쟁·crash 복구 시험이 완료된 것으로 부르면 안 된다.

## AD02 보완과 실제 합성

guard1은 exact snapshot 필드 집합, status/RAW_PASS scope, exact boolean flags, ID순서/choice coverage, 정수 source_checks=3×rows, finite/nonnegative source_maximum_difference<=1e-12, expected5 source mapping, canonical choice digest를 검사한다. 기존 checkpoint digest의 sort_keys/ensure_ascii=False/compact separators와 일치하며 guard가 allow_nan=False로 비정상 숫자를 더 엄격히 거부한다.

실제 합성 receipt는 positive1/invalid19(중복0), model_fit/heldout_truth_read/wholefalse를 기록한다. 실행 auditor1/guard1 현재2SHA가 모두 일치한다. 누락·추가 필드, scope/status/flags bool-int 차이, count·오차 오류, digest·source·row coverage·payload tamper 거부가 source와 receipt에 대응한다. 합성은 정상 선택 추론이 아닌 메타데이터 검사다.

guard가 choice 내부 active/reference_day/level schema를 단독으로 검사하지 않는 것은 driver3가 곧이어 arithmetic.verify의 exact choice schema·원day activation·train membership·finite level을 전수 검사하므로 현재 통합 경로에서 허용 가능한 책임 분담이다. guard 단독 PASS를 완전한 choice 유효성/선택 정확성으로 사용해서는 안 된다.

## 실행 전 남은 등록 조건

등록기에서 SG2_choice_sources_sha256의 정확5 path를 현재 production plan의 SG2 source5집합으로 직접 구성하고 assembly3 전이핀과 동일한 SHA인지 확인해야 한다. guard의 len5와 mapping 동등만으로 기대한5경로의 정당성이 자동 증명되지는 않는다.

새 registrar는 전체 assembly3 등록/pins, raw/PFN 등록·registry, driver/library/completion/storage/guard import source, 합성 arithmetic3SHA 및 choice2SHA의 실제 receipt 연결, exact24family/3seed/4stage/tolerance·failclosed 정책을 봉인해야 한다. 현재 driver는 ownSHA와 actual4 module.__file__/sourceSHA를 강제하며 신규 등록이 이 계약을 채워야 한다. 아직 실행된 등록으로 주장할 수 없다.

## 유지할 한계

bound와 level는 여전히 저장값에 조건부다. 학습 label bounds의 독립 min/max, 참조 level 기원·fresh 선택, 모델 matrix/y/cache 및 input causality는 별도 전체 verifier가 책임진다. 모든 row×stage 비교와 ±1e-12 threshold 거부를 유지해야 하고 actual boundary행을 임의 제외하거나 tolerance를 완화해서는 안 된다.

saved baseline/candidate final↔post_SG2_clip 연결은 유지된다. result/root complete wholefalse/no-heldtruth/no-model 표시도 유지된다. 이번 검사로 실제 원66 산술 완료, historical GPU equality, 원 검증기 성능, 후보 채택 또는 전체 목표 완료를 주장할 수 없다.
