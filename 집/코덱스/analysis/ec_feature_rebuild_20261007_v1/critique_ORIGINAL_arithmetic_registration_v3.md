# 원66 조건부 산술 실제 등록 v3 독립 검토

2026-10-07. 등록/registrar3/driver3/SG2 source 연결 및 저장 합성 receipt를 읽기 전용으로 검토했다. 원66 replay, 모델/query, 평가정답 숫자, 채점, GPU, worker 개입은 하지 않았다.

## 판정

AD03의 실제 등록 봉인까지 확인했다. 현재 조건부 산술 verifier 실행을 위한 새 핵심 차단 결함은 발견하지 않았다. 실행은 정확 raw66/PFN66/assembled66 completion을 모두 요구하며 whole gate 및 score 허용은 계속 false다.

## 현재 SHA·계보

- 실제 `ORIGINAL_DOMAIN24_arithmetic_registration_v3.json`은 270핀이다. 기본환경 269 MATCH/불일치0, pandas 1핀은 공식 escalated 읽기 SHA `3f0e93be963b0ccf84860992e2b9c5597132e1fbb51a14ba749a1b4f1b0df14b`로 일치 확인했다.
- assembly3 전체257 source mapping을 충돌 없이 상속하고 assembly3 등록 파일 자체도 핀한다. verifier ownSHA는 현재 driver3와 같다.
- arithmetic 실제 합성의 실행3SHA와 choice metadata 실제 합성 실행2SHA는 현 등록핀과 모두 같다. registrar는 각각 status/정밀도·대조수/invalid 길이/poison 또는 양성수와 no-model/no-truth/wholefalse 및 실행SHA 전체 mapping을 대조한다. 저장된 합성의 invalid14/19는 앞 독립 검토에서 중복 없이 확인했다.
- 정확 SG2 source5는 original_sg2_plan1, blk_sg2_refonly1·2, kernel2, `집/클로드/submission14_ec_sg2/sg2post.py`다. 실제 sg1.SG2_SOURCE 정의가 이 마지막 파일이며 plan의 required5와 동일하다. 모든5SHA는 전체 source핀에도 포함된다.

## 고정 정책·재개 경계

등록은 Decimal60/절대오차1e-12와 abs(abs(delta)-.30)<=1e-12의 failclosed를 고정한다. source replay 조사만 허용하고 tolerance 조절을 금지한다. driver/library의 현재 구현과 일치한다. 원66×3seed×(baseline+24)×4stage, raw5346/PFN528/assembled132 명단을 대상으로 한다.

fold마다 fresh 전체 산술을 재계산하고 엄격 같은 객체의 arithmetic receipt와 대조한다. O_EXCL singlewriter, 기존 바이트 보존, 최종66 receipt/source/producer/assembled 출력 및 complete 재해시 정책이 유지된다. 이것은 완료 fold를 검사 없이 건너뛰는 resume가 아니다. source 계획의 실제 4module path/SHA 검사도 그대로다.

등록 model_fit/heldout_truth_read/whole_pipeline_gate_passed/score_permitted 모두 false다. 실제 원66 replay는 아직0이며 실제 결과 receipt/분모/모호경계행이 확인된 상태가 아니다. 이 조건부 산술 등록은 full producer 수치 검증·fresh label bounds/SG2 level 근거·참조 선택 재현·query feature 인과 gate를 대신하지 않는다. 전체 scorer가 truth-read 전에 그 별도 gate를 요구해야 한다.

## 남는 범위 한계

완성 producer/assembled 파일 불변 운영 전제와 storage helper의 순차 snapshot·부분 lock/실제 crash·동시 두process 시험 한계는 이전 plan3 비평 그대로다. registrar가 invalid 목록의 exact집합 대신 길이를 검사하지만 현재 receipt 실제 목록과 실행SHA는 독립 대조되어 있다. 후속 소비자는 현재 등록/receipt hash를 확인해야 하며 완료 문자열이나 count만 별도로 이용하면 안 된다.

이번 등록 검토는 모델 개선·후보 채택·역사적 GPU 동일성·전체 목표 완료를 판정하지 않는다. 원 TM111/P2LOO/EL1 및 최초 미사용 확인과 의미 있는 최종 정리파일 조건은 여전히 남는다.
