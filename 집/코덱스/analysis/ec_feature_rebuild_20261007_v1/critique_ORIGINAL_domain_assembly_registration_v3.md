# 원66 도메인 조립 등록 v3 보완 독립 검토

검토일: 2026-10-07. 읽기 전용 소스·등록·합성 receipt와 현재 파일 SHA를 대조했다. 조립/모델/query/채점/평가 정답 숫자/GPU/worker 조작은 실행하지 않았다. 이전 상세 비평은 `critique_ORIGINAL_domain_assembly_registration_v2.md`를 유지한다.

## 판정

AR01의 합성 실행 receipt → auditor/helper 현재 소스 연결 누락은 v3에서 수용 가능한 수준으로 보완되었다. 새 핵심 조립 실행 차단 결함은 발견하지 않았다. 단, producer 원66 전부 완료와 정확 manifest가 실행 전제이며 독립 전체 수치·인과 gate 전 채점 금지는 그대로다.

## 독립 확인

- 실제 등록 `ORIGINAL_DOMAIN24_pipeline_registration_v3.json` 257핀 중 기본환경 256핀 MATCH, 불일치 0이다. pandas 단일 파일은 공식 escalation 읽기 해시 `3f0e93be963b0ccf84860992e2b9c5597132e1fbb51a14ba749a1b4f1b0df14b`로 나머지 1핀도 MATCH 확인했다.
- 새 synthetic receipt의 실행 소스 4개(auditor2/helper1/checkpoint1·2) SHA는 현재 파일 및 등록 pin과 모두 일치한다. 정확11개 check, 중복 없는 길이11, model_fit=false, heldout_truth_read=false, whole_pipeline_gate_passed=false를 실제 receipt에서 확인했다.
- registrar3는 단순 완료 문자열 대신 정확11 check 집합·길이와 실행4SHA의 전체 dictionary 동등 및 현재 등록 pin 일치를 강제한다. 이전 PASS와 다른 현재 파일을 기계적으로 연결하는 문제가 닫혔다.
- assembler3 전체 텍스트는 assembler2에서 등록 파일 포인터 v2→v3, 출력 root v2→v3를 치환한 결과와 정확히 같다. 고정 가중치·시드·24family·clip/shrink/SG2 및 완성132파일 경계는 변경되지 않았다.

## 유지할 범위 한계

합성 감사는 파일의 실행 후 SHA를 기록한다. 이것은 실제 실행 객체의 `__file__`/function identity 및 코드 실행 중 변경에 대한 원자적 관측까지 입증하지는 않는다. 다만 이번 변경은 신선한 실행 receipt와 현 등록 pin을 직접 연결한다. 실제 조립에서는 기존 11개 모듈 actual 경로/SHA 검사와 source 재검사를 반드시 통과해야 한다.

11검사는 저장·재개 보존 검사다. 실제 두 process 경쟁·process crash, 모델 수치, candidate 전수 산술, 미래 입력 교란, 역사적 GPU 동일성, 원66 전체 모델의 성능을 증명하지 않는다. 이전 v2의 leaf symlink/부분 lock 쓰기/연속 검사 snapshot 한계도 사라진 것으로 해석하지 않는다.

원66 raw5346/PFN528 manifest 확인 전 feature/label 로드를 금지하는 코드와 모든 whole=false 표시가 유지된다. baseline 각 행×3seed SG2 source audit만으로 candidate·전체 pipeline causality를 승인하지 않는다. 현재 조립 등록 완료는 전체 목표 완료나 후보 채택/제출 승인에 해당하지 않는다.
