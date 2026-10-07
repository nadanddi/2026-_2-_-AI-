# 등록4 보완 독립 재검토

2026-10-07. 이전 비평 v1을 보존하고 새 `register_original_raw_fit_v4.py` 소스와 저장 계약 SHA만 재검토했다. 실행·학습·예측·정답·채점 0이다.

**ORR01/ORR02의 핵심 결함은 소스상 닫혔다.** 등록4는 preparation/runtime/resume/stat independent receipt의 recorded code SHA를 현 감사 소스와 각각 연결한다. stats 8개 모두 PASS/87블록/8640행, 추가 integrity code·등록·draw SHA/200k 및 negative10 code/guard SHA·전부 PASS를 검사하고 증거를 source pin에 포함한다. 시작 시 `-O`를 RuntimeError로 거부한다. 실제 negative exit1 자체는 부모 보고이며 이 리뷰에서 재실행하지 않았다.

읽기 접근이 거부되었던 pandas 파일은 정식 승인된 읽기 전용 `Get-FileHash`로 확인했다. 현 SHA `3f0e93be963b0ccf84860992e2b9c5597132e1fbb51a14ba749a1b4f1b0df14b`로 저장 runtime pin과 일치한다. 따라서 runtime19개 current SHA는 이 검토에서 **19 MATCH**로 보완된다. 이는 모델 실행 승인을 뜻하지 않는다.

등록4가 출력할 이름은 기존에 생성되지 않은 `DOMAIN24_original_raw_fit_registration_v3.json`이다. raw3의 고정 소비 경로와 맞고 기존 산출물을 덮어쓰지 않으므로 이름 자체는 결함이 아니다. 다만 실행한 등록기 버전은 v4임을 기록해야 한다.

잔여 비차단 권고: extras 추가 시 기존 동일 경로 pin과 충돌 검사를 공통화하면 향후 변경에도 더 명확히 실패한다. integrity/status와 no-fit/no-target 필드도 정확히 검사하면 메타데이터 계약을 더 좁힐 수 있다. 현재 저장 소스·영수증 연결에서 상충하는 실제 증거는 발견하지 못했다.

**새 핵심 raw-fit blocker는 찾지 못했다.** 단, 준비66/396 및 independent preparation PASS가 실제 생성된 뒤 등록4의 전수 SHA 검사가 성공해야 학습을 열 수 있다. 등록4는 이 조건을 계속 요구한다. 현재 상태를 실제 등록/fit PASS로 바꾸어 보고할 수 없다. 원66 전체 PFN·후처리·causal/numeric·독립 fullgate 뒤 정답 parse, DIAG/TM p 교집합과 모든 seed×TM/P2/EL 방향 강제, 최초 미사용 1회 판정은 미래 scorer의 의무로 남는다. SG2 RAW_PASS 및 역사적 기준선 동일성 한계와 raw3의 표본 prefix 감사 한계도 v1과 동일하다.
