# 등록5 추가 차이 독립 점검

2026-10-07. 등록4→등록5 소스 diff 및 현 SHA만 점검했다. 등록/모델/예측/정답/채점 실행은 하지 않았다. 현 등록5 SHA는 `7ed100a70bfcf55c5aa0c6062b7b96da62bd6defbb3491247aa9a17397bd825c`다.

추가 수정은 설명과 일치한다. runtime의 target_values_read=false, statcheck의 target_values_read/model_fit=false, negative 감사의 model_fit/heldout_truth_loaded/full_model_score_gate=false를 정확히 강제한다. integrity의 python_actual을 실제 runtime capture environment와 대조한다. extras 병합은 이미 상속한 동일 절대경로 pin이 있으면 같은 hash만 수용하고 충돌을 거부한다. 등록5 및 upgrade5 자신도 pin 목록에 추가된다. 이전 receipt→감사 소스 연결과 시작의 -O RuntimeError는 유지된다.

**ORR01의 extras 충돌 잔여 권고도 닫혔으며 새 핵심 blocker는 없다.** 등록4의 실제 -O 거부는 부모가 수행한 증거이며, 등록5는 같은 차단 소스를 유지한다. 여기서 등록5의 실제 실행 PASS를 주장하지 않는다. runtime19핀의 승인된 읽기 검산 근거는 비평 v2와 같다.

다음 실행 등록기는 등록5이고 출력은 아직 생성되지 않은 `DOMAIN24_original_raw_fit_registration_v3.json`, 소비자는 rawrunner3다. 이를 명확히 기록하면 버전 이름 차이는 기능 결함이 아니다. 전체66/396 준비와 독립 준비검산을 실제 통과하기 전 fit0 유지, 이후 raw 완료만으로 full-model 채점 허용 금지, 원24 전부 수행·고정 통계·최초 미사용 1회 규칙은 그대로다.
