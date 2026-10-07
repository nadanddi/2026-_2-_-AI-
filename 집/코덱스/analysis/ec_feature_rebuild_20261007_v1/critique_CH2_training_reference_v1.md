# CH2 학습 reference 준비 독립 비평

2026-10-07. `prepare_ch2_training_reference_v1.py`와 실제 `CH2_training_reference_preparation_v1.json`을 source/receipt로 검토했다. 신규 모델·합성 검사·query 추론·정답 채점은 실행하지 않았다.

## 판정과 실제 확인

학습 reference 준비 자체를 막을 핵심 오류는 발견하지 못했다. 현재 receipt의8개 source SHA256을 PowerShell로 다시 확인했고 불일치0이다. 저장 reference 파일의 현재 SHA도 receipt와 일치한다. 저장된5520학습행/230기록/201링크/17cyclic component, JSON_roundtrip_exact=True 및 fit/score/등록=False는 source의 처리 범위와 맞는다.

이 판정은 학습 참조 생성과 저장 정합성에 한정된다. graph의 물리적 정확성, query source 배정, 실제 누수 감사, 모델 성능이나 후보 채택의 PASS가 아니다.

## 필터와 lineage

- train/query/gap 집합의 비중첩을 먼저 검사한다. train_X와 train_y 모두 row_id가 train에 속하는지 확인한 뒤 숫자로 바꾼다. 중복 train ID를 거부하고 최종 입력/EC ID 집합이 layout train과 정확히 일치해야 한다. 입력은 RAW14, 정답은 sub_ec만 수치 변환한다. sub_temp·query/gap 숫자 사용 경로는 없다. CSV 문자열 및 전체 hash 바이트를 읽는다는 기존 표현은 유지해야 한다.
- 입력 finite 검사는 loader의 입력 loop 직후에는 없지만 `PrefixSourceCH2` constructor가 모든 train raw 값의 finite/None 및 EC finite를 강제하므로 유효한 reference 저장까지 우회되지 않는다. constructor는 full24h·train ID 정확집합·samefarm/one-to-one link·고정 flank 및 query membership을 검사한다.
- independent audit의 status와 query/gap 수치 사용0을 확인하고 그 audit가 지목한 source SHA를 현재 파일과 대조한다. 합성 receipt의48검사와 성능 미평가를 확인하고 해당 source SHA도 검사한다. graph→layout SHA와 layout의 원자료 SHA를 검증한다. 원 graph links를 그대로 사용하며 규모를 independent audit와 비교한다.
- scale은 독립 train-only audit의 farm별 `weather_scales_train_only`에서 받는다. query 기반 refit 경로가 없다. 실제 graph의 cyclic root 개수와 graph cycle 개수의 동일성을 constructor 결과에서 검사한다. 원 배정을 새 최적해로 교체하거나 동률 graph를 재선택하지 않는다.

## JSON replay의 강도와 한계

원 reference를 immutable constructor로 만든 뒤 JSON serialize/deserialize하여 다시 constructor를 통과시킨다. root mapping·cyclic roots·train bounds·common support 및 모든 train hour의 원값을 직접 비교한다. 이는 파일 작성 여부만 확인하는 receipt보다 강한 재현 검사다. `allow_nan=False`도 저장 비유한값을 차단한다.

다만 replay.scales와 model.scales의 명시적 동일성 assert는 없다. scale은 동일한 JSON finite 숫자를 그대로 재입력하므로 현재 경로의 결함 증거는 아니지만, ‘전체 immutable 상태 직접 비교’라는 설명을 엄밀히 하려면 scale 비교를 다음 버전 또는 실제 replay verifier에 추가해야 한다. 현재 비교는 메모리 `encoded`의 roundtrip이며, 파일을 다시 읽어 constructor를 재생한 감사까지 실행한 것은 아니다. 저장 파일 SHA를 확인했으므로 현재 바이트 정합성은 확인되지만 실제 query runner는 파일에서 다시 읽고 digest와 상태를 검증해야 한다.

## 실제 후보 등록 전 유지할 조건

미래 loader는 receipt의 완료 문자열만 보지 말고8개 직접 SHA와 independent/synthetic receipt의 내부 source pins를 전이적으로 검증하고, reference 파일 SHA·canonical payload digest·layout SHA·train ID/full24/graph/scale를 fresh 확인해야 한다. preparation은 source 파일 SHA를 고정했으나 실제 import module.__file__/runtime fingerprint를 기록하지 않았다. 실제 inference runner에서는 source pin과 실제 import 경로를 연결한다. assertion 기반 경계이므로 Python 최적화 모드(-O)를 허용하지 않도록 실행 정책도 고정한다.

original validators에 이 BLK reference 파일을 재사용하면 안 된다. 각 fold의 허용 train 입력·정답과 MASK/기존 purge로 graph와 scale를 다시 만든다. CP01/02가 닫힌 prefix3를 실제 loader가 우회하지 않도록 정확 packet 집합과 소비 ID를 감사한다. N04의3방법·scope·seed·조합 순서·누적 비교 수·원검증기·최초 미사용 판정은 아직 미등록이다. 이 준비 산출물은 등록이나 성능 증거를 대신하지 않는다.
