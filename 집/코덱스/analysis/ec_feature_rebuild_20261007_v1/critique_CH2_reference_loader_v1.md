# CH2 저장 reference loader 독립 검토

2026-10-07. `ch2_reference_loader_v1.py`와 실제 `CH2_training_reference_file_replay_v1.json`을 읽기 검토했다. reference의 값 재계산·query 추론·모델·채점·합성 코드를 실행하지 않고 파일 바이트 hash만 새로 비교했다.

## 판정

직전 preparation 비평에서 남긴 파일 실제 재독해·scale 명시 비교·실제 import 경로·최적화 실행 금지·전이적 source 검사 항목은 이 loader에서 보완됐다. 이 준비/replay 경로의 새 core blocker는 발견하지 못했다. 현재 receipt에 등록된12개 직접/전이 source SHA 불일치0, 저장 reference SHA·loader 자체 SHA·import source SHA도 각각 일치한다.

## source에서 확인한 보완

- `load_reference` 진입 시 `__debug__`와 `sys.flags.optimize`를 확인하고 -O를 명시 RuntimeError로 거부한다. 이후 assertion 기반 경계가 비활성화되는 실행을 허용하지 않는다.
- 실제 import된 prefix module의 `__file__` 절대경로를 own folder의 prefix3 파일과 비교한다. 이 파일은 preparation source pin에도 포함되며, 실행 receipt에 실제 경로와 source SHA를 기록한다.
- preparation/audit/synthetic receipt가 지목하는 source를 합치며 같은 path의 hash 충돌을 거부하고 각각 현재 SHA를 확인한다. 끝에서도 다시 source SHA를 검사한다. 단순 완료 문자열만으로 통과하지 않는다.
- reference 파일 SHA를 먼저 검사하고 실제 파일을 읽어 JSON decode한 payload의 canonical digest를 preparation 기록과 비교한다. payload source map은 preparation map과 정확히 일치해야 한다. layout SHA와 원 graph links의 정확 동등성을 검사한다.
- record 중복 key를 거부하며 immutable constructor로 full24h/정확 train 집합/finite/graph/flank 경계를 다시 확인한다. graph 규모와 cyclic component 수를 기록과 비교한다. payload scale을 independent train-only audit의 scale과 직접 비교하고 constructor 내부 scale도 다시 비교한다. 파일에서 읽은 모든 원 record 값이 constructor 복사 이후 동일한지 검사한다.

따라서 이 실제 file replay는 직전 메모리 encoded roundtrip보다 강하다. 원 source graph를 새 최적 graph로 바꾸지 않으며, query 선택·보간·점수 호출이 없다.

## 미완료 경계

이번 독립 확인은 현재 파일 hash와 source 검토이며 canonical digest·record/scale 비교를 별도 실행해 재검산하지 않았다. 그 비교의 실제 PASS는 저장 replay receipt의 실행 증거이고, 현재 hash 일치로 source/산출물 정합성을 확인한 것이다.

loader가 생성하는 details의12개 source map에는 **preparation receipt 자체의 raw SHA와 loader 자체 SHA**가 들어 있지 않다. loader SHA는 최상위 실행 receipt에 따로 있다. 현재 준비 replay를 무효화하는 문제는 아니지만 미래 inference 등록/gate는 preparation receipt·replay receipt·loader·reference 파일을 포함하여 봉인하고 재확인해야 한다. 이 출력 map을 ‘모든 관련 파일의 완전한 폐쇄 목록’으로 그대로 쓰면 안 된다. 같은 path conflict 검사는 문자열 key 기준이므로, 미래 일반화된 pin 수집기는 resolve한 절대경로 기준으로 중복/충돌을 검사하는 편이 명확하다.

기존 N04 후보 통계·scope/seed·조합 순서·전검증기·최초 미사용 판정은 여전히 미등록이다. 원검증기에는 BLK 학습 reference를 복사하지 않고 각 허용 train/MASK/purge에서 재생성해야 한다. `FRESH_TRAIN_REFERENCE_FILE_REPLAY_PASS_NO_QUERY_OR_SCORE`라는 제한된 status는 적절하다. 이 PASS는 query 인과 gate·물리적 연결 정확도·예측 개선·바닐라 전수 완결 또는 차단된 실행 우회 근거가 아니다.
