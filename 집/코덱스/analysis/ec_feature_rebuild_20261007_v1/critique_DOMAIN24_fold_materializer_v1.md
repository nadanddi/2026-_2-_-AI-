# 원66fold 특징 materializer 사전 독립 비평

2026-10-07. `original_fold_features_v1.py`와 직접 사용하는 특징/loader source를 읽고 preparation등록2의 현재109개 SHA를 비교했다. mismatch0이다. probe·query·모델·score를 실행하지 않았고 actual66준비/첫fold probe 완료를 주장하지 않는다.

## 현재 코드 수용 범위

기본 `require_all66=True`는 complete status·66fold·396prefixcheck·등록SHA·정확66개 파일명 집합과 모든 파일 currentSHA를 먼저 요구한다. complete가 없거나 일부 fold만 있는 상태에서는 기본 호출이 통과하지 못한다. 선택 fold도 payload digest·status·등록SHA·validator/fold·정확 train/query/forbidden ID 및 truth/fit/score=False를 확인한다.

선택된 각24family의 fresh matrix는 준비 receipt와 columns/trainSHA/querySHA/allmissingtraincolumns **전체 dict 동등성**으로 비교한다. selected/saved/등록 family key집합과 exact additionalET 열을 검증한다. baseline47열과 extra를 분리하고 기존baseline명은 중복 추가하지 않는다.24개의 exact candidate집합을 강제하므로 BLK rank에 따른 후보 누락 경로가 없다. query/tr row순서와1187domain열도 확인한다.

## 미래 모델 fit 전 반드시 연결할 조건

**OFM01 — probe 출력의 학습 사용 차단.** `require_all66=False`는 명시 preparation probe의 용도로 적절하지만 ctx·train labels·실제 matrices를 True와 같은 형태로 반환한다. docstring과 기본True만으로 이 결과를 fit에 쓰지 못하도록 막을 수는 없다. future model runner는 complete fresh검증을 필수로 호출하고 `details['require_all66'] is True`를 fit 직전에 assert해야 한다. 가능하면 model용 production API에서 False 인자를 노출하지 않고 probe API를 분리한다. 이 자료구조를 받는 모형등록/gate에 complete receiptSHA·정확66file SHA를 연결한다. 현재 probe를 막는 오류가 아니라 미래 model-fit gate의 차단 조건이다.

**OFM02 — 실제 import 및 새 loader pin.** `from blk_baseline_data_v1 import *`와 함수직접import가 실제로 어느 module을 가져왔는지 검사하지 않는다.109pin은 지정된 preparation source의 내용이며 새 materializer 자체는 아직 포함되지 않는다. future model 등록은 materializer·baseline/domain/checkpoint/context/env 및 동적 AST 추출 source의 actualmodule.__file__/SHA를 연결해야 한다. 함수 wildcard 대신 module 명시 import는 경로검사와 dependency 명확성에 도움이 된다. 현재 freshmatrixSHA일치는 특징의 재현에 강한 제약을 주지만 모든 import lineage를 입증하지는 않는다.

## label/MASK·인과 범위

fold loader의 train은 정확 ordered_train_ids, query는 ordered_query_ids, gap은 input_forbidden_ids다. BLKContext는 train EC/sub_temp를 읽고 query/gap label은 읽지 않는다. 학습 EC와 input source는 원자료 hash가 고정됐으며 train labelSHA는 details에 기록한다. temperature label이 공개 학습 reference에 존재하는 것과 EC model에 쓰는 것은 구분한다. `prepare_reference`의 calendar fit은 reference_inputs만 사용하고 query에는 day metadata를 전달한다.

이 materializer는 TEST_X 원파일을 직접 새 MASK로 만드는 코드가 아니다. 기존 등록/registry의 허용train·query·forbidden 및 recorddisjoint 경계를 그대로 따른다. `build_domain`은 train+query 입력을 한 frame에 넣지만 family가 같은record의 현재/과거만 쓰는 전제 아래 학습record와 queryrecord 분리로 query→train 유입을 막는다. 따라서 ‘fresh matrix가 receipt와 같음’은 준비와 재현의 동등성이지 독립 누수 증명은 아니다. 원66registry의 train/test MASK·purge·recorddisjoint와 준비prefix감사를 유지해야 한다. full query를 수치 변환하는 offline preparation 범위와 실제 평가행 인과 특징 범위를 구분하고, 모델의 query normalization/캐시정책은 별도 등록한다.

## 재개·검사 표현

complete 파일의66/396 count와 파일SHA를 검사하는 것은 전체 preparation gate다. 현재 materializer가396prefix검사를 독립적으로 다시 수행하거나66fold 전체를 재구성하는 것은 아니다. 선택fold24freshmatrix를 재구성하며 future model loop가 각fold를 호출해야 전체fresh 검사가 된다. complete 누락을 probe=True 의미로 자동 fallback하는 경로는 없고 False는 명시 인자뿐인 점은 타당하다.

부정확한 runtime을 version-string 검사로 일부 막지만 실제 imported library path·thread setting은 직접 확인하지 않는다. future model runner의 CPU/cache/threads/params/generator fingerprints는 별도 필수다. completeSHA/path를 details에 기록하면 future receipt의 연결이 더 명확해진다.

매fold 호출마다109source+66receipt SHA 및1187feature를 fresh 확인하므로72/원66×24×3seed 모델루프에서 무조건 반복하면 준비검사 비용이 커질 수 있다. cache 최적화를 하더라도 행/열/label/matrix/source/complete digest 동일성을 지키고 모든candidate/seed 시작의 guard를 생략하지 않는다. 성능 절약을 이유로 False를 생산 경로에 쓰면 안 된다.

현재 코드에 새 core preparation blocker는 발견하지 못했다. OFM01/02는 미래 model등록/fit 전에 닫아야 하며, actualprobe와all66complete 증거가 나오기 전 모델 준비 완료·성능검증 완료를 보고할 수 없다.
