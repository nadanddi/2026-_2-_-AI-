# BLK CPU 공개 캐시 경로 독립 소스 검토 v1

2026-10-07. 읽기 전용 라이브러리·기준선 코드 검토이며 모델 실행, GPU 사용, 보류 정답 열람, 성능 채점은 하지 않았다. 기존 `critique_BLK_CPU_padding_v1.md`를 대체하거나 수정하지 않는 추가 기록이다. 실행 중 수치 감사의 완료 여부와 실제 객체 경로는 확인하지 않았다.

## CACH01 — 기본 경로의 query 의존 전처리는 소스에서 확인됨 (P1)

`blk_pfn_baseline_v1.py:51`은 `fit_mode`를 지정하지 않는다. 로컬 `regressor.py:621`의 V2 factory는 model_path와 n_estimators만 기본 변경한다. 생성자 `regressor.py:297` 기본은 `fit_preprocessors`이며, `base.py:385`는 `InferenceEngineCachePreprocessing`을 선택한다. 이 경로는 CPU 전처리기를 학습 입력으로 fit하지만 모델 forward에는 학습+query 입력을 함께 전달한다. 실제 작업 객체의 클래스·fit_mode는 별도 실행 로그로 봉인해야 한다.

`architectures/tabpfn_v2.py:637`의 `_embed_features`는 feature_cache가 없으면 전체 행에서 열 선택 mask와 그룹 배율을 fit한다. 결측 대치 평균과 standard scaler만 학습 행으로 fit한다. `:1139`의 `_constant_feature_mask`는 모든 행을 첫 행과 비교하며, `:1194`의 `_fit_feature_group_scaling`은 같은 전체 행 mask의 개수로 배율을 정한다. 열 제거 함수는 남은 열을 그룹 앞쪽으로 이동하므로 query에 의해 mask가 달라지면 열 배치까지 달라질 수 있다. NaN 비교 의미와 대치 후 mask도 구별해야 한다.

따라서 train-only attention을 근거로 전체 모델이 query 간 독립이라고 결론내릴 수 없다. 현재 자료에서 실제 어떤 mask가 변했는지, 기존 1440개 예측에 얼마의 영향이 있었는지는 미검증이다. 최소 8행 복제는 고유 값 집합을 늘리지 않지만 이 전체 행 전처리 구조를 학습 전용으로 바꾸지 않는다. 수치 PASS만으로 규정 gate를 열면 안 된다.

## CACH02 — 공개 fit_with_cache는 학습 전용 feature cache를 만드는 경로 (조건부 권고)

`base.py:396`의 `fit_with_cache`는 `InferenceEngineExplicitKVCache`를 선택한다. `inference.py`의 생성자와 `_build_cache`는 ensemble member의 X_train/y_train만 사용해 모델에 `return_kv_cache=True`로 전달한다. V2 forward는 이 학습 입력에서 feature cache와 각 layer KV, 결측 target embedding을 저장한다.

이후 `_call_model`은 변환한 query를 `kv_cache=...`, `x_is_test_only=True`로 전달한다. V2 forward의 using_cache 분기는 `feature_cache=kv_cache.feature_cache`, num_train_labels=0으로 `_embed_features`를 호출한다. 열 선택 mask, 대치 평균, scaler mean/std, 그룹 mask와 사용 특징 수를 모두 캐시에서 재사용한다. 이 소스 구조는 query 입력으로 해당 통계를 다시 fit하지 않는 정책에 적합하다.

권고: 라이브러리 수정보다 공개 옵션을 사용하고 새 기준선·별도 출력 폴더로 등록한다. `fit_mode='fit_with_cache'`, `kv_cache_precision='auto'`, CPU/float32, 동일 학습 ID·특징 순서·seed·ensemble·weight를 명시한다. precision 기본 None은 architecture 지원에 따라 int8가 될 수 있으므로 생략하지 않는다. 캐시 선택으로 학습 context나 샘플 수를 변경하는 것은 별도 정책 변경이다.

## CACH03 — 동등성 및 새 gate의 한계 (P1)

이 전환은 단순 GEMV/GEMM 수치 보정이 아니다. 이전 경로가 전체 행에서 정한 상수 열·그룹 통계를 새 경로는 학습 행만으로 정한다. 그러므로 같은 weight/seed라도 이전 raw1440 출력과 항상 같은 수학적 함수라고 주장할 수 없다. 실제 CPU 연산 원인은 아직 특정하지 못했다. 기존 single-row FAIL 및 모든 raw·등록·감사는 그대로 보존한다.

새 실행 전에 다음을 고정해야 한다.

1. 실제 model architecture, engine class, fit_mode, cache precision, inference dtype, thread/메모리/chunk 설정과 소스 SHA. factory 기본값이 아니라 실현된 값 기록.
2. 캐시 생성의 입력 ID/특징 SHA 및 query 포함 0. feature cache 각 tensor의 이름·shape·dtype·SHA를 생성 후와 예측 후 비교. 모델 내부 캐시 변화가 결과를 누적 변경하지 않는지 순서 반전·반복 호출 확인.
3. 문맥 5~8 각각 단일 및 2~7행, 8행 이상, 전체1440, 재정렬/분할/반복 호출을 같은 고정 atol1e-6로 점검. 미래·다른 farm query 교란과 현재 prefix 변경 대조도 유지. 필요하다면 작은 batch 수치 정책은 새 캐시 경로 안에서 별도 등록.
4. 실제 CPU 전처리 pipeline의 transform이 query 통계를 fit하지 않음 확인 및 학습+MASK 규칙 유지. feature cache만 안전해도 상위 adapter/feature builder/후처리 전체가 자동 PASS인 것은 아님.
5. 재생성한 PFN 출력·R3 조합·SG2·후처리의 새 lineage를 연결하고 기존 assembly/gate를 재사용하지 않음. 먼저 규정·재현 gate 통과 후 보류 정답 채점.

캐시 고정은 합법적 처리 구조에 관한 소스 근거다. 모든 작은 batch의 수치 허용오차, 실제 저장 객체, 라이브러리 실행 경로, 전체 baseline PASS는 아직 실행 증거가 필요하다. 현재 independent source review만으로 개선·채택·최종 모델을 허용하지 않는다.

## 읽은 로컬 소스 SHA256

| 파일 (.analysis-tools/extra/tabpfn 아래) | SHA256 |
| --- | --- |
| architectures/tabpfn_v2.py | bc4f947a605dbb05aac1f0654e4bb376542d10981c8df2562455b6091c9cd6ac |
| inference.py | a9eb1ca6475737cca17070768f43225479a28fe17a52f61a7aba12908b1881d7 |
| base.py | 5df137ae5512d4061456e6138c7859ee866a0a870cfdf084b739f69629b78706 |
| regressor.py | 5b3e2120a7f727cfc3bdd76b8249b6a5ae8ccef01806601f5e19eb9fb8333480 |

PowerShell Get-FileHash로 읽은 파일을 직접 대조했다. 실제 실행 import __file__와 일치한다는 증거는 본 리뷰에서 만들지 않았다.
