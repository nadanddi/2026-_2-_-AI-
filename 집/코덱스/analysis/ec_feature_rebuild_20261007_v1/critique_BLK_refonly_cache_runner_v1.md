# BLK 학습 전용 PFN 캐시 runner 독립 초안 비평 v1

2026-10-07. `blk_pfn_refonly_cache_v1.py` 최초 읽은 초안 기준. 작성 중 파일은 부모가 보완할 수 있으므로 아래 결함이 최신 버전에도 남아 있는지는 재확인해야 한다. 모델 fit/predict, 정답 열람, 채점, GPU 실행은 하지 않았다.

## 판단 범위

공개 `fit_with_cache`/`kv_cache_precision='auto'` 명시, 기존 uncached 출력 보존, 별도 폴더, 동일 2000개 context ID 검증, 공식 `_embed_features` 경로 관측은 앞선 all-query 통계 fit 문제에 대한 적절한 수정 방향이다. 현재 초안은 실행 전이며 규정·수치 PASS 증거가 아니다. 기존 CPU 수치 실패는 그대로 유효한 기록이다.

## RC01 — 관측 guard가 실제 실행됐음을 양성 확인해야 함 (P1, gate 전 필수)

`ReferenceOnlyTrace`는 관측한 호출에 대해 fit=2000행/2000라벨/캐시 없음, predict=캐시 있음/라벨0을 강제하며 mask/group fit을 예측 단계에서 차단한다. 단, trace.call/mask/group 목록의 비어 있지 않음과 정확한 호출 수를 assert하지 않는다. 다른 구현이나 별도 import 경로가 wrapper를 우회하면 trace가 비어도 마지막 검사로 차단되지 않는다. model 이름 문자열과 feature_cache 존재만으로 공식 함수 실행까지 보장할 수 없다.

fit 직후 최소 관측양성을 확인하라. 현재 소스/4 estimator 기준 embed fit4회, group fit4회, mask fit8회가 예상된다(group 함수가 mask를 다시 호출함). 내부 실행이 다른 경우 검증 규칙을 실행 전에 설명하고 고정한다. predict 단계에도 각 실제 호출의 embed 관측을 확인한다. patched 함수의 원래 module/__file__/source SHA, 모델과 executor의 실제 class module/file을 같이 저장한다. 캐시 없는 상태를 일부러 예측으로 넘기는 작은 별도 unit guard 확인은 소스 검토만으로 대체하지 않는다.

2000행 guard는 학습 행 수를 검사하지만 출처를 검사하지 않는다. 실입력 X[ix]/y[ix]의 순서·dtype·shape·SHA와 ID SHA를 연결하고 ensemble member의 실제 전처리 행 수/subsample 설정을 기록해야 '같은 2000 context'까지 재현된다. 이 리뷰는 예상 default subsampling을 실행 확인하지 않았다.

## RC02 — 저장 파일 pin과 런타임 pin은 다름 (P1, gate 전 필수)

library_sha256는 예상 ROOT 파일을 해시한다. 현재 import된 tabpfn/architecture/engine이 그 파일이라는 확인은 없다. `tabpfn.__file__`, `architecture.__file__`, executor와 model class의 inspect source path를 resolve하여 해시 대상과 일치시키라. 실제 model.get_params 및 inference_config_, dtype/device/ensemble member 수, kv cache dtype, autocast/메모리/chunk 설정도 기록한다. weight SHA는 좋지만 새 runner 및 adapter·AST 추출 함수·상위 준비 코드와 실제 입력 행렬이 같은 lineage로 이어져야 한다.

X/Q/y의 실제 matrix SHA가 필요하다. train 목표 y는 공개 학습 정답이며 해시 기록은 보류 정답 채점과 구별된다. 평가 prefix 규정은 이 캐시 guard가 상위 prepare_query를 검사하는 것은 아니므로 기존 feature/calendar 미래 교란 감사를 유지한다. 여기서 other-query poison은 이후 query만으로 제한하지 않았으므로 pure-future-only 검사라고 표기하지 않은 점은 타당하다.

## RC03 — feature statistics 불변과 전체 cache 불변을 구별 (P1, gate 전 필수)

cache_fingerprint는 6개 feature statistics의 tensor SHA만 검사한다. KV key/value, test_y_embedding, CPU preprocessing pipeline의 fitted state는 검사하지 않는다. 현 구현은 'feature statistics unchanged'로 보고할 수 있지만 '전체 cache/model state 불변'으로 확대할 수 없다. 모든 KV 및 test target embedding의 dtype/shape/SHA를 before/after 대조하거나 그 한계를 명시하고 반복 호출 증거를 추가하라. CPU transform fitted state가 query 입력을 fit하지 않는다는 기존 소스 검토 역시 별도 근거이다.

full1440 예측을 모든 교란/역순/소배치 뒤 다시 실행하여 처음과 고정1e-6 비교를 추가하라. 같은 single/subset 반복도 호출 순서 누적 상태 영향을 검사한다. 초안은 전반적 numerical check를 갖췄지만 full repeat 검사가 없다. finite96/8/단일8/2~7정역이 PASS해도 모든 가능한 query에 대한 증명은 아니며 source train-only 구조와 함께 제한된 근거로 보고해야 한다.

## RC04 — 실패·재개와 출력 검증에 누락 (P2, 실행 전 권고)

lock 획득 후 torch thread 설정, X/Q/y 변환, probe/index 생성이 try 밖에 있다. 이 구간 오류는 failure 기록/finally 해제 경로를 건너뛰고 lock을 남긴다. lock 직후부터 보호 구간으로 포함하고 정리 오류가 원래 예외를 가리지 않게 한다. 동일 프로세스에서 interop thread 설정 재호출이 실패할 수 있으므로 환경 실현 값도 체크한다.

resume 분기는 saved.registration/pred hash와 audit.status/prediction file SHA만 검사한다. audit.registration/code/adapter/library pin, saved.row_ids/context/길이/finite, audit의 실제 required check keys·고정 tolerance·관측 trace 양성·cache 불변이 없거나 틀려도 PASS 문자열만으로 skip할 수 있다. gate가 이를 모두 검사하도록 보장하거나 runner 재개 시 동일 검증을 수행한다. 모든 4개 파일의 완결 SHA를 complete.json에 묶으면 재현 연결이 명확하다. 기존 complete가 존재할 때 마지막 assert가 실패하는 재실행 정책도 '불변 completed run 확인 후 정상 종료'인지 '실행 거부'인지 명시한다.

예측 output을 audit보다 먼저 저장하고 중간 종료시 불완전 pair를 거부하는 정책은 보존 우선이라 타당하다. 수치 FAIL의 output/audit를 남기고 assertion으로 멈추는 것도 타당하다. 단 예외가 audit 작성 전이면 failure 기록에서 어느 context·phase가 실패했는지 같이 저장할 필요가 있다.

## 후속 판정

RC01~03 실현값·호출 증거와 실제 4문맥 결과가 필요하다. 소스 관점에서 명시 공개 캐시 경로는 앞선 query 통계 fit을 고치는 방향이지만, 현재 초안을 규정 통과 baseline 또는 개선 모델로 판정하지 않는다. 이전 uncached와의 차이는 DIAGNOSTIC_ONLY로 유지하고 새 PFN/R3/SG2 assembly lineage와 독립 gate를 통해서만 채점 단계로 연결한다.
