# BLK reference-only cache runner v3 독립 재검토

2026-10-07. `blk_pfn_refonly_cache_v3.py` 읽기 전용 검토. 코드 실행·모델 fit/predict·보류 정답 열람·채점·GPU 사용 0. 부모가 실제 runner를 시작했다고 전달했지만 본 리뷰는 해당 실행 결과를 확인하지 않았다. 원 v1/v2와 기존 uncached FAIL은 보존 대상이다.

## 기존 지적의 소스상 닫힘

| 지적 | v3 확인 | 판정 |
| --- | --- | --- |
| RC01 관측 guard가 비어도 통과 | fit embed4/group4/mask8 및 prediction embed>=104 강제 | 소스상 닫힘, 실제 trace 대기 |
| RC02 예상 파일 해시와 실제 import 분리 | 7개 module.__file__ resolve 일치 강제, 실제 engine/architecture 이름과 모듈, fit_mode/cache precision, X/Q/y SHA 저장 | 주요 결함 소스상 닫힘, 실현값·추가 설정은 아래 잔여 |
| RC03 feature stats만 불변 검사 | layer별 KV key/value와 test target embedding까지 전후 SHA, 교란 후96/전체1440 재예측 | 소스상 닫힘, 실제 불변·수치 결과 대기 |
| RC04 lock 이후 try 밖 실패 | torch 설정/행렬/probe를 lock 직후 try 안으로 이동 | 주요 고아 lock 위험 소스상 닫힘 |
| RC04 재개 검증 약함 | audit 등록/code/adapter pin, output row ID/길이/finite, 검사 수, trace 수/fit행수, cache 전후 equality 추가 | 상당 부분 보완, 완전 닫힘은 아님 |

26개 predict 호출(full1+96subset1+reverse8 1+single8+2~7정역12+poison1+repeat96 1+repeatfull1)이 각각 4개 estimator를 통과하면 104 embed 호출이다. 현재 소스의 >=104는 이에 맞는다. 학습에는 4 embed/4 group/8 mask가 예상되며 `_fit_feature_group_scaling`이 내부에서 mask를 한 번 더 호출하는 경로와 일치한다. 각 호출 내부의 phase/labels/cache assertion을 함께 유지한 점은 타당하다.

## 잔여 RCV301 — 원래 device 검사가 아님 (P2, 최종 gate 전 보완)

cache_fingerprint는 tensor를 `.detach().cpu().contiguous()`로 복사한 다음 `tensor.device.type=='cpu'`를 검사한다. 이는 원래 캐시가 CPU에 있었음을 입증하지 않는다. 생성자 device='cpu'와 env_extra 소스 경로는 CPU 의도 근거이지만 `GPU_used:false`를 실제 확인했다고 주장하려면 복사 전 원본 key/value/feature/target device를 기록·검사하고 executor.get_devices()/model parameter device·dtype도 확인한다. float32 assertion은 복사로 dtype이 변하지 않으므로 여전히 유효하다. 실제 GPU 사용을 발견했다는 뜻은 아니며 증거 범위의 결함이다.

runtime_state는 model.memory_saving_mode와 thread 값 등을 기록하지만 thread값/precision/device에 모두 assertion을 걸지는 않는다. 실제 n_estimators/random_state/n_preprocessing_jobs/inference_config_, cache ensemble member 수 및 max_batched_test_rows/autocast 설정도 receipt에 남기면 재현이 명확하다. matrix SHA와 함께 dtype/shape/columns/ordered context ID를 연결하라. 현재 생성자 값은 명시되어 있으므로 새 성능 정책 변경이 필요하다는 지적은 아니다.

## 잔여 RCV302 — 재개와 최종 receipt는 gate가 다시 검증해야 함 (P2)

resume는 checks 길이18을 검사하지만 정확한 key 집합, atol1e-6/rtol0, max_difference와 계산값 일치, 모든 값의 nonnegative finite를 확인하지 않는다. saved.context/fit_mode, fit trace 각 항목의 rows2000/labels2000/cacheFalse 및 prediction trace labels0/cacheTrue도 skip 시 다시 검증하지 않는다. input_matrix_sha256/runtime_state는 기록되지만 현재 입력 및 기대 정책과 재대조하지 않는다. 처음 정상 실행에서는 guard가 내용을 보장하지만 재개 또는 독립 gate는 PASS 문자열과 수만으로 동일 근거를 보장할 수 없다.

complete.json은 4개 output/audit 파일 SHA를 담지 않는다. 새 assembly/gate가 각 파일 SHA·등록 pin·실제 18키·trace 내용·원 device·4 context completeness를 직접 묶어서 검증하면 이 잔여를 닫을 수 있다. completed run 재호출 마지막 assert와 failure 파일의 context/phase 부재, finally lock 읽기 오류가 원예외를 가릴 가능성은 운영상 제한으로 남는다. 본 리뷰는 기존 파일 수정을 요구하지 않으며 필요하면 새 gate/receipt 버전에서 해결한다.

## 실행 및 채택 범위

소스상 이전의 핵심 내부 query 통계 fit 문제에 대해 train-only 공개 캐시+공식 경로 관측+양성 호출 수+전체 캐시 내용 해시가 갖춰졌다. CPU 진단 실행을 막아야 할 새로운 명백한 입력 누수는 이 초안에서 발견하지 못했다. 이는 실제 source 경로 관측 PASS, 수치 PASS 또는 전체 모델 gate PASS와 같지 않다.

4 context 실제 PASS 후 새 PFN 출력/R3/SG2/후처리 assembly와 위 잔여 receipt를 독립 gate로 연결해야 한다. 옛 uncached와의 차이는 진단만 가능하며 uncached 정책의 numerical FAIL을 지우거나 '동일 baseline 재현'으로 해석하면 안 된다. 보류 정답을 아직 읽지 않는 정책을 유지한다.
