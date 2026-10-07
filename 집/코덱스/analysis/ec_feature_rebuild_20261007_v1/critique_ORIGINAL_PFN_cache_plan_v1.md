# 원66 PFN cached baseline 초안 독립 비평

2026-10-07. `run_original_pfn_cache_v1.py`, 재사용 cache3/minbatch8, BLK strict receipt2 소스만 읽었다. 모델 생성·학습·예측·query·보류 정답·채점·GPU 실행은 하지 않았다. registrar 미작성/등록0/fit0인 초안에 대한 검토다.

## 판단과 fit 전 보완

train-only cached 경로 자체는 이전 BLK 수정 정책과 일치한다. 다만 **원66 runner의 저장/재개 검증이 BLK strict receipt보다 약하다.** 다음 두 항목을 새 버전에서 강화하고 실제 runtime/등록 계약을 봉인한 후 실행하는 것이 타당하다.

### OPFN01 — 캐시 footprint와 trace 재개 검증 누락

`validate_saved`는 before==after만 확인한다. 빈 dict/list나 축소된 동일 두 footprint도 이 조건을 만족한다. cache4개, train_shape=[1,2000], 정확6개 feature statistic key, KV12층 각 key/value, FP32/양의 shape/2000학습축/target embedding/64자리 hash를 검사하지 않는다. 실제 새 fit의 fingerprint 함수가 일부 검사를 하더라도 저장 artifact 재개 및 future verifier가 동일 조건을 검증하는 문제는 남는다.

BLK strict receipt2의 구조 검사를 이식한다. trace는 정확 fit4와 predict116(고정 호출 정책을 유지할 경우), 전체 calls=fit+predict, phase 집합 정확2개, fit reference2000, predict labels0/cache true, predict row수 집합 {8,min(96,nquery),nquery}를 확인한다. 지금 `len(pred)>=116`, `rows>=8`만으로는 미지 phase 및 기대하지 않은 batch를 허용한다. memory_saving_mode도 기록·검사하고 engine/architecture를 suffix가 아닌 실제 등록한 정확한 qualified name으로 대조한다.

### OPFN02 — 실행 코드·모델·weight의 prospective 등록

아직 registrar가 없어 현재 `reg` 계약을 충족했다는 근거는 없다. 원66 정확 fold roster/4context/264fit/528파일, ordered train/query/forbidden IDs, 모든 context2000개를 미리 고정해야 한다. 각 fold의 train행≥2000 및 query행≥8을 fit 전 확인한다. context은 seed5..8로 그 fold의 train 순서에서만 재생한다. 원 BLK context ID를 재사용하는 주장은 금지한다.

정확 FULL38 순서, local v2 weight path/SHA, estimators4/seed/devicecpu/FP32/preprocessjobs1/fit_with_cache/kv_auto/minbatch8/memory policy/thread4/interop1을 등록·대조한다. rawreg의 전체 source pins와 complete66/독립 준비검산, 통계 봉인 링크를 전이 포함한다. minbatch8 및 baseline/context/checkpoint/env_extra/threadpool/numpy/pandas 실제 import source/경로도 pin과 runtime binding에 연결한다. 현재 runtime_paths에는 cache/feature/Torch/TabPFN 및7라이브러리는 있지만 adapter 등의 actual import binding은 빠져 있다. source SHA 검사만으로 동일 이름의 다른 imported callable을 검증하지 못한다.

## 소스에서 수용한 내용

- production loader는 rawreg를 통해 complete66 및 all24 fresh matrix gate를 요구한다. query/forbidden은 context에 들어갈 수 없고 y 전체의 길이·유한성·actual SHA를 확인한다. context matrix/reference y/query matrix SHA와 준비 영수증을 저장 contract에 연결한다.
- 캐시 fit에는 X[ix]/y[ix] reference2000만 들어간다. ReferenceOnlyTrace는 fit의 2000행/labels2000/cache없음, predict의 labels0/cache있음과 prediction 중 mask/group fit 금지를 관측·차단한다. minbatch8은 제공된 마지막 행만 복사하므로 자체적으로 새 query 정보를 추가하지 않는다.
- 4 context×66=264 model fits이고 각 fit은 estimators4다. 이를 264개의 estimator fitting으로 부르면 부정확하다. 저장 output+audit264쌍=528파일이다.
- 21 checks=기존 numeric18+fresh prefix3. 실제 predict 호출은 full/scattered/reverse/독립single8/batch2..7정역12/poison/repeat2/prefix3 =29회, 각4 architecture embed이면 predict trace116개다. fit trace4, mask8/group4 정책과 산술상 맞는다.
- `-O` 금지, CPU-only Torch build, 실제 환경 일치, source 재확인, exact output row IDs와 pred digest/유한성, pair 미완성 시 보존·중단, fold/final complete 정확 재개 비교를 유지한다. whole_baseline_complete=false/no-score 경계도 적절하다.

## 인과성·failure·gate 한계

prefix3은 첫/중간/끝의 현재·이전 입력에서 FULL을 재구성하고 나머지 query를 교란하여 대조한다. 평가행 전수 causal 감사는 아니며 h0/자정/모든 fold의 모든 row 경계를 대표하지 않는다. 8행 poison도 protected 한 행과 다른 query batch 영향의 제한된 검사다. 원 full-model gate에는 전수 prefix, 후처리, source/matrix/weight/receipt current SHA 및 독립 검산이 별도로 필요하다. cached 내부 query-stat fit 금지와 batch 수치 검사는 구분한다.

숫자차이 또는 cache변경 FAIL은 출력을 보존하고 validate_saved에서 멈춘다. 반면 fit/trace exception·비유한 예측은 pair 쓰기 전에 중단해 별도 failure receipt가 없다. 실패 단계/exception/source/context/정답미열람을 새 별도 failure artifact로 남기는 것은 권고하며 PASS를 대체해서는 안 된다. 출력 후 audit 기록 실패 때 단독 출력이 남고 재개가 차단되는 것은 안전한 중단이므로 자동 삭제·자동 재fit으로 우회하지 않는다. lock 초기 기록이 try 밖인 stale lock 가능성도 남으며 정확 PID/start time 확인 뒤 처리해야 한다.

Torch thread/interop 설정이 기존 BLK 실행과 동일하더라도 현재 프로세스에서 이미 interop 작업이 시작됐다면 set_num_interop_threads 호출이 실패할 수 있다. fresh process로 시작하고 실패를 기록한다. 모델 cache의 원 tensor device 직접 기록 부재 한계는 기존과 같고 CPU-only wheel+명시 CPU 설정을 별도 증거로 사용한다.

이 초안은 역사적 GPU/uncached 출력 동일성을 주장하지 않는다. 원66 PFN raw 완료는 PFN/R3 혼합·shrink 한 번·clip·SG2와 fullgate 완료가 아니다. PFN policy/문맥4·weights를 결과에 따라 바꾸지 않고 전24 원검증기 및 고정 통계, 최초 미사용 1회 판정을 유지해야 한다.
