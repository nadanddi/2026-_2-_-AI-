# 원66 PFN runner2·규칙1·등록기2 사전 비평

2026-10-07. 소스·저장 metadata·current SHA만 검토했다. PFN 생성/fit/예측/query/보류 정답/채점/GPU 실행0이며 부모의 raw worker는 건드리지 않았다.

**OPFN01의 핵심 재개 검증 누락은 닫혔다. OPFN02의 runtime/source 등록 보완도 소스상 수용한다. 새 핵심 fit blocker는 발견하지 못했다.** 다만 registrar2는 아직 미실행이므로 실제 PFN 등록 PASS는 아니다. 실제 등록·source 전수 검사 성공 전 PFN fit은 계속 차단한다.

## 현재 증거

- plan capture code, runner2, synthetic audit code, rules1의 recorded SHA가 각각 현 파일과 일치한다. plan runtime/source27핀도26개 기본 MATCH+접근이 거부된 pandas1개 정식 승인 읽기 SHA MATCH로 현27개 모두 일치한다.
- plan의66fold×4=264 context 각각에 대해 ID2000개·중복0·그 fold train에만 포함·query/금지 ID 혼입0을 독립 metadata 검사했다. seed RNG 순서의 재생은 capture와 future runner가 담당한다. 이번 검토는 모델 context 숫자 matrix 재구성 검사가 아니다.
- synthetic21개 저장 PASS이며 현재 audit/rules source에 연결된다. trace/check값은 합성이고 cache shape는 과거 실제 BLK metadata에서 가져왔다. 원66 actual fit·수치 PASS라고 부를 수 없다.

## 수용한 변경

rules1은 exact4 cache, train_shape[1,2000], exact6통계 key/dtype/shape, exact12 KV층/key/value, target embedding, 원deviceCPU 및64자리 SHA를 검사한다. runner2는 tensor를 `.cpu()`로 복사하기 **전에** 원 device를 확인하고 메타데이터에 CPU를 기록한다. 기존 원device 직접 증거 부재를 이 새 runner의 실제 fit audit에서는 보완할 수 있다.

exact120 trace=4fit+116predict와 phase집합,2000행 fit/labels, cache true/labels0 prediction 및 batch row집합, mask8/group4, exact qualified engine/architecture, memoryauto/FP32/thread4/interop1을 재개와 새 출력 모두 검사한다. architecture실제1개와 ensemble cache4개는 서로 다른 수다. **이전 비평 v1의 architecture4 요구와 runner1의 len4 조건은 잘못된 것이며 runner2의 exact1이 실제 BLK 기록과 맞다.**

실제 adapter/baseline/context/env_extra/checkpoint/threadpool/numpy/pandas runtime binding이 추가됐다. 등록기2는 rawreg 전214핀을 상속하고 충돌을 거부하며 plan capture/runner SHA, source27, context train-only, local weightsSHA,38열·CPU cached 정책, synthetic21의 code/rules SHA/no-fit/no-target을 연결한다. -O 거부 및528파일/wholebaselinefalse/noadoption 경계가 있다.

## 잔여 비차단 권고와 실행 한계

등록기의38열 조건은 유일성·개수뿐이다. 현재 capture source가 parent columns==FULL을 실제 검사했고 pinned이므로 현 계약의 오류 증거는 없지만, 등록기·runner에도 reg columns==실제 FULL 순서를 직접 대조하면 계약이 더 명확하다. 등록 model_arguments/CPU_policy를 runner 생성자 kwargs 및 상태와 직접 대조하고 memory_saving_mode='auto'를 생성자에 명시하면 pinned default 의존을 줄인다. 현재 소스의 kwargs와 실제 fit 후 상태 검사는 등록 정책과 일치한다.

future registrar에는 synthetic real_query_predictions=0와 parent_metadata recorded SHA의 current 연결, runtime module paths의 기대 workspace/library 위치·정확7라이브러리 집합 검사를 직접 추가하는 것도 권고한다. 현재 audit/capture source가 pinned이고 보류 숫자값을 읽지 않는 구현인 점을 함께 확인했다.

정확한 fingerprint shape는 현 V2/weights의 기대 구조를 의도적으로 제한한다. 실제 원fold에서 구조나 dtype/trace count가 달라지면 새 fit은 실패해야 하며 PASS를 위해 조건을 사후 완화하지 않는다. 실패 및 pair 미완성은 원값을 보존한다. 예외 단계의 별도 failure receipt와 lock 최초 쓰기 실패의 stale lock 문제는 v1의 회복 한계로 남는다.

21 numerical checks/3 prefix는 여전히 유한 표본 검사다. 원행 전수 causal·원feature train MASK·PFN/R3 조합·shrink한번/clip/SG2·독립 currentlineage gate를 대체하지 않는다. CPU cached baseline은 과거 GPU/uncached 또는 다른 SG2 범위의 동일성을 주장할 수 없다. 24원후보 전수·고정 원통계·최초 미사용1회 판정은 유지한다. raw/PFN 동시 fit 여부는 메모리·CPU 실제 상태로 결정할 운영 문제이며 이 source 리뷰에서 실행 승인을 대신하지 않는다.
