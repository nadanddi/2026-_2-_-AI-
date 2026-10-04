# GATE25 / RIDGE26 실행 전 독립 비평

run_v2.py를 정적으로 읽었다. 현재 준비 JSON이 아직 생성되지 않아 preparation PASS를 확인하지 못했다. analysis-verification 관점에서 **실험 설계는 지지하되 현 runner의 재현·provenance 감사 보강 전 실제 실행 수용은 보류**한다. 후보 성능을 계산하거나 작성자 코드를 실행하지 않았으며 학습·원시 EC/test/잠금/EL1 조회는0이다.

## 지지하는 설계

- 내부 모델 타깃은 b의 공개 label이며 내부 baseline은 a에서 학습한 기존 OOF CPU/PFN 결과다. anchor reference=a, outer reference=tr로 분리했다. 현재 코드에서 query/b 자신의 target을 anchor lookup에 직접 넣는 경로는 발견하지 못했다.
- 동일 farm/pass·엄격 earlierday 필터가 anchor 후보에 적용된다. query가 reference를 갱신하거나 다른 query의 label을 참조하지 않는다. query signature는 자기 day의 현재까지 available prefix mean이다. test를 concat해 z 통계를 fit하던 SG2와 달리 reference-hour median/z만 사용한다.
- a/b/tr/q IDs를 이전 검증 preparation의 manifest와 대조하고 a,b가 outer tr의 부분집합인지 확인한다. a→b와tr→q의 ±1 purge도 검사한다. a∩b를 명시적으로 다시 assert하는 것은 권장하지만 이전 IDs의 정확한 고정이 유지되면 직접 자기참조 위험은 줄어든다.
- GATE label은 clip한 trial correction의 실제 b SSE 개선 부호이고 weight는 그 절대 크기다. 이는 supervised meta target이며 label 자체를 feature에 넣지 않는다. RIDGE는 b hourly residual을 학습한다. 두 안의 보정 계수·clip·threshold를 outer 점수로 고르지 않는 방향은 적절하다.
- 보정은 완성된 actualv2 baseline 이후 한 번 더한 후 bounds clip이라는 신규 정의다. 기존 raw ensemble 안의 LGB를 교체한다고 설명하면 안 된다. GATE 최대 제안 delta는±.3, RIDGE 최대 delta는±.06이다. 최종 clip 때문에 실효 candidate−baseline은 이보다 작을 수 있다.

## 실행 전 필요한 감사

1. **P1: 산술 문턱과 최초 감사.** close는현재 `<1e-10`이다. 이전 고정 실제예측/산술 audit의1e-12보다 느슨하다. native repeat, 저장 모델 독립 forward, scalar prefix/clip, query order/single/other query/동일 input prefix 감사는 원1e-12로 유지해야 한다. 현재 first는 fit_predict 두 번과 fit dict 동일성만 검사한다. batch/state·미래 입력 불변성을 별도로 검증하지 않으며 scaler와학습을query별로다시fit한 “single 감사”로 대체하면 안 된다.

2. **P1: prior provenance의 신선한 결속.** PREV를 load해 status와 현재 dictionary/캐시 SHA를 신뢰한다. 이전 final-loss preparation SHA `df4ff2c3b7617676c61e4cee4b8e9690f9312a7cca448c0caec189cb5051b860`의 expected pin을 명시하지 않는다. 새 preparation에 prior SHA를 기록하는 것만으로 원본 검증 receipt와 결속됐다고 할 수 없다. 기존 원R3/PFN provenance guard와 full_inner의 label/feature/context/bounds/runtime/source 검증을 적절한 기존 guard로 다시 호출하거나, 불변 검증 receipt와 전체 source/input/cache pins를 명시적으로 검증해야 한다. 이 의견은 cache가 실제 오염됐다는 주장과는 다르다.

3. **P1: 매 fit freshness와 실패 보존.** actual 시작의 preparation dict 동일성만 확인하며 각 fit 직전 source/input/cache/preregistration hashes를 재확인하지 않는다. prereg 파일은 존재만 확인한다. 이전 실험 수준의 immutable config/runtime/source/input/cache/prereg/whole receipt gate가 필요하다. 실패 시 raw/query prediction·오류·partial artefact를 보존하는 failure receipt가 없고 디렉터리 존재로 재시작을 막을 뿐이다. 파일이 만들어진 뒤 별도 수정·재사용을 금지하고 새 버전에서만 보완하는 운영 조건도 명시해야 한다.

4. **P1: 저장 모델과 독립 replay 근거.** fit JSON은 n/positive/mean/scale/coef/intercept/model_none을 저장하지만 scaler variance/sample count, exact model parameters, eligible row IDs/hash, labels/weights/feature matrix hash가 없다. GATE가 한 클래스라 model=None이면 constant probability도 직접 저장하지 않는다(positive/n으로 추론 가능하더라도 명시적 기록 권장). 독립 verifier는 고정 b-only 표준화와 cost weight를 재생하고 sigmoid/Ridge forward 및 one output clip을 계산할 수 있어야 한다. 모델/전처리/feature order/dtype/finite/constant branch의 strict schema와 corruption rejection을 준비해야 한다.

## 범위와 구현 주의

- signature·median/z는 hour별로 reference 전체 farm/pass를 pool한다. anchor 선택만 samefarm/pass/earlierday다. CFG에서 reference라는 말을 거리 후보와 fitted processor에 동일하게 쓰면 혼동된다. 현재 ref-only pooled preprocessing을 의도한 것이라면 이를 선언해야 한다. query 이후 시점의 **학습** records를 fitted 전처리에 쓰는 것은 stored training model과 구분해 검토할 일이며 query future-input 사용이라고 자동 판정하지 않는다.
- prefix는24개 full day를 요구하고 index가 positional RangeIndex임을 가정한다. neighbors의 out[ii]도 같은 전제를 갖는다. 현재 a/b/q의 reset_index로 full batch는 맞지만 부분 prefix·single-row·permuted-index 감사는 별도 안전한 함수나 keyed 정렬/재매핑이 필요하다. 실제 query batch 크기가 예측에 영향을 주지 않는지 모델을고정해 확인해야 한다.
- GATE eligible에 abs(cost)>1e-14가 추가돼 있다. 그 threshold와 가중치의 평균1 정규화 모집단을 prereg/config에 수치로 고정해야 한다. eligible0이면 scaler fit 전에 명시적인 중단과 failure receipt가 필요하다. 한 클래스 fallback을 성능 실패로 임의 바꾸지 않고 사전 정의대로 기록해야 한다.
- NaN prefixskip은 현재까지 finite 관측만 누적하고, 없으면 ref-hour median으로 대체한다. all-empty column의 median0 fallback을 선언하고 mean/std/feature가 모두 finite임을 prep 단계에서 검사해야 한다. raw IDs/hour uniqueness 및24h coverage도 검사해야 한다. median0 자체가 forbidden target imputation이라는 증거는 없다.
- 이 안은 단일 inner heldout b를 meta training에 사용한다. 전체 outer train crossfit meta 학습 또는 새 untouched 검증이라고 설명하면 안 된다. GATE와RIDGE를 순차 실행하더라도 alpha=.025/26 및strict15/seed별 DIAG 조정CI를 고정하고 GATE 결과를 보고 RIDGE recipe를 바꾸지 않는다.

현재 direct target leakage보다 주된 미비는 신선한 provenance·감사·저장 모델 재현이다. 독립 whole는66셀/aggregate/source/runtime/first/checkpoint guards 완료 전에 외부점수를 계산하지 않아야 한다. 이 조건을 새코드/prepare/합성 반례로 구체화한 뒤 실행하는 것이 타당하다.
