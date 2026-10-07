# 원 66폴드 raw3·등록3 독립 사전 비평

검토일: 2026-10-07. 소스·저장 계약·현재 파일 SHA만 읽었다. 모델 생성/학습/예측, 실제 query 실행, 보류 정답 숫자 열람, 성능 채점, GPU 실행은 하지 않았다.

## 판단

raw3의 실제 y SHA, 생성자 계약, 실제 import 경로, 재개 imputer·수치감사 계약은 이전 지적을 구체적으로 보완한다. 현재 확인한 증거에서 새로운 학습 누수는 찾지 못했다. 그러나 **등록3는 실행 감사 영수증과 그 감사 소스의 연결을 직접 강제하지 않는다. 등록 전 새 버전에서 보완해야 한다.** 또한 현재 `complete.json`과 raw fit 등록 파일 모두 없다. 66폴드 준비·독립 검산 완료 전 학습 차단은 유지한다.

## 실제 확인한 증거

- runtime, resume synthetic, statistics independent receipt의 `code_sha256`가 각각 capture/audit/crosscheck 현 소스 SHA와 일치한다. runtime과 synthetic이 가리키는 raw3 SHA도 현 raw3와 일치한다.
- 런타임은 Python 3.12.10 / NumPy 2.5.3 / pandas 3.0.1 / sklearn 1.9.1 / LightGBM 4.7.0을 기록한다. ET/LGB/MLP × 47/1414/6464의 정확한 9개 생성자 계약이며 fit/query prediction 0이다.
- runtime source pin 19개 중 **18개 MATCH, pandas `__init__.py` 1개는 접근 거부로 독립 읽기 불가**다. 이를 19개 전부 재검증 PASS라고 보고하면 안 된다. 저장 capture 자체의 실제 SHA 기록과 등록기의 향후 전수 SHA 검사와 구별한다.
- 재개 합성 영수증의 13개 검사 PASS는 원 학습/실제 query/보류 정답 0인 계약 검사다. 실제 66폴드 학습·재개 성공이나 LGB/MLP의 모든 고유 전처리 상태 감사 증거로 확대할 수 없다.
- 통계 독립 검산은 저장 200,000 draw / 87블록 / 8,640행 membership 및 상수 손실·누락·비유한 값 8개 검사를 보고한다. 독립 구현은 Counter/struct를 쓰지만 동일 지정 RNG를 재생하므로 RNG 자체를 독립화한 검산은 아니다.
- 추가 integrity 소스는 source/draw SHA, layout digest, Python 3.12, farm별 count 총합, DIAG/TM 양의 분모를 검사한다. 합성 10개 모두 PASS이며 감사·guard recorded SHA는 현 소스와 일치한다. 저장 integrity는 `full_model_score_gate=false`, `heldout_truth_loaded=false`를 명시한다.

## 등록 전 필수 보완

### ORR01 — PASS 영수증과 실행 감사 소스 연결

등록3는 `runtime.runner_sha256`, `synthetic.runner_sha256`, `statscheck.registration_sha256`를 검사하지만 각 영수증의 `code_sha256`와 현 실행 감사 소스 SHA를 비교하지 않는다. `extras`에 현 감사 소스를 pin하는 것만으로는 과거 PASS가 그 소스로 생성되었음을 입증하지 못한다. 현재 실제 파일은 일치하지만 재개·향후 변경에 대한 fail-closed 조건이 빠져 있다.

새 등록기는 runtime→capture, synthetic→resume audit, prepcheck→preparation crosscheck, statscheck→statistics crosscheck 연결을 각각 검사해야 한다. statscheck의 8개 검사 모두 PASS, 정확한 87블록/8640행 및 no-target/no-fit도 검사한다. 새 integrity/audit 영수증을 필수 증거로 사용할 경우 해당 code/guard SHA와 statistics 등록 SHA를 검사하고 전이 source pin에 넣는다. source map을 합칠 때 `extras`도 동일 경로 hash 충돌을 거부한다.

### ORR02 — 등록기 최적화 실행 차단

raw3는 `-O`를 명시 거부하지만 등록3는 주요 검증이 `assert`이고 자체 `-O` 거부가 없다. 새 등록기 시작에 `__debug__`/`sys.flags.optimize`를 확인하고 `RuntimeError`로 차단해야 한다. 정상 실행 명령을 쓰겠다는 약속과 파일 내부의 차단은 별개다.

위 항목은 pinned 등록3를 고치라는 뜻이 아니다. 새 이름으로 보완한 등록기·등록을 사용해야 한다. 실제 준비가 완료되지 않아 지금 등록기를 실행하지 않은 판단은 타당하다.

## raw3에서 수용한 변경과 남은 한계

`fit_one`은 production loader의 `require_all66=True`, loader SHA, 준비 완료 SHA, 유한한 y/행수/실제 little-endian y SHA를 확인한다. 9개 등록 생성자 계약 및 runtime paths를 매 fit과 재개 시 대조한다. train median/all-missing/effective columns, contract/row IDs/pred digest, audit 수·유한성·비음수성·최댓값·fit/query 행수의 정확한 재개 검사가 있다. baseline 594 + domain ET 4752 = 5346개, 폴드당 81개, 모든 24family를 수행하고 완료 요약 재개 비교도 유지한다.

다만 raw 수치감사는 reverse 전행, scattered 최대32행, 첫/중간/끝 최대3 prefix다. 66폴드 전체 query causal 경계를 전수 증명하는 검사가 아니다. 이후 whole-model causal gate에서 전수 prefix 및 후처리를 확인해야 한다. MLP early stopping의 .12 내부 split은 fold train 안에서만 이루어지지만 시간 블록 검증과 동일한 내부 검증이 아니며, baseline 고정 정책이라는 한계를 유지한다.

ET fit jobs4→predict jobs1 전환은 소스에 있으며 기록 생성자 계약은 전환 전 n_jobs4다. 이를 fit 후 runtime 파라미터 기록으로 부르면 안 된다. lock 파일 최초 기록이 try 밖에 있어 기록 실패 때 stale lock이 남는 회복 한계도 있다. 정확 PID/start time 확인 없이 자동 제거하지 않는다.

## 전체 모델·통계 경계

등록 recipe는 R3(.6 ET/.3 LGB/.1 MLP) .6 + train-only cached PFN .4, context5..8/2000행/n_est4, 한 번 shrink, train bounds clip, SG2 원 `rowday>=179` 정책을 명시한다. 이것은 prospective CPU 기준선 정책이다. 과거 GPU 제출과 같다는 주장도, 과거 WT2의 all-TM SG2 적용과 같은 기준선이라는 주장도 허용되지 않는다. RAW_PASS SG2를 후에 QUERY_ROLE로 바꾸거나 BLK 결과로 선택하면 별도 변경이다.

PFN/후처리는 아직 구현·실행·감사 완료가 아니라 recipe다. raw 완료만으로 원 TM111/P2LOO/EL1 채점이나 채택을 열 수 없다. 이후 scorer는 전체 cached PFN/R3+조립/후처리 전수 gate 및 독립 검사, 현 source/draw/layout/결과 SHA를 **정답 parse 전** 확인해야 한다. 24후보 × 기존 3seed 방향, DIAG/TM 양 p<.025/84 교집합은 미래 scorer에서 기계적으로 강제해야 한다. 현재 통계 utility와 integrity에는 전체 모델 gate가 없으므로 OS01/OS02는 그 미래 경계가 구현될 때까지 열린다.

BLK54의 소급 변경, BLK FAIL에 따른 원24 제외, 이 탐색의 채택 허가, 최초 미사용 seed/layout 1회 판정 대체는 모두 허용되지 않는다. 통계 draw 무결성의 malformed-file 검사 누락은 추가 integrity 합성10개로 보완되었지만, 이것이 모델 누수나 성능을 검증한 것은 아니다.
