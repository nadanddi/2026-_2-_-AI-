# family21 whole verifier v1 독립 비평

2026-10-04 · 집 코덱스 · ec_lgb_verifier_critic

대상: `verify_full_v1.py` 정적 소스. 실제 candidate CSV/점수/부분 출력은 읽지 않았고 fit/predict/score/worker 실행은 0이다. 부모가 보고한 합성48 PASS는 독립 실행하지 않았다. 이 문서는 실행 결과가 아닌 코드 감사이다. 부모 설명대로 v1은 placeholder 때문에 실제 verify 시작에서 중단된다. 아래는 그 placeholder만 채웠을 때에도 남는 결함이다.

## 실제 검증 전에 막아야 하는 항목

1. **actual whole/aggregate 미검사.** `fit['cells']`는 길이66만 검사한다. 중복·다른 family/cell·내용 위조도 통과할 수 있다. 각 실제 meta의 고정 순서 목록을 모아 fit cells와 완전 동일하게 비교하고, expected 22fold×3seed 키/순서와 검사한다. `oof.csv`는 전혀 읽지 않으므로 누락/오염 aggregate도 PASS한다. aggregate schema·키·순서·83160행·전 열을 실제66 CSV 연결본과 비교하고 whole manifest/aggregate SHA까지 확인해야 한다. 모든 완료검사 이전에 rmse/boot 호출 금지.

2. **fresh candidate signature 부재.** 새 meta는 status/CSV SHA만 검사한다. 23특징 학습 여부, BASE14 순서, DP1 순서, seed, fresh single LGB만 교체했는지, 모델 params, train/query IDs/특징/공개 targets/bounds, runtime, input/cache/dependency SHA를 계산해 expected signature와 정확히 일치시켜야 한다. CSV SHA는 파일-기록 정합이며 모델 출처 증거가 아니다. source 핀은 run만 검사하고 import한 실제 source·env·support·core 및 준비 manifest 22개 재구성이 없다.

3. **first audit 부채.** first status PASS 및 첫 meta의 SHA만 검사한다. NaN/음수/문턱 초과 error, 다른 signature, 잘못된 first cell, 빠진 fresh_fit/order/single/prefixmodel 감사, 원 BASE14 재현 실패를 전부 허용한다. first의 exact schema, identity/signature, RAW/FINAL 문턱, 유한·비음수 수치 및 전 error 집합, row/hour coverage, first original BASE14 결과와 source/cache SHA를 확인해야 한다. 저장 감사의 수치·SHA 확인을 예측 replay라고 표현하면 안 된다. 다른65 meta에 first SHA가 들어가는 것도 거부한다.

4. **원 R3 함수 재사용은 provenance 전체 검사가 아니다.** 부모 TabDPT `guard_original_r3` 내부는 public 목표·ID·member 산술·bounds·metadata key/digest를 확인한다. `provenance.shared.input_sha256['train_X.csv']`, `shared.core_sha256`, `environment` 고정 검사는 그 함수 밖 TabDPT caller에 있었다. LGB caller에도 이 검사 및 metadata SHA 기록이 필요하다. `old['r3_seed']`와 `r3['raw_r3']`, `old['baseline_seed']`와 baseline을 직접 비교해야 한다. baseline NPZ/PFN4/context provenance·SHA/cacheguard/trainingmeta source 검사도 없으며 PFN context는 ordered IDs 일치 외 중복·정확 schema·유한성·길이와 pinned receipt를 확인해야 한다.

5. **CSV/JSON exact schema 결여.** CSV 필요한 속성은 접근하지만 추가/중복 열을 거부하지 않는다. 수치열 shape/finite, key dtype(정수/문자), exact column order, JSON 정확 필수 key/금지 key·version·family·run identity를 고정한다. `len(folds)==22`, `len(scores)==15`만으로 verifier 이름/원래 fold 구성/seed별 행 수가 고정되지는 않는다. fold keys/각fold orderedIDs/query count, validator×seed 15개의 정확 집합을 검사한다.

6. **실제 runtime 미핀.** env/numpy/pandas는 top-level에서 import하지만 버전/실제 module path/source digest 검사 없음. imported R3 verifier의 `bootstrap_runtime`은 support SHA만 확인하며 TabDPT runtime을 검증하지 않는다. LGB용 pinned Python/numpy/pandas/sklearn/lightgbm 및 source/runtime probe 결과를 runner/prep/first/66meta에 동일하게 묶어야 한다. ACL오류는 readonly 진단 대상이며 버전핀 완화 근거가 아니다. `python -O`에서는 assert 제거되므로 `sys.flags.optimize==0`을 fail-closed 방식으로 보장하거나 핵심 검사에 명시 예외를 사용한다.

## 산술·bootstrap·판정

- flatten .48/.24/.08/.20 + scalar farm/day prefix + one final clip은 독립 산술 경로로 적절하다. raw-old=.24ΔLGB 대조도 있다. baseline 역시 core.shrink만 믿지 말고 scalar baseline 경로를 추가한다. lo/hi는 공개 train 범위 대조하지만 finite·lo≤hi를 명시한다.
- `rmse`는 fsum과 NumPy 수치를 비교하지만 엄격 개선 부호 일치검사가 없다. 극소 개선/동률에서 두 경로의 부호가 다르면 fail해야 한다. `candidate < baseline`를 직접 판정하고 pct는 표시값으로 남기는 편이 명확하다. baseline0도 명시 처리한다.
- boot의 두 경로는 같은 rng20261003+seed와 동일 farm순서의 5일 sorted-day block/20k 추출이며 손실합 구성은 독립이다. 이는 올바른 산술 검산 방향이다. 반드시 farm 집합 F13/F47·360farm/day·24hour 완전성, 각seed DIAG8640, block개수/count, tail block 정책을 검사한다. 예상 무작위 index/draw 수, p≥0 동률 포함, adjusted quantile[alpha,1-alpha]와 NumPy quantile 방법(linear)을 manifest로 고정한다. 비교 tolerance 통과는 엄격 p/CI 경계가 같다는 증거가 아니므로 독립 경로에서 p와 CI·판정도 각각 계산하여 일치시킨다.
- v1의 strict15칸 + seed별 p<.025/21 + adjustedCI 상한<0은 부모가 고정한 family21 규칙과 맞는다. HANDOFF의 다른 변경 규칙으로 사후 완화하지 않는다. PUBLIC_PASS_PENDING_REVIEW는 채택 확정이 아니다. 반복 공개 검증·별도 untouched holdout 부재 및 first saved audit만 확인했다는 한계를 result에 저장한다.
- high행744만 검사하지 말고 high31일/ordinary329일을 검사한다. bias도 fsum/NumPy 두 경로로 검산한다. segment 진단은 채택 추가조건으로 뒤늦게 전환하지 않는다.

## 특징 정보 및 보존

- DP1 AST에서 두 함수만 추출하고 전체 동료 모듈을 실행하지 않는 것은 적절하다. 실제 소스의 day_feats는 hour sort와 prefix 연산이며 source읽기 기준 미래행을 직접 참조하지 않는다. 그러나 v1은 full-day 연산과 prefix 재구성/미래오염/row-order 반례를 자체 검사하지 않는다. 23특징 signature는 training/query 값과 BASE14 원재현을 함께 고정해야 한다. DP1 source에는 fillna(0)가 있으므로 실제 운영5열 missing/nonfinite/negative/시간중복 검사를 준비·입력 출처에 묶는다.
- `open('x')`와 사전 exists 검사는 기존 결과 덮어쓰기를 막지만 scoreCSV→segmentCSV→JSON 중간 실패는 부분 결과를 남긴다. 실패 파일을 성공처럼 재사용하거나 지우지 말고 다음 버전 출력으로 복구한다. 최종 JSON에 scores/segments/fit/first/aggregate/prereg SHA와 complete status를 기록해야 한다. 체크 숫자가 많다는 사실은 누락된 검사를 보충하지 않는다.
- CHECKS 전역값은 같은 interpreter 재진입 시 누적된다. 단독 실행·실행명/version/run identity를 기록하고 검증마다 초기화하거나 검사수 의미를 명시한다.

## v2 합성 오염반례 체크리스트

실제 candidate가 없는 합성 schema만 이용한다. 정상66meta/83160 구조 자체를 합성 fixture로 만들거나 검증 함수를 추출해 검사하고, 아래 각 1개 변경이 점수함수 전에 거부되는지 확인한다.

1. fit cells 중복/누락/순서변경; aggregate 1행누락·raw1값변경·행순서변경·추가열.
2. runner/prep/runtime/input/core/support/cache SHA 1개 변경; 23features 순서/값 1개 변경; train/query target/ID 변경.
3. first signature 변경·NaN/음수/초과error·error항목누락·firstoriginalBASE14 실패·다른firstcell·비first meta firstSHA 삽입.
4. 원 R3 provenance shared/runtime만 변경하고 metadata key를 다시 계산: 내부 guard만으로 통과할 수 있는 반례를 caller가 거부해야 한다.
5. ordered train/query/context ID 변경·중복·PFN 값 변경·old R3/baseline 열만 변경.
6. shuffled hour/farm/day·tail block·동률/미세부호차·clip 전후 경계·미래행 운영입력변경 후 prefix 불변·타farm 변경 후 다른farm 불변.
7. synthetic 기존 출력 존재와 실제 결과 목적지 존재: overwrite0 및 부분 산출물을 success로 표시하지 않음.

v1 감사 판정: **실제 whole 사용 불가**. v2 source 핀/fixture 검증이 완료된 후에도 실제66cell 완료 전에는 성능·채택을 판정하지 않는다. 이 감사는 실제 수치 검산, 모델 재현, 전체 데이터 규정 증명을 수행하지 않았다.
