# 실제 A 중첩 OOF 실행 전 독립 비평 v1

검토일: 2026-10-06 · 집 코덱스 · 독립 비평 담당. 실제 학습 결과 0개인 시점의 설계/소스 검토다. 기존 source를 수정하거나 모델을 학습하지 않았다.

## 판정

**학습 로직에서 외부 정답이 내부 모델 입력에 들어가는 명백한 결함은 발견하지 않았다. 그러나 v1 상태로 장시간 학습을 시작하는 것은 보류 권고한다.** P1 검산 및 중단 재개 결함을 새 버전으로 먼저 보완하면 기존 분할·모델 설정을 바꾸지 않고 실행할 수 있다. 성능 개선이나 규정의 완전 충족을 뜻하는 판정은 아니다.

실제 읽은 파일: plan_v1.md, run_v1.py, verify_v1.py, prelaunch_v2.json, preparation_v1.json의 선택 키, 원 A 사례 run_v1.py, support.py, EC core run.py, season.py 및 사용자 검증 규칙. preparation 전체를 출력하지 않았다. registration_v1.json은 검토 시점에 아직 존재하지 않아 등록의 실제 완료 여부는 검증하지 않았다.

## 독립 재계산으로 확인한 범위

- stdlib로 preparation 80개 record의 ID 유일성, 학습/query/외부 query 분리, 같은 농장 query ±1일 제거, 외부 학습 ID 범위 포함을 다시 검사했다.
- 각 farm 날짜를 정렬해 `(i//5)%4`를 독립 구성했고 query ID와 순서까지 일치했다. 각 외부 fold의 내부 query 합집합은 외부 학습 행과 정확히 같으며 중복이 없다.
- 외부 DIAG10=10, A=5, B=5, 합계20; 내부 4개씩80. train=3,840~4,440행, query=1,392~1,728행. 시드3개 기준 예정 시간행 출현370,800개, h0/6/12/23 예정 사례행 출현61,800개다. 이는 고유 독립 표본 수가 아니다.
- core.features는 같은 농장·날짜의 h0와 현재까지 expanding 특징만 사용한다. train/query는 날짜 단위로 나뉜다. seasonal에서 벡터는 내부 학습 날짜 키로 한정되고 query 날의 전체 외기 벡터를 mapping에 넘기지 않는다.
- ET/LGB/MLP 설정과 혼합·평활·clip은 원 소스와 대조했다. MLP 학습행 전처리는 pipeline 안에서 적합하고 LGB 입력에는 imputation을 추가하지 않았다. thread 수만 별도 지정한다. 실제 같은 환경의 재적합 일치 여부는 아직 실행 전이라 미확인이다.

## P1 — 실행 전 보완 권고

### 1. 검산이 원 정답과 y의 일치 여부를 확인하지 않는다

근거: verify_v1.py:58~75는 CSV `y`의 일평균으로 flags를 다시 만들 뿐 원 공개 OOF 정답 및 preparation의 `query_target_sha`와 비교하지 않는다. 잘못된 정답과 그 정답에서 생성된 flags가 함께 저장되어도 정합 검사를 통과할 수 있다. 생성 코드가 `b.sub_ec`를 사용한다는 사실은 독립 검산의 대체물이 아니다.

개선: 등록된 공개 OOF SHA를 확인하고 DIAG10 seed7 원 정답을 row_id별로 복원하여 모든 CSV y와 직접 대조한다. 각 record query 순서의 y array SHA도 `query_target_sha`와 비교한다. farm/day/hour는 row_id 파싱값과, v/k/s는 파일 문맥과 대조한다. 학습 정답 SHA와 clip min/max도 재계산하면 더 좋다.

차단 범위: 원 정답을 검증한 사례 집계 및 완료 주장. 학습식 자체를 변경할 필요는 없다. 완료 검산을 신뢰할 수 있게 실행 전에 보완할 것을 권고한다. 신뢰도 높음(코드상 미검사 확인).

### 2. 완료 검산 재실행이 항상 파일 생성에서 실패한다

근거: verify_v1.py:85,87,89,94는 기존 verification JSON·문서·ZIP·bundle을 모두 exclusive create로 쓴다. 정상 완료 후 verify_v1.py를 다시 실행하면 첫 proof 저장에서 FileExistsError가 발생한다. 같은 파일:86의 설명은 정상 완료 후 이 명령으로 검산하라고 안내하므로 구현과 설명이 모순이다. stdlib 임시 파일로 exclusive-create 재실행 실패를 확인했다.

개선: 검산의 기본 동작을 read-only로 만들고, 산출물 작성은 첫 finalize에서만 수행한다. 이미 존재하는 산출물은 새 계산과 구조/해시를 비교해 검산하고 변경하지 않는다. 새 검산 결과가 필요하면 명시적으로 새로운 버전 이름으로 저장한다.

차단 범위: 재현/반복 검산 및 정상 완료 선언. 신뢰도 높음.

### 3. 마지막 단계에서 중단되면 같은 코드로 재개할 수 없다

근거: run_v1.py:135에서 nested_cases를 만든 후 136 receipt 또는 137 verifier가 실패하면, 다음 실행은 동일 cases 존재를 금지하는135에서 실패한다. verifier도 proof 작성 후 문서/ZIP 오류가 나면 기존 proof 때문에 재실행이 막힌다. lock을 자동 삭제하지 않는 것은 적절하지만 lock 문제를 해결해도 마지막 단계가 재개되지 않는다. 개별 NPZ→JSON 작성도 비원자적이라 NPZ만 있고 JSON이 없는 중단 상태에서 cache()가 실패한다.

개선: 완성 캐시/CSV는 기존 해시와 새 계산을 비교해 재사용한다. 계산 완료·receipt·검산·문서·ZIP 단계를 분리하여 독립 재개 가능하게 만든다. 새 캐시는 임시 파일에 작성→내용 검사→완성 이름으로 이전하는 방식을 고려한다. 불완전 기존 파일을 삭제/덮어쓰지 말고 분리 보존 후 새로운 attempt 이름으로 복구한다. lock은 PID·source·실제 프로세스 확인 후 명시적 복구 절차로 취급한다.

차단 범위: 현재 계획이 보장한 중단 재개. 장시간 계산의 낭비 방지를 위해 시작 전 보완 권고. 신뢰도 높음.

### 4. CSV 숫자의 NaN을 오차 검산에서 놓칠 수 있다

근거: verify_v1.py는 NPZ raw의 finite만 검사한다. CSV의 raw_A/A/prefix_A/raw_pfn/y 등에 finite를 확인하지 않는다. `max(0., abs(float('nan')))`는 0.0이므로65,73의 오차 누적은 NaN을 반영하지 못할 수 있다. stdlib로 이 동작을 확인했다. 일부 flags 검사가 다른 오류를 잡을 수 있지만 finite 전수검사를 대신할 수 없다.

개선: 모든 수치열의 finite·길이·범위를 별도 전수검사한 뒤 오차를 계산한다. raw_r3도 해당 R3 NPZ raw와 직접 대조한다. NaN 오염 사례를 만들어 verifier가 확실히 거부하는지 확인한다.

차단 범위: 검산 PASS의 무오류 주장. 신뢰도 높음.

## P2 — 완료 범위/서명 보완

### 5. 시드 합의 CSV가 계획에 있지만 구현에 없다

근거: plan은 경계 사례 및 시드 합의 CSV를 약속한다. run_v1.py:135는 nested_cases만 생성한다. verify의 case_counts도 시드별 집계여서 3시드가 같은 날에 합의했는지 보존하지 않는다.

개선: `(v,k,farm,day,hour)`별 세 시드가 정확히 존재하는지 확인하고 high·hard_high·missed_high·hard_low 투표 수, prefix 최소/최대, 평균 등을 별도 합의 CSV로 저장 및 독립 검산한다. 또는 시작 전에 약속 범위를 축소하되 사용자 목표와 맞는지 설명한다. 모델 학습 차단 사유는 아니지만 완료했다고 하려면 해결해야 한다.

### 6. 검산에서 runtime/dependency/kind 서명이 부분만 확인된다

근거: verify_v1.py:34~35는 준비 SHA·seed·record SHA만 비교한다. metadata.signature의 runtime/dependencies/kind를 prep 및 현재 실제 파일과 대조하지 않는다. receipt.source_sha, component_files, outer_oof_files와 실제 파일 집합의 유일성/완전성도 직접 검사하지 않는다. run은 p 재계산 비교를 하므로 원 실행에는 방어가 있으나 standalone verifier의 방어는 더 약하다.

개선: 각 JSON signature 전체를 기대값과 비교하고 dependency 파일 현재 SHA를 확인한다. 런타임/체크포인트 서명도 확인하되 캐시 검산과 실제 재학습에 필요한 환경을 구분해 정확히 기록한다. receipt 파일 집합을 예상560 NPZ+60 OOF+cases(+consensus)와 대조한다. R3 et/lgb/mlp/train_raw 및 PFN context 배열의 길이/finite도 검사한다.

## 해석 제한과 후속 개선

- 현재 내부 query에서 전체 날짜 정답 평균으로 high를 정의하는 것은 지도학습 label 용도다. 미래 시점의 값을 예측 input으로 쓰지 않으면 그 자체는 누수가 아니다. 향후 구별기 학습에서는 y/high/다른 flags를 input에 넣지 않는다.
- 동일 날짜의 두 온실/농장 입력이 비슷할 수 있으므로 ±1일 동일 농장 purge만으로 모든 날씨 상관을 제거했다고 말할 수 없다. 기존 A와 동일한 평가 설계를 유지한다는 목적은 충족하지만 새로운 독립 일반화 성능의 증거는 아니다.
- MLP early stopping은 행 단위 무작위 내부 holdout이고 인접 시간행이 겹친다. 원 A와 동일성 때문에 이번 실행에서 임의로 바꿀 사안은 아니다. 과적합 방지의 독립 날짜 검증이라고 설명하면 안 된다.
- season 적합은 허용된 학습집합의 하루 외기를 사용한다. query 미래 입력을 보지 않는 것은 확인했다. 운영위 원 PDF 미확보이므로 규정의 전면 적합을 독립 확정하지 않는다.
- 여러 외부 fold에 같은 날짜가 반복되므로370,800행/61,800사례를 독립 n으로 사용하지 않는다. 후속 구별기 학습은 외부 fold별 OOF와 그 fold의 외부 A를 짝지어야 한다. threshold 재선택과 OOF 전역 재분할을 하지 않는다.
- 첫 PFN8행 배치 재검사와 첫 R3/PFN 재적합은 유용하지만 모든 query의 배치 인과성과 모든80문맥의 bitwise 재현을 증명한 것은 아니다. 보고서의 검증 범위 표현을 현재 감사 수준에 맞춘다.

## 최종 실행 게이트

1. P1 1~4를 새 verifier/run 버전에 반영하고 합성 오염 거부·재검산 두 번·최종화 중단 재개를 확인한다.
2. 계획의 시드합의 산출물을 구현하거나 범위를 명확히 고정한다.
3. 새 코드/plan/preparation SHA를 실제 첫 fit 전에 등록한다. v1 기존 파일은 보존한다.
4. 완료 후 원 공개 정답 대조,560구성원/60OOF/370,800시간행/61,800사례행 예상포괄성 및 산술 독립 검산을 수행한다.
5. 그 뒤 별도의 혹독한 결과 비평을 진행한다. 현재 보고서로 성능 개선을 결론짓지 않는다.
