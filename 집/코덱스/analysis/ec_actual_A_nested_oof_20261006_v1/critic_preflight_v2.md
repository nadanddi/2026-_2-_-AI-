# 실제 A 중첩 OOF v3 독립 재검토

2026-10-06 · 집 코덱스 독립 비평 담당. 실제 모델 fit 없이 run_v3.py / verify_v3.py / preparation_v3.json을 재검토했다. 기존 source 및 기존 보고서는 수정하지 않았다.

## 판정

**v1에서 제기한 실행 전 P1 네 건은 v3에 반영됐다. 현재 추가로 실행을 차단할 명백한 결함은 발견하지 않았다.** 준비 서명 등록·등록 후 SHA 확인·첫 실제 재적합 감사가 완료되기 전까지 실제 실행 및 재현 PASS를 주장할 수 없다. 완료 뒤 결과 독립 검산과 혹독한 결과 비평은 여전히 필요하다.

## 모델과 분할 변경 여부

AST를 독립 파싱해 v1/v3의 `models`, `r3`, `pfn`, `splits`, `prep`, `runtime`, `ordered`, `ids`, `ar` 함수가 정확히 동일함을 확인했다. 모델 적합·입력·시드·PFN 샘플 선택·혼합식·분할 알고리즘의 변경은 없다. 변경은 저장·서명·최종화 및 검산/합의 산출물에 있다.

preparation_v3가 생성된 뒤 `records`, `runtime`, `inputs`, `full`, `base`, `core_params`, `seeds`, `pfn_seeds`, `pfn_config`, `threads`를 v1과 직접 대조해 완전 일치했다. 80개 문맥의 모든 ID·target SHA·feature SHA·season notes·bounds가 records 비교에 포함됐다. v1에서 독립 재계산한 외부20/내부80 분할, 예정370,800시간행·61,800사례행 출현은 그대로다.

## 문제별 개선조치 수용 여부

| v1 지적 | v3 확인 내용 | 판정 |
|---|---|---|
| 원 정답 대조 누락 | 공개 OOF 입력 SHA 확인→DIAG10 seed7 원 정답 row_id별 복원→각 OOF y 대조, record query_target_sha 별도 대조 | 수용 |
| 검산 재실행 x모드 실패 | `--check-only`는 새 proof를 다시 계산해 기존 proof와 비교하고 파일을 새로 만들지 않음; 정상 finalize도 JSON/doc 동일성 비교 후 재사용 | 수용 |
| 최종화 중단 재개 불가 | CSV compare-or-create, JSON/doc/ZIP 부분 파일 작성 후 promotion, 기존 ZIP entry SHA 대조; cases/receipt 기존 존재만으로 실패하지 않음 | 수용 |
| CSV NaN 오차 검산 누락 | y/y_day/A/raw_r3/raw_pfn/raw_A/clip_lo/clip_hi/prefix_A finite 검사 후 산술 감사 | 수용 |
| 시드 합의 CSV 없음 | v/k/farm/day/hour별 high·y_day·prefix최소최대·세가지 flags 투표 생성, 독립검산에서 시드3개와 투표/범위 재계산 | 수용 |
| signature 부분 검사 | metadata runtime/dependencies/kind를 prep와 대조, 실제 dependency 소스 및 원 입력 SHA 재확인 | 주요 부분 수용 |

구성원 NPZ/JSON은 같은 staging 디렉터리에서 완성한 뒤 전체 디렉터리를 완성 위치로 rename한다. 따라서 NPZ만 완료 위치에 있고 JSON이 아직 없는 두 단계 저장 문제를 피했다. 중단된 staging은 보존되고 완성 캐시로 오인되지 않는다. lock 자동 삭제를 금지한 정책은 유지되어 있어 비정상 종료 후 실제 worker 종료 및 PID/source 확인이 필요하다.

## 학습 없는 독립 합성 확인

- run_v3와 verify_v3에서 실제 `load`/`save` 함수 AST만 추출해 stdlib 임시 폴더에서 실행했다. 첫 저장→같은 JSON 재저장으로 기존 byte 유지→다른 JSON 저장은 AssertionError로 거부를 각각 확인했다. source를 import하지 않아 모델·데이터 bootstrap 및 학습은 실행하지 않았다.
- 임시 stage 디렉터리에 NPZ/JSON 두 파일을 만든 뒤 디렉터리 rename으로 두 파일이 함께 완성 위치로 이동함을 확인했다. 이는 코드 경로의 보통 성공 동작 확인이며 실제 정전/디스크 장애의 fsync 내구성까지 검증한 것은 아니다.
- 실제80문맥 캐시와 전체 CSV가 아직 없으므로 전체 `--check-only`를 두 번 실행한 결과는 아직 없다. CSV exact round-trip과 후처리1e-12 일치도 역시 실제 완료 후 확인할 항목이다.

## 남은 P2 — 이번 학습을 차단하지 않지만 완료/후속 사용 때 보완

1. **CSV raw_r3 직접 검산 누락.** finite 검사와 `.8*NPZ R3+.2*PFN` raw_A 복원은 한다. 그러나 저장된 CSV raw_r3 값 자체는 NPZ raw와 대조하지 않는다. 향후 gate input으로 raw_r3를 쓴다면 OOF 예측의 잘못된 부수열이 입력에 들어갈 수 있으므로 직접 대조를 추가하거나 후속 데이터 준비 단계에서 확실히 검증한다.
2. **CSV ID 문맥 파싱 대조 누락.** OOF row_id 순서는 엄격히 검증하지만 각 행 farm/day/hour를 row_id에서 독립 파싱하여 대조하지 않는다. v/k/s도 파일명 문맥과 독립 비교하지 않는다. 생성 코드에는 이들 값이 원 b 및 현재 loop에서 직접 오므로 명백한 생성 오류는 발견하지 않았지만 standalone 검산의 방어범위는 제한된다.
3. **check-only의 bundle 범위.** `--check-only`는 fresh 산술 proof 비교 뒤 return하므로 ZIP CRC/entry SHA 및 bundle SHA는 다시 검사하지 않는다. 최초 정상 finalize와 보통 verifier 재실행은 ZIP을 검증한다. 따라서 사용자 보고에서 check-only를 재현 ZIP까지 검증한 것으로 표현하면 안 된다. 완료 후 bundle을 따로 검산하거나 check-only에도 bundle 검사 경로를 더할 수 있다.
4. **receipt 집합/부수배열 검산 범위.** receipt.source_sha, count 숫자, file list의 정확한 예상 집합·중복 금지, R3 train_raw/et/lgb/mlp의 모든 배열 shape·finite가 별도 전수검사되지는 않는다. 필수560 구성원/60OOF를 verifier가 직접 열고 계산하는 핵심 포괄성은 확인한다. cache metadata runtime은 prep 기록과 같음을 보일 뿐 standalone 검산 시 현재 런타임 전체를 다시 계산한 비교는 아니다. 실 재학습에서는 prep 재계산 equality가 이 차이를 보완한다.

## 결과 해석과 다음 게이트

준비·코드 검토의 통과는 실제 캐시 검산이나 성능 통과를 뜻하지 않는다. 첫 DIAG10/k0/j0 R3 seed7 및 PFN seed1 재적합, PFN query 첫8행 배치 비교가 통과한 뒤 계산을 계속해야 한다. 전체 OOF 완료 후 원정답·포괄성·후처리·시드합의·캐시 SHA를 독립 확인하고 결과 비평을 수행한다.

자동 문서가 확정 사용자 보고 전에 혹독한 비평을 요구하며 새 구별기 성능/RMSE 개선을 주장하지 않도록 바뀐 점은 적절하다. 여러 외부 fold에 반복되는 날짜는 독립 표본으로 세지 않고, 다음 구별기는 외부 fold별 OOF와 해당 외부 query 예측만 짝지어 사용한다. 이번 검토로 미래 input 금지 규정 전체의 원문 적합을 확정하지 않으며 운영위 원 PDF 미확보 한계를 유지한다.
