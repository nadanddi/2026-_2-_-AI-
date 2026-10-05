# 첫 내부 문맥 R3 실제 저장 결과 독립 비평

2026-10-06 · 집 코덱스 독립 비평 담당. DIAG10/외부0/내부0의 R3 시드7·101·2024 저장 캐시만 검사했다. 새 모델 적합, worker 재시작, source 변경, query 성능 채점은 하지 않았다. PFN·최종 A·80문맥 완료의 판정이 아니다.

## 현재 판정

**첫 R3 세 시드 저장물의 구성·서명·ID·산술은 독립 검사 PASS다. 그러나 실제 partial 검산기에서 BOM 처리 오류가 발생했으므로 전체 완료 검산 경로는 보완이 필요하다.** 현재 캐시 성공을 성능 개선이나 전체 OOF 성공으로 확대해서는 안 된다.

## 직접 확인한 실제 값

| 시드 | 학습행/날짜 | query행/날짜 | fsum 혼합 최대오차 | 첫 재적합 기록 최대오차 |
|---|---:|---:|---:|---:|
| 7 | 4,128 / 172 | 1,680 / 70 | 2.220446049250313e-16 | 2.220446049250313e-16 |
| 101 | 4,128 / 172 | 1,680 / 70 | 2.220446049250313e-16 | 미수행(null) |
| 2024 | 4,128 / 172 | 1,680 / 70 | 2.220446049250313e-16 | 미수행(null) |

- NPZ actual SHA와 JSON SHA 일치, record SHA·prepared SHA·시드·runtime·dependencies가 preparation_v4 첫 record 및 준비값과 일치했다.
- row_id는 등록된 query와 순서까지 일치하며 train_row_id도 동일하다. 학습/query/외부 query 사이 교집합0, 같은 농장 query ±1일 purge 충족을 독립 재검사했다.
- raw/et/lgb/mlp는 각각1,680개, train_raw는4,128개이며 shape·finite 전수검사를 통과했다. `.6ET+.3LGB+.1MLP`를 Python math.fsum으로 재계산한 결과가 표의 최대오차다.
- 원 공개 OOF를 utf-8-sig로 독립 읽고 첫 record의 train/query target SHA 및 bounds를 재구성하여 등록값과 일치함을 확인했다. FULL/BASE 특징 이름에는 정답·high·hard/missed flags가 포함되지 않는다.
- 첫 재적합 오차는 저장 JSON의 실제 run 감사기록을 확인한 값이다. 독립 비평 담당이 재학습한 값이 아니며7번 시드에만 해당한다. 최초 코드가 fresh clone 재학습 후 이 수치를 기록하는 점은 사전 소스 검토로 확인했다.

## P1 — 부분 검산 실패의 실제 원인 및 개선 피드백

partial_audit_v1.log의 verify_v4.py:35에서 `KeyError: row_id`를 직접 확인했다. 원 public OOF는 row_id 열이 있으며, 파일을 `encoding='utf-8'`로 csv.DictReader에 넘겨 헤더가 `\ufeffrow_id`가 된 것이 원인이다. stdlib로 fieldnames와 첫 행을 확인하여 원인을 재현했다. row_id 열이 없다는 해석은 맞지 않으므로 초기 의심을 BOM 원인으로 정정했다.

개선: 원 CSV를 변경하지 않고 새 독립 검산기의 rows()에서 `encoding='utf-8-sig'`로 읽는다. 원 등록 run_v4/verify_v4를 현재 worker 실행 중 수정하거나 학습을 재시작하지 않는다. 외부 별도 verifier 버전으로 기존 source/준비/캐시 서명을 대조하고 final BOM 실패 후 남는 lock을 실제 worker 종료 확인 뒤 처리하여 검산을 이어갈 수 있다. 단, 등록한 원 verifier가 실패한 사실과 독립 보완 verifier의 PASS를 구분해서 기록해야 한다.

차단 범위: v4 partial 및 자동 전체 verifier의 실행. 현재 R3 캐시의 산술/ID 검사는 별도 경로에서 수행했으므로 해당 오류가 R3 적합 결과를 무효화한다는 근거는 없다. 성능 모델·분할·입력 변경이나 재학습으로 해결할 사안이 아니다. 신뢰도 높음.

## 혹독한 해석 비평

1. **“R3가 잘 맞는다”는 주장은 허용되지 않는다.** 저장 JSON의 train_raw_rmse는 in-sample 값이며 독립 query 성능과 같지 않다. 이번 비평에서 query RMSE를 계산하지 않았고 부분 성능 판정을 하지 않았다. 작은 학습 오차는 과적합 여부를 해소하지 못한다.
2. **“재현성이 전 시드·전 문맥에서 입증됐다”는 주장도 허용되지 않는다.** seed7 첫 문맥 재적합만 실제 감사됐다. 다른 시드는 반복미수행으로 명시한다.
3. **“실제 A가 완료됐다”는 표현은 이르다.** 실제 A는 R3와 PFN4 평균의 혼합 및 평활·clip까지 필요하다. 현재 이 보고서는 R3 캐시3개만 평가한다.
4. **외부query 정답을 쓰지 않았다는 확인은 유지된다.** 해당 ID가 내부 train/query에 모두 배제됐고, 정답이 feature 목록에 없다. 모델 적합 자체의 범위는 실행 코드 및 ID 서명에 기반한다. 미저장 학습객체 내부를 검토했다고 과장하지 않는다.

## 다음 피드백

먼저 BOM 보완 독립 검산기를 만들고 같은 첫 R33개를 partial 재검산한다. PFN 첫 문맥 완료 뒤 문맥 선택·query/외부query 분리·batch8·repeat·finite를 별도 검사한다. 전체80문맥 결과가 준비되면 독립 전체 검산과 최종 결과 비평을 수행하며, 현재 worker를 그대로 유지한다.
