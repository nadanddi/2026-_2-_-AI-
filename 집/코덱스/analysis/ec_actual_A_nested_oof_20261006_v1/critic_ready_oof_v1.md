# 완료된 DIAG10 외부0~3 OOF 및 첫 PFN 독립 비평

2026-10-06 · 집 코덱스 독립 비평 담당. 실행 중 worker는 건드리지 않고 고정된 완료 subset만 읽었다. 독립 감사 코드 critic_ready_oof_audit_v1.py와 실제 실행 결과 critic_ready_oof_audit_v1.json을 함께 보존한다. 새 학습·재시작·부분 성능 채점은0이다.

## 판정과 근거 범위

**DIAG10 외부0~3의 내부16문맥·구성원112개·OOF CSV12개의 정답/ID/서명/문맥/산술 감사는 PASS다.** 최대 산술 차이는4.440892098500626e-16으로 사전1e-12보다 작다. 추가적인 실행 차단 문제는 이번 고정 subset에서 발견하지 않았다. 전체80문맥 완료, A/B 검증기 완료, 성능 개선을 뜻하지 않는다.

- 감사한 시간행 출현76,392개, h0/6/12/23 사례행 출현12,732개. 12CSV의 row_id는 각 등록 외부 학습 ID와 순서까지 일치하며, 날짜별24시간 및 내부fold 단일 소속을 검사했다.
- registered source/dependency SHA, 원 공개 정답 입력 SHA 및 NPZ SHA·prepared/record/runtime/dependency/kind/seed metadata를 대조했다. 원 공개 정답을 utf-8-sig로 읽고 첫16문맥 학습/query target SHA 및 학습 min/max bounds도 재계산했다.
- 구성원 raw 및 R3 et/lgb/mlp/train_raw의 shape·finite 전수 확인. CSV의 수치 finite, y 원정답 정확 일치, row_id좌표 farm/day/hour 및 파일 문맥 v/k/j/s 일치 확인.
- 각 내부 query 집합이 외부 학습 행을 정확히 한 번 덮으며 query/외부query가 train/PFN context에 들어가지 않는다. 같은농장 query±1일 purge도 검사했다.

## 첫 PFN 문맥의 엄격 비평

DIAG10/0/0의 PFN seed1~4마다 context2000행·query1680행이다. np.random.default_rng(seed)의 등록 choice를 독립 재생해 context_row_id 순서까지 정확히 같음을 확인했다. context 유일성 및 학습집합 subset이고 query/외부query 교집합0임을 확인했다.

seed1 JSON의 실제 worker 재적합 repeat_maxdiff는0.0이다. seed2~4 repeat는 null로 미실시다. 네 시드 모두 first8_batch_invariance=true가 기록되어 있다. 이것은 worker가 실제 수행한 감사기록을 검토한 것이며 독립 비평 담당이 재학습·재추론한 결과는 아니다. 모든query의 모든 prefix 배치 불변성을 입증한 것은 아니다.

## 독립 산술 재계산

NumPy 평균/원 pandas 후처리를 다시 부르지 않고, Python math.fsum과 명시 날짜별 시간정렬 루프로 재계산했다.

| 비교 | 최대 차이 |
|---|---:|
| .6ET+.3LGB+.1MLP ↔ R3 NPZ | 4.440892098500626e-16 |
| PFN4 fsum 평균 ↔ CSV raw_pfn | 0.0 |
| NPZ R3 ↔ CSV raw_r3 | 0.0 |
| .8R3+.2PFN평균 ↔ CSV raw_A | 0.0 |
| raw 평활 후 내부train범위 clip ↔ CSV A | 2.220446049250313e-16 |
| A 현재까지 prefix 평균 ↔ CSV prefix_A | 2.220446049250313e-16 |
| 원정답 일평균 ↔ CSV y_day | 5.551115123125783e-17 |

high/hard_high/missed_high/hard_low flags도 재계산해 전행 일치했다. 이는 문턱 정의의 구현 일치 검사이며 성능/선택 효용 평가가 아니다.

## 문제점·개선 피드백

1. **부분OOF를 전체OOF로 설명할 위험.** 검토범위는 완료된 DIAG10 외부0~3뿐이다. 현재 진행 중인 DIAG10외부4 및 A/B 결과를 포함했다고 말하지 않는다. 개선: 보고마다 validator/fold/context/CSV 개수를 명시하고 전체80 완료 receipt와 별도 구분한다.
2. **반복 행을 독립 표본 수로 해석할 위험.** 외부fold와 시드에 같은 날짜가 반복된다. 76,392행은 출현 수이며 독립 n이 아니다. 개선: 후속 구별기 데이터는 외부fold별로 짝지어 유지하고 전역 결합 재분할을 하지 않는다.
3. **PFN repeat/batch 기록을 독립 재학습으로 과장할 위험.** 현재 별도 비평은 서명된 기록과 context/산술 재생이다. 개선: first seed1 repeat만0이고 나머지는미실시로 명시한다. 과도한 모든query batch안전 주장 없이 감사범위 first8로 제한한다.
4. **중간 숫자를 좋은 성능으로 오인할 위험.** 이번 감사는 query RMSE·선택률·구별기 AUC를 채점하지 않았다. 개선: 전체 OOF 완료 검산을 먼저 하고 구별기 성능은 별도 사전등록 실험에서 검증한다.
5. **BOM 보완 경로의 provenance.** 원등록 verifier4는 BOM오류를 유지하며 별도 verifier5/finish5가 최종 검산을 처리한다. 개선: 완료 보고에서 원 worker exit와 별도 최종검산 exit를 각각 남기고, 이 부분 독립 감사 PASS로 원verifier 오류가 사라졌다고 설명하지 않는다.

## 다음 게이트

worker·source·분할·계수를 그대로 유지해 남은 문맥을 완료한다. 최종 receipt4와 별도 verification5/bundle5를 확인하고 전체560구성원·60OOF·예정370,800시간행/61,800사례행·seed consensus를 fresh 재검산한다. 이후 독립 최종 비평에서 누수·재현·구성 동일성·결론 제한 및 개선책을 평가한다. 현재 subset 감사로 전체 완료를 선언하지 않는다.
