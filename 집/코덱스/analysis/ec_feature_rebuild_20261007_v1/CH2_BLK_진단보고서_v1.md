# CH2 고정 이웃 사슬 BLK 진단

사전 고정한 6개 비교는 모두 통계 기준을 통과하지 못했다. 두 방법은 세 시드 모두 오차를 줄였지만, 효과가 8개 중 같은 한 블록에만 나타났다. 이 결과로 CH2 전체 또는 이웃 원자료의 무용함을 주장하지 않는다.

## 고정 비교와 결과

CPU reference-only cached 기준선, BLK 1,440행·60일·8블록, 시드47/1414/6464. 주 비교는 QUERY_ROLE, RAW_PASS는 적용 범위 진단이다. 기존6+도메인48+이번6=누적60비교, P(worse)<.025/60을 실행 전에 고정했다. 이전 도메인 등록의 alpha54는 수정하지 않았다.

| 범위 | 방법 | 평균 시드 RMSE 변화 | P(worse) | 판정 |
|---|---|---:|---:|---|
| BLK_QUERY_ROLE | FLANK_SOURCE_MATCH | -0.0011914510 | 0.31868407 | 진단 FAIL |
| BLK_QUERY_ROLE | PAST_QUERY_PREFIX_STATE | -0.0002800081 | 0.31868407 | 진단 FAIL |
| BLK_QUERY_ROLE | CH2_REFONLY_GUARD | +0.0000000000 | 1.00000000 | 진단 FAIL |
| BLK_RAW_PASS | FLANK_SOURCE_MATCH | -0.0010606324 | 0.31868407 | 진단 FAIL |
| BLK_RAW_PASS | PAST_QUERY_PREFIX_STATE | -0.0002230161 | 0.31868407 | 진단 FAIL |
| BLK_RAW_PASS | CH2_REFONLY_GUARD | +0.0000000000 | 1.00000000 | 진단 FAIL |

CH2_REFONLY_GUARD는 지원0행이라 기준선과 같았다. FLANK_SOURCE_MATCH는25행, PAST_QUERY_PREFIX_STATE는14행을 지원했다. 활성4비교의 block SSE 변화는 index3에서만 음수이고 나머지7개는0이다. 재표집에서 그 블록이 빠지면 변화0이므로, 0을 악화와 함께 세는 고정 통계에서 p≈.319이다.

## 검증과 한계

- 94개 등록 pin·현재98개 gate pin을 확인했다. 실제360개 forbidden-query 교란/역순 검사, 모든1,440행 prefix 재생,1,560회 소비 로그,25,920개 독립 끝점 산술 검사를 통과했다. 산술 최대차1.11e-16.
- Decimal60 독립 검산은1,152 RMSE·48블록SSE·6 bootstrap p와 CI·576셀 고유키·6요약평균을 확인했다.
- 일반1,368행/고EC72행3일과 앞·가운데·뒤, 농장,24시간별 효과를 사전에 고정한576셀 CSV에 보존했다. 이 사후 구분으로 적용 규칙을 바꾸지 않는다.
- BLK는모두pass1이고 고EC3일뿐이다. 기준선은과거GPU출력과의바이트동일성을주장하지않는CPUcachedrecipe이다. 정답을 이용한 사슬은 물리적인 온실 동이나 실제 시간 순서를 증명하지 않는다.
- 원TM111/P2LOO/EL1 검증과 최초미사용seed/layout1회, 도메인·문헌·전체데이터 단계, 최종 보고서는 남아 있다. 이번6안은튜닝없이종료하며전체목표는계속진행한다.

## 보존한 실패와 보완

등록v1의 CSV 열 이름 오류와 partialCSV6을 보존했다. v2 등록 뒤 query adapter의 전체 schema 선검사·실제 import경로를 보완한v3를 등록한 후추론했다. verifier2의tuple/list직렬화비교실패는실제row600에서원인확인후새verifier3로JSON정규화비교만수정했다. 예측·통계기준은변경하지않았다. score spec2는추론후·정답채점전에pointer수정되었으며created_before_inference=False를기록했다(등록script의기존콘솔문구는이시점과맞지않으므로사용하지않는다).

## 근거

- CH2_BLK_registration_v3.json / feature_candidates_v8.csv
- checkpoints/CH2_BLK_v1/predictions.json / audit.json
- CH2_BLK_verified_gate_v3.json / CH2_gate_serialization_diagnosis_v1.json
- CH2_BLK_diagnostic_results_v1.json / CH2_BLK_diagnostic_cells_v1.csv
- CH2_BLK_score_independent_crosscheck_v1.json
- critique_CH2_actual_gate_v1.md / critique_CH2_scored_diagnostics_v1.md
