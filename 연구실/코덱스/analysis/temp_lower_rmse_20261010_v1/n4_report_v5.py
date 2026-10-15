"""Generate a source-linked pilot report after completed arrays/checks only."""
from pathlib import Path
import json
H=Path(__file__).resolve().parent
j=lambda n:json.loads((H/n).read_text(encoding='utf-8'))
r=j('n4_results_v2.json');v=j('n4_verify_all_v1.json');g=j('n4_groups_v2.json')
assert v['passed'] and r['recipe_sha']==v['recipe_sha'] and not r['adoption']
for s in r['results']:
    assert abs(r['results'][s]['base_rmse']-g[s]['overall']['base_rmse'])<1e-12
    assert abs(r['results'][s]['candidate_rmse']-g[s]['overall']['candidate_rmse'])<1e-12
rows=[]
for s,m in r['results'].items():
    rows.append(f"| {s} | {m['base_rmse']:.8f} | {m['candidate_rmse']:.8f} | {m['pct']:+.4f}% | {m['rows']} |")
judge='예비 진입 기준 통과. 채택은 보류하고 별도 확인이 필요하다.' if r['pilot_pass'] else '예비 진입 기준 미달. 이 후보를 채택하지 않는다.'
d=r['results']['DIAG10']
text=f'''# 온도 발견 연결 실험 결과

2026-10-10 연구실 코덱스. {judge}

## 재동기화
집 클로드 Drive3산출물 재복사 완료, 원격 Git 신규0. 확인한 집 일지는10-09까지이며 이후 작업 부재를 증명하지 않는다. `동기화_확인_v1.md` 참조.

## 전체 조합 예비 결과
기준과 후보가 동일 BASE/PFN을 공유하며 CODEX 보완만 바뀐다. 원9회와 동일한 혼합식이나 **PFN 신규49 한 표본**을 쓰므로 원9회8표본이나 평가 RMSE0.5127의 개선을 입증하지 못한다.

| 검증 | 기준 RMSE | 후보 RMSE | 변화 | 행 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

DIAG p_worse={d['p_worse']:.6f}, CI={d['ci']}, alpha=.025/6. 세 검증 방향과 DIAG 유의조건을 실행 전에 고정했다. 기존 데이터 재사용·사후 가설 선택 편향은 이 보정만으로 해결되지 않는다.

## 발견을 연결하며 갱신한 해석
| 기존 발견 | 연결한 시험 | 갱신 |
|---|---|---|
| 다른 농장에도 온도 변화가 있다 |48농장 Δprior LOFO→타깃 누적1열 N2 |41/48농장 prior 개선이 타깃 개선으로 이어지지 않았다. 타깃 DIAG-.0481/EXT10+.2157/EXT12+.4020%, 기각 |
| 하루 수준 오차가 크다 |실제 train y0 수준+변화 분리 N3 |변화 MSE는 줄었으나 전체 악화. 기준과 같은 h0분해에서 일반 검증의 교차항 상쇄 약화와 EXT12 수준 악화를 확인. 공유 anchor의 대수결합을 물리 원인으로 해석하지 않음 |
| 전문가: 배지-공기 분리·관수 수온·흰색 멀칭 가능성 |기존BASE 일사량 대리 특징→IRR4→원8~10℃ gate N4(분석자 가설, 전문가 직접 일사량 설명 없음) |새 관측정보가 아닌 기존 모델 정보의 재배치. 추운 행의 실패 후 만든 사후 가설이므로 새 배치와 전체 혼합으로 확인 |

## 검증과 비평
원 패키지54파일/공식3입력/checkpoint hash 현재 일치. 특징 감사 첫fold4,464,900비교셀 PASS(v1 actual; 수정된 재개 옵션은 불변6함수 AST 확인 후 증거 상속). 이것은 전체 예측 인과성 검사나 모든 fold/모든 cutoff의 증거가 아니다.

첫fold 독립 비평은 buffer/orderedIDs/훈련 가중치/season/context49 표본순서와 혼합을 재검산했다. 전체 scalar 검산 {v['checks']}항목 통과, 최대 혼합식 차이 {v['max_formula_gap']:.3g}. 각fold 두 baseline 시드의 예측이 exact같음을 코드에서 확인했다. `n4_groups_v2.json`은 MSE변화=2E[기준오차×보완값]+E[보완값²]도 확인했다. 독립 최종 비평 결과는 별도 문서가 우선한다.

원9회 정확 재현 소수6자리 불일치(최대0.004595℃)는 미해결. 새 제출/확정 구성 없음. 원PFN8 전체·실제 stochastic seed별·확인 layout·전체모델 예측 인과 검사 없이는 개선 확정 불가.

## 근거
- n4_registration_v2.json / n4_results_v2.json / n4_verify_all_v1.json / n4_groups_v2.json
- N4_PLAN_REVIEW_v1.md / N4_CODE_REVIEW_v2.md / N4_AUDIT_INHERITANCE_REVIEW_v1.md / N4_MID_REVIEW_v1.md
- n2_target_results_v1.json / n3_results_v1.json / n3_baseline_decomposition_v1.json / N3_INTERPRETATION_REVIEW_v2.md
'''
with (H/'온도_연결실험_종합_v1.md').open('x',encoding='utf-8') as f:f.write(text)
print('N4_REPORT_SAVED',judge)
