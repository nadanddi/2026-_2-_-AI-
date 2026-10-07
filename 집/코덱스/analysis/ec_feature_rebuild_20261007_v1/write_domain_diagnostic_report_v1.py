"""Produce an immutable diagnostic report and update a NEW candidate registry version."""
from pathlib import Path
import csv,json,hashlib,math
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
result=json.loads((HERE/'DOMAIN24_BLK_diagnostic_results_v1.json').read_text(encoding='utf-8'))
check=json.loads((HERE/'DOMAIN24_score_independent_crosscheck_v2.json').read_text(encoding='utf-8'))
assert check['status']=='PASS' and check['result_sha256']==sha(HERE/'DOMAIN24_BLK_diagnostic_results_v1.json')
assert result['variant_count']==48 and len(result['results'])==48
assert result['cumulative_current_BLK_variant_count']==54
assert not any(r['BLK_screen_pass'] for r in result['results'])
rows=sorted(result['results'],key=lambda r:r['mean_seed_delta_RMSE'])
lines=['# 도메인24 BLK 진단 — 전체 목표 중간 결과',
'', '24묶음 × 2적용범위 × 3시드의 48비교 모두 사전 진단 기준을 통과하지 못했다. 새 후보 채택은 없다.',
'', '표본은 1,440행·60기록일·8블록이다. 고EC는 3기록일·72행뿐이다. 같은 노출 표본의 탐색 결과이며 새 확인 검증이 아니다.',
'', '기준은 세 시드 모두 RMSE 감소 및 블록 부트스트랩 P(worse) < 0.025/54이다. 기존 등록의 누적54 보정을 그대로 적용했다. 이후 CH2의 누적60 기준으로 이 진단을 소급 수정하지 않았다.',
'', '학습 전용 캐시 CPU 기준선과 비교했다. 과거 GPU 제출본과 출력 동일성을 주장하지 않는다. 두 범위는 가짜 평가 역할에 SG2를 적용한 QUERY_ROLE 및 원래 일차 gate를 유지한 RAW_PASS이다.',
'', '| 순위 | 범위 | 후보 | 묶음 | 시드 평균 RMSE차 | 전 시드 감소 | P(worse) | 진단 |',
'|---:|---|---|---|---:|---|---:|---|']
for i,r in enumerate(rows,1):
    whole=[c for c in r['cells'] if c['segment']=='전체']
    assert len(whole)==3
    delta=math.fsum(c['candidate_RMSE']-c['baseline_RMSE'] for c in whole)/3
    assert abs(delta-r['mean_seed_delta_RMSE'])<1e-12
    lines.append(f"| {i} | {r['scope']} | {r['candidate']} | {r['family']} | {delta:+.9f} | {r['all_three_seeds_improve']} | {r['p_worse']:.9f} | FAIL |")
lines+=['', '## 일반·고EC 및 블록 위치', '',
'각 숫자는 세 시드의 RMSE차 평균이다. 고EC 구분은 진단용 정답 기준이며 적용 정책을 정하는 데 쓰지 않았다.', '',
'| 범위 | 후보 | 일반 | 고EC | 앞 | 가운데 | 뒤 |',
'|---|---|---:|---:|---:|---:|---:|']
segments=sorted({c['segment'] for r in rows for c in r['cells']})
# Report exact available segment labels; do not invent missing categories.
lines+=['', '실제 저장 세그먼트: '+', '.join(segments), '']
for r in rows:
    grouped={s:[c['delta_RMSE'] for c in r['cells'] if c['segment']==s and c['rows']] for s in segments}
    def mean(s):
        vals=grouped.get(s,[])
        return f'{math.fsum(vals)/len(vals):+.9f}' if vals else '미제공'
    lines.append('| '+r['scope']+' | '+r['candidate']+' | '+' | '.join(mean(s) for s in ['일반','고EC','앞','가운데','뒤'])+' |')
lines+=['', '## 검증과 한계', '',
'- 전체 처리 검증: 독립 산술 518,400개, 최대 차이 4.440892098500626e-16. 유한 검사이며 모든 가능한 입력에 대한 증명은 아니다.',
'- 별도 Decimal60 코드가 48안·864셀·1,728 RMSE와 블록 손실·부트스트랩 P/CI를 검산했다. 모델 자체를 독립 재학습한 검사는 아니다.',
'- 반론: 평균 감소가 있어도 사용할 수 있지 않은가? 답: 사전 고정한 반복성 기준을 통과하지 않았으므로 채택하지 않는다. 신뢰도 높음은 이번 진단 FAIL 판정에만 해당한다.',
'- 반론: 도메인 특징 자체가 무효인가? 답: 이 작은 BLK 표본만으로 일반화할 수 없다. 전체 24묶음을 TM111/P2LOO/EL1와 세 시드에서 계속 시험한다.',
'- 원66폴드 준비, 공개 이웃/CH2 원검증기, 문헌24, 데이터142·원열 부분집합/상호작용, 미사용 시드·배치 최초1회 및 최종 전체 보고서는 남아 있다.',
'', '근거: DOMAIN24_verified_gate_v1.json, DOMAIN24_execution_score_v1.json, DOMAIN24_BLK_diagnostic_results_v1.json, DOMAIN24_score_independent_crosscheck_v2.json.']
with (HERE/'DOMAIN24_BLK_진단보고서_v1.md').open('x',encoding='utf-8') as handle:handle.write('\n'.join(lines)+'\n')
with (HERE/'feature_candidates_v9.csv').open(encoding='utf-8-sig',newline='') as handle:
    reader=csv.DictReader(handle);fields=reader.fieldnames;registry=list(reader)
assert len(registry)==202
for record in registry:
    if record['candidate_id'] in {r['candidate'] for r in rows}:
        entries=[r for r in rows if r['candidate']==record['candidate_id']]
        assert len(entries)==2
        record['status']='BLK_DIAGNOSTIC_FAIL_ORIGINAL_ALL24_REQUIRED'
        record['performance']=json.dumps({r['scope']:{'delta_RMSE':r['mean_seed_delta_RMSE'],'p_worse':r['p_worse'],'pass':False} for r in entries},ensure_ascii=False)
with (HERE/'feature_candidates_v10.csv').open('x',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(registry)
print('Immutable domain BLK diagnostic report + candidate registry10 written; no adoption or submission')
