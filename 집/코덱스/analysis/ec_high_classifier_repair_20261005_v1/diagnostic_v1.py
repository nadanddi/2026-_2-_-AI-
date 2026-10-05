from pathlib import Path
import json,csv
H=Path(__file__).resolve().parent;OUT=H.parents[3]/'집/코덱스/local'/H.name
items=[]
for s in [7,101,2024]:
    b=json.loads((OUT/f'model_DIAG10_0_{s}.json').read_text(encoding='utf-8'))
    with (OUT/f'meta_DIAG10_0_{s}.csv').open(encoding='utf-8',newline='') as f:meta=list(csv.DictReader(f))
    with (OUT/f'outer_DIAG10_0_{s}.csv').open(encoding='utf-8',newline='') as f:outer=list(csv.DictReader(f))
    high=[r for r in meta if int(r['hour'])==23 and int(r['high'])];bad=next(r for r in outer if r['row_id']=='F47_157_23')
    assert all(float(r['prefix_A'])>=.9 for r in high) and int(bad['high'])==1 and float(bad['p'])<b['cuts']['GUARD']
    items.append(dict(seed=s,meta_high_days=len(high),meta_high_prefix_min=min(float(r['prefix_A']) for r in high),meta_high_score_min=min(float(r['p']) for r in high),guard_cut=b['cuts']['GUARD'],missed_day=bad['row_id'],missed_prefix=float(bad['prefix_A']),missed_score=float(bad['p']),train_high_min=min(z['counts']['train_high'] for z in b['split'])))
payload=dict(status='PASS_RAW_CALIBRATION_SUPPORT_CHECK',items=items,caution='Observed calibration support mismatch, not proof of a unique causal mechanism; no threshold retuning.')
with (H/'diagnostic_v1.json').open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
print(json.dumps(payload,ensure_ascii=False))
