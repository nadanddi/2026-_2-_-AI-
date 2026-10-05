from pathlib import Path
import json,csv
H=Path(__file__).resolve().parent;OUT=H.parents[3]/'집/코덱스/local'/H.name
rc=json.loads((H/'receipt_v2.json').read_text(encoding='utf-8'));records=[]
for rec in rc['files']:
    if rec['context']!='outer' or rec['mode']!='HARD':continue
    trrec=next(r for r in rc['files'] if r['v']==rec['v'] and r['k']==rec['k'] and r['s']==rec['s'] and r['mode']=='HARD' and r['context']=='train')
    with (OUT/trrec['path']).open(newline='',encoding='utf-8') as f:tr=[r for r in csv.DictReader(f) if int(r['hour'])==23]
    with (OUT/rec['path']).open(newline='',encoding='utf-8') as f:q=[r for r in csv.DictReader(f) if int(r['hour'])==23]
    for r in q:
        base=float(r['prefix_A'])>=.9
        if not base or int(r['GUARD']):continue
        phase=int(r['day'])>=179;pool=[z for z in tr if z['farm']==r['farm'] and (int(z['day'])>=179)==phase];high=[z for z in pool if int(z['high'])];scores=[float(z['prefix_proxy']) for z in high]
        records.append(dict(v=rec['v'],k=rec['k'],seed=rec['s'],row_id=r['row_id'],high=int(r['high']),prefix_A=float(r['prefix_A']),prefix_proxy=float(r['prefix_proxy']),p=float(r['p']),train_context_days=len(pool),train_context_high=len(high),train_high_proxy_min=min(scores) if scores else None,train_high_proxy_max=max(scores) if scores else None))
payload=dict(status='PASS_RAW_CONTEXT_SUPPORT',records=records,caution='Post-hoc context and proxy comparison only; no revised threshold or selected subset score.')
with (H/'diagnostic_v1.json').open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
print(json.dumps({'diag_extra_high': [r for r in records if r['v']=='DIAG10' and r['seed']==7 and r['high']],'A_extra_high_context0':sum(r['v']=='A' and r['high'] and r['train_context_high']==0 for r in records),'all_extra_high':sum(r['high'] for r in records)},ensure_ascii=False))
