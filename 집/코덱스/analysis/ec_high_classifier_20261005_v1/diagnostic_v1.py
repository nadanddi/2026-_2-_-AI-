import json,csv
from pathlib import Path
H=Path(__file__).resolve().parent;OUT=H.parents[3]/'집/코덱스/local'/H.name
result=[]
for mode in ['PAST','ALL']:
    for s in [7,101,2024]:
        for k in range(10):
            b=json.loads((OUT/f'model_{mode}_{k}_{s}.json').read_text())
            with (OUT/f'meta_{mode}_{k}_{s}.csv').open(newline='',encoding='utf-8') as f:q=[r for r in csv.DictReader(f) if int(r['hour'])==23]
            with (OUT/f'outer_{mode}_{k}_{s}.csv').open(newline='',encoding='utf-8') as f:outer=[r for r in csv.DictReader(f) if int(r['hour'])==23]
            zero_high=sum(int(r['high'])==1 and float(r['p'])==0 for r in q)
            result.append(dict(mode=mode,seed=s,k=k,threshold=b['threshold'],meta_constants=[m['constant'] for m in b['meta']],meta_high=sum(int(r['high']) for r in q),meta_high_zero_score=zero_high,outer_n=len(outer),outer_fp=sum(int(r['high'])==0 and float(r['p'])>=b['threshold'] for r in outer)))
payload=dict(status='PASS_SAVED_MODELS_AND_CSV_DIAGNOSIS',records=result,zero_cut_cells=sum(r['threshold']==0 for r in result),zero_cut_outer_fp_one_seed={m:sum(r['outer_fp'] for r in result if r['mode']==m and r['seed']==7 and r['threshold']==0) for m in ['PAST','ALL']},caution='Diagnostic only; no revised threshold evaluated or selected.')
with (H/'diagnostic_v1.json').open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in payload.items() if k!='records'}))
