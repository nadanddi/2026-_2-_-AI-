"""Fresh stdlib-only check of three recommendation-driving descriptive numbers."""
from pathlib import Path
import csv,json,math,hashlib,collections
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
OOF=ROOT/'집/코덱스/local/ec_lgb_operation_20261004_v1/oof.csv'
whole=json.loads((H.parent/'ec_lgb_operation_20261004_v1/full_verification_v3.json').read_text(encoding='utf-8'))
assert hashlib.sha256(OOF.read_bytes()).hexdigest()==whole['aggregate_sha256']
by_id=collections.defaultdict(list)
with OOF.open(encoding='utf-8',newline='') as f:
    for r in csv.DictReader(f):
        if r['validator']=='DIAG10':by_id[r['row_id']].append(r)
assert len(by_id)==8640 and all(len(rs)==3 for rs in by_id.values())
days=collections.defaultdict(list)
for rid,rs in by_id.items():
    assert {int(r['seed']) for r in rs}=={7,101,2024}
    ys={float(r['y']) for r in rs};assert len(ys)==1
    y=ys.pop();p=math.fsum(float(r['baseline']) for r in rs)/3
    days[(rs[0]['farm'],int(rs[0]['day']))].append((y,p))
ordinary=[];high=[]
for key,rows in days.items():
    assert len(rows)==24
    errors=[p-y for y,p in rows];mean=math.fsum(errors)/24
    record=dict(key=key,sse=math.fsum(e*e for e in errors),level=24*mean*mean)
    (high if math.fsum(y for y,p in rows)/24>=1 else ordinary).append(record)
assert len(high)==31 and len(ordinary)==329
total=math.fsum(r['sse'] for r in ordinary)
values=dict(ordinary_day_share_pct=100*math.fsum(r['level'] for r in ordinary)/total,
 ordinary_top10_sse_share_pct=100*math.fsum(r['sse'] for r in sorted(ordinary,key=lambda r:-r['sse'])[:10])/total,
 high_day_share_pct=100*math.fsum(r['level'] for r in high)/math.fsum(r['sse'] for r in high))
saved=json.loads((H/'results_v3.json').read_text(encoding='utf-8'))['summaries']
expected=dict(ordinary_day_share_pct=saved['baseline/mean/ordinary/all']['day_share_pct'],
 ordinary_top10_sse_share_pct=saved['baseline/mean/ordinary/all']['top10_sse_share_pct'],
 high_day_share_pct=saved['baseline/mean/high/all']['day_share_pct'])
assert all(abs(values[k]-expected[k])<1e-10 for k in values)
with (H/'key_findings_crosscheck_v1.json').open('x',encoding='utf-8') as f:
    json.dump(dict(status='PASS',values=values,ordinary_days=329,high_days=31,rows=8640,
      actual_fit=0,test_reads=0,raw_ec_reads=0,limitation='Repeated public validation; retrospective decomposition, not attainable improvement.'),f,indent=2)
print(json.dumps(values))
