from pathlib import Path
import csv, math, json, hashlib
from collections import defaultdict
from decimal import Decimal, localcontext
H=Path(__file__).resolve().parent; ROOT=H.parents[3]
P=ROOT/'제출/09회차_2026-10-06(팀)/submission_14.csv'
T=ROOT/'공용/대회자료/정형데이터/참가자_배포/test_X.csv'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with P.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
with T.open(encoding='utf-8-sig',newline='') as f:test=list(csv.DictReader(f))
assert len(rows)==len(test)==1440 and len({r['row_id'] for r in rows})==1440
assert {r['row_id'] for r in rows}=={r['row_id'] for r in test}
assert sha(P)=='f2fe5cf7e5299c2c0f2ac39f0e071066744b0c75ea19e1d7dff1e3a6282ba18a'
g=defaultdict(list)
for r in rows:
    key=tuple(r['row_id'].split('_')[:2]);g[key].append(r['sub_ec'])
assert len(g)==60 and {len(v) for v in g.values()}=={24}
means=[math.fsum(map(float,v))/24 for v in g.values()]
assert all(math.isfinite(x) for x in means)
terms=[max(x-1,0)**2 for x in means]
L=math.sqrt(math.fsum(terms)/60)
# Algebraically distinct form: positive part of daily prediction SUM above 24.
Lsum=math.sqrt(math.fsum(max(math.fsum(map(float,v))-24,0)**2 for v in g.values())/(60*24**2))
assert abs(L-Lsum)<1e-15
with localcontext() as ctx:
    ctx.prec=65
    ss=sum((max(sum(map(Decimal,v))-Decimal(24),Decimal(0))**2 for v in g.values()),Decimal(0))
    ld=(ss/Decimal(60*24**2)).sqrt()
assert abs(L-float(ld))<1e-15
producer=json.loads((H/'evaluation_absence_bound_v1.json').read_text(encoding='utf-8'))
# Deliberately do not import any producer module. Verify its reported numeric bound recursively.
nums=[]
def walk(v):
    if isinstance(v,dict):
        for k,x in v.items():
            if isinstance(x,(int,float)) and not isinstance(x,bool):nums.append((k,float(x)))
            walk(x)
    elif isinstance(v,list):
        for x in v:walk(x)
walk(producer)
assert any(abs(x-L)<1e-14 for _,x in nums),nums
out=dict(status='INDEPENDENT_LB_BOUND_PASS',rows=1440,days=60,rows_per_day=24,lower_bound_fsum=L,lower_bound_sum_form=Lsum,lower_bound_decimal=str(ld),official_score=.1384,conservative_display_upper=.1385,bound_exceeds_upper=L>.1385,lower_bound_margin=L-.1385,sources={str(P):sha(P),str(T):sha(T),str(H/'evaluation_absence_bound_v1.json'):sha(H/'evaluation_absence_bound_v1.json'),str(ROOT/'공용/대회자료/정형데이터/정형데이터_문제설명서.pdf'):sha(ROOT/'공용/대회자료/정형데이터/정형데이터_문제설명서.pdf'),str(ROOT/'공용/대회자료/공고/온라인미션_중간안내_20261005.pdf'):sha(ROOT/'공용/대회자료/공고/온라인미션_중간안내_20261005.pdf'),str(__file__):sha(__file__)},individual_label_inversion=False,model_change=False)
with (H/'evaluation_absence_bound_independent_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(out,ensure_ascii=False,indent=2))
