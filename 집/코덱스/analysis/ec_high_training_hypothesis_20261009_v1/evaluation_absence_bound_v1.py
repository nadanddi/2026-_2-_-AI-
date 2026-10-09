"""Fixed aggregate Jensen lower bound; no hidden targets or model changes."""
import csv, json, math, hashlib, sys
from pathlib import Path
from collections import defaultdict
from decimal import Decimal, localcontext
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
H=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
assert (H/'PLAN_lb_v1.md').exists() and (H/'critique_lb_plan_v1.md').exists()
P=ROOT/'제출/09회차_2026-10-06(팀)/submission_14.csv'
expected='f2fe5cf7e5299c2c0f2ac39f0e071066744b0c75ea19e1d7dff1e3a6282ba18a'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P)==expected
rows=list(csv.DictReader(P.open(encoding='utf-8-sig',newline='')))
test=list(csv.DictReader((Path(env.DATA)/'test_X.csv').open(encoding='utf-8-sig',newline='')))
assert len(rows)==len(test)==1440 and [r['row_id'] for r in rows]==[r['row_id'] for r in test]
assert len(set(r['row_id'] for r in rows))==1440
D=defaultdict(list)
for r in rows:
    assert math.isfinite(float(r['sub_ec']))
    D[r['row_id'].rsplit('_',1)[0]].append(r['sub_ec'])
assert len(D)==60 and all(len(v)==24 for v in D.values())
cutoff=1.0;score_upper=0.1385
means=[math.fsum(float(x) for x in v)/24 for v in D.values()]
bound=math.sqrt(math.fsum(max(m-cutoff,0)**2 for m in means)/60)
with localcontext() as ctx:
    ctx.prec=60
    dd=[sum((Decimal(x) for x in v),Decimal(0))/Decimal(24) for v in D.values()]
    exact=(sum((max(m-Decimal(1),Decimal(0))**2 for m in dd),Decimal(0))/Decimal(60)).sqrt()
assert abs(bound-float(exact))<1e-12
out=dict(submission_sha=sha(P),script_sha=sha(Path(__file__)),plan_sha=sha(H/'PLAN_lb_v1.md'),rows=1440,days=60,day_rows=24,high_day_mean_cutoff=cutoff,reported_rmse=0.1384,score_upper=score_upper,lower_bound_float=bound,lower_bound_decimal=str(exact),exceeds_upper=bound>score_upper,scope='conditional_on_all_1440_rows_equal_weight_RMSE',adoption=False,hidden_targets_used=False)
with (H/'evaluation_absence_bound_v1.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(out,ensure_ascii=False))
