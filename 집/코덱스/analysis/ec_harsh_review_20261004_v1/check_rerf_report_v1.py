"""Validate saved aggregate report only. No new predictions, tuning, or fit."""
from pathlib import Path
import csv,json,math,hashlib
from decimal import Decimal,localcontext
ROOT=Path(__file__).resolve().parents[4]
SRC=ROOT/'집/코덱스/analysis/ec_regression_enhanced_20261004_v1'
OUT=Path(__file__).resolve().parent
with (SRC/'scores_v1.csv').open(encoding='utf-8-sig',newline='') as f:scores=list(csv.DictReader(f))
with (SRC/'segments_v1.csv').open(encoding='utf-8-sig',newline='') as f:segments=list(csv.DictReader(f))
cells=[]
for seed in [7,101,2024]:
    s=next(r for r in scores if r['validator']=='DIAG10' and int(r['seed'])==seed)
    h=next(r for r in segments if r['segment']=='high' and int(r['seed'])==seed)
    n=next(r for r in segments if r['segment']=='ordinary' and int(r['seed'])==seed)
    assert int(h['n'])==744 and int(n['n'])==7896 and int(s['n'])==8640
    assert int(h['days'])==31 and int(n['days'])==329
    for col in ['baseline','candidate']:
        combined=math.sqrt(math.fsum([int(h['n'])*float(h[col])**2,int(n['n'])*float(n[col])**2])/int(s['n']))
        assert abs(combined-float(s[col]))<1e-12
        with localcontext() as ctx:
            ctx.prec=40
            independent=((Decimal(h['n'])*Decimal(h[col])**2+Decimal(n['n'])*Decimal(n[col])**2)/Decimal(s['n'])).sqrt()
            assert abs(float(independent)-combined)<1e-12
    change={name:100*(float(row['candidate'])/float(row['baseline'])-1) for name,row in [('overall',s),('high',h),('ordinary',n)]}
    assert abs(change['overall']-float(s['change_pct']))<1e-12
    with localcontext() as ctx:
        ctx.prec=40
        for name,row in [('overall',s),('high',h),('ordinary',n)]:
            assert abs(float(100*(Decimal(row['candidate'])/Decimal(row['baseline'])-1))-change[name])<1e-11
    hsse=int(h['n'])*(float(h['candidate'])**2-float(h['baseline'])**2)
    nsse=int(n['n'])*(float(n['candidate'])**2-float(n['baseline'])**2)
    dm=float(n['candidate'])**2-float(n['baseline'])**2
    biaspart=float(n['candidate_bias'])**2-float(n['baseline_bias'])**2
    varpart=dm-biaspart
    cells.append(dict(seed=seed,change_pct=change,high_delta_sse=hsse,ordinary_delta_sse=nsse,ordinary_delta_mse=dm,ordinary_delta_bias_sq=biaspart,ordinary_delta_residual_variance=varpart,residual_variance_share_of_ordinary_loss_increase=varpart/dm))
result=dict(status='PASS',scope='Fixed saved aggregate score/segment reaggregation only',cells=cells,hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC/'scores_v1.csv',SRC/'segments_v1.csv',SRC/'run.py',SRC/'result_v1.json',SRC/'first_fold_verification_v1.json']})
(OUT/'rerf_report_check_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result['cells'],ensure_ascii=False,indent=2))
