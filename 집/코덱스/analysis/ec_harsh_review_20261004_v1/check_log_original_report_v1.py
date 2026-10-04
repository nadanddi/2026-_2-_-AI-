"""Independent arithmetic from published aggregate OOF tables only."""
from pathlib import Path
import csv,json,math,hashlib
from decimal import Decimal,localcontext
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
SRC=ROOT/'집/코덱스/analysis/ec_log_partition_original_20261004_v1'
RATIO=ROOT/'집/코덱스/analysis/ec_log_partition_mean_20261004_v1'
def table(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
scores=table(SRC/'scores_v1.csv');segments=table(SRC/'segments_v1.csv');ratio=table(RATIO/'scores_v1.csv')
result=json.loads((SRC/'result_v1.json').read_text(encoding='utf-8'))
fit=json.loads((SRC/'fit_audit_v1.json').read_text(encoding='utf-8'))
ratio_idx={(r['validator'],r['seed']):r for r in ratio}
assert len(scores)==15 and sum(int(r['n']) for r in scores)==result['rows']==83160
assert result['decision']=='REJECT' and result['public_pass'] is False
assert result['family']==18 and abs(result['alpha']-.025/18)<1e-15
assert all(v['p_worse']>result['alpha'] and v['ci'][1]>0 for v in result['bootstrap'].values())
comparison=[]
for row in scores:
    percent=100*(float(row['candidate'])/float(row['baseline'])-1)
    assert abs(percent-float(row['change_pct']))<1e-11
    prev=ratio_idx[row['validator'],row['seed']]
    assert row['n']==prev['n'] and row['baseline']==prev['baseline']
    direct=100*(float(row['candidate'])/float(prev['candidate'])-1)
    with localcontext() as ctx:
        ctx.prec=40
        assert abs(float(100*(Decimal(row['candidate'])/Decimal(row['baseline'])-1))-percent)<1e-11
        assert abs(float(100*(Decimal(row['candidate'])/Decimal(prev['candidate'])-1))-direct)<1e-11
    comparison.append(dict(validator=row['validator'],seed=int(row['seed']),original_vs_ratio_pct=direct))

cells=[]
for seed in [7,101,2024]:
    o=next(r for r in scores if r['validator']=='DIAG10' and int(r['seed'])==seed)
    h=next(r for r in segments if r['segment']=='high' and int(r['seed'])==seed)
    n=next(r for r in segments if r['segment']=='ordinary' and int(r['seed'])==seed)
    assert (int(h['n']),int(n['n']),int(o['n']))==(744,7896,8640)
    assert (int(h['days']),int(n['days']))==(31,329)
    for col in ['baseline','candidate']:
        reconstructed=math.sqrt(math.fsum(int(r['n'])*float(r[col])**2 for r in [h,n])/int(o['n']))
        assert abs(reconstructed-float(o[col]))<1e-12
        with localcontext() as ctx:
            ctx.prec=40
            independent=((Decimal(h['n'])*Decimal(h[col])**2+Decimal(n['n'])*Decimal(n[col])**2)/Decimal(o['n'])).sqrt()
            assert abs(float(independent)-reconstructed)<1e-12
    parts=[]
    for name,row in [('high',h),('ordinary',n)]:
        rmse0,rmse1=float(row['baseline']),float(row['candidate'])
        bias0,bias1=float(row['baseline_bias']),float(row['candidate_bias'])
        dmse=rmse1**2-rmse0**2;dbias=bias1**2-bias0**2
        dvar=(rmse1**2-bias1**2)-(rmse0**2-bias0**2)
        assert abs(dmse-dbias-dvar)<1e-15
        with localcontext() as ctx:
            ctx.prec=40
            rr0,rr1,bb0,bb1=map(Decimal,[row['baseline'],row['candidate'],row['baseline_bias'],row['candidate_bias']])
            assert abs(float((rr1**2-rr0**2)-(bb1**2-bb0**2))-dvar)<1e-15
        parts.append(dict(segment=name,change_pct=100*(rmse1/rmse0-1),delta_sse=int(row['n'])*dmse,
                          bias0=bias0,bias1=bias1,mean_prediction_change=bias1-bias0,
                          delta_mse=dmse,delta_bias_sq=dbias,delta_error_variance=dvar))
    total=math.fsum(p['delta_sse'] for p in parts)
    assert total<0 and all(p['delta_sse']<0 for p in parts)
    cells.append(dict(seed=seed,parts=parts,ordinary_share_of_sse_gain=parts[1]['delta_sse']/total))

assert fit['status']=='PASS' and len(fit['cells'])==66
assert len({(r['validator'],r['fold'],r['seed']) for r in fit['cells']})==66
assert max(r['prior_partition_maxdiff'] for r in fit['cells'])<1e-9
record=dict(status='PASS',scope='Aggregate table arithmetic only; no raw model, OOF rescoring, bootstrap rerun or fit',
            cells=cells,improved_score_cells=sum(float(r['change_pct'])<0 for r in scores),comparison=comparison,
            stored_p_worse={k:v['p_worse'] for k,v in result['bootstrap'].items()},
            stored_prior_ratio_reproduction_maxdiff=max(r['prior_partition_maxdiff'] for r in fit['cells']),
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC/'scores_v1.csv',SRC/'segments_v1.csv',SRC/'result_v1.json',SRC/'fit_audit_v1.json',SRC/'run.py']})
(OUT/'log_original_report_check_v1.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='hashes'},ensure_ascii=False,indent=2))
