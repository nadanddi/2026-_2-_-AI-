"""Fixed completed public OOF audit only. Never fit, optimize weights or read holdout."""
from pathlib import Path
import csv,json,math,hashlib
from collections import defaultdict
from decimal import Decimal,localcontext
ROOT=Path(__file__).resolve().parents[4]
H=ROOT/'집/코덱스/analysis/ec_log_partition_original_20261004_v1'
SRC=ROOT/'집/코덱스/local/ec_log_partition_original_20261004_v1'
OUT=Path(__file__).resolve().parent
def table(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
result=json.loads((H/'result_v1.json').read_text(encoding='utf-8'))
scores={(r['validator'],int(r['seed'])):r for r in table(H/'scores_v1.csv')}
segments={(int(r['seed']),r['segment']):r for r in table(H/'segments_v1.csv')}
groups=defaultdict(list);hashes={};seen=set()
for cell in result['split_audit']:
    v,k=cell['validator'],cell['fold']
    assert v in ['DIAG10','A','B','EXT10','EXT12']
    for seed in [7,101,2024]:
        path=SRC/f'{v}_{k}_{seed}.csv';rows=table(path);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(rows)>0
        for r in rows:
            assert r['validator']==v and int(r['fold'])==k and int(r['seed'])==seed
            key=(v,k,seed,r['row_id']);assert key not in seen;seen.add(key)
            r['yy']=float(r['y']);r['bb']=float(r['baseline']);r['cc']=float(r['candidate'])
            assert all(math.isfinite(r[c]) for c in ['yy','bb','cc'])
        groups[v,seed].extend(rows)
assert len(hashes)==66 and len(seen)==83160

def metrics(rows):
    n=len(rows);out={}
    for alias,col in [('baseline','bb'),('candidate','cc')]:
        e=[r[col]-r['yy'] for r in rows]
        out[alias]=math.sqrt(math.fsum(x*x for x in e)/n)
        out[alias+'_bias']=math.fsum(e)/n
    return out

checks=[]
for (v,seed),rows in groups.items():
    row=scores[v,seed];actual=metrics(rows)
    assert len(rows)==int(row['n'])
    for col in ['baseline','candidate']:assert abs(actual[col]-float(row[col]))<1e-12
    e=[r['bb']-r['yy'] for r in rows];d=[r['cc']-r['bb'] for r in rows]
    derivative=2*math.fsum(a*b for a,b in zip(e,d))/len(rows)
    delta_sq=math.fsum(x*x for x in d)/len(rows)
    delta_mse=actual['candidate']**2-actual['baseline']**2
    assert abs(derivative+delta_sq-delta_mse)<1e-12
    with localcontext() as ctx:
        ctx.prec=40
        # Independent decimal sufficient-statistic computation on exact binary floats.
        num=sum((Decimal(a)*Decimal(b) for a,b in zip(e,d)),Decimal(0))
        assert abs(float(2*num/Decimal(len(rows)))-derivative)<1e-12
    checks.append(dict(validator=v,seed=seed,n=len(rows),rmse=actual,posthoc_derivative_at_zero=derivative,
                       mean_delta_sq=delta_sq,endpoint_delta_mse=delta_mse))

segment_checks=[]
for seed in [7,101,2024]:
    rows=groups['DIAG10',seed];assert len(rows)==8640 and len({r['row_id'] for r in rows})==8640
    days=defaultdict(list)
    for r in rows:days[r['farm'],int(r['day'])].append(r['yy'])
    high={k for k,y in days.items() if math.fsum(y)/len(y)>=1}
    assert len(days)==360 and len(high)==31
    sels={'high':[r for r in rows if (r['farm'],int(r['day'])) in high],
          'ordinary':[r for r in rows if (r['farm'],int(r['day'])) not in high],
          'late':[r for r in rows if int(r['day'])>=179],
          'F13':[r for r in rows if r['farm']=='F13'],
          'F47':[r for r in rows if r['farm']=='F47']}
    for name,rr in sels.items():
        expected=segments[seed,name];actual=metrics(rr)
        assert len(rr)==int(expected['n'])
        assert len({(r['farm'],r['day']) for r in rr})==int(expected['days'])
        for col,val in actual.items():assert abs(val-float(expected[col]))<1e-12
        segment_checks.append(dict(seed=seed,segment=name,n=len(rr),metrics=actual))

record=dict(status='PASS',scope='Re-sum fixed completed public OOF and posthoc derivative only; no weight selection, model fit or independent score claim',
            public_occurrence_rows=len(seen),files=len(hashes),score_cells=checks,segment_cells=segment_checks,
            negative_derivative_cells=sum(c['posthoc_derivative_at_zero']<0 for c in checks),hashes=hashes)
(OUT/'log_original_oof_verification_v1.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k in ['status','scope','public_occurrence_rows','files','negative_derivative_cells']},ensure_ascii=False,indent=2))
print(json.dumps([{k:c[k] for k in ['validator','seed','posthoc_derivative_at_zero','endpoint_delta_mse']} for c in checks],ensure_ascii=False,indent=2))
