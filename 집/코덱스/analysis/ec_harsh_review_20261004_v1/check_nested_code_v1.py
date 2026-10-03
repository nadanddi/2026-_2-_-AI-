"""Execute isolated reviewed helpers on synthetic data; never import/run model script."""
from pathlib import Path
import ast, json, math, hashlib
from decimal import Decimal, localcontext
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT/'집/코덱스/analysis/ec_nested_high_specialist_20261004_v1'
OUT = Path(__file__).resolve().parent
source = (SRC/'run.py').read_text(encoding='utf-8')
tree = ast.parse(source)
selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ['prefix', 'full_inner', 'specialist']]
ns = {'np':np, 'INNER':Path('synthetic'), 'LAMBDA':.01}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SRC/'run.py'), 'exec'), ns)

# Prefix independently calculated row by row, with two farms and shuffled hours.
frame = pd.DataFrame({'farm':['F47','F13','F13','F47','F13','F47'], 'day':[1]*6, 'hour':[2,1,0,0,2,1]})
p = np.asarray([1.1,.8,1.2,.7,.6,.9])
pp = ns['prefix'](p,frame)
expected=[]
for i,row in frame.iterrows():
    rows=[j for j,r in frame.iterrows() if r.farm==row.farm and r.day==row.day and r.hour<=row.hour]
    expected.append(math.fsum(float(p[j]) for j in rows)/len(rows))
assert np.max(abs(pp-expected)) < 1e-12
with localcontext() as ctx:
    ctx.prec=40
    exact=[]
    for i,row in frame.iterrows():
        rows=[j for j,r in frame.iterrows() if r.farm==row.farm and r.day==row.day and r.hour<=row.hour]
        exact.append(float(sum((Decimal(str(p[j])) for j in rows),Decimal(0))/Decimal(len(rows))))
    assert np.max(abs(pp-exact)) < 1e-12

# Full-query denominator and weight KKT, independently checked by Decimal.
cases=[]
for name,e,d in [('interior',[-.1,0.,0.,.05],[.2,0.,0.,.1]),
                 ('zero',[.1,.1],[.2,.1]),('upper',[-2.,-2.],[.2,.1]),
                 ('all_gate_zero',[9.,-9.],[0.,0.])]:
    n=len(e);num=math.fsum(a*b for a,b in zip(e,d))/n
    den=math.fsum(a*a for a in d)/n+.01
    w=float(np.clip(-num/den,0,.5))
    with localcontext() as ctx:
        ctx.prec=40
        de=[Decimal(str(x)) for x in e];dd=[Decimal(str(x)) for x in d]
        dn=sum((a*b for a,b in zip(de,dd)),Decimal(0))/Decimal(n)
        ds=sum((a*a for a in dd),Decimal(0))/Decimal(n)+Decimal('.01')
        dw=min(Decimal('.5'),max(Decimal(0),-dn/ds))
        assert abs(w-float(dw))<1e-12
    grad=2*(num+den*w)
    assert (w==0 and grad>=-1e-12) or (w==.5 and grad<=1e-12) or abs(grad)<1e-12
    cases.append(dict(name=name,weight=w,gradient=grad))

class NeverFit:
    def et(self,*args,**kwargs):
        raise AssertionError('No model construction or training permitted')

# Actual fallback helper returns None model; first-fold audit would assert against it.
train = pd.DataFrame({'row_id':['F13_001_00','F13_001_01'], 'farm':['F13']*2,'day':[1]*2,'hour':[0,1],'sub_ec':[.9,1.0]})
fallback=np.asarray([.45,.55])
sp,model,hs,info=ns['specialist'](NeverFit(),train,train,[],7,fallback)
assert np.array_equal(sp,fallback) and model is None and info['fallback'] and info['n_days']==1

# Synthetic matching caches pass existing full_inner despite duplicate train and query IDs.
lab=pd.DataFrame({'row_id':['F13_001_00','F13_010_00'],'farm':['F13']*2,'day':[1,10],'hour':[0,0],'sub_ec':[1.,.5]})
idx=lab.set_index('row_id')
payload={'inner_train_id':np.asarray(['F13_001_00']*2), 'row_id':np.asarray(['F13_010_00']*2), 'lo':np.asarray(1.), 'hi':np.asarray(1.)}
for seed in [7,101,2024]: payload[f'r3_{seed}']=np.asarray([.6,.6])
def mock_arrayfile(path):
    if path.name.endswith('_cpu.npz'):return payload
    seed=int(path.stem.rsplit('_',1)[-1]);ix=np.random.default_rng(seed).choice(2,2,replace=False)
    return {'row_id':payload['row_id'], 'context_row_id':payload['inner_train_id'][ix], 'prediction':np.asarray([.7,.7])}
ns['arrayfile']=mock_arrayfile
a,b,z,bag=ns['full_inner']('SYNTH',0,lab,idx)
assert a.row_id.duplicated().any() and b.row_id.duplicated().any()

record=dict(status='PASS',scope='Isolated code helpers on synthetic fixtures only; no real predictions, labels, fit or original-script import',
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            prefix_rows=len(frame),prefix_maxdiff=float(np.max(abs(pp-expected))),weight_cases=cases,
            fallback_returns_none_model=True,existing_full_inner_accepts_duplicate_ids=True,
            duplicated_train_rows=len(a),duplicated_query_rows=len(b))
(OUT/'nested_code_check_v1.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
