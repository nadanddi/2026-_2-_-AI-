"""Independent csv/math.fsum descriptive audit; no pandas, numpy or fitting."""
from pathlib import Path
import csv, math, json, hashlib
from collections import defaultdict

H = Path(__file__).resolve().parent
R = H.parents[3]
O = H / 'results_v2'
INPUT = R / '연구실/코덱스/local' / H.name / 'inputs/oof.csv'

def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def mean(v):
    return math.fsum(v) / len(v)

def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),(a,b)

records = [r for r in read(INPUT) if r['validator']=='DIAG10']
byrow = defaultdict(list)
byseedday = defaultdict(list)
for r in records:
    byrow[r['row_id']].append(r)
    byseedday[(r['seed'],r['farm'],int(r['day']))].append(r)
assert len(records)==25920 and len(byrow)==8640
assert all(len(q)==3 and {r['seed'] for r in q}=={'7','101','2024'} for q in byrow.values())
assert all(len({r['y'] for r in q})==1 for q in byrow.values())

weights = {'raw_et':.48,'raw_lgb':.24,'raw_mlp':.08,'old_pfn_raw':.20}
mix_byrow = defaultdict(list)
member_byrow = defaultdict(lambda:defaultdict(list))
max_reconstruct = 0.
for key,q in byseedday.items():
    q.sort(key=lambda r:int(r['hour']))
    assert [int(r['hour']) for r in q]==list(range(24))
    raw = [math.fsum(float(r[c])*w for c,w in weights.items()) for r in q]
    sm = [.5*v+.5*mean(raw[:i+1]) for i,v in enumerate(raw)]
    for i,r in enumerate(q):
        pred=max(float(r['clip_lo']),min(float(r['clip_hi']),sm[i]))
        max_reconstruct=max(max_reconstruct,abs(pred-float(r['baseline'])))
        close(pred,float(r['baseline']))
        mix_byrow[r['row_id']].append(sm[i])
    for c in weights:
        v=[float(r[c]) for r in q]
        for i,r in enumerate(q):
            member_byrow[r['row_id']][c].append(.5*v[i]+.5*mean(v[:i+1]))

rawpath=R/'data/train_X.csv'
# Use bootstrap only to locate data if the conventional path does not exist.
if not rawpath.exists():
    import sys
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(R/'집/클로드/research'))
    import env
    rawpath=Path(env.DATA)/'train_X.csv'
vent={}
with rawpath.open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        if r['row_id'] in byrow:
            vent[r['row_id']]=float(r['act_vent']) if r['act_vent'] else float('nan')
assert len(vent)==8640

dayrows=defaultdict(list)
for rid,q in byrow.items():
    r=q[0]
    dayrows[(r['farm'],int(r['day']))].append(dict(id=rid,y=float(r['y']),p=mean([float(t['baseline']) for t in q]),vent=vent[rid]))
computed={}
for key,q in dayrows.items():
    assert len(q)==24
    e=[r['p']-r['y'] for r in q]
    bias=mean(e)
    sse=math.fsum(t*t for t in e)
    shape=math.fsum((t-bias)**2 for t in e)
    close(sse,24*bias*bias+shape)
    computed[key]=dict(n=24,ymean=mean([r['y'] for r in q]),pmean=mean([r['p'] for r in q]),bias=bias,sse=sse,sse_day=24*bias*bias,sse_shape=shape,ventzero=mean([float(r['vent']==0) for r in q]))
days=read(O/'days.csv')
for r in days:
    key=(r['farm'],int(r['day']))
    for c,v in computed[key].items():
        close(v,float(r[c]))
groups=defaultdict(list)
for key,d in computed.items():
    if d['ymean']<1:
        groups['closed' if d['ventzero']>=.8 else 'rest'].append((key,d))
assert len(groups['closed'])==61 and len(groups['rest'])==268
summary={}
official=json.loads((O/'summary.json').read_text(encoding='utf-8'))
for gr,q in groups.items():
    ds=[d for _,d in q]
    sse=math.fsum(d['sse'] for d in ds)
    summary[gr]=dict(days=len(ds),rmse=math.sqrt(sse/(24*len(ds))),bias=mean([d['bias'] for d in ds]),sse=sse,day_share=math.fsum(d['sse_day'] for d in ds)/sse,shape_mse=math.fsum(d['sse_shape'] for d in ds)/(24*len(ds)),level_mse=math.fsum(d['sse_day'] for d in ds)/(24*len(ds)))
    for c,v in summary[gr].items():close(v,official[gr][c])

strata=read(O/'strata.csv')
for r in strata:
    if r['facets']=='farm':
        farm=r['stratum'].strip("() ,'")
        ds=[d for (f,_),d in groups[r['group']] if f==farm]
        assert len(ds)==int(r['days'])
        close(math.sqrt(math.fsum(d['sse'] for d in ds)/(24*len(ds))),float(r['rmse']))
decomp=json.loads((O/'composition_decomposition.json').read_text(encoding='utf-8'))
for r in decomp:
    nc=math.fsum(s['nc'] for s in r['strata']);nr=math.fsum(s['nr'] for s in r['strata'])
    comp=math.fsum((s['nc']/nc-s['nr']/nr)*(s['mc']+s['mr'])/2 for s in r['strata'])
    within=math.fsum((s['nc']/nc+s['nr']/nr)*(s['mc']-s['mr'])/2 for s in r['strata'])
    close(comp,r['composition']);close(within,r['within'])
    close(comp+within,r['closed_mse']-r['rest_mse'])

allocation=[]
officialalloc=read(O/'member_sse_allocation.csv')
for gr,q in groups.items():
    rr=[r for key,_ in q for r in dayrows[key]]
    for c,w in weights.items():
        val=math.fsum(w*(mean(member_byrow[r['id']][c])-r['y'])*(mean(mix_byrow[r['id']])-r['y']) for r in rr)
        name={'raw_et':'et','raw_lgb':'lgb','raw_mlp':'mlp','old_pfn_raw':'pfn'}[c]
        close(val,float(next(a['sse_allocation'] for a in officialalloc if a['group']==gr and a['member']==name)))
        allocation.append(dict(group=gr,member=name,sse_allocation=val))

out=dict(status='PASS',method='csv + math.fsum; independently recomputed from public OOF and matching train_X row IDs',rows=8640,days=360,summary=summary,max_seed_prediction_reconstruction_error=max_reconstruct,checked=['all 360 day metrics','61/268 membership from raw act_vent','seed-wise nonlinear clipping','farm RMSE strata','Kitagawa identities','member covariance SSE allocations'],inputs_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),allocation=allocation)
target=H/'critic_verification_v1.json'
with target.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(out,ensure_ascii=False,indent=2))
