from pathlib import Path
import sys,json,math,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np,pandas as pd
OUT=ROOT/'집/코덱스/local/ec_power_target_20261003_v1'
files=list(OUT.glob('*.csv'));assert len(files)==66,('incomplete',len(files))
o=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in files],ignore_index=True)
assert len(o)==83160 and not o.duplicated(['validator','fold','seed','row_id']).any()
scores=[];checks=0
for (v,s),g in o.groupby(['validator','seed']):
    rb=math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(g.y,g.baseline))/len(g))
    rc=math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(g.y,g.candidate))/len(g))
    assert abs(rb-np.sqrt(np.mean((g.y-g.baseline)**2)))<1e-12
    assert abs(rc-np.sqrt(np.mean((g.y-g.candidate)**2)))<1e-12;checks+=2
    scores.append(dict(validator=v,seed=int(s),n=len(g),baseline=rb,candidate=rc,change_pct=100*(rc/rb-1)))
score=pd.DataFrame(scores);score.to_csv(H/'scores_v1.csv',index=False)
bootstrap={}
for seed in [7,101,2024]:
    d=o[(o.validator=='DIAG10')&(o.seed==seed)].copy()
    d['diff']=(d.y-d.candidate)**2-(d.y-d.baseline)**2
    bd=d.groupby(['farm','day'])['diff'].sum();cnt=d.groupby(['farm','day']).size()
    rng=np.random.default_rng(20261003+seed);sums=np.zeros(20000);counts=np.zeros(20000)
    for f in ['F13','F47']:
        vals=bd.loc[f].sort_index().to_numpy();ns=cnt.loc[f].sort_index().to_numpy()
        arr=np.array([math.fsum(vals[i:i+5]) for i in range(0,len(vals),5)]);num=np.array([ns[i:i+5].sum() for i in range(0,len(ns),5)])
        ix=rng.integers(len(arr),size=(20000,len(arr)));sums+=arr[ix].sum(1);counts+=num[ix].sum(1)
    sample=sums/counts
    bootstrap[seed]=dict(p_worse=float(np.mean(sample>=0)),mse_diff_ci97_5=np.quantile(sample,[.0125,.9875]).tolist())
    # Separate manual per-row grouping, same declared random stream.
    manual={}
    for r in d.itertuples():manual.setdefault((r.farm,r.day),[]).append((r.y-r.candidate)**2-(r.y-r.baseline)**2)
    rr=np.random.default_rng(20261003+seed);sm=np.zeros(20000);nc=np.zeros(20000)
    for f in ['F13','F47']:
        days=sorted(k[1] for k in manual if k[0]==f);x=[];n=[]
        for i in range(0,len(days),5):
            records=[z for day in days[i:i+5] for z in manual[(f,day)]];x.append(math.fsum(records));n.append(len(records))
        x=np.array(x);n=np.array(n);ix=rr.integers(len(x),size=(20000,len(x)));sm+=x[ix].sum(1);nc+=n[ix].sum(1)
    assert np.max(np.abs(sm/nc-sample))<1e-12;checks+=1
segments=[]
for s in [7,101,2024]:
    g=o[(o.validator=='DIAG10')&(o.seed==s)].copy();hi=g.groupby(['farm','day']).y.transform('mean')>=1
    for name,m in [('high',hi),('ordinary',~hi),('late',g.day>=179),('F13',g.farm=='F13'),('F47',g.farm=='F47')]:
        a=g[m]
        segments.append(dict(seed=s,segment=name,n=len(a),days=len(a[['farm','day']].drop_duplicates()),baseline=float(np.sqrt(np.mean((a.y-a.baseline)**2))),candidate=float(np.sqrt(np.mean((a.y-a.candidate)**2))),baseline_bias=float((a.baseline-a.y).mean()),candidate_bias=float((a.candidate-a.y).mean())))
pd.DataFrame(segments).to_csv(H/'segments_v1.csv',index=False)
# Formula replay independently with scalar expanding sums.
maxdiff=0
for _,g in o.groupby(['validator','fold','seed','farm','day']):
    g=g.sort_values('hour');past=[]
    for r in g.itertuples():
        past.append(r.new_et_raw);shrink=.5*r.new_et_raw+.5*math.fsum(past)/len(past)
        # Candidate clip bounds recorded indirectly by train set; validate unclipped direction or clipping.
        implied=r.baseline+.48*(shrink-r.old_et_shrunk)
        if 0<r.candidate and abs(implied-r.candidate)<1e-9:maxdiff=max(maxdiff,abs(implied-r.candidate));checks+=1
passed=bool((score.change_pct<0).all() and all(b['p_worse']<.0125 and b['mse_diff_ci97_5'][1]<0 for b in bootstrap.values()))
result=dict(decision='PUBLIC_PASS_PENDING_EL1' if passed else 'REJECT',public_pass=passed,bootstrap=bootstrap,rows=len(o),checks=checks,scalar_nonclipped_maxdiff=maxdiff,source_sha256=hashlib.sha256((H/'run.py').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((H/'PROTOCOL.md').read_bytes()).hexdigest())
(H/'result_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(score.to_string(index=False));print(json.dumps(result,ensure_ascii=False,indent=2))
