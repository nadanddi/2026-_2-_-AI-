from pathlib import Path
import sys, json, math, hashlib
sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'))
import support as S
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
OUT=ROOT/'집/코덱스/local/source_history_20261003_v1'
W=['out_temp','out_hum','out_rad','out_wspd']
CHECKS=[]
def ck(n,b):
    assert b,n
    CHECKS.append(n)
def save(p,x):
    assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def roles(raw,keys):
    vec={k:g.sort_values('hour')[W].to_numpy() for k,g in raw.groupby(['farm','day']) if k in keys}
    proposals={}; pairs=[]
    for (f,d),a in vec.items():
        k=(f,d+1)
        if k in vec and a.shape==(24,4) and np.isfinite(a).all() and np.array_equal(a,vec[k]):
            pairs.append(((f,d),k))
            proposals.setdefault((f,d),set()).add(0); proposals.setdefault(k,set()).add(1)
    conflict={k for k,v in proposals.items() if len(v)>1}
    good=[(a,b) for a,b in pairs if a not in conflict and b not in conflict]
    return {k:r for a,b in good for k,r in [(a,0),(b,1)]},good
def prefix(raw,cols):
    a=raw.sort_values(['farm','day','hour']).copy()
    a[cols]=a.groupby(['farm','day'])[cols].transform(lambda x:x.expanding().mean())
    return a.set_index(['farm','day','hour'])[cols]
def probs(raw,cols,keys,role,table):
    p={}; fitrows=[]
    for f in ['F13','F47']:
        train=[k for k in role if k[0]==f]
        query=sorted(k for k in keys if k[0]==f)
        y=np.array([role[k] for k in train])
        for h in range(24):
            xi=[(*k,h) for k in train]; qi=[(*k,h) for k in query]
            if len(set(y))<2:
                pred=np.full(len(qi),.5)
            else:
                model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=1,max_iter=2000))
                model.fit(table.reindex(xi).to_numpy(),y)
                pred=model.predict_proba(table.reindex(qi).to_numpy())[:,1]
            for k,v in zip(qi,pred):p[k]=float(v)
        fitrows.extend(train)
    return p,fitrows
def correction(frame,ref,p,daily):
    delta=np.zeros(len(frame)); n=np.zeros(len(frame),int); latest=np.full(len(frame),-1,int)
    for (f,d),ix in frame.groupby(['farm','day']).indices.items():
        history=[(int(q),float(y)) for (ff,q),y in daily.items() if ff==f and 0<d-q<=12 and (d>=179)==(q>=179)]
        if not history:continue
        qs=np.array([q for q,y in history]); ys=np.array([y for q,y in history])
        pp=np.array([p[(f,int(q),23)] for q in qs]); decay=np.exp(-(d-qs)/6)
        for i in ix:
            ph=p[(f,int(d),int(frame.iloc[i].hour))]
            weights=decay*(ph*pp+(1-ph)*(1-pp))
            if weights.sum()>0:
                level=np.dot(weights,ys)/weights.sum()
                delta[i]=.1*np.exp(-(d-qs.max())/6)*level;n[i]=len(qs);latest[i]=qs.max()
    return np.maximum(ref+np.clip(delta,-.5,.5),0),n,latest
def rm(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def boot(g,seed):
    d=g.assign(diff=(g.candidate-g.y)**2-(g.baseline-g.y)**2)
    b=d.groupby(['farm','day'])['diff'].sum()
    blocks=[]
    for f in ['F13','F47']:
        vals=b.loc[f].sort_index().to_numpy();blocks.append([vals[i:i+5].sum() for i in range(0,len(vals),5)])
    rng=np.random.default_rng(seed); totals=np.zeros(20000)
    for a in blocks:
        a=np.array(a);totals+=a[rng.integers(len(a),size=(20000,len(a)))].sum(1)
    return float(np.mean(totals>=0)),np.quantile(totals,[.025,.975]).tolist()
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    lab,core,_,folds,outer=S.loadec()
    raw=core.identify(pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id']+W+core.RAW))
    raw=raw[raw.farm.isin(['F13','F47'])].sort_values(['farm','day','hour']).reset_index(drop=True)
    table=prefix(raw,core.RAW);keys=set(zip(lab.farm,lab.day))
    allrole,pairs=roles(raw,keys)
    structure=[]
    for f in ['F13','F47']:
        starts=sorted(a[1] for a,b in pairs if a[0]==f)
        structure.append(dict(farm=f,public_complete_pairs=len(starts),start_parity_switches=sum(a%2!=b%2 for a,b in zip(starts,starts[1:]))))
    rows=[];aud=[]
    with threadpool_limits(limits=2):
        for name,k,tm,vm in folds:
            tr=lab[tm].copy();va=lab[vm].sort_values(['farm','day','hour']).reset_index(drop=True)
            tk=set(zip(tr.farm,tr.day));vk=set(zip(va.farm,va.day));r,pair=roles(raw,tk)
            ck('role_subset_train',set(r)<=tk and not(set(r)&vk))
            ck('buffer',all(not(f==ff and abs(d-q)<=1) for f,d in tk for ff,q in vk))
            p,fit=probs(raw,core.RAW,tk|vk,r,table)
            ck('fit_subset_train',set(fit)<=tk)
            z=dict(np.load(S.OUT/f'E_{name}_{k}_cpu.npz'))
            ids=set(z['row_id']);trainids=set(z['inner_train_id'])
            ck('nested_no_outer_validation',set(z['outer_train_id']).isdisjoint(va.row_id) and ids.isdisjoint(va.row_id) and trainids.isdisjoint(va.row_id))
            ck('nested_independent',ids.isdisjoint(trainids) and ids<=set(tr.row_id))
            b=lab.set_index('row_id').reindex(z['row_id']).reset_index()
            bag=np.mean([np.load(S.OUT/f'E_{name}_{k}_pfn_{s}.npz')['prediction'] for s in range(1,5)],axis=0)
            audit=dict(validator=name,fold=k,train_days=len(tk),validation_days=len(vk),pair_train_days=len(r))
            # Query futures/external channels cannot enter the prefix at sampled h=6.
            if name=='DIAG10' and k==0:
                f,d=next(iter(vk));mut=raw.copy();sel=(mut.farm==f)&(mut.day==d)
                mut.loc[sel&(mut.hour>6),core.RAW]=99999
                mut.loc[mut.farm!=f,core.RAW]=-99999
                mut[W]=88888
                changed=prefix(mut,core.RAW)
                ck('future_weather_otherfarm_prefix_invariance',np.array_equal(table.loc[(f,d,6)].to_numpy(),changed.loc[(f,d,6)].to_numpy(),equal_nan=True))
            for seed in [7,101,2024]:
                base=outer[(outer.validator==name)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy()
                ck('baseline_finite',np.isfinite(base).all())
                nested=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),z['lo'],z['hi'])
                yd=b.assign(residual=b.sub_ec.to_numpy()-nested).groupby(['farm','day']).residual.mean()
                pred,n,last=correction(va,base,p,yd)
                ck('history_past',all(q<d or q==-1 for q,d in zip(last,va.day)))
                ck('history_train',all(q==-1 or (f,int(q)) in tk for f,q in zip(va.farm,last)))
                a=va[['row_id','farm','day','hour']].copy();a['y']=va.sub_ec;a['baseline']=base;a['candidate']=pred
                a['validator']=name;a['fold']=k;a['seed']=seed;a['pB']=[p[(f,int(d),int(h))] for f,d,h in zip(va.farm,va.day,va.hour)];a['history_n']=n;a['last_day']=last
                rows.append(a)
            aud.append(audit);print(name,k,'pair train days',len(r),flush=True)
    o=pd.concat(rows,ignore_index=True);o.to_csv(OUT/'oof.csv',index=False)
    score=[]
    for (v,s),g in o.groupby(['validator','seed']):
        rb=rm(g.y.to_numpy(),g.baseline.to_numpy());rc=rm(g.y.to_numpy(),g.candidate.to_numpy())
        score.append(dict(validator=v,seed=int(s),n=len(g),baseline=rb,candidate=rc,change_pct=100*(rc/rb-1)))
        for c,x in [('baseline',rb),('candidate',rc)]:
            independent=math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(g.y,g[c]))/len(g))
            ck('fsum_rmse',abs(independent-x)<1e-12)
    sc=pd.DataFrame(score);sc.to_csv(HERE/'scores_v1.csv',index=False)
    bs={s:boot(o[(o.validator=='DIAG10')&(o.seed==s)],20261003+s) for s in [7,101,2024]}
    public_pass=bool((sc.change_pct<0).all() and all(p<.025 and ci[1]<0 for p,ci in bs.values()))
    save(HERE/'result_v1.json',dict(public_pass=public_pass,decision='PUBLIC_PASS_PENDING_EL1' if public_pass else 'REJECT',bootstrap=bs,structure=structure,fold_audit=aud,checks=CHECKS,protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest()))
    print(sc.to_string(index=False));print('DECISION',public_pass,flush=True)
if __name__=='__main__':main()
