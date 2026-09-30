import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import hashlib,json
from datetime import datetime,timezone,timedelta
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ACT=['act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
IND=['in_temp','in_hum','in_co2']
NAMES=['closed_sink','fan_vpd','saturation_cooling','heat_vpd']

def features(raw):
    x=raw.copy()
    x['farm']=x.row_id.str[:3];x['day']=x.row_id.str[4:7].astype(int);x['hour']=x.row_id.str[8:].astype(int)
    x=x.sort_values(['farm','day','hour']).reset_index(drop=True)
    g=x.groupby(['farm','day'])
    es=6.112*np.exp(17.67*x.in_temp/(x.in_temp+243.5))
    x['ah']=216.7*es*x.in_hum/100/(x.in_temp+273.15)
    x['vpd']=(.6108*np.exp(17.27*x.in_temp/(x.in_temp+237.3))*(1-x.in_hum/100)).clip(lower=0)
    prev=g[['ah','in_temp','in_hum','act_vent','act_fog']].shift(1)
    valid=x.hour.eq(0)|(x[['ah','act_vent','act_fog']].notna().all(axis=1)&prev[['ah','act_vent','act_fog']].notna().all(axis=1))
    closed=x.act_vent.eq(0)&prev.act_vent.eq(0)&x.act_fog.eq(0)&prev.act_fog.eq(0)
    x['closed_sink_h']=np.where(x.hour.eq(0),0,np.where(closed,(prev.ah-x.ah).clip(lower=0),0))
    x.loc[~valid,'closed_sink_h']=np.nan
    x['fan_vpd_h']=x.vpd*x.act_circfan.clip(0,100)/100
    x['heat_vpd_h']=x.vpd*x.act_heating.clip(0,100)/100
    validcool=x.hour.eq(0)|(x.in_temp.notna()&prev.in_temp.notna()&prev.in_hum.notna())
    x['saturation_cooling_h']=np.where(x.hour.eq(0),0,np.where(prev.in_hum.ge(95),(prev.in_temp-x.in_temp).clip(lower=0),0))
    x.loc[~validcool,'saturation_cooling_h']=np.nan
    for n in NAMES:
        x[n]=x.groupby(['farm','day'])[n+'_h'].transform(lambda s:s.cumsum().where(s.expanding().count().eq(np.arange(1,len(s)+1))))
    return x

def main():
    here=Path(__file__).resolve().parent
    out=ROOT/'연구실/코덱스/local/ec_management_d14_v1'/datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    hashes={}
    def read(p):
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        return pd.read_csv(p)
    for p in (here/'run.py',here/'PROTOCOL.md'):
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    old=ROOT/'집/코덱스/analysis/local';frames=[]
    for f in (0,2,4,6,8,9):
        folder='ec_three_seed_ensemble_cv/20260928_035914' if f<8 else 'ec_locked_confirmation/20260928_044934'
        z=read(old/folder/f'fold{f}.csv');z['v2']=z['blend' if f<8 else 'candidate'];z['fold']=f
        frames.append(z[['row_id','farm','day','hour','sub_ec','v2','fold']])
    v=pd.concat(frames,ignore_index=True)
    assert len(v)==5616 and v.row_id.is_unique
    raw=read(Path(env.DATA)/'train_X.csv');raw=raw[raw.row_id.isin(v.row_id)].copy()
    feat=features(raw)
    future=raw.copy();future.loc[future.row_id.str[-2:].astype(int).gt(6),IND+ACT]=777
    test=features(future)
    early=feat.hour.le(6)
    np.testing.assert_allclose(feat.loc[early,NAMES],test.loc[early,NAMES],equal_nan=True,rtol=0,atol=0)
    truncated=features(raw[raw.row_id.str[-2:].astype(int).le(6)])
    np.testing.assert_allclose(feat.loc[early,NAMES],truncated[NAMES],equal_nan=True,rtol=0,atol=0)
    daily=v.groupby(['farm','day']).sub_ec.mean().rename('y')
    e=v[v.hour.eq(6)].set_index(['farm','day'])[['fold','v2']].join(daily)
    e=e.join(feat[feat.hour.eq(6)].set_index(['farm','day'])[NAMES])
    h20=read(old/'ec_moisture_flux_h20/20260929_123524/input_only_moisture.csv')
    e=e.join(h20[h20.hour.eq(6)].set_index(['farm','day'])[['ah_gap','vent_flux_cum','closed_rise_cum']])
    prefix=feat[early]
    control=[]
    for h in (0,6):
        a=feat[feat.hour.eq(h)].set_index(['farm','day'])[IND+ACT].add_suffix('_h'+str(h));e=e.join(a);control+=list(a)
    for c in ACT:
        name=c+'_mean';e[name]=prefix.groupby(['farm','day'])[c].mean();control.append(name)
    for c in IND:
        gg=prefix.groupby(['farm','day'])[c]
        for name,vals in [(c+'_mean',gg.mean()),(c+'_std',gg.std()),(c+'_coverage',gg.count()/7),
            (c+'_rough',gg.apply(lambda s:s.diff().diff().abs().mean()))]:
            e[name]=vals;control.append(name)
    control+=['ah_gap','vent_flux_cum','closed_rise_cum']
    e=e.reset_index().sort_values(['farm','day']).reset_index(drop=True)
    e['late']=e.day.ge(179);e['target']=e.y-e.v2
    assert len(e)==234 and e[['target','v2']].notna().all().all()
    raw_missing={c:float(e[c].isna().mean()) for c in NAMES}
    complete=e[NAMES].notna().all(axis=1)
    for c in control+NAMES:
        e[c]=e[c].fillna(e.groupby('farm')[c].transform('median')).fillna(e[c].median())
    def matrices(frame):
        z=frame[control].to_numpy(float)
        sd=z.std(axis=0);z=(z-z.mean(axis=0))/np.where(sd>1e-12,sd,1)
        c=np.column_stack([np.ones(len(frame)),frame.farm.eq('F47'),frame.late,frame.day/100,frame.v2,frame.v2**2,z])
        u,s,_=np.linalg.svd(c,full_matrices=False);rank=int(np.sum(s>s[0]*1e-10));q=u[:,:rank]
        y=frame.target.to_numpy(float);y=y-q@(q.T@y)
        x=frame[NAMES].to_numpy(float);x=x-q@(q.T@x)
        xn=np.linalg.norm(x,axis=0);yn=np.linalg.norm(y)
        usable=xn>1e-8*np.maximum(np.linalg.norm(frame[NAMES].to_numpy(float),axis=0),1)
        x[:,~usable]=0
        r=(x.T@y)/(np.maximum(xn,1e-15)*yn)
        return r,x/np.maximum(xn,1e-15),y,q,rank,float(s[0]/s[rank-1])
    r,x,y,q,rank,condition=matrices(e)
    variants={}
    top=np.argsort(np.abs(y))[-5:]
    for name,frame in [('F13',e[e.farm.eq('F13')]),('F47',e[e.farm.eq('F47')]),
        ('without_top5',e.drop(top)),('without_high',e[e.y.lt(1.2)]),('complete_case',e[complete]),
        *[(f'without_fold{f}',e[e.fold.ne(f)]) for f in (0,2,4,6,8,9)]]:
        rr,_,_,_,rk,cond=matrices(frame);variants[name]=dict(days=len(frame),rank=rk,condition_number=cond,r=rr.tolist())
    rng=np.random.default_rng(300914);groups=list(e.groupby(['farm','fold']).indices.values())
    null=[]
    for _ in range(100):
        p=np.tile(y[:,None],(1,100))
        for j in range(100):
            for idx in groups:p[idx,j]=y[rng.permutation(idx)]
        p=p-q@(q.T@p)
        null.extend(np.max(np.abs(x.T@p)/np.maximum(np.linalg.norm(p,axis=0),1e-15),axis=0))
    null=np.asarray(null);pmax=[float((1+np.sum(null>=abs(a)))/(10001)) for a in r]
    signals=[]
    for k,name in enumerate(NAMES):
        sign=np.sign(r[k]);ok=pmax[k]<.01
        for farm in ('F13','F47'):
            rr=variants[farm]['r'][k]
            ok &= bool(np.sign(rr)==sign and abs(rr)>=.15 and int((complete&e.farm.eq(farm)).sum())>=80)
        for nm in ('without_top5','without_high'):
            rr=variants[nm]['r'][k];ok &= bool(np.sign(rr)==sign and abs(rr)>=.10)
        for f in (0,2,4,6,8,9):ok &= bool(np.sign(variants[f'without_fold{f}']['r'][k])==sign)
        signals.append(dict(name=name,partial_r=float(r[k]),p_max=pmax[k],gate_pass=bool(ok)))
    e.to_csv(out/'daily_features.csv',index=False)
    result=dict(status='PASS',days=len(e),controls=control,control_rank=rank,condition_number=condition,signals=signals,sensitivity=variants,
        missing_fraction=raw_missing,complete_days=int(complete.sum()),permutations=10000,seed=300914,
        causality_future_perturbation='PASS',causality_prefix_only='PASS',final_lock_scored=False,
        test_read=False,model_trained=False,sha256=hashes)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sha256','controls')},ensure_ascii=False,indent=2))
    print('OUTPUT',out)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
