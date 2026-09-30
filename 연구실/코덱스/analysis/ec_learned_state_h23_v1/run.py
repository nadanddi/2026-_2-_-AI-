import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import hashlib,json
from datetime import datetime,timezone,timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
OLD=ROOT/'집/코덱스/analysis/local'
ACT=['act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
IND=['in_temp','in_hum','in_co2']
WEATHER=['out_temp','out_hum','out_rad','out_wspd']
RAW=IND+ACT+WEATHER
BASE=IND+ACT+['day','hr_sin','hr_cos','midnight']+[v+'_h0' for v in ACT+IND]+[n for v in ACT for n in (v+'_tdm',v+'_tdz')]
LAGS=[c+f'_lag{k}' for k in range(1,7) for c in RAW]
AUX=['day','hr_sin','hr_cos','farm47','in_temp']+ACT+WEATHER+LAGS
TARGET=['dah','dco2']
DIRECT=BASE+WEATHER+LAGS+TARGET
STATE=[f'{p}_{n}' for p in ('ah','co2') for n in ('expected','innovation','expected_cum','innovation_cum')]
FOLDS=(0,2,4,6,8,9)
SEEDS=(7,101)

def identify(raw):
    x=raw.copy();x['farm']=x.row_id.str[:3];x['day']=x.row_id.str[4:7].astype(int);x['hour']=x.row_id.str[8:10].astype(int)
    return x.sort_values(['farm','day','hour']).reset_index(drop=True)

def features(raw):
    x=identify(raw);assert x.row_id.is_unique
    x['hr_sin']=np.sin(2*np.pi*x.hour/24);x['hr_cos']=np.cos(2*np.pi*x.hour/24)
    x['midnight']=x.hour.eq(0).astype(float);x['farm47']=x.farm.eq('F47').astype(float)
    g=x.groupby(['farm','day'],sort=False);key=pd.MultiIndex.from_frame(x[['farm','day']])
    h0=x[x.hour.eq(0)].set_index(['farm','day'])
    for c in ACT+IND:x[c+'_h0']=h0[c].reindex(key).to_numpy()
    for c in ACT:
        x[c+'_tdm']=g[c].transform(lambda s:s.expanding().mean())
        x[c+'_tdz']=g[c].transform(lambda s:s.eq(0).astype(float).where(s.notna()).expanding().mean())
    lagged={}
    for k in range(1,7):
        shifted=g[RAW].shift(k);valid=(x.hour-g.hour.shift(k)).eq(k)
        for c in RAW:lagged[c+f'_lag{k}']=shifted[c].where(valid)
    x=pd.concat([x,pd.DataFrame(lagged,index=x.index)],axis=1)
    def ah(t,r):return 216.7*6.112*np.exp(17.67*t/(t+243.5))*r/100/(t+273.15)
    x['dah']=ah(x.in_temp,x.in_hum)-ah(x.in_temp_lag1,x.in_hum_lag1)
    x['dco2']=x.in_co2-x.in_co2_lag1
    return x

def split_mask(frame,days):
    near={(f,int(d)+k) for f,d in days for k in (-1,0,1)}
    return np.array([(f,int(d)) not in near for f,d in frame[['farm','day']].itertuples(index=False,name=None)])

def keys(frame):return sorted({(f,int(d)) for f,d in frame[['farm','day']].itertuples(index=False,name=None)})

def model(seed,trees=400,leaf=1):
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),ExtraTreesRegressor(
        n_estimators=trees,min_samples_leaf=leaf,max_features=1.0,random_state=seed,n_jobs=4))

def teacher(tr,va,seed,trees=200):
    use=tr[TARGET].notna().all(axis=1)&tr.hour.gt(0)
    fit=tr[use];assert len(fit)>10
    y=fit[TARGET].to_numpy(float);scale=y.std(axis=0);scale=np.maximum(scale,1e-8)
    m=model(seed,trees,8);m.fit(fit[AUX],y/scale)
    m.steps[-1][1].n_jobs=1
    pred=m.predict(va[AUX])*scale;pred[va.hour.eq(0)]=0
    return pred,m,scale

def crossfit(tr,seed,trees=200):
    groups=(tr.day//14+tr.farm.str[1:].astype(int))%3
    pred=np.empty((len(tr),2));audit=[]
    for k in range(3):
        select=groups.eq(k).to_numpy();va=tr[select];held=keys(va)
        fit=tr[split_mask(tr,set(held))];p,_,_=teacher(fit,va,seed+1000+k,trees)
        pred[select]=p;audit.append({'inner':k,'train_days':keys(fit),'held_days':held})
    assert np.isfinite(pred).all()
    return pred,audit

def states(frame,pred):
    s=frame[['farm','day','hour']].reset_index(drop=True).copy();target=frame[TARGET].to_numpy(float)
    for j,name in enumerate(('ah','co2')):
        s[name+'_expected']=pred[:,j]
        s[name+'_innovation']=target[:,j]-pred[:,j]
        s.loc[s.hour.eq(0),name+'_innovation']=0
        for typ in ('expected','innovation'):
            c=name+'_'+typ
            s[c+'_cum']=s.groupby(['farm','day'])[c].transform(lambda v:v.cumsum().where(v.expanding().count().eq(np.arange(1,len(v)+1))))
    return s[STATE]

def shrink(delta,frame):
    z=frame[['farm','day','hour']].reset_index(drop=True).copy();z['delta']=delta
    z=z.sort_values(['farm','day','hour']);avg=z.groupby(['farm','day']).delta.transform(lambda s:s.expanding().mean())
    out=np.empty(len(z));out[z.index]=(.5*z.delta+.5*avg).to_numpy();return out

def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))

def bootstrap(frame,a,b,seed):
    z=frame[['farm','day','sub_ec',a,b]].copy()
    z['delta']=(z.sub_ec-z[a])**2-(z.sub_ec-z[b])**2
    daily=z.groupby(['farm','day']).delta.mean().to_numpy();rng=np.random.default_rng(seed)
    vals=np.concatenate([daily[rng.integers(len(daily),size=(1000,len(daily)))].mean(axis=1) for _ in range(20)])
    return {'delta_mse':float(daily.mean()),'ci975':np.quantile(vals,[.0125,.9875]).tolist(),'p_worse':float(np.mean(vals>=0))}

def main():
    stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    out=ROOT/'연구실/코덱스/local/ec_learned_state_h23_v1'/stamp;out.mkdir(parents=True,exist_ok=False)
    hashes={}
    def hashed(p):
        p=Path(p);hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
    def read(p):return pd.read_csv(hashed(p))
    for p in (HERE/'run.py',HERE/'PROTOCOL.md',HERE/'test_run.py'):hashed(p)
    raw=read(Path(env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(('F13','F47'))].copy()
    lab=features(raw);assert len(lab)==9600
    # Verify real-data causality before EC training.
    ids=lab.loc[lab.farm.eq('F13')&lab.day.eq(100)&lab.hour.le(6),'row_id']
    changed=raw.copy();mask=changed.row_id.str.startswith('F47')|changed.row_id.str.startswith('F13_100_').mul(changed.row_id.str[-2:].astype(int).gt(6))
    changed.loc[mask,RAW]=777
    cols=list(dict.fromkeys(DIRECT+AUX+TARGET))
    pd.testing.assert_frame_equal(lab.set_index('row_id').loc[ids,cols],features(changed).set_index('row_id').loc[ids,cols])
    pd.testing.assert_frame_equal(lab.set_index('row_id').loc[ids,cols],features(raw[raw.row_id.isin(ids)]).set_index('row_id').loc[ids,cols])
    pd.testing.assert_frame_equal(lab,features(raw.sample(frac=1,random_state=230930)))
    lockpath=ROOT/'집/코덱스/analysis/codex_independent/ec_final_lock/locked_days.json'
    lock={(r['farm'],int(r['day'])) for r in json.loads(hashed(lockpath).read_text(encoding='utf-8'))['selected']}
    lab=lab.merge(read(Path(env.DATA)/'train_y.csv')[['row_id','sub_ec']],on='row_id',validate='one_to_one')
    lab=lab.merge(read(OLD/'rl_ec_v1/20260927_173801/splits.csv')[['row_id','fold']],on='row_id',validate='one_to_one')
    assert lab.sub_ec.notna().all()
    hold=set(keys(lab[lab.fold.isin((8,9))]));dev=lab[split_mask(lab,hold|lock)].copy()
    scores=[];rows=[];audits=[];auxscores=[]
    for fold in FOLDS:
        folder='ec_three_seed_ensemble_cv/20260928_035914' if fold<8 else 'ec_locked_confirmation/20260928_044934'
        saved=read(OLD/folder/f'fold{fold}.csv');va=lab.set_index('row_id').loc[saved.row_id].reset_index()
        np.testing.assert_allclose(va.sub_ec,saved.sub_ec,rtol=0,atol=1e-12)
        v2=saved['blend' if fold<8 else 'candidate'].to_numpy(float)
        held=set(keys(va));assert not held&lock
        tr=dev[split_mask(dev,held)] if fold<8 else dev.copy()
        assert not set(keys(tr)) & {(f,d+k) for f,d in held|lock|hold for k in (-1,0,1)}
        low,high=float(tr.sub_ec.min()),float(tr.sub_ec.max())
        for seed in SEEDS:
            print(f'START fold={fold} seed={seed} train_days={len(tr)//24}',flush=True)
            tp,inner=crossfit(tr,seed);vp,tm,scale=teacher(tr,va,seed)
            # Real model prediction order must not alter results.
            reverse=tm.predict(va[AUX].iloc[::-1])[::-1]*scale;reverse[va.hour.eq(0)]=0
            np.testing.assert_allclose(vp,reverse,rtol=0,atol=0)
            ts,vs=states(tr,tp),states(va,vp)
            # Learned-state prefix is also invariant when subsequent predictions/inputs are removed.
            early=va.hour.le(6).to_numpy();es=states(va[early],vp[early])
            np.testing.assert_allclose(vs.loc[early].to_numpy(),es.to_numpy(),equal_nan=True,rtol=0,atol=0)
            train=tr.reset_index(drop=True).join(ts);valid=va.reset_index(drop=True).join(vs)
            predictions={}
            for name,feat in [('base',BASE),('direct_et',DIRECT),('state_et',DIRECT+STATE)]:
                m=model(seed);m.fit(train[feat],train.sub_ec)
                m.steps[-1][1].n_jobs=1;predictions[name]=m.predict(valid[feat])
            rec=va[['row_id','farm','day','hour','sub_ec']].copy();rec['v2']=v2
            for name,pred in predictions.items():rec[name]=pred
            for name,et in [('direct','direct_et'),('candidate','state_et')]:rec[name]=np.clip(v2+.24*shrink(predictions[et]-predictions['base'],va),low,high)
            rec['fold']=fold;rec['seed']=seed
            aux=va[['row_id','farm','day','hour',*TARGET]].copy()
            for j,n in enumerate(TARGET):aux['pred_'+n]=vp[:,j]
            aux.to_csv(out/f'aux_fold{fold}_seed{seed}.csv',index=False,float_format='%.17g')
            rec.to_csv(out/f'fold{fold}_seed{seed}.csv',index=False,float_format='%.17g')
            record={'fold':fold,'seed':seed,'train_days':len(tr)//24,**{n:rmse(rec.sub_ec,rec[n]) for n in ['v2','direct','candidate','base','direct_et','state_et']}}
            record['vs_v2']=record['candidate']/record['v2']-1;record['vs_direct']=record['candidate']/record['direct']-1
            scores.append(record);rows.append(rec);auxscores.append(aux.assign(seed=seed))
            audits.append({'fold':fold,'seed':seed,'train_days':keys(tr),'held_days':keys(va),'inner':inner})
            (out/'progress.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
            (out/'training_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
            print(json.dumps(record),flush=True)
    allrows=pd.concat(rows,ignore_index=True);allaux=pd.concat(auxscores,ignore_index=True);summary={}
    screen=all(s['vs_v2']<0 and s['vs_direct']<0 for s in scores)
    for seed in SEEDS:
        g=allrows[allrows.seed.eq(seed)];a=allaux[allaux.seed.eq(seed)];s={n:rmse(g.sub_ec,g[n]) for n in ('v2','direct','candidate','base','direct_et','state_et')}
        s['comparisons']={n:bootstrap(g,'candidate',n,seed+230930) for n in ('v2','direct')}
        s['by_farm']={f:{n:rmse(q.sub_ec,q[n]) for n in ('v2','direct','candidate')} for f,q in g.groupby('farm')}
        s['aux']={}
        for n in TARGET:
            valid=a[n].notna()&a.hour.gt(0);q=a[valid];mse=float(np.mean((q[n]-q['pred_'+n])**2));zero=float(np.mean(q[n]**2))
            s['aux'][n]={'rows':len(q),'rmse':mse**.5,'persistence_rmse':zero**.5,'skill_vs_zero':1-mse/zero}
            screen &= mse<zero
        for b in s['comparisons'].values():screen &= b['p_worse']<.0125 and b['ci975'][1]<0
        for b in s['by_farm'].values():screen &= b['candidate']<b['v2'] and b['candidate']<b['direct']
        daily=g.groupby(['farm','day']).sub_ec.mean();highidx={k for k,v in daily.items() if v>=1.2}
        highmask=np.array([(f,int(d)) in highidx for f,d in g[['farm','day']].itertuples(index=False,name=None)])
        s['subsets']={n:{'rows':len(q),**{c:rmse(q.sub_ec,q[c]) for c in ('v2','direct','candidate')}} for n,q in [('high',g[highmask]),('ordinary',g[~highmask]),('early',g[g.hour.le(6)])]}
        summary[str(seed)]=s
    result={'hypothesis':'H23','scores':scores,'per_seed':summary,'screen_pass':bool(screen),'causality':'PASS','prediction_order':'PASS','nested_audit':'PASS','final_lock_scored':False,'test_X_read':False,'submission_created':False,'sha256':hashes}
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('OUTPUT',str(out),flush=True);print(json.dumps({k:v for k,v in result.items() if k not in ('scores','sha256')},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
