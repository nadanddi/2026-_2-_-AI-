from pathlib import Path
import os, sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import csv,json,math,hashlib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
RAW=['out_temp','out_hum','out_rad','out_wspd','in_temp','in_hum','in_co2','act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
ACT=['act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog','out_rad']
def read(p):
    with p.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def score(y,p):
    a=float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
    b=math.sqrt(math.fsum((float(x)-float(v))**2 for x,v in zip(y,p))/len(y))
    assert math.isclose(a,b,abs_tol=1e-12)
    return a
def features(x):
    x=x.sort_values(['farm','day','hour']).copy();g=x.groupby(['farm','day'],sort=False)
    t=pd.DataFrame(index=x.index)
    t['day']=x.day;t['hour']=x.hour
    for k in [1,2]:
        t[f'sin{k}']=np.sin(x.hour*np.pi*k/12);t[f'cos{k}']=np.cos(x.hour*np.pi*k/12)
    i=t.copy();i['temp']=x.in_temp;i['temp2']=x.in_temp**2
    i['h0']=g.in_temp.transform(lambda z:z.iloc[0]);i['cumtemp']=g.in_temp.transform(lambda a:a.expanding().mean())
    for l in [1,3]: i[f'lag{l}']=g.in_temp.shift(l).fillna(i.h0)
    i['d0']=i.temp-i.h0
    a=t.copy()
    for c in ACT:
        a[c]=x[c];a[c+'_h0']=g[c].transform(lambda z:z.iloc[0]);a[c+'_cum']=g[c].transform(lambda z:z.expanding().mean())
    return {'I':i,'A':a,'C':t}
def main():
    lockpath=Path(env.CODEX)/'ec_final_lock/locked_days.json'
    lock={(r['farm'],int(r['day'])) for r in json.loads(lockpath.read_text(encoding='utf-8'))['selected']}
    records=[]
    for r in read(Path(env.DATA)/'train_y.csv'):
        f,d,h=r['row_id'].split('_')
        if f not in ['F13','F47'] or (f,int(d)) in lock: continue
        records.append({'row_id':r['row_id'],'farm':f,'day':int(d),'hour':int(h),'sub_ec':float(r['sub_ec']),'sub_temp':float(r['sub_temp'])})
    y=pd.DataFrame(records).sort_values(['farm','day','hour']).reset_index(drop=True)
    raw=pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id']+RAW)
    d=y.merge(raw,on='row_id',validate='one_to_one')
    splitpath=ROOT/'집/코덱스/analysis/local/rl_ec_v1/20260927_173801/splits.csv'
    d=d.merge(pd.read_csv(splitpath)[['row_id','block','fold']],on='row_id',validate='one_to_one').sort_values(['farm','day','hour']).reset_index(drop=True)
    assert len(d)==8640 and d.row_id.is_unique
    g=d.groupby(['farm','day']);assert g.size().eq(24).all()
    d['level']=g.sub_ec.transform('mean');d['shape']=d.sub_ec-d.level
    # Construct causal inputs over the full input panel, so input-only lock/gap rows remain available.
    allx=raw[raw.row_id.str[:3].isin(['F13','F47'])].copy()
    ids=allx.row_id.str.split('_',expand=True);allx['farm']=ids[0];allx['day']=ids[1].astype(int);allx['hour']=ids[2].astype(int)
    allx=allx.sort_values(['farm','day','hour']).reset_index(drop=True)
    fs=features(allx);base=allx[['row_id']]
    design={k:base.join(v).set_index('row_id').loc[d.row_id].reset_index(drop=True) for k,v in fs.items()}
    for lag in [0,1,3]:
        tt=g.sub_temp.shift(lag).fillna(g.sub_temp.transform('first')) if lag else d.sub_temp
        center=tt-tt.groupby([d.farm,d.day]).transform('mean')
        design[f'T{lag}']=pd.DataFrame({'dt':center,'dt2':center**2})
    # Future and other-farm changes cannot affect current/past same-farm causal designs.
    cutoff_day=int(allx.loc[allx.farm.eq('F13'),'day'].median());past=allx.farm.eq('F13') & ((allx.day*24+allx.hour)<=cutoff_day*24+6)
    for mask in [allx.farm.eq('F47'),allx.farm.eq('F13') & ~past]:
        changed=allx.copy();changed.loc[mask,RAW]=changed.loc[mask,RAW]*7+999
        new=features(changed)
        for k in ['I','A','C']: pd.testing.assert_frame_equal(fs[k].loc[past],new[k].loc[past])
    daily=d.groupby(['farm','day']).agg(level=('sub_ec','mean'),fold=('fold','first'),block=('block','first'),sub_temp_mean=('sub_temp','mean'),in_temp_mean=('in_temp','mean'),in_temp_min=('in_temp','min'),in_temp_max=('in_temp','max')).reset_index()
    for c in ACT: daily[c]=g[c].mean().to_numpy()
    daily['farm47']=daily.farm.eq('F47').astype(float);daily['day2']=daily.day**2
    ld={'L-T':daily[['farm47','sub_temp_mean']].assign(temp2=daily.sub_temp_mean**2),
        'L-I':daily[['farm47','in_temp_mean','in_temp_min','in_temp_max']].assign(temp2=daily.in_temp_mean**2),
        'L-A':daily[['farm47']+ACT], 'L-C':daily[['farm47','day','day2']]}
    pred={k:np.full(len(d),np.nan) for k in design};lp={k:np.full(len(daily),np.nan) for k in ld};foldscores=[]
    forbidden={(f,z+j) for f,z in lock for j in [-1,0,1]}
    for fold in range(10):
        va=d.fold.eq(fold);vdays=set(d.loc[va,['farm','day']].itertuples(index=False,name=None))
        blocked=forbidden|{(f,z+j) for f,z in vdays for j in [-1,0,1]}
        tr=np.array([(f,z) not in blocked for f,z in zip(d.farm,d.day)])
        for k,x in design.items():
            m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=1))
            m.fit(x.loc[tr],d.loc[tr,'shape']);pred[k][va]=m.predict(x.loc[va])
            foldscores.append({'model':k,'fold':fold,'train_rmse':score(d.loc[tr,'shape'],m.predict(x.loc[tr])),'val_rmse':score(d.loc[va,'shape'],pred[k][va])})
        vl=daily.fold.eq(fold);tl=np.array([(f,z) not in blocked for f,z in zip(daily.farm,daily.day)])
        for k,x in ld.items():
            m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=1))
            m.fit(x.loc[tl],daily.loc[tl,'level']);lp[k][vl]=m.predict(x.loc[vl])
            foldscores.append({'model':k,'fold':fold,'train_rmse':score(daily.loc[tl,'level'],m.predict(x.loc[tl])),'val_rmse':score(daily.loc[vl,'level'],lp[k][vl])})
        print(f'fold {fold} complete',flush=True)
    oof=pd.read_csv(ROOT/'집/코덱스/local/ec_restart_phase3_20261001_v1/oof_predictions.csv')
    oo=oof[oof.validator.eq('DIAG10')].set_index('row_id').loc[d.row_id]
    assert np.allclose(oo.sub_ec.to_numpy(),d.sub_ec.to_numpy(),atol=5e-15,rtol=0)
    d['v2']=oo.v2.to_numpy();e=d.v2-d.sub_ec
    em=e.groupby([d.farm,d.day]).transform('mean')
    baseline={'overall':score(d.sub_ec,d.v2),'level':float(np.sqrt(np.mean(em**2))),'shape':float(np.sqrt(np.mean((e-em)**2)))}
    result={'status':'PASS','baseline':baseline,'models':{},'causality_checks':'PASS','lock_scored':False,'test_X_read':False,'rows':len(d),'days':len(daily)}
    rng=np.random.default_rng(261002);draws={}
    # Share bootstrap block resamples across the ten prespecified equations.
    for f in ['F13','F47']:
        n=daily.loc[daily.farm.eq(f),'block'].nunique();draws[f]=rng.integers(0,n,size=(20000,n))
    for k,p in {**pred,**lp}.items():
        assert np.isfinite(p).all()
        frame=d if k in pred else daily;target=d['shape'] if k in pred else daily.level
        errors=(target.to_numpy()-p)**2;tot=np.zeros((20000,2))
        for f in ['F13','F47']:
            ff=frame.farm.eq(f);z=pd.DataFrame({'block':frame.loc[ff,'block'],'n':1,'ss':errors[ff]})
            z=z.groupby('block')[['n','ss']].sum().to_numpy();tot+=z[draws[f]].sum(axis=1)
        ci=np.quantile(np.sqrt(tot[:,1]/tot[:,0]),[.0025,.9975]).tolist();threshold=.05 if k in pred else .1
        strata={}
        for label,mask in [('F13',frame.farm.eq('F13')),('F47',frame.farm.eq('F47')),('early',frame.day.lt(179)),('late',frame.day.ge(179))]:
            strata[label]={'n':int(mask.sum()),'rmse':score(target[mask],p[mask])}
        result['models'][k]={'rmse':score(target,p),'ci995':ci,'threshold':threshold,'threshold_pass':bool(ci[1]<=threshold and all(strata[f]['rmse']<=threshold for f in ['F13','F47'])),'strata':strata,'availability':('causal_inputs' if k in ['I','A','C','L-C'] else 'oracle_unavailable')}
    for k,p in pred.items(): d[k]=p
    for k,p in lp.items():daily[k]=p
    d[['row_id','farm','day','hour','block','fold','sub_ec','shape','v2']+list(pred)].to_csv(HERE/'shape_oof.csv',index=False)
    daily.to_csv(HERE/'level_oof.csv',index=False);pd.DataFrame(foldscores).to_csv(HERE/'fold_scores.csv',index=False)
    result['hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(env.DATA)/'train_X.csv',Path(env.DATA)/'train_y.csv',splitpath,lockpath,HERE/'PROTOCOL.md',Path(__file__)]}
    (HERE/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1): main()
