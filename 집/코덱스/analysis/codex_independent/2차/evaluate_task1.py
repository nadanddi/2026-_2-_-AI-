# -*- coding: utf-8 -*-
import sys,json,hashlib,warnings
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
ROOT=H.parents[2]
sys.path.insert(0,str(ROOT/'.analysis-tools/python'))
sys.path.insert(0,str(ROOT/'research'))
import env
import numpy as np
import pandas as pd
import common
import train_flags_v6 as TF
import features_v4 as F4
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from resid_reset_features import build_features,FEATURE_COLUMNS,PHYSICS_COLUMNS,USABLE
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)

def folds(d):
    fs=[{f:set() for f in ['F13','F47']} for _ in range(10)]
    for f in ['F13','F47']:
        ds=sorted(d.loc[d.farm==f,'day'].unique())
        for i in range(0,len(ds),5): fs[(i//5)%10][f].update(map(int,ds[i:i+5]))
    return fs

def model_predict(tr,va,w=None):
    lin=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.))
    kw={} if w is None else {'ridge__sample_weight':w}
    lin.fit(tr[PHYSICS_COLUMNS],tr.sub_temp.to_numpy(),**kw)
    model=LGBMRegressor(n_estimators=220,learning_rate=.035,num_leaves=12,max_depth=-1,min_child_samples=100,reg_lambda=15,verbosity=-1,n_jobs=4,random_state=726)
    model.fit(tr[FEATURE_COLUMNS],tr.sub_temp.to_numpy()-lin.predict(tr[PHYSICS_COLUMNS]),sample_weight=w)
    return lin.predict(va[PHYSICS_COLUMNS])+model.predict(va[FEATURE_COLUMNS])

def comparison(d,b,p):
    if not len(d): return {'n':0,'days':0}
    y=d.sub_temp.to_numpy(); g=pd.factorize(pd.MultiIndex.from_frame(d[['farm','day']]))[0]
    n=np.bincount(g); sb=np.bincount(g,weights=(b-y)**2); sp=np.bincount(g,weights=(p-y)**2)
    rng=np.random.default_rng(726); ix=rng.integers(0,len(n),(2000,len(n)))
    diff=np.sqrt(sp[ix].sum(1)/n[ix].sum(1))-np.sqrt(sb[ix].sum(1)/n[ix].sum(1))
    rb=float(np.sqrt(sb.sum()/n.sum())); rp=float(np.sqrt(sp.sum()/n.sum()))
    return {'n':len(d),'days':len(n),'base':rb,'new':rp,'delta':rp-rb,'ci95':np.quantile(diff,[.025,.975]).tolist(),'bias_base':float(np.mean(b-y)),'bias_new':float(np.mean(p-y))}

def masks(d):
    basic={'all':np.ones(len(d),bool),'clean':d.clean.to_numpy(),'clean_low_noise':d.clean.to_numpy()&d.low_noise.to_numpy()}
    subs={'all':np.ones(len(d),bool),'hour00_03':d.hour.to_numpy()<4,'hour04_23':d.hour.to_numpy()>=4,'second_cold10':(d.day.to_numpy()>=179)&(d.day_ph3_min.to_numpy()<10),'F47_second_cold10':(d.farm.to_numpy()=='F47')&(d.day.to_numpy()>=179)&(d.day_ph3_min.to_numpy()<10)}
    for f in ['F13','F47']:
        for part in [1,2]: subs[f+'_part'+str(part)]=(d.farm.to_numpy()==f)&((d.day.to_numpy()<179) if part==1 else (d.day.to_numpy()>=179))
    return {a+'|'+b:ma&mb for a,ma in basic.items() for b,mb in subs.items()}

def summarize(d,base,preds):
    out={}
    for v,p in preds.items():
        out[v]={k:comparison(d.loc[m].reset_index(drop=True),base[m],(.8*base+.2*p)[m]) for k,m in masks(d).items()}
    # 가중치의 효과는 동일한 20% 혼합 간에도 짝지어 비교.
    for v in ['weighted','weighted_foldlocal']:
        out[v+'_vs_unweighted']={k:comparison(d.loc[m].reset_index(drop=True),(.8*base+.2*preds['unweighted'])[m],(.8*base+.2*preds[v])[m]) for k,m in masks(d).items()}
    return out

def main():
    tx,ty,sx=common.load_raw()
    watch=[ROOT/'research/train_flags_v6.py',ROOT/'research/causality_v6.py',ROOT/'research/eval_v6.py',ROOT/'analysis/codex_independent/independent_analysis.py',ROOT/'analysis/codex_independent/보고서.md',ROOT/'.gitignore']+[Path(common.DATA)/f for f in ['train_X.csv','train_y.csv','test_X.csv']]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in watch}
    (H/'original_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    z=build_features(tx,sx)
    saved=np.load(ROOT/'research/local/eval_v6_oof.npz',allow_pickle=True)
    d=z.merge(ty[['row_id','sub_temp']],on='row_id').set_index('row_id').loc[saved['row_id']].reset_index()
    assert d.sub_temp.notna().all() and len(d)==9600
    # 1차 후보 피처 및 예측 일치를 검증한다.
    sys.path.insert(0,str(H.parent))
    import independent_analysis as old
    oz=old.features(pd.concat([tx,sx],ignore_index=True)).set_index('row_id').loc[z.row_id]
    aa=z[FEATURE_COLUMNS].to_numpy(dtype=np.float64); bb=oz[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    assert np.array_equal(np.ascontiguousarray(aa).view(np.uint64),np.ascontiguousarray(bb).view(np.uint64))
    w=TF.row_weights(d,.2,w_noisy=.2)
    d['clean']=TF.row_weights(d,0.,radius=3)>=1
    nd=TF.noisy_days(); threshold=float(nd.noise_score.quantile(.75))
    nd['low_noise']=nd.noise_score<threshold
    d=d.merge(nd[['farm','day','noise_score','noisy','low_noise','cold']],on=['farm','day'],how='left')
    ph=F4.phys_features().set_index('row_id').loc[d.row_id,'ph_in_temp_3'].to_numpy()
    d['ph3']=ph; d['day_ph3_min']=d.groupby(['farm','day']).ph3.transform('min')
    d['weight']=w
    configs={'DIAG10':folds(d)}
    for t in [8,10,12]:
        days=d.loc[d.day_ph3_min<t,['farm','day']].drop_duplicates()
        configs['EXT'+str(t)]=[{f:set(days.loc[days.farm==f,'day']) for f in ['F13','F47']}]
    results={'noise_threshold':threshold,'counts':{'n':len(d),'clean':int(d.clean.sum()),'clean_low_noise':int((d.clean&d.low_noise).sum()),'noisy_days':int(nd.noisy.sum()),'top25_days':int((~nd.low_noise).sum()),'weights':{str(v):int((w==v).sum()) for v in np.unique(w)}},'feature_parity_bits':True,'evaluations':{}}
    allpred={}
    for name,fs in configs.items():
        base=pd.Series(saved['F60ND__'+name],index=saved['row_id']).loc[d.row_id].to_numpy()
        preds={v:np.full(len(d),np.nan) for v in ['unweighted','weighted','weighted_foldlocal']}
        foldres=[]
        for j,fd in enumerate(fs):
            tr,va=common.split_mask(d,fd)
            original=TF._train_frame
            try:
                allowed=set(d.loc[tr,'row_id'])
                TF._train_frame=lambda:tx[tx.row_id.isin(allowed)].sort_values(['farm','t']).reset_index(drop=True).copy()
                wl=TF.row_weights(d.loc[tr],.2,w_noisy=.2)
            finally: TF._train_frame=original
            for v,ww in [('unweighted',None),('weighted',w[tr]),('weighted_foldlocal',wl)]:
                preds[v][va]=model_predict(d.loc[tr],d.loc[va],ww)
            foldres.append({'fold':j+1,'train_rows':int(tr.sum()),'val_rows':int(va.sum()),'metrics':summarize(d.loc[va].reset_index(drop=True),base[va],{v:p[va] for v,p in preds.items()})})
            print(name,'fold',j+1,'done',flush=True)
        valid=np.isfinite(base)
        assert all(np.array_equal(np.isfinite(p),valid) for p in preds.values())
        results['evaluations'][name]={'aggregate':summarize(d.loc[valid].reset_index(drop=True),base[valid],{v:p[valid] for v,p in preds.items()}),'folds':foldres}
        for v,p in preds.items():allpred[name+'__'+v]=p
        print(name,'all',{v:results['evaluations'][name]['aggregate'][v]['all|all'] for v in preds},flush=True)
    prev=np.load(H.parent/'sub_temp_independent_oof.npz',allow_pickle=True)
    oldp=pd.Series(prev['resid_reset'],index=prev['row_id']).loc[d.row_id].to_numpy()
    results['unweighted_oof_max_abs_difference']=float(np.max(np.abs(oldp-allpred['DIAG10__unweighted'])))
    assert results['unweighted_oof_max_abs_difference']<1e-10
    np.savez_compressed(H/'validation_predictions.npz',row_id=d.row_id.to_numpy(),**allpred)
    d.to_json(H/'validation_rows.json',orient='records',force_ascii=False,double_precision=15)
    (H/'validation_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print('ALL DONE',flush=True)
if __name__=='__main__':main()
