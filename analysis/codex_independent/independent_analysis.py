# -*- coding: utf-8 -*-
"""독립 분석. 기존 파일 읽기 전용; 출력은 이 스크립트 폴더에만 생성."""
import sys, os, json, hashlib, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / '.analysis-tools/python'))
sys.path.insert(0, str(ROOT / 'research'))
import env
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
import common

def features(a):
    a = a[a.farm.isin(['F13','F47'])].sort_values(['farm','t']).reset_index(drop=True).copy()
    z = a[['row_id','farm','day','hour','t']].copy()
    z['farm_id'] = (a.farm == 'F47').astype(float)
    z['sin'] = np.sin(a.hour * np.pi/12)
    z['cos'] = np.cos(a.hour * np.pi/12)
    z['second'] = (a.day >= 179).astype(float)
    cols = common.USABLE
    for c in cols:
        s = a[c]
        gb = s.groupby([a.farm,a.day])
        z[c] = s
        z[c+'_h0'] = s.where(a.hour == 0).groupby([a.farm,a.day]).ffill()
        z[c+'_mean'] = gb.transform(lambda x: x.expanding().mean())
        z[c+'_std'] = gb.transform(lambda x: x.expanding().std())
        z[c+'_diff'] = gb.diff()
        for hl in ([1,3,8] if c in ['in_temp','out_temp','out_rad','act_heating'] else []):
            z[c+'_reset'+str(hl)] = gb.transform(lambda x: x.ewm(halflife=hl).mean())
            z[c+'_hist'+str(hl)] = s.groupby(a.farm).transform(lambda x: x.ewm(halflife=hl).mean())
        prev = s.groupby(a.farm).shift().where(a.groupby('farm').t.diff() == 1)
        z[c+'_seam'] = (s-prev).where(a.hour == 0).groupby([a.farm,a.day]).ffill()
    z['delta'] = a.in_temp-a.out_temp
    return z

def folds(d):
    out = [{f:set() for f in ['F13','F47']} for _ in range(10)]
    for f in ['F13','F47']:
        ds = sorted(d.loc[d.farm == f,'day'].unique())
        for i in range(0,len(ds),5):
            out[(i//5)%10][f].update(int(x) for x in ds[i:i+5])
    return out

def ridge():
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.0))

def lgb():
    return LGBMRegressor(n_estimators=220,num_leaves=12,max_depth=-1,min_child_samples=100,learning_rate=.035,reg_lambda=15,verbosity=-1,n_jobs=4,random_state=726)

def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))

def compare(d,y,b,p):
    groups=pd.factorize(pd.MultiIndex.from_frame(d[['farm','day']]))[0]
    n=np.bincount(groups); sb=np.bincount(groups,weights=(y-b)**2); sp=np.bincount(groups,weights=(y-p)**2)
    rng=np.random.default_rng(726)
    ids=rng.integers(0,len(n),(2000,len(n)))
    delta=np.sqrt(sp[ids].sum(1)/n[ids].sum(1))-np.sqrt(sb[ids].sum(1)/n[ids].sum(1))
    result={'rmse':rmse(y,p),'delta':rmse(y,p)-rmse(y,b),'ci95':np.quantile(delta,[.025,.975]).tolist()}
    for name,m in [('early',d.hour.values<4),('late',d.hour.values>=4),('second',d.day.values>=179),('cold',d.in_temp.values<10)]:
        if m.sum(): result[name]={'n':int(m.sum()),'base':rmse(y[m],b[m]),'new':rmse(y[m],p[m])}
    return result

def main():
    tx,ty,sx=common.load_raw()
    a=pd.concat([tx,sx],ignore_index=True)
    z=features(a)
    # 미래/다른 온실 입력 및 테스트 전체 분포 교란: 8개 절단점 전 행 확인.
    audits=[]
    for farm in ['F13','F47']:
        for hour in [0,3,12,23]:
            cut=205*24+hour
            aa=a.copy()
            change=(aa.farm != farm)|(aa.t>cut)
            aa.loc[change,common.USABLE]=aa.loc[change,common.USABLE]*17+133
            zz=features(aa)
            m=(z.farm==farm)&(z.t<=cut)
            pd.testing.assert_frame_equal(z.loc[m],zz.loc[m])
            audits.append({'farm':farm,'cut_t':cut,'rows_checked':int(m.sum()),'pass':True})
    basecols=[c for c in z.columns if c not in ['row_id','farm','t'] and '_hist' not in c and '_seam' not in c]
    histcols=[c for c in z.columns if c not in ['row_id','farm','t']]
    physreset=['farm_id','sin','cos','in_temp','out_temp','out_rad','act_heating']+[c for c in z if '_reset' in c]
    physhist=physreset+[c for c in z if '_hist' in c]
    result={'audit':audits,'feature_count':{'reset':len(basecols),'history':len(histcols)},'targets':{}}
    for target,file,key in [('sub_temp','eval_v6_oof.npz','F60ND__DIAG10'),('sub_ec','oof_ec_diag.npz','oof')]:
        saved=np.load(ROOT/'research/local'/file,allow_pickle=True)
        d=z.merge(ty[['row_id',target]],on='row_id').merge(pd.DataFrame({'row_id':saved['row_id'],'baseline':saved[key]}),on='row_id')
        d=d[d[target].notna()&np.isfinite(d.baseline)].reset_index(drop=True)
        assert d.row_id.is_unique
        y=d[target].to_numpy(); b=d.baseline.to_numpy()
        fd=folds(d)
        names=['ridge_reset','ridge_history','resid_reset','resid_history'] if target=='sub_temp' else ['ec_reset','ec_highweight']
        pred={name:np.full(len(d),np.nan) for name in names}
        for k,fold in enumerate(fd):
            tr,va=common.split_mask(d,fold)
            for name in names:
                history=name.endswith('history')
                cols=histcols if history else basecols
                if target=='sub_temp':
                    pc=physhist if history else physreset
                    lin=ridge().fit(d.loc[tr,pc],y[tr])
                    pv=lin.predict(d.loc[va,pc])
                    if name.startswith('resid'):
                        res=y[tr]-lin.predict(d.loc[tr,pc])
                        model=lgb().fit(d.loc[tr,cols],res)
                        pv=pv+model.predict(d.loc[va,cols])
                else:
                    w=np.where(y[tr]>1.5,3.,1.) if name=='ec_highweight' else None
                    model=lgb().fit(d.loc[tr,cols],y[tr],sample_weight=w)
                    pv=model.predict(d.loc[va,cols])
                pred[name][va]=pv
            print(target,'fold',k+1,'/10',flush=True)
        assert all(np.isfinite(p).all() for p in pred.values())
        e=b-y
        d['error']=e
        day=d.groupby(['farm','day']).agg(n=('error','size'),bias=('error','mean'),ecmean=(target,'mean'))
        dm=d.groupby(['farm','day']).error.transform('mean').to_numpy()
        diag={'n':len(d),'days':len(day),'baseline':rmse(y,b),'day_sse_share':float(np.sum(dm**2)/np.sum(e**2)),
              'within_day_rmse':rmse(e,dm),'hours':{},'candidates':{},'daily_correlations':{}}
        for h in range(24):
            m=d.hour.values==h
            diag['hours'][str(h)]={'n':int(m.sum()),'rmse':rmse(y[m],b[m]),'bias':float(e[m].mean())}
        # 事後診断の相関。モデルの入力には使用しない。
        for h in [0,3,12,23]:
            snap=d[d.hour==h].set_index(['farm','day']).join(day[['bias']],rsuffix='_day')
            cs=['in_temp_h0','in_hum_h0','in_co2_h0','in_temp_mean','act_heating_mean','act_thermal_mean','act_vent_mean','in_temp_seam','in_co2_seam','day']
            corr={c:float(snap[c].corr(snap.bias)) for c in cs if c in snap}
            diag['daily_correlations'][str(h)]=corr
        seam=d.in_temp_seam.to_numpy()
        for name,m in [('early_jump3',(d.hour.to_numpy()<4)&(np.abs(seam)>=3)),('early_smalljump',(d.hour.to_numpy()<4)&(np.abs(seam)<3))]:
            diag[name]={'n':int(m.sum()),'rmse':rmse(y[m],b[m]),'bias':float(e[m].mean()),'seam_error_corr':float(pd.Series(seam[m]).corr(pd.Series(e[m])))}
        for name,p in pred.items():
            diag['candidates'][name]=compare(d,y,b,p)
            diag['candidates'][name+'_blend20']=compare(d,y,b,.8*b+.2*p)
        if target=='sub_ec':
            high=y>1.5
            diag['high']={'rows':int(high.sum()),'days':int(d.loc[high,['farm','day']].drop_duplicates().shape[0]),'sse_share':float(np.sum(e[high]**2)/np.sum(e**2)),'bias':float(e[high].mean()),'rmse':rmse(y[high],b[high])}
            from sklearn.metrics import roc_auc_score,average_precision_score
            for name,p in [('baseline',b)]+list(pred.items()):
                diag.setdefault('high_detection',{})[name]={'auc':float(roc_auc_score(high,p)),'ap':float(average_precision_score(high,p)),'high_rmse':rmse(y[high],p[high]),'low_rmse':rmse(y[~high],p[~high])}
        d[['row_id','farm','day','hour',target,'baseline','error']].to_json(HERE/(target+'_diagnostic.json'),orient='records',force_ascii=False)
        np.savez_compressed(HERE/(target+'_independent_oof.npz'),row_id=d.row_id.values,**pred)
        result['targets'][target]=diag
        print(target,'RESULT',json.dumps(diag['candidates']),flush=True)
    result['input_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(common.DATA)/'train_X.csv',Path(common.DATA)/'train_y.csv',Path(common.DATA)/'test_X.csv',ROOT/'research/local/eval_v6_oof.npz',ROOT/'research/local/oof_ec_diag.npz']}
    with open(HERE/'results.json','w',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=True)
    print('DONE',flush=True)

if __name__=='__main__': main()

