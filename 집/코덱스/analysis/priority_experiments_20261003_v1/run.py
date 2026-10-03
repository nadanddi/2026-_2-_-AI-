"""Precommitted priority queue. No test prediction or reserved-label reads."""
from pathlib import Path
import os, sys, json, hashlib, time
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0, str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'))
import support as S
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits

OUT=ROOT/'집/코덱스/local/priority_experiments_20261003_v1'
OUT.mkdir(parents=True,exist_ok=True)
CUTS=[0,3,6,12,18,23]
SEEDS=[7,101,2024]
ARMS=['EC_DAY20','EC_STATE20','T_DIRECT20','T_GAP20','T_GAPSPLIT20','EC_RARE_ET','T_RARE_GAP20','EC_KNN20','T_KNN20','EC_H0CHANGE']
K=len(ARMS)
RAWT=S.common.USABLE
CHECKS=[]
def check(name, condition, **kwargs):
    if not condition: raise AssertionError((name,kwargs))
    CHECKS.append(dict(name=name,passed=True,**kwargs))
def jsonout(p,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def reg(seed,trees=160,leaf=3):
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),ExtraTreesRegressor(n_estimators=trees,min_samples_leaf=leaf,n_jobs=4,random_state=seed))
def classifier(seed,leaf=3):
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),ExtraTreesClassifier(n_estimators=160,min_samples_leaf=leaf,n_jobs=4,random_state=seed))
def probability(m,x):
    classes=m.steps[-1][1].classes_
    return m.predict_proba(x)[:,list(classes).index(1)] if 1 in classes else np.zeros(len(x))
def prefix_tables(lab,cols):
    ans={}
    for c in CUTS:
        d=lab.loc[lab.hour<=c].sort_values(['farm','day','hour'])
        g=d.groupby(['farm','day'],sort=True)
        mean=g[cols].mean().add_suffix('_mean')
        std=g[cols].std().fillna(0).add_suffix('_std')
        first=d.loc[d.hour==0].set_index(['farm','day'])[cols]
        last=d.loc[d.hour==c].set_index(['farm','day'])[cols]
        f=pd.concat([mean,std,first.add_suffix('_h0'),last.add_suffix('_current'),(last-first).add_suffix('_delta')],axis=1)
        f['farm_id']=(f.index.get_level_values('farm')=='F47').astype(float)
        ans[c]=f
    return ans
def xday(frame, table):
    days=frame[['farm','day']].drop_duplicates().set_index(['farm','day'])
    x=table.reindex(days.index).copy()
    if 'season' in frame:
        x['season']=frame.groupby(['farm','day']).season.first().reindex(x.index)
    check('one_row_per_day',len(x)==len(days) and x.index.is_unique)
    return x
def level_prediction(tr,va,tables,yday,seed,kind):
    predictions=np.zeros(len(va)); probabilities=np.full(len(va),np.nan)
    bins=np.searchsorted(CUTS,va.hour,side='right')-1
    for j,c in enumerate(CUTS):
        mask=bins==j
        if not mask.any():continue
        x=xday(tr,tables[c]); ids=pd.MultiIndex.from_frame(va.loc[mask,['farm','day']]); q=tables[c].reindex(ids).copy()
        if 'season' in tr:q['season']=va.loc[mask,'season'].to_numpy()
        y=yday.reindex(x.index).to_numpy()
        check('no_query_day_neighbors',not bool(set(x.index)&set(ids)))
        if kind=='knn':
            model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),KNeighborsRegressor(n_neighbors=min(7,len(x)),weights='distance',n_jobs=1))
            model.fit(x,y);p=model.predict(q)
        elif kind=='state':
            labels=(y>=1).astype(int)
            cl=classifier(seed);cl.fit(x,labels);pr=probability(cl,q);parts=[]
            for val in (0,1):
                use=labels==val
                if use.any():
                    m=reg(seed);m.fit(x.loc[use],y[use]);parts.append(m.predict(q))
                else:parts.append(np.full(len(q),np.mean(y)))
            p=(1-pr)*parts[0]+pr*parts[1];probabilities[mask]=pr
        else:
            model=reg(seed);model.fit(x,y);p=model.predict(q)
        predictions[mask]=p
    return predictions,probabilities
def record(va,target,arm,name,k,seed,context,ref,pred,**extra):
    a=va[['row_id','farm','day','hour']].copy()
    a['y']=va['sub_ec' if target=='EC' else 'sub_temp'].to_numpy()
    for key,val in dict(target=target,arm=arm,validator=name,fold=k,seed=seed,context=context,baseline=ref,candidate=pred,**extra).items():a[key]=val
    check('finite_prediction',np.isfinite(a[['y','baseline','candidate']].to_numpy()).all())
    return a
def temp_refs(va,outer,name,seed,context):
    d=outer[(outer.validator==name)&(outer.base_seed==seed)&(outer.context==context)]
    p=np.column_stack([d[d.member==m].set_index('row_id').prediction.reindex(va.row_id).to_numpy() for m in ['BASE','CODEX','PFN']])
    g=S.gate(va)
    ref=.4*p[:,0]+(.6-.4*g)*p[:,1]+.4*g*p[:,2]
    return ref
def audit_baseline():
    lab,_,_,_,folds,outer=S.loadtemp();rows=[];daily=[]
    for name,k,fd in folds:
        _,vm=S.common.split_mask(lab,fd);va=lab[vm].copy()
        for seed in (7,101):
            for context in ('1-8','17-24'):
                ref=temp_refs(va,outer,name,seed,context)
                old=outer[(outer.validator==name)&(outer.base_seed==seed)&(outer.context==context)&(outer.member=='W30G')].set_index('row_id').prediction.reindex(va.row_id).to_numpy()
                a=record(va,'TEMP','BASELINE_UPDATE',name,k,seed,context,old,ref);rows.append(a)
                a['air']=va.in_temp.to_numpy()
                b=a.groupby(['farm','day']).agg(y=('y','mean'),air=('air','mean'),old_sse=('baseline',lambda x:0))
                b['old_rmse']=a.assign(err=(old-a.y)**2).groupby(['farm','day']).err.mean()**.5
                b['new_rmse']=a.assign(err=(ref-a.y)**2).groupby(['farm','day']).err.mean()**.5
                b['air_complete']=a.groupby(['farm','day']).air.count().eq(24)
                b['gap']=b.y-b.air;b['validator']=name;b['fold']=k;b['seed']=seed;b['context']=context
                daily.append(b.reset_index())
    pd.concat(rows).to_csv(OUT/'baseline_update_oof.csv',index=False)
    pd.concat(daily).drop(columns='old_sse').to_csv(HERE/'W40G_daily_audit.csv',index=False)
    print('BASELINE_UPDATE_DONE',flush=True)
def ec_phase(phase):
    lab,core,wv,folds,outer=S.loadec()
    tables=prefix_tables(lab,core.RAW)
    et_cache=pd.read_csv(ROOT/'집/클로드/research/local/ec3_DI1_all.csv',float_precision='round_trip').set_index(['validator','validation_fold','row_id'])
    for name,k,tm,vm in folds:
        path=OUT/f'{phase}_{name}_{k}.csv'
        if path.exists():continue
        tr,va=S.seasonal(lab[tm],lab[vm],wv);tr=tr.reset_index(drop=True);va=va.reset_index(drop=True)
        check('ec_train_validation_ids',not bool(set(tr.row_id)&set(va.row_id)))
        days=tr.groupby(['farm','day']).sub_ec.mean();frames=[]
        for seed in SEEDS:
            b=outer[(outer.validator==name)&(outer.validation_fold==k)&(outer.seed==seed)].set_index('row_id').reindex(va.row_id)
            ref=b.season_v2.to_numpy();lo,hi=tr.sub_ec.min(),tr.sub_ec.max()
            if phase in ('EC_DAY20','EC_STATE20','EC_KNN20'):
                level,pr=level_prediction(tr,va,tables,days,seed,{'EC_DAY20':'reg','EC_STATE20':'state','EC_KNN20':'knn'}[phase])
                cm=pd.Series(ref).groupby([va.farm,va.day]).transform(lambda s:s.expanding().mean()).to_numpy()
                pred=np.clip(ref+.2*(level-cm),lo,hi)
                frames.append(record(va,'EC',phase,name,k,seed,'1-4',ref,pred,level_prediction=level,high_probability=pr))
            else:
                cols=['season' if c=='day' else c for c in core.FULL]
                old=et_cache.loc[[(name,k,r) for r in va.row_id],f'etS_{seed}'].to_numpy()
                model=core.et(seed)
                if phase=='EC_RARE_ET':
                    w=np.array([4. if days.loc[(f,d)]>=1 else 1. for f,d in zip(tr.farm,tr.day)])
                    p=core.predict_model(model,tr,va,cols,w)
                else:
                    tr=tr.copy();va=va.copy()
                    for c in core.RAW:
                        tr[c+'_change0']=tr[c]-tr[c+'_h0'];va[c+'_change0']=va[c]-va[c+'_h0']
                    cols=[c.replace('_h0','_change0') if c.endswith('_h0') else c for c in cols]
                    p=core.predict_model(model,tr,va,cols)
                pred=np.clip(ref+core.shrink(.48*(p-old),va),lo,hi)
                frames.append(record(va,'EC',phase,name,k,seed,'1-4',ref,pred,new_et=p,old_et=old))
        pd.concat(frames).to_csv(path,index=False)
        print(f'{phase} {name}/{k} done train={len(tr)//24} query={len(va)//24}',flush=True)
def lgb(seed):
    return LGBMRegressor(n_estimators=220,learning_rate=.035,num_leaves=12,min_child_samples=100,reg_lambda=15,verbosity=-1,n_jobs=2,random_state=seed,deterministic=True,force_col_wise=True)
def temp_phase(phase):
    lab,_,_,weights,folds,outer=S.loadtemp();tables=prefix_tables(lab,RAWT)
    for name,k,fd in folds:
        path=OUT/f'{phase}_{name}_{k}.csv'
        if path.exists():continue
        tm,vm=S.common.split_mask(lab,fd);tr=lab[tm].reset_index(drop=True).copy();va=lab[vm].reset_index(drop=True).copy();w=weights[tm]
        med=float(tr.in_temp.median());air=tr.in_temp.fillna(med).to_numpy();qair=va.in_temp.fillna(med).to_numpy()
        gap=tr.sub_temp.to_numpy()-air;tg=tr.assign(gap=gap).groupby(['farm','day']).gap.mean()
        labels=np.array([int(tg.loc[(f,d)]>=0) for f,d in zip(tr.farm,tr.day)]);rare=np.array([4. if abs(tg.loc[(f,d)])>=2 else 1. for f,d in zip(tr.farm,tr.day)])
        frames=[]
        for seed in (7,101):
            if phase=='T_KNN20':
                p,_=level_prediction(tr,va,tables,tg,seed,'knn');p=qair+p
            elif phase=='T_GAPSPLIT20':
                cl=classifier(seed,leaf=10);cl.fit(tr[S.FEATURE_COLUMNS],labels,extratreesclassifier__sample_weight=np.ones(len(tr))/24)
                prob=probability(cl,va[S.FEATURE_COLUMNS]);parts=[]
                for state in (0,1):
                    use=labels==state
                    if use.sum()>=24:
                        m=lgb(seed);m.fit(tr.loc[use,S.FEATURE_COLUMNS],gap[use],sample_weight=w[use]);parts.append(m.predict(va[S.FEATURE_COLUMNS]))
                    else:parts.append(np.full(len(va),np.average(gap,weights=w)))
                p=qair+(1-prob)*parts[0]+prob*parts[1]
            else:
                m=lgb(seed);y=tr.sub_temp if phase=='T_DIRECT20' else gap
                m.fit(tr[S.FEATURE_COLUMNS],y,sample_weight=w*(rare if phase=='T_RARE_GAP20' else 1))
                p=m.predict(va[S.FEATURE_COLUMNS])+(0 if phase=='T_DIRECT20' else qair)
            for context in ('1-8','17-24'):
                ref=temp_refs(va,outer,name,seed,context)
                frames.append(record(va,'TEMP',phase,name,k,seed,context,ref,.8*ref+.2*p,alternative=p))
        pd.concat(frames).to_csv(path,index=False)
        print(f'{phase} {name}/{k} done train={len(tr)//24} query={len(va)//24}',flush=True)
def risk_phase():
    for target in ('EC','TEMP'):
        if target=='EC':
            lab,core,wv,folds,outer=S.loadec();cols=core.RAW
        else:
            lab,_,_,_,f0,outer=S.loadtemp();folds=[(name,k,*S.common.split_mask(lab,fd)) for name,k,fd in f0];cols=RAWT
        tables=prefix_tables(lab,cols)
        for name,k,tm,vm in folds:
            path=OUT/f'RISK_{target}_{name}_{k}.csv'
            if path.exists():continue
            va=lab[vm].reset_index(drop=True).copy();prefix=('E' if target=='EC' else 'T')+f'_{name}_{k}'
            cache=S.OUT;z=dict(np.load(cache/f'{prefix}_cpu.npz'))
            b=lab.set_index('row_id').reindex(z['row_id']).reset_index()
            check('risk_nested_ids',set(z['inner_train_id']).isdisjoint(set(z['row_id'])) and set(z['outer_train_id']).isdisjoint(set(va.row_id)))
            if target=='EC':
                a=lab.set_index('row_id').reindex(z['inner_train_id']).reset_index();_,b=S.seasonal(a,b,wv);_,va=S.seasonal(lab[tm],va,wv)
                pfn=S.OUT/f'{prefix}_pfn_1.npz'
                bag=np.mean([np.load(S.OUT/f'{prefix}_pfn_{s}.npz')['prediction'] for s in range(1,5)],axis=0)
                base=np.clip(core.shrink(.8*z['r3_7']+.2*bag,b),z['lo'],z['hi']);qref=outer[(outer.validator==name)&(outer.validation_fold==k)&(outer.seed==7)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy();truth='sub_ec';th=.1
            else:
                bag=np.mean([np.load(S.OUT/f'{prefix}_pfn_{s}.npz')['prediction'] for s in range(1,9)],axis=0)
                g=S.gate(b);base=.4*z['base_7']+(.6-.4*g)*z['codex_7']+.4*g*bag
                qref=temp_refs(va,outer,name,7,'1-8');truth='sub_temp';th=.5
            b=b.reset_index(drop=True);va=va.reset_index(drop=True)
            daily=b.assign(e=(base-b[truth].to_numpy())**2).groupby(['farm','day']).e.mean()**.5
            y=(daily>th).astype(int);risk=np.zeros(len(va));bins=np.searchsorted(CUTS,va.hour,side='right')-1
            for j,c in enumerate(CUTS):
                use=bins==j
                if not use.any():continue
                x=xday(b,tables[c]);idx=pd.MultiIndex.from_frame(va.loc[use,['farm','day']]);q=tables[c].reindex(idx).copy()
                if 'season' in b:q['season']=va.loc[use,'season'].to_numpy()
                # Prefix predicted mean comes from nested out-of-fold experts.
                x['pred_mean']=b.loc[b.hour<=c].assign(p=base[b.hour<=c]).groupby(['farm','day']).p.mean().reindex(x.index)
                q['pred_mean']=va.loc[va.hour<=c].assign(p=qref[va.hour<=c]).groupby(['farm','day']).p.mean().reindex(idx).to_numpy()
                cl=classifier(7,leaf=5);cl.fit(x,y.reindex(x.index));risk[use]=probability(cl,q)
            a=record(va,target,'RISK_DIAGNOSTIC',name,k,7,'1-4' if target=='EC' else '1-8',qref,qref,risk=risk)
            a.to_csv(path,index=False);print(f'RISK {target} {name}/{k} done inner_days={len(y)}',flush=True)
def main():
    phase=sys.argv[1]
    with threadpool_limits(limits=2):
        if phase=='baseline':audit_baseline()
        elif phase=='risk':risk_phase()
        elif phase.startswith('EC_'):ec_phase(phase)
        else:temp_phase(phase)
    jsonout(OUT/f'{phase}_audit.json',dict(status='PASS',checks=CHECKS,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),phase=phase))
    print(f'{phase}_DONE',flush=True)
if __name__=='__main__':main()
