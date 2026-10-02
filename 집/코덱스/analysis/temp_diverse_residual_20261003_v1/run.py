from pathlib import Path
import sys,json,time,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/temp_season_20261002_v1/reference'))
import numpy as np,pandas as pd
import common,harness,temp_mask_v1 as TM,train_flags_v6 as TF
from anal_q1_errors import diag_folds
from resid_reset_features import build_features,FEATURE_COLUMNS,PHYSICS_COLUMNS
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
OUT=ROOT/'집/코덱스/local/temp_diverse_residual_20261003_v1';OUT.mkdir(parents=True,exist_ok=True)
def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    start=time.time();common.load_raw=TM.masked_loader
    try:lab,ct,phc=TM.build_world()
    finally:common.load_raw=TM.ORIG;harness._CACHE.clear()
    tx,_,sx=TM.masked_loader();cf=build_features(tx,sx).set_index('row_id')
    for c in FEATURE_COLUMNS:
        if c not in lab:lab[c]=cf.loc[lab.row_id,c].to_numpy()
    assert len(lab)==9600 and lab.row_id.is_unique
    w=TF.row_weights(lab,.2,w_noisy=.2);zpath=Path(env.LOCAL)/'temp_mask_v1_oof.npz';z=dict(np.load(zpath,allow_pickle=True))
    assert np.array_equal(lab.row_id.to_numpy(),z['row_id'])
    dmin=lab.groupby(['farm','day']).ph_in_temp_3.min();sets=[('DIAG10',diag_folds(lab))]
    for t in (10,12):
        ix=dmin[dmin<t].index;sets.append((f'EXT{t}',[{f:set(int(d) for ff,d in ix if ff==f) for f in common.TARGET_FARMS}]))
    rows=[];stats=[];maxdiff=0.
    for name,fds in sets:
        predictions={tag:{s:np.full(len(lab),np.nan) for s in (726,727)} for tag in ['CB','ET']};fid=np.full(len(lab),-1)
        for k,fd in enumerate(fds):
            tm,vm=common.split_mask(lab,fd);tr,va=lab[tm].copy(),lab[vm].copy();cp=OUT/f'{name}_{k}.npz'
            if cp.exists():payload=dict(np.load(cp,allow_pickle=True));assert np.array_equal(payload['row_id'],va.row_id.to_numpy())
            else:
                payload={'row_id':va.row_id.to_numpy(dtype=str)}
                lin=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.))
                lin.fit(tr[PHYSICS_COLUMNS],tr.sub_temp.to_numpy(),ridge__sample_weight=w[tm]);btr,bva=lin.predict(tr[PHYSICS_COLUMNS]),lin.predict(va[PHYSICS_COLUMNS]);resid=tr.sub_temp.to_numpy()-btr
                imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(tr[FEATURE_COLUMNS]);xtr,xva=imp.transform(tr[FEATURE_COLUMNS]),imp.transform(va[FEATURE_COLUMNS])
                for seed in (726,727):
                    baseline=TM.codex_fit_predict(tr,va,w[tm],seed);diff=float(np.max(np.abs(baseline-z[f'{name}__CODEX__{seed}'][vm])));assert diff<1e-8;maxdiff=max(maxdiff,diff)
                    payload[f'baseline_{seed}']=baseline
                    cb=CatBoostRegressor(iterations=600,depth=5,learning_rate=.04,l2_leaf_reg=10,random_strength=1,bootstrap_type='Bayesian',bagging_temperature=1,loss_function='RMSE',random_seed=seed,thread_count=4,task_type='CPU',verbose=False,allow_writing_files=False)
                    et=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=20,max_features=.8,n_jobs=4,random_state=seed)
                    for tag,model in [('CB',cb),('ET',et)]:
                        model.fit(xtr,resid,sample_weight=w[tm]);pred=bva+model.predict(xva);payload[f'{tag}_{seed}']=pred
                        payload[f'{tag}_train_{seed}']=rmse(btr+model.predict(xtr),tr.sub_temp.to_numpy());payload[f'{tag}_val_{seed}']=rmse(pred,va.sub_temp.to_numpy())
                        print(f'{name}/{k}/{seed}/{tag} member valRMSE {float(payload[f"{tag}_val_{seed}"]):.6f}, elapsed {time.time()-start:.0f}s',flush=True)
                np.savez(cp,**payload)
            for seed in (726,727):
                diff=float(np.max(np.abs(payload[f'baseline_{seed}']-z[f'{name}__CODEX__{seed}'][vm])));assert diff<1e-8;maxdiff=max(maxdiff,diff)
                for tag in predictions:
                    predictions[tag][seed][vm]=payload[f'{tag}_{seed}'];stats.append(dict(validator=name,fold=k,seed=seed,member=tag,ntrain=len(tr),nval=len(va),train_rmse=float(payload[f'{tag}_train_{seed}']),val_rmse=float(payload[f'{tag}_val_{seed}'])))
            fid[vm]=k
        pfs=[np.load(Path(env.LOCAL)/f'web_tabpfn_v2_temp_{name}.npy').mean(0),np.load(Path(env.LOCAL)/f'web_tabpfn_v6_temp_{name}.npy')[:8].mean(0)]
        g=np.where(np.isnan(lab.in_temp),1.,np.clip((lab.in_temp.to_numpy()-8)/2,0,1))
        for bs,cs in [(7,726),(101,727)]:
            base,cx=z[f'{name}__MASK__{bs}'],z[f'{name}__CODEX__{cs}']
            for context,pfn in zip(['1-8','17-24'],pfs):
                ref=(.4+.1*g)*base+(.6-.4*g)*cx+.3*g*pfn
                for tag,pred in predictions.items():
                    cand=ref+.1*g*(pred[cs]-base);ok=np.isfinite(ref)&np.isfinite(cand);assert np.array_equal(ok,fid>=0)
                    assert np.array_equal(cand[ok&(g==0)],ref[ok&(g==0)])
                    df=lab.loc[ok,['row_id','farm','day','hour','in_temp','sub_temp']].copy();df['validator']=name;df['fold']=fid[ok];df['seed']=cs;df['context']=context;df['member']=tag;df['base']=ref[ok];df['candidate']=cand[ok];df['new_member']=pred[cs][ok];rows.append(df)
    oof=pd.concat(rows,ignore_index=True);oof.to_csv(OUT/'oof.csv',index=False);summary=[];rng=np.random.default_rng(20261003)
    for (tag,name,seed,context),df in oof.groupby(['member','validator','seed','context'],sort=True):
        a=rmse(df.base.to_numpy(),df.sub_temp.to_numpy());b=rmse(df.candidate.to_numpy(),df.sub_temp.to_numpy());folds=[rmse(x.candidate.to_numpy(),x.sub_temp.to_numpy())-rmse(x.base.to_numpy(),x.sub_temp.to_numpy()) for _,x in df.groupby('fold')]
        item=dict(member=tag,validator=name,seed=int(seed),context=context,baseline_rmse=a,candidate_rmse=b,delta_pct=100*(b/a-1),fold_delta_std=float(np.std(folds)),folds_better=int(np.sum(np.array(folds)<0)),folds_total=len(folds))
        if name=='DIAG10':
            df=df.copy();df['block']=df.farm+'_'+(df.day//5).astype(str);df['dmse']=(df.candidate-df.sub_temp)**2-(df.base-df.sub_temp)**2
            blocks=df.groupby('block').dmse.agg(['sum','count']);ix=rng.integers(0,len(blocks),size=(20000,len(blocks)));samples=blocks['sum'].to_numpy()[ix].sum(1)/blocks['count'].to_numpy()[ix].sum(1)
            item.update(p_worse=float(np.mean(samples>=0)),ci=np.quantile(samples,[.00625,.99375]).tolist())
        summary.append(item);print(json.dumps(item),flush=True)
    verdicts={tag:('PASS' if all(s['delta_pct']<0 for s in summary if s['member']==tag) and all(s['p_worse']<.00625 and s['ci'][1]<0 for s in summary if s['member']==tag and s['validator']=='DIAG10') else 'REJECT') for tag in ['CB','ET']}
    output=dict(verdicts=verdicts,summary=summary,member_training_scores=stats,cache_maxdiff=maxdiff,elapsed=time.time()-start,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'run.py',HERE/'PROTOCOL.md',zpath]})
    (OUT/'result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(verdicts),flush=True)
if __name__=='__main__':main()
