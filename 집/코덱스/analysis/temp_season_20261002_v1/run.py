from pathlib import Path
import sys, json, hashlib, time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(HERE/'reference'))
import numpy as np
import pandas as pd
import common, harness
import temp_mask_v1 as TM
import train_flags_v6 as TF
from resid_reset_features import build_features, FEATURE_COLUMNS, PHYSICS_COLUMNS
from anal_q1_errors import diag_folds
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
import importlib.util
sp=importlib.util.spec_from_file_location('season_fixed',ROOT/'집/코덱스/analysis/ec_submission10_season_20261002_v1/season.py')
season=importlib.util.module_from_spec(sp);sp.loader.exec_module(season)
OUT=ROOT/'집/코덱스/local/temp_season_20261003_v1'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))
def main():
    start=time.time()
    common.load_raw=TM.masked_loader
    try:lab,ct,phc=TM.build_world()
    finally:common.load_raw=TM.ORIG;harness._CACHE.clear()
    tx,ty,sx=TM.masked_loader()
    cf=build_features(tx,sx).set_index('row_id')
    # Match the original temp_mask cache construction exactly.
    for c in FEATURE_COLUMNS:
        if c not in lab:lab[c]=cf.loc[lab.row_id,c].values
    assert lab.row_id.is_unique and len(lab)==9600
    assert np.isfinite(lab.sub_temp).all()
    raw,_,_=TM.ORIG()
    vec=season.vectors(raw[raw.farm.isin(['F13','F47'])])
    w=TF.row_weights(lab,.2,w_noisy=.2)
    zpath=Path(env.LOCAL)/'temp_mask_v1_oof.npz'
    z=np.load(zpath,allow_pickle=True)
    assert np.array_equal(lab.row_id.to_numpy(),z['row_id'])
    dmin=lab.groupby(['farm','day']).ph_in_temp_3.min()
    sets=[('DIAG10',diag_folds(lab))]
    for threshold in (10,12):
        ix=dmin[dmin<threshold].index
        sets.append((f'EXT{threshold}',[{f:set(int(d) for ff,d in ix if ff==f) for f in common.TARGET_FARMS}]))
    metadata={'baseline_cache_sha256':sha(zpath),'rows':len(lab),'cols':list(FEATURE_COLUMNS),'candidate_cols':[('season' if c=='day' else c) for c in FEATURE_COLUMNS], 'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [HERE/'run.py',HERE/'PROTOCOL.md',Path(sp.origin)]},'folds':[]}
    rows=[];maxdiff=0.
    for name,folds in sets:
        candidates={s:np.full(len(lab),np.nan) for s in (726,727)}
        refs={s:np.full(len(lab),np.nan) for s in (726,727)}
        folds_id=np.full(len(lab),-1)
        for k,fd in enumerate(folds):
            tm,vm=common.split_mask(lab,fd)
            tr,va=lab[tm].copy(),lab[vm].copy()
            tdays=tr[['farm','day']].drop_duplicates();qdays=va[['farm','day']].drop_duplicates().reset_index(drop=True)
            assert not set(map(tuple,tdays.to_numpy()))&set(map(tuple,qdays.to_numpy()))
            vtrain={key:vec[key] for key in map(tuple,tdays.to_numpy())}
            sm,query,notes=season.mapping(tdays,qdays,vtrain)
            # Query weather does not exist in the mapping input; reorder query rows to verify independent interpolation.
            _,reverse,_=season.mapping(tdays,qdays.iloc[::-1],vtrain)
            assert np.array_equal(query,reverse[::-1])
            tr['season']=[sm[(f,int(d))] for f,d in zip(tr.farm,tr.day)]
            qmap={(f,int(d)):v for (f,d),v in zip(qdays.itertuples(index=False,name=None),query)}
            va['season']=[qmap[(f,int(d))] for f,d in zip(va.farm,va.day)]
            cols=[('season' if c=='day' else c) for c in FEATURE_COLUMNS]
            cp=OUT/f'{name}_{k}.npz'
            if cp.exists():
                cache=np.load(cp)
                assert np.array_equal(cache['row_id'],va.row_id.to_numpy())
            else:
                payload={'row_id':va.row_id.to_numpy(),'season':va.season.to_numpy()}
                lin=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.))
                lin.fit(tr[PHYSICS_COLUMNS],tr.sub_temp.to_numpy(),ridge__sample_weight=w[tm])
                residual=tr.sub_temp.to_numpy()-lin.predict(tr[PHYSICS_COLUMNS]);physics=lin.predict(va[PHYSICS_COLUMNS])
                for seed in (726,727):
                    for tag,fs in [('base',FEATURE_COLUMNS),('season',cols)]:
                        m=LGBMRegressor(n_estimators=220,learning_rate=.035,num_leaves=12,max_depth=-1,min_child_samples=100,reg_lambda=15,verbosity=-1,n_jobs=4,random_state=seed)
                        m.fit(tr[fs],residual,sample_weight=w[tm]);payload[f'{tag}_{seed}']=physics+m.predict(va[fs])
                np.savez(cp,**payload);cache=payload
            for seed in (726,727):
                refs[seed][vm]=cache[f'base_{seed}'];candidates[seed][vm]=cache[f'season_{seed}']
                diff=float(np.max(np.abs(refs[seed][vm]-z[f'{name}__CODEX__{seed}'][vm])))
                maxdiff=max(maxdiff,diff)
                assert diff<1e-8,f'baseline mismatch {name}/{k}/{seed}: {diff}'
            folds_id[vm]=k
            metadata['folds'].append({'validator':name,'fold':k,'ntrain':len(tr),'nval':len(va),'season_fit':notes})
            print(f'{name}/{k} baseline/cache PASS maxdiff {maxdiff:.3g}, elapsed {time.time()-start:.0f}s',flush=True)
        pfpaths=[Path(env.LOCAL)/f'web_tabpfn_v2_temp_{name}.npy',Path(env.LOCAL)/f'web_tabpfn_v6_temp_{name}.npy']
        pfs=[np.load(pfpaths[0]).mean(0),np.load(pfpaths[1])[:8].mean(0)]
        metadata.setdefault('pfn_sha256',{}).update({str(p):sha(p) for p in pfpaths})
        g=np.where(np.isnan(lab.in_temp),1.,np.clip((lab.in_temp.to_numpy()-8)/2,0,1))
        for bs,cs in [(7,726),(101,727)]:
            base=z[f'{name}__MASK__{bs}']
            for group,pfn in zip(['1-8','17-24'],pfs):
                a=(.4+.1*g)*base+(.6-.4*g)*refs[cs]+.3*g*pfn
                b=(.4+.1*g)*base+(.6-.4*g)*candidates[cs]+.3*g*pfn
                ok=np.isfinite(a)&np.isfinite(b)
                assert np.array_equal(ok,folds_id>=0)
                df=lab.loc[ok,['row_id','farm','day','hour','in_temp','sub_temp']].copy()
                df['validator']=name;df['seed']=cs;df['context']=group;df['fold']=folds_id[ok];df['base']=a[ok];df['candidate']=b[ok]
                rows.append(df)
    oof=pd.concat(rows,ignore_index=True);oof.to_csv(OUT/'oof.csv',index=False)
    summary=[];rng=np.random.default_rng(20261003)
    for (name,seed,context),df in oof.groupby(['validator','seed','context'],sort=True):
        a=rmse(df.base.to_numpy(),df.sub_temp.to_numpy());b=rmse(df.candidate.to_numpy(),df.sub_temp.to_numpy())
        item={'validator':name,'seed':int(seed),'context':context,'baseline_rmse':a,'candidate_rmse':b,'delta_pct':100*(b/a-1),'fold_delta_std':float(np.std([rmse(x.candidate.to_numpy(),x.sub_temp.to_numpy())-rmse(x.base.to_numpy(),x.sub_temp.to_numpy()) for _,x in df.groupby('fold')]))}
        if name=='DIAG10':
            df=df.copy();df['block']=df.farm+'_'+(df.day//5).astype(str);df['dmse']=(df.candidate-df.sub_temp)**2-(df.base-df.sub_temp)**2
            blocks=df.groupby('block').dmse.agg(['sum','count']);ix=rng.integers(0,len(blocks),size=(20000,len(blocks)))
            samples=blocks['sum'].to_numpy()[ix].sum(1)/blocks['count'].to_numpy()[ix].sum(1)
            item.update(p_worse=float(np.mean(samples>=0)),ci95=np.quantile(samples,[.025,.975]).tolist())
        summary.append(item);print(json.dumps(item),flush=True)
    adopted=all(s['delta_pct']<0 for s in summary) and all(s['p_worse']<.025 and s['ci95'][1]<0 for s in summary if s['validator']=='DIAG10')
    metadata.update(baseline_cache_maxdiff=maxdiff,verdict='PASS' if adopted else 'REJECT',elapsed_seconds=time.time()-start,summary=summary)
    (OUT/'result.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('T-S1',metadata['verdict'],flush=True)
if __name__=='__main__':main()
