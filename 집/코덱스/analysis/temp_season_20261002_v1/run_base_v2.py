import run as R
from sklearn.linear_model import LinearRegression
import cold_v5
from screen_v6 import temp_members
from lightgbm import LGBMRegressor
OUT=R.ROOT/'집/코덱스/local/temp_season_base_20261003_v1'
OUT.mkdir(parents=True,exist_ok=True)
np,pd=R.np,R.pd
def main():
    start=R.time.time();R.common.load_raw=R.TM.masked_loader
    try:lab,ct,phc=R.TM.build_world()
    finally:R.common.load_raw=R.TM.ORIG;R.harness._CACHE.clear()
    assert 'day' in ct
    raw,_,_=R.TM.ORIG();vec=R.season.vectors(raw[raw.farm.isin(['F13','F47'])])
    w=R.TF.row_weights(lab,.2,w_noisy=.2)
    z=np.load(R.Path(R.env.LOCAL)/'temp_mask_v1_oof.npz',allow_pickle=True)
    assert np.array_equal(lab.row_id.to_numpy(),z['row_id'])
    dmin=lab.groupby(['farm','day']).ph_in_temp_3.min();sets=[('DIAG10',R.diag_folds(lab))]
    for t in (10,12):
        ix=dmin[dmin<t].index;sets.append((f'EXT{t}',[{f:set(int(d) for ff,d in ix if ff==f) for f in R.common.TARGET_FARMS}]))
    rows=[];metadata={'source_hashes':{str(p.relative_to(R.ROOT)):R.sha(p) for p in [R.HERE/'run_base_v2.py',R.HERE/'PROTOCOL_T_S2.md']},'candidate_cols':[('season' if c=='day' else c) for c in ct],'folds':[]};maxdiff=0.
    for name,fds in sets:
        refs={s:np.full(len(lab),np.nan) for s in (726,727)};cands={s:np.full(len(lab),np.nan) for s in (726,727)};fid=np.full(len(lab),-1)
        for k,fd in enumerate(fds):
            tm,vm=R.common.split_mask(lab,fd);tr,va=lab[tm].copy(),lab[vm].copy()
            td=tr[['farm','day']].drop_duplicates();qd=va[['farm','day']].drop_duplicates().reset_index(drop=True)
            sm,sq,notes=R.season.mapping(td,qd,{key:vec[key] for key in map(tuple,td.to_numpy())})
            tr['season']=[sm[(f,int(d))] for f,d in zip(tr.farm,tr.day)]
            qm={(f,int(d)):q for (f,d),q in zip(qd.itertuples(index=False,name=None),sq)};va['season']=[qm[(f,int(d))] for f,d in zip(va.farm,va.day)]
            cscols=[('season' if c=='day' else c) for c in ct];payload={'row_id':va.row_id.to_numpy(dtype=str),'season':va.season.to_numpy()}
            imp=R.SimpleImputer(strategy='median').fit(tr[phc]);lin=LinearRegression().fit(imp.transform(tr[phc]),tr.sub_temp.to_numpy(),sample_weight=w[tm])
            btr,bva=lin.predict(imp.transform(tr[phc])),lin.predict(imp.transform(va[phc]));resid=tr.sub_temp.to_numpy()-btr
            cp=OUT/f'{name}_{k}.npz'
            if cp.exists():payload=np.load(cp,allow_pickle=True)
            else:
                for bs,cs in [(7,726),(101,727)]:
                    cold_v5.SEED=bs
                    members=temp_members(tr,va,ct,phc,w[tm]);old=.65*members['res']+.25*members['ridge']+.1*members['nys']
                    m=cold_v5.lgbh();m.fit(tr[cscols],resid,sample_weight=w[tm]);newres=bva+m.predict(va[cscols])
                    new=old+.65*(newres-members['res'])
                    payload[f'base_{cs}']=old;payload[f'season_{cs}']=new
                    payload[f'train_rmse_{cs}']=R.rmse(btr+m.predict(tr[cscols]),tr.sub_temp.to_numpy())
                    payload[f'val_rmse_{cs}']=R.rmse(newres,va.sub_temp.to_numpy())
                np.savez(cp,**payload)
            for bs,cs in [(7,726),(101,727)]:
                refs[cs][vm]=payload[f'base_{cs}'];cands[cs][vm]=payload[f'season_{cs}']
                diff=float(np.max(np.abs(refs[cs][vm]-z[f'{name}__MASK__{bs}'][vm])));maxdiff=max(maxdiff,diff)
                assert diff<1e-8,f'BASE mismatch {name}/{k}/{bs}: {diff}'
            fid[vm]=k;metadata['folds'].append(dict(validator=name,fold=k,season_fit=notes))
            print(f'{name}/{k} BASE/cache PASS {maxdiff:.3g}, elapsed {R.time.time()-start:.0f}s',flush=True)
        pfs=[np.load(R.Path(R.env.LOCAL)/f'web_tabpfn_v2_temp_{name}.npy').mean(0),np.load(R.Path(R.env.LOCAL)/f'web_tabpfn_v6_temp_{name}.npy')[:8].mean(0)]
        g=np.where(np.isnan(lab.in_temp),1.,np.clip((lab.in_temp.to_numpy()-8)/2,0,1))
        for bs,cs in [(7,726),(101,727)]:
            cx=z[f'{name}__CODEX__{cs}']
            for context,pfn in zip(['1-8','17-24'],pfs):
                a=(.4+.1*g)*refs[cs]+(.6-.4*g)*cx+.3*g*pfn;b=(.4+.1*g)*cands[cs]+(.6-.4*g)*cx+.3*g*pfn
                ok=np.isfinite(a)&np.isfinite(b);assert np.array_equal(ok,fid>=0)
                df=lab.loc[ok,['row_id','farm','day','hour','in_temp','sub_temp']].copy()
                df['validator']=name;df['seed']=cs;df['context']=context;df['fold']=fid[ok];df['base']=a[ok];df['candidate']=b[ok];rows.append(df)
    oof=pd.concat(rows,ignore_index=True);oof.to_csv(OUT/'oof.csv',index=False);summary=[];rng=np.random.default_rng(20261003)
    for (name,seed,context),df in oof.groupby(['validator','seed','context'],sort=True):
        a=R.rmse(df.base.to_numpy(),df.sub_temp.to_numpy());b=R.rmse(df.candidate.to_numpy(),df.sub_temp.to_numpy())
        s=dict(validator=name,seed=int(seed),context=context,baseline_rmse=a,candidate_rmse=b,delta_pct=100*(b/a-1))
        if name=='DIAG10':
            df=df.copy();df['block']=df.farm+'_'+(df.day//5).astype(str);df['dmse']=(df.candidate-df.sub_temp)**2-(df.base-df.sub_temp)**2
            blocks=df.groupby('block').dmse.agg(['sum','count']);ix=rng.integers(0,len(blocks),size=(20000,len(blocks)));samples=blocks['sum'].to_numpy()[ix].sum(1)/blocks['count'].to_numpy()[ix].sum(1)
            s.update(p_worse=float(np.mean(samples>=0)),ci95=np.quantile(samples,[.0125,.9875]).tolist())
        summary.append(s);print(R.json.dumps(s),flush=True)
    passed=all(s['delta_pct']<0 for s in summary) and all(s['p_worse']<.0125 and s['ci95'][1]<0 for s in summary if s['validator']=='DIAG10')
    metadata.update(summary=summary,baseline_cache_maxdiff=maxdiff,verdict='PASS' if passed else 'REJECT',elapsed_seconds=R.time.time()-start)
    (OUT/'result.json').write_text(R.json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8');print('T-S2',metadata['verdict'],flush=True)
if __name__=='__main__':main()
