"""Final full-public-label training only after the registered 360-day CV is complete."""
from pathlib import Path
import sys,json,time,gc
sys.dont_write_bytecode=True
import run_v2 as P
import recipe_ec_v1 as C
from predict_model_v1 import predict_bundle
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits

def final_fit():
    reg=P.checkpins();s=json.loads((P.H/'final_score_v1.json').read_text(encoding='utf8'));assert s['status']=='FULL_DIAG10' and s['folds']==list(range(10))
    assert (P.H/'critique_midpoint_v1.md').exists()
    D=P.L/'model_full_clean_v1';assert not D.exists();D.mkdir()
    # Read the remaining official labels only here, after all fixed CV scores are saved.
    x=pd.read_csv(P.DATA/'train_X.csv');x=x[x.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True)
    y=pd.read_csv(P.DATA/'train_y.csv',usecols=['row_id','sub_ec'],float_precision='round_trip');y=y[y.row_id.isin(x.row_id)].reset_index(drop=True)
    assert len(x)==len(y)==9600 and x.row_id.is_unique and y.row_id.is_unique and set(x.row_id)==set(y.row_id)
    f=C.features(x[['row_id']+C.RAW]).merge(y,on='row_id',validate='one_to_one');sf,empty,notes=C.season(f,f.iloc[:0],x)
    removed,_,detail,snap=P.select(sf,empty)
    cx,cy,cl=P.dataset(x,y,removed,P.L/'dataset_full_train');P.save(P.L/'dataset_full_train/selection_detail_v1.json',detail);np.savez_compressed(P.L/'dataset_full_train/selector_snapshot_v1.npz',**snap)
    t=C.features(cx[['row_id']+C.RAW]).merge(cy,on='row_id',validate='one_to_one');t,_,notes=C.season(t,t.iloc[:0],cx)
    td=list(t[['farm','day']].drop_duplicates().itertuples(index=False,name=None));bounds=[float(t.sub_ec.min()),float(t.sub_ec.max())]
    # Reload/inference probes use only old public rows; do not rescore the consumed 40-day set.
    public_ids={rid for rec in reg['folds'] for rid in rec['query_ids']};pmeta=C.identify(cx[cx.row_id.isin(public_ids)])
    probe_days=set(pmeta[['farm','day']].drop_duplicates().groupby('farm').head(2).itertuples(index=False,name=None))
    probe_x=cx[[(farm,int(day)) in probe_days for farm,day in zip(C.identify(cx).farm,C.identify(cx).day)]].copy();assert len(probe_x)==96
    q=C.frozen_season(C.features(probe_x[['row_id']+C.RAW]),notes,td)
    assert np.max(abs(P.ordered(t,q.row_id).season.to_numpy()-q.season.to_numpy()))<1e-12
    models=[];raw_probe=np.zeros(len(q));receipts=[]
    for seed in C.SEEDS:
        for kind in C.WEIGHTS:
            start=time.monotonic();cols=C.feature_columns(kind);model=C.factory(kind,seed)
            with threadpool_limits(limits=2):
                model.fit(t[cols],t.sub_ec.to_numpy(float))
                if kind=='et':model.steps[-1][1].n_jobs=1
                pred=np.asarray(model.predict(q[cols]),float);assert np.isfinite(pred).all()
            raw_probe+=C.WEIGHTS[kind]*pred/len(C.SEEDS)
            fp=D/f'{kind}_{seed}_v1.joblib';joblib.dump(model,fp,compress=3);del model;gc.collect();loaded=joblib.load(fp)
            with threadpool_limits(limits=2):again=np.asarray(loaded.predict(q[cols]),float)
            gap=float(np.max(abs(pred-again)));assert gap<1e-12;del loaded;gc.collect()
            models.append(dict(kind=kind,seed=seed,file=fp.name,sha=P.sha(fp),columns=cols));receipts.append(dict(kind=kind,seed=seed,train_rows=len(t),train_feature_hash=P.fhash(t,['row_id','sub_ec']+cols),reload_maxdiff=gap,seconds=time.monotonic()-start));print('FINAL_FIT',kind,seed,'days',len(t)//24,'reloadgap',gap,flush=True)
    manifest=dict(status='TRAINED_EXPERIMENTAL_MODEL',training_rows=len(t),training_days=sorted((str(ff),int(dd)) for ff,dd in td),removed_days=sorted((str(ff),int(dd)) for ff,dd in removed),bounds=bounds,models=models,weights=C.WEIGHTS,seeds=list(C.SEEDS),season_notes=notes,code_hashes={n:P.sha(P.H/n) for n in ['recipe_ec_v1.py','season_transform_v1.py','predict_model_v1.py']},recipe='R3 all components refitted on cleaned official labelled rows; causal prefix shrink .5/.5; clean target clip',registration_sha=P.sha(P.H/'registration_v1.json'),cv_score_sha=P.sha(P.H/'final_score_v1.json'),clean_data_hashes=cl['hashes'],versions=reg['versions'],consumed_40_days_in_final_training=True,consumed_40_days_rescored=False,adoption=False,submission_artifact=False)
    P.save(D/'model_manifest_v1.json',manifest);P.save(D/'fit_receipts_v1.json',receipts)
    direct=np.clip(C.shrink(raw_probe,q),*bounds);again=predict_bundle(D,probe_x).set_index('row_id').loc[q.row_id].prediction.to_numpy();gap=float(np.max(abs(direct-again)));assert gap<1e-10
    causal=[];base=pd.Series(again,index=q.row_id)
    for farm in ('F13','F47'):
        meta=C.identify(probe_x);clock=meta.day*24+meta.hour;cut=int(clock[meta.farm==farm].quantile(.5));allowed=(meta.farm==farm)&(clock<=cut)
        changed=probe_x.copy();changed.loc[~allowed,C.RAW]=changed.loc[~allowed,C.RAW]*13+97
        altered=predict_bundle(D,changed).set_index('row_id').prediction;ids=meta.loc[allowed,'row_id'];diff=float(np.max(abs(base.loc[ids]-altered.loc[ids])));assert diff<1e-10;causal.append(dict(farm=farm,cut=cut,rows=len(ids),future_other_farm_maxdiff=diff))
        # Dropping every future and other-farm input gives identical allowed-row predictions.
        prefix=probe_x.loc[allowed].copy();prefix_pred=predict_bundle(D,prefix).set_index('row_id').prediction;diff=float(np.max(abs(base.loc[ids]-prefix_pred.loc[ids])));assert diff<1e-10;causal[-1]['future_other_farm_deleted_maxdiff']=diff
    shuffled=predict_bundle(D,probe_x.sample(frac=1,random_state=20261009)).set_index('row_id').prediction;assert float(np.max(abs(base-shuffled.reindex(base.index))))<1e-10
    weather=probe_x.copy();weather[C.W]=weather[C.W]*37+913;w_pred=predict_bundle(D,weather).set_index('row_id').prediction;assert float(np.max(abs(base-w_pred.reindex(base.index))))<1e-10
    P.csv_new(D/'replay_probe_input_v1.csv',probe_x);P.csv_new(D/'replay_probe_prediction_v1.csv',pd.DataFrame({'row_id':q.row_id,'prediction':direct}))
    P.save(P.H/'final_model_checks_v1.json',dict(status='PASS',model_manifest_sha=P.sha(D/'model_manifest_v1.json'),training_rows=len(t),training_days=len(td),removed_days=len(removed),removed_rows=9600-len(t),inmemory_vs_reload_maxdiff=gap,causal=causal,input_order_invariant=True,query_weather_unused=True,consumed_40_days_validation_reads=0,full400_labels_loaded_after_cv_score_sha=P.sha(P.H/'final_score_v1.json'),adoption=False));print('FINAL_MODEL_COMPLETE',len(t),'rows',len(td),'days removed',len(removed),flush=True)
if __name__=='__main__':final_fit()
