"""사전 고정된 잠금 40일 최종 확인. 예측 완료 후 정답을 단 한 번 연다."""
from pathlib import Path
import sys, json, csv, os, gc
sys.dont_write_bytecode=True
import run_dc4 as d
import numpy as np
from threadpoolctl import threadpool_limits

OUT=d.OUT/'locked_confirmation'

def predict(kind,tr,q,core):
    p=OUT/f'{kind}_predictions.npz'
    base={'script':d.engine.sha(__file__),'protocol':d.engine.sha(d.HERE/'PROTOCOL.md'),
          'transform':d.engine.sha(d.__file__),'core':d.engine.sha(d.common.CORE_PATH),
          'training_rows':tr.row_id.tolist(),'query_rows':q.row_id.tolist(),
          'features':{'FULL':core.FULL,'BASE':core.BASE},'environment':d.engine.environment_manifest()}
    key=d.engine.canonical_hash(base)
    cached=d.engine.load_cache(p,key,q.row_id)
    if cached:return cached[0]
    r3=[]
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    for seed in d.SEEDS:
        e=core.et(seed);e.steps[-1][1].set_params(n_jobs=2)
        l=core.lg(seed,'tweedie').set_params(n_jobs=2)
        m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128,64),alpha=.01,learning_rate_init=.001,
                max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=seed))
        preds=[]
        for model,cols in [(e,core.FULL),(l,core.BASE),(m,core.BASE)]:
            with threadpool_limits(limits=2):model.fit(tr[cols],tr.sub_ec);preds.append(np.asarray(model.predict(q[cols]),float))
        r3.append(.6*preds[0]+.3*preds[1]+.1*preds[2]);del e,l,m,preds;gc.collect()
        print(f'잠금 예측 {kind} R3 {seed} 완료; 잠금 정답 미열람',flush=True)
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(4);bags=[]
    x=tr[core.FULL].to_numpy(np.float32);query=q[core.FULL].to_numpy(np.float32)
    for seed in [1,2,3,4]:
        ix=np.random.default_rng(seed).choice(len(tr),size=min(2000,len(tr)),replace=False)
        model=TabPFNRegressor.create_default_for_version(ModelVersion.V2,model_path=str(d.engine.CKPT),device='cpu',
             n_estimators=4,random_state=seed,ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1)
        with threadpool_limits(limits=4):
            model.fit(x[ix],tr.sub_ec.to_numpy(float)[ix]);pred=np.asarray(model.predict(query),float)
            assert np.array_equal(pred[:8],np.asarray(model.predict(query[:8]),float))
        bags.append(pred);del model;gc.collect()
        print(f'잠금 예측 {kind} PFN {seed} 완료; 잠금 정답 미열람',flush=True)
    raw=.8*np.mean(r3,axis=0)+.2*np.mean(bags,axis=0)
    arrays={'row_id':q.row_id.to_numpy(str),'prediction':d.common.finish(raw,tr,q),
            'raw_r3_by_seed':np.stack(r3),'raw_pfn_by_context':np.stack(bags)}
    d.engine.write_cache(p,key,arrays,{'provenance':base,'final_lock_labels_opened':False})
    return arrays

def main():
    # Public evidence must exist first; outcome never changes the fixed models or formula.
    assert (d.HERE/'et_replication_result.json').exists() and (d.HERE/'v2_integration_result.json').exists()
    assert json.loads((d.HERE/'et_replication_result.json').read_text(encoding='utf-8'))['replication_pass']
    OUT.mkdir(parents=True,exist_ok=True)
    ledger=OUT/'LOCK_CONSUMED_ONCE.json'
    assert not ledger.exists(),'잠금 확인을 이미 소비했습니다. 정답 재채점 금지.'
    raw,lab,folds,locks,core,base=d.setup()
    tr=lab[d.common.near_mask(lab,locks)].reset_index(drop=True)
    q=core.features(raw)
    q=q[[(f,int(day)) in locks for f,day in zip(q.farm,q.day)]].reset_index(drop=True)
    assert len(tr)==7344 and len(tr[['farm','day']].drop_duplicates())==306
    assert len(q)==960 and len(q[['farm','day']].drop_duplicates())==40
    seasonal_tr,seasonal_q,checks=d.transform(raw,tr,q,check=True)
    day=predict('day',tr,q,core)
    season=predict('season',seasonal_tr,seasonal_q,d.seasonal_core(core))
    assert day['row_id'].tolist()==season['row_id'].tolist()==q.row_id.tolist()
    assert np.isfinite(day['prediction']).all() and np.isfinite(season['prediction']).all()
    # Exclusive ledger is created BEFORE first numeric conversion of sealed labels.
    record={'status':'LABELS_OPENING','script':d.engine.sha(__file__),'protocol':d.engine.sha(d.HERE/'PROTOCOL.md'),
            'formula':'season_v2_ensemble_rmse < day_v2_ensemble_rmse','rows':960,
            'prediction_files':{p.name:d.engine.sha(p) for p in OUT.glob('*predictions.npz')},
            'training_rows':7344,'training_days':306,'season_checks':checks,'allowed_reuse':False}
    with ledger.open('x',encoding='utf-8') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
    values={};ids=set(q.row_id)
    with (d.common.DATA/'train_y.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['row_id'] in ids:values[row['row_id']]=float(row['sub_ec'])
    assert set(values)==ids
    y=np.asarray([values[k] for k in q.row_id],float)
    dr=d.engine.rmse(y,day['prediction']);sr=d.engine.rmse(y,season['prediction'])
    independent_day=d.common.rmse(y,day['prediction']);independent_season=d.common.rmse(y,season['prediction'])
    assert abs(dr-independent_day)<1e-12 and abs(sr-independent_season)<1e-12
    result=record|{'status':'CONSUMED','day_rmse':dr,'season_rmse':sr,'delta_rmse':sr-dr,
                 'locked_gate_pass':sr<dr,'labels_read_once':True,'n_labels':len(y),'independent_rmse_pass':True,
                 'candidate_adopted':False,'independent_full_review_pending':True}
    d.engine.atomic_json(d.HERE/'locked_confirmation_result.json',result)
    # Preserve first-open ledger unchanged; final result is a separate file.
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
