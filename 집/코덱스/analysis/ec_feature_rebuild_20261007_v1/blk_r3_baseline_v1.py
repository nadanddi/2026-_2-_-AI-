"""Fit CPU R3 baseline members, preserve resumable raw predictions; no scoring."""
from pathlib import Path
import json,os,platform,time
from blk_baseline_data_v1 import *
import sklearn,lightgbm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits
from checkpoint_v1 import atomic,digest
from checkpoint_v2 import start_ticks

def model(name,seed):
    if name=='ET':return make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(n_estimators=600,max_features=1.,min_samples_leaf=1,n_jobs=4,random_state=seed))
    if name=='LGB':return lightgbm.LGBMRegressor(n_estimators=800,learning_rate=.03,num_leaves=31,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=1,deterministic=True,force_col_wise=True,n_jobs=4,verbose=-1,random_state=seed,objective='tweedie',tweedie_variance_power=1.5)
    return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),alpha=.01,learning_rate_init=.001,max_iter=800,early_stopping=True,n_iter_no_change=25,validation_fraction=.12,random_state=seed))

def main():
    layoutpath=HERE/'BLK_layout_v2.json'
    layout=json.loads(layoutpath.read_text(encoding='utf-8'))
    source_hashes={name:sha(Path(env.DATA)/name) for name in layout['source_sha256']}
    assert source_hashes==layout['source_sha256'],'actual inputs changed before fit'
    ctx=BLKContext(layout)
    x,tr,table=prepare_reference(ctx)
    q=prepare_query(ctx,sorted(ctx.query_ids),table)
    data_audit=audit(ctx,q,table)
    assert len(tr)==5520 and len(q)==1440
    environment={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__,'device':'CPU','threadpool_limit':1}
    dependencies=[Path(__file__),HERE/'blk_baseline_data_v1.py',HERE/'blk_context_v1.py',HERE/'checkpoint_v1.py',HERE/'checkpoint_v2.py',CORE,DC4,DEPLOY]
    registration={'status':'BASELINE_R3_ONLY_NO_SCORE','layout_sha256':sha(layoutpath),'sources':source_hashes,
        'dependencies':{str(p.relative_to(ROOT)):sha(p) for p in dependencies},'environment':environment,
        'ordered_train_ids':tr.row_id.tolist(),'ordered_query_ids':q.row_id.tolist(),
        'anchor_ids':sorted(ctx.reference_labels),'calendar_fit_ids':tr.row_id.tolist(),
        'PFN_context_ids_NOT_EXECUTED':{str(s):tr.row_id.iloc[np.random.default_rng(s).choice(len(tr),2000,replace=False)].tolist() for s in [5,6,7,8]},
        'feature_columns':{'ET':FULL_R3,'LGB':BASE_R3,'MLP':BASE_R3,'PFN_NOT_EXECUTED':FULL},
        'data_audit':data_audit,'seeds':[47,1414,6464],
        'R3_weights':[.6,.3,.1],'PFN_weight_NOT_EXECUTED':.4,
        'stages_saved':'raw ET/LGB/MLP only, no shrink/clip/SG2/endpoint/held-out labels'}
    folder=HERE/'checkpoints/BLK_R3_v1';folder.mkdir(parents=True,exist_ok=True)
    reg=folder/'registration.json'
    if reg.exists():assert json.loads(reg.read_text(encoding='utf-8'))==registration
    else:atomic(reg,json.dumps(registration,ensure_ascii=False,allow_nan=False))
    lock=folder/'RUN_WRITER_LOCK.json'
    token={'pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),'registration_sha256':sha(reg)}
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(token,f)
    try:
        for seed in registration['seeds']:
            for name,cols in [('ET',FULL_R3),('LGB',BASE_R3),('MLP',BASE_R3)]:
                out=folder/f'{name}_seed{seed}.json'
                if out.exists():
                    saved=json.loads(out.read_text(encoding='utf-8'))
                    assert saved['registration_sha256']==sha(reg) and saved['row_ids']==q.row_id.tolist()
                    assert saved['pred_sha256']==digest(saved['pred'])
                    continue
                assert all(sha(ROOT/p)==v for p,v in registration['dependencies'].items())
                assert all(sha(Path(env.DATA)/p)==v for p,v in source_hashes.items())
                started=time.monotonic()
                print(f'BLK baseline {name} seed{seed} fitting CPU',flush=True)
                m=model(name,seed)
                with threadpool_limits(limits=1):
                    m.fit(tr[cols],tr.sub_ec.to_numpy(float))
                    if name=='ET':m.steps[-1][1].n_jobs=1
                    pred=np.asarray(m.predict(q[cols]),float)
                    single=np.asarray(m.predict(q.iloc[:8][cols]),float)
                    difference=float(np.max(np.abs(pred[:8]-single)))
                    assert difference<=1e-6
                assert np.isfinite(pred).all()
                data={'registration_sha256':sha(reg),'member':name,'seed':seed,'row_ids':q.row_id.tolist(),'pred':pred.tolist(),
                      'pred_sha256':digest(pred.tolist()),'single_batch_max_difference':difference,
                      'duration_seconds':time.monotonic()-started,'heldout_truth_loaded':False}
                atomic(out,json.dumps(data,ensure_ascii=False,allow_nan=False))
                print(f'{name} seed{seed} raw checkpoint complete, {data["duration_seconds"]:.1f}s',flush=True)
        status=folder/'complete.json'
        if not status.exists():atomic(status,json.dumps({'status':'R3_RAW_MEMBERS_COMPLETE','whole_baseline_complete':False,'heldout_scored':False,'remaining':['PFN CPU contexts5..8','one shrink/clip','ref-only SG2/query-role/raw-pass','final full-model causal audit','endpoint comparison']}))
    finally:
        if json.loads(lock.read_text(encoding='utf-8'))==token:lock.unlink()

if __name__=='__main__':main()
