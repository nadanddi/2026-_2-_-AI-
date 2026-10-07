"""CPU-only PFN v2 contexts 5..8, pinned existing weights, no held-out labels."""
import os,json,platform,time,gc
from blk_baseline_data_v1 import *
import env_extra
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
import torch,tabpfn
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from threadpoolctl import threadpool_limits
from checkpoint_v1 import atomic,digest
from checkpoint_v2 import start_ticks

def main():
    weights=Path.home()/'AppData/Roaming/tabpfn/tabpfn-v2-regressor.ckpt'
    assert sha(weights)=='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
    regpath=HERE/'checkpoints/BLK_R3_v1/registration.json'
    reg=json.loads(regpath.read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==v for p,v in reg['dependencies'].items())
    assert all(sha(Path(env.DATA)/p)==v for p,v in reg['sources'].items())
    layoutpath=HERE/'BLK_layout_v2.json'
    assert sha(layoutpath)==reg['layout_sha256']
    ctx=BLKContext(json.loads(layoutpath.read_text(encoding='utf-8')))
    _,tr,table=prepare_reference(ctx)
    q=prepare_query(ctx,reg['ordered_query_ids'],table)
    assert tr.row_id.tolist()==reg['ordered_train_ids']
    folder=HERE/'checkpoints/BLK_PFN_CPU_v1';folder.mkdir(parents=True,exist_ok=True)
    registration={'R3_registration_sha256':sha(regpath),'weights_path':str(weights),'weights_sha256':sha(weights),
        'code_sha256':sha(__file__),'environment':{'python':platform.python_version(),'torch':torch.__version__,'tabpfn':tabpfn.__version__,'device':'cpu','torch_threads':4,'torch_interop':1,'threadpool_limit':4},
        'columns':FULL,'contexts':reg['PFN_context_ids_NOT_EXECUTED'],'query_ids':q.row_id.tolist(),
        'n_estimators':4,'precision':'float32','random_state':'context_seed','training_rows_per_context':2000,
        'model_version':'V2','heldout_truth_loaded':False,'CPU_GPU_numerical_identity_NOT_CLAIMED':True}
    rp=folder/'registration.json'
    if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==registration
    else:atomic(rp,json.dumps(registration,ensure_ascii=False))
    lock=folder/'RUN_WRITER_LOCK.json'
    token={'pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),'registration_sha256':sha(rp)}
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(token,f)
    try:
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        X=tr[FULL].to_numpy(np.float32);Q=q[FULL].to_numpy(np.float32);y=tr.sub_ec.to_numpy(float)
        for seed in [5,6,7,8]:
            out=folder/f'context{seed}.json'
            if out.exists():
                saved=json.loads(out.read_text(encoding='utf-8'))
                assert saved['registration_sha256']==sha(rp) and saved['pred_sha256']==digest(saved['pred'])
                continue
            started=time.monotonic()
            ix=np.random.default_rng(seed).choice(len(tr),size=2000,replace=False)
            assert tr.row_id.iloc[ix].tolist()==registration['contexts'][str(seed)]
            m=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cpu',model_path=str(weights),n_estimators=4,random_state=seed,
                ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1)
            print(f'BLK TabPFN context{seed} CPU fit/predict starts',flush=True)
            with threadpool_limits(limits=4):
                m.fit(X[ix],y[ix]);pred=np.asarray(m.predict(Q),float)
                small=np.asarray(m.predict(Q[:8]),float)
                difference=float(np.max(np.abs(pred[:8]-small)))
                assert difference<=1e-6,('query batch sensitivity',difference)
            assert np.isfinite(pred).all()
            data={'registration_sha256':sha(rp),'context':seed,'row_ids':q.row_id.tolist(),'pred':pred.tolist(),'pred_sha256':digest(pred.tolist()),
                'single_batch_max_difference':difference,'duration_seconds':time.monotonic()-started,'heldout_truth_loaded':False}
            atomic(out,json.dumps(data,ensure_ascii=False,allow_nan=False))
            print(f'context{seed} raw CPU checkpoint complete {data["duration_seconds"]:.1f}s',flush=True)
            del m;gc.collect()
        out=folder/'complete.json'
        if not out.exists():atomic(out,json.dumps({'status':'PFN_RAW_CONTEXTS_COMPLETE','whole_baseline_complete':False,'heldout_scored':False}))
    finally:
        if json.loads(lock.read_text(encoding='utf-8'))==token:lock.unlink()

if __name__=='__main__':main()
