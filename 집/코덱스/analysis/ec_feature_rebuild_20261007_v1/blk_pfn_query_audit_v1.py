"""Re-fit one frozen CPU context for scattered, order, single, poisoned-batch checks."""
import argparse,json,os,time
from blk_baseline_data_v1 import *
import env_extra
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from threadpoolctl import threadpool_limits
from checkpoint_v1 import atomic,digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--context',type=int,choices=[5,6,7,8],required=True)
    seed=parser.parse_args().context
    folder=HERE/'checkpoints/BLK_PFN_CPU_v1'
    rp=folder/'registration.json';reg=json.loads(rp.read_text(encoding='utf-8'))
    original=folder/f'context{seed}.json';saved=json.loads(original.read_text(encoding='utf-8'))
    assert saved['registration_sha256']==sha(rp) and digest(saved['pred'])==saved['pred_sha256']
    rregpath=HERE/'checkpoints/BLK_R3_v1/registration.json'
    rreg=json.loads(rregpath.read_text(encoding='utf-8'))
    assert reg['R3_registration_sha256']==sha(rregpath)
    assert all(sha(ROOT/p)==s for p,s in rreg['dependencies'].items())
    assert all(sha(Path(env.DATA)/p)==s for p,s in rreg['sources'].items())
    assert sha(reg['weights_path'])==reg['weights_sha256']
    layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
    ctx=BLKContext(layout);_,tr,table=prepare_reference(ctx)
    ids=reg['query_ids'];q=prepare_query(ctx,ids,table)
    assert tr.row_id.tolist()==rreg['ordered_train_ids'] and FULL==reg['columns']
    ix=np.random.default_rng(seed).choice(len(tr),size=2000,replace=False)
    assert tr.row_id.iloc[ix].tolist()==reg['contexts'][str(seed)]
    probes=[f'{b["farm"]}_{b["query_days"][i]:03d}_{h:02d}' for b in layout['blocks'] for i in [0,len(b['query_days'])//2,len(b['query_days'])-1] for h in [0,5,12,23]]
    indices=[ids.index(r) for r in probes]
    Q=q[FULL].to_numpy(np.float32);expected=np.asarray(saved['pred'])[indices]
    out=HERE/f'BLK_PFN_context{seed}_query_audit_v1.json';assert not out.exists()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    m=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cpu',model_path=reg['weights_path'],n_estimators=4,random_state=seed,
        ignore_pretraining_limits=True,inference_precision=torch.float32,n_preprocessing_jobs=1)
    start=time.monotonic();differences={}
    with threadpool_limits(limits=4):
        m.fit(tr[FULL].to_numpy(np.float32)[ix],tr.sub_ec.to_numpy(float)[ix])
        subset=np.asarray(m.predict(Q[indices]),float)
        differences['scattered96_vs_original1440']=float(np.max(np.abs(subset-expected)))
        order=[0,13,26,39,52,65,78,95]
        reversed_pred=np.asarray(m.predict(Q[np.asarray(indices)[order[::-1]]]),float)[::-1]
        differences['reversed_scattered8_vs_scattered96']=float(np.max(np.abs(reversed_pred-subset[order])))
        single=float(np.asarray(m.predict(Q[[indices[-1]]]),float)[0])
        differences['single_last_row_vs_original1440']=abs(single-expected[-1])
        # Protect earliest F13 query row. Poison later/other-farm query feature rows
        # in this batch. Reference and protected row remain identical.
        poison=Q[np.asarray(indices)[[0,1,2,3,24,48,72,95]]].copy()
        poison[1:]=np.nan_to_num(poison[1:],nan=0)*13+97
        mixed=np.asarray(m.predict(poison),float)
        differences['protected_first_row_future_other_farm_poison_batch']=abs(mixed[0]-expected[0])
    maximum=max(differences.values())
    status='PASS' if maximum<=1e-6 else 'FAIL_PRESERVE_DO_NOT_SCORE'
    atomic(out,json.dumps({'status':status,'context':seed,'probes':probes,'checks':differences,'max_difference':maximum,'atol':1e-6,'rtol':0,
        'code_sha256':sha(__file__),'PFN_registration_sha256':sha(rp),'original_context_sha256':sha(original),
        'heldout_truth_loaded':False,'GPU_used':False,'duration_seconds':time.monotonic()-start,
        'scope':'all8 blocks, front/middle/back, hours0/5/12/23; order8, single1, poisonedbatch1. Finite empirical checks, not exhaustive input-domain proof'},ensure_ascii=False,allow_nan=False))
    print(json.dumps({'context':seed,'status':status,'differences':differences}),flush=True)
    assert status=='PASS',differences

if __name__=='__main__':main()
