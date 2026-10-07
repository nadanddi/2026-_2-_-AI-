"""CPU PFN with reference-only internal feature statistics and explicit KV cache.

Preserves the earlier uncached outputs. Same weights/context/features/estimators,
but fit_with_cache is a declared correctness repair, not numerical identity claim.
"""
import gc, json, os, platform, time
from blk_baseline_data_v1 import *
import env_extra
for name in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:
    os.environ[name]='1'
import torch, tabpfn
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.architectures import tabpfn_v2 as architecture
from threadpoolctl import threadpool_limits
from checkpoint_v1 import atomic, digest
from checkpoint_v2 import start_ticks
from blk_pfn_minbatch_v1 import predict_minbatch8

LIBRARY_FILES=[
    '.analysis-tools/extra/tabpfn/architectures/tabpfn_v2.py',
    '.analysis-tools/extra/tabpfn/inference.py',
    '.analysis-tools/extra/tabpfn/base.py',
    '.analysis-tools/extra/tabpfn/regressor.py',
    '.analysis-tools/extra/tabpfn/preprocessing/ensemble.py',
    '.analysis-tools/extra/tabpfn/preprocessing/pipeline_interface.py',
    '.analysis-tools/extra/tabpfn/preprocessing/steps/add_fingerprint_features_step.py',
]

def cache_fingerprint(model):
    cache=model.executor_.kv_caches
    assert len(cache)==4
    result=[]
    for item in cache:
        assert item.train_shape[1]==2000 and item.feature_cache is not None
        assert set(item.feature_cache)=={'column_selection_mask','feature_means','scaler_mean','scaler_std','ng_non_constant_mask','ng_num_used_features'}
        values={}
        for name,value in item.feature_cache.items():
            value=value.detach().cpu().contiguous()
            values[name]={'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest()}
        result.append({'train_shape':list(item.train_shape),'feature_statistics':values})
    return result

class ReferenceOnlyTrace:
    """Observe the official path and fail if any prediction attempts a stats fit."""
    def __init__(self):
        self.phase='fit'; self.calls=[]; self.mask_fit_rows=[]; self.group_fit_rows=[]
        self.original_embed=architecture.TabPFNV2._embed_features
        self.original_mask=architecture._constant_feature_mask
        self.original_group=architecture._fit_feature_group_scaling
    def __enter__(self):
        trace=self
        def embed(model, x, **kwargs):
            using_cache=kwargs.get('feature_cache') is not None
            labels=kwargs['num_train_labels']
            if trace.phase=='fit':
                assert not using_cache and len(x)==labels==2000, ('non-reference cache build',len(x),labels)
            else:
                assert using_cache and labels==0, 'Prediction attempted internal feature fitting'
            trace.calls.append({'phase':trace.phase,'rows':len(x),'num_train_labels':labels,'uses_feature_cache':using_cache})
            return trace.original_embed(model,x,**kwargs)
        def mask(x):
            assert trace.phase=='fit', 'Prediction attempted all-row constant mask'
            assert len(x)==2000
            trace.mask_fit_rows.append(len(x)); return trace.original_mask(x)
        def group(x):
            assert trace.phase=='fit', 'Prediction attempted all-row feature-group fitting'
            assert len(x)==2000
            trace.group_fit_rows.append(len(x)); return trace.original_group(x)
        architecture.TabPFNV2._embed_features=embed
        architecture._constant_feature_mask=mask
        architecture._fit_feature_group_scaling=group
        return self
    def __exit__(self,*args):
        architecture.TabPFNV2._embed_features=self.original_embed
        architecture._constant_feature_mask=self.original_mask
        architecture._fit_feature_group_scaling=self.original_group

def main():
    old_folder=HERE/'checkpoints/BLK_PFN_CPU_v1'
    old_path=old_folder/'registration.json'; old=json.loads(old_path.read_text(encoding='utf-8'))
    rpath=HERE/'checkpoints/BLK_R3_v1/registration.json'; rreg=json.loads(rpath.read_text(encoding='utf-8'))
    assert old['R3_registration_sha256']==sha(rpath)
    assert all(sha(ROOT/p)==s for p,s in rreg['dependencies'].items())
    assert all(sha(Path(env.DATA)/p)==s for p,s in rreg['sources'].items())
    assert sha(old['weights_path'])==old['weights_sha256']
    layout_path=HERE/'BLK_layout_v2.json'; assert sha(layout_path)==rreg['layout_sha256']
    layout=json.loads(layout_path.read_text(encoding='utf-8'))
    ctx=BLKContext(layout); _,tr,table=prepare_reference(ctx)
    ids=old['query_ids']; q=prepare_query(ctx,ids,table)
    assert tr.row_id.tolist()==rreg['ordered_train_ids'] and FULL==old['columns']
    folder=HERE/'checkpoints/BLK_PFN_CPU_REFONLY_v1'; folder.mkdir(parents=True,exist_ok=True)
    reg={**old,'parent_uncached_registration_sha256':sha(old_path),'code_sha256':sha(__file__),
        'fit_mode':'fit_with_cache','kv_cache_precision':'auto','minimum_inference_rows':8,
        'padding':'own last supplied query row copies only; no new information',
        'adapter_sha256':sha(HERE/'blk_pfn_minbatch_v1.py'),
        'library_sha256':{p:sha(ROOT/p) for p in LIBRARY_FILES},
        'inference_engine_required':'InferenceEngineExplicitKVCache','architecture_required':'TabPFNV2',
        'internal_statistics_policy':'fit all internal masks/means/std/group counts on reference2000 only; never fit at prediction',
        'uncached_numeric_identity_NOT_CLAIMED':True,'atol':1e-6,'rtol':0,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'tabpfn':tabpfn.__version__,
            'device':'cpu','torch_threads':4,'torch_interop':1,'threadpool_limit':4}}
    rp=folder/'registration.json'
    if rp.exists(): assert json.loads(rp.read_text(encoding='utf-8'))==reg
    else: atomic(rp,json.dumps(reg,ensure_ascii=False))
    lock=folder/'RUN_WRITER_LOCK.json'; token={'pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),'registration_sha256':sha(rp)}
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w',encoding='utf-8') as f: json.dump(token,f)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    X=tr[FULL].to_numpy(np.float32); Q=q[FULL].to_numpy(np.float32); y=tr.sub_ec.to_numpy(float)
    probes=[f'{b["farm"]}_{b["query_days"][i]:03d}_{h:02d}' for b in layout['blocks']
        for i in [0,len(b['query_days'])//2,len(b['query_days'])-1] for h in [0,5,12,23]]
    indices=[ids.index(r) for r in probes]; order=[0,13,26,39,52,65,78,95]
    try:
        for seed in [5,6,7,8]:
            out=folder/f'context{seed}.json'; audit_path=folder/f'context{seed}_audit.json'
            if out.exists() or audit_path.exists():
                assert out.exists() and audit_path.exists(), 'Incomplete output pair; preserve and investigate'
                saved=json.loads(out.read_text(encoding='utf-8')); audit=json.loads(audit_path.read_text(encoding='utf-8'))
                assert saved['registration_sha256']==sha(rp) and digest(saved['pred'])==saved['pred_sha256']
                assert audit['status']=='PASS' and audit['prediction_file_sha256']==sha(out)
                continue
            started=time.monotonic(); ix=np.random.default_rng(seed).choice(len(tr),size=2000,replace=False)
            assert tr.row_id.iloc[ix].tolist()==reg['contexts'][str(seed)]
            print(f'context{seed}: reference-only cache build starts',flush=True)
            model=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cpu',model_path=old['weights_path'],
                n_estimators=4,random_state=seed,ignore_pretraining_limits=True,inference_precision=torch.float32,
                n_preprocessing_jobs=1,fit_mode='fit_with_cache',kv_cache_precision='auto')
            with threadpool_limits(limits=4), ReferenceOnlyTrace() as trace:
                model.fit(X[ix],y[ix])
                assert type(model.executor_).__name__==reg['inference_engine_required']
                assert all(type(m).__name__==reg['architecture_required'] for m in model.models_)
                before=cache_fingerprint(model); trace.phase='predict'
                print(f'context{seed}: cache built; full1440 prediction starts',flush=True)
                pred=predict_minbatch8(model,Q); expected=pred[indices]; differences={}
                subset=predict_minbatch8(model,Q[indices])
                differences['scattered96_vs_full1440']=float(np.max(np.abs(subset-expected)))
                reversed_pred=predict_minbatch8(model,Q[np.asarray(indices)[order[::-1]]])[::-1]
                differences['reversed8_vs_full1440']=float(np.max(np.abs(reversed_pred-expected[order])))
                singles=[abs(predict_minbatch8(model,Q[[indices[p]]])[0]-expected[p]) for p in order]
                differences['independent_single8_vs_full1440']=float(max(singles))
                for count in range(2,8):
                    selected=np.asarray(indices)[order[:count]]; target=expected[order[:count]]
                    for reversed_order in [False,True]:
                        inp=Q[selected[::-1]] if reversed_order else Q[selected]
                        result=predict_minbatch8(model,inp)
                        if reversed_order: result=result[::-1]
                        differences[f'batch{count}_reverse{reversed_order}']=float(np.max(np.abs(result-target)))
                poison=Q[np.asarray(indices)[[0,1,2,3,24,48,72,95]]].copy()
                poison[1:]=np.nan_to_num(poison[1:],nan=0)*13+97
                differences['protected_first_row_other_query_poison']=float(abs(predict_minbatch8(model,poison)[0]-expected[0]))
                after=cache_fingerprint(model); assert before==after, 'Internal feature statistics mutated at inference'
            assert np.isfinite(pred).all() and len(pred)==1440
            old_pred=np.asarray(json.loads((old_folder/f'context{seed}.json').read_text(encoding='utf-8'))['pred'])
            saved={'registration_sha256':sha(rp),'context':seed,'row_ids':ids,'pred':pred.tolist(),
                'pred_sha256':digest(pred.tolist()),'heldout_truth_loaded':False,'fit_mode':'fit_with_cache'}
            atomic(out,json.dumps(saved,ensure_ascii=False,allow_nan=False))
            maximum=max(differences.values()); status='PASS' if maximum<=1e-6 else 'FAIL_PRESERVE_DO_NOT_SCORE'
            audit={'status':status,'context':seed,'registration_sha256':sha(rp),'prediction_file_sha256':sha(out),
                'code_sha256':sha(__file__),'adapter_sha256':reg['adapter_sha256'],'checks':differences,
                'max_difference':maximum,'atol':1e-6,'rtol':0,'reference_only_trace':trace.calls,
                'constant_mask_fit_rows':trace.mask_fit_rows,'group_statistics_fit_rows':trace.group_fit_rows,
                'feature_cache_before':before,'feature_cache_after':after,'feature_cache_unchanged':True,
                'uncached_parent_max_difference_DIAGNOSTIC_ONLY':float(np.max(np.abs(pred-old_pred))),
                'probes':probes,'heldout_truth_loaded':False,'GPU_used':False,'duration_seconds':time.monotonic()-started,
                'limits':['Finite empirical numerical tests; internal train-only statistics additionally enforced by observed runtime guard',
                    'No byte identity claim with earlier uncached or historical GPU output','Other-query poison includes earlier rows too; not pure future-only test']}
            atomic(audit_path,json.dumps(audit,ensure_ascii=False,allow_nan=False))
            print(f'context{seed}: {status}, maxbatchdifference={maximum}, duration={audit["duration_seconds"]:.1f}s',flush=True)
            assert status=='PASS', differences
            del model; gc.collect()
        complete=folder/'complete.json'; assert not complete.exists()
        atomic(complete,json.dumps({'status':'PFN_REFERENCE_ONLY_RAW_CONTEXTS_AND_AUDITS_COMPLETE',
            'registration_sha256':sha(rp),'whole_baseline_complete':False,'heldout_scored':False}))
    except BaseException as exc:
        failure=folder/f'failure_{time.time_ns()}.json'
        atomic(failure,json.dumps({'status':'FAILED_PRESERVE_DO_NOT_SCORE','exception':repr(exc),
            'registration_sha256':sha(rp),'heldout_truth_loaded':False,'GPU_used':False},ensure_ascii=False))
        raise
    finally:
        if json.loads(lock.read_text(encoding='utf-8'))==token: lock.unlink()

if __name__=='__main__': main()
