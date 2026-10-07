"""Original66 CPU PFN baseline contexts, train-only cache; no score or candidate fit."""
from pathlib import Path
import json,os,time,gc,hashlib,platform,importlib,sys
import original_fold_features_v2 as feature_module
import blk_pfn_refonly_cache_v3 as cache_module
from blk_baseline_data_v1 import *
from blk_pfn_minbatch_v1 import predict_minbatch8
from checkpoint_v1 import atomic,digest
from checkpoint_v2 import start_ticks
import original_pfn_cache_rules_v1 as rules_module
import blk_baseline_data_v1 as baseline_module
import blk_context_v1 as context_module
import blk_pfn_minbatch_v1 as adapter_module
import checkpoint_v1 as checkpoint_module
import checkpoint_v2 as checkpoint2_module
import env_extra as extra_module
import threadpoolctl as pool_module
torch=cache_module.torch
tabpfn=cache_module.tabpfn
TabPFNRegressor=cache_module.TabPFNRegressor
ModelVersion=cache_module.ModelVersion
threadpool_limits=cache_module.threadpool_limits
CHECK_KEYS={'scattered_vs_full','reversed8_vs_full','independent_single8_vs_full',
    'protected_first_row_other_query_poison','repeated_scattered_after_other_checks','repeated_full_after_other_checks',
    *[f'batch{n}_reverse{rev}' for n in range(2,8) for rev in [False,True]],
    *[f'fresh_prefix_probe{i}' for i in range(3)]}

def runtime_paths():
    modules={'features':feature_module,'cache_template':cache_module,'torch':torch,'tabpfn':tabpfn,
        'rules':rules_module,'baseline':baseline_module,'context':context_module,'adapter':adapter_module,
        'checkpoint':checkpoint_module,'checkpoint2':checkpoint2_module,'env':env,'env_extra':extra_module,
        'threadpoolctl':pool_module,'numpy':np,'pandas':pd} 
    for relative in cache_module.LIBRARY_FILES:
        suffix=relative.split('/tabpfn/',1)[1][:-3].replace('/','.')
        modules['tabpfn.'+suffix]=importlib.import_module('tabpfn.'+suffix)
    return {name:str(Path(module.__file__).resolve()) for name,module in modules.items()}

def verify_sources(reg):
    assert all(sha(p)==v for p,v in reg['source_sha256'].items())
    assert runtime_paths()==reg['runtime_module_paths']
    assert torch.version.cuda is None and '+cpu' in torch.__version__
    assert {'python':platform.python_version(),'torch':torch.__version__,'tabpfn':tabpfn.__version__}==reg['environment']

def validate_saved(saved,audit,contract,ids,path):
    assert saved['status']=='ORIGINAL_PFN_REFERENCE_ONLY_RAW_PASS_NO_SCORE'
    assert saved['contract']==contract and saved['row_ids']==ids
    assert len(saved['pred'])==len(ids) and np.isfinite(saved['pred']).all()
    assert digest(saved['pred'])==saved['pred_sha256']
    assert saved['heldout_truth_loaded'] is False
    assert audit['status']=='PASS' and audit['contract']==contract and audit['prediction_file_sha256']==sha(path)
    assert audit['auditor_sha256']==sha(rules_module.__file__)
    assert audit['code_sha256']==sha(__file__)
    rules_module.audit_metadata(audit,len(ids),CHECK_KEYS)

def cache_fingerprint(model):
    for cache in model.executor_.kv_caches:
        assert all(t.device.type=='cpu' for t in cache.feature_cache.values())
        assert cache.test_y_embedding.device.type=='cpu'
        for layer in cache.kv.values():
            assert layer.key.device.type==layer.value.device.type=='cpu'
    result=cache_module.cache_fingerprint(model)
    for cache in result:
        for value in cache['feature_statistics'].values():value['device']='cpu'
        cache['target_embedding']['device']='cpu'
        for layer in cache['kv'].values():
            for value in layer.values():value['device']='cpu'
    rules_module.fingerprint(result)
    return result

def predict_context(reg,rawreg,fold,bundle,seed,folder):
    ctx,tr,q,calendar,_,_,details=bundle
    verify_sources(reg)
    assert details['require_all66'] is True
    ids=q.row_id.tolist();ix=np.random.default_rng(seed).choice(len(tr),size=2000,replace=False)
    context_ids=tr.row_id.iloc[ix].tolist()
    keyname=f'{fold["validator"]}_fold{fold["fold"]}'
    assert context_ids==reg['contexts'][keyname][str(seed)]
    assert len(context_ids)==len(set(context_ids))==2000
    assert set(context_ids)<=set(fold['ordered_train_ids'])
    assert not(set(context_ids)&(set(fold['ordered_query_ids'])|set(fold['input_forbidden_ids'])))
    X=tr[FULL].to_numpy(np.float32);Q=q[FULL].to_numpy(np.float32);y=tr.sub_ec.to_numpy(float)
    assert len(y)==len(tr) and np.isfinite(y).all()
    assert hashlib.sha256(y.astype('<f8').tobytes()).hexdigest()==details['train_label_sha256']
    contract={'registration_sha256':sha(HERE/'DOMAIN24_original_pfn_registration_v2.json'),
        'raw_fit_registration_sha256':sha(HERE/'DOMAIN24_original_raw_fit_registration_v3.json'),
        'validator':fold['validator'],'fold':fold['fold'],'context_seed':seed,'columns':FULL,
        'context_ids_sha256':digest(context_ids),'ordered_query_ids_sha256':digest(ids),
        'prepared_fold_sha256':details['preparation_receipt_sha256'],
        'train_context_matrix_sha256':hashlib.sha256(X[ix].tobytes()).hexdigest(),
        'reference_labels_sha256':hashlib.sha256(y[ix].tobytes()).hexdigest(),
        'full_query_matrix_sha256':hashlib.sha256(Q.tobytes()).hexdigest()}
    out=folder/f'context{seed}.json';ap=folder/f'context{seed}_audit.json'
    if out.exists() or ap.exists():
        assert out.exists() and ap.exists(),'Incomplete output pair preserved; investigate before resume'
        validate_saved(json.loads(out.read_text(encoding='utf-8')),json.loads(ap.read_text(encoding='utf-8')),contract,ids,out)
        return out,ap
    started=time.monotonic();probe_indices=sorted(set([0,len(q)//2,len(q)-1]));prefix={}
    assert len(probe_indices)==3
    for i,rid in enumerate(ids[j] for j in probe_indices):
        packet=ctx.query_prefix(rid);forbidden=set(ctx._query)-set(packet);saved={k:ctx._query[k] for k in forbidden}
        try:
            for k in forbidden:ctx._query[k]={c:99999. for c in saved[k]}
            short=prepare_query(ctx,[rid],calendar).set_index('row_id')[FULL].to_numpy(np.float32)
            np.testing.assert_allclose(short,Q[[probe_indices[i]]],atol=0,rtol=0,equal_nan=True)
            prefix[i]=short
        finally:ctx._query.update(saved)
    print(f'{keyname} PFNcontext{seed}: train-only CPU cache fit',flush=True)
    model=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cpu',model_path=reg['weights_path'],
        n_estimators=4,random_state=seed,ignore_pretraining_limits=True,inference_precision=torch.float32,
        n_preprocessing_jobs=1,fit_mode='fit_with_cache',kv_cache_precision='auto')
    with threadpool_limits(limits=4),cache_module.ReferenceOnlyTrace() as trace:
        model.fit(X[ix],y[ix]);assert type(model.executor_).__name__=='InferenceEngineExplicitKVCache'
        assert all(type(m).__name__=='TabPFNV2' for m in model.models_)
        before=cache_fingerprint(model);trace.phase='predict'
        pred=predict_minbatch8(model,Q);indices=np.linspace(0,len(Q)-1,min(96,len(Q)),dtype=int)
        order=np.linspace(0,len(indices)-1,8,dtype=int);expected=pred[indices];differences={}
        subset=predict_minbatch8(model,Q[indices])
        differences['scattered_vs_full']=float(np.max(np.abs(subset-expected)))
        reverse=predict_minbatch8(model,Q[indices[order[::-1]]])[::-1]
        differences['reversed8_vs_full']=float(np.max(np.abs(reverse-expected[order])))
        differences['independent_single8_vs_full']=float(max(abs(predict_minbatch8(model,Q[[indices[i]]])[0]-expected[i]) for i in order))
        for count in range(2,8):
            selected=indices[order[:count]];target=expected[order[:count]]
            for rev in [False,True]:
                p=predict_minbatch8(model,Q[selected[::-1] if rev else selected])
                if rev:p=p[::-1]
                differences[f'batch{count}_reverse{rev}']=float(np.max(np.abs(p-target)))
        poison=Q[indices[order]].copy();poison[1:]=np.nan_to_num(poison[1:],nan=0)*13+97
        differences['protected_first_row_other_query_poison']=float(abs(predict_minbatch8(model,poison)[0]-expected[order[0]]))
        differences['repeated_scattered_after_other_checks']=float(np.max(np.abs(predict_minbatch8(model,Q[indices])-expected)))
        differences['repeated_full_after_other_checks']=float(np.max(np.abs(predict_minbatch8(model,Q)-pred)))
        for i,px in prefix.items():differences[f'fresh_prefix_probe{i}']=float(abs(predict_minbatch8(model,px)[0]-pred[probe_indices[i]]))
        after=cache_fingerprint(model)
        state={'fit_mode':model.fit_mode,'kv_cache_precision':model.executor_.kv_cache_precision,
            'engine':type(model.executor_).__module__+'.'+type(model.executor_).__name__,
            'architectures':[type(m).__module__+'.'+type(m).__name__ for m in model.models_],
            'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
            'inference_precision':str(model.inference_precision),'memory_saving_mode':model.memory_saving_mode}
    assert set(differences)==CHECK_KEYS
    assert np.isfinite(pred).all() and np.isfinite(list(differences.values())).all()
    maximum=max(differences.values());status='PASS' if maximum<=1e-6 and before==after else 'FAIL_PRESERVE_DO_NOT_SCORE'
    saved={'status':'ORIGINAL_PFN_REFERENCE_ONLY_RAW_PASS_NO_SCORE' if status=='PASS' else 'FAIL_PRESERVE_DO_NOT_SCORE',
        'contract':contract,'row_ids':ids,'pred':pred.tolist(),'pred_sha256':digest(pred.tolist()),'heldout_truth_loaded':False}
    audit={'status':status,'code_sha256':sha(__file__),'auditor_sha256':sha(rules_module.__file__),'contract':contract,'checks':differences,'max_difference':maximum,
        'reference_only_trace':trace.calls,'constant_mask_fit_rows':trace.mask_fit_rows,'group_statistics_fit_rows':trace.group_fit_rows,
        'feature_cache_before':before,'feature_cache_after':after,'feature_cache_unchanged':before==after,
        'runtime_state':state,'GPU_used':False,'heldout_truth_loaded':False,'duration_seconds':time.monotonic()-started,
        'limits':['Finite batch and three prefix probes; full-model causal gate still required','No historical GPU/uncached equality claim']}
    verify_sources(reg)
    assert not out.exists() and not ap.exists();atomic(out,json.dumps(saved,ensure_ascii=False,allow_nan=False))
    audit['prediction_file_sha256']=sha(out);atomic(ap,json.dumps(audit,ensure_ascii=False,allow_nan=False))
    validate_saved(saved,audit,contract,ids,out)
    del model;gc.collect()
    print(f'{keyname} PFNcontext{seed} PASS max{maximum} {audit["duration_seconds"]:.1f}s',flush=True)
    return out,ap

def main():
    if not __debug__ or sys.flags.optimize:raise RuntimeError('Python -O prohibited')
    assert Path(feature_module.__file__).resolve()==(HERE/'original_fold_features_v2.py').resolve()
    assert Path(cache_module.__file__).resolve()==(HERE/'blk_pfn_refonly_cache_v3.py').resolve()
    regpath=HERE/'DOMAIN24_original_pfn_registration_v2.json';reg=json.loads(regpath.read_text(encoding='utf-8'))
    assert reg['status']=='REGISTERED_ORIGINAL66_PFN_REFERENCE_ONLY_BEFORE_FIT'
    assert reg['runner_sha256']==sha(__file__) and reg['context_seeds']==[5,6,7,8] and reg['context_fits']==264
    verify_sources(reg)
    rawpath=HERE/'DOMAIN24_original_raw_fit_registration_v3.json';rawreg=json.loads(rawpath.read_text(encoding='utf-8'))
    assert reg['raw_fit_registration_sha256']==sha(rawpath)
    registry=json.loads((HERE/'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    assert len(registry['folds'])==66
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    root=HERE/'checkpoints/DOMAIN24_ORIGINAL_PFN_CACHE_v2';root.mkdir(parents=True,exist_ok=True)
    token={'pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),'registration_sha256':sha(regpath)}
    lock=root/'RUN_WRITER_LOCK.json';outputs={}
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w',encoding='utf-8') as h:json.dump(token,h)
    try:
        for fold in registry['folds']:
            bundle=feature_module.load_production_fold(fold['validator'],fold['fold'],rawreg)
            folder=root/f'{fold["validator"]}_fold{fold["fold"]}';folder.mkdir(parents=True,exist_ok=True)
            for seed in [5,6,7,8]:
                for path in predict_context(reg,rawreg,fold,bundle,seed,folder):outputs[str(path.relative_to(root))]=sha(path)
            summary={'status':'ORIGINAL_FOLD_PFN4_RAW_AND_AUDITS_COMPLETE_NO_SCORE','registration_sha256':sha(regpath),
                'files_sha256':{k:v for k,v in outputs.items() if k.startswith(folder.name+'/')},'heldout_truth_loaded':False}
            assert len(summary['files_sha256'])==8
            path=folder/'complete.json'
            if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==summary
            else:atomic(path,json.dumps(summary,ensure_ascii=False))
        assert len(outputs)==528;verify_sources(reg)
        summary={'status':'ORIGINAL66_PFN264_CONTEXTS_RAW_AND_AUDITS_COMPLETE_NO_SCORE',
            'registration_sha256':sha(regpath),'files_sha256':outputs,'whole_baseline_complete':False,'heldout_truth_loaded':False}
        path=root/'complete.json'
        if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==summary
        else:atomic(path,json.dumps(summary,ensure_ascii=False))
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))==token:lock.unlink()
if __name__=='__main__':main()
