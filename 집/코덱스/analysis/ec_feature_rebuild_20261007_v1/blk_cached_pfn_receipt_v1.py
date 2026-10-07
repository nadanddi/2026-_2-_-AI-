"""Strict current-source verifier for all four reference-only CPU PFN receipts.

No fitting, query labels, scoring, or adoption. Recomputes reference/query feature
matrix hashes using the existing causal builder, after source/ID checks.
"""
import json, math
from blk_baseline_data_v1 import *
import env_extra
import torch, tabpfn
from checkpoint_v1 import digest

CHECK_KEYS={
    'scattered96_vs_full1440','reversed8_vs_full1440','independent_single8_vs_full1440',
    'protected_first_row_other_query_poison','repeated96_after_other_checks',
    'repeated_full1440_after_other_checks',
    *[f'batch{n}_reverse{reverse}' for n in range(2,8) for reverse in [False,True]],
}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def valid_hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def verify_cached_pfn():
    folder=HERE/'checkpoints/BLK_PFN_CPU_REFONLY_v3'; rp=folder/'registration.json'
    rfolder=HERE/'checkpoints/BLK_R3_v1'; rpath=rfolder/'registration.json'
    reg=read(rp); rreg=read(rpath); complete=read(folder/'complete.json')
    assert complete['status']=='PFN_REFERENCE_ONLY_RAW_CONTEXTS_AND_AUDITS_COMPLETE'
    assert complete['registration_sha256']==sha(rp) and not complete['heldout_scored']
    assert reg['R3_registration_sha256']==sha(rpath)
    assert reg['code_sha256']==sha(HERE/'blk_pfn_refonly_cache_v3.py')
    assert reg['source_template_sha256']==sha(HERE/'blk_pfn_refonly_cache_v1.py')
    assert reg['parent_runner_sha256']==sha(HERE/'blk_pfn_refonly_cache_v2.py')
    assert reg['adapter_sha256']==sha(HERE/'blk_pfn_minbatch_v1.py')
    assert reg['fit_mode']=='fit_with_cache' and reg['kv_cache_precision']=='auto'
    assert reg['minimum_inference_rows']==8 and reg['atol']==1e-6 and reg['rtol']==0
    assert reg['model_version']=='V2' and reg['n_estimators']==4 and reg['precision']=='float32'
    assert reg['columns']==FULL==rreg['feature_columns']['PFN_NOT_EXECUTED']
    assert reg['contexts']==rreg['PFN_context_ids_NOT_EXECUTED']
    assert reg['query_ids']==rreg['ordered_query_ids'] and not reg['heldout_truth_loaded']
    assert all(sha(ROOT/p)==s for p,s in reg['library_sha256'].items())
    assert all(sha(record['path'])==record['sha256'] for record in reg['runtime_module_sources'].values())
    assert all(sha(ROOT/p)==s for p,s in rreg['dependencies'].items())
    assert all(sha(Path(env.DATA)/p)==s for p,s in rreg['sources'].items())
    assert sha(reg['weights_path'])==reg['weights_sha256']=='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
    assert reg['environment']['torch']==torch.__version__ and reg['environment']['tabpfn']==tabpfn.__version__
    assert torch.version.cuda is None and '+cpu' in torch.__version__, 'Require the registered CPU-only Torch runtime'
    assert reg['environment']['device']=='cpu'
    layout_path=HERE/'BLK_layout_v2.json'; assert sha(layout_path)==rreg['layout_sha256']
    layout=read(layout_path); ctx=BLKContext(layout)
    _,tr,table=prepare_reference(ctx); ids=reg['query_ids']; q=prepare_query(ctx,ids,table)
    assert tr.row_id.tolist()==rreg['ordered_train_ids'] and q.row_id.tolist()==ids
    assert len(ids)==len(set(ids))==1440 and len(tr)==5520
    X=tr[FULL].to_numpy(np.float32); Q=q[FULL].to_numpy(np.float32); y=tr.sub_ec.to_numpy(float)
    qhash=hashlib.sha256(Q.tobytes()).hexdigest(); train_ids=set(tr.row_id)
    paths=[rp,folder/'complete.json']; max_difference=0.; context_receipts={}
    expected_probes=[f'{b["farm"]}_{b["query_days"][i]:03d}_{h:02d}' for b in layout['blocks']
        for i in [0,len(b['query_days'])//2,len(b['query_days'])-1] for h in [0,5,12,23]]
    for seed in [5,6,7,8]:
        output=folder/f'context{seed}.json'; ap=folder/f'context{seed}_audit.json'
        saved=read(output); audit=read(ap)
        assert saved['registration_sha256']==audit['registration_sha256']==sha(rp)
        assert saved['context']==audit['context']==seed and saved['row_ids']==ids
        assert saved['fit_mode']=='fit_with_cache' and not saved['heldout_truth_loaded']
        assert len(saved['pred'])==1440 and all(math.isfinite(p) for p in saved['pred'])
        assert saved['pred_sha256']==digest(saved['pred'])
        assert audit['status']=='PASS' and audit['prediction_file_sha256']==sha(output)
        assert audit['code_sha256']==reg['code_sha256'] and audit['adapter_sha256']==reg['adapter_sha256']
        assert audit['source_template_sha256']==reg['source_template_sha256'] and audit['parent_runner_sha256']==reg['parent_runner_sha256']
        assert audit['atol']==1e-6 and audit['rtol']==0 and set(audit['checks'])==CHECK_KEYS
        assert all(math.isfinite(v) and 0<=v<=1e-6 for v in audit['checks'].values())
        assert audit['max_difference']==max(audit['checks'].values())
        assert audit['probes']==expected_probes and not audit['heldout_truth_loaded'] and not audit['GPU_used']
        calls=audit['reference_only_trace']; fits=[c for c in calls if c['phase']=='fit']; predicts=[c for c in calls if c['phase']=='predict']
        assert len(fits)==4 and len(predicts)>=104 and len(calls)==len(fits)+len(predicts)
        assert all(c['rows']==c['num_train_labels']==2000 and c['uses_feature_cache'] is False for c in fits)
        assert all(c['rows'] in [8,96,1440] and c['num_train_labels']==0 and c['uses_feature_cache'] is True for c in predicts)
        assert audit['constant_mask_fit_rows']==[2000]*8 and audit['group_statistics_fit_rows']==[2000]*4
        state=audit['runtime_state']
        assert state['fit_mode']=='fit_with_cache' and state['kv_cache_precision']=='auto'
        assert state['engine']=='tabpfn.inference.InferenceEngineExplicitKVCache'
        assert state['architectures'] and all(n=='tabpfn.architectures.tabpfn_v2.TabPFNV2' for n in state['architectures'])
        assert state['threads']==4 and state['interop_threads']==1 and state['inference_precision']=='torch.float32'
        assert state['memory_saving_mode']=='auto'
        before=audit['feature_cache_before']; assert before==audit['feature_cache_after'] and audit['feature_cache_unchanged'] is True
        assert len(before)==4
        for item in before:
            assert item['train_shape'][1]==2000 and item['kv']
            assert set(item['feature_statistics'])=={'column_selection_mask','feature_means','scaler_mean','scaler_std','ng_non_constant_mask','ng_num_used_features'}
            stats=list(item['feature_statistics'].values())+[item['target_embedding']]
            for kv in item['kv'].values():
                assert set(kv)=={'key','value'}
                assert all(v['dtype']=='torch.float32' for v in kv.values())
                stats+=list(kv.values())
            assert all(valid_hash(v['sha256']) and v['shape'] and isinstance(v['dtype'],str) for v in stats)
        ix=np.random.default_rng(seed).choice(len(tr),size=2000,replace=False)
        context_ids=tr.row_id.iloc[ix].tolist(); assert context_ids==reg['contexts'][str(seed)]
        assert len(set(context_ids))==2000 and set(context_ids)<=train_ids and not(set(context_ids)&set(ids))
        expected={'train':hashlib.sha256(X[ix].tobytes()).hexdigest(),
            'reference_labels':hashlib.sha256(y[ix].tobytes()).hexdigest(),'full_query_features':qhash}
        assert audit['input_matrix_sha256']==expected
        paths += [output,ap]; max_difference=max(max_difference,audit['max_difference'])
        context_receipts[str(seed)]={'prediction_sha256':sha(output),'audit_sha256':sha(ap),'matrix_sha256':expected}
    return {'status':'VERIFIED_REFERENCE_ONLY_PFN_RECEIPTS','registration_sha256':sha(rp),
        'receipt_files_sha256':{str(p.relative_to(HERE)):sha(p) for p in paths},
        'code_sha256':sha(__file__),'context_receipts':context_receipts,'max_batch_difference':max_difference,
        'CPU_only_runtime_verified':True,'torch_cuda_build':torch.version.cuda,'heldout_truth_loaded':False,
        'limits':['Actual original tensor device was not recorded before .cpu(); CPU-only wheel and explicit CPU model configuration are separate evidence',
            'Cached policy differs from earlier uncached policy; not a mathematical identity claim','Finite numerical probes, not exhaustive input-domain proof']}

if __name__=='__main__':
    receipt=verify_cached_pfn(); out=HERE/'BLK_cached_PFN_receipt_v1.json'
    assert not out.exists(); out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print('All four reference-only CPU PFN receipts verified; whole baseline gate still separate',flush=True)
