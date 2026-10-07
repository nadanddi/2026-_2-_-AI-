"""Strict metadata contracts shared by prospective original PFN runner and verifier."""
import math
FEATURE_DTYPES={'column_selection_mask':'torch.bool','feature_means':'torch.float32',
    'scaler_mean':'torch.float32','scaler_std':'torch.float32','ng_non_constant_mask':'torch.bool',
    'ng_num_used_features':'torch.int64'}
ENGINE='tabpfn.inference.InferenceEngineExplicitKVCache'
ARCHITECTURE='tabpfn.architectures.tabpfn_v2.TabPFNV2'
def tensor(value,dtype,shape=None):
    assert set(value)=={'shape','dtype','sha256','device'}
    assert value['dtype']==dtype and value['device']=='cpu'
    dims=value['shape'];assert isinstance(dims,list) and dims and all(type(n) is int and n>0 for n in dims)
    if shape is not None:assert dims==shape
    h=value['sha256'];assert isinstance(h,str) and len(h)==64 and set(h)<=set('0123456789abcdef')
def fingerprint(value):
    assert isinstance(value,list) and len(value)==4
    for cache in value:
        assert set(cache)=={'train_shape','feature_statistics','kv','target_embedding'}
        assert cache['train_shape']==[1,2000]
        stats=cache['feature_statistics'];assert set(stats)==set(FEATURE_DTYPES)
        groups=stats['feature_means']['shape'][0]
        assert type(groups) is int and groups>0
        for name,dtype in FEATURE_DTYPES.items():tensor(stats[name],dtype,[groups,1 if name=='ng_num_used_features' else 2])
        tensor(cache['target_embedding'],'torch.float32',[1,192])
        assert set(cache['kv'])=={str(i) for i in range(12)}
        for layer in cache['kv'].values():
            assert set(layer)=={'key','value'}
            for v in layer.values():tensor(v,'torch.float32',[groups+1,2000,1,32])
def audit_metadata(audit,query_rows,check_keys):
    assert set(audit['checks'])==set(check_keys)
    vals=list(audit['checks'].values())
    assert vals and all(math.isfinite(v) and v>=0 for v in vals)
    assert audit['max_difference']==max(vals)<=1e-6
    fingerprint(audit['feature_cache_before']);fingerprint(audit['feature_cache_after'])
    assert audit['feature_cache_before']==audit['feature_cache_after'] and audit['feature_cache_unchanged'] is True
    trace=audit['reference_only_trace']
    assert len(trace)==120 and {v['phase'] for v in trace}=={'fit','predict'}
    fit=[v for v in trace if v['phase']=='fit'];pred=[v for v in trace if v['phase']=='predict']
    assert len(fit)==4 and len(pred)==116
    assert all(v['rows']==v['num_train_labels']==2000 and v['uses_feature_cache'] is False for v in fit)
    sizes={8,min(96,query_rows),query_rows}
    assert all(v['num_train_labels']==0 and v['uses_feature_cache'] is True and v['rows'] in sizes for v in pred)
    assert audit['constant_mask_fit_rows']==[2000]*8 and audit['group_statistics_fit_rows']==[2000]*4
    state=audit['runtime_state']
    assert state['fit_mode']=='fit_with_cache' and state['kv_cache_precision']=='auto'
    assert state['engine']==ENGINE and state['architectures']==[ARCHITECTURE]
    assert state['threads']==4 and state['interop_threads']==1
    assert state['inference_precision']=='torch.float32' and state['memory_saving_mode']=='auto'
    assert audit['heldout_truth_loaded'] is False and audit['GPU_used'] is False
