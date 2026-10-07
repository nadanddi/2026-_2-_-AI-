"""Synthetic strict-cache/trace metadata tests; no PFN import, model or labels."""
from pathlib import Path
import copy,json,hashlib
import original_pfn_cache_rules_v1 as rules
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=HERE/'checkpoints/BLK_PFN_CPU_REFONLY_v3/context5_audit.json'
old=json.loads(parent.read_text(encoding='utf-8'))
audit=copy.deepcopy(old)
for footprint in ['feature_cache_before','feature_cache_after']:
    for cache in audit[footprint]:
        for v in cache['feature_statistics'].values():v['device']='cpu'
        cache['target_embedding']['device']='cpu'
        for layer in cache['kv'].values():
            for v in layer.values():v['device']='cpu'
keys={str(i) for i in range(21)};audit['checks']=dict.fromkeys(keys,0.);audit['max_difference']=0.
audit['reference_only_trace']=[{'phase':'fit','rows':2000,'num_train_labels':2000,'uses_feature_cache':False} for _ in range(4)]
audit['reference_only_trace'] += [{'phase':'predict','rows':24,'num_train_labels':0,'uses_feature_cache':True} for _ in range(116)]
rules.audit_metadata(audit,24,keys)
tests=[{'case':'valid_synthetic_contract_with_prior_cache_shapes','PASS':True}]
bad_cases={}
bad=copy.deepcopy(audit);bad['feature_cache_before']=bad['feature_cache_after']=[];bad_cases['empty_identical_cache']=bad
bad=copy.deepcopy(audit);bad['feature_cache_before'].pop();bad_cases['missing_cache']=bad
bad=copy.deepcopy(audit);bad['feature_cache_before'][0]['train_shape']=[1,1999];bad_cases['wrong_train_shape']=bad
bad=copy.deepcopy(audit);bad['feature_cache_before'][0]['feature_statistics'].pop('scaler_std');bad_cases['missing_stat']=bad
bad=copy.deepcopy(audit);bad['feature_cache_before'][0]['kv'].pop('11');bad_cases['missing_KV_layer']=bad
bad=copy.deepcopy(audit);bad['feature_cache_before'][0]['kv']['0']['key']['dtype']='torch.float64';bad_cases['wrong_KV_dtype']=bad
bad=copy.deepcopy(audit);bad['feature_cache_before'][0]['kv']['0']['key']['device']='cuda:0';bad_cases['original_GPU_device']=bad
bad=copy.deepcopy(audit);bad['feature_cache_before'][0]['target_embedding']['sha256']='x'*64;bad_cases['invalid_hash']=bad
bad=copy.deepcopy(audit);bad['reference_only_trace'].append({'phase':'unknown'});bad_cases['unknown_extra_phase']=bad
bad=copy.deepcopy(audit);bad['reference_only_trace'][-1]['num_train_labels']=1;bad_cases['query_labels']=bad
bad=copy.deepcopy(audit);bad['reference_only_trace'][-1]['uses_feature_cache']=False;bad_cases['query_refit_flag']=bad
bad=copy.deepcopy(audit);bad['reference_only_trace'][-1]['rows']=9;bad_cases['unexpected_batch_rows']=bad
bad=copy.deepcopy(audit);bad['reference_only_trace'].pop();bad_cases['missing_predict_trace']=bad
bad=copy.deepcopy(audit);bad['runtime_state']['architectures']*=4;bad_cases['wrong_architecture_count']=bad
bad=copy.deepcopy(audit);bad['runtime_state']['engine']='other.InferenceEngineExplicitKVCache';bad_cases['engine_suffix_only']=bad
bad=copy.deepcopy(audit);bad['runtime_state']['memory_saving_mode']=False;bad_cases['wrong_memory_policy']=bad
bad=copy.deepcopy(audit);bad['checks'].pop('20');bad_cases['missing_numerical_check']=bad
bad=copy.deepcopy(audit);bad['checks']['20']=float('nan');bad_cases['nonfinite_difference']=bad
bad=copy.deepcopy(audit);bad['constant_mask_fit_rows'][0]=2001;bad_cases['wrong_fit_rows']=bad
bad=copy.deepcopy(audit);bad['heldout_truth_loaded']=True;bad_cases['heldout_flag']=bad
for name,bad in bad_cases.items():
    try:rules.audit_metadata(bad,24,keys)
    except (AssertionError,KeyError,TypeError):tests.append({'case':name+'_rejected','PASS':True})
    else:raise AssertionError(name+' accepted')
record={'status':'ORIGINAL_PFN_STRICT_METADATA_SYNTHETIC_PASS','tests':tests,
        'code_sha256':sha(__file__),'rules_sha256':sha(rules.__file__),'parent_metadata_sha256':sha(parent),
        'PFN_import':False,'model_fit':False,'real_query_predictions':0,'target_values_read':False,
        'limits':['Synthetic trace/check values; cache shapes copied from earlier audited BLK metadata',
                  'Not evidence of actual original66 PFN fit or numerical accuracy']}
with (HERE/'DOMAIN24_original_pfn_rules_synthetic_audit_v1.json').open('x',encoding='utf-8') as h:
    json.dump(record,h,ensure_ascii=False,indent=2)
print(f'Original PFN strict metadata synthetic {len(tests)} PASS; no PFN model/truth')
