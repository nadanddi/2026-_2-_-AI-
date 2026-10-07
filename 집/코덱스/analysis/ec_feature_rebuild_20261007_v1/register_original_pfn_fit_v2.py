"""Register all original66 train-only PFN contexts before any PFN fitting."""
from pathlib import Path
import json,hashlib,ast,sys
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    if not __debug__ or sys.flags.optimize:raise RuntimeError('Python -O prohibited')
    out=HERE/'DOMAIN24_original_pfn_registration_v2.json';assert not out.exists()
    rawpath=HERE/'DOMAIN24_original_raw_fit_registration_v3.json';raw=read(rawpath)
    assert raw['status']=='REGISTERED_ORIGINAL66_R3_DOMAIN24_RAW_FIT_BEFORE_FIT'
    assert raw['candidate_fit_count']==4752 and raw['baseline_fit_count']==594
    sources=dict(raw['source_sha256']);assert all(sha(p)==v for p,v in sources.items())
    planpath=HERE/'DOMAIN24_original_pfn_runtime_plan_v2.json';plan=read(planpath)
    assert plan['status']=='ORIGINAL_PFN_CONTEXTS_AND_RUNTIME_PLAN_CAPTURED_NO_FIT_REGISTRATION'
    assert plan['code_sha256']==sha(HERE/'capture_original_pfn_plan_v2.py')
    assert plan['runner_sha256']==sha(HERE/'run_original_pfn_cache_v2.py')
    assert plan['model_fits']==plan['query_predictions']==0 and plan['target_values_read'] is False
    assert plan['context_seeds']==[5,6,7,8] and plan['context_fits']==264 and plan['context_rows']==2000
    registry=read(HERE/'original_validator_endpoint_registry_v1.json')
    assert len(registry['folds'])==66
    assert plan['registry_sha256']==sha(HERE/'original_validator_endpoint_registry_v1.json')
    names={f'{f["validator"]}_fold{f["fold"]}' for f in registry['folds']}
    assert set(plan['contexts'])==names and len(plan['columns'])==38 and len(set(plan['columns']))==38
    for f in registry['folds']:
        train=set(f['ordered_train_ids']);query=set(f['ordered_query_ids']);gap=set(f['input_forbidden_ids'])
        assert len(train)>=2000 and len(query)>=8 and not(train&query or train&gap or query&gap)
        assert set(plan['contexts'][f'{f["validator"]}_fold{f["fold"]}'])=={'5','6','7','8'}
        for ids in plan['contexts'][f'{f["validator"]}_fold{f["fold"]}'].values():
            assert len(ids)==len(set(ids))==2000 and set(ids)<=train and not(set(ids)&(query|gap))
    assert plan['fit_mode']=='fit_with_cache' and plan['kv_cache_precision']=='auto'
    assert plan['model_version']=='V2' and plan['n_estimators']==4 and plan['precision']=='float32'
    assert plan['minimum_inference_rows']==8 and plan['atol']==1e-6 and plan['rtol']==0
    assert sha(plan['weights_path'])==plan['weights_sha256']
    audit=read(HERE/'DOMAIN24_original_pfn_rules_synthetic_audit_v1.json')
    assert audit['status']=='ORIGINAL_PFN_STRICT_METADATA_SYNTHETIC_PASS'
    assert audit['code_sha256']==sha(HERE/'audit_original_pfn_rules_v1.py')
    assert audit['rules_sha256']==sha(HERE/'original_pfn_cache_rules_v1.py')
    assert len(audit['tests'])==21 and all(t['PASS'] for t in audit['tests'])
    assert audit['model_fit'] is False and audit['PFN_import'] is False and audit['target_values_read'] is False
    for p,v in plan['source_sha256'].items():
        assert p not in sources or sources[p]==v
        sources[p]=v
    extras=['DOMAIN24_original_raw_fit_registration_v3.json','DOMAIN24_original_pfn_runtime_plan_v2.json',
            'capture_original_pfn_plan_v2.py','original_pfn_cache_rules_v1.py','run_original_pfn_cache_v2.py',
            'upgrade_original_pfn_runner_v2.py','audit_original_pfn_rules_v1.py',
            'DOMAIN24_original_pfn_rules_synthetic_audit_v1.json','register_original_pfn_fit_v2.py',
            'critique_ORIGINAL_PFN_cache_plan_v1.md','critique_ORIGINAL_PFN_cache_plan_v2.md']
    for n in extras:
        p=HERE/n;key=str(p.resolve());value=sha(p)
        assert key not in sources or sources[key]==value
        sources[key]=value
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
    assert all(sha(p)==v for p,v in sources.items())
    result={**plan,'status':'REGISTERED_ORIGINAL66_PFN_REFERENCE_ONLY_BEFORE_FIT',
        'raw_fit_registration_sha256':sha(rawpath),'source_sha256':sources,
        'registrar_sha256':sha(__file__),'output_file_count':528,'whole_baseline_complete':False,
        'heldout_truth_loaded':False,'adoption_permitted':False,'GPU_used':False,
        'model_arguments':{'device':'cpu','n_estimators':4,'inference_precision':'torch.float32',
            'n_preprocessing_jobs':1,'ignore_pretraining_limits':True,'fit_mode':'fit_with_cache',
            'kv_cache_precision':'auto','memory_saving_mode':'auto'},
        'history':'New CPU reference-only cached baseline, no historical GPU/uncached equality claim',
        'limits':['264 PFN raw contexts only; full R3 mix/shrink/clip/SG2 gates before any score',
                  'Three sampled prefix checks do not prove all-row causality','All24 original candidates remain mandatory']}
    with out.open('x',encoding='utf-8') as h:json.dump(result,h,ensure_ascii=False,indent=2,allow_nan=False)
    print(f'Original PFN264 train-only contexts registered {len(sources)} pins; no fit execution')
if __name__=='__main__':main()
