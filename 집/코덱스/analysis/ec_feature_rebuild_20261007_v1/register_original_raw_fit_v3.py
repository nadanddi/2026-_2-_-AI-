"""Seal original66 raw-fit contracts only after complete preparation and independent checks."""
from pathlib import Path
import hashlib,json,ast
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def main():
    out=HERE/'DOMAIN24_original_raw_fit_registration_v3.json'
    assert not out.exists()
    prepreg=read(HERE/'DOMAIN24_original_preparation_registration_v2.json')
    folder=HERE/'checkpoints/DOMAIN24_ORIGINAL_PREPARATION_v2'
    complete=read(folder/'complete.json')
    prepcheck=read(HERE/'DOMAIN24_original_preparation_independent_crosscheck_v1.json')
    assert complete['status']=='ORIGINAL66_FEATURES_PREPARED_NOT_MODELS_VALIDATED'
    assert complete['folds']==66 and complete['prefix_checks']==396
    assert complete['registration_sha256']==sha(HERE/'DOMAIN24_original_preparation_registration_v2.json')
    assert prepcheck['status']=='INDEPENDENT_ORIGINAL66_PREPARATION_METADATA_PASS'
    assert prepcheck['complete_sha256']==sha(folder/'complete.json')
    assert prepcheck['folds_checked']==66 and prepcheck['candidate_matrix_signatures_checked']==1584
    registry=read(HERE/'original_validator_endpoint_registry_v1.json')
    assert len(registry['folds'])==66
    expected={f'{f["validator"]}_fold{f["fold"]}.json' for f in registry['folds']}
    assert set(complete['files_sha256'])==expected
    assert all(sha(folder/n)==v for n,v in complete['files_sha256'].items())
    runtime=read(HERE/'DOMAIN24_original_runtime_contract_v1.json')
    assert runtime['status']=='ACTUAL_ORIGINAL_MODEL_RUNTIME_CONSTRUCTORS_CAPTURED_NO_FIT'
    assert runtime['model_fits']==runtime['query_predictions']==0
    assert set(runtime['model_contracts'])=={f'{n}_seed{s}' for n in ['ET','LGB','MLP'] for s in [47,1414,6464]}
    assert runtime['runner_sha256']==sha(HERE/'run_domain_original_raw_v3.py')
    synthetic=read(HERE/'DOMAIN24_original_raw_resume_synthetic_audit_v1.json')
    assert synthetic['runner_sha256']==runtime['runner_sha256']
    assert len(synthetic['tests'])==13 and all(t['PASS'] for t in synthetic['tests'])
    assert synthetic['model_fits']==synthetic['real_query_predictions']==synthetic['real_heldout_labels_read']==0
    stats=read(HERE/'DOMAIN24_original_statistics_registration_v2.json')
    statscheck=read(HERE/'DOMAIN24_original_statistics_independent_crosscheck_v1.json')
    assert statscheck['status']=='INDEPENDENT_ORIGINAL_STATISTICS_DRAW_AND_SYNTHETIC_PASS'
    assert statscheck['registration_sha256']==sha(HERE/'DOMAIN24_original_statistics_registration_v2.json')
    assert statscheck['draws_replayed']==200000 and len(statscheck['synthetic_tests'])==8
    assert stats['draws']==200000 and stats['comparisons']==84 and stats['alpha']==.025/84
    sources=dict(prepreg['sources_sha256'])
    for group in [runtime['source_sha256'],stats['source_sha256']]:
        for p,v in group.items():
            assert p not in sources or sources[p]==v
            sources[p]=v
    extras=['DOMAIN24_original_preparation_registration_v2.json',
        'DOMAIN24_original_preparation_independent_crosscheck_v1.json','crosscheck_original_preparation_v1.py',
        'original_fold_features_v2.py','run_domain_original_raw_v3.py','upgrade_original_raw_runner_v3.py',
        'audit_original_raw_resume_v1.py','DOMAIN24_original_raw_resume_synthetic_audit_v1.json',
        'capture_original_runtime_v1.py','DOMAIN24_original_runtime_contract_v1.json',
        'DOMAIN24_original_statistics_registration_v2.json','DOMAIN24_original_bootstrap_draws_v2.bin',
        'crosscheck_original_statistics_v1.py','DOMAIN24_original_statistics_independent_crosscheck_v1.json',
        'critique_DOMAIN24_original_statistics_v1.md','critique_DOMAIN24_original_raw_plan_v2.md',
        'register_original_raw_fit_v3.py']
    for name in extras:
        p=HERE/name;sources[str(p.resolve())]=sha(p)
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
    for name,v in complete['files_sha256'].items():sources[str((folder/name).resolve())]=v
    sources[str((folder/'complete.json').resolve())]=sha(folder/'complete.json')
    assert all(sha(p)==v for p,v in sources.items())
    recipe={'R3_weights':[.6,.3,.1],'R3_total_weight':.6,'PFN_total_weight':.4,
        'PFN_context_seeds':[5,6,7,8],'PFN_reference_rows_per_context':2000,'PFN_estimators':4,
        'PFN_policy':'Official reference-only fit_with_cache, same local v2 weights; no query stats fits',
        'shrink':'Once, .5 current + .5 same farm/day prediction prefix mean',
        'clip':'Fold train-label min/max before and after SG2',
        'SG2':'RefOnlySG2, original rowday>=179 gate (BLK_RAW_PASS policy), no fake query-role forcing',
        'candidate_change':'One domain family added to ET only; LGB/MLP/PFN fixed to that fold baseline',
        'full_score_gate':'All original cached PFN/R3 and full mixed postprocessing causal/numeric gates plus independent audit BEFORE any heldout truth parsing',
        'historical_GPU_equivalence_claimed':False}
    result={'status':'REGISTERED_ORIGINAL66_R3_DOMAIN24_RAW_FIT_BEFORE_FIT',
        'runner_sha256':runtime['runner_sha256'],'source_sha256':sources,
        'preparation_complete_sha256':sha(folder/'complete.json'),
        'registry_sha256':sha(HERE/'original_validator_endpoint_registry_v1.json'),
        'fold_roster':[{'validator':f['validator'],'fold':f['fold'],
                       'train_rows':len(f['ordered_train_ids']),'query_rows':len(f['ordered_query_ids'])} for f in registry['folds']],
        'seeds':[47,1414,6464],'candidate_cap':24,'family_map':prepreg['family_map'],
        'candidate_fit_count':4752,'baseline_fit_count':594,'environment':runtime['environment'],
        'runtime_module_paths':runtime['runtime_module_paths'],'model_contracts':runtime['model_contracts'],
        'statistics_registration_sha256':sha(HERE/'DOMAIN24_original_statistics_registration_v2.json'),
        'statistics_draws_sha256':stats['draw_file_sha256'],
        'statistics_crosscheck_sha256':sha(HERE/'DOMAIN24_original_statistics_independent_crosscheck_v1.json'),
        'performance_read_before_fit':False,'performance_read_scope':'Original-domain full-model heldout scores only; earlier BLK exposed',
        'full_model_recipe':recipe,'heldout_score_permitted':False,'adoption_permitted':False,
        'CPU_policy':{'threadpool_limit':1,'ET_fit_jobs':4,'ET_predict_jobs':1,'LGB_jobs':4},
        'limits':['Raw stage does not include PFN/postprocess or allow score/adoption',
                  'All24 original families mandatory regardless BLK rank',
                  'Same known labels; final firstunused seed/layout one-shot rule still pending']}
    with out.open('x',encoding='utf-8') as h:json.dump(result,h,ensure_ascii=False,indent=2,allow_nan=False)
    print(f'Original66 raw fits registered, {len(sources)} pins, 4752+594 fit cells; no fit execution')
if __name__=='__main__':main()
