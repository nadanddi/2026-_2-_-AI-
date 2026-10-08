"""Stdlib snapshot of completed original raw predictions; no truth numbers/score."""
from pathlib import Path
import hashlib,json,math,sys
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def main():
    if not __debug__ or sys.flags.optimize:raise RuntimeError('Python -O prohibited')
    assert sys.argv[1:]==['--snapshot','2']
    regpath=HERE/'DOMAIN24_original_raw_fit_registration_v3.json'
    reg=json.loads(regpath.read_text(encoding='utf-8'))
    assert all(sha(p)==v for p,v in reg['source_sha256'].items())
    folds=json.loads((HERE/'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))['folds']
    lookup={(f['validator'],f['fold']):f for f in folds}
    root=HERE/'checkpoints/DOMAIN24_ORIGINAL_RAW_v3';files=sorted(root.glob('*_fold*/*_seed*.json'))
    assert files
    results=[]
    for path in files:
        saved=json.loads(path.read_text(encoding='utf-8'));contract=saved['contract']
        fold=lookup[contract['validator'],contract['fold']];cid=contract['candidate'];seed=contract['seed']
        assert path.name==f'{cid}_seed{seed}.json' and path.parent.name==f'{fold["validator"]}_fold{fold["fold"]}'
        assert saved['status']=='ORIGINAL_RAW_MODEL_PREDICTION_PASS_NO_SCORE'
        assert contract['registration_sha256']==sha(regpath) and seed in [47,1414,6464]
        assert cid in reg['family_map'] or cid in ['BASELINE_ET','BASELINE_LGB','BASELINE_MLP']
        assert saved['row_ids']==fold['ordered_query_ids']
        assert len(saved['pred'])==len(saved['row_ids']) and all(math.isfinite(v) for v in saved['pred'])
        assert digest(saved['pred'])==saved['pred_sha256'] and saved['heldout_truth_loaded'] is False
        assert contract['runtime_module_paths']==reg['runtime_module_paths']
        member=contract['member'];assert contract['model_contract']==reg['model_contracts'][f'{member}_seed{seed}']
        prep_path=HERE/'checkpoints/DOMAIN24_ORIGINAL_PREPARATION_v2'/f'{fold["validator"]}_fold{fold["fold"]}.json'
        assert contract['preparation_receipt_sha256']==sha(prep_path)
        prep=json.loads(prep_path.read_text(encoding='utf-8'))
        matrix=contract['matrices'];assert len(matrix['columns'])==len(set(matrix['columns']))
        if cid in reg['family_map']:
            expected=prep['candidate_matrices'][cid]
            assert matrix['train_matrix_sha256']==expected['train_sha256']
            assert matrix['query_matrix_sha256']==expected['query_sha256']
            assert matrix['columns']==expected['columns']
            assert member=='ET'
            assert saved['imputer']['all_missing_train_columns']==expected['all_missing_train_columns']
        if member in ['ET','MLP']:
            imputer=saved['imputer'];empty=imputer['all_missing_train_columns']
            assert imputer['statistics_sha256']==imputer['independent_train_median_sha256']
            assert imputer['effective_columns']==[c for c in matrix['columns'] if c not in empty]
            assert imputer['fit_reference_only'] is True
        else:assert member=='LGB' and saved['imputer']=={}
        audit=saved['audit'];errors=audit['reverse_scattered_prefix_max_differences']
        assert audit['prefix_prediction_checks']==3 and len(errors)==5
        assert all(math.isfinite(v) and v>=0 for v in errors)
        assert audit['all_errors_max']==max(errors)<=1e-6
        assert audit['fit_rows']==len(fold['ordered_train_ids']) and audit['query_rows']==len(fold['ordered_query_ids'])
        results.append({'path':str(path.relative_to(HERE)),'sha256':sha(path),'candidate':cid,'seed':seed,
                        'validator':fold['validator'],'fold':fold['fold'],'PASS':True})
    assert all(sha(p)==v for p,v in reg['source_sha256'].items())
    result={'status':'PARTIAL_ORIGINAL_RAW_METADATA_SNAPSHOT_PASS_NOT_FULL_GATE',
        'code_sha256':sha(__file__),'registration_sha256':sha(regpath),'files_checked':len(results),'files':results,
        'heldout_truth_values_read':False,'score_computed':False,
        'limits':['Snapshot only; does not prove worker completed5346fits or full matrix/model independent replay',
                  'Imputer median agreement is saved-receipt agreement, not fresh independent numerical reconstruction']}
    with (HERE/'DOMAIN24_original_raw_metadata_snapshot_v2.json').open('x',encoding='utf-8') as h:
        json.dump(result,h,ensure_ascii=False,indent=2)
    print(f'Original completed raw metadata {len(results)} PASS; partial snapshot, no scores')
if __name__=='__main__':main()

