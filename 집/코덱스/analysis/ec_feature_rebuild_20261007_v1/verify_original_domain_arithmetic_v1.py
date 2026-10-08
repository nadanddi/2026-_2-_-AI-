"""Full original66 saved-stage Decimal replay; conditional gate, no scores."""
from pathlib import Path
import hashlib
import json
import sys
import original_completion_receipt_v2 as completion
import original_independent_arithmetic_v1 as arithmetic

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_sources(reg):
    assert all(sha(p) == value for p, value in reg['sources_sha256'].items())


def stable_json(path, expected_sha):
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_sha
    value = json.loads(data.decode('utf-8'))
    assert Path(path).read_bytes() == data
    return value


def main():
    if not __debug__ or sys.flags.optimize: raise RuntimeError('Python -O prohibited')
    regpath = HERE/'ORIGINAL_DOMAIN24_arithmetic_registration_v1.json'
    reg = json.loads(regpath.read_text(encoding='utf-8'))
    assert reg['status'] == 'REGISTERED_ORIGINAL66_CONDITIONAL_ARITHMETIC_NO_SCORE'
    assert reg['verifier_sha256'] == sha(__file__)
    pinned_sources(reg)
    for module,name in ((arithmetic,'original_independent_arithmetic_v1.py'),
                        (completion,'original_completion_receipt_v2.py')):
        path = Path(module.__file__).resolve()
        assert path == (HERE/name).resolve() and sha(path) == reg['sources_sha256'][str(path)]
    registry = json.loads((HERE/'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    rawregpath = HERE/'DOMAIN24_original_raw_fit_registration_v3.json'
    pfnregpath = HERE/'DOMAIN24_original_pfn_registration_v2.json'
    rawreg = json.loads(rawregpath.read_text(encoding='utf-8'))
    rawroot = HERE/'checkpoints/DOMAIN24_ORIGINAL_RAW_v3'
    pfnroot = HERE/'checkpoints/DOMAIN24_ORIGINAL_PFN_CACHE_v2'
    assembled = HERE/'checkpoints/DOMAIN24_ORIGINAL_ASSEMBLED_v3'
    rawmanifest = completion.verify_all(rawroot,registry,rawreg,'raw',sha(rawregpath))
    pfnmanifest = completion.verify_all(pfnroot,registry,rawreg,'pfn',sha(pfnregpath))
    sealed_path = assembled/'complete.json'; sealed_sha = sha(sealed_path)
    sealed = stable_json(sealed_path,sealed_sha)
    assert sealed['status'] == 'ORIGINAL66_DOMAIN24_ASSEMBLED_NOT_FULL_SCORE_GATE'
    assert sealed['whole_pipeline_gate_passed'] is False and sealed['heldout_truth_loaded'] is False
    assembly_reg_sha = sha(HERE/'ORIGINAL_DOMAIN24_pipeline_registration_v3.json')
    assert sealed['registration_sha256'] == assembly_reg_sha
    assert sealed['raw_completion_sha256'] == rawmanifest['completion_sha256']
    assert sealed['pfn_completion_sha256'] == pfnmanifest['completion_sha256']
    expected = {f'{fold["validator"]}_fold{fold["fold"]}/{suffix}.json'
                for fold in registry['folds'] for suffix in ('predictions','audit')}
    assert len(expected) == 132
    completion.verify_manifest(sealed['files_sha256'], expected,
                               lambda name: completion.contained_path(assembled,name).read_bytes())
    result = {}; comparisons = 0; maxima = {s:0. for s in arithmetic.STAGES}
    for fold in registry['folds']:
        pinned_sources(reg)
        name = f'{fold["validator"]}_fold{fold["fold"]}'
        saved = stable_json(assembled/name/'predictions.json',sealed['files_sha256'][name+'/predictions.json'])
        audit = stable_json(assembled/name/'audit.json',sealed['files_sha256'][name+'/audit.json'])
        ids = fold['ordered_query_ids']
        assert saved['row_ids'] == ids and saved['validator'] == fold['validator'] and saved['fold'] == fold['fold']
        assert saved['pipeline_registration_sha256'] == audit['registration_sha256'] == assembly_reg_sha
        assert audit['predictions_sha256'] == sealed['files_sha256'][name+'/predictions.json']
        assert audit['status'] == 'ORIGINAL_FOLD_ASSEMBLY_COMPLETE_NOT_SCORE_GATE'
        assert saved['whole_pipeline_gate_passed'] is audit['whole_pipeline_gate_passed'] is False
        assert saved['heldout_truth_loaded'] is audit['heldout_truth_loaded'] is False
        choices = audit['choice_snapshot']['choices']
        assert audit['choice_snapshot']['row_ids'] == ids
        assert audit['choice_snapshot']['source_checks'] == 3*len(ids)
        assert len(saved['stage_outputs']) == 75
        assert set(saved['baseline']) == set(saved['candidates']) == {'47','1414','6464'}
        train_days = {arithmetic.row_key(rid)[:2] for rid in fold['ordered_train_ids']}
        assert not train_days.intersection({arithmetic.row_key(rid)[:2] for rid in ids})
        def load_pred(base,manifest,cid):
            relative = name+'/'+cid+'.json'
            obj = stable_json(base/relative,manifest['files_sha256'][relative])
            assert obj['row_ids'] == ids and obj['heldout_truth_loaded'] is False
            return obj['pred']
        pfns = {s:load_pred(pfnroot,pfnmanifest,f'context{s}') for s in (5,6,7,8)}
        fold_results = {}
        for seed in (47,1414,6464):
            lgb = load_pred(rawroot,rawmanifest,f'BASELINE_LGB_seed{seed}')
            mlp = load_pred(rawroot,rawmanifest,f'BASELINE_MLP_seed{seed}')
            assert set(saved['candidates'][str(seed)]) == set(rawreg['family_map'])
            for cid in ('BASELINE_ET',*sorted(rawreg['family_map'])):
                tag = f'{cid}_seed{seed}'
                et = load_pred(rawroot,rawmanifest,tag)
                stages = saved['stage_outputs'][tag]
                final = saved['baseline'][str(seed)] if cid == 'BASELINE_ET' else saved['candidates'][str(seed)][cid]
                assert final == stages['post_SG2_clip']
                check = arithmetic.verify(ids,et,lgb,mlp,pfns,saved['training_label_bounds'],
                                          choices,stages,train_days)
                fold_results[tag] = check
                comparisons += sum(check['stage_comparisons'].values())
                for stage,value in check['maximum_difference'].items(): maxima[stage] = max(maxima[stage],value)
        assert len(fold_results) == 75
        result[name] = fold_results
        print(name,'conditional Decimal arithmetic75 PASS; not full model gate',flush=True)
    assert len(result) == 66
    assert comparisons == sum(len(f['ordered_query_ids']) for f in registry['folds'])*75*4
    for base, manifest in ((rawroot,rawmanifest),(pfnroot,pfnmanifest)):
        assert all(sha(base/n) == v for n,v in manifest['files_sha256'].items())
        assert sha(base/'complete.json') == manifest['completion_sha256']
        assert all(sha(base/n/'complete.json') == v for n,v in manifest['fold_completion_sha256'].items())
    assert sha(sealed_path) == sealed_sha
    assert all(sha(assembled/n) == v for n,v in sealed['files_sha256'].items())
    pinned_sources(reg)
    out = {'status':'ORIGINAL66_CONDITIONAL_DECIMAL_ARITHMETIC_PASS_NOT_FULL_GATE',
           'fold_results':result,'comparisons':comparisons,'maximum_difference':maxima,
           'registration_sha256':sha(regpath),'assembly_completion_sha256':sealed_sha,
           'heldout_truth_read':False,'model_fit':False,'whole_pipeline_gate_passed':False,
           'limits':['Bounds and SG2 choices supplied by producer, not independently selected here',
                     'Fresh input/label lineage and query-input causality require separate full verifier',
                     'No model refit or historical GPU numerical identity proof']}
    with (HERE/'ORIGINAL_DOMAIN24_arithmetic_result_v1.json').open('x',encoding='utf-8') as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print('Full original66 conditional arithmetic PASS; scoring still closed')


if __name__ == '__main__': main()
