"""Independent stdlib audit of all original preparation receipts; no matrices or scores."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()

def main():
    if not __debug__ or sys.flags.optimize:
        raise RuntimeError('Optimized execution prohibited')
    regpath = HERE / 'DOMAIN24_original_preparation_registration_v2.json'
    reg = json.loads(regpath.read_text(encoding='utf-8'))
    for path, expected in reg['sources_sha256'].items():
        assert sha(path) == expected, path
    registry_path = HERE / 'original_validator_endpoint_registry_v1.json'
    assert sha(registry_path) == reg['registry_sha256']
    folds = json.loads(registry_path.read_text(encoding='utf-8'))['folds']
    assert len(folds) == 66 and len(reg['family_map']) == 24
    folder = HERE / 'checkpoints/DOMAIN24_ORIGINAL_PREPARATION_v2'
    complete_path = folder / 'complete.json'
    complete = json.loads(complete_path.read_text(encoding='utf-8'))
    assert complete['status'] == 'ORIGINAL66_FEATURES_PREPARED_NOT_MODELS_VALIDATED'
    assert complete['registration_sha256'] == sha(regpath)
    assert complete['folds'] == 66 and complete['prefix_checks'] == 396
    assert complete['heldout_truth_loaded'] is False
    assert complete['model_fit'] is False and complete['performance_evaluated'] is False
    expected_files = {f'{f["validator"]}_fold{f["fold"]}.json' for f in folds}
    assert set(complete['files_sha256']) == expected_files
    assert {p.name for p in folder.glob('*_fold*.json')} == expected_files
    checks = 0
    candidates = 0
    summaries = []
    for fold in folds:
        name = f'{fold["validator"]}_fold{fold["fold"]}.json'
        path = folder / name
        assert sha(path) == complete['files_sha256'][name]
        record = json.loads(path.read_text(encoding='utf-8'))
        payload = dict(record)
        signature = payload.pop('payload_sha256')
        assert canonical(payload) == signature
        assert record['registration_sha256'] == sha(regpath)
        assert record['status'] == 'PREPARED_FEATURES_NOT_FIT_OR_SCORE'
        assert record['validator'] == fold['validator'] and record['fold'] == fold['fold']
        for key in ('ordered_train_ids', 'ordered_query_ids', 'input_forbidden_ids'):
            assert record[key] == fold[key]
            assert len(record[key]) == len(set(record[key]))
        train, query, forbidden = (set(record[k]) for k in ('ordered_train_ids', 'ordered_query_ids', 'input_forbidden_ids'))
        assert not train & query and not train & forbidden and not query & forbidden
        assert record['heldout_truth_loaded'] is False
        assert record['model_fit'] is False and record['performance_evaluated'] is False
        assert record['domain_columns'] == 1187 and record['prefix_checks'] == 6
        assert set(record['candidate_matrices']) == set(reg['family_map'])
        for cid, matrix in record['candidate_matrices'].items():
            extra = reg['family_map'][cid]['additional_ET_columns']
            assert matrix['columns'][-len(extra):] == extra if extra else True
            assert len(matrix['columns']) == len(set(matrix['columns']))
            assert set(matrix) == {'columns', 'train_sha256', 'query_sha256', 'all_missing_train_columns'}
            for key in ('train_sha256', 'query_sha256'):
                value = matrix[key]
                assert len(value) == 64 and set(value) <= set('0123456789abcdef')
            empty = matrix['all_missing_train_columns']
            assert len(empty) == len(set(empty)) and set(empty) <= set(matrix['columns'])
            candidates += 1
        checks += record['prefix_checks']
        summaries.append({'validator': fold['validator'], 'fold': fold['fold'],
                          'train_rows': len(train), 'query_rows': len(query), 'receipt_sha256': sha(path)})
    assert checks == 396 and candidates == 1584
    assert all(sha(p) == value for p, value in reg['sources_sha256'].items())
    result = {'status': 'INDEPENDENT_ORIGINAL66_PREPARATION_METADATA_PASS',
              'code_sha256': sha(__file__), 'registration_sha256': sha(regpath),
              'complete_sha256': sha(complete_path), 'source_pins_checked': len(reg['sources_sha256']),
              'folds_checked': 66, 'candidate_matrix_signatures_checked': candidates,
              'reported_sampled_prefix_checks': checks, 'folds': summaries,
              'model_fit': False, 'heldout_score_loaded': False,
              'limits': ['Receipt integrity only; does not independently reconstruct matrices or repeat prefix tests',
                         'Production fit must reconstruct all24 matrices in each fold and compare hashes']}
    out = HERE / 'DOMAIN24_original_preparation_independent_crosscheck_v1.json'
    with out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print('Independent stdlib original66 preparation metadata PASS; 1584 signatures, no fit/score')

if __name__ == '__main__':
    main()
