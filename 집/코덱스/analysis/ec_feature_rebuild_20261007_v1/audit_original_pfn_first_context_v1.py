"""Actual first original PFN receipt metadata, no model import or query truth."""
from pathlib import Path
import hashlib
import json
import math
import sys
import original_pfn_cache_rules_v1 as rules

HERE = Path(__file__).resolve().parent
CHECK_KEYS = {'scattered_vs_full', 'reversed8_vs_full', 'independent_single8_vs_full',
              'protected_first_row_other_query_poison', 'repeated_scattered_after_other_checks',
              'repeated_full_after_other_checks',
              *[f'batch{n}_reverse{rev}' for n in range(2, 8) for rev in (False, True)],
              *[f'fresh_prefix_probe{i}' for i in range(3)]}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def main():
    if not __debug__ or sys.flags.optimize: raise RuntimeError('Python -O prohibited')
    driverpath = HERE / 'DOMAIN24_original_pfn_driver_registration_v3.json'
    driver = json.loads(driverpath.read_text(encoding='utf-8'))
    assert driver['status'] == 'REGISTERED_PFN_DRIVER_PATH_FIX_NO_MODEL_CHANGE'
    assert all(sha(p) == v for p, v in driver['sources_sha256'].items())
    regpath = HERE / 'DOMAIN24_original_pfn_registration_v2.json'
    rawpath = HERE / 'DOMAIN24_original_raw_fit_registration_v3.json'
    reg = json.loads(regpath.read_text(encoding='utf-8'))
    assert driver['producer_registration_sha256'] == sha(regpath)
    registry = json.loads((HERE / 'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    fold = next(f for f in registry['folds'] if f['validator'] == 'DIAG10' and f['fold'] == 0)
    ids = fold['ordered_query_ids']
    context = reg['contexts']['DIAG10_fold0']['5']
    assert len(context) == len(set(context)) == 2000
    assert set(context) <= set(fold['ordered_train_ids'])
    assert not set(context).intersection(set(ids) | set(fold['input_forbidden_ids']))
    folder = HERE / 'checkpoints/DOMAIN24_ORIGINAL_PFN_CACHE_v2/DIAG10_fold0'
    prediction_path = folder / 'context5.json'
    audit_path = folder / 'context5_audit.json'
    pred_blob, audit_blob = prediction_path.read_bytes(), audit_path.read_bytes()
    pred_sha, audit_sha = hashlib.sha256(pred_blob).hexdigest(), hashlib.sha256(audit_blob).hexdigest()
    saved, audit = json.loads(pred_blob), json.loads(audit_blob)
    contract = saved['contract']
    assert saved['status'] == 'ORIGINAL_PFN_REFERENCE_ONLY_RAW_PASS_NO_SCORE'
    assert saved['row_ids'] == ids and saved['heldout_truth_loaded'] is False
    assert len(saved['pred']) == len(ids)
    assert all(type(p) is float and math.isfinite(p) for p in saved['pred'])
    assert saved['pred_sha256'] == digest(saved['pred'])
    assert contract['registration_sha256'] == sha(regpath)
    assert contract['raw_fit_registration_sha256'] == sha(rawpath)
    assert contract['validator'] == 'DIAG10' and contract['fold'] == 0 and contract['context_seed'] == 5
    assert contract['columns'] == reg['columns']
    assert contract['context_ids_sha256'] == digest(context)
    assert contract['ordered_query_ids_sha256'] == digest(ids)
    assert contract['prepared_fold_sha256'] == sha(HERE / 'checkpoints/DOMAIN24_ORIGINAL_PREPARATION_v2/DIAG10_fold0.json')
    for name in ('train_context_matrix_sha256', 'reference_labels_sha256', 'full_query_matrix_sha256'):
        value = contract[name]
        assert type(value) is str and len(value) == 64 and set(value) <= set('0123456789abcdef')
    assert audit['status'] == 'PASS' and audit['contract'] == contract
    assert audit['prediction_file_sha256'] == pred_sha
    assert audit['code_sha256'] == sha(HERE / 'run_original_pfn_cache_v2.py')
    assert audit['auditor_sha256'] == sha(HERE / 'original_pfn_cache_rules_v1.py')
    rules.audit_metadata(audit, len(ids), CHECK_KEYS)
    assert prediction_path.read_bytes() == pred_blob and audit_path.read_bytes() == audit_blob
    assert all(sha(p) == v for p, v in driver['sources_sha256'].items())
    result = {'status': 'FIRST_ORIGINAL_PFN_CONTEXT_METADATA_PASS_NOT_FULL_MODEL_GATE',
              'prediction_sha256': pred_sha, 'audit_sha256': audit_sha,
              'driver_registration_sha256': sha(driverpath), 'producer_registration_sha256': sha(regpath),
              'query_rows': len(ids), 'reference_context_rows': 2000,
              'numeric_checks': len(CHECK_KEYS), 'max_difference': audit['max_difference'],
              'preprocessing_fit_trace_entries': 4, 'prediction_trace_entries': 116,
              'cache_unchanged': True, 'GPU_used': False, 'heldout_truth_read': False,
              'model_fit': False, 'whole_pipeline_gate_passed': False,
              'code_sha256': sha(__file__),
              'limits': ['One completed context only, not original264 completion',
                         'Saved matrix/label digests checked structurally, not fresh numerical reconstruction',
                         'No independent model refit or mixed postprocessing/score gate']}
    with (HERE / 'original_pfn_first_context_metadata_audit_v1.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(f'First original PFN context metadata PASS: {len(ids)} rows,21checks,120trace; no full gate')


if __name__ == '__main__': main()
