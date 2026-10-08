"""Bind saved predictions to fresh original-fold matrices; never fit or score.

The caller must register sources and verify complete manifests independently.
Saved producer numeric audits do not substitute for an independent model refit
or the full mixed prediction/SG2 causal gate.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import run_domain_original_raw_v3 as raw_producer
import original_pfn_cache_rules_v1 as pfn_rules
from checkpoint_v1 import digest
from blk_baseline_data_v1 import HERE, FULL, FULL_R3, BASE_R3, np, sha

PFN_KEYS = {'scattered_vs_full', 'reversed8_vs_full', 'independent_single8_vs_full',
            'protected_first_row_other_query_poison', 'repeated_scattered_after_other_checks',
            'repeated_full_after_other_checks',
            *[f'batch{n}_reverse{rev}' for n in range(2, 8) for rev in (False, True)],
            *[f'fresh_prefix_probe{i}' for i in range(3)]}


def require(condition, message):
    if not condition: raise ValueError(message)


def matrix_sha(frame):
    values = np.array(frame, dtype='<f8', order='C', copy=True)
    values[np.isnan(values)] = np.nan
    return hashlib.sha256(values.tobytes()).hexdigest()


def numeric_sha(values):
    array = np.array(values, dtype='<f8', order='C', copy=True)
    array[np.isnan(array)] = np.nan
    return hashlib.sha256(array.tobytes()).hexdigest()


def read_prediction(path, ids, status):
    path = Path(path)
    blob = path.read_bytes(); saved = json.loads(blob)
    require(saved['status'] == status and saved['heldout_truth_loaded'] is False,
            'prediction status/truth mismatch')
    require(saved['row_ids'] == ids and len(saved['pred']) == len(ids), 'query row mismatch')
    require(all(type(v) is float and math.isfinite(v) for v in saved['pred']), 'nonfinite or invalid prediction')
    require(saved['pred_sha256'] == digest(saved['pred']), 'prediction digest mismatch')
    return saved, blob


def validate_raw(path, reg, fold, seed, member, candidate, tx, qx, y, details):
    if not __debug__ or sys.flags.optimize: raise RuntimeError('Python -O prohibited')
    require(details['require_all66'] is True and details['heldout_truth_loaded'] is False,
            'fresh sealed original66 preparation required')
    require(raw_producer.runtime_paths() == reg['runtime_module_paths'], 'raw runtime path mismatch')
    require(seed in reg['seeds'] and member in ('ET', 'LGB', 'MLP'), 'unregistered seed/member')
    if candidate.startswith('BASELINE_'):
        require(candidate == 'BASELINE_' + member, 'baseline member/name mismatch')
        required_columns = FULL_R3 if member == 'ET' else BASE_R3
    else:
        require(member == 'ET' and candidate in reg['family_map'], 'unregistered domain family')
        required_columns = FULL_R3 + reg['family_map'][candidate]['additional_ET_columns']
    require(list(tx.columns) == required_columns, 'matrix columns differ from registered model recipe')
    require(list(tx.columns) == list(qx.columns), 'train/query column mismatch')
    require(len(y) == len(tx) and np.isfinite(y).all(), 'invalid training labels')
    require(numeric_sha(y) == details['train_label_sha256'], 'fresh train label digest mismatch')
    ids = fold['ordered_query_ids']
    require(len(qx) == len(ids) and len(tx) == len(fold['ordered_train_ids']), 'fresh matrix row mismatch')
    require(list(qx.index) == ids and list(tx.index) == fold['ordered_train_ids'], 'fresh matrix ID order mismatch')
    expected = {'registration_sha256': sha(HERE / 'DOMAIN24_original_raw_fit_registration_v3.json'),
                'validator': fold['validator'], 'fold': fold['fold'], 'seed': seed,
                'member': member, 'candidate': candidate,
                'preparation_receipt_sha256': details['preparation_receipt_sha256'],
                'matrices': {'train_matrix_sha256': matrix_sha(tx), 'query_matrix_sha256': matrix_sha(qx),
                             'train_labels_sha256': numeric_sha(y), 'columns': list(tx.columns)},
                'model_contract': reg['model_contracts'][f'{member}_seed{seed}'],
                'runtime_module_paths': reg['runtime_module_paths']}
    saved, blob = read_prediction(path, ids, 'ORIGINAL_RAW_MODEL_PREDICTION_PASS_NO_SCORE')
    require(saved['contract'] == expected, 'saved raw/fresh matrix contract mismatch')
    imputer = {}
    if member in ('ET', 'MLP'):
        values = tx.to_numpy(float)
        empty = [c for c, missing in zip(tx.columns, np.isnan(values).all(axis=0)) if missing]
        median = np.nanmedian(values, axis=0)
        imputer = {'statistics_sha256': numeric_sha(median),
                   'independent_train_median_sha256': numeric_sha(median),
                   'all_missing_train_columns': empty,
                   'effective_columns': [c for c in tx if c not in empty], 'fit_reference_only': True}
    require(saved['imputer'] == imputer, 'saved imputer/fresh training median mismatch')
    audit = saved['audit']; errors = audit['reverse_scattered_prefix_max_differences']
    require(audit['prefix_prediction_checks'] == 3 and len(errors) == 5, 'raw numeric audit count mismatch')
    require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in errors), 'invalid raw differences')
    require(audit['all_errors_max'] == max(errors) <= 1e-6, 'raw prediction audit failed')
    require(audit['fit_rows'] == len(tx) and audit['query_rows'] == len(ids), 'raw audit row mismatch')
    require(Path(path).read_bytes() == blob, 'raw output changed during consumption')
    return saved['pred'], {'path': str(Path(path).resolve()),
                           'sha256': hashlib.sha256(blob).hexdigest(), 'fresh_contract_matched': True}


def validate_pfn(path, audit_path, reg, fold, seed, tr, q, details):
    if not __debug__ or sys.flags.optimize: raise RuntimeError('Python -O prohibited')
    require(details['require_all66'] is True and sys.byteorder == 'little', 'fresh preparation/platform mismatch')
    require(details['heldout_truth_loaded'] is False, 'preparation read heldout truth')
    require(type(seed) is int and seed in (5, 6, 7, 8), 'unregistered PFN context seed')
    require(reg['columns'] == FULL, 'PFN columns differ from registered recipe')
    ids = fold['ordered_query_ids']
    require(tr.row_id.tolist() == fold['ordered_train_ids'] and q.row_id.tolist() == ids, 'PFN fresh matrix order mismatch')
    ix = np.random.default_rng(seed).choice(len(tr), size=2000, replace=False)
    context = tr.row_id.iloc[ix].tolist()
    name = f'{fold["validator"]}_fold{fold["fold"]}'
    require(context == reg['contexts'][name][str(seed)] and len(set(context)) == 2000, 'PFN context replay mismatch')
    require(set(context) <= set(fold['ordered_train_ids']) and not set(context).intersection(
        set(ids) | set(fold['input_forbidden_ids'])), 'PFN context includes forbidden row')
    y = tr.sub_ec.to_numpy(float)
    require(np.isfinite(y).all() and numeric_sha(y) == details['train_label_sha256'], 'PFN training label digest mismatch')
    X = tr[FULL].to_numpy(np.float32); Q = q[FULL].to_numpy(np.float32)
    contract = {'registration_sha256': sha(HERE / 'DOMAIN24_original_pfn_registration_v2.json'),
                'raw_fit_registration_sha256': sha(HERE / 'DOMAIN24_original_raw_fit_registration_v3.json'),
                'validator': fold['validator'], 'fold': fold['fold'], 'context_seed': seed,
                'columns': FULL, 'context_ids_sha256': digest(context), 'ordered_query_ids_sha256': digest(ids),
                'prepared_fold_sha256': details['preparation_receipt_sha256'],
                'train_context_matrix_sha256': hashlib.sha256(X[ix].tobytes()).hexdigest(),
                'reference_labels_sha256': hashlib.sha256(y[ix].tobytes()).hexdigest(),
                'full_query_matrix_sha256': hashlib.sha256(Q.tobytes()).hexdigest()}
    saved, blob = read_prediction(path, ids, 'ORIGINAL_PFN_REFERENCE_ONLY_RAW_PASS_NO_SCORE')
    require(saved['contract'] == contract, 'PFN saved/fresh context matrix contract mismatch')
    audit_blob = Path(audit_path).read_bytes(); audit = json.loads(audit_blob)
    require(audit['status'] == 'PASS' and audit['contract'] == contract, 'PFN audit contract mismatch')
    require(audit['prediction_file_sha256'] == hashlib.sha256(blob).hexdigest(), 'PFN audit/output mismatch')
    require(audit['code_sha256'] == sha(HERE / 'run_original_pfn_cache_v2.py') and
            audit['auditor_sha256'] == sha(pfn_rules.__file__), 'PFN audit source mismatch')
    pfn_rules.audit_metadata(audit, len(ids), PFN_KEYS)
    require(Path(path).read_bytes() == blob and Path(audit_path).read_bytes() == audit_blob, 'PFN outputs changed during consumption')
    return saved['pred'], {'prediction_sha256': hashlib.sha256(blob).hexdigest(),
                           'audit_sha256': hashlib.sha256(audit_blob).hexdigest(),
                           'fresh_context_matrix_matched': True}

