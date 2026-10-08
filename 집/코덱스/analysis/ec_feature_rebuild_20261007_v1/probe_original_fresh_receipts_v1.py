"""Fresh first-fold raw81 + PFN contexts5/6 receipt audit, no model fit or scores."""
from pathlib import Path
import json
import sys
import original_fold_features_v2 as features
from original_saved_model_validation_v1 import validate_raw, validate_pfn
from original_completion_receipt_v2 import verify_fold
from blk_baseline_data_v1 import HERE, sha, FULL_R3, BASE_R3


def main():
    if not __debug__ or sys.flags.optimize: raise RuntimeError('Python -O prohibited')
    regpath = HERE / 'ORIGINAL_fresh_receipt_probe_registration_v1.json'
    reg = json.loads(regpath.read_text(encoding='utf-8'))
    assert reg['status'] == 'REGISTERED_FIRST_FOLD_FRESH_RECEIPT_AUDIT_NO_SCORE'
    assert reg['probe_sha256'] == sha(__file__)
    assert all(sha(p) == v for p, v in reg['sources_sha256'].items())
    rawpath = HERE / 'DOMAIN24_original_raw_fit_registration_v3.json'
    pfnpath = HERE / 'DOMAIN24_original_pfn_registration_v2.json'
    rawreg = json.loads(rawpath.read_text(encoding='utf-8'))
    pfnreg = json.loads(pfnpath.read_text(encoding='utf-8'))
    registry = json.loads((HERE / 'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    fold = next(f for f in registry['folds'] if f['validator'] == 'DIAG10' and f['fold'] == 0)
    rawroot = HERE / 'checkpoints/DOMAIN24_ORIGINAL_RAW_v3'
    manifest = verify_fold(rawroot, fold, rawreg, 'raw', sha(rawpath))
    ctx, tr, q, calendar, domain, matrices, details = features.load_production_fold('DIAG10', 0, rawreg)
    assert set(ctx.reference_labels) == set(fold['ordered_train_ids'])
    assert not set(ctx.reference_labels).intersection(ctx.query_ids | ctx.gap_ids)
    y = tr.sub_ec.to_numpy(float)
    rawreceipts = []
    for seed in (47, 1414, 6464):
        for member, columns in (('ET', FULL_R3), ('LGB', BASE_R3), ('MLP', BASE_R3)):
            candidate = f'BASELINE_{member}'
            tx = tr[columns].copy(); tx.index = tr.row_id
            qx = q[columns].copy(); qx.index = q.row_id
            _, record = validate_raw(rawroot / f'DIAG10_fold0/{candidate}_seed{seed}.json', rawreg,
                                     fold, seed, member, candidate, tx, qx, y, details)
            rawreceipts.append(record)
        for candidate, (tx, qx) in sorted(matrices.items()):
            _, record = validate_raw(rawroot / f'DIAG10_fold0/{candidate}_seed{seed}.json', rawreg,
                                     fold, seed, 'ET', candidate, tx, qx, y, details)
            rawreceipts.append(record)
        print(f'Fresh DIAG10fold0 seed{seed} raw27 saved contracts PASS', flush=True)
    assert len(rawreceipts) == 81
    pfnreceipts = []
    folder = HERE / 'checkpoints/DOMAIN24_ORIGINAL_PFN_CACHE_v2/DIAG10_fold0'
    for seed in (5, 6):
        _, record = validate_pfn(folder / f'context{seed}.json', folder / f'context{seed}_audit.json',
                                 pfnreg, fold, seed, tr, q, details)
        pfnreceipts.append(record)
        print(f'Fresh DIAG10fold0 PFNcontext{seed} matrix/labels/cache receipt PASS', flush=True)
    assert all(sha(p) == v for p, v in reg['sources_sha256'].items())
    assert all(sha(rawroot / name) == value for name, value in manifest['files_sha256'].items())
    result = {'status': 'FIRST_FOLD_FRESH_RECEIPTS_PASS_NOT_FULL_MODEL_GATE',
              'registration_sha256': sha(regpath), 'probe_sha256': sha(__file__),
              'validator': 'DIAG10', 'fold': 0, 'raw_saved_contracts': rawreceipts,
              'pfn_saved_contracts': pfnreceipts, 'query_rows': len(q), 'training_rows': len(tr),
              'raw_contract_count': 81, 'pfn_context_count': 2,
              'heldout_truth_read': False, 'model_fit': False, 'whole_pipeline_gate_passed': False,
              'limits': ['Only first fold and two PFN contexts, not original66 completeness',
                         'Fresh feature/label/imputer/context reconstruction, no independent model refit',
                         'No mixed postprocessing/SG2 performance or whole causal score gate']}
    with (HERE / 'ORIGINAL_fresh_receipt_probe_result_v1.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print('Fresh receipts raw81/PFN2 PASS; no fit, truth or full score gate')


if __name__ == '__main__': main()
