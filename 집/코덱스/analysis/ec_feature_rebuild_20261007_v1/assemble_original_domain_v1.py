"""Original66 domain24 fixed mixed predictions; no heldout truth or score gate.

Require raw5346/PFN264 completion first. Every consumed output is checked
against fresh features/reference labels before frozen mix/shrink/clip/SG2.
An independent full numerical/causal verifier remains mandatory for scoring.
"""
from pathlib import Path
import json
import sys
import original_fold_features_v2 as features
import original_saved_model_validation_v2 as validation
import original_completion_receipt_v2 as completion
import original_postprocess_kernel_v2 as kernel
import original_sg2_plan_v1 as sgplan
from blk_baseline_data_v1 import HERE, FULL_R3, BASE_R3, sha
from checkpoint_v1 import atomic


def sources(reg):
    if not all(sha(p) == v for p, v in reg['sources_sha256'].items()):
        raise ValueError('assembly source changed')


def main():
    if not __debug__ or sys.flags.optimize: raise RuntimeError('Python -O prohibited')
    regpath = HERE / 'ORIGINAL_DOMAIN24_pipeline_registration_v1.json'
    reg = json.loads(regpath.read_text(encoding='utf-8'))
    assert reg['status'] == 'REGISTERED_ORIGINAL66_DOMAIN24_ASSEMBLY_NO_SCORE'
    assert reg['assembler_sha256'] == sha(__file__)
    sources(reg)
    for loaded, name in ((features, 'original_fold_features_v2.py'),
                         (validation, 'original_saved_model_validation_v2.py'),
                         (completion, 'original_completion_receipt_v2.py'),
                         (kernel, 'original_postprocess_kernel_v2.py'),
                         (sgplan, 'original_sg2_plan_v1.py'),
                         (validation.raw_producer, 'run_domain_original_raw_v3.py'),
                         (validation.pfn_rules, 'original_pfn_cache_rules_v1.py')):
        path = Path(loaded.__file__).resolve()
        assert path == (HERE / name).resolve() and sha(path) == reg['sources_sha256'][str(path)]
    registry = json.loads((HERE / 'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    rawpath = HERE / 'DOMAIN24_original_raw_fit_registration_v3.json'
    pfnpath = HERE / 'DOMAIN24_original_pfn_registration_v2.json'
    rawreg = json.loads(rawpath.read_text(encoding='utf-8'))
    pfnreg = json.loads(pfnpath.read_text(encoding='utf-8'))
    assert len(rawreg['family_map']) == 24 and rawreg['seeds'] == [47, 1414, 6464]
    rawroot = HERE / 'checkpoints/DOMAIN24_ORIGINAL_RAW_v3'
    pfnroot = HERE / 'checkpoints/DOMAIN24_ORIGINAL_PFN_CACHE_v2'
    # Closed until both exact66 manifests exist; no feature/label load precedes these checks.
    rawmanifest = completion.verify_all(rawroot, registry, rawreg, 'raw', sha(rawpath))
    pfnmanifest = completion.verify_all(pfnroot, registry, rawreg, 'pfn', sha(pfnpath))
    root = HERE / 'checkpoints/DOMAIN24_ORIGINAL_ASSEMBLED_v1'
    root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for fold in registry['folds']:
        sources(reg)
        name = f'{fold["validator"]}_fold{fold["fold"]}'
        folder = root / name
        # Existing completed artifacts are preserved. Independent validation is
        # required before adding a resume branch in a later registered version.
        assert not folder.exists(), 'assembly already exists; preserve and investigate before resuming'
        ctx, tr, q, calendar, domain, matrices, details = features.load_production_fold(
            fold['validator'], fold['fold'], rawreg)
        ids = fold['ordered_query_ids']; y = tr.sub_ec.to_numpy(float)
        assert tr.row_id.tolist() == fold['ordered_train_ids'] and q.row_id.tolist() == ids
        assert set(ctx.reference_labels) == set(fold['ordered_train_ids'])
        assert not set(ctx.reference_labels).intersection(ctx.query_ids | ctx.gap_ids)
        lower, upper = float(y.min()), float(y.max())
        pfns, pfnreceipts = {}, []
        for seed in (5, 6, 7, 8):
            path = pfnroot / name / f'context{seed}.json'
            auditpath = pfnroot / name / f'context{seed}_audit.json'
            values, receipt = validation.validate_pfn(path, auditpath, pfnreg, fold, seed, tr, q, details)
            assert receipt['prediction_sha256'] == pfnmanifest['files_sha256'][f'{name}/{path.name}']
            assert receipt['audit_sha256'] == pfnmanifest['files_sha256'][f'{name}/{auditpath.name}']
            pfns[seed] = values; pfnreceipts.append(receipt)
        plan = sgplan.OriginalSG2Plan(ctx, ids, reg['sources_sha256'])
        saved = {'row_ids': ids, 'baseline': {}, 'candidates': {}, 'stage_outputs': {},
                 'pipeline_registration_sha256': sha(regpath), 'validator': fold['validator'],
                 'fold': fold['fold'], 'training_label_bounds': [lower, upper],
                 'fresh_preparation_details': details, 'heldout_truth_loaded': False,
                 'whole_pipeline_gate_passed': False}
        rawreceipts = []
        for seed in (47, 1414, 6464):
            predictions = {}
            for member, columns in (('ET', FULL_R3), ('LGB', BASE_R3), ('MLP', BASE_R3)):
                cid = f'BASELINE_{member}'
                tx = tr[columns].copy(); tx.index = tr.row_id
                qx = q[columns].copy(); qx.index = q.row_id
                path = rawroot / name / f'{cid}_seed{seed}.json'
                values, receipt = validation.validate_raw(path, rawreg, fold, seed, member, cid, tx, qx, y, details)
                assert receipt['sha256'] == rawmanifest['files_sha256'][f'{name}/{path.name}']
                predictions[cid] = values; rawreceipts.append(receipt)
            for cid, (tx, qx) in sorted(matrices.items()):
                path = rawroot / name / f'{cid}_seed{seed}.json'
                values, receipt = validation.validate_raw(path, rawreg, fold, seed, 'ET', cid, tx, qx, y, details)
                assert receipt['sha256'] == rawmanifest['files_sha256'][f'{name}/{path.name}']
                predictions[cid] = values; rawreceipts.append(receipt)
            for cid in ('BASELINE_ET', *sorted(rawreg['family_map'])):
                stages = kernel.assemble(ids, predictions[cid], predictions['BASELINE_LGB'],
                                         predictions['BASELINE_MLP'], pfns, lower, upper, plan.apply)
                tag = f'{cid}_seed{seed}'
                saved['stage_outputs'][tag] = stages
                if cid == 'BASELINE_ET':
                    saved['baseline'][str(seed)] = stages['post_SG2_clip']
                    mapping = dict(zip(ids, stages['pre_SG2_clip']))
                    for rid in ids:
                        f, d, h = kernel.key(rid)
                        prefix = {f'{f}_{d:03d}_{j:02d}': mapping[f'{f}_{d:03d}_{j:02d}'] for j in range(h + 1)}
                        plan.audit_source(rid, prefix)
                else:
                    saved['candidates'].setdefault(str(seed), {})[cid] = stages['post_SG2_clip']
            print(f'{name} seed{seed}: baseline+domain24 mixed; SG2 baseline all-row source audited', flush=True)
        assert len(rawreceipts) == 81 and len(pfnreceipts) == 4
        assert len(saved['stage_outputs']) == 75
        choices = plan.snapshot()
        assert choices['source_checks'] == 3 * len(ids)
        folder.mkdir(parents=True, exist_ok=False)
        atomic(folder / 'predictions.json', json.dumps(saved, ensure_ascii=False, allow_nan=False))
        audit = {'status': 'ORIGINAL_FOLD_ASSEMBLY_COMPLETE_NOT_SCORE_GATE',
                 'predictions_sha256': sha(folder / 'predictions.json'), 'raw_receipts': rawreceipts,
                 'pfn_receipts': pfnreceipts, 'choice_snapshot': choices,
                 'registration_sha256': sha(regpath), 'heldout_truth_loaded': False,
                 'whole_pipeline_gate_passed': False,
                 'limits': ['Source replay covers all baseline rows; independent candidate arithmetic remains required',
                            'No fresh future-input poison or full mixed causal audit here',
                            'No independent model refit or historical GPU identity claim']}
        atomic(folder / 'audit.json', json.dumps(audit, ensure_ascii=False, allow_nan=False))
        for path in (folder / 'predictions.json', folder / 'audit.json'):
            outputs[path.relative_to(root).as_posix()] = sha(path)
        sources(reg)
    assert len(outputs) == 132
    # Recheck all consumed immutable producer outputs at completion.
    for base, manifest in ((rawroot, rawmanifest), (pfnroot, pfnmanifest)):
        assert all(sha(base / name) == value for name, value in manifest['files_sha256'].items())
    sources(reg)
    complete = {'status': 'ORIGINAL66_DOMAIN24_ASSEMBLED_NOT_FULL_SCORE_GATE',
                'files_sha256': outputs, 'registration_sha256': sha(regpath),
                'raw_completion_sha256': rawmanifest['completion_sha256'],
                'pfn_completion_sha256': pfnmanifest['completion_sha256'],
                'heldout_truth_loaded': False, 'whole_pipeline_gate_passed': False}
    atomic(root / 'complete.json', json.dumps(complete, ensure_ascii=False, allow_nan=False))
    print('Original66 assembly complete; independent full gate and scoring still closed')


if __name__ == '__main__': main()
