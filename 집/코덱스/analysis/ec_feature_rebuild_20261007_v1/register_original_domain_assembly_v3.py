"""Freeze assembly source chain; no inputs, labels, fitting or scoring."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not __debug__ or sys.flags.optimize: raise RuntimeError('Python -O prohibited')
    pins = {}
    prior = HERE/'ORIGINAL_fresh_receipt_probe_registration_v2.json'
    reg = json.loads(prior.read_text(encoding='utf-8'))
    for p, value in reg['sources_sha256'].items():
        assert sha(p) == value
        pins[p] = value
    pins[str(prior.resolve())] = sha(prior)
    additions = ('assemble_original_domain_v3.py', 'register_original_domain_assembly_v3.py',
                 'original_postprocess_kernel_v2.py', 'original_sg2_plan_v1.py',
                 'original_assembly_checkpoint_v1.py', 'audit_original_assembly_checkpoint_v2.py',
                 'original_assembly_checkpoint_synthetic_audit_v2.json',
                 'original_postprocess_kernel_synthetic_audit_v2.json',
                 'ORIGINAL_SG2_source_probe_result_v2.json',
                 'critique_ORIGINAL_domain_assembly_plan_v1.md',
                 'critique_ORIGINAL_domain_assembly_plan_v2.md',
                 'critique_ORIGINAL_SG2_actual_probe_v2.md',
                 'critique_ORIGINAL_fresh_receipt_actual_v2.md',
                 'DOMAIN24_original_statistics_registration_v2.json',
                 'original_domain_statistics_v2.py', 'checkpoint_v1.py', 'checkpoint_v2.py',
                 'blk_baseline_data_v1.py')
    for name in additions:
        p = (HERE/name).resolve(); value = sha(p)
        if str(p) in pins: assert pins[str(p)] == value
        pins[str(p)] = value
    synthetic = json.loads((HERE/'original_assembly_checkpoint_synthetic_audit_v2.json').read_text(encoding='utf-8'))
    assert synthetic['status'] == 'SYNTHETIC_ASSEMBLY_STORAGE_PASS_NOT_FULL_MODEL_GATE'
    assert synthetic['check_count'] == 11 and synthetic['whole_pipeline_gate_passed'] is False
    assert synthetic['model_fit'] is False and synthetic['heldout_truth_read'] is False
    expected_checks = {'second writer refused', 'complete pair exact fresh resume preserves bytes',
                       'prediction mismatch refused and preserved',
                       'prediction-only partial resumed with preserved bytes SHA',
                       'matching audit-only partial resumed',
                       'wrong audit refused BEFORE missing prediction write', 'invalid JSON preserved',
                       'global complete exact resume preserves bytes', 'boolean versus integer mismatch rejected',
                       'owned writer lock removed after success', 'owned writer lock removed after exception'}
    assert len(synthetic['checks']) == 11 and set(synthetic['checks']) == expected_checks
    expected_sources = {str((HERE/n).resolve()): sha(HERE/n) for n in
                        ('audit_original_assembly_checkpoint_v2.py', 'original_assembly_checkpoint_v1.py',
                         'checkpoint_v1.py', 'checkpoint_v2.py')}
    assert synthetic['executed_sources_sha256'] == expected_sources
    assert all(pins.get(path) == value for path,value in expected_sources.items())
    result = {'status': 'REGISTERED_ORIGINAL66_DOMAIN24_ASSEMBLY_NO_SCORE',
              'sources_sha256': pins, 'assembler_sha256': sha(HERE/'assemble_original_domain_v3.py'),
              'required_before_feature_label_loading': ['raw original66 exact5346 manifest',
                                                       'PFN original66 exact528 manifest'],
              'recipe': {'R3': {'ET': 0.6, 'LGB': 0.3, 'MLP': 0.1},
                         'R3_weight': 0.6, 'PFN_weight': 0.4, 'PFN_context_seeds': [5, 6, 7, 8],
                         'prefix_shrink_once': 0.5, 'clip': 'fold train EC min/max before and after SG2',
                         'SG2': 'RAW_PASS day >=179; distinct samefarm train-only reference',
                         'candidate_change': 'ET family only; other raw members fixed'},
              'seeds': [47, 1414, 6464], 'families': 24, 'folds': 66,
              'resume': 'fresh complete fold recomputation; strict JSON object equality; existing bytes preserved',
              'writer': 'O_EXCL PID + Windows creation ticks; no automatic stale lock deletion',
              'final_rehash': ['all producer outputs', 'producer root/fold completion receipts',
                               'all132 assembled outputs', 'registered sources'],
              'heldout_truth_read': False, 'model_fit': False, 'whole_pipeline_gate_passed': False,
              'score_gate': 'Independent full numerical/input causality verification mandatory before score',
              'limits': ['No independent refit or historical GPU prediction identity claim',
                         'Assembly source audit covers baseline rows; candidate arithmetic audited separately']}
    with (HERE/'ORIGINAL_DOMAIN24_pipeline_registration_v3.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print('Original domain assembly registered', len(pins), 'pins; no assembly/model/score execution')


if __name__ == '__main__': main()
