"""Bind fresh receipt audit to both unchanged model producers before execution."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not __debug__ or sys.flags.optimize: raise RuntimeError('Python -O prohibited')
    pins = {}
    for name in ('DOMAIN24_original_raw_driver_registration_v4.json',
                 'DOMAIN24_original_pfn_driver_registration_v3.json'):
        p = HERE / name
        driver = json.loads(p.read_text(encoding='utf-8'))
        for source, value in driver['sources_sha256'].items():
            assert sha(source) == value
            if source in pins: assert pins[source] == value
            pins[source] = value
        pins[str(p.resolve())] = sha(p)
    for name in ('original_saved_model_validation_v2.py', 'original_completion_receipt_v2.py',
                 'probe_original_fresh_receipts_v2.py', 'register_original_fresh_receipt_probe_v2.py',
                 'original_completion_manifest_audit_v2.json',
                 'critique_ORIGINAL_fresh_receipt_plan_v1.md', 'critique_ORIGINAL_completion_receipt_v2.md', 'critique_ORIGINAL_PFN_first_context_v1.md'):
        p = (HERE / name).resolve()
        value = sha(p)
        if str(p) in pins: assert pins[str(p)] == value
        pins[str(p)] = value
    result = {'status': 'REGISTERED_FIRST_FOLD_FRESH_RECEIPT_AUDIT_NO_SCORE',
              'sources_sha256': pins, 'probe_sha256': sha(HERE / 'probe_original_fresh_receipts_v2.py'),
              'validator': 'DIAG10', 'fold': 0, 'raw_contract_count': 81,
              'pfn_contexts': [5, 6], 'fresh_feature_loader_requires_all66_prepared': True,
              'model_fit': False, 'heldout_truth_read': False, 'whole_pipeline_gate_passed': False}
    with (HERE / 'ORIGINAL_fresh_receipt_probe_registration_v2.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(f'Fresh first-fold receipt audit registered {len(pins)} pins; no data/model/score')


if __name__ == '__main__': main()

