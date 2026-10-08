"""Seal one-fold SG2 source/prefix audit before reading reference inputs/EC."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not __debug__ or sys.flags.optimize: raise RuntimeError('Python -O prohibited')
    rawpath = HERE / 'DOMAIN24_original_raw_fit_registration_v3.json'
    raw = json.loads(rawpath.read_text(encoding='utf-8'))
    pins = dict(raw['source_sha256'])
    assert all(sha(p) == value for p, value in pins.items())
    additions = ['original_sg2_plan_v1.py', 'original_postprocess_kernel_v2.py',
                 'audit_original_sg2_plan_v1.py', 'original_sg2_plan_synthetic_audit_v1.json',
                 'audit_original_postprocess_kernel_v2.py', 'original_postprocess_kernel_synthetic_audit_v2.json',
                 'critique_ORIGINAL_SG2_plan_kernel_v2.md', 'probe_original_sg2_source_v1.py',
                 'register_original_sg2_source_probe_v1.py', 'DOMAIN24_original_raw_fit_registration_v3.json']
    for name in additions:
        p = (HERE / name).resolve(); value = sha(p)
        if str(p) in pins: assert pins[str(p)] == value
        pins[str(p)] = value
    result = {'status': 'REGISTERED_ONE_FOLD_SG2_SOURCE_AUDIT_NO_SCORE',
              'probe_sha256': sha(HERE / 'probe_original_sg2_source_v1.py'),
              'producer_registration_sha256': sha(rawpath), 'sources_sha256': pins,
              'validator': 'DIAG10', 'fold': 0, 'all_row_source_audit': True,
              'prediction_prefix': '.8 + .001*hour, synthetic and fixed',
              'poison_probes': 'first query day per farm/pass1-or-pass2, hours0/6/23',
              'fresh_choice_for_poison': True, 'heldout_truth_read': False,
              'model_fit': False, 'score_computed': False, 'whole_pipeline_gate_passed': False}
    with (HERE / 'ORIGINAL_SG2_source_probe_registration_v1.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(f'SG2 source probe registered: {len(pins)} pins; no data or score')


if __name__ == '__main__': main()
