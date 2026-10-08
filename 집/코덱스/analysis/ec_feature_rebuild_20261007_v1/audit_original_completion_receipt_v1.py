"""Manifest fault injection and actual first-fold storage check; no scores."""
from pathlib import Path
import hashlib
import json
from original_completion_receipt_v1 import verify_manifest, verify_fold, verify_all, sha

HERE = Path(__file__).resolve().parent


def main():
    data = {'FOLD0/a.json': b'first file', 'FOLD0/b.json': b'second file'}
    expected = set(data)
    manifest = {name: hashlib.sha256(blob).hexdigest() for name, blob in data.items()}
    assert verify_manifest(manifest, expected, data.__getitem__) == manifest
    rejected = 0
    cases = [({}, expected), ({**manifest, 'FOLD0/c.json': '0'*64}, expected),
             ({'FOLD0/a.json': manifest['FOLD0/a.json']}, expected),
             ({'../outside.json': '0'*64}, {'../outside.json'}),
             ({'/root/a.json': '0'*64}, {'/root/a.json'}),
             ({'FOLD0\\a.json': '0'*64}, {'FOLD0\\a.json'}),
             ({'FOLD0/../a.json': '0'*64}, {'FOLD0/../a.json'}),
             ({'FOLD0//a.json': '0'*64}, {'FOLD0//a.json'}),
             ({'FOLD0/a.json': 'A'*64}, {'FOLD0/a.json'}),
             ({'FOLD0/a.json': True}, {'FOLD0/a.json'})]
    for bad, names in cases:
        reads = []
        def guarded(name):
            reads.append(name); raise AssertionError('unsafe contract reached file access')
        try: verify_manifest(bad, names, guarded)
        except ValueError: rejected += 1
        else: raise AssertionError('invalid manifest accepted')
        assert not reads
    try: verify_manifest(manifest, expected, lambda name: data[name] + b'tamper')
    except ValueError: rejected += 1
    else: raise AssertionError('tampered output accepted')
    # One actual sealed fold exists while the original66 worker is incomplete.
    regpath = HERE / 'DOMAIN24_original_raw_fit_registration_v3.json'
    rawreg = json.loads(regpath.read_text(encoding='utf-8'))
    registry = json.loads((HERE / 'original_validator_endpoint_registry_v1.json').read_text(encoding='utf-8'))
    fold = next(f for f in registry['folds'] if f['validator'] == 'DIAG10' and f['fold'] == 0)
    root = HERE / 'checkpoints/DOMAIN24_ORIGINAL_RAW_v3'
    actual = verify_fold(root, fold, rawreg, 'raw', sha(regpath))
    assert len(actual['files_sha256']) == 81
    full_rejected = False
    try: verify_all(root, registry, rawreg, 'raw', sha(regpath))
    except FileNotFoundError:
        full_rejected = True
    assert full_rejected, 'expected current incomplete original66 state changed; investigate before claiming'
    result = {'status': 'COMPLETION_MANIFEST_AUDIT_PASS_NOT_NUMERICAL_OR_FULL_MODEL_GATE',
              'synthetic_unsafe_contracts_rejected_before_read': len(cases),
              'synthetic_tampered_output_rejected': True, 'total_rejections': rejected,
              'actual_first_fold_files_checked': 81, 'actual_first_fold_manifest': actual,
              'incomplete_original66_refused': full_rejected,
              'sources_sha256': {str(p): sha(p) for p in
                    (Path(__file__), HERE / 'original_completion_receipt_v1.py', regpath)},
              'heldout_truth_read': False, 'model_fit': False, 'whole_pipeline_gate_passed': False,
              'limits': ['Storage manifest only; does not validate fitted models or fresh matrices',
                         'PFN/raw producer strict receipts, SG2 and independent full causal gate remain mandatory']}
    with (HERE / 'original_completion_manifest_audit_v1.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(f'Manifest audit PASS: actual81, unsafe10, tamper1; incomplete66 refused, no scores')


if __name__ == '__main__': main()
