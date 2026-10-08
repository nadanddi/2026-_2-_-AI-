"""Synthetic preservation/resume checks; no data, model fit or scores."""
from pathlib import Path
import hashlib
import json
import tempfile
import original_assembly_checkpoint_v1 as storage

HERE = Path(__file__).resolve().parent


def main():
    checks = []
    pred = {'row_ids': ['synthetic'], 'prediction': [0.25], 'whole_pipeline_gate_passed': False}
    def audit(h): return {'predictions_sha256': h, 'whole_pipeline_gate_passed': False}
    with tempfile.TemporaryDirectory(prefix='assembly_resume_synthetic_', dir=HERE) as tmp:
        root = Path(tmp)
        with storage.single_writer(root, 'a' * 64):
            try:
                with storage.single_writer(root, 'a' * 64): pass
            except RuntimeError: checks.append('second writer refused')
            else: raise AssertionError('second writer entered')
            folder = root / 'fold'
            storage.persist_pair(folder, pred, audit)
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            storage.persist_pair(folder, pred, audit)
            assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
            checks.append('complete pair exact fresh resume preserves bytes')
            try: storage.persist_pair(folder, dict(pred, prediction=[0.26]), audit)
            except ValueError: pass
            else: raise AssertionError('different fresh predictions accepted')
            assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
            checks.append('prediction mismatch refused and preserved')
            partial = root / 'partial'; partial.mkdir()
            pp = partial / 'predictions.json'
            pp.write_text(json.dumps(pred, indent=2), encoding='utf-8')
            old = pp.read_bytes()
            storage.persist_pair(partial, pred, audit)
            assert pp.read_bytes() == old
            assert json.loads((partial/'audit.json').read_text())['predictions_sha256'] == hashlib.sha256(old).hexdigest()
            checks.append('prediction-only partial resumed with preserved bytes SHA')
            auditonly = root/'auditonly'; auditonly.mkdir()
            h = hashlib.sha256(storage.blob(pred).encode()).hexdigest()
            (auditonly/'audit.json').write_text(storage.blob(audit(h)), encoding='utf-8')
            storage.persist_pair(auditonly, pred, audit)
            checks.append('matching audit-only partial resumed')
            bad = root/'bad'; bad.mkdir()
            ap = bad/'audit.json'; ap.write_text(storage.blob(audit('b'*64)), encoding='utf-8')
            old = ap.read_bytes()
            try: storage.persist_pair(bad, pred, audit)
            except ValueError: pass
            else: raise AssertionError('wrong audit accepted')
            assert ap.read_bytes() == old and not (bad/'predictions.json').exists()
            checks.append('wrong audit refused BEFORE missing prediction write')
            pp = root/'invalid.json'; pp.write_text('{broken', encoding='utf-8')
            try: storage.persist_complete(pp, pred)
            except json.JSONDecodeError: pass
            else: raise AssertionError('invalid artifact overwritten')
            assert pp.read_text() == '{broken'
            checks.append('invalid JSON preserved')
            cp = root/'complete.json'; storage.persist_complete(cp, pred)
            old = cp.read_bytes(); storage.persist_complete(cp, pred)
            assert old == cp.read_bytes()
            checks.append('global complete exact resume preserves bytes')
            wrong = dict(pred, whole_pipeline_gate_passed=0)
            try: storage.persist_complete(cp, wrong)
            except ValueError: checks.append('boolean versus integer mismatch rejected')
            else: raise AssertionError('type mismatch accepted')
        assert not (root/'RUN_WRITER_LOCK.json').exists()
        checks.append('owned writer lock removed after success')
        try:
            with storage.single_writer(root, 'a'*64): raise ValueError('synthetic interruption')
        except ValueError: pass
        assert not (root/'RUN_WRITER_LOCK.json').exists()
        checks.append('owned writer lock removed after exception')
    result = {'status': 'SYNTHETIC_ASSEMBLY_STORAGE_PASS_NOT_FULL_MODEL_GATE',
              'checks': checks, 'check_count': len(checks), 'model_fit': False,
              'heldout_truth_read': False, 'whole_pipeline_gate_passed': False,
              'limits': ['No actual process crash or simultaneous two-process race injected',
                         'No original66 assembly or numeric/model/causality gate']}
    with (HERE/'original_assembly_checkpoint_synthetic_audit_v1.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print('Synthetic assembly storage PASS', len(checks))


if __name__ == '__main__': main()
