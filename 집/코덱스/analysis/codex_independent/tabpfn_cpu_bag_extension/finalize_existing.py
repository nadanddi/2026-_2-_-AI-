"""Verify completed checkpoints and write the JSON summary after a dtype error."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
NEW = Path(sys.argv[1]).resolve()
OLD = ROOT / 'local/tabpfn_cpu_bag/20260927_182052'
assert NEW.is_dir() and ROOT / 'local/tabpfn_cpu_bag_extension' in NEW.parents


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


manifest = json.loads((NEW / 'manifest.json').read_text(encoding='utf-8'))
assert manifest['code_hash'] == sha(HERE.with_name('run.py'))
assert manifest['protocol_hash'] == sha(HERE.with_name('PROTOCOL.md'))
assert manifest['previous_bag_manifest_hash'] == sha(OLD / 'manifest.json')
assert manifest['previous_bag_result_hash'] == sha(OLD / 'result.json')
for fold in (0, 2):
    assert manifest['previous_bag_predictions'][f'fold{fold}'] == sha(OLD / f'fold{fold}_bag.csv')

stored = json.loads((NEW / 'partial_results.json').read_text(encoding='utf-8'))
old_scores = json.loads((OLD / 'result.json').read_text(encoding='utf-8'))['scores']
scores = {}
for fold in (0, 2, 4, 6):
    folder = OLD if fold in (0, 2) else NEW
    row = pd.read_csv(folder / f'fold{fold}_bag.csv')
    assert row.row_id.is_unique and row.sub_ec.notna().all()
    if fold in (4, 6):
        members = []
        for seed in (1, 2, 3, 4):
            member = pd.read_csv(NEW / f'fold{fold}_context{seed}.csv')
            assert member.row_id.tolist() == row.row_id.tolist()
            assert np.isfinite(member.member).all()
            members.append(member.member.to_numpy(float))
        np.testing.assert_allclose(row.bag, np.mean(members, axis=0), rtol=0, atol=1e-12)
    for seed in (7, 101):
        key = f'fold{fold}_base{seed}'
        truth = row.sub_ec.to_numpy(float)
        baseline = np.sqrt(np.mean((truth - row[f'baseline_{seed}'].to_numpy(float)) ** 2))
        candidate = np.sqrt(np.mean((truth - row[f'blend_{seed}'].to_numpy(float)) ** 2))
        saved = old_scores[key] if fold in (0, 2) else stored[key]
        assert abs(baseline - saved['baseline']) < 1e-12
        assert abs(candidate - saved['candidate']) < 1e-12
        scores[key] = dict(baseline=float(baseline), candidate=float(candidate),
                           relative_change=float(candidate/baseline-1))

changes = [v['relative_change'] for v in scores.values()]
result = dict(scores=scores, all_improve=bool(all(x < 0 for x in changes)),
              mean_relative_change=float(np.mean(changes)),
              passes_search=bool(all(x < 0 for x in changes) and np.mean(changes) <= -.01),
              confirmation_scored=False, adopted=False,
              finalized_from_saved_predictions=True,
              note='Original runner completed predictions and scores, but JSON serialization of a numpy bool failed.')
(NEW / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
