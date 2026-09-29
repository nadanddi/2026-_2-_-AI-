"""Finalize a fully predicted run from saved, checked CSVs after interruption."""
import sys
import json
from pathlib import Path

V1 = Path(__file__).resolve().parents[1] / 'rl_ec_v1'
sys.path.insert(0, str(V1))
import env  # register the read-only local numerical libraries

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[2] / 'local/tabpfn_cpu_bag/20260927_182052'


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main():
    manifest = json.loads((RUN / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['folds'] == [0, 2]
    assert manifest['context_seeds'] == [1, 2, 3, 4]
    assert manifest['base_seeds'] == [7, 101]
    assert manifest['offline'] is True and manifest['device'] == 'cpu'
    partial = json.loads((RUN / 'partial_results.json').read_text(encoding='utf-8'))
    assert set(partial) == {f'fold{fold}_base{seed}' for fold in (0, 2) for seed in (7, 101)}

    scores = {}
    frames = []
    for fold in (0, 2):
        frame = pd.read_csv(RUN / f'fold{fold}_bag.csv')
        assert frame.row_id.is_unique and frame.sub_ec.notna().all()
        members = []
        for seed in (1, 2, 3, 4):
            member = pd.read_csv(RUN / f'fold{fold}_context{seed}.csv')
            assert member.row_id.tolist() == frame.row_id.tolist()
            assert np.isfinite(member.member.to_numpy(float)).all()
            members.append(member.member.to_numpy(float))
        actual_bag = np.mean(members, axis=0)
        assert np.allclose(frame.bag.to_numpy(float), actual_bag, rtol=0, atol=1e-12)
        for seed in (7, 101):
            a = rmse(frame.sub_ec, frame[f'baseline_{seed}'])
            b = rmse(frame.sub_ec, frame[f'blend_{seed}'])
            prior = partial[f'fold{fold}_base{seed}']
            assert abs(a - prior['baseline']) < 1e-12
            assert abs(b - prior['candidate']) < 1e-12
            assert abs(b / a - 1 - prior['relative_change']) < 1e-12
            scores[f'fold{fold}_base{seed}'] = dict(baseline=a, candidate=b,
                                                      relative_change=b / a - 1)
        frames.append(frame)
    delta = [x['relative_change'] for x in scores.values()]
    result = dict(scores=scores, all_improve=all(x < 0 for x in delta),
                  mean_relative_change=float(np.mean(delta)),
                  passes_search=bool(all(x < 0 for x in delta) and np.mean(delta) <= -.01),
                  confirmation_scored=False, adopted=False,
                  finalized_from_verified_checkpoints=True,
                  note='Existing runner was interrupted after saving all four scores; no model refit.')
    (RUN / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with (RUN / 'progress.log').open('a', encoding='utf-8') as f:
        f.write('FINALIZED FROM VERIFIED CHECKPOINTS ' + json.dumps(result) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
