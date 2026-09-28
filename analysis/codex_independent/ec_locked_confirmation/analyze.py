"""Predeclared supporting diagnostics for the once-scored confirmation folds."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def score(frame):
    y = frame.sub_ec.to_numpy(float)
    a = float(np.sqrt(np.mean((y - frame.baseline.to_numpy(float)) ** 2)))
    b = float(np.sqrt(np.mean((y - frame.candidate.to_numpy(float)) ** 2)))
    days = frame[['farm', 'day']].drop_duplicates().shape[0] if {'farm', 'day'} <= set(frame.columns) else 1
    return dict(rows=len(frame), days=days,
                baseline=a, candidate=b, relative_change=b/a-1)


def resample(frame, unit, seed):
    rows = frame.copy()
    rows['base_sq'] = (rows.sub_ec - rows.baseline) ** 2
    rows['cand_sq'] = (rows.sub_ec - rows.candidate) ** 2
    groups = {farm: g.groupby(unit)[['base_sq', 'cand_sq']].sum().to_numpy(float)
              for farm, g in rows.groupby('farm')}
    rng = np.random.default_rng(seed)
    changes = np.empty(5000)
    for i in range(len(changes)):
        sampled = sum(t[rng.integers(0, len(t), len(t))].sum(axis=0)
                      for t in groups.values())
        changes[i] = np.sqrt(sampled[1] / sampled[0]) - 1
    return dict(block_counts={farm: len(t) for farm, t in groups.items()},
                draws=len(changes), seed=seed,
                ci95=[float(x) for x in np.quantile(changes, [.025, .975])],
                share_improved=float(np.mean(changes < 0)))


def main():
    out = Path(sys.argv[1]).resolve()
    assert out.is_dir() and ROOT / 'local/ec_locked_confirmation' in out.parents
    assert (out / 'result.json').is_file()
    frames = [pd.read_csv(out / f'fold{fold}.csv') for fold in (8, 9)]
    combined = pd.concat(frames, ignore_index=True)
    assert combined.row_id.is_unique and len(combined) > 0
    result = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    assert abs(score(combined)['relative_change'] - result['pooled']['relative_change']) < 1e-12

    source = Path(os.environ['AGRI_SOURCE_ROOT']).resolve()
    train = pd.read_csv(source / '온라인대회자료/정형데이터/참가자_배포/train_X.csv')
    train = train[train.row_id.isin(combined.row_id)].copy()
    assert len(train) == len(combined)
    train['farm'] = train.row_id.str[:3]
    train['day'] = train.row_id.str[4:7].astype(int)
    grouped = train.groupby(['farm', 'day'])
    sealed = ((grouped.act_circfan.mean() < 10) &
              (grouped.act_vent.apply(lambda s: (s == 0).mean()) > .85))
    sealed = sealed.rename('sealed').reset_index()
    combined = combined.merge(sealed, on=['farm', 'day'], validate='many_to_one')
    assert combined.sealed.notna().all()

    day = combined.groupby(['farm', 'day']).apply(
        lambda g: pd.Series({
            'baseline': score(g)['baseline'], 'candidate': score(g)['candidate'],
            'gain_sq': float((((g.sub_ec-g.baseline)**2)-((g.sub_ec-g.candidate)**2)).sum())
        }), include_groups=False).reset_index()
    positive = day.gain_sq[day.gain_sq > 0].sort_values(ascending=False)
    diagnostics = dict(code_hash=sha(HERE), run_result_hash=sha(out / 'result.json'),
        input_hash=sha(source / '온라인대회자료/정형데이터/참가자_배포/train_X.csv'),
        pooled=score(combined),
        by_farm={farm: score(g) for farm, g in combined.groupby('farm')},
        by_regime={str(k): score(g) for k, g in combined.groupby('sealed')},
        by_farm_regime={f'{farm}_{flag}': score(g)
                        for (farm, flag), g in combined.groupby(['farm', 'sealed'])},
        days=dict(total=len(day), improved=int((day.candidate < day.baseline).sum()),
                  median_relative_change=float(np.median(day.candidate/day.baseline-1)),
                  top5_share_of_positive_gain=float(positive.head(5).sum()/positive.sum())
                  if len(positive) else None),
        bootstrap_day=resample(combined, 'day', 290928),
        bootstrap_five_day=resample(combined, 'block', 290929),
        note='Sealed flag uses entire-day inputs for diagnosis only, not for prediction.')
    (out / 'diagnostics.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2),
                                          encoding='utf-8')
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
