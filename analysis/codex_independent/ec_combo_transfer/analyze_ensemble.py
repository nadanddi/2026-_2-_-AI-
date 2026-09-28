"""Descriptive average of the two stored EC combo validation seed predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'local/ec_combo_transfer'


def score(frame):
    y = frame.sub_ec.to_numpy(float)
    old = float(np.sqrt(np.mean((y - frame.v2.to_numpy(float))**2)))
    new = float(np.sqrt(np.mean((y - frame.combo.to_numpy(float))**2)))
    return {'days': frame[['farm', 'day']].drop_duplicates().shape[0],
            'rows': len(frame), 'v2': old, 'combo': new,
            'relative_change': new / old - 1}


def main():
    run = sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / 'result.json').exists())[-1]
    batches = []
    for fold in (8, 9):
        a = pd.read_csv(run / f'fold{fold}_seed11.csv')
        b = pd.read_csv(run / f'fold{fold}_seed12.csv')
        assert a.row_id.tolist() == b.row_id.tolist()
        assert np.array_equal(a.sub_ec.to_numpy(float), b.sub_ec.to_numpy(float))
        assert a.sealed.tolist() == b.sealed.tolist()
        m = a[['row_id', 'farm', 'day', 'fold', 'sub_ec', 'sealed']].copy()
        m['v2'] = (a.v2.to_numpy(float) + b.v2.to_numpy(float)) / 2
        m['combo'] = (a.combo.to_numpy(float) + b.combo.to_numpy(float)) / 2
        batches.append(m)
    x = pd.concat(batches, ignore_index=True)
    assert x.row_id.is_unique and len(x) == 1920
    spans = {'F13': (185, 239), 'F47': (183, 237)}
    x['test_span'] = [spans[r.farm][0] <= r.day <= spans[r.farm][1]
                      for r in x.itertuples()]
    result = {'status': 'POST_HOC_DESCRIPTIVE_ENSEMBLE', 'all': score(x)}
    for name in ('fold', 'farm', 'test_span', 'sealed'):
        result[name] = {str(k): score(g) for k, g in x.groupby(name)}
    x['gain'] = (x.sub_ec - x.v2)**2 - (x.sub_ec - x.combo)**2
    days = x.groupby(['farm', 'day']).gain.sum().sort_values(ascending=False)
    positive = days[days > 0]
    top5 = positive.head(5)
    rest = x.set_index(['farm', 'day']).drop(index=top5.index).reset_index()
    result['day_robustness'] = {
        'improved_days': int((days > 0).sum()),
        'top5_share_positive_gain': float(top5.sum() / positive.sum()),
        'without_top5': score(rest),
    }
    result['note'] = ('Computed after seeing per-seed results; descriptive only. '
                      'Uses public validation labels; no test labels or predictions.')
    (run / 'ensemble_diagnostic.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
