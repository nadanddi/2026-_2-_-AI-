"""H12 analysis-only comparison to an earlier label-checked source-link proxy."""
import env  # first project import

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / '온라인대회자료/정형데이터/참가자_배포'
SEARCH = ROOT / 'analysis/local/ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM = ROOT / 'analysis/local/ec_locked_confirmation/20260928_044934'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--links', required=True, type=Path)
    ap.add_argument('--predicted', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    proxy = pd.read_csv(a.links)
    proxy = proxy[proxy.d_from.lt(proxy.d_to)].copy()
    pred = pd.read_csv(a.predicted)
    assert len(pred) == 400 and pred[['farm', 'day']].drop_duplicates().shape[0] == 400
    ids = []
    for fold in (0, 2, 4, 6, 8, 9):
        p = (SEARCH if fold < 8 else CONFIRM) / f'fold{fold}.csv'
        ids.extend(pd.read_csv(p, usecols=['row_id']).row_id)
    oof = pd.DataFrame({'row_id': ids})
    oof['farm'] = oof.row_id.str[:3]
    oof['day'] = oof.row_id.str[4:7].astype(int)
    oof = oof[['farm', 'day']].drop_duplicates()
    y = pd.read_csv(DATA / 'train_y.csv', usecols=['row_id', 'sub_ec'])
    y['farm'] = y.row_id.str[:3]
    y['day'] = y.row_id.str[4:7].astype(int)
    high = y[y.farm.isin(('F13', 'F47'))].groupby(['farm', 'day']).sub_ec.mean().ge(1.2).rename('high').reset_index()
    z = proxy.merge(oof, left_on=['farm', 'd_to'], right_on=['farm', 'day'])
    z = z.merge(pred[['farm', 'day', 'link_day']], on=['farm', 'day'], validate='one_to_one')
    z = z.merge(high, on=['farm', 'day'], validate='one_to_one')
    z['covered'] = z.link_day.notna()
    z['hit'] = z.link_day.eq(z.d_from)
    z['period'] = np.where(z.day < 179, 'early', 'late')
    def stat(g):
        return {'n': int(len(g)), 'covered': int(g.covered.sum()),
                'hits': int(g.hit.sum()),
                'coverage': float(g.covered.mean()),
                'accuracy_all': float(g.hit.mean()),
                'accuracy_when_covered': float(g.loc[g.covered, 'hit'].mean()) if g.covered.any() else None,
                'fixed_dminus2_hits': int(g.d_from.eq(g.day-2).sum())}
    result = {'proxy_sha256': hashlib.sha256(a.links.read_bytes()).hexdigest(),
              'predicted_sha256': hashlib.sha256(a.predicted.read_bytes()).hexdigest(),
              'overall': stat(z),
              'by_farm_period': {f'{f}_{p}': stat(g) for (f, p), g in z.groupby(['farm', 'period'])},
              'by_high_day': {str(bool(h)): stat(g) for h, g in z.groupby('high')},
              'caveat': 'The proxy is label-checked but not true source ID. It was read only after H12 scores were fixed.'}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
