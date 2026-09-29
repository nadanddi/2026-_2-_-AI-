"""Analysis-only H11 fixed-lag link audit; never feeds proxy links to a model."""
import env  # first project import

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SEARCH = ROOT / 'analysis/local/ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM = ROOT / 'analysis/local/ec_locked_confirmation/20260928_044934'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--links', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    links = pd.read_csv(a.links)
    links = links[links.d_from.lt(links.d_to)].copy()
    links['period'] = links.d_to.map(lambda d: 'early' if d < 179 else 'late')
    links['hit'] = links.d_from.eq(links.d_to - 2)
    oof_days = []
    for f in (0, 2, 4, 6, 8, 9):
        path = (SEARCH if f < 8 else CONFIRM) / f'fold{f}.csv'
        ids = pd.read_csv(path, usecols=['row_id']).row_id
        oof_days.extend((s[:3], int(s[4:7])) for s in ids)
    days = pd.DataFrame(oof_days, columns=['farm', 'd_to']).drop_duplicates()
    found = links.merge(days, on=['farm', 'd_to'])
    def stats(z):
        return {'n': int(len(z)), 'hits': int(z.hit.sum()), 'hit_rate': float(z.hit.mean())}
    result = {'link_sha256': hashlib.sha256(a.links.read_bytes()).hexdigest(),
              'all_proxy_links': stats(links), 'oof_proxy_links': stats(found),
              'oof_days': int(len(days)), 'linked_oof_days': int(found[['farm', 'd_to']].drop_duplicates().shape[0]),
              'by_farm_period': {f'{farm}_{period}': stats(g)
                  for (farm, period), g in found.groupby(['farm', 'period'])},
              'caveat': 'Proxy links were created in earlier analysis with label checks; they are not ground-truth source IDs and are never model features.'}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
