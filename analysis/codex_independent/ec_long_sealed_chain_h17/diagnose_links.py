"""Analysis-only 1-4 step chain accuracy against earlier label-checked proxy."""
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
    ap.add_argument('--proxy', required=True, type=Path)
    ap.add_argument('--memory', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    proxy = pd.read_csv(a.proxy)
    proxy = proxy[proxy.d_from.lt(proxy.d_to)].copy()
    pred = pd.read_csv(a.memory)
    true_map = {(r.farm, int(r.d_to)): int(r.d_from) for r in proxy.itertuples(index=False)}
    pred_map = {(r.farm, int(r.day)): None if pd.isna(r.link_day) else int(r.link_day)
                for r in pred.itertuples(index=False)}
    days = set()
    for fold in (0, 2, 4, 6, 8, 9):
        ids = pd.read_csv((SEARCH if fold < 8 else CONFIRM)/f'fold{fold}.csv', usecols=['row_id']).row_id
        days.update((s[:3], int(s[4:7])) for s in ids)
    stats = {}
    for depth in range(1, 5):
        n, hit = 0, 0
        for key in days:
            t, p = key, key
            valid = True
            for _ in range(depth):
                tf = true_map.get(t)
                pf = pred_map.get(p)
                if tf is None or pf is None:
                    valid = False
                    break
                t = (key[0], tf); p = (key[0], pf)
            if valid:
                n += 1
                hit += int(t == p)
        stats[str(depth)] = {'comparable_oof_days': n, 'hits': hit,
                             'hit_rate': hit/n if n else None}
    result = {'proxy_sha256': hashlib.sha256(a.proxy.read_bytes()).hexdigest(),
              'memory_sha256': hashlib.sha256(a.memory.read_bytes()).hexdigest(),
              'depth_accuracy': stats,
              'caveat': 'Proxy source links are label-checked prior analysis, not true IDs; read only after H17 scores.'}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
