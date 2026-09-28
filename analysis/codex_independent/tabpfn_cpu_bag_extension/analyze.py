"""Describe farm and five-day-block uncertainty for fixed saved predictions."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
OLD = ROOT / 'local/tabpfn_cpu_bag/20260927_182052'
NEW = Path(sys.argv[1]).resolve()
assert NEW.is_dir() and ROOT / 'local/tabpfn_cpu_bag_extension' in NEW.parents

frames = []
for fold in (0, 2, 4, 6):
    path = (OLD if fold in (0, 2) else NEW) / f'fold{fold}_bag.csv'
    frame = pd.read_csv(path)
    frame['fold'] = fold
    frames.append(frame)
all_rows = pd.concat(frames, ignore_index=True)
assert all_rows.row_id.is_unique
assert all_rows[['farm', 'day', 'block']].notna().all().all()


def score(rows):
    result = {}
    for seed in (7, 101):
        truth = rows.sub_ec.to_numpy(float)
        base = rows[f'baseline_{seed}'].to_numpy(float)
        blend = rows[f'blend_{seed}'].to_numpy(float)
        a = np.sqrt(np.mean((truth - base) ** 2))
        b = np.sqrt(np.mean((truth - blend) ** 2))
        result[str(seed)] = dict(baseline=float(a), candidate=float(b),
                                 relative_change=float(b/a-1))
    return result


by_farm = {farm: score(group) for farm, group in all_rows.groupby('farm')}
by_fold_farm = {f'{fold}_{farm}': score(group)
                for (fold, farm), group in all_rows.groupby(['fold', 'farm'])}

def bootstrap(group_col, seed):
    # Fixed-prediction resampling; training variance is deliberately excluded.
    rows = all_rows.copy()
    for model_seed in (7, 101):
        rows[f'base_sq_{model_seed}'] = (rows.sub_ec - rows[f'baseline_{model_seed}']) ** 2
        rows[f'blend_sq_{model_seed}'] = (rows.sub_ec - rows[f'blend_{model_seed}']) ** 2
    cols = [f'{kind}_sq_{model_seed}' for kind in ('base', 'blend') for model_seed in (7, 101)]
    groups = {farm: group.groupby(group_col)[cols].sum().to_numpy(float)
              for farm, group in rows.groupby('farm')}
    rng = np.random.default_rng(seed)
    boot = np.empty(3000)
    for repeat in range(len(boot)):
        sampled = sum((table[rng.integers(0, len(table), size=len(table))].sum(axis=0)
                       for table in groups.values()))
        boot[repeat] = np.mean([
            np.sqrt(sampled[2+i] / sampled[i]) - 1 for i in (0, 1)])
    return dict(method=f'farm-stratified {group_col} blocks, fixed saved predictions',
                group_counts={farm: len(table) for farm, table in groups.items()},
                draws=len(boot), seed=seed,
                mean_relative_change=float(np.mean(boot)),
                ci95=[float(x) for x in np.quantile(boot, [.025, .975])],
                ci99=[float(x) for x in np.quantile(boot, [.005, .995])],
                probability_improvement=float(np.mean(boot < 0)))

output = dict(rows=len(all_rows), pooled=score(all_rows), by_farm=by_farm,
              by_fold_farm=by_fold_farm,
              bootstrap_day=bootstrap('day', 62026),
              bootstrap_five_day=bootstrap('block', 62027))
(NEW / 'block_analysis.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(output, ensure_ascii=False, indent=2))
