"""D7: target-day nearest neighbors using only input prefixes."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / '온라인대회자료/정형데이터/참가자_배포'
LOCAL = ROOT / 'analysis/local'
SEARCH = LOCAL / 'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM = LOCAL / 'ec_locked_confirmation/20260928_044934'
LOCK = HERE.parent / 'ec_final_lock/locked_days.json'
INPUTS = ['in_temp', 'in_hum', 'in_co2', 'act_vent', 'act_circfan', 'act_co2']
INDOOR = INPUTS[:3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame():
    paths = [*(SEARCH / f'fold{i}.csv' for i in (0, 2, 4, 6)),
             CONFIRM / 'fold8.csv', CONFIRM / 'fold9.csv']
    oof = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    oof['v2'] = oof['blend'].where(oof['blend'].notna(), oof['candidate']) if 'candidate' in oof else oof['blend']
    assert len(oof) == 5616 and oof.row_id.is_unique
    lock = {(d['farm'], d['day']) for d in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    assert not any((f, int(d)) in lock for f, d in zip(oof.farm, oof.day))

    x = pd.read_csv(DATA / 'train_X.csv', usecols=['row_id', *INPUTS])
    x = x[x.row_id.isin(oof.row_id)].copy()
    assert len(x) == len(oof)
    x['farm'] = x.row_id.str[:3]
    x['day'] = x.row_id.str[4:7].astype(int)
    x['hour'] = x.row_id.str[8:10].astype(int)
    x = x[x.hour.le(6)]
    assert len(x) == 234 * 7
    mean = x.groupby(['farm', 'day'], sort=True)[INPUTS].mean().add_prefix('avg_')
    h0 = x[x.hour.eq(0)].set_index(['farm', 'day'])[INDOOR].add_prefix('h0_')
    h6 = x[x.hour.eq(6)].set_index(['farm', 'day'])[INDOOR].add_prefix('h6_')
    features = pd.concat([mean, h0, h6], axis=1).reset_index()
    features['day_scaled'] = features.day / 30
    y = oof.groupby(['farm', 'day'], as_index=False).agg(
        y=('sub_ec', 'mean'), p=('v2', 'mean'), count=('row_id', 'size'))
    assert len(y) == 234 and y['count'].eq(24).all()
    return features.merge(y, on=['farm', 'day'], validate='one_to_one'), paths


def match(a):
    result = []
    cols = [c for c in a if c.startswith(('avg_', 'h0_', 'h6_'))] + ['day_scaled']
    for farm, group in a.groupby('farm'):
        g = group.sort_values('day').reset_index(drop=True)
        raw = g[cols].to_numpy(float)
        med = np.nanmedian(raw, axis=0)
        raw = np.where(np.isfinite(raw), raw, med)
        iqr = np.nanquantile(raw, .75, axis=0) - np.nanquantile(raw, .25, axis=0)
        sd = np.std(raw, axis=0)
        scale = np.where(iqr > 0, iqr, np.where(sd > 0, sd, 1.))
        z = (raw - med) / scale
        days = g.day.to_numpy(int)
        distance = np.sqrt(np.mean((z[:, None, :] - z[None, :, :]) ** 2, axis=2))
        allowed = (np.abs(days[:, None] - days[None, :]) >= 2) & (np.abs(days[:, None] - days[None, :]) <= 30)
        assert allowed.any(axis=1).all()
        nearest = np.where(allowed, distance, np.inf).argmin(axis=1)
        for i, j in enumerate(nearest):
            result.append(dict(farm=farm, day=int(days[i]), match_day=int(days[j]),
                               distance=float(distance[i, j]), y=float(g.y[i]),
                               match_y=float(g.y[j]), p=float(g.p[i]),
                               match_p=float(g.p[j]), y_gap=float(abs(g.y[i] - g.y[j])),
                               residual_gap=float(abs((g.y[i] - g.p[i]) - (g.y[j] - g.p[j])))))
    return pd.DataFrame(result)


def random_reference(a):
    rng = np.random.default_rng(2907)
    farm_data = []
    for _, g in a.groupby('farm'):
        days = g.day.to_numpy(int)
        allowed = (np.abs(days[:, None] - days[None, :]) >= 2) & (np.abs(days[:, None] - days[None, :]) <= 30)
        farm_data.append((g.y.to_numpy(float), [np.flatnonzero(row) for row in allowed]))
    medians = []
    for _ in range(10000):
        gaps = []
        for y, allowed in farm_data:
            js = [rng.choice(candidates) for candidates in allowed]
            gaps.extend(np.abs(y - y[js]))
        medians.append(float(np.median(gaps)))
    return np.asarray(medians)


def main():
    a, paths = frame()
    pairs = match(a)
    cutoff = float(pairs.distance.quantile(.25))
    close = pairs[pairs.distance.le(cutoff)]
    large = close[close.y_gap.ge(.4)].sort_values('y_gap', ascending=False)
    ref = random_reference(a)
    result = dict(days=len(a), directed_pairs=len(pairs),
                  unique_pairs=len({(r.farm, min(r.day, r.match_day), max(r.day, r.match_day))
                                    for r in pairs.itertuples()}),
                  distance_quartiles=[float(x) for x in pairs.distance.quantile([.25, .5, .75])],
                  y_gap_median=float(pairs.y_gap.median()),
                  y_gap_p90=float(pairs.y_gap.quantile(.9)),
                  close_pairs=len(close), close_large_gap=len(large),
                  close_large_gap_fraction=float(len(large) / len(close)),
                  close_large_gap_by_farm=large.farm.value_counts().to_dict(),
                  examples=large.head(12).to_dict('records'),
                  random_median_gap_ci95=[float(x) for x in np.quantile(ref, [.025, .975])],
                  random_median_gap_mean=float(np.mean(ref)),
                  random_fraction_at_or_below_observed=float(np.mean(ref <= pairs.y_gap.median())),
                  hashes={str(p): sha(p) for p in [HERE / 'PROTOCOL.md', HERE / 'run.py',
                                                   DATA / 'train_X.csv', LOCK, *paths]},
                  test_X_read=False, final_lock_labels_read=False,
                  hidden_labels_read=False, submission_created=False)
    out = LOCAL / 'ec_observational_aliasing' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    pairs.to_csv(out / 'pairs.csv', index=False)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
