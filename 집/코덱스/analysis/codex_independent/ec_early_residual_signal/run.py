"""D8: max-stat corrected early-input partial correlations with EC v2 day residual."""
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
FOLDS = (0, 2, 4, 6, 8, 9)
RAW_COLS = ['out_temp', 'out_hum', 'out_rad', 'out_wspd',
            'in_temp', 'in_hum', 'in_co2', 'act_shade',
            'act_heating', 'act_circfan', 'act_vent']
SIGNALS = ['out_temp_0', 'out_hum_0', 'out_rad_0', 'out_wspd_0',
           'in_temp_change', 'in_hum_change', 'in_co2_change',
           'in_temp_range', 'in_hum_range', 'in_co2_range',
           'act_shade_mean', 'act_heating_mean',
           'act_circfan_mean', 'act_vent_mean']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    paths = [(SEARCH if f < 8 else CONFIRM) / f'fold{f}.csv' for f in FOLDS]
    frames = []
    for f, path in zip(FOLDS, paths):
        z = pd.read_csv(path)
        z['v2'] = z['blend'] if f < 8 else z['candidate']
        frames.append(z[['row_id', 'farm', 'day', 'hour', 'sub_ec', 'v2']])
    oof = pd.concat(frames, ignore_index=True)
    assert len(oof) == 5616 and oof.row_id.is_unique
    lock = {(r['farm'], r['day']) for r in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    assert not any((f, int(d)) in lock for f, d in zip(oof.farm, oof.day))
    y = oof.groupby(['farm', 'day'], as_index=False).agg(y=('sub_ec', 'mean'), count=('row_id', 'size'))
    assert len(y) == 234 and y['count'].eq(24).all()
    h6p = oof[oof.hour.eq(6)][['farm', 'day', 'v2']]
    assert len(h6p) == 234
    day = y.merge(h6p, on=['farm', 'day'], validate='one_to_one')
    x = pd.read_csv(DATA / 'train_X.csv', usecols=['row_id', *RAW_COLS])
    x = x[x.row_id.isin(oof.row_id)].copy()
    assert len(x) == len(oof)
    x['farm'] = x.row_id.str[:3]
    x['day'] = x.row_id.str[4:7].astype(int)
    x['hour'] = x.row_id.str[8:10].astype(int)
    x = x[x.hour.le(6)].copy()
    assert len(x) == 234 * 7
    early = x.groupby(['farm', 'day'], sort=True)
    at0 = x[x.hour.eq(0)].set_index(['farm', 'day'])
    at6 = x[x.hour.eq(6)].set_index(['farm', 'day'])
    f = pd.DataFrame(index=at0.index)
    for name in RAW_COLS[:4]:
        f[name + '_0'] = at0[name]
    for name in RAW_COLS[4:7]:
        f[name + '_change'] = at6[name] - at0[name]
        f[name + '_range'] = early[name].max() - early[name].min()
    for name in RAW_COLS[7:]:
        f[name + '_mean'] = early[name].mean()
    assert list(f.columns) == SIGNALS
    day = day.merge(f.reset_index(), on=['farm', 'day'], validate='one_to_one')
    day['residual'] = day.y - day.v2
    day['section'] = (day.day >= 179).astype(float)
    return day, paths


def residualize(x, controls):
    beta = np.linalg.lstsq(controls, x, rcond=None)[0]
    return x - controls @ beta


def correlations(day):
    controls = np.column_stack([
        np.ones(len(day)), day.farm.eq('F47').to_numpy(float),
        day.section.to_numpy(float), day.day.to_numpy(float) / 100,
        day.v2.to_numpy(float), day.v2.to_numpy(float) ** 2])
    target = residualize(day.residual.to_numpy(float), controls)
    features = []
    included = []
    for name in SIGNALS:
        parts = []
        for _, g in day.groupby('farm', sort=False):
            raw = g[name].to_numpy(float)
            med = np.nanmedian(raw)
            med = med if np.isfinite(med) else 0.
            parts.append(pd.Series(np.where(np.isfinite(raw), raw, med), index=g.index))
        x = pd.concat(parts).sort_index().to_numpy(float)
        if np.std(x) == 0:
            continue
        r = residualize(x, controls)
        sd = np.linalg.norm(r)
        if sd < 1e-12:
            continue
        features.append(r / sd)
        included.append(name)
    assert included
    z = np.column_stack(features)
    rnorm = target / np.linalg.norm(target)
    corr = rnorm @ z
    by_farm = {}
    for farm, g in day.groupby('farm'):
        idx = g.index.to_numpy()
        yy = target[idx]
        xx = z[idx]
        yy = yy - yy.mean()
        xx = xx - xx.mean(axis=0)
        farm_corr = yy @ xx / (np.linalg.norm(yy) * np.linalg.norm(xx, axis=0))
        by_farm[farm] = dict(zip(included, [float(v) for v in farm_corr]))
    return target, z, included, corr, by_farm


def main():
    day, paths = load()
    residual, z, included, corr, by_farm = correlations(day)
    winner = int(np.argmax(np.abs(corr)))
    rng = np.random.default_rng(2910)
    groups = [np.array(idx, dtype=int) for idx in day.groupby(['farm', 'section']).indices.values()]
    max_abs = np.empty(10000)
    for i in range(10000):
        shuffled = residual.copy()
        for idx in groups:
            shuffled[idx] = residual[rng.permutation(idx)]
        max_abs[i] = np.max(np.abs((shuffled / np.linalg.norm(shuffled)) @ z))
    pmax = float(np.mean(max_abs >= abs(corr[winner])))
    strong = all(np.sign(by_farm[f][included[winner]]) == np.sign(corr[winner])
                 and abs(by_farm[f][included[winner]]) >= .15 for f in ('F13', 'F47'))
    result = dict(days=len(day), signals_planned=len(SIGNALS), signals_tested=len(included),
                  tested=[dict(name=n, partial_corr=float(c), by_farm={f:by_farm[f][n]
                                                               for f in ('F13', 'F47')})
                          for n, c in zip(included, corr)],
                  winner=included[winner], winner_abs_corr=float(abs(corr[winner])),
                  max_stat_permutation_p=pmax, same_direction_farms=bool(strong),
                  h9_signal_gate=bool(pmax < .01 and strong),
                  permutation_draws=10000, permutation_seed=2910,
                  hashes={str(p):sha(p) for p in [HERE/'PROTOCOL.md', HERE/'run.py',
                                                 DATA/'train_X.csv', LOCK, *paths]},
                  test_X_read=False, final_lock_labels_read=False,
                  hidden_labels_read=False, submission_created=False)
    out = LOCAL/'ec_early_residual_signal'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'hashes'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
