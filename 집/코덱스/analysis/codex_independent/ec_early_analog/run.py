"""H7 early-input nearest-day EC level, on prespecified OOF days."""
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
FOLDS = (0, 2, 4, 6, 8, 9)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def days_and_oof():
    x = pd.read_csv(DATA / 'train_X.csv', usecols=['row_id', *INPUTS])
    x = x[x.row_id.str[:3].isin(('F13', 'F47'))].copy()
    x['farm'] = x.row_id.str[:3]
    x['day'] = x.row_id.str[4:7].astype(int)
    x['hour'] = x.row_id.str[8:10].astype(int)
    assert len(x) == 9600 and x.row_id.is_unique
    y = pd.read_csv(DATA / 'train_y.csv', usecols=['row_id', 'sub_ec'])
    target = x[['row_id', 'farm', 'day']].merge(y, on='row_id', validate='one_to_one')
    target = target.groupby(['farm', 'day'], as_index=False).agg(y=('sub_ec', 'mean'), n=('row_id', 'size'))
    assert len(target) == 400 and target.n.eq(24).all() and target.y.notna().all()
    early = x[x.hour.le(6)].copy()
    avg = early.groupby(['farm', 'day'])[INPUTS].mean().add_prefix('avg_')
    h0 = early[early.hour.eq(0)].set_index(['farm', 'day'])[INDOOR].add_prefix('h0_')
    h6 = early[early.hour.eq(6)].set_index(['farm', 'day'])[INDOOR].add_prefix('h6_')
    days = pd.concat([avg, h0, h6], axis=1).reset_index()
    days['day_scaled'] = days.day / 30
    days = days.merge(target[['farm', 'day', 'y']], on=['farm', 'day'], validate='one_to_one')
    assert len(days) == 400
    frames, paths = [], []
    for fold in FOLDS:
        path = (SEARCH if fold < 8 else CONFIRM) / f'fold{fold}.csv'
        z = pd.read_csv(path)
        z['v2'] = z['blend'] if fold < 8 else z['candidate']
        z['fold'] = fold
        frames.append(z[['row_id', 'farm', 'day', 'hour', 'sub_ec', 'v2', 'fold']])
        paths.append(path)
    oof = pd.concat(frames, ignore_index=True)
    assert len(oof) == 5616 and oof.row_id.is_unique
    return days, oof, paths


def analog_levels(train, valid, cols):
    a = train[cols].to_numpy(float)
    b = valid[cols].to_numpy(float)
    med = np.nanmedian(a, axis=0)
    med = np.where(np.isfinite(med), med, 0.)
    a = np.where(np.isfinite(a), a, med)
    b = np.where(np.isfinite(b), b, med)
    iqr = np.quantile(a, .75, axis=0) - np.quantile(a, .25, axis=0)
    sd = np.std(a, axis=0)
    scale = np.where(iqr > 0, iqr, np.where(sd > 0, sd, 1.))
    a = (a - med) / scale
    b = (b - med) / scale
    distance = np.sqrt(np.mean((b[:, None, :] - a[None, :, :]) ** 2, axis=2))
    order = np.argsort(distance, axis=1, kind='stable')[:, :5]
    nearest = np.take_along_axis(distance, order, axis=1)
    weights = 1 / (nearest + .05)
    labels = train.y.to_numpy(float)[order]
    return np.sum(weights * labels, axis=1) / weights.sum(axis=1), nearest[:, 0]


def main():
    days, oof, paths = days_and_oof()
    cols = [c for c in days if c.startswith(('avg_', 'h0_', 'h6_'))] + ['day_scaled']
    lock = {(z['farm'], z['day']) for z in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    assert len(lock) == 40
    chunks, training = [], []
    for fold in FOLDS:
        v = oof[oof.fold.eq(fold)].copy()
        val = {(f, int(d)) for f, d in v[['farm', 'day']].itertuples(index=False, name=None)}
        assert not (val & lock)
        excluded = {(f, d + j) for f, d in val | lock for j in (-1, 0, 1)}
        train = days[[((f, int(d)) not in excluded) for f, d in
                      days[['farm', 'day']].itertuples(index=False, name=None)]]
        valid = days[[((f, int(d)) in val) for f, d in
                      days[['farm', 'day']].itertuples(index=False, name=None)]].copy()
        assert len(valid) == len(val)
        for farm in ('F13', 'F47'):
            tr = train[train.farm.eq(farm)]
            va = valid[valid.farm.eq(farm)]
            assert len(tr) >= 60 and len(va) > 0
            analog, distance = analog_levels(tr, va, cols)
            valid.loc[valid.farm.eq(farm), 'analog'] = analog
            valid.loc[valid.farm.eq(farm), 'distance'] = distance
        v = v.merge(valid[['farm', 'day', 'analog', 'distance']], on=['farm', 'day'], validate='many_to_one')
        gate = v.hour.ge(6) & v.distance.le(.15)
        v['h7'] = np.where(gate, .9 * v.v2 + .1 * v.analog, v.v2)
        assert np.isfinite(v[['v2', 'h7']].to_numpy(float)).all()
        chunks.append(v)
        training.append(dict(fold=fold, train_days=len(train), valid_days=len(valid),
                             gate_days=int(v.loc[gate, ['farm', 'day']].drop_duplicates().shape[0]),
                             gate_rows=int(gate.sum())))
    all_rows = pd.concat(chunks, ignore_index=True)
    sections = [('overall', all_rows)]
    sections += [(f'fold{f}', all_rows[all_rows.fold.eq(f)]) for f in FOLDS]
    sections += [(farm, all_rows[all_rows.farm.eq(farm)]) for farm in ('F13', 'F47')]
    metrics = {}
    for name, g in sections:
        base, cand = rmse(g.sub_ec, g.v2), rmse(g.sub_ec, g.h7)
        metrics[name] = dict(v2_rmse=base, h7_rmse=cand, relative_change=cand / base - 1)
    all_rows['delta_sq'] = (all_rows.sub_ec - all_rows.h7)**2 - (all_rows.sub_ec - all_rows.v2)**2
    delta = all_rows.groupby(['farm', 'day']).delta_sq.mean().to_numpy(float)
    assert len(delta) == 234
    rng = np.random.default_rng(2908)
    draws = np.empty(20000)
    for i in range(len(draws)):
        draws[i] = delta[rng.integers(len(delta), size=len(delta))].mean()
    ci = [float(x) for x in np.quantile(draws, [.025, .975])]
    passes = all(metrics[f'fold{f}']['relative_change'] < 0 for f in FOLDS)
    passes = passes and all(metrics[f]['relative_change'] < 0 for f in ('F13', 'F47')) and ci[1] < 0
    result = dict(training=training, metrics=metrics, bootstrap_mse_delta_95ci=ci,
                  bootstrap_p_worse=float(np.mean(draws >= 0)), screen_pass=bool(passes),
                  hashes={str(p): sha(p) for p in [HERE / 'PROTOCOL.md', HERE / 'run.py',
                                                   DATA / 'train_X.csv', DATA / 'train_y.csv', LOCK, *paths]},
                  final_lock_scored=False, test_X_read=False, hidden_labels_read=False,
                  submission_created=False)
    out = LOCAL / 'ec_early_analog' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'hashes'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
