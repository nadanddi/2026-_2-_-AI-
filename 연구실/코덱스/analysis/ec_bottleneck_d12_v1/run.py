"""Read-only public EC diagnostics; no new model or final-lock scoring."""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').exists())
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env
import hashlib
import json
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = ROOT / '연구실/코덱스/local/ec_bottleneck_d12_v1' / datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
OUT.mkdir(parents=True, exist_ok=False)
HASHES = {}


def read(path):
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def identify(df):
    df = df.copy()
    parts = df.row_id.str.extract(r'^(F\d{2})_(\d{3})_(\d{2})$')
    assert parts.notna().all().all()
    df['farm'] = parts[0]
    df['day'] = parts[1].astype(int)
    df['hour'] = parts[2].astype(int)
    assert df.hour.between(0, 23).all()
    return df


for p in (HERE / 'PROTOCOL.md', HERE / 'run.py'):
    HASHES[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
data = Path(env.DATA)
raw = identify(read(data / 'train_X.csv'))
test = identify(read(data / 'test_X.csv'))
labels = read(data / 'train_y.csv')
sample = read(data / 'sample_submission.csv')
for z in (raw, test, labels, sample):
    assert z.row_id.is_unique and z.row_id.notna().all()
assert test.row_id.equals(sample.row_id)
assert not set(raw.row_id) & set(test.row_id)
assert set(labels.row_id).issubset(set(raw.row_id))
cols = [c for c in test if c not in ('row_id', 'farm', 'day', 'hour')]
mask = [c for c in cols if test[c].isna().all()]
target = raw[raw.farm.isin(['F13', 'F47'])]
availability = pd.DataFrame({
    'column': cols,
    'target_nonmissing': [int(target[c].notna().sum()) for c in cols],
    'test_nonmissing': [int(test[c].notna().sum()) for c in cols],
    'mask_excluded': [c in mask for c in cols],
})
availability.to_csv(OUT / 'availability.csv', index=False)
audit = dict(train_rows=len(raw), label_rows=len(labels), test_rows=len(test),
             train_farms=int(raw.farm.nunique()), target_input_rows=len(target),
             target_label_rows=int(labels.row_id.str[:3].isin(['F13', 'F47']).sum()),
             missing_label_ids=len(set(raw.row_id) - set(labels.row_id)),
             train_test_overlap=0, mask_excluded=mask,
             test_days=int(test.groupby(['farm', 'day']).ngroups),
             target_complete_days=int(target.groupby(['farm', 'day']).size().eq(24).sum()),
             target_total_days=int(target.groupby(['farm', 'day']).ngroups))
frames = []
old = ROOT / '집/코덱스/analysis/local'
for fold in (0, 2, 4, 6, 8, 9):
    folder = 'ec_three_seed_ensemble_cv/20260928_035914' if fold < 8 else 'ec_locked_confirmation/20260928_044934'
    z = read(old / folder / f'fold{fold}.csv')
    z['v2'] = z['blend' if fold < 8 else 'candidate']
    z['fold'] = fold
    frames.append(z[['row_id', 'farm', 'day', 'hour', 'sub_ec', 'baseline', 'v2', 'fold']])
v = pd.concat(frames, ignore_index=True)
assert len(v) == 5616 and v.row_id.is_unique
check = labels.set_index('row_id').loc[v.row_id, 'sub_ec'].to_numpy()
np.testing.assert_allclose(v.sub_ec, check, rtol=0, atol=1e-12)
assert np.isfinite(v[['sub_ec', 'baseline', 'v2']]).all().all()
daily = v.groupby(['farm', 'day']).agg(n=('row_id', 'size'), fold=('fold', 'first'),
    y=('sub_ec', 'mean'), baseline=('baseline', 'mean'), v2=('v2', 'mean'))
assert len(daily) == 234 and daily.n.eq(24).all()
daily['high'] = daily.y.ge(1.2)
decomp = []
for model in ('baseline', 'v2'):
    v['e_' + model] = v.sub_ec - v[model]
    stats = v.groupby(['farm', 'day'])['e_' + model].agg(['mean', lambda a: float(np.sum(a ** 2))])
    daily['bias_' + model] = stats['mean']
    daily['sse_' + model] = stats.iloc[:, 1]
    daily['level_' + model] = 24 * stats['mean'] ** 2
    daily['shape_' + model] = daily['sse_' + model] - daily['level_' + model]
    assert daily['shape_' + model].min() > -1e-10
    total = float(daily['sse_' + model].sum())
    level = float(daily['level_' + model].sum())
    shape = float(daily['shape_' + model].sum())
    assert abs(total - level - shape) < 1e-10
    decomp.append(dict(model=model, rmse=float(np.sqrt(total / len(v))),
        level_share=level / total, shape_share=shape / total,
        oracle_remove_level_rmse=float(np.sqrt(shape / len(v))),
        oracle_remove_shape_rmse=float(np.sqrt(level / len(v)))))
hours = []
for hour in (0, 6, 12, 23):
    early = v[v.hour.eq(hour)].set_index(['farm', 'day']).reindex(daily.index)
    daily[f'p{hour}'] = early.v2
    for name, g in [('all', daily), *list(daily.groupby(level='farm'))]:
        residual = g.y - g[f'p{hour}']
        hours.append(dict(hour=hour, farm=name, days=len(g),
            high_auc=float(roc_auc_score(g.high, g[f'p{hour}'])),
            high_mean_y=float(g.loc[g.high, 'y'].mean()),
            high_mean_prediction=float(g.loc[g.high, f'p{hour}'].mean()),
            high_mean_residual=float(residual[g.high].mean()),
            ordinary_mean_residual=float(residual[~g.high].mean()),
            day_level_proxy_rmse=float(np.sqrt(np.mean(residual ** 2)))))
calibration = []
for hour in (0, 6):
    selected = daily[daily[f'p{hour}'].ge(0.8)]
    groups = [('all', -1, selected)]
    groups += [(farm, int(fold), g) for (farm, fold), g in selected.groupby(['farm', 'fold'])]
    for farm, fold, g in groups:
        r = g.y - g[f'p{hour}']
        calibration.append(dict(hour=hour, farm=farm, fold=fold, days=len(g),
            high_days=int(g.high.sum()), mean_residual=float(r.mean()),
            positive_days=int(r.gt(0).sum()), negative_days=int(r.lt(0).sum())))
segments = []
for name, g in [('all', daily), ('high', daily[daily.high]), ('ordinary', daily[~daily.high])]:
    row = dict(segment=name, days=len(g))
    for m in ('baseline', 'v2'):
        row[m + '_rmse'] = float(np.sqrt(g['sse_' + m].sum() / (24 * len(g))))
        row[m + '_sse_share'] = float(g['sse_' + m].sum() / daily['sse_' + m].sum())
    row['sse_reduction'] = float(g.sse_baseline.sum() - g.sse_v2.sum())
    row['level_reduction'] = float(g.level_baseline.sum() - g.level_v2.sum())
    row['shape_reduction'] = float(g.shape_baseline.sum() - g.shape_v2.sum())
    segments.append(row)
pd.DataFrame(hours).to_csv(OUT / 'hours.csv', index=False)
pd.DataFrame(calibration).to_csv(OUT / 'early_calibration.csv', index=False)
pd.DataFrame(segments).to_csv(OUT / 'segments.csv', index=False)
daily.to_csv(OUT / 'daily.csv')
folds = []
for fold, g in daily.groupby('fold'):
    folds.append(dict(fold=int(fold), days=len(g), high_days=int(g.high.sum()),
        baseline=float(np.sqrt(g.sse_baseline.sum() / (24 * len(g)))),
        v2=float(np.sqrt(g.sse_v2.sum() / (24 * len(g)))),
        v2_level_share=float(g.level_v2.sum() / g.sse_v2.sum())))
pd.DataFrame(folds).to_csv(OUT / 'folds.csv', index=False)
result = dict(status='PASS', diagnostic_only=True, final_lock_scored=False,
    audit=audit, rows=len(v), days=len(daily), decomposition=decomp,
    segments=segments, early_hours=hours, early_calibration=calibration, folds=folds,
    sha256=HASHES)
(OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: result[k] for k in ('status', 'audit', 'decomposition', 'segments', 'folds')}, ensure_ascii=False, indent=2))
print('OUTPUT', OUT)
