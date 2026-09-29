"""Input-only midnight clusters versus stored EC out-of-fold day residuals."""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
SOURCE = Path(os.environ['AGRI_SOURCE_ROOT']).resolve()
DATA = SOURCE / '온라인대회자료/정형데이터/참가자_배포'
SEARCH = ROOT / 'local/ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM = ROOT / 'local/ec_locked_confirmation/20260928_044934'
COLS = ['in_temp', 'in_hum', 'in_co2', 'act_vent', 'act_shade',
        'act_thermal', 'act_heating', 'act_circfan', 'act_co2', 'act_fog']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clusters(train):
    first = train[train.row_id.str.endswith('_00')].copy()
    first['farm'] = first.row_id.str[:3]
    first['day'] = first.row_id.str[4:7].astype(int)
    first = first[first.farm.isin(('F13', 'F47'))]
    assert len(first) == 400 and first[['farm', 'day']].duplicated().sum() == 0
    parts = []
    for farm, group in first.groupby('farm', sort=True):
        matrix = SimpleImputer(strategy='median').fit_transform(group[COLS])
        matrix = StandardScaler().fit_transform(matrix)
        labels = KMeans(n_clusters=4, n_init=20, random_state=0).fit_predict(matrix)
        item = group[['farm', 'day']].copy()
        item['cluster'] = labels.astype(int)
        parts.append(item)
    return pd.concat(parts, ignore_index=True)


def within_section_stat(day, residual):
    a = day.copy()
    a['residual'] = residual
    section_mean = a.groupby(['farm', 'section']).residual.transform('mean')
    cluster_mean = a.groupby(['farm', 'section', 'cluster']).residual.transform('mean')
    total = float(np.sum((a.residual - section_mean) ** 2))
    between = float(np.sum((cluster_mean - section_mean) ** 2))
    return between / total


def main():
    train_path = DATA / 'train_X.csv'
    tr = pd.read_csv(train_path, usecols=['row_id', *COLS])
    cl = clusters(tr)
    files = [*[SEARCH / f'fold{fold}.csv' for fold in (0, 2, 4, 6)],
             CONFIRM / 'fold8.csv', CONFIRM / 'fold9.csv']
    frames = []
    for file in files:
        a = pd.read_csv(file)
        if 'blend' in a:
            a = a.rename(columns={'blend': 'candidate'})
        frames.append(a[['row_id', 'farm', 'day', 'sub_ec', 'baseline', 'candidate']])
    oof = pd.concat(frames, ignore_index=True)
    assert len(oof) == 5616 and oof.row_id.is_unique
    oof['base_resid'] = oof.sub_ec - oof.baseline
    oof['cand_resid'] = oof.sub_ec - oof.candidate
    day = oof.groupby(['farm', 'day'], as_index=False).agg(
        base_resid=('base_resid', 'mean'), cand_resid=('cand_resid', 'mean'),
        rows=('row_id', 'size'))
    assert len(day) == 234 and day.rows.eq(24).all()
    day = day.merge(cl, on=['farm', 'day'], validate='one_to_one')
    day['section'] = np.where(day.day < 179, 'first', 'second')
    rng = np.random.default_rng(20260929)
    groups = [idx.to_numpy() for _, idx in
              day.reset_index().groupby(['farm', 'section'])['index']]
    # Each permutation preserves farm, section, and their cluster counts.
    metrics = {}
    for name in ('base_resid', 'cand_resid'):
        residual = day[name].to_numpy(float)
        observed = within_section_stat(day, residual)
        random_stats = np.empty(10000)
        for iteration in range(len(random_stats)):
            permuted = day.cluster.to_numpy().copy()
            for idx in groups:
                permuted[idx] = rng.permutation(permuted[idx])
            sample = day.copy()
            sample['cluster'] = permuted
            random_stats[iteration] = within_section_stat(sample, residual)
        metrics[name] = dict(explained_fraction=observed,
                             permutation_fraction_ge_observed=float(np.mean(random_stats >= observed)),
                             null_q95=float(np.quantile(random_stats, .95)))
    detail = day.groupby(['farm', 'section', 'cluster'], as_index=False).agg(
        days=('day', 'size'), base_bias=('base_resid', 'mean'),
        cand_bias=('cand_resid', 'mean'))
    enough = detail[(detail.section == 'second') & (detail.days >= 5) &
                    (detail.cand_bias.abs() >= .05)]
    qualified = set(enough.farm) == {'F13', 'F47'}
    v2 = metrics['cand_resid']
    result = dict(status='POST_HOC_SOURCE_SEASON_PROBE', public_oof_days=len(day),
                  midnight_features=COLS, clusters_per_farm=4,
                  metrics=metrics, cells=detail.to_dict('records'),
                  passes_screen=bool(v2['explained_fraction'] >= .05 and
                                     v2['permutation_fraction_ge_observed'] < .01 and qualified),
                  hidden_labels_read=False, test_input_read=False,
                  note='Exploratory; previous studies reused these public labels.',
                  sha256={str(p):sha(p) for p in [HERE, train_path, *files]})
    out = ROOT / 'local/ec_source_season_probe' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key != 'sha256'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
