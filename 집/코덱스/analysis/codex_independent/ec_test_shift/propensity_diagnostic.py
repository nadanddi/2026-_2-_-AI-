"""Post-hoc covariate-shift check, using only unlabeled daily input summaries.

This is an overlap diagnostic, not a model-selection gate or estimate of the
hidden leaderboard RMSE. Full-day summaries include future inputs and must
never be used as prediction features.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ['AGRI_SOURCE_ROOT']).resolve()
DATA = SOURCE / '온라인대회자료/정형데이터/참가자_배포'
SEARCH = ROOT / 'local/ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM = ROOT / 'local/ec_locked_confirmation/20260928_044934'
FEATURE_SETS = {
    'simple': ['in_temp_mean', 'sealed'],
    'expanded': ['in_temp_mean', 'in_temp_min', 'sealed',
                 'act_circfan_mean', 'act_vent_zero', 'act_heating_mean',
                 'in_co2_roughness'],
}


def daily(path, is_test):
    cols = ['row_id', 'in_temp', 'in_co2', 'act_circfan', 'act_vent', 'act_heating']
    x = pd.read_csv(path, usecols=cols)
    x['farm'] = x.row_id.str[:3]
    x['day'] = x.row_id.str[4:7].astype(int)
    x = x[x.farm.isin(['F13', 'F47'])].copy()
    x['hour'] = x.row_id.str[-2:].astype(int)
    x = x.sort_values(['farm', 'day', 'hour'])
    g = x.groupby(['farm', 'day'], sort=True)
    out = g.agg(in_temp_mean=('in_temp', 'mean'),
                in_temp_min=('in_temp', 'min'),
                act_circfan_mean=('act_circfan', 'mean'),
                act_vent_zero=('act_vent', lambda s: float((s == 0).mean())),
                act_heating_mean=('act_heating', 'mean')).reset_index()
    rough = g.in_co2.apply(lambda s: float(s.diff().abs().median())).rename('in_co2_roughness')
    out = out.merge(rough.reset_index(), on=['farm', 'day'], validate='one_to_one')
    out['sealed'] = ((out.act_circfan_mean < 10) & (out.act_vent_zero > .85)).astype(int)
    out['is_test'] = int(is_test)
    assert g.size().eq(24).all(), 'Expected exactly 24 rows per greenhouse-day'
    assert out[FEATURE_SETS['expanded']].notna().all().all()
    return out


def predictions():
    files = [(SEARCH / f'fold{i}.csv', 'search') for i in (0, 2, 4, 6)]
    files += [(CONFIRM / f'fold{i}.csv', 'confirmation') for i in (8, 9)]
    frames = []
    for path, split in files:
        frame = pd.read_csv(path, usecols=['row_id', 'sub_ec', 'baseline',
                                           'blend' if split == 'search' else 'candidate'])
        frame = frame.rename(columns={'blend': 'candidate'})
        frame['farm'] = frame.row_id.str[:3]
        frame['day'] = frame.row_id.str[4:7].astype(int)
        frame['split'] = split
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    assert out.row_id.is_unique
    assert out.groupby(['farm', 'day']).size().eq(24).all()
    return out


def score(frame, weight):
    w = np.asarray(weight, float)
    y = frame.sub_ec.to_numpy(float)
    b = frame.baseline.to_numpy(float)
    c = frame.candidate.to_numpy(float)
    eb = float(np.sqrt(np.average((y - b)**2, weights=w)))
    ec = float(np.sqrt(np.average((y - c)**2, weights=w)))
    days = frame[['farm', 'day']].drop_duplicates().shape[0]
    day_weights = frame[['farm', 'day']].assign(weight=w).groupby(['farm', 'day']).weight.first().to_numpy()
    ess = float(day_weights.sum()**2 / np.square(day_weights).sum())
    return {'days': days, 'rows': len(frame), 'effective_days': ess,
            'baseline': eb, 'candidate': ec, 'relative_change': ec / eb - 1}


def gain_concentration(frame, weight):
    a = frame[['farm', 'day', 'sub_ec', 'baseline', 'candidate']].copy()
    a['weight'] = np.asarray(weight, float)
    a['gain'] = a.weight * ((a.sub_ec - a.baseline)**2 -
                            (a.sub_ec - a.candidate)**2)
    by_day = a.groupby(['farm', 'day']).gain.sum().sort_values(ascending=False)
    positive = by_day[by_day > 0]
    top = positive.head(5)
    rest = a.set_index(['farm', 'day']).drop(index=top.index).reset_index()
    return {
        'improved_days': int((by_day > 0).sum()),
        'top5_share_positive_gain': float(top.sum() / positive.sum()),
        'relative_change_excluding_top5': score(rest, rest.weight)['relative_change'],
    }


def evaluate(daily_all, pred, cols):
    x = daily_all[cols].to_numpy(float)
    y = daily_all.is_test.to_numpy(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(
        C=1., max_iter=1000, class_weight='balanced'))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260928)
    probability = cross_val_predict(clf, x, y, cv=cv, method='predict_proba')[:, 1]
    auc = float(roc_auc_score(y, probability))
    clf.fit(x, y)
    odds = clf.predict_proba(x)[:, 1]
    odds = np.clip(odds, .01, .99) / (1 - np.clip(odds, .01, .99))
    train = daily_all[~daily_all.is_test.astype(bool)][['farm', 'day']].copy()
    train['weight'] = odds[~daily_all.is_test.astype(bool)]
    merged = pred.merge(train, on=['farm', 'day'], how='left', validate='many_to_one')
    assert merged.weight.notna().all()
    groups = {}
    for split in ('search', 'confirmation', 'all'):
        a = merged if split == 'all' else merged[merged.split == split]
        groups[split] = {
            'unweighted': score(a, np.ones(len(a))),
            'weighted': score(a, a.weight.clip(upper=10)),
            'weighted_gain_concentration': gain_concentration(a, a.weight.clip(upper=10)),
            'weight_max_before_cap': float(a.weight.max()),
            'weighted_by_farm': {
                farm: score(part, part.weight.clip(upper=10))
                for farm, part in a.groupby('farm', sort=True)
            },
        }
    return {'input_only_oof_auc': auc, 'groups': groups,
            'train_weight_quantiles': train.weight.quantile([0, .25, .5, .75, .9, .99, 1]).to_dict()}


def main():
    train = daily(DATA / 'train_X.csv', False)
    test = daily(DATA / 'test_X.csv', True)
    assert len(train) == 400 and len(test) == 60
    daily_all = pd.concat([train, test], ignore_index=True)
    pred = predictions()
    output = {
        'status': 'POST_HOC_UNLABELED_COVARIATE_SHIFT_DIAGNOSTIC',
        'feature_sets': {name: evaluate(daily_all, pred, cols)
                         for name, cols in FEATURE_SETS.items()},
        'note': 'Full-day summaries use future inputs: diagnostic only. '
                'Covariate reweighting assumes conditional target behavior transfers; '
                'unobserved irrigation and label shift may violate that assumption. '
                'Do not use for candidate selection or hidden-score claims.',
    }
    dest = ROOT / 'local/ec_test_propensity' / datetime.now().strftime('%Y%m%d_%H%M%S')
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'result.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
