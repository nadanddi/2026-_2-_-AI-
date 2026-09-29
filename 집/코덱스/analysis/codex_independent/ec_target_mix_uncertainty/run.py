"""Post-hoc day bootstrap standardized to unlabeled test regime counts."""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
SOURCE = Path(os.environ['AGRI_SOURCE_ROOT']).resolve()
DATA = SOURCE / '온라인대회자료/정형데이터/참가자_배포'
OOF = ROOT / 'local/ec_locked_confirmation/20260928_044934'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def regime(frame):
    frame = frame[['row_id', 'act_circfan', 'act_vent']].copy()
    frame = frame[frame.row_id.str[:3].isin(('F13', 'F47'))].copy()
    frame['farm'] = frame.row_id.str[:3]
    frame['day'] = frame.row_id.str[4:7].astype(int)
    days = frame.groupby(['farm', 'day'], sort=True)
    assert days.size().eq(24).all()
    seal = ((days.act_circfan.mean() < 10) &
            (days.act_vent.apply(lambda s: (s == 0).mean()) > .85))
    return seal.rename('sealed').reset_index()


def main():
    paths = [OOF / 'fold8.csv', OOF / 'fold9.csv']
    oof = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    assert len(oof) == 1920 and oof.row_id.is_unique
    assert set(oof.farm) == {'F13', 'F47'}
    train_path, test_path = DATA / 'train_X.csv', DATA / 'test_X.csv'
    train_regime = regime(pd.read_csv(train_path))
    test_regime = regime(pd.read_csv(test_path))
    val = oof.merge(train_regime, on=['farm', 'day'], validate='many_to_one')
    assert val.sealed.notna().all() and len(val) == 1920
    val['base_sq'] = (val.sub_ec - val.baseline) ** 2
    val['cand_sq'] = (val.sub_ec - val.candidate) ** 2
    days = val.groupby(['farm', 'day', 'sealed'], as_index=False).agg(
        base_sq=('base_sq', 'mean'), cand_sq=('cand_sq', 'mean'), rows=('row_id', 'size'))
    assert days.rows.eq(24).all() and len(days) == 80
    counts = test_regime.groupby(['farm', 'sealed']).size().to_dict()
    assert sum(counts.values()) == 60 and len(counts) == 4
    rng = np.random.default_rng(20260928)
    reps = 100000
    b = np.zeros(reps)
    c = np.zeros(reps)
    point_b = point_c = 0.0
    strata = []
    for (farm, sealed), target_days in sorted(counts.items()):
        group = days[(days.farm == farm) & (days.sealed == sealed)]
        assert len(group) > 0
        base = group.base_sq.to_numpy(float)
        cand = group.cand_sq.to_numpy(float)
        draw = rng.integers(0, len(group), size=(reps, int(target_days)))
        b += base[draw].sum(axis=1)
        c += cand[draw].sum(axis=1)
        point_b += int(target_days) * base.mean()
        point_c += int(target_days) * cand.mean()
        strata.append(dict(farm=farm, sealed=bool(sealed),
                           public_confirmation_days=len(group),
                           unlabeled_test_days=int(target_days),
                           public_baseline_rmse=float(np.sqrt(base.mean())),
                           public_candidate_rmse=float(np.sqrt(cand.mean()))))
    delta = np.sqrt(c / b) - 1
    raw = np.sqrt(days.cand_sq.mean() / days.base_sq.mean()) - 1
    result = dict(
        status='POST_HOC_STANDARDIZED_DAY_BOOTSTRAP', repetitions=reps,
        random_seed=20260928, public_confirmation_days=80,
        unlabeled_test_days=60, public_confirmation_change=float(raw),
        standardized_public_baseline_rmse=float(np.sqrt(point_b / 60)),
        standardized_public_candidate_rmse=float(np.sqrt(point_c / 60)),
        standardized_public_change=float(np.sqrt(point_c / point_b) - 1),
        bootstrap_change_quantiles={str(q):float(np.quantile(delta, q))
                                    for q in (.025, .5, .975)},
        bootstrap_fraction_worse=float(np.mean(delta > 0)),
        strata=strata, hidden_labels_read=False, test_prediction_created=False,
        caveat='Post-hoc; assumes within-farm/regime exchangeability; not a hidden-score CI.',
        sha256={str(path):sha256(path) for path in
                [HERE, *paths, train_path, test_path]})
    out = ROOT / 'local/ec_target_mix_uncertainty' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
