"""Post-hoc sealed-day diagnostic; never used as an inference feature."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
SOURCE = Path(os.environ['AGRI_SOURCE_ROOT']).resolve()
DATA = SOURCE / '온라인대회자료/정형데이터/참가자_배포'
CV = ROOT / 'local/ec_three_seed_ensemble_cv/20260928_035914'


def identify(frame):
    a = frame.copy()
    a['farm'] = a.row_id.str[:3]
    a['day'] = a.row_id.str[4:7].astype(int)
    return a


def sealed_days(frame):
    a = identify(frame)
    g = a.groupby(['farm', 'day'])
    fans = g.act_circfan.mean()
    vents = g.act_vent.apply(lambda col: (col == 0).mean())
    return ((fans < 10) & (vents > .85)).rename('sealed').reset_index()


def scores(frame):
    truth = frame.sub_ec.to_numpy(float)
    baseline = frame.baseline.to_numpy(float)
    candidate = frame.blend.to_numpy(float)
    a = np.sqrt(np.mean((truth-baseline)**2))
    b = np.sqrt(np.mean((truth-candidate)**2))
    days = frame[['farm','day']].drop_duplicates()
    return dict(rows=len(frame),days=len(days),baseline=float(a),candidate=float(b),
                relative_change=float(b/a-1))


def main():
    train = pd.read_csv(DATA/'train_X.csv')
    train = train[train.row_id.str[:3].isin(('F13','F47'))]
    test = pd.read_csv(DATA/'test_X.csv')
    tr_sealed,te_sealed=sealed_days(train),sealed_days(test)
    cv=pd.concat([pd.read_csv(CV/f'fold{f}.csv') for f in (0,2,4,6)],ignore_index=True)
    cv=cv.merge(tr_sealed,on=['farm','day'],validate='many_to_one')
    assert cv.sealed.notna().all() and cv.row_id.is_unique
    by_regime={str(sealed):scores(g) for sealed,g in cv.groupby('sealed')}
    by_farm_regime={f'{farm}_{sealed}':scores(g)
                    for (farm,sealed),g in cv.groupby(['farm','sealed'])}
    day=cv.groupby(['farm','day','sealed']).apply(
        lambda g: float((((g.sub_ec-g.baseline)**2)-((g.sub_ec-g.blend)**2)).sum()),
        include_groups=False).rename('gain_sq').reset_index()
    top=day.sort_values('gain_sq',ascending=False).head(5)
    output=dict(status='DIAGNOSTIC',train_sealed_days=int(tr_sealed.sealed.sum()),
                train_total_days=len(tr_sealed),test_sealed_days=int(te_sealed.sealed.sum()),
                test_total_days=len(te_sealed),cv=by_regime,by_farm_regime=by_farm_regime,
                top5_positive_gain_days=dict(sealed_count=int(top.sealed.sum()),
                                             positive_total=float(top.gain_sq.sum()),
                                             days=[{'farm':r.farm,'day':int(r.day),'sealed':bool(r.sealed),
                                                    'gain_sq':float(r.gain_sq)} for r in top.itertuples()]),
                note='Full-day sealed flag uses future inputs for diagnosis only, never for predictions.')
    (CV/'regime_analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
