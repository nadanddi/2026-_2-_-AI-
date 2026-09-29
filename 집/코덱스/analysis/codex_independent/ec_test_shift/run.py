"""Unlabeled test-input distribution diagnostic for the frozen EC candidate."""
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
BASE = SOURCE / 'research/submissions/submission_07.csv'
CANDIDATE = ROOT / 'local/ec_tabpfn_local_candidate/20260928_032419/candidate_temp06_ec_tabpfn_v2_cpu.csv'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stats(frame):
    delta = frame.delta.to_numpy(float)
    return dict(rows=len(frame), days=frame[['farm', 'day']].drop_duplicates().shape[0],
                mean_delta=float(delta.mean()), rms_delta=float(np.sqrt(np.mean(delta**2))),
                median_delta=float(np.median(delta)),
                q05_delta=float(np.quantile(delta, .05)),
                q95_delta=float(np.quantile(delta, .95)),
                max_abs_delta=float(np.max(np.abs(delta))),
                share_candidate_higher=float(np.mean(delta > 0)),
                share_abs_delta_gt_005=float(np.mean(np.abs(delta) > .05)))


def main():
    test = pd.read_csv(DATA / 'test_X.csv')
    base = pd.read_csv(BASE)
    candidate = pd.read_csv(CANDIDATE)
    assert len(test) == len(base) == len(candidate) == 1440
    assert test.row_id.tolist() == base.row_id.tolist() == candidate.row_id.tolist()
    assert np.array_equal(base.sub_temp.to_numpy(float), candidate.sub_temp.to_numpy(float))
    test['farm'] = test.row_id.str[:3]
    test['day'] = test.row_id.str[4:7].astype(int)
    grouped = test.groupby(['farm', 'day'])
    sealed = ((grouped.act_circfan.mean() < 10) &
              (grouped.act_vent.apply(lambda s: (s == 0).mean()) > .85))
    sealed = sealed.rename('sealed').reset_index()
    a = test[['row_id', 'farm', 'day']].copy()
    a['delta'] = candidate.sub_ec.to_numpy(float) - base.sub_ec.to_numpy(float)
    a = a.merge(sealed, on=['farm', 'day'], validate='many_to_one')
    assert a.sealed.notna().all()
    by_day = a.groupby(['farm', 'day', 'sealed']).apply(
        lambda g: pd.Series({'mean_delta':float(g.delta.mean()),
                             'rms_delta':float(np.sqrt(np.mean(g.delta.to_numpy(float)**2))),
                             'max_abs_delta':float(np.max(np.abs(g.delta.to_numpy(float))))}),
        include_groups=False).reset_index()
    top = by_day.sort_values('rms_delta', ascending=False).head(8)
    result = dict(status='UNLABELED_DIAGNOSTIC',
        code_hash=sha(HERE), input_hash=sha(DATA / 'test_X.csv'),
        baseline_hash=sha(BASE), candidate_hash=sha(CANDIDATE),
        overall=stats(a), by_farm={f:stats(g) for f,g in a.groupby('farm')},
        by_regime={str(flag):stats(g) for flag,g in a.groupby('sealed')},
        by_farm_regime={f'{f}_{flag}':stats(g)
                        for (f,flag),g in a.groupby(['farm','sealed'])},
        top_rms_shift_days=[dict(farm=r.farm, day=int(r.day), sealed=bool(r.sealed),
                                 mean_delta=float(r.mean_delta), rms_delta=float(r.rms_delta),
                                 max_abs_delta=float(r.max_abs_delta)) for r in top.itertuples()],
        temperature_identical=True, hidden_labels_read=False,
        note='Sealed flag uses full-day future inputs only for diagnosis; shifts cannot establish RMSE direction.')
    out = ROOT / 'local/ec_test_shift' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
