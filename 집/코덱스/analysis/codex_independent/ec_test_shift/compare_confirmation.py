"""Compare unlabeled EC prediction movements on confirmation and test inputs."""
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
CV = ROOT / 'local/ec_locked_confirmation/20260928_044934'
TEST_SHIFT = ROOT / 'local/ec_test_shift/20260928_100001/result.json'
BASE = SOURCE / 'research/submissions/submission_07.csv'
CANDIDATE = ROOT / 'local/ec_tabpfn_local_candidate/20260928_032419/candidate_temp06_ec_tabpfn_v2_cpu.csv'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed_lookup(raw):
    raw = raw.copy()
    raw['farm'] = raw.row_id.str[:3]
    raw['day'] = raw.row_id.str[4:7].astype(int)
    g = raw.groupby(['farm', 'day'])
    result = ((g.act_circfan.mean() < 10) &
              (g.act_vent.apply(lambda s: (s == 0).mean()) > .85))
    return result.rename('sealed').reset_index()


def attach(frame, raw):
    a = frame.copy()
    a['farm'] = a.row_id.str[:3]
    a['day'] = a.row_id.str[4:7].astype(int)
    a = a.merge(sealed_lookup(raw), on=['farm', 'day'], validate='many_to_one')
    assert a.sealed.notna().all()
    return a


def movement(frame):
    d = frame.delta.to_numpy(float)
    return dict(rows=len(frame), days=frame[['farm','day']].drop_duplicates().shape[0],
                mean=float(d.mean()), rms=float(np.sqrt(np.mean(d*d))),
                share_abs_gt_005=float(np.mean(np.abs(d) > .05)))


def summarize(frame):
    return dict(overall=movement(frame),
                by_farm={f:movement(g) for f,g in frame.groupby('farm')},
                by_regime={str(flag):movement(g) for flag,g in frame.groupby('sealed')},
                by_farm_regime={f'{f}_{flag}':movement(g)
                                for (f,flag),g in frame.groupby(['farm','sealed'])})


def main():
    cv = pd.concat([pd.read_csv(CV / f'fold{fold}.csv',
                               usecols=['row_id','baseline','candidate']) for fold in (8,9)],
                   ignore_index=True)
    assert len(cv) == 1920 and cv.row_id.is_unique
    cv['delta'] = cv.candidate - cv.baseline
    train = pd.read_csv(DATA / 'train_X.csv')
    train = train[train.row_id.isin(cv.row_id)]
    assert len(train) == len(cv)
    cv = attach(cv[['row_id','delta']], train)

    base, candidate = pd.read_csv(BASE), pd.read_csv(CANDIDATE)
    test = pd.read_csv(DATA / 'test_X.csv')
    assert test.row_id.tolist() == base.row_id.tolist() == candidate.row_id.tolist()
    te = pd.DataFrame({'row_id':test.row_id,
                       'delta':candidate.sub_ec.to_numpy(float)-base.sub_ec.to_numpy(float)})
    te = attach(te, test)
    c, t = summarize(cv), summarize(te)
    ratio = {}
    for kind in ('overall','by_farm','by_regime','by_farm_regime'):
        if kind == 'overall':
            ratio['overall'] = float(t[kind]['rms']/c[kind]['rms'])
        else:
            ratio[kind] = {key:float(t[kind][key]['rms']/c[kind][key]['rms'])
                           for key in t[kind] if key in c[kind]}
    out = ROOT / 'local/ec_test_shift_confirmation_comparison' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status='UNLABELED_DIAGNOSTIC', code_hash=sha(HERE),
                  cv_file_hashes={str(f):sha(CV / f'fold{f}.csv') for f in (8,9)},
                  test_shift_result_hash=sha(TEST_SHIFT),
                  confirmation=c, test=t, test_to_confirmation_rms_ratio=ratio,
                  hidden_labels_used_for_comparison=False,
                  note='Confirmation predictions were originally scored, but this comparison uses only their prediction differences. Full-day sealed flag is diagnostic only.')
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
