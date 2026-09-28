"""Post-hoc EC check on confirmation days inside the unlabeled test day span."""
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(frame):
    y = frame.sub_ec.to_numpy(float)
    a = float(np.sqrt(np.mean((y-frame.baseline.to_numpy(float))**2)))
    b = float(np.sqrt(np.mean((y-frame.candidate.to_numpy(float))**2)))
    return dict(rows=len(frame), days=frame[['farm','day']].drop_duplicates().shape[0],
                baseline=a, candidate=b, relative_change=b/a-1,
                mean_prediction_shift=float((frame.candidate-frame.baseline).mean()))


def main():
    test = pd.read_csv(DATA / 'test_X.csv', usecols=['row_id'])
    test['farm'] = test.row_id.str[:3]
    test['day'] = test.row_id.str[4:7].astype(int)
    spans = test.groupby('farm').day.agg(['min','max']).to_dict('index')
    cv = pd.concat([pd.read_csv(CV / f'fold{f}.csv') for f in (8,9)], ignore_index=True)
    assert len(cv) == 1920 and cv.row_id.is_unique
    cv['within_test_day_span'] = [spans[r.farm]['min'] <= r.day <= spans[r.farm]['max']
                                  for r in cv.itertuples()]
    result = dict(status='POST_HOC_TRANSFER_DIAGNOSTIC',
        code_hash=sha(HERE), test_input_hash=sha(DATA / 'test_X.csv'),
        fold_hashes={str(f):sha(CV / f'fold{f}.csv') for f in (8,9)},
        test_day_span={f:{k:int(v) for k,v in span.items()} for f,span in spans.items()},
        pooled=score(cv),
        within_test_day_span=score(cv[cv.within_test_day_span]),
        outside_test_day_span=score(cv[~cv.within_test_day_span]),
        by_farm_within={f:score(g) for f,g in cv[cv.within_test_day_span].groupby('farm')},
        note='Same published confirmation labels, partitioned after seeing scores; no new independent test or model selection.')
    out = ROOT / 'local/ec_test_period_diagnostic' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                      encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
