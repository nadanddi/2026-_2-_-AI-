"""Post-hoc EC transfer diagnostic across stored search and confirmation predictions."""
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
TEST_X = SOURCE / '온라인대회자료/정형데이터/참가자_배포/test_X.csv'
SEARCH = ROOT / 'local/ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM = ROOT / 'local/ec_locked_confirmation/20260928_044934'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(frame):
    y = frame.sub_ec.to_numpy(float)
    a = float(np.sqrt(np.mean((y-frame.baseline.to_numpy(float))**2)))
    b = float(np.sqrt(np.mean((y-frame.candidate.to_numpy(float))**2)))
    return dict(rows=len(frame), days=frame[['farm','day']].drop_duplicates().shape[0],
                baseline=a, candidate=b, relative_change=b/a-1)


def summarize(frame):
    in_span = frame[frame.in_test_span]
    out_span = frame[~frame.in_test_span]
    return dict(all=score(frame), within_test_span=score(in_span),
                outside_test_span=score(out_span),
                within_by_farm={f:score(g) for f,g in in_span.groupby('farm')})


def main():
    test = pd.read_csv(TEST_X, usecols=['row_id'])
    test['farm'] = test.row_id.str[:3]
    test['day'] = test.row_id.str[4:7].astype(int)
    spans = test.groupby('farm').day.agg(['min','max']).to_dict('index')
    files = [(SEARCH / f'fold{f}.csv','search') for f in (0,2,4,6)] + [
             (CONFIRM / f'fold{f}.csv','confirmation') for f in (8,9)]
    frames=[]
    for file, split in files:
        raw = pd.read_csv(file)
        col = 'blend' if split == 'search' else 'candidate'
        a = raw[['row_id','farm','day','sub_ec','baseline',col]].rename(columns={col:'candidate'}).copy()
        a['source'] = split
        frames.append(a)
    pooled = pd.concat(frames, ignore_index=True)
    assert pooled.row_id.is_unique
    pooled['in_test_span'] = [spans[r.farm]['min'] <= r.day <= spans[r.farm]['max']
                              for r in pooled.itertuples()]
    output = dict(status='POST_HOC_TRANSFER_DIAGNOSTIC',
        code_hash=sha(HERE), test_input_hash=sha(TEST_X),
        prediction_hashes={file.name+'_'+split:sha(file) for file,split in files},
        test_day_spans={f:{k:int(v) for k,v in span.items()} for f,span in spans.items()},
        search=summarize(pooled[pooled.source.eq('search')]),
        confirmation=summarize(pooled[pooled.source.eq('confirmation')]),
        pooled_descriptive=summarize(pooled),
        note='Dates selected after observing published validation results. Search and confirmation share the historical public dataset; this is not independent evidence and must not tune the candidate.')
    out = ROOT / 'local/ec_test_period_all_folds' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'result.json').write_text(json.dumps(output, ensure_ascii=False, indent=2),
                                      encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
