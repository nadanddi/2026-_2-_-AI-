"""Explain test-period validation heterogeneity using full-day diagnostic regimes."""
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
SEARCH = ROOT / 'local/ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM = ROOT / 'local/ec_locked_confirmation/20260928_044934'


def score(a):
    y = a.sub_ec.to_numpy(float)
    b = float(np.sqrt(np.mean((y-a.baseline.to_numpy(float))**2)))
    c = float(np.sqrt(np.mean((y-a.candidate.to_numpy(float))**2)))
    return dict(days=a[['farm','day']].drop_duplicates().shape[0], rows=len(a),
                baseline=b,candidate=c,relative_change=c/b-1)


def main():
    test = pd.read_csv(DATA/'test_X.csv',usecols=['row_id'])
    test['farm']=test.row_id.str[:3]
    test['day']=test.row_id.str[4:7].astype(int)
    spans=test.groupby('farm').day.agg(['min','max']).to_dict('index')
    train=pd.read_csv(DATA/'train_X.csv')
    train['farm']=train.row_id.str[:3]
    train['day']=train.row_id.str[4:7].astype(int)
    g=train.groupby(['farm','day'])
    sealed=((g.act_circfan.mean()<10)&(g.act_vent.apply(lambda s:(s==0).mean())>.85))
    sealed=sealed.rename('sealed').reset_index()
    frames=[]
    for fold in (0,2,4,6,8,9):
        split='search' if fold<8 else 'confirmation'
        file=(SEARCH if fold<8 else CONFIRM)/f'fold{fold}.csv'
        a=pd.read_csv(file)
        if fold<8:
            a=a.rename(columns={'blend':'candidate'})
        a=a[['row_id','farm','day','sub_ec','baseline','candidate']].copy()
        a['split']=split
        a['in_span']=[spans[r.farm]['min']<=r.day<=spans[r.farm]['max'] for r in a.itertuples()]
        frames.append(a)
    a=pd.concat(frames,ignore_index=True).merge(sealed,on=['farm','day'],validate='many_to_one')
    assert a.row_id.is_unique
    grouped={f'{split}_{span}_{regime}':score(group)
             for (split,span,regime),group in a.groupby(['split','in_span','sealed'])}
    output=dict(status='POST_HOC_TRANSFER_DIAGNOSTIC',groups=grouped,
                note='Regime uses full-day future inputs for diagnosis only; no candidate tuning or independent validation.')
    out=ROOT/'local/ec_test_period_regime'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
