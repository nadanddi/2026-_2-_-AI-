"""Fixed H5: replace four-context TabPFN mean by median, causal delta."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
LOCAL=ROOT/'analysis/local'
SEARCH=LOCAL/'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM=LOCAL/'ec_locked_confirmation/20260928_044934'
FIRST=LOCAL/'tabpfn_cpu_bag/20260927_182052'
EXT=LOCAL/'tabpfn_cpu_bag_extension/20260928_013953'


def rmse(g,c):
    return float(np.sqrt(np.mean((g.sub_ec-g[c])**2)))


def main():
    frames=[]
    for fold in (0,2,4,6,8,9):
        z=pd.read_csv((SEARCH if fold<8 else CONFIRM)/f'fold{fold}.csv')
        z['v2']=z['blend'] if fold<8 else z['candidate']
        z['fold']=fold
        path=(FIRST if fold in (0,2) else EXT) if fold<8 else CONFIRM
        members=[]
        for seed in (1,2,3,4):
            c=pd.read_csv(path/f'fold{fold}_context{seed}.csv')
            assert c.row_id.tolist()==z.row_id.tolist()
            members.append(c.member.to_numpy(float))
        a=np.stack(members,axis=1)
        z['delta']=np.median(a,axis=1)-np.mean(a,axis=1)
        frames.append(z[['row_id','farm','day','fold','sub_ec','v2','delta']])
    x=pd.concat(frames,ignore_index=True)
    assert len(x)==5616 and x.row_id.is_unique
    x['hour']=x.row_id.str[8:10].astype(int)
    x.sort_values(['farm','day','hour'],inplace=True)
    x['delta_prefix']=x.groupby(['farm','day']).delta.transform(
        lambda s:s.expanding().mean())
    x['h5']=x.v2+.2*(.5*x.delta+.5*x.delta_prefix)
    assert np.isfinite(x[['v2','h5']].to_numpy(float)).all()
    metrics={name:dict(rows=len(g),baseline=rmse(g,'v2'),candidate=rmse(g,'h5'),
                       relative_change=rmse(g,'h5')/rmse(g,'v2')-1)
             for name,g in [('all',x),
                            *[(f'fold{i}',x[x.fold.eq(i)]) for i in (0,2,4,6,8,9)],
                            *[(f,x[x.farm.eq(f)]) for f in ('F13','F47')]]}
    x['delta_sq']=(x.sub_ec-x.h5)**2-(x.sub_ec-x.v2)**2
    day=x.groupby(['farm','day']).delta_sq.mean().to_numpy()
    assert len(day)==234
    rng=np.random.default_rng(2907)
    sample=np.empty(20000)
    for i in range(len(sample)):
        sample[i]=day[rng.integers(len(day),size=len(day))].mean()
    ci=np.quantile(sample,[.025,.975]).tolist()
    passed=all(metrics[f'fold{i}']['relative_change']<0 for i in (0,2,4,6,8,9))
    passed=passed and all(metrics[f]['relative_change']<0 for f in ('F13','F47')) and ci[1]<0
    result=dict(metrics=metrics,day_mse_delta_95ci=ci,
                day_bootstrap_p_worse=float(np.mean(sample>=0)),
                mean_absolute_context_aggregate_change=float(x.delta.abs().mean()),
                screen_pass=bool(passed),test_input_read=False,
                final_lock_scored=False,submission_created=False)
    out=LOCAL/'ec_context_median'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
