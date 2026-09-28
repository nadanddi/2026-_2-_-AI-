"""Fixed H4 current-row disagreement gate on saved public EC v2 OOF."""
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
DATA=ROOT/'온라인대회자료/정형데이터/참가자_배포'


def rmse(g,col):
    return float(np.sqrt(np.mean((g.sub_ec-g[col])**2)))


def main():
    frames=[]
    for fold in (0,2,4,6,8,9):
        z=pd.read_csv((SEARCH if fold<8 else CONFIRM)/f'fold{fold}.csv')
        z['v2']=z['blend'] if fold<8 else z['candidate']
        z['b']=z['baseline']
        z['fold']=fold
        path=(FIRST if fold in (0,2) else EXT) if fold<8 else CONFIRM
        contexts=[]
        for seed in (1,2,3,4):
            c=pd.read_csv(path/f'fold{fold}_context{seed}.csv')
            assert c.row_id.tolist()==z.row_id.tolist()
            contexts.append(c.member.to_numpy(float))
        z['t']=np.mean(contexts,axis=0)
        frames.append(z[['row_id','farm','day','fold','sub_ec','b','t','v2']])
    x=pd.concat(frames,ignore_index=True)
    assert len(x)==5616 and x.row_id.is_unique
    temp=pd.read_csv(DATA/'train_X.csv',usecols=['row_id','in_temp'])
    x=x.merge(temp,on='row_id',validate='one_to_one')
    assert np.isfinite(x[['sub_ec','b','t','v2']].to_numpy(float)).all()
    x['gate']=(x.b-x.t).abs().ge(.15)&x.in_temp.ge(10)
    x['h4']=np.where(x.gate,.75*x.v2+.25*x.t,x.v2)
    assert np.isfinite(x.h4.to_numpy(float)).all()
    metrics={name:dict(rows=len(g),gate_rows=int(g.gate.sum()),
                       baseline=rmse(g,'v2'),candidate=rmse(g,'h4'),
                       relative_change=rmse(g,'h4')/rmse(g,'v2')-1)
             for name,g in [('all',x),
                            *[(f'fold{i}',x[x.fold.eq(i)]) for i in (0,2,4,6,8,9)],
                            *[(f,x[x.farm.eq(f)]) for f in ('F13','F47')],
                            ('gate',x[x.gate]) ]}
    x['delta_sq']=(x.sub_ec-x.h4)**2-(x.sub_ec-x.v2)**2
    days=x.groupby(['farm','day']).delta_sq.mean().to_numpy()
    assert len(days)==234
    rng=np.random.default_rng(2906)
    samples=np.empty(20000)
    for i in range(len(samples)):
        samples[i]=days[rng.integers(len(days),size=len(days))].mean()
    ci=np.quantile(samples,[.025,.975]).tolist()
    passed=all(metrics[f'fold{i}']['relative_change']<0 for i in (0,2,4,6,8,9))
    passed=passed and all(metrics[f]['relative_change']<0 for f in ('F13','F47')) and ci[1]<0
    result=dict(formula='if |b-t|>=0.15 and in_temp>=10: 0.75*v2+0.25*t; else v2',
                metrics=metrics,day_mse_delta_95ci=ci,
                day_bootstrap_p_worse=float(np.mean(samples>=0)),
                screen_pass=bool(passed),test_input_read=False,
                final_lock_scored=False,submission_created=False)
    out=LOCAL/'ec_disagreement_gate'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
