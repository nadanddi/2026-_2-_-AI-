"""D5: saved EC model disagreement and member error, no fitting."""
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


def summarize(g,total_sq):
    n=len(g)
    out=dict(rows=n,days=int(g.groupby(['farm','day']).ngroups),
             error_share_of_v2=float(np.sum((g.sub_ec-g.v2)**2)/total_sq))
    for col in ('b','t','v2'):
        e=g.sub_ec-g[col]
        out[f'{col}_rmse']=float(np.sqrt(np.mean(e**2)))
        out[f'{col}_residual']=float(e.mean())
    return out


def main():
    frames=[]
    for fold in (0,2,4,6,8,9):
        source=(SEARCH if fold<8 else CONFIRM)
        z=pd.read_csv(source/f'fold{fold}.csv')
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
        z['t_std']=np.std(contexts,axis=0)
        frames.append(z[['row_id','farm','day','fold','sub_ec','b','t','t_std','v2']])
    a=pd.concat(frames,ignore_index=True)
    assert len(a)==5616 and a.row_id.is_unique
    x=pd.read_csv(DATA/'train_X.csv',usecols=['row_id','in_temp'])
    a=a.merge(x,on='row_id',validate='one_to_one')
    assert np.isfinite(a[['sub_ec','b','t','v2']].to_numpy(float)).all()
    a['disagree']=(a.b-a.t).abs().ge(.15)
    total_sq=float(np.sum((a.sub_ec-a.v2)**2))
    groups={'all':a,'disagree':a[a.disagree],
            'agree':a[~a.disagree],
            'cold_disagree':a[a.disagree&a.in_temp.lt(10)],
            **{f'{farm}_disagree':a[a.disagree&a.farm.eq(farm)]
               for farm in ('F13','F47')},
            **{f'fold{fold}_disagree':a[a.disagree&a.fold.eq(fold)]
               for fold in (0,2,4,6,8,9)}}
    stats={name:summarize(g,total_sq) if len(g) else None
           for name,g in groups.items()}
    main=stats['disagree']
    farm=[stats[f'{f}_disagree'] for f in ('F13','F47')]
    folds=[stats[f'fold{i}_disagree'] for i in (0,2,4,6,8,9)]
    proceed=bool(main['rows']>=300 and main['days']>=25
                 and main['v2_rmse']-main['t_rmse']>=.01
                 and all(s and s['v2_rmse']-s['t_rmse']>=.01 for s in farm)
                 and sum(s is not None and s['t_rmse']<s['v2_rmse'] for s in folds)>=4)
    result=dict(threshold=.15,stats=stats,
                context_std_disagree_mean=float(a.loc[a.disagree,'t_std'].mean()),
                proceed_to_h4=proceed,test_input_read=False,
                final_lock_labels_read=False,submission_created=False)
    out=LOCAL/'ec_member_disagreement'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
