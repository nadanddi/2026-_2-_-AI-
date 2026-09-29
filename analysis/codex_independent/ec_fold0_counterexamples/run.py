"""D6: daywise H4 loss structure, especially fold 0; no new model."""
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
    a=pd.concat(frames,ignore_index=True)
    assert len(a)==5616 and a.row_id.is_unique
    raw=pd.read_csv(DATA/'train_X.csv',usecols=['row_id','in_temp','act_circfan','act_vent'])
    a=a.merge(raw,on='row_id',validate='one_to_one')
    a['gate']=(a.b-a.t).abs().ge(.15)&a.in_temp.ge(10)
    a['h4']=np.where(a.gate,.75*a.v2+.25*a.t,a.v2)
    a['delta_sq']=(a.sub_ec-a.h4)**2-(a.sub_ec-a.v2)**2
    a['t_minus_b']=a.t-a.b
    # Non-gated input summaries are diagnostic context, not prediction features.
    day=a.groupby(['farm','day','fold'],as_index=False).agg(
        rows=('row_id','size'),changed=('gate','sum'),
        delta_sse=('delta_sq','sum'),y=('sub_ec','mean'),p=('v2','mean'),
        t=('t','mean'),b=('b','mean'))
    changed=a[a.gate].groupby(['farm','day'],as_index=False).agg(
        changed_t=('in_temp','mean'),changed_fan=('act_circfan','mean'),
        changed_vent=('act_vent','mean'),changed_t_minus_b=('t_minus_b','mean'))
    day=day.merge(changed,on=['farm','day'],how='left',validate='one_to_one')
    folds=[]
    for fold,g in day.groupby('fold'):
        harms=g[g.delta_sse.gt(0)].sort_values('delta_sse',ascending=False)
        total_harm=float(harms.delta_sse.sum())
        folds.append(dict(fold=int(fold),days=len(g),changed_days=int(g.changed.gt(0).sum()),
                          harmful_days=len(harms),net_delta_sse=float(g.delta_sse.sum()),
                          total_positive_harm=total_harm,
                          top1_harm_share=float(harms.delta_sse.head(1).sum()/total_harm) if total_harm>0 else None,
                          top2_harm_share=float(harms.delta_sse.head(2).sum()/total_harm) if total_harm>0 else None))
    zero=day[day.fold.eq(0)].sort_values('delta_sse',ascending=False)
    result=dict(folds=folds,fold0_days=zero.to_dict('records'),
                test_input_read=False,lock_labels_read=False,submission_created=False)
    out=LOCAL/'ec_fold0_counterexamples'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    day.to_csv(out/'day_table.csv',index=False,float_format='%.17g')
    print(json.dumps({'folds':folds,'fold0_top_harm':zero.head(8).to_dict('records'),
                      'fold0_top_gain':zero.tail(5).to_dict('records')},
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
