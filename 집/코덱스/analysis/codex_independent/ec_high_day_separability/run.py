"""OOF high-EC day ranking and calibration diagnostic, no model fitting."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[3]
R=ROOT/'analysis/local'
SEARCH=R/'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM=R/'ec_locked_confirmation/20260928_044934'
DATA=ROOT/'온라인대회자료/정형데이터/참가자_배포'


def oof_days():
    frames=[]
    for path in [*[SEARCH/f'fold{i}.csv' for i in (0,2,4,6)],
                 CONFIRM/'fold8.csv',CONFIRM/'fold9.csv']:
        z=pd.read_csv(path)
        z['v2']=z['blend'] if 'blend' in z else z['candidate']
        frames.append(z[['row_id','farm','day','sub_ec','v2']])
    rows=pd.concat(frames,ignore_index=True)
    assert len(rows)==5616 and rows.row_id.is_unique
    assert np.isfinite(rows[['sub_ec','v2']].to_numpy(float)).all()
    rows['hour']=rows.row_id.str[8:10].astype(int)
    first=rows[rows.hour.eq(0)][['farm','day','v2']].rename(columns={'v2':'p0'})
    day=rows.groupby(['farm','day'],as_index=False).agg(y=('sub_ec','mean'),p=('v2','mean'),
                                                        count=('row_id','size'))
    assert len(day)==234 and day['count'].eq(24).all()
    day=day.merge(first,on=['farm','day'],validate='one_to_one')
    x=pd.read_csv(DATA/'train_X.csv',usecols=['row_id','act_circfan','act_vent'])
    x=x[x.row_id.str[:3].isin(('F13','F47'))].copy()
    x['farm']=x.row_id.str[:3];x['day']=x.row_id.str[4:7].astype(int)
    reg=x.groupby(['farm','day'],as_index=False).agg(
        fan=('act_circfan','mean'),vent_zero=('act_vent',lambda v:float((v==0).mean())))
    reg['sealed']=reg.fan.lt(10)&reg.vent_zero.gt(.85)
    day=day.merge(reg[['farm','day','sealed']],on=['farm','day'],validate='one_to_one')
    day['section']=np.where(day.day<179,'first','second')
    day['high']=day.y.ge(1.2)
    day['residual']=day.y-day.p
    return day


def metrics(group,score):
    n=len(group);positive=int(group.high.sum());negative=n-positive
    out=dict(days=n,high_days=positive,prevalence=float(group.high.mean()),
             mean_y=float(group.y.mean()),mean_p=float(group[score].mean()))
    if positive>=3 and negative>=3:
        out['roc_auc']=float(roc_auc_score(group.high,group[score]))
        out['average_precision']=float(average_precision_score(group.high,group[score]))
        selected=group.nlargest(positive,score)
        out['top_k_recall']=float(selected.high.sum()/positive)
    else:
        out.update(roc_auc=None,average_precision=None,top_k_recall=None)
    return out


def main():
    d=oof_days()
    dimensions={'all':d,'F13':d[d.farm.eq('F13')],
                'F47':d[d.farm.eq('F47')],
                'first':d[d.section.eq('first')],
                'second':d[d.section.eq('second')],
                'sealed':d[d.sealed],
                'nonsealed':d[~d.sealed]}
    rank={name:{score:metrics(group,score) for score in ('p','p0')}
          for name,group in dimensions.items()}
    bins=[-np.inf,.6,.9,1.2,np.inf]
    labels=['lt_0.6','0.6_to_0.9','0.9_to_1.2','ge_1.2']
    d['p_bin']=pd.cut(d.p,bins=bins,labels=labels,right=False)
    calibration=d.groupby('p_bin',observed=False).agg(
        days=('day','size'),mean_prediction=('p','mean'),mean_label=('y','mean'),
        high_days=('high','sum'),high_rate=('high','mean'),mean_residual=('residual','mean'))
    high=d[d.high]
    result=dict(ranking=rank,calibration=calibration.reset_index().to_dict('records'),
                high_day_mean_label=float(high.y.mean()),
                high_day_mean_prediction=float(high.p.mean()),
                high_day_mean_residual=float(high.residual.mean()),
                calibration_hypothesis_selected=bool(rank['all']['p']['roc_auc']>=.85 and
                    rank['F13']['p']['roc_auc'] is not None and rank['F13']['p']['roc_auc']>=.75 and
                    rank['F47']['p']['roc_auc'] is not None and rank['F47']['p']['roc_auc']>=.75),
                hidden_labels_read=False,test_input_read=False,
                no_submission_file_created=True)
    out=R/'ec_high_day_separability'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
