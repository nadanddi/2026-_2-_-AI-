"""D4: fixed current CO2 actuator split within the D3 high-error gate."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
LOCAL=ROOT/'analysis/local'
DATA=ROOT/'온라인대회자료/정형데이터/참가자_배포'
SEARCH=LOCAL/'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM=LOCAL/'ec_locked_confirmation/20260928_044934'


def main():
    frames=[]
    for fold in (0,2,4,6,8,9):
        z=pd.read_csv((SEARCH if fold<8 else CONFIRM)/f'fold{fold}.csv')
        z['v2']=z['blend'] if fold<8 else z['candidate']
        z['fold']=fold
        frames.append(z[['row_id','farm','day','fold','sub_ec','v2']])
    oof=pd.concat(frames,ignore_index=True)
    assert len(oof)==5616 and oof.row_id.is_unique
    cols=['row_id','act_circfan','act_vent','act_co2']
    x=pd.read_csv(DATA/'train_X.csv',usecols=cols)
    x=oof[['row_id']].merge(x,on='row_id',validate='one_to_one')
    z=oof.merge(x,on='row_id',validate='one_to_one')
    z=z[z.act_circfan.lt(10)&z.act_vent.eq(0)&z.v2.ge(.6)].copy()
    assert len(z)==965
    z['co2_on']=z.act_co2.gt(0)
    z['residual']=z.sub_ec-z.v2
    cells=[]
    for on,g in z.groupby('co2_on'):
        cells.append(dict(co2_on=bool(on),rows=len(g),
                          days=int(g.groupby(['farm','day']).ngroups),
                          mean_residual=float(g.residual.mean()),
                          rmse=float(np.sqrt(np.mean(g.residual**2)))))
    daily=z.groupby(['farm','day','fold'],as_index=False).agg(
        residual=('residual','mean'),co2_fraction=('co2_on','mean'),n=('row_id','size'))
    daily['co2_majority']=daily.co2_fraction.ge(.5)
    assert len(daily)==56
    means=daily.groupby('co2_majority').residual.mean()
    if len(means)!=2:
        difference=None;ci=[None,None];pworse=None
    else:
        difference=float(means[True]-means[False])
        rng=np.random.default_rng(2903)
        a=daily[daily.co2_majority].residual.to_numpy()
        b=daily[~daily.co2_majority].residual.to_numpy()
        samples=np.empty(20000)
        for i in range(len(samples)):
            samples[i]=a[rng.integers(len(a),size=len(a))].mean()-b[rng.integers(len(b),size=len(b))].mean()
        ci=np.quantile(samples,[.025,.975]).tolist()
        pworse=float(np.mean(samples*difference<=0))
    groups=[]
    for kind,key,g in [*(('farm',f,g) for f,g in daily.groupby('farm')),
                       *(('fold',int(f),g) for f,g in daily.groupby('fold'))]:
        m=g.groupby('co2_majority').residual.mean()
        groups.append(dict(kind=kind,key=key,days=len(g),
                           co2_majority_days=int(g.co2_majority.sum()),
                           difference=(float(m[True]-m[False]) if len(m)==2 else None)))
    farm=[g for g in groups if g['kind']=='farm']
    folds=[g for g in groups if g['kind']=='fold']
    pass_gate=bool(difference is not None and min(daily.co2_majority.sum(),(~daily.co2_majority).sum())>=15
                   and abs(difference)>=.10 and ci[0]*ci[1]>0
                   and all(g['difference'] is not None and g['difference']*difference>0 for g in farm)
                   and sum(g['difference'] is not None and g['difference']*difference>0 for g in folds)>=4)
    result=dict(rows=len(z),days=len(daily),cells=cells,
                day_group_counts={str(k):int(v) for k,v in daily.co2_majority.value_counts().items()},
                day_mean_residual_difference=difference,
                day_bootstrap_difference_95ci=ci,
                day_bootstrap_opposite_probability=pworse,
                stratified=groups,proceed_to_h3=pass_gate,
                test_input_read=False,lock_labels_read=False,submission_created=False)
    out=LOCAL/'ec_co2_regime_diagnostic'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
