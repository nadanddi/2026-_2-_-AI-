"""D3: a current-row ventilation gate in saved public EC v2 OOF."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT/'analysis/local'
DATA = ROOT/'온라인대회자료/정형데이터/참가자_배포'
SEARCH = LOCAL/'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM = LOCAL/'ec_locked_confirmation/20260928_044934'


def metrics(g, total_sq):
    err = g.sub_ec-g.v2
    return dict(rows=int(len(g)), days=int(g.groupby(['farm','day']).ngroups),
                mean_residual=float(err.mean()), rmse=float(np.sqrt(np.mean(err**2))),
                squared_error_share=float(np.sum(err**2)/total_sq))


def main():
    frames=[]
    for fold in (0,2,4,6,8,9):
        z=pd.read_csv((SEARCH if fold<8 else CONFIRM)/f'fold{fold}.csv')
        z['v2']=z['blend'] if fold<8 else z['candidate']
        z['fold']=fold
        frames.append(z[['row_id','farm','day','fold','sub_ec','v2']])
    oof=pd.concat(frames,ignore_index=True)
    assert len(oof)==5616 and oof.row_id.is_unique
    x=pd.read_csv(DATA/'train_X.csv',usecols=['row_id','act_circfan','act_vent'])
    x=x.merge(oof[['row_id']],on='row_id',validate='one_to_one')
    z=oof.merge(x,on='row_id',validate='one_to_one')
    assert len(z)==5616
    z['hour']=z.row_id.str[8:10].astype(int)
    z['gate']=z.act_circfan.lt(10)&z.act_vent.eq(0)
    z['p_high']=z.v2.ge(.6)
    total_sq=float(np.sum((z.sub_ec-z.v2)**2))
    cells=[]
    for (gate,p_high),g in z.groupby(['gate','p_high']):
        cells.append(dict(gate=bool(gate),p_high=bool(p_high),**metrics(g,total_sq)))
    stratified=[]
    for (farm,fold,gate,p_high),g in z.groupby(['farm','fold','gate','p_high']):
        stratified.append(dict(farm=farm,fold=int(fold),gate=bool(gate),
                               p_high=bool(p_high),**metrics(g,total_sq)))
    h0=z[z.hour.eq(0)][['farm','day','gate','p_high']]
    daily=z.groupby(['farm','day'],as_index=False).agg(
        y=('sub_ec','mean'),p=('v2','mean'),gate_fraction=('gate','mean'),
        fan=('act_circfan','mean'),vent_zero=('act_vent',lambda s:float(s.eq(0).mean())))
    daily=daily.merge(h0,on=['farm','day'],validate='one_to_one')
    daily['residual']=daily.y-daily.p
    daily['sealed_after_day']=daily.fan.lt(10)&daily.vent_zero.gt(.85)
    daycells=[]
    for (gate,p_high),g in daily.groupby(['gate','p_high']):
        daycells.append(dict(gate0=bool(gate),p_high0=bool(p_high),days=len(g),
                             mean_residual=float(g.residual.mean()),
                             sealed_after_day=int(g.sealed_after_day.sum())))
    active=z[z.gate & z.p_high]
    active_days=int(active.groupby(['farm','day']).ngroups)
    farm_residuals={farm:float((active[active.farm.eq(farm)].sub_ec-
                                active[active.farm.eq(farm)].v2).mean())
                    for farm in ('F13','F47')}
    fold_residuals={str(fold):float((active[active.fold.eq(fold)].sub_ec-
                                   active[active.fold.eq(fold)].v2).mean())
                    for fold in (0,2,4,6,8,9)}
    gate_pass=(active_days>=20 and all(v>=.05 for v in farm_residuals.values())
               and sum(v>0 for v in fold_residuals.values())>=4)
    result=dict(rows=len(z),days=len(daily),cells=cells,stratified=stratified,
                midnight_cells=daycells,active_days=active_days,
                active_farm_residuals=farm_residuals,
                active_fold_residuals=fold_residuals,gate_pass=bool(gate_pass),
                test_input_read=False,new_lock_labels_read=False,
                submission_created=False)
    out=LOCAL/'ec_causal_seal_diagnostic'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='stratified'},
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
