"""Honest fold-specific midnight setpoint screen against saved EC v2 OOF."""
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
LOCK = Path(__file__).resolve().parents[1]/'ec_final_lock/locked_days.json'
COLS = ['day','farm47','in_temp','in_hum','in_co2','act_vent',
        'act_shade','act_thermal','act_heating','act_circfan',
        'act_co2','act_fog','out_temp','out_hum']


def score(g, col):
    return float(np.sqrt(np.mean(np.square(g.sub_ec-g[col]))))


def fit_ridge(tr, va):
    x=tr[COLS].to_numpy(float)
    z=va[COLS].to_numpy(float)
    with np.errstate(all='ignore'):
        med=np.nanmedian(x,axis=0)
    med=np.where(np.isfinite(med),med,0)
    assert np.isfinite(med).all()
    x=np.where(np.isfinite(x),x,med)
    z=np.where(np.isfinite(z),z,med)
    mean=x.mean(axis=0)
    std=x.std(axis=0)
    std=np.where(std>1e-12,std,1)
    x=(x-mean)/std
    z=(z-mean)/std
    a=np.column_stack([np.ones(len(x)),x])
    b=np.column_stack([np.ones(len(z)),z])
    penalty=np.diag(np.r_[0.0,np.full(len(COLS),20.0)])
    beta=np.linalg.solve(a.T@a+penalty,a.T@tr.y.to_numpy(float))
    return b@beta


def main():
    raw=pd.read_csv(DATA/'train_X.csv')
    raw=raw[raw.row_id.str[:3].isin(('F13','F47'))].copy()
    raw['farm']=raw.row_id.str[:3]
    raw['day']=raw.row_id.str[4:7].astype(int)
    raw['hour']=raw.row_id.str[8:10].astype(int)
    raw['farm47']=raw.farm.eq('F47').astype(float)
    assert len(raw)==9600 and raw.row_id.is_unique
    y=pd.read_csv(DATA/'train_y.csv',usecols=['row_id','sub_ec'])
    lab=raw[['row_id','farm','day']].merge(y,on='row_id',validate='one_to_one')
    assert lab.sub_ec.notna().all()
    daily_y=lab.groupby(['farm','day'],as_index=False).agg(
        y=('sub_ec','mean'),n=('row_id','size'))
    assert len(daily_y)==400 and daily_y.n.eq(24).all()
    midnight=raw[raw.hour.eq(0)][['farm',*COLS]].copy()
    assert len(midnight)==400
    days=midnight.merge(daily_y[['farm','day','y']],on=['farm','day'],validate='one_to_one')
    locked=json.loads(LOCK.read_text(encoding='utf-8'))['selected']
    lock={(r['farm'],r['day']) for r in locked}
    assert len(lock)==40
    folds=[]
    for fold in (0,2,4,6,8,9):
        z=pd.read_csv((SEARCH if fold<8 else CONFIRM)/f'fold{fold}.csv')
        z['v2']=z['blend'] if fold<8 else z['candidate']
        z['fold']=fold
        folds.append(z[['row_id','farm','day','fold','sub_ec','v2']])
    oof=pd.concat(folds,ignore_index=True)
    assert len(oof)==5616 and oof.row_id.is_unique
    scored=[];training=[]
    for fold in (0,2,4,6,8,9):
        v=oof[oof.fold.eq(fold)].copy()
        val={(f,int(d)) for f,d in v[['farm','day']].itertuples(index=False,name=None)}
        assert not (val & lock)
        excluded=val|lock
        near={(f,d+j) for f,d in excluded for j in (-1,0,1)}
        tr=days[[((f,int(d)) not in near) for f,d in
                 days[['farm','day']].itertuples(index=False,name=None)]].copy()
        va=days[[((f,int(d)) in val) for f,d in
                 days[['farm','day']].itertuples(index=False,name=None)]].copy()
        assert len(va)==len(val) and len(tr)>=150
        va['m0']=fit_ridge(tr,va)
        v=v.merge(va[['farm','day','m0']],on=['farm','day'],validate='many_to_one')
        v=v.merge(raw[['row_id','act_circfan','act_vent']],on='row_id',validate='one_to_one')
        gate=v.act_circfan.lt(10)&v.act_vent.eq(0)&v.v2.ge(.6)
        v['h2']=np.where(gate,.8*v.v2+.2*v.m0,v.v2)
        assert np.isfinite(v[['v2','h2']].to_numpy(float)).all()
        scored.append(v)
        training.append(dict(fold=fold,training_days=len(tr),validation_days=len(va),
                             gate_rows=int(gate.sum()),v2_rmse=score(v,'v2'),
                             h2_rmse=score(v,'h2')))
    a=pd.concat(scored,ignore_index=True)
    metrics={name:dict(v2_rmse=score(g,'v2'),h2_rmse=score(g,'h2'),
                       relative_change=score(g,'h2')/score(g,'v2')-1)
             for name,g in [('overall',a),*[ (f'fold{i}',a[a.fold.eq(i)])
                                          for i in (0,2,4,6,8,9)],
                            *[(f,a[a.farm.eq(f)]) for f in ('F13','F47')]]}
    a['delta_sq']=(a.sub_ec-a.h2)**2-(a.sub_ec-a.v2)**2
    daily=a.groupby(['farm','day']).delta_sq.mean().to_numpy()
    assert len(daily)==234
    rng=np.random.default_rng(2902)
    sample=np.empty(20000)
    for i in range(len(sample)):
        sample[i]=daily[rng.integers(len(daily),size=len(daily))].mean()
    ci=np.quantile(sample,[.025,.975]).tolist()
    passed=all(metrics[f'fold{i}']['relative_change']<0 for i in (0,2,4,6,8,9))
    passed=passed and all(metrics[f]['relative_change']<0 for f in ('F13','F47')) and ci[1]<0
    result=dict(training=training,metrics=metrics,
                bootstrap_mse_delta_95ci=ci,
                bootstrap_p_worse=float(np.mean(sample>=0)),
                screen_pass=bool(passed),locked_days_scored=False,
                test_input_read=False,submission_created=False)
    out=LOCAL/'ec_midnight_setpoint'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
