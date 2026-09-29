"""H3 high-day probability mixture, fixed public OOF screen only."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path(__file__).resolve().parents[3]
LOCAL=ROOT/'analysis/local'
DATA=ROOT/'온라인대회자료/정형데이터/참가자_배포'
SEARCH=LOCAL/'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM=LOCAL/'ec_locked_confirmation/20260928_044934'
LOCK=Path(__file__).resolve().parents[1]/'ec_final_lock/locked_days.json'
CURRENT=['in_temp','in_hum','in_co2','out_rad','act_vent',
         'act_circfan','act_shade','act_heating','act_co2','act_fog']
PREFIX=['in_temp','in_hum','act_vent','act_circfan','act_heating']
FEATURES=['day','hour','farm47',*CURRENT,*[x+'_prefix' for x in PREFIX]]


def rmse(y,p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def load_features():
    x=pd.read_csv(DATA/'train_X.csv',usecols=['row_id',*CURRENT])
    x=x[x.row_id.str[:3].isin(('F13','F47'))].copy()
    x['farm']=x.row_id.str[:3]
    x['day']=x.row_id.str[4:7].astype(int)
    x['hour']=x.row_id.str[8:10].astype(int)
    x['farm47']=x.farm.eq('F47').astype(int)
    x.sort_values(['farm','day','hour'],inplace=True)
    assert len(x)==9600 and x.row_id.is_unique
    for c in PREFIX:
        x[c+'_prefix']=x.groupby(['farm','day'])[c].transform(
            lambda s:s.expanding().mean())
    assert x.groupby(['farm','day']).size().eq(24).all()
    return x


def load_oof():
    frames=[]
    for fold in (0,2,4,6,8,9):
        z=pd.read_csv((SEARCH if fold<8 else CONFIRM)/f'fold{fold}.csv')
        z['v2']=z['blend'] if fold<8 else z['candidate']
        z['fold']=fold
        frames.append(z[['row_id','farm','day','fold','sub_ec','v2']])
    oof=pd.concat(frames,ignore_index=True)
    assert len(oof)==5616 and oof.row_id.is_unique
    return oof


def main():
    x=load_features()
    oof=load_oof()
    locked=json.loads(LOCK.read_text(encoding='utf-8'))['selected']
    lock={(r['farm'],int(r['day'])) for r in locked}
    assert len(lock)==40
    # Labels for the final lock are excluded before forming daily targets.
    y=pd.read_csv(DATA/'train_y.csv',usecols=['row_id','sub_ec'])
    y['farm']=y.row_id.str[:3]
    y['day']=y.row_id.str[4:7].astype(int)
    y=y[y.farm.isin(('F13','F47'))].copy()
    assert len(y)==9600 and y.sub_ec.notna().all()
    y=y[[((f,int(d)) not in lock) for f,d in
         y[['farm','day']].itertuples(index=False,name=None)]].copy()
    daily=y.groupby(['farm','day'],as_index=False).agg(
        mean_ec=('sub_ec','mean'),n=('row_id','size'))
    assert len(daily)==360 and daily.n.eq(24).all()
    daily['high']=daily.mean_ec.ge(1.2)
    rows=x.merge(daily[['farm','day','mean_ec','high']],on=['farm','day'],how='inner',
                 validate='many_to_one')
    assert len(rows)==8640
    outputs=[];fold_info=[]
    for fold in (0,2,4,6,8,9):
        va=oof[oof.fold.eq(fold)].copy()
        val={(f,int(d)) for f,d in va[['farm','day']].itertuples(index=False,name=None)}
        assert not (val&lock)
        near={(f,d+j) for f,d in val|lock for j in (-1,0,1)}
        tr=rows[[((f,int(d)) not in near) for f,d in
                 rows[['farm','day']].itertuples(index=False,name=None)]].copy()
        assert tr.groupby(['farm','day']).ngroups>=150
        dt=tr[['farm','day','mean_ec','high']].drop_duplicates(['farm','day'])
        q0=float(dt.high.mean())
        mu1=float(dt.loc[dt.high,'mean_ec'].mean())
        mu0=float(dt.loc[~dt.high,'mean_ec'].mean())
        assert 0<q0<1 and mu1>mu0
        model=HistGradientBoostingClassifier(
            max_iter=150,learning_rate=.05,max_leaf_nodes=7,
            min_samples_leaf=100,l2_regularization=10,
            early_stopping=False,random_state=2904)
        model.fit(tr[FEATURES],tr.high.astype(int),
                  sample_weight=np.full(len(tr),1/24))
        va=va.merge(x[['row_id',*FEATURES,'act_circfan','act_vent']]
                    .loc[:,lambda d:~d.columns.duplicated()],
                    on='row_id',validate='one_to_one')
        q=model.predict_proba(va[FEATURES])[:,1]
        gate=va.act_circfan.lt(10)&va.act_vent.eq(0)&va.v2.ge(.6)
        va['h3']=va.v2+np.where(gate,.20*(q-q0)*(mu1-mu0),0)
        assert np.isfinite(va[['v2','h3']].to_numpy(float)).all()
        outputs.append(va[['row_id','farm','day','fold','sub_ec','v2','h3']])
        fold_info.append(dict(fold=fold,train_days=len(dt),high_train_days=int(dt.high.sum()),
                              q0=q0,mu0=mu0,mu1=mu1,gate_rows=int(gate.sum()),
                              q_gate_mean=float(q[gate].mean()),
                              baseline=rmse(va.sub_ec,va.v2),candidate=rmse(va.sub_ec,va.h3)))
    a=pd.concat(outputs,ignore_index=True)
    metrics={name:dict(v2_rmse=rmse(g.sub_ec,g.v2),h3_rmse=rmse(g.sub_ec,g.h3),
                       relative_change=rmse(g.sub_ec,g.h3)/rmse(g.sub_ec,g.v2)-1)
             for name,g in [('all',a),*[ (f'fold{i}',a[a.fold.eq(i)]) for i in (0,2,4,6,8,9)],
                            *[(f,a[a.farm.eq(f)]) for f in ('F13','F47')]]}
    a['delta_sq']=(a.sub_ec-a.h3)**2-(a.sub_ec-a.v2)**2
    day_delta=a.groupby(['farm','day']).delta_sq.mean().to_numpy()
    assert len(day_delta)==234
    rng=np.random.default_rng(2905)
    sample=np.empty(20000)
    for i in range(len(sample)):
        sample[i]=day_delta[rng.integers(len(day_delta),size=len(day_delta))].mean()
    ci=np.quantile(sample,[.025,.975]).tolist()
    passed=all(metrics[f'fold{i}']['relative_change']<0 for i in (0,2,4,6,8,9))
    passed=passed and all(metrics[f]['relative_change']<0 for f in ('F13','F47')) and ci[1]<0
    result=dict(fold_info=fold_info,metrics=metrics,day_mse_delta_95ci=ci,
                bootstrap_p_worse=float(np.mean(sample>=0)),screen_pass=bool(passed),
                test_input_read=False,lock_labels_in_training=False,
                submission_created=False)
    out=LOCAL/'ec_high_day_mixture'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
