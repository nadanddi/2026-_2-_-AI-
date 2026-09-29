"""Compare public EC OOF regimes with input-only evaluation regimes."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'온라인대회자료/정형데이터/참가자_배포'
R=ROOT/'analysis/local'
SEARCH=R/'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM=R/'ec_locked_confirmation/20260928_044934'


def load_oof():
    paths=[*[SEARCH/f'fold{i}.csv' for i in (0,2,4,6)],
           CONFIRM/'fold8.csv',CONFIRM/'fold9.csv']
    a=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    a['prediction']=a['blend'] if 'blend' in a else a['candidate']
    assert len(a)==5616 and a.row_id.is_unique
    a['error']=a.sub_ec-a.prediction
    a['day_error']=a.groupby(['farm','day']).error.transform('mean')
    return a


def regimes(filename):
    x=pd.read_csv(DATA/filename,usecols=['row_id','in_temp','act_vent','act_circfan'])
    x=x[x.row_id.str[:3].isin(('F13','F47'))].copy()
    x['farm']=x.row_id.str[:3]
    x['day']=x.row_id.str[4:7].astype(int)
    x['hour']=x.row_id.str[8:10].astype(int)
    d=x.groupby(['farm','day'],as_index=False).agg(
        t0=('in_temp','first'),fan=('act_circfan','mean'),
        vent_zero=('act_vent',lambda z:float((z==0).mean())),
        count=('row_id','size'))
    assert d['count'].eq(24).all()
    d['sealed']=d.fan.lt(10)&d.vent_zero.gt(.85)
    d['cold0']=d.t0.le(10)
    d['section']=np.where(d.day<179,'first','second')
    return d


def stats(d):
    return dict(days=len(d),sealed=int(d.sealed.sum()),sealed_rate=float(d.sealed.mean()),
                cold0=int(d.cold0.sum()),cold0_rate=float(d.cold0.mean()),
                sealed_and_cold0=int((d.sealed&d.cold0).sum()))


def main():
    oof=load_oof()
    tr=regimes('train_X.csv')
    te=regimes('test_X.csv')
    days=oof[['farm','day']].drop_duplicates()
    tr['is_oof']=tr.set_index(['farm','day']).index.isin(days.set_index(['farm','day']).index)
    summary={}
    for name,d in [('all_training',tr),('oof',tr[tr.is_oof]),
                   ('non_oof_training',tr[~tr.is_oof]),('evaluation',te)]:
        summary[name]={'all':stats(d)}
        for (farm,section),g in d.groupby(['farm','section']):
            summary[name][f'{farm}_{section}']=stats(g)
    error={}
    oof=oof.merge(tr[['farm','day','sealed','cold0','section']],on=['farm','day'],validate='many_to_one')
    for (farm,section,sealed),g in oof.groupby(['farm','section','sealed']):
        mse=float(np.mean(g.error**2));level=float(np.mean(g.day_error**2))
        shape=float(np.mean((g.error-g.day_error)**2))
        error[f'{farm}_{section}_sealed{int(sealed)}']=dict(
            days=int(g[['farm','day']].drop_duplicates().shape[0]),
            rmse=float(np.sqrt(mse)),level_rmse=float(np.sqrt(level)),
            within_day_rmse=float(np.sqrt(shape)),level_fraction=level/mse)
    late=oof[oof.section.eq('second')]
    late_mse=float(np.mean(late.error**2))
    late_level=float(np.mean(late.day_error**2))
    late_shape=float(np.mean((late.error-late.day_error)**2))
    result=dict(regimes=summary,oof_error=error,
                late_oof_rmse=float(np.sqrt(late_mse)),
                late_oof_level_rmse=float(np.sqrt(late_level)),
                late_oof_shape_rmse=float(np.sqrt(late_shape)),
                hidden_labels_read=False,no_submission_file_created=True)
    late_reg=tr[tr.is_oof & tr.section.eq('second')]
    result['evaluation_sealed_rate_over_late_oof']=float(te.sealed.mean()/late_reg.sealed.mean()) if late_reg.sealed.mean()>0 else None
    result['late_oof_unrepresentative_by_protocol']=bool(te.sealed.mean()>=2*late_reg.sealed.mean())
    out=R/'ec_validation_gap'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
