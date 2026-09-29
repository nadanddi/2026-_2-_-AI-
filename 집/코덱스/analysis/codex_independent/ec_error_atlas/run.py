"""Public EC v2 OOF error decomposition by predefined regimes."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'온라인대회자료/정형데이터/참가자_배포'
R=ROOT/'analysis/local'
SEARCH=R/'ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM=R/'ec_locked_confirmation/20260928_044934'
INPUT_COLS=['in_temp','in_hum','in_co2','act_vent','act_shade',
            'act_thermal','act_heating','act_circfan','act_co2','act_fog']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions():
    paths=[*[SEARCH/f'fold{i}.csv' for i in (0,2,4,6)],
           CONFIRM/'fold8.csv',CONFIRM/'fold9.csv']
    frames=[]
    for path in paths:
        z=pd.read_csv(path)
        z['v2']=z['blend'] if 'blend' in z else z['candidate']
        frames.append(z[['row_id','farm','day','sub_ec','baseline','v2']])
    a=pd.concat(frames,ignore_index=True)
    assert len(a)==5616 and a.row_id.is_unique
    assert np.isfinite(a[['sub_ec','baseline','v2']].to_numpy(float)).all()
    return a,paths


def day_inputs():
    x=pd.read_csv(DATA/'train_X.csv',usecols=['row_id',*INPUT_COLS])
    x=x[x.row_id.str[:3].isin(('F13','F47'))].copy()
    x['farm']=x.row_id.str[:3]
    x['day']=x.row_id.str[4:7].astype(int)
    x['hour']=x.row_id.str[8:10].astype(int)
    x.sort_values(['farm','day','hour'],inplace=True)
    agg=x.groupby(['farm','day'],as_index=False).agg(
        t0=('in_temp','first'),fan=('act_circfan','mean'),
        vent_zero=('act_vent',lambda s:float((s==0).mean())),
        rows=('row_id','size'))
    assert len(agg)==400 and agg.rows.eq(24).all()
    agg['sealed']=agg.fan.lt(10)&agg.vent_zero.gt(.85)
    agg['cold0']=agg.t0.le(10)
    agg['section']=np.where(agg.day<179,'first','second')
    h0=x[x.hour.eq(0)][['farm','day',*INPUT_COLS]].copy()
    assert len(h0)==400
    groups=[]
    for farm,g in h0.groupby('farm',sort=True):
        m=SimpleImputer(strategy='median').fit_transform(g[INPUT_COLS])
        m=StandardScaler().fit_transform(m)
        labels=KMeans(n_clusters=4,n_init=20,random_state=0).fit_predict(m)
        item=g[['farm','day']].copy();item['cluster']=labels.astype(int)
        groups.append(item)
    agg=agg.merge(pd.concat(groups),on=['farm','day'],validate='one_to_one')
    return agg


def metric(g,total_sq):
    n=len(g)
    assert n>0
    sqv=float(g.sq_v2.sum());sqb=float(g.sq_base.sum())
    level=float(g.sq_level.sum())
    return dict(days=n,rows=24*n,rmse_v2=float(np.sqrt(sqv/(24*n))),
                rmse_baseline=float(np.sqrt(sqb/(24*n))),
                relative_change=float(np.sqrt(sqv/sqb)-1),
                mean_residual=float(g.residual.mean()),
                error_share_of_all=float(sqv/total_sq),
                level_fraction=float(level/sqv),
                high_ec_days=int(g.high_ec.sum()))


def table(day,cols,total_sq):
    output=[]
    for key,g in day.groupby(cols,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        output.append(dict(zip(cols,[v.item() if hasattr(v,'item') else v for v in key]))|
                      metric(g,total_sq))
    return sorted(output,key=lambda z:z['error_share_of_all'],reverse=True)


def main():
    a,paths=predictions()
    a['e_v2']=a.sub_ec-a.v2
    a['e_base']=a.sub_ec-a.baseline
    daily=a.groupby(['farm','day'],as_index=False).agg(
        y=('sub_ec','mean'),p=('v2','mean'),residual=('e_v2','mean'),
        sq_v2=('e_v2',lambda s:float(np.sum(s**2))),
        sq_base=('e_base',lambda s:float(np.sum(s**2))),
        rows=('row_id','size'))
    assert len(daily)==234 and daily.rows.eq(24).all()
    daily['sq_level']=24*daily.residual**2
    daily['high_ec']=daily.y.ge(1.2)
    daily=daily.merge(day_inputs(),on=['farm','day'],validate='one_to_one')
    total_sq=float(daily.sq_v2.sum())
    specs={'farm':['farm'],'section':['section'],'sealed':['sealed'],
           'cold0':['cold0'],'high_ec':['high_ec'],
           'farm_section':['farm','section'],
           'farm_section_sealed':['farm','section','sealed'],
           'farm_section_cold':['farm','section','cold0'],
           'farm_section_sealed_cold':['farm','section','sealed','cold0'],
           'farm_section_cluster':['farm','section','cluster']}
    result=dict(days=len(daily),rows=len(a),overall=metric(daily,total_sq),
                tables={name:table(daily,cols,total_sq) for name,cols in specs.items()},
                top_days=daily.nlargest(20,'sq_v2')[['farm','day','y','p','residual',
                     'sealed','cold0','section','cluster','sq_v2']].to_dict('records'),
                sha256={str(p):sha(p) for p in [DATA/'train_X.csv',*paths]},
                hidden_labels_read=False,test_input_read=False,
                no_submission_file_created=True)
    assert abs(result['overall']['error_share_of_all']-1)<1e-10
    out=R/'ec_error_atlas'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,rows in result['tables'].items():
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('tables','top_days','sha256')},
                     ensure_ascii=False,indent=2))
    print('TOP STABLE CELLS')
    for name in ('farm_section_sealed_cold','farm_section_cluster'):
        print(name,json.dumps([z for z in result['tables'][name] if z['days']>=5][:8],ensure_ascii=False))


if __name__=='__main__':
    main()
