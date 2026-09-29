"""Preprocessing checks with fixed features and fixed histogram-boosting settings."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.analysis-tools/python'))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from profile_tabular import load,OUT,TARGETS
from causal_features import build_all


def short_ffill(x):
    parts=[]
    for _,g in x.groupby('farm'):
        g=g.sort_values('time')
        a=g.set_index('time').reindex(range(g.time.min(),g.time.max()+1))
        cols=['in_temp','in_hum','in_co2']
        a[cols]=a[cols].ffill(limit=3)
        parts.append(a.loc[g.time].reset_index())
    return pd.concat(parts,ignore_index=True)


def model():
    return HistGradientBoostingRegressor(max_iter=160,learning_rate=.05,max_leaf_nodes=15,
        min_samples_leaf=50,l2_regularization=10,early_stopping=False,random_state=20260919)


def main():
    x=load('train_X.csv'); y=load('train_y.csv').set_index('row_id')
    x=x[x.farm.isin(['F13','F47'])].copy()
    features,groups=build_all(x)
    filled,_=build_all(short_ffill(x))
    # Retain original observation availability after filling.
    availability=groups['quality']+[c for c in groups['smooth'] if '_coverage' in c or c.endswith('_age')]
    filled[availability]=features[availability]
    x=x.set_index('row_id').loc[features.index]
    columns=sum([groups[k] for k in ['raw','clock','lags','smooth','physics','day_context','quality']],[])
    aligned=x.day-x.farm.map({'F13':2,'F47':0})
    records=[]
    for start,end in [(150,160),(170,180),(209,217)]:
        # Match public-label usage; every validation target is held out.
        va=aligned.ge(start)&aligned.lt(end); tr=~va
        for f in ['F13','F47']:
            for t in TARGETS:
                yy=y[t].reindex(features.index)
                train=tr&x.farm.eq(f)&yy.notna(); valid=va&x.farm.eq(f)&yy.notna()
                for variant in ['base','ffill_3hours','drop_current_missing','downweight_far_days','residual_indoor_temp']:
                    if variant=='residual_indoor_temp' and t!='sub_temp':continue
                    xx=filled if variant=='ffill_3hours' else features
                    actual_train=train.copy()
                    if variant=='drop_current_missing':actual_train&=x[['in_temp','in_hum','in_co2']].notna().all(axis=1)
                    weights=None
                    if variant=='downweight_far_days':
                        # Distance to the fixed fold, using metadata only. The bandwidth is preselected, not tuned.
                        distance=(aligned[actual_train]-(start+end-1)/2).abs()
                        weights=np.exp(-distance/60).to_numpy()
                    yytrain=yy[actual_train]
                    offset=pd.Series(0.,index=features.index)
                    if variant=='residual_indoor_temp':
                        fallback=float(x.loc[actual_train,'in_temp'].mean())
                        offset=x.in_temp.fillna(fallback)
                        yytrain=yytrain-offset[actual_train]
                    fit=model().fit(xx.loc[actual_train,columns],yytrain,sample_weight=weights)
                    pred=fit.predict(xx.loc[valid,columns])+offset[valid].to_numpy()
                    err=pred-yy[valid].to_numpy()
                    records.append(dict(start=start,end=end,farm=f,target=t,variant=variant,n_train=int(actual_train.sum()),n=int(valid.sum()),rmse=float(np.sqrt(np.mean(err**2)))))
        print(f'Finished preprocessing {start}:{end}',flush=True)
        pd.DataFrame(records).to_csv(OUT/'preprocessing_validation.csv',index=False)
    df=pd.DataFrame(records); df['sse']=df.rmse**2*df.n
    summary=df.groupby(['target','variant']).agg(sse=('sse','sum'),n=('n','sum')); summary['rmse']=np.sqrt(summary.sse/summary.n)
    summary.to_csv(OUT/'preprocessing_validation_summary.csv')
    print(summary.to_string())


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
