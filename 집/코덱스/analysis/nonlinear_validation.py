"""Fixed-parameter histogram boosting ablations; no test predictions or submissions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.analysis-tools/python'))
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from profile_tabular import load, OUT, TARGETS
from causal_features import build_all


def main():
    purge='--purge-weather' in sys.argv
    prefix='weather_purged' if purge else 'nonlinear'
    x=load('train_X.csv'); y=load('train_y.csv').set_index('row_id')
    x=x[x.farm.isin(['F13','F47'])].copy()
    feat,groups=build_all(x)
    x=x.set_index('row_id').loc[feat.index]
    aligned=x.day-x.farm.map({'F13':2,'F47':0})
    names={
       'raw_clock':['raw','clock'],
       'history':['raw','clock','lags','smooth','physics'],
       'day_context':['raw','clock','day_context','quality'],
       'history_day_context':['raw','clock','lags','smooth','physics','day_context','quality'],
    }
    signatures={}
    if purge:
        names={k:v for k,v in names.items() if k in ['raw_clock','history_day_context']}
        for f,g in x.groupby('farm'):
            daily=g.pivot(index='day',columns='hour',values=['out_temp','out_hum','out_rad','out_wspd'])
            assert not daily.isna().any().any()
            signatures[f]=pd.util.hash_pandas_object(daily,index=False)
    result=[]
    params=dict(max_iter=160,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=50,
                l2_regularization=10.,early_stopping=False,random_state=20260919)
    for mode in ['past_only','blocked_train_both_sides']:
        for start,end in [(150,160),(170,180),(189,197),(209,217)]:
            valid=aligned.ge(start)&aligned.lt(end)
            train=aligned.lt(start) if mode=='past_only' else ~valid
            for f in ['F13','F47']:
                local_train=train&x.farm.eq(f)
                removed=0
                if purge:
                    va_days=x.loc[valid&x.farm.eq(f),'day'].unique()
                    sig=signatures[f]
                    matched_days=sig.index[sig.isin(sig.reindex(va_days))]
                    exclusion=x.day.isin(matched_days)&x.farm.eq(f)
                    removed=int((local_train&exclusion).sum())
                    local_train&=~exclusion
                for target in TARGETS:
                    yy=y[target].reindex(feat.index)
                    tr=local_train&yy.notna(); va=valid&x.farm.eq(f)&yy.notna()
                    for name,keys in names.items():
                        columns=sum([groups[k] for k in keys],[])
                        model=HistGradientBoostingRegressor(**params)
                        model.fit(feat.loc[tr,columns],yy[tr])
                        prediction=model.predict(feat.loc[va,columns])
                        error=prediction-yy[va].to_numpy()
                        result.append(dict(mode=mode,start=start,end=end,farm=f,target=target,features=name,
                            n_train=int(tr.sum()),n_valid=int(va.sum()),purged_train_rows=removed,rmse=float(np.sqrt(np.mean(error**2)))))
            print(f'Finished {mode} {start}:{end}',flush=True)
            pd.DataFrame(result).to_csv(OUT/f'{prefix}_validation.csv',index=False)
    df=pd.DataFrame(result); df['sse']=df.rmse**2*df.n_valid
    summary=df.groupby(['mode','target','features']).agg(sse=('sse','sum'),n=('n_valid','sum'))
    summary['rmse']=np.sqrt(summary.sse/summary.n)
    summary.drop(columns='sse').to_csv(OUT/f'{prefix}_validation_summary.csv')
    (OUT/f'{prefix}_parameters.json').write_text(json.dumps(params,indent=2),encoding='utf8')
    print(summary.drop(columns='sse').to_string())


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
