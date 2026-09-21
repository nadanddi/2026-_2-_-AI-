"""EC-specific controls, humidity, loss and regime-model ablations.

No leaderboard or test labels. Same fold rules as previous descriptive experiments.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.analysis-tools/python'))
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from profile_tabular import load,OUT
from ec_features import add_ec_features


def regressor():
    return HistGradientBoostingRegressor(max_iter=160,learning_rate=.05,max_leaf_nodes=15,
        min_samples_leaf=50,l2_regularization=10,early_stopping=False,random_state=20260919)


def fit_predict(a,b,y,kind):
    prob=None
    if kind=='mixture':
        high=y>1
        clf=HistGradientBoostingClassifier(max_iter=120,learning_rate=.05,max_leaf_nodes=7,
            min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=20260919)
        if high.any() and (~high).any():
            clf.fit(a,high); prob=clf.predict_proba(b)[:,1]
            low=regressor().fit(a.loc[~high],y[~high]).predict(b)
            upper=regressor().fit(a.loc[high],y[high]).predict(b)
            prediction=(1-prob)*low+prob*upper
        else:
            prediction=regressor().fit(a,y).predict(b)
    elif kind=='log':
        prediction=np.expm1(regressor().fit(a,np.log1p(y)).predict(b))
    else:
        prediction=regressor().fit(a,y).predict(b)
    return prediction,prob


def main():
    x=load('train_X.csv'); yy=load('train_y.csv').set_index('row_id').sub_ec
    x=x[x.farm.isin(['F13','F47'])].copy()
    features,groups=add_ec_features(x)
    x=x.set_index('row_id').loc[features.index]; yy=yy.reindex(features.index)
    aligned=x.day-x.farm.map({'F13':2,'F47':0})
    base=sum([groups[k] for k in ['raw','clock','lags','smooth','physics','day_context','quality']],[])
    controls=base+groups['ec_controls']; enriched=controls+groups['humidity']
    variants={
        'base':(base,'direct'),
        'controls':(controls,'direct'),
        'controls_humidity':(enriched,'direct'),
        'controls_humidity_no_day':([c for c in enriched if c!='day'],'direct'),
        'log_ec':(enriched,'log'),
        'mixture_ec':(enriched,'mixture'),
    }
    results=[]; rows=[]
    for mode in ['past_only','blocked_train_both_sides']:
        for start,end in [(150,160),(170,180),(209,217)]:
            val=aligned.ge(start)&aligned.lt(end)
            train=aligned.lt(start) if mode=='past_only' else ~val
            for f in ['F13','F47']:
                tr=train&x.farm.eq(f)&yy.notna(); va=val&x.farm.eq(f)&yy.notna()
                for name,(columns,kind) in variants.items():
                    pred,prob=fit_predict(features.loc[tr,columns],features.loc[va,columns],yy[tr],kind)
                    actual=yy[va].to_numpy(); high=actual>1; err=pred-actual
                    results.append(dict(mode=mode,start=start,end=end,farm=f,variant=name,n=len(actual),n_high=int(high.sum()),
                        rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err))),
                        high_sse=float((err[high]**2).sum()),low_sse=float((err[~high]**2).sum()),
                        high_auc=float(roc_auc_score(high,prob)) if prob is not None and 0<high.sum()<len(high) else None))
                    for j,idx in enumerate(features.index[va]):
                        rows.append(dict(row_id=idx,mode=mode,start=start,farm=f,variant=name,actual=float(actual[j]),prediction=float(pred[j])))
            print(f'Finished EC {mode} {start}:{end}',flush=True)
            pd.DataFrame(results).to_csv(OUT/'ec_validation.csv',index=False)
    df=pd.DataFrame(results); df['sse']=df.rmse**2*df.n
    summary=df.groupby(['mode','variant']).agg(sse=('sse','sum'),n=('n','sum'),high_sse=('high_sse','sum'),low_sse=('low_sse','sum'),n_high=('n_high','sum'))
    summary['rmse']=np.sqrt(summary.sse/summary.n)
    summary['high_rmse']=np.sqrt(summary.high_sse/summary.n_high)
    summary['low_rmse']=np.sqrt(summary.low_sse/(summary.n-summary.n_high))
    summary.to_csv(OUT/'ec_validation_summary.csv')
    pd.DataFrame(rows).to_csv(OUT/'ec_local_predictions.csv',index=False)
    print(summary[['n','rmse','high_rmse','low_rmse']].to_string())


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
