"""Diagnostic ridge ablations on training-only chronological blocks.

These are exploratory validation scores, not competition scores or a final model.
All target imputation is forbidden. X imputation/scaling uses fold training only.
"""
import json
import numpy as np
import pandas as pd
from profile_tabular import load, OUT, TARGETS
from causal_features import build_all


def ridge(train,test,y,alpha=10.):
    train=np.asarray(train,dtype=float); test=np.asarray(test,dtype=float)
    keep=np.isfinite(train).any(axis=0)
    train,test=train[:,keep],test[:,keep]
    means=np.nanmean(train,axis=0)
    train=np.where(np.isfinite(train),train,means)
    test=np.where(np.isfinite(test),test,means)
    scales=np.maximum(train.std(axis=0),1e-6)
    train=(train-means)/scales; test=(test-means)/scales
    center=y.mean()
    coef=np.linalg.solve(train.T@train+np.eye(train.shape[1])*alpha,train.T@(y-center))
    return test@coef+center


def main():
    x=load('train_X.csv'); y=load('train_y.csv').set_index('row_id')
    x=x[x.farm.isin(['F13','F47'])].copy()
    features,groups=build_all(x)
    x=x.set_index('row_id').loc[features.index]
    sets={
        'raw':groups['raw'],
        'raw_clock':groups['raw']+groups['clock'],
        'raw_clock_lags':sum([groups[k] for k in ['raw','clock','lags']],[]),
        'raw_clock_smooth':sum([groups[k] for k in ['raw','clock','smooth']],[]),
        'history_physics':sum([groups[k] for k in ['raw','clock','lags','smooth','physics']],[]),
        'history_physics_trend':sum([groups[k] for k in ['raw','clock','lags','smooth','physics','trend']],[]),
    }
    result=[]
    # Offset F13 by -2 to pair analogous supplied train/test gap schedules.
    # This is NOT a claim of common calendar dates or aligned weather.
    # Fold grouping only; no cross-farm values are features.
    aligned=x.day-x.farm.map({'F13':2,'F47':0})
    for mode in ['past_only','blocked_train_both_sides']:
        for start,end in [(120,130),(150,160),(170,180),(189,197),(209,217),(239,244)]:
            valid=aligned.ge(start)&aligned.lt(end)
            train=aligned.lt(start) if mode=='past_only' else ~valid
            for f in ['F13','F47']:
                tr=train&x.farm.eq(f); va=valid&x.farm.eq(f)
                if not va.any():continue
                for target in TARGETS:
                    yy=y[target].reindex(features.index)
                    train_mask=tr&yy.notna(); val_mask=va&yy.notna()
                    for name,columns in sets.items():
                        pred=ridge(features.loc[train_mask,columns],features.loc[val_mask,columns],yy[train_mask].to_numpy())
                        error=pred-yy[val_mask].to_numpy()
                        result.append(dict(mode=mode,start=start,end=end,farm=f,target=target,features=name,
                            n_train=int(train_mask.sum()),n_valid=int(val_mask.sum()),
                            rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(np.abs(error)))))
                    base=float(yy[train_mask].mean())
                    err=base-yy[val_mask].to_numpy()
                    result.append(dict(mode=mode,start=start,end=end,farm=f,target=target,features='train_mean',
                        n_train=int(train_mask.sum()),n_valid=int(val_mask.sum()),rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(np.abs(err)))))
            print(f'Finished {mode} block {start}:{end}',flush=True)
    out=pd.DataFrame(result)
    out.to_csv(OUT/'feature_validation.csv',index=False)
    out['sse']=out.rmse**2*out.n_valid
    aggregate=out.groupby(['mode','target','features']).agg(sse=('sse','sum'),n=('n_valid','sum'))
    aggregate['rmse']=np.sqrt(aggregate.sse/aggregate.n)
    aggregate.drop(columns='sse').to_csv(OUT/'feature_validation_summary.csv')
    print(aggregate.drop(columns='sse').to_string())
    (OUT/'feature_groups.json').write_text(json.dumps(groups,indent=2),encoding='utf8')


if __name__=='__main__':main()
