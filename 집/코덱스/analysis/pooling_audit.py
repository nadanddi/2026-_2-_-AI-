"""Controlled linear transfer check with common inputs and farm intercepts."""
import numpy as np
import pandas as pd
from profile_tabular import load, OUT
from validate_features import ridge


def main():
    x=load('train_X.csv'); y=load('train_y.csv')
    d=x.merge(y[['row_id','sub_temp']],on='row_id',how='left')
    common=['in_temp','in_hum','in_co2']
    z=d[common].copy()
    for c in common:z[c+'_missing']=d[c].isna().astype(float)
    z['hour_sin']=np.sin(2*np.pi*d.hour/24); z['hour_cos']=np.cos(2*np.pi*d.hour/24)
    z=pd.concat([z,pd.get_dummies(d.farm,prefix='farm',dtype=float)],axis=1)
    aligned=d.day-d.farm.map({'F13':2,'F47':0}).fillna(0)
    rows=[]
    for start,end in [(150,160),(170,180),(189,197),(209,217)]:
        va=d.farm.isin(['F13','F47'])&aligned.ge(start)&aligned.lt(end)&d.sub_temp.notna()
        for scope in ['all_51','target_2','separate_target_farms']:
            predictions=pd.Series(index=d.index[va],dtype=float)
            for farm in (['F13','F47'] if scope=='separate_target_farms' else ['both']):
                train=~va&d.sub_temp.notna()
                val=va.copy()
                if scope=='target_2':train&=d.farm.isin(['F13','F47'])
                elif scope=='separate_target_farms':
                    train&=d.farm.eq(farm); val&=d.farm.eq(farm)
                predictions.loc[val[val].index]=ridge(z.loc[train],z.loc[val],d.loc[train,'sub_temp'].to_numpy())
            for f in ['F13','F47']:
                ids=d.index[va&d.farm.eq(f)]
                rmse=np.sqrt(np.mean((predictions.loc[ids]-d.loc[ids,'sub_temp'])**2))
                rows.append(dict(start=start,end=end,scope=scope,farm=f,n=len(ids),rmse=rmse))
    result=pd.DataFrame(rows); result.to_csv(OUT/'pooling_validation.csv',index=False)
    result['sse']=result.rmse**2*result.n
    a=result.groupby('scope').agg(sse=('sse','sum'),n=('n','sum')); a['rmse']=np.sqrt(a.sse/a.n)
    print(a.to_string())


if __name__=='__main__':main()
