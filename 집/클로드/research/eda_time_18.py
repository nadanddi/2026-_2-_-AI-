"""Q5c/Q1m: cross-midnight lags/EWMs vs day-reset versions (small LGBM, day-block CV)."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import GroupKFold
tX, ty, sX = common.load_raw()
X = pd.concat([tX,sX]).merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left')
X = X[X.farm.isin(['F13','F47'])].sort_values(['farm','t']).reset_index(drop=True)
base=['in_temp','in_hum','in_co2','out_temp','out_rad','act_heating','act_thermal','act_shade','act_vent','act_circfan','act_co2','hour']
def feats(D, reset):
    F=D[base].copy(); F['farm']=(D.farm=='F47').astype(int)
    key = ['farm','day'] if reset else ['farm']
    g=D.groupby(key)
    for c in ['in_temp','in_hum','act_heating']:
        for L in [1,2,3,4,6]:
            v=g[c].shift(L)
            if reset:  # fall back to earliest same-day value
                v=v.fillna(g[c].transform('first'))
            F[f'{c}_lag{L}']=v
        for hl in [3,6,24]:
            F[f'{c}_ewm{hl}']=g[c].transform(lambda s: s.ewm(halflife=hl,ignore_na=True).mean())
    return F
res={}
for reset in [False,True]:
    F=feats(X,reset)
    for tgt in ['sub_temp','sub_ec']:
        m=X[tgt].notna().values
        Fm=F[m]; y=X.loc[m,tgt].values; grp=(X.loc[m,'day']//10).values; hr=X.loc[m,'hour'].values
        pred=np.zeros(len(y))
        for tr,te in GroupKFold(5).split(Fm,y,grp):
            mdl=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=30,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,n_jobs=2,verbose=-1)
            mdl.fit(Fm.iloc[tr],y[tr]); pred[te]=mdl.predict(Fm.iloc[te])
        e=pred-y
        res[(reset,tgt)]=(np.sqrt(np.mean(e**2)), np.sqrt(np.mean(e[hr<=3]**2)), np.sqrt(np.mean(e[hr>3]**2)))
        print('reset' if reset else 'cross', tgt, 'RMSE all %.4f  h0-3 %.4f  h4-23 %.4f'%res[(reset,tgt)])
