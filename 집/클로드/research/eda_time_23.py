"""Q3c: residual target (sub_temp - air-temperature anchor) vs raw target, scored on cold held-out rows."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import GroupKFold
src=open('eda_time_20.py',encoding='utf-8').read()
src=src[:src.index("for mode in")]
exec(src)
F=feats(X,'none')
m=X.sub_temp.notna().values
Fm=F[m].reset_index(drop=True); y=X.loc[m,'sub_temp'].values; grp=(X.loc[m,'day']//10).values; it=X.loc[m,'in_temp'].values
# anchor: day-reset EWM of in_temp with halflife 3 (air leads substrate by ~3h)
anchors={'raw':np.zeros(len(y)),'in_temp':Fm.in_temp.fillna(Fm.in_temp_ewm3).values,'ewm3':Fm.in_temp_ewm3.values,'lag3':Fm.in_temp_lag3.fillna(Fm.in_temp_ewm3).values}
for name,a in anchors.items():
    pred=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(Fm,y,grp):
        mdl=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=30,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,n_jobs=2,verbose=-1)
        mdl.fit(Fm.iloc[tr],y[tr]-a[tr]); pred[te]=mdl.predict(Fm.iloc[te])+a[te]
    e=pred-y
    print(f'{name:8s} all {np.sqrt(np.mean(e**2)):.4f}  in_temp<10 {np.sqrt(np.mean(e[it<10]**2)):.4f} (n={np.sum(it<10)})  in_temp<8 {np.sqrt(np.mean(e[it<8]**2)):.4f}  bias<10 {np.mean(e[it<10]):+.3f}')
# extrapolation-style check: train only on days with daily mean in_temp>=12, test on colder days
dm=X[m].groupby(['farm','day']).in_temp.transform('mean').values
trm=dm>=12; tem=dm<11
for name,a in anchors.items():
    mdl=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=30,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,n_jobs=2,verbose=-1)
    mdl.fit(Fm[trm],y[trm]-a[trm]); e=mdl.predict(Fm[tem])+a[tem]-y[tem]
    print(f'extrap warm->cold {name:8s} RMSE {np.sqrt(np.mean(e**2)):.4f} bias {np.mean(e):+.3f} n={tem.sum()}')
