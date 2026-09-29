"""Q1n: value of causal source-signature features (same-day expanding actuator stats) for LGBM."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import GroupKFold
import importlib.util, sys
spec=importlib.util.spec_from_file_location('e18','eda_time_18.py')
tX, ty, sX = common.load_raw()
X = pd.concat([tX,sX]).merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left')
X = X[X.farm.isin(['F13','F47'])].sort_values(['farm','t']).reset_index(drop=True)
base=['in_temp','in_hum','in_co2','out_temp','out_rad','act_heating','act_thermal','act_shade','act_vent','act_circfan','act_co2','hour']
def feats(D, sig, reset_short):
    F=D[base].copy(); F['farm']=(D.farm=='F47').astype(int)
    g=D.groupby('farm'); gd=D.groupby(['farm','day'])
    for c in ['in_temp','in_hum','act_heating']:
        for L in [1,2,3,4,6]:
            v=g[c].shift(L)
            if reset_short: v=gd[c].shift(L).fillna(gd[c].transform('first'))
            F[f'{c}_lag{L}']=v
        for hl in [3,6,24]:
            F[f'{c}_ewm{hl}']=g[c].transform(lambda s: s.ewm(halflife=hl,ignore_na=True).mean())
    if sig:
        for c in ['act_circfan','act_shade','act_thermal','act_co2','act_heating','act_fog','act_vent','in_co2','in_hum']:
            F[f'{c}_h0']=gd[c].transform('first')
            F[f'{c}_dexp']=gd[c].transform(lambda s: s.expanding().mean())
            F[f'{c}_dzero']=gd[c].transform(lambda s: (s==0).expanding().mean())
        F['fan_eq_heat_dexp']=(D.act_circfan==D.act_heating).groupby([D.farm,D.day]).transform(lambda s: s.expanding().mean())
        F['shade_eq_th_dexp']=(D.act_shade==D.act_thermal).groupby([D.farm,D.day]).transform(lambda s: s.expanding().mean())
    return F
for sig,rs in [(False,False),(True,False),(True,True)]:
    F=feats(X,sig,rs)
    out=[]
    for tgt in ['sub_temp','sub_ec']:
        m=X[tgt].notna().values
        Fm=F[m]; y=X.loc[m,tgt].values; grp=(X.loc[m,'day']//10).values; hr=X.loc[m,'hour'].values
        pred=np.zeros(len(y))
        for tr,te in GroupKFold(5).split(Fm,y,grp):
            mdl=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=30,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,n_jobs=2,verbose=-1)
            mdl.fit(Fm.iloc[tr],y[tr]); pred[te]=mdl.predict(Fm.iloc[te])
        e=pred-y
        out.append('%s all %.4f h0-3 %.4f'%(tgt,np.sqrt(np.mean(e**2)),np.sqrt(np.mean(e[hr<=3]**2))))
    print('sig',sig,'reset_short',rs,' | '.join(out))
