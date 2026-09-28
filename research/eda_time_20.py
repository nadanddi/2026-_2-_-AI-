"""Q1o: 'same-source previous day' features.  Today's hour-0 actuator signature is matched to the
hour-0 signatures of the previous 8 days (all strictly earlier inputs of the same farm); the matched
day's daily input summaries become features.  Compared with the plain previous-day summaries."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import GroupKFold
tX, ty, sX = common.load_raw()
X = pd.concat([tX,sX]).merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left')
X = X[X.farm.isin(['F13','F47'])].sort_values(['farm','t']).reset_index(drop=True)
base=['in_temp','in_hum','in_co2','out_temp','out_rad','act_heating','act_thermal','act_shade','act_vent','act_circfan','act_co2','hour']
SIG=['act_circfan','act_shade','act_thermal','act_co2','act_heating','act_fog','act_vent']
SUMC=['in_temp','in_hum','in_co2','act_heating','act_circfan','act_thermal']
def feats(D, mode):
    F=D[base].copy(); F['farm']=(D.farm=='F47').astype(int)
    g=D.groupby('farm'); gd=D.groupby(['farm','day'])
    for c in ['in_temp','in_hum','act_heating']:
        for L in [1,2,3,4,6]:
            F[f'{c}_lag{L}']=gd[c].shift(L).fillna(gd[c].transform('first'))
        for hl in [3,6,24]:
            F[f'{c}_ewm{hl}']=g[c].transform(lambda s: s.ewm(halflife=hl,ignore_na=True).mean())
    for c in SIG+['in_co2','in_hum']:
        F[f'{c}_h0']=gd[c].transform('first')
        F[f'{c}_dexp']=gd[c].transform(lambda s: s.expanding().mean())
    if mode=='none': return F
    rows=[]
    for f,d in D.groupby('farm'):
        h0=d[d.hour==0].set_index('day')[SIG]
        sd=h0.std()+1e-6
        daym=d.groupby('day')[SUMC].mean(); dayend=d[d.hour>=20].groupby('day')[SUMC].mean()
        days=sorted(d.day.unique())
        for x in days:
            cands=[p for p in days if x-8<=p<x]
            if mode=='prev': pick=max(cands) if cands else None
            else:
                if not cands or x not in h0.index: pick=None
                else:
                    dist=((h0.loc[cands]-h0.loc[x])/sd).abs().sum(1)
                    pick=dist.idxmin()
            r={'farm':f,'day':x,'gap':(x-pick) if pick else np.nan}
            for c in SUMC:
                r[f'src_{c}_m']=daym[c].get(pick,np.nan) if pick else np.nan
                r[f'src_{c}_end']=dayend[c].get(pick,np.nan) if pick else np.nan
            rows.append(r)
    S=pd.DataFrame(rows)
    M=D[['farm','day']].merge(S,on=['farm','day'],how='left')
    for c in M.columns[2:]: F[c]=M[c].values
    return F
for mode in ['none','prev','match']:
    F=feats(X,mode); out=[]
    for tgt in ['sub_temp','sub_ec']:
        m=X[tgt].notna().values
        Fm=F[m]; y=X.loc[m,tgt].values; grp=(X.loc[m,'day']//10).values; hr=X.loc[m,'hour'].values
        pred=np.zeros(len(y))
        for tr,te in GroupKFold(5).split(Fm,y,grp):
            mdl=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=30,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,n_jobs=2,verbose=-1)
            mdl.fit(Fm.iloc[tr],y[tr]); pred[te]=mdl.predict(Fm.iloc[te])
        e=pred-y
        out.append('%s all %.4f h0-3 %.4f'%(tgt,np.sqrt(np.mean(e**2)),np.sqrt(np.mean(e[hr<=3]**2))))
    print(mode,' | '.join(out), ' mean gap', round(F['gap'].mean(),2) if 'gap' in F else '')
