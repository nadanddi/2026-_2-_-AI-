import env, common
import numpy as np, pandas as pd
tX, ty, sX = common.load_raw()
tX['is_test']=False; sX['is_test']=True
A=pd.concat([tX,sX]); A=A[A.farm.isin(['F13','F47'])].sort_values(['farm','t'])
def dstats(df):
    df=df.sort_values('hour')
    c=df.in_co2.values; T=df.in_temp.values; a=df.act_co2.values; h=df.in_hum.values
    d1=np.diff(c); d2=np.diff(c,2); dT=np.diff(T); dT2=np.diff(T,2)
    noco2=(a[1:]==0)
    def ac1(x):
        x=x[~np.isnan(x)]; 
        return np.corrcoef(x[:-1],x[1:])[0,1] if len(x)>3 and x.std()>0 else np.nan
    return pd.Series(dict(co2_d1=np.nanmedian(np.abs(d1)),co2_d2=np.nanmedian(np.abs(d2)),co2_d1_noco2=np.nanmedian(np.abs(d1[noco2])) if noco2.any() else np.nan,
        T_d1=np.nanmedian(np.abs(dT)),T_d2=np.nanmedian(np.abs(dT2)),hum_d2=np.nanmedian(np.abs(np.diff(h,2))),
        co2_ac1_d1=ac1(d1),T_ac1_d1=ac1(dT),
        co2_mean=np.nanmean(c),T_mean=np.nanmean(T),
        co2_dec=np.mean(np.round(c[~np.isnan(c)])%10==0), T_rng=np.nanmax(T)-np.nanmin(T)))
S=A.groupby(['farm','day']).apply(dstats)
S=S.join(A.groupby(['farm','day']).is_test.first())
q=pd.read_csv('local/anal_q3f_dayscore.csv').set_index(['farm','day'])
dc=pd.read_csv('local/deep_cal_11_days.csv').set_index(['farm','day'])
S=S.join(q[['p_oof','q']]).join(dc[['chain','node','cal']])
S['grp']=np.where(S.is_test,'TEST',np.where(S.q.astype(str).str.startswith('Q4'),'Q4','Q1-3'))
S.loc[S.grp.eq('Q1-3') & S.q.isna(),'grp']='nolabel'
S.to_csv('local/audit3_10_daystats.csv')
cols=['co2_d1','co2_d2','co2_d1_noco2','T_d1','T_d2','hum_d2','co2_ac1_d1','T_ac1_d1','co2_mean','T_mean','T_rng']
print(S.groupby('grp')[cols].median().round(3).to_string())
print(S.groupby(['grp','farm']).size())
# thresholds: Q4 lower quartile of co2_d2
thr=S[S.grp=='Q4'].co2_d2.quantile(.5); thr2=S[S.grp=='Q1-3'].co2_d2.quantile(.9)
print('Q4 median co2_d2',thr,'Q1-3 p90',thr2)
for g in ['Q1-3','Q4','TEST']:
    x=S[S.grp==g]; print(g,'frac co2_d2>=Q4 median %.2f  >= Q1-3 p90 %.2f'%((x.co2_d2>=thr).mean(),(x.co2_d2>=thr2).mean()))
print(S[S.grp=='TEST'].sort_values('co2_d2',ascending=False)[['co2_d1','co2_d2','T_d1','co2_ac1_d1','co2_mean']].head(12).round(2))
