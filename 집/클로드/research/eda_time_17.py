"""Q5b: optimal lag of air->substrate, and whether lags crossing midnight carry information."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX,sX]).merge(ty[['row_id','sub_temp']],on='row_id',how='left').sort_values(['farm','t'])
for f in ['F13','F47','F05']:
    D=X[X.farm==f].set_index('t')
    it=D.in_temp
    out={}
    for k in range(0,9):
        D[f'L{k}']=it.reindex(D.index-k).values
    # residual of sub_temp around day mean using same-day hours 6..23 only (all lags within day)
    E=D[(D.hour>=8)].dropna(subset=["sub_temp","in_temp"]+[f"L{k}" for k in range(9)])
    g=E.groupby('day')
    st=E.sub_temp-g.sub_temp.transform('mean')
    print(f,'within-day corr by lag (hours>=8):', {k: round(np.corrcoef(st, E[f'L{k}']-g[f'L{k}'].transform('mean'))[0,1],3) for k in range(9)})
    # level: corr of sub_temp with lagged in_temp in raw level, split by whether lag crosses midnight
    Z=D.dropna(subset=["sub_temp","in_temp"])
    for k in [1,2,3,4]:
        a=Z[Z.hour>=k]; b=Z[Z.hour<k]
        ra=np.sqrt(np.mean((a.sub_temp-a[f'L{k}'])**2 - 0)); 
        # fit simple linear per group
        def r2(z):
            z=z.dropna(subset=[f'L{k}']); c=np.polyfit(z[f'L{k}'],z.sub_temp,1); e=z.sub_temp-np.polyval(c,z[f'L{k}']); return round(np.sqrt(np.mean(e**2)),3)
        def r2own(z):
            c=np.polyfit(z.in_temp,z.sub_temp,1); e=z.sub_temp-np.polyval(c,z.in_temp); return round(np.sqrt(np.mean(e**2)),3)
        print(f,' lag',k,'rows with lag inside day: RMSE lin(sub~in_temp[t-k])',r2(a),'vs lin(sub~in_temp[t])',r2own(a),'| lag crosses midnight:',r2(b),'vs',r2own(b))
