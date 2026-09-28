"""Q7: are F13/F47 days copies of other farms' days (inside sensors)?"""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True)
P = X.pivot_table(index=['farm','day'], columns='hour', values='in_temp')
Q = X.pivot_table(index=['farm','day'], columns='hour', values='in_hum')
P = P.dropna(); Q=Q.reindex(P.index)
tgt = P.index.get_level_values(0).isin(['F13','F47'])
A = P[tgt].values; B = P[~tgt].values; Bi = P[~tgt].index
Qa = Q[tgt].values; Qb=Q[~tgt].values
res=[]
for i,(key) in enumerate(P[tgt].index):
    d = np.abs(B - A[i]).mean(1)
    j = np.argmin(d)
    dh = np.nanmean(np.abs(Qb[j]-Qa[i]))
    res.append((key[0],key[1],Bi[j][0],Bi[j][1],d[j],dh))
r = pd.DataFrame(res,columns=['farm','day','src','sday','mae_t','mae_h'])
print(r.mae_t.describe()); print((r.mae_t<0.05).mean(), (r.mae_t<0.3).mean())
print(r.sort_values('mae_t').head(20))
r.to_csv('local/eda_time_8_nn.csv',index=False)
