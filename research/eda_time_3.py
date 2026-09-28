"""Q1c: successor structure across both farms, parity, and jump size to best successor."""
import env, common
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True)
X = X[X.farm.isin(['F13','F47'])]
C = ['in_temp','in_hum','in_co2']
sd = {c: X[c].std() for c in C}
keys=[]; E=[]; S=[]; SL=[]
for f in ['F13','F47']:
    d = X[X.farm==f]
    P = {c: d.pivot_table(index='day', columns='hour', values=c) for c in C}
    for dd in P['in_temp'].index:
        keys.append((f,dd))
        E.append([P[c].loc[dd,23]/sd[c] for c in C]); S.append([P[c].loc[dd,0]/sd[c] for c in C])
        SL.append([(P[c].loc[dd,23]-P[c].loc[dd,22])/sd[c] for c in C])
E=np.array(E); S=np.array(S); SL=np.array(SL)
D = np.sqrt(((E[:,None]+0.5*SL[:,None]-S[None])**2).sum(-1))
for i in range(len(keys)): D[i,i]=np.nan
keys_df = pd.DataFrame(keys, columns=['farm','day'])
out=[]
for i,(f,dd) in enumerate(keys):
    r = D[i]
    if np.all(np.isnan(r)): continue
    j = np.nanargmin(r); f2,d2 = keys[j]
    out.append((f,dd,f2,d2-dd, r[j]))
o = pd.DataFrame(out, columns=['farm','day','farm2','off','dist'])
print(pd.crosstab([o.farm,o.farm2], o.off.clip(-5,5)))
o['par']=o.day%2
print('same-farm off==2 by parity'); print(o[o.farm==o.farm2].groupby(['farm','par']).off.apply(lambda s:(s==2).mean()))
print(o[o.farm==o.farm2].groupby(['farm','par']).off.apply(lambda s:(s==1).mean()))
# raw jump in in_temp for true d->d+1, d->d+2 by parity
for f in ['F13','F47']:
    d = X[X.farm==f]; P=d.pivot_table(index='day',columns='hour',values='in_temp')
    for k in [1,2]:
        for par in [0,1]:
            v=[abs(P.loc[dd,23]-P.loc[dd+k,0]) for dd in P.index if dd%2==par and dd+k in P.index]
            print(f,'k',k,'parity',par,'mean |jump|',round(np.nanmean(v),3), 'n',len(v))
