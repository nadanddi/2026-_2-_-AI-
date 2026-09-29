import env, common
import numpy as np, pandas as pd, itertools
tX, ty, sX = common.load_raw()
A=pd.concat([tX,sX])
A=A[A.farm.isin(['F13','F47'])]
g=pd.read_csv('local/audit3_03_groups.csv').set_index(['farm','day'])
acts=['act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog','in_temp','in_co2']
V={c:A.pivot_table(index=['farm','day'],columns='hour',values=c) for c in acts}
same=[];rnd=[]
rng=np.random.default_rng(0)
keys=list(g.index)
for k,s in g.groupby('mine'):
    for a,b in itertools.combinations(list(s.index),2): same.append((a,b))
for _ in range(2000):
    a,b=rng.choice(len(keys),2,replace=False); rnd.append((keys[a],keys[b]))
def stats(pairs,c):
    eq=[];cor=[]
    for a,b in pairs:
        x=V[c].loc[a].values; y=V[c].loc[b].values
        eq.append(np.nanmax(np.abs(x-y))<1e-9)
        if np.nanstd(x)>0 and np.nanstd(y)>0: cor.append(np.corrcoef(np.nan_to_num(x),np.nan_to_num(y))[0,1])
    return np.mean(eq).round(3), np.round(np.median(cor),3) if cor else None
for c in acts:
    print(c,'same-date identical,medcorr',stats(same,c),' random',stats(rnd,c))
# same-date pair types
def typ(a,b): return 'within' if a[0]==b[0] else 'cross'
for t in ['within','cross']:
    ps=[p for p in same if typ(*p)==t]
    print(t,len(ps),{c:stats(ps,c)[0] for c in ['act_thermal','act_shade','act_vent','act_heating','act_co2']})
