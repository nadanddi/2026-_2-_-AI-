import env, common
import numpy as np, pandas as pd
from collections import Counter
from scipy.sparse.csgraph import connected_components
tX, ty, sX = common.load_raw()
tX['is_test']=False; sX['is_test']=True
A=pd.concat([tX,sX])
B=A[A.farm.isin(['F13','F47','F32'])]
P=B.pivot_table(index=['farm','day'],columns='hour',values='out_temp')
P=P.dropna()
idx=P.index; M=P.values
D=np.sqrt(((M[:,None]-M[None])**2).mean(-1))
adj=(D<0.05)
nc,lab=connected_components(adj,directed=False)
lab=pd.Series(lab,index=idx)
isF32=np.array([f=='F32' for f,d in idx])
print('components',nc)
# F32 matches
m32=[(f,d) for (f,d),l in lab.items() if f!='F32' and (lab[isF32]==l).any()]
print('F13/F47 days sharing exact out_temp with F32:',len(m32))
# closest F32 day for each F13 day
i13=np.where(~isF32)[0]; i32=np.where(isF32)[0]
if len(i32):
    dm=D[np.ix_(i13,i32)].min(1); print('F13/47 -> nearest F32 out_temp rmse quantiles',np.quantile(dm,[0,.1,.5]).round(2))
L=lab[~isF32]
grp=L.groupby(L).apply(lambda s:list(s.index))
print('cal groups (F13+F47)',len(grp))
print(Counter(tuple(sorted(Counter(f for f,d in v).items())) for v in grp).most_common(10))
# compare with deep_cal_11
dc=pd.read_csv('local/deep_cal_11_days.csv').set_index(['farm','day'])
j=pd.DataFrame({'mine':L}).join(dc[['cal','chain','is_test']])
# are my groups consistent with their cal? each my-group -> unique cal?
print('my groups mapping to >1 cal:', (j.groupby('mine').cal.nunique()>1).sum(), ' their cals mapping to >1 my group:', (j.groupby('cal').mine.nunique()>1).sum(), 'their ncal', j.cal.nunique())
j.to_csv('local/audit3_03_groups.csv')
# test days: group partners
t=j[j.is_test]
for (f,d),r in t.iterrows():
    mem=[x for x in grp[r.mine] if x!=(f,d)]
    pass
tp=[]
for (f,d),r in t.iterrows():
    mem=grp[r.mine]
    tp.append((f,d,[(ff,dd) for ff,dd in mem if (ff,dd)!=(f,d)]))
for x in tp: print(x)
