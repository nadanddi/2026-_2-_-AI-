import env
import numpy as np, pandas as pd
S=pd.read_csv('local/audit3_10_daystats.csv').set_index(['farm','day'])
g=pd.read_csv('local/audit3_03_groups.csv').set_index(['farm','day'])
S=S.join(g.mine); S['Q4']=S.grp.eq('Q4')
tr=S[S.grp.isin(['Q1-3','Q4'])]
within=[];cross=[]
for k,s in tr.groupby('mine'):
    it=list(s.itertuples())
    for i in range(len(it)):
        for j in range(i+1,len(it)):
            a,b=it[i],it[j]
            (within if a.Index[0]==b.Index[0] else cross).append((a.Q4,b.Q4))
for n,p in [('within-farm same date',within),('cross-farm same date',cross)]:
    p=np.array(p); print(n,len(p),'P(Q4|partner Q4)=%.2f base %.2f'%((p[:,0]&p[:,1]).sum()*2/max(1,p.sum()),tr.Q4.mean()))
# runs in record
for f in ['F13','F47']:
    x=tr.loc[f].Q4; d=x[x].index.tolist()
    print(f,'Q4 days:',d)
# joint rough signature: low ac1 & high d2
def sig(x): return ((x.co2_ac1_d1<0.25)&(x.co2_d2>=6)).mean()
for gname in ['Q1-3','Q4','TEST']: print(gname,'signature rate %.2f'%sig(S[S.grp==gname]))
print(S[S.grp=='TEST'][lambda x:(x.co2_ac1_d1<0.25)&(x.co2_d2>=6)][['co2_d1','co2_d2','co2_ac1_d1','T_d1']])
# distribution quantiles of co2_d2 and ac1
for c in ['co2_d2','co2_ac1_d1','T_d1','T_ac1_d1']:
    print(c,{gname:np.round(S[S.grp==gname][c].quantile([.1,.5,.9]).values,2).tolist() for gname in ['Q1-3','Q4','TEST']})
