import env
import numpy as np, pandas as pd
S=pd.read_csv('local/audit3_10_daystats.csv').set_index(['farm','day'])
S['rough']=(S.co2_d2>=S[S.grp=='Q4'].co2_d2.quantile(.25))
S['Q4']=S.grp.eq('Q4')
tr=S[S.grp.isin(['Q1-3','Q4'])]
print('chain values', S.chain.nunique(), S.chain.value_counts().head(10).to_dict())
print('Q4 by chain (train):'); print(tr.groupby('chain').Q4.agg(['size','mean']).round(2).T.to_string())
print('rough by chain in TEST:'); print(S[S.grp=='TEST'].groupby('chain').rough.agg(['size','mean']).round(2).T.to_string())
from scipy.stats import chi2_contingency
ct=pd.crosstab(tr.chain,tr.Q4); print('chi2 p chain vs Q4', chi2_contingency(ct)[1])
# segment
tr=tr.assign(seg=np.where(tr.index.get_level_values(1)<179,1,2))
print(tr.groupby(['farm','seg']).Q4.agg(['size','mean']).round(2))
# persistence along record: Q4 at d vs d+1, d+2
for f in ['F13','F47']:
    x=tr.loc[f].Q4
    for k in [1,2,3]:
        a=x.reindex(x.index+k).values; b=x.values; m=~pd.isna(a)
        print(f,'lag',k,'P(Q4 next|Q4) %.2f  P(Q4 next|not) %.2f'%(a[m][b[m]].astype(float).mean(),a[m][~b[m]].astype(float).mean()))
# same-date partners concordance
g=pd.read_csv('local/audit3_03_groups.csv').set_index(['farm','day'])
tr=tr.join(g.mine)
pairs=[]
for k,s in tr.groupby('mine'):
    v=s.Q4.values
    for i in range(len(v)):
        for j in range(i+1,len(v)): pairs.append((v[i],v[j]))
p=np.array(pairs); base=tr.Q4.mean()
print('same-date pair both Q4 %.3f, expected %.3f; P(Q4|partner Q4) %.2f'%((p[:,0]&p[:,1]).mean(),base**2,(p[:,0]&p[:,1]).sum()/max(1,p[:,0].sum())))
