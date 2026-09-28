"""Q2/Q3: map every farm-day to a calendar anchor through identical outside-weather groups."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
g=pd.read_csv('local/eda_time_9_groups.csv')
g=g[g.farm.isin(['F13','F47'])]
for f in ['F13','F47']:
    L=pd.read_csv(f'local/eda_time_10_{f}.csv').set_index('day')
    gg=g[g.farm==f].set_index('day')
    rows=[]
    for x in L.index:
        gid=gg.gid.get(x)
        mem=g[(g.gid==gid)] if gid is not None else g.iloc[0:0]
        same=mem[(mem.farm==f)&(mem.day!=x)].day.tolist()
        oth=mem[mem.farm!=f].day.tolist()
        rows.append((x,L.test[x],L.cl[x],same,oth))
    R=pd.DataFrame(rows,columns=['day','test','cl','sib_same_farm','sib_other_farm']).set_index('day')
    late=R[R.index>=179]
    print(f,'--- late part (>=179): same-farm siblings and cluster of sibling')
    for x,r in late.iterrows():
        sc=[ (s, int(L.cl.get(s,-1))) for s in r.sib_same_farm]
        print(x,'T' if r.test else ' ', 'cl',r.cl,'sib',sc,'other',r.sib_other_farm)
