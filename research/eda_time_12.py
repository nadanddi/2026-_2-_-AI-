"""Q1k: does source (fingerprint cluster) switching explain midnight jumps and EC level?"""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True).merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left')
for f in ['F13','F47']:
    L=pd.read_csv(f'local/eda_time_10_{f}.csv').set_index('day')
    d=X[X.farm==f]
    P={c:d.pivot_table(index='day',columns='hour',values=c).reindex(L.index) for c in ['sub_temp','sub_ec','in_temp']}
    rows=[]
    for x in L.index:
        if x-1 in L.index:
            rows.append((x, L.cl[x]==L.cl[x-1], abs(P['sub_temp'].loc[x,0]-P['sub_temp'].loc[x-1,23]), abs(P['sub_ec'].loc[x,0]-P['sub_ec'].loc[x-1,23]), abs(P['in_temp'].loc[x,0]-P['in_temp'].loc[x-1,23])))
    R=pd.DataFrame(rows,columns=['day','same','jst','jec','jit'])
    print(f, R.groupby('same')[['jst','jec','jit']].mean().round(3).assign(n=R.groupby('same').size()))
    lab=L.dropna(subset=['ec'])
    tot=lab.ec.var(); within=lab.groupby('cl').ec.var().mul(lab.groupby('cl').size()-1).sum()/(len(lab)-1)
    print(f,' EC daily var explained by 4-cluster id: %.3f' % (1-within/tot))
    tot=lab.st.var(); within=lab.groupby('cl').st.var().mul(lab.groupby('cl').size()-1).sum()/(len(lab)-1)
    print(f,' sub_temp daily var explained by cluster id: %.3f' % (1-within/tot))
