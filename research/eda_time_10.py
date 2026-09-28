"""Q1i: cluster days into latent source greenhouses by actuator fingerprint."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True).merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left')
X['test']=X.row_id.isin(sX.row_id)
g = pd.read_csv('local/eda_time_9_groups.csv')
acts=['act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
def fp(d):
    r={}
    for a in acts:
        r[a+'_m']=d[a].mean(); r[a+'_z']=(d[a]==0).mean(); r[a+'_n']=d[a].nunique()
    r['fan_eq_heat']=(d.act_circfan==d.act_heating).mean()
    r['shade_eq_th']=(d.act_shade==d.act_thermal).mean()
    r['co2_m']=d.in_co2.mean(); r['hum_m']=d.in_hum.mean()
    return pd.Series(r)
for f in ['F13','F47']:
    d=X[X.farm==f]
    F=d.groupby('day').apply(fp)
    lab=d.groupby('day').agg(ec=('sub_ec','mean'),st=('sub_temp','mean'),it=('in_temp','mean'),test=('test','max'))
    Z=StandardScaler().fit_transform(F.fillna(0))
    for k in [2,3,4,5,6]:
        km=KMeans(k,n_init=20,random_state=0).fit(Z)
        print(f,k,'silhouette',round(silhouette_score(Z,km.labels_),3))
    km=KMeans(4,n_init=20,random_state=0).fit(Z)
    F['cl']=km.labels_
    L=lab.join(F[['cl','act_circfan_m','act_shade_m','act_thermal_m','act_co2_m','fan_eq_heat','shade_eq_th']])
    L=L.join(g[g.farm==f].set_index('day')[['gid','gsize']])
    print(L.groupby('cl').agg(n=('ec','size'),ntest=('test','sum'),ec=('ec','mean'),ecsd=('ec','std'),st=('st','mean'),it=('it','mean'),fan=('act_circfan_m','mean'),shade=('act_shade_m','mean'),th=('act_thermal_m','mean'),fe=('fan_eq_heat','mean'),se=('shade_eq_th','mean'),dmin=('ec',lambda s: s.index.min()),dmax=('ec',lambda s: s.index.max())).round(3))
    # same-weather siblings within farm: same cluster?
    sib=L.dropna(subset=['gid']).groupby('gid').cl.agg(lambda s: (len(s)>1, s.nunique()==len(s)))
    sib=sib[sib.map(lambda t:t[0])]
    print(' within-farm sibling pairs in different clusters:', np.mean([t[1] for t in sib]).round(3), 'n', len(sib))
    L.to_csv(f'local/eda_time_10_{f}.csv')
    print(''.join(str(c) for c in L.cl.values))
