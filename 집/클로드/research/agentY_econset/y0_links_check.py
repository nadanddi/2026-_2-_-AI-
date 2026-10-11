import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env, pickle, pandas as pd, numpy as np, os
R=pickle.load(open(r"C:\work\farmai\집\클로드\research\local\linker_v3\base.pkl",'rb'))
r=R['recs'].reset_index(drop=True)
T=pd.read_pickle(r"C:\work\farmai\집\클로드\research\local\linker_v3\truth.pkl")
y=pd.read_csv(os.path.join(env.DATA,'train_y.csv')); y['rid']=y.row_id.str[:7]
m=y.groupby('rid').sub_ec.mean()
r['rid']=r.farm+'_'+r.day.map(lambda d:f"{d:03d}")
T['eb']=m.reindex(r.loc[T.b,'rid']).values; T['ea']=m.reindex(r.loc[T.a1,'rid']).values
for c,g in T.groupby('cert'):
    print('cert',c,len(g),'corr',np.corrcoef(g.eb,g.ea)[0,1].round(3),'spearman',g[['eb','ea']].corr('spearman').iloc[0,1].round(3),'med|diff|',np.median(abs(g.eb-g.ea)).round(3),'sameFarm',g.same_farm.mean().round(2))
print(T.groupby('cert').c1.describe())
for q in [0.1,0.2,0.3,0.5,1.0]:
    g=T[(~T.cert)&(T.c1<q)]; 
    if len(g)>5: print('noncert c1<',q,len(g),g[['eb','ea']].corr('spearman').iloc[0,1].round(3))
print('a1 used by multiple b (cert):', T[T.cert].a1.duplicated().sum())
print('cross-farm cert:', (~T[T.cert].same_farm).sum())
