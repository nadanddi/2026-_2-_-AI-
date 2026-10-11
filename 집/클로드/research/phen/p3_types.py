from load import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
D=load(); D['key']=D.farm+'_'+D.rec.astype(str)
P={'n':range(0,7),'m':range(7,12),'a':range(12,17),'e':range(17,24)}
A=['act_vent','act_shade','act_thermal','act_heating','act_circfan','act_co2','act_fog']
rows=[]
for k,g in D.groupby('key'):
    g=g.set_index('h')
    d={'key':k,'farm':k[:3],'rec':g.rec.iloc[0],'src':g.src.iloc[0],'phase':g.phase.iloc[0]}
    for a in A:
        for p,hh in P.items(): d[f'{a}_{p}']=g.loc[list(hh),a].mean()
    d['Tin_n']=g.loc[list(P['n']),'in_temp'].mean(); d['Tin_max']=g.in_temp.max()
    d['RH_mean']=g.in_hum.mean(); d['CO2_n']=g.loc[list(P['n']),'in_co2'].mean(); d['Tout']=g.out_temp.mean(); d['Rad']=g.out_rad.sum()
    d['ec_mean']=g.sub_ec.mean(); d['ec25_mean']=g.ec25.mean(); d['ec0']=g.sub_ec.get(0)
    for h in [0,6,9,12,15,18,23]: d[f'ecs{h}']=g.sub_ec.get(h)-g.sub_ec.mean(); d[f'e25s{h}']=g.ec25.get(h)-g.ec25.mean()
    d['dTs_n']=g.loc[list(P['n']),'dTs'].mean(); d['dTs_d']=g.loc[list(P['a']),'dTs'].mean()
    d['sub_n']=g.loc[list(P['n']),'sub_temp'].mean()
    rows.append(d)
R=pd.DataFrame(rows)
feat=[c for c in R.columns if c.startswith('act_')]+['Tin_n','Tin_max','RH_mean','CO2_n']
# drop all-zero/near-constant
feat=[f for f in feat if R[f].std()>0.5]
Z=R[feat].fillna(R[feat].median())
tr=R.src=='train'
sc=StandardScaler().fit(Z[tr])
best=None
for k in [4,5,6]:
    km=KMeans(k,n_init=20,random_state=0).fit(sc.transform(Z[tr]))
    print(k,round(km.inertia_))
K=5
km=KMeans(K,n_init=20,random_state=0).fit(sc.transform(Z[tr]))
R['type']=km.predict(sc.transform(Z))
pd.set_option('display.width',300)
cen=R.groupby('type')[feat].mean().round(0)
print(cen.T.to_string())
g=R.groupby('type')
S=pd.DataFrame({'n_1차tr':g.apply(lambda x:((x.phase==1)&(x.src=='train')).sum()),
 'n_2차tr':g.apply(lambda x:((x.phase==2)&(x.src=='train')).sum()),'n_평가':g.apply(lambda x:(x.src=='test').sum()),
 'F47비율':g.apply(lambda x:(x.farm=='F47').mean()).round(2),
 'Tout':g.Tout.mean().round(1),'Rad':g.Rad.mean().round(0),'Tin_n':g.Tin_n.mean().round(1),'RH':g.RH_mean.mean().round(0),
 'EC평균':g.ec_mean.mean().round(3),'EC중앙':g.ec_mean.median().round(3),'EC25평균':g.ec25_mean.mean().round(3),
 'dTs_밤':g.dTs_n.mean().round(2),'dTs_오후':g.dTs_d.mean().round(2),'배지밤':g.sub_n.mean().round(1)})
print(S.to_string())
print('EC 하루모양(원EC, 날평균 대비)');print(g[[f'ecs{h}' for h in [0,6,9,12,15,18,23]]].mean().round(3).to_string())
print('EC25 하루모양');print(g[[f'e25s{h}' for h in [0,6,9,12,15,18,23]]].mean().round(3).to_string())
# share by farm x src
print(pd.crosstab([R.farm,R.src,R.phase],R.type,normalize='index').round(2))
# per-farm EC per type (train)
print(R[tr].groupby(['farm','type']).ec_mean.agg(['count','mean']).round(3).unstack(0))
# within-farm-phase2 interleaving: type sequence
for f in ['F13','F47']:
    s=R[(R.farm==f)&(R.phase==2)].sort_values('rec')
    print(f,''.join(str(t) if src=='train' else chr(ord('a')+t) for t,src in zip(s.type,s.src)))
R.to_csv(os.path.join(os.path.dirname(__file__),'types.csv'),index=False)
