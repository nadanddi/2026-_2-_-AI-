from common import *
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
A,S=load()
H=A[A.gh.isin(["F13","F47"])].copy()
ECd=H.groupby(["gh","day"]).sub_ec.mean()
V=["in_temp","in_hum","in_co2","act_vent","act_shade","act_thermal","act_heating","act_circfan","act_co2","act_fog","out_temp","out_rad"]
H=H.sort_values(["gh","day","hr"])
for v in V:
    H[v+"_cm"]=H.groupby(["gh","day"])[v].transform(lambda s:s.expanding().mean())
rows=[]
for g in ["F13","F47"]:
    T=day_table(A,g)
    for _,r in T.iterrows():
        if np.isnan(r.twin): continue
        a,b=int(r.twin),int(r.day)
        if (g,a) in ECd and (g,b) in ECd and not np.isnan(ECd[(g,a)]) and not np.isnan(ECd[(g,b)]) and abs(ECd[(g,a)]-ECd[(g,b)])>0.1:
            rows.append((g,a,int(ECd[(g,a)]>ECd[(g,b)]),a)); rows.append((g,b,int(ECd[(g,b)]>ECd[(g,a)]),a))
L=pd.DataFrame(rows,columns=["gh","day","hi","date"])
D=H.merge(L,on=["gh","day"])
F=["hr"]+V+[v+"_cm" for v in V]
for g in ["F13","F47"]:
    d=D[D.gh==g]; dates=d.date.unique(); rng=np.random.RandomState(0); rng.shuffle(dates)
    k=np.array_split(dates,5); p=np.zeros(len(d))
    for kk in k:
        te=d.date.isin(kk).values
        m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=15,min_child_samples=50,verbose=-1).fit(d[~te][F],d[~te].hi)
        p[te]=m.predict_proba(d[te][F])[:,1]
    print(g,"pairs",len(d)//48,"AUC hourly",round(roc_auc_score(d.hi,p),3), "by hour AUC", [round(roc_auc_score(d.hi[d.hr==h],p[d.hr.values==h]),2) for h in [0,6,12,18,23]])
