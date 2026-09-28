from cv import *
import pickle
from sklearn.model_selection import GroupKFold
F=add_twin(F_ALL[F_ALL.gh.isin(["F13","F47"])],A)
F=F[F.sub_temp.notna()&(F.is_test==0)].copy()
cols=feats(F,True)
F["grp"]=F.gh+F.day.astype(str)
oof=np.zeros(len(F))
for tr,va in GroupKFold(5).split(F,groups=F.grp):
    m=lgb.train({**P,"seed":0},lgb.Dataset(F.iloc[tr][cols],F.iloc[tr].sub_temp),600)
    oof[va]=m.predict(F.iloc[va][cols])
F["res"]=F.sub_temp-oof
dres=F.groupby(["gh","day"]).res.agg(lambda s: np.sqrt(np.mean(s**2)))
bad=set(dres[dres>2.0].index); print("suspect train days (daily rmse>2):",len(bad),"of",len(dres)); print(sorted(bad))
badrow=set(zip(F.gh[F.res.abs()>4],F.day[F.res.abs()>4],F.hr[F.res.abs()>4]))
pickle.dump((bad,badrow),open("suspect.pkl","wb"))
filt=lambda tr,g,B: ~pd.Series([(a,b) in bad for a,b in zip(tr.gh,tr.day)],index=tr.index)
res={}
for k,c in {"t_clean_base":dict(),"t_clean_l7":dict(n=1500,params=dict(num_leaves=7)),"t_clean_l63":dict(n=600,params=dict(num_leaves=63))}.items():
    M,o=run("sub_temp",twin=True,seeds=(0,1,2),trfilter=filt,**c); summarize(k,M,o); res[k]=(M,o)
pickle.dump(res,open("res_clean.pkl","wb"))
