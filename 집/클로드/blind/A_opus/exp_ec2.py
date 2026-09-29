from cv import *
import pickle
# date keys (weather fingerprint) for "same-date" ablation -- analysis only
R=pd.read_pickle("daykeys.pkl"); gid={k:i for i,k in enumerate(R.ot.unique())}; R["gid"]=R.ot.map(gid)
DK=dict(zip(zip(R.gh,R.day),R.gid))
def no_same_date(tr,g,B):
    ds={DK.get((g,d)) for d in B}
    return ~pd.Series([DK.get((a,b)) in ds for a,b in zip(tr.gh,tr.day)],index=tr.index)
def causal(tr,g,B): return tr.day<B[0]
wcols=[c for c in F_ALL.columns if c.startswith("out_") or c.startswith("ot_") or c.startswith("orad")]
res={}
cfg={"ec_no_weather":dict(extra_drop=wcols),
     "ec_no_same_date_rows":dict(trfilter=no_same_date),
     "ec_causal_train":dict(trfilter=causal),
     "temp_no_same_date_rows":dict(trfilter=no_same_date,target="sub_temp"),
     "temp_causal_train":dict(trfilter=causal,target="sub_temp")}
for k,c in cfg.items():
    t=c.pop("target","sub_ec")
    M,o=run(t,twin=False,seeds=(0,1),**c); summarize(k,M,o); res[k]=(M,o)
pickle.dump(res,open("res_ec2.pkl","wb"))
