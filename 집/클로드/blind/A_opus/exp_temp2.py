from cv import *
import pickle
res={}
xcols=[c for c in F_ALL.columns if "_x" in c or "ewx" in c]
cfg={
 "base_twin":dict(twin=True),
 "huber1":dict(twin=True,params=dict(objective="huber",alpha=1.0)),
 "l1":dict(twin=True,params=dict(objective="l1")),
 "no_cross_midnight":dict(twin=True,extra_drop=xcols),
 "n1500_lr02":dict(twin=True,n=1500,params=dict(learning_rate=0.02)),
}
for k,c in cfg.items():
    M,o=run("sub_temp",seeds=(0,1),**c); summarize(k,M,o); res[k]=(M,o)
pickle.dump(res,open("res_temp2.pkl","wb"))
