from cv import *
import pickle
res={}
cfg={
 "ec_notwin":dict(twin=False),
 "ec_twin":dict(twin=True),
 "ec_twin_huber":dict(twin=True,params=dict(objective="huber",alpha=0.3)),
 "ec_twin_l1":dict(twin=True,params=dict(objective="l1")),
}
for k,c in cfg.items():
    M,o=run("sub_ec",seeds=(0,1),**c); summarize(k,M,o); res[k]=(M,o)
pickle.dump(res,open("res_ec1.pkl","wb"))
