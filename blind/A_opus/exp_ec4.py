from cv import *
import pickle
wcols=[c for c in F_ALL.columns if c.startswith("out_") or c.startswith("ot_") or c.startswith("orad")]
xcols=[c for c in F_ALL.columns if "_x" in c or "ewx" in c]
res={}
cfg={"nw":dict(extra_drop=wcols),
     "nw_nx":dict(extra_drop=wcols+xcols),
     "nw_n400":dict(extra_drop=wcols,n=400),
     "nw_mcs150":dict(extra_drop=wcols,params=dict(min_child_samples=150)),
     "nw_leaves7_n1500":dict(extra_drop=wcols,n=1500,params=dict(num_leaves=7)),
     "nw_twin":dict(extra_drop=wcols,twin=True)}
for k,c in cfg.items():
    tw=c.pop("twin",False)
    M,o=run("sub_ec",twin=tw,seeds=(0,1,2),**c); summarize(k,M,o); res[k]=(M,o)
pickle.dump(res,open("res_ec4.pkl","wb"))
