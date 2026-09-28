from cv import *
import pickle
wcols=[c for c in F_ALL.columns if c.startswith("out_") or c.startswith("ot_") or c.startswith("orad")]
res={}
cfg={"ec_nw_l3_n3000":dict(t="sub_ec",extra_drop=wcols,n=3000,params=dict(num_leaves=3)),
     "ec_nw_l7_n1500_twin":dict(t="sub_ec",twin=True,extra_drop=wcols,n=1500,params=dict(num_leaves=7)),
     "ec_nw_l7_n1500_ff5":dict(t="sub_ec",extra_drop=wcols,n=1500,params=dict(num_leaves=7,feature_fraction=0.5)),
     "t_l7_n1500":dict(t="sub_temp",twin=True,n=1500,params=dict(num_leaves=7)),
     "t_l15_n1200":dict(t="sub_temp",twin=True,n=1200,params=dict(num_leaves=15)),
     "t_l63_n600":dict(t="sub_temp",twin=True,n=600,params=dict(num_leaves=63))}
for k,c in cfg.items():
    t=c.pop("t"); tw=c.pop("twin",False)
    M,o=run(t,twin=tw,seeds=(0,1,2),**c); summarize(k,M,o); res[k]=(M,o)
pickle.dump(res,open("res_5.pkl","wb"))
