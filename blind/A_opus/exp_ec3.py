import cv
from cv import *
from feat_extra import add_fingerprint
import pickle
cv.F_ALL=add_fingerprint(cv.F_ALL[cv.F_ALL.gh.isin(["F13","F47"])])
res={}
for k,c in {"ec_fp":dict(),"ec_fp_log":dict(log=True),"temp_fp":dict(target="sub_temp")}.items():
    t=c.pop("target","sub_ec"); lg=c.pop("log",False)
    if lg:
        cv.F_ALL["log_ec"]=np.log(cv.F_ALL.sub_ec); cv.DROP.add("log_ec")
        M,o=run("log_ec",twin=False,seeds=(0,1)); o={k2:(np.exp(v[0]),np.exp(v[1])) for k2,v in o.items()}
        M=np.array([[np.sqrt(np.mean((v[0]-v[1])**2)) for v in o.values()]])
    else:
        M,o=run(t,twin=(t=="sub_temp"),seeds=(0,1),**c)
    summarize(k,M,o); res[k]=(M,o)
pickle.dump(res,open("res_ec3.pkl","wb"))
