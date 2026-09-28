from cv import *
import pickle
res={}
M,o=run("sub_temp",ghs=tuple(sorted(F_ALL.gh.unique())),twin=True,seeds=(0,),n=1500,params=dict(learning_rate=0.05,num_leaves=63)); summarize("temp_allgh",M,o); res["allgh"]=(M,o)
w=lambda tr: np.where(tr.gh.isin(["F13","F47"]),5.0,1.0)
M,o=run("sub_temp",ghs=tuple(sorted(F_ALL.gh.unique())),twin=True,seeds=(0,),n=1500,params=dict(learning_rate=0.05,num_leaves=63),weight=w); summarize("temp_allgh_w5",M,o); res["allgh_w5"]=(M,o)
pickle.dump(res,open("res_temp3.pkl","wb"))
