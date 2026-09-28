from features import *
import time
t0=time.time()
A,S=load()
ghs=sorted(A.gh.unique())
F=build(A,ghs)
F.to_pickle("feat_all.pkl"); print(F.shape, time.time()-t0)
