import sys, os, time, importlib.util
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
t=time.time(); sys.argv=['x']
HERE=r"C:\work\farmai\집\클로드\research"
s=importlib.util.spec_from_file_location("ta2", os.path.join(HERE,"ta2_anchor_debiased_v1.py")); A2=importlib.util.module_from_spec(s); s.loader.exec_module(A2)
R=A2.A1.frames(); print(list(R), time.time()-t)
for k,(M,c) in R.items(): print(k, M.shape, len(c), M.columns.tolist()[:12], M.day.min(), M.day.max(), (M.day>=179).sum())
M,c=R['DIAG10']; os.makedirs(os.path.join(HERE,'agentY_bias','local'),exist_ok=True)
for k,(M,c) in R.items():
    D=M[['row_id','farm','day','hour','sub_temp','in_temp','fold']].copy()
    for i,q in enumerate(c): D['p%d'%i]=q
    D.to_csv(os.path.join(HERE,'agentY_bias','local','temp_%s.csv'%k),index=False)
