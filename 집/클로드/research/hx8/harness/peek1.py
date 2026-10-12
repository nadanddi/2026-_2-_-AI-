import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import pandas as pd, numpy as np
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(env.ROOT,"집","클로드","research","ec0h","rise","dm11_calendar_v1.csv")); print(CAL.columns.tolist()); DAY=dict(zip(CAL.rec,CAL.day))
E0 = pd.read_csv(os.path.join(env.LOCAL,"dmt3e_E0_days.csv")); E0d=dict(zip(E0.rec,zip(E0.seg,E0.day)))
S = pd.read_csv(r"C:\work\farmai\제출\13회차_2026-10-12(나)\submission_22.csv"); S["rec"]=S.row_id.str[:7]; P=S.groupby("rec").sub_ec.mean()
TL = pd.read_pickle(os.path.join(S3,"truth.pkl")); pre={rid(int(b)):(rid(int(a)),bool(c)) for a,b,c in zip(TL.a1,TL.b,TL.cert)}
for f,lo,hi in (("F13",210,245),("F47",205,240)):
    for n in range(lo,hi):
        r=f"{f}_{n:03d}"
        if r not in idx: print(r,"MISSING(deleted)", DAY.get(r)); continue
        i=idx[r]; tr=TRAIN[i]
        print(r, "TRAIN" if tr else "EVAL ", "day",DAY.get(r), "E0",E0d.get(r), "ymean %.3f y0 %.3f y23 %.3f"%(np.nanmean(EC[i]),EC[i,0],EC[i,23]) if tr else "pred %.3f"%P.get(r,np.nan), "pre",pre.get(r))
