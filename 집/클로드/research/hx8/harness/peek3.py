import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import pandas as pd, numpy as np
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
E0 = pd.read_csv(os.path.join(env.LOCAL,"dmt3e_E0_days.csv"))
E0["y"]=[float(np.nanmean(EC[idx[r]])) for r in E0.rec]
H=E0[E0.y>=1].sort_values(["seg","day"])
print(H.to_string(index=False))
# per seg, per day: max y
g=E0.groupby(["seg","day"]).y.agg(["max","count"]).reset_index()
for s,G in g.groupby("seg"):
    print(s, " ".join(f"{int(d)}:{m:.2f}" for d,m in zip(G.day,G["max"])))
