# H-HOUSE h4: temp pass-2 loose-chain lag-5 partial corr (+.56) - outlier check
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
OUT = r"C:\work\farmai\집\클로드\research\agentX_house\local"
CH = pd.read_csv(os.path.join(OUT, "chains_loose.csv")); PR = pd.read_csv(os.path.join(OUT, "preds_temp_loose.csv"))
d = dict(zip(PR.rid, PR.d)); CH = CH[CH.chain >= 0]
rows = []
for c, g in CH.groupby("chain"):
    s = g.sort_values("pos").rid.tolist()
    for i in range(5, len(s)):
        t, a1, a5 = s[i], s[i-1], s[i-5]
        if int(t[4:]) >= 179 and all(x in d for x in (t, a1, a5)): rows.append((t, a5, d[t], d[a1], d[a5]))
X = pd.DataFrame(rows, columns=["t", "a5", "dt", "d1", "d5"]); print(X.round(2).to_string(index=False))
for ex in ([], ["F47_191"]):
    Y = X[~X.t.isin(ex) & ~X.a5.isin(ex)]
    e0 = Y.dt - np.polyval(np.polyfit(Y.d1, Y.dt, 1), Y.d1); e5 = Y.d5 - np.polyval(np.polyfit(Y.d1, Y.d5, 1), Y.d1)
    print("excl", ex, "n", len(Y), "partial r %.2f p %.3f, spearman %.2f" % (*pearsonr(e0, e5), spearmanr(e0, e5)[0]))
