# -*- coding: utf-8 -*-
"""Q2/Q4: build source chains from links; describe; EC level vs day / cal / chain (analysis)."""
import env  # noqa
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
S = pd.read_csv(env.LOCAL + "/deep_cal_10_daysum.csv")
L = pd.read_csv(env.LOCAL + "/deep_cal_10_links.csv")
S["node"] = np.arange(len(S))
nid = {(f, d): i for i, (f, d) in enumerate(zip(S.farm, S.day))}
r = [nid[(f, x)] for f, x in zip(L.farm, L.d_from)]; c = [nid[(f, y)] for f, y in zip(L.farm, L.d_to)]
G = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(S), len(S)))
nc, lab = connected_components(G, directed=False)
S["chain"] = lab
cs = S.groupby("chain").agg(farm=("farm", "first"), n=("day", "size"), cal0=("cal", "min"), cal1=("cal", "max"),
                            ntest=("is_test", "sum"), dmin=("day", "min"), dmax=("day", "max"))
print("chains:", nc)
print(cs.sort_values(["farm", "n"], ascending=[True, False]).head(30).to_string())
# regularity inside long chains: parity of day index in first pass
S.to_csv(env.LOCAL + "/deep_cal_11_days.csv", index=False)
for f in ["F13", "F47"]:
    big = cs[(cs.farm == f)].sort_values("n", ascending=False).index[:2]
    for ch in big:
        s = S[S.chain == ch].sort_values("cal")
        fp = s[s.day <= 178]
        print(f, "chain", ch, "n", len(s), "first-pass parity share even: %.2f" % (fp.day % 2 == 0).mean(),
              "test days:", s[s.is_test].day.tolist())
# EC level analysis (labelled days)
E = S.dropna(subset=["ec_m"]).copy()
from sklearn.linear_model import LinearRegression
def r2(cols, df):
    X = pd.get_dummies(df[cols].astype(str)) if any(df[c].dtype == object for c in cols) else df[cols]
    return LinearRegression().fit(X, df.ec_m).score(X, df.ec_m)
import itertools
for f in ["F13", "F47"]:
    e = E[E.farm == f].copy()
    e["second"] = (e.day > 178).astype(int)
    print(f, "corr(ec, day) %.2f | corr(ec, cal) %.2f | corr within first pass: day %.2f cal %.2f"
          % (e.ec_m.corr(e.day), e.ec_m.corr(e.cal), e[e.second == 0].ec_m.corr(e[e.second == 0].day), e[e.second == 0].ec_m.corr(e[e.second == 0].cal)))
    # second-pass EC vs first-pass EC at same cal
    fp = e[e.second == 0].groupby("cal").ec_m.mean()
    sp = e[e.second == 1].copy(); sp["fp_same_cal"] = sp.cal.map(fp)
    print("   second pass labelled days %d: EC mean %.3f ; first-pass same-cal mean %.3f ; corr %.2f ; first-pass overall %.3f"
          % (len(sp), sp.ec_m.mean(), sp.fp_same_cal.mean(), sp.ec_m.corr(sp.fp_same_cal), e[e.second == 0].ec_m.mean()))
    # polynomial fits
    for nm, x in [("day", e.day), ("cal", e.cal)]:
        for deg in (1, 3):
            p = np.polyfit(x, e.ec_m, deg); res = e.ec_m - np.polyval(p, x)
            print("   R2 ec~poly%d(%s): %.3f" % (deg, nm, 1 - res.var() / e.ec_m.var()))
