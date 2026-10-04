# -*- coding: utf-8 -*-
"""FZ0 (forensics, fixed before running; 2026-10-05 집 클로드; data-generation-forensics skill).
Problem: high-EC corrections (HG1 / AF0 / AF0b) fail because of a few FALSE gated days
(model and anchors say high, label ~0.6: F13 231 / 233, F47 216, ...; 6.316, 6.318, 6.319).
Step 1 ORACLE: on DIAG10 R3S rows (hk0_rows_v1.csv), HGB-type correction (pm_h >= .9, both
anchors >= 1, p + .5 (mean(a1, a2) - pm_h)) applied to TRUE high days only vs SG2 -> the
prize if false days could be recognised.
Step 2 hypotheses (fingerprints predicted before looking at results):
  H1 operation regime (e.g. CO2 dosing / shading programme differs on false days; eyeball
     of F13 231/233, F47 216 showed act_co2 mean 29-36 vs 1-2 on F13 217 / F47 229 / 231):
     must see: a day-level input with AUC(false vs true) >= .85 and same direction in both
     farms; must NOT see: true highs spread over the same range.
  H2 greenhouse-wide reset (both 동 moderate on the same date): must see the pair sibling of
     a false day also < 1 while true high days' siblings are low (< .6) as usual 동A;
     must NOT see: false days' siblings high.
  H3 information absent: no day-level input (non-causal, full day) separates false from
     true gated days: best univariate AUC < .80 or family-wise p >= .05, and the
     cross-validated multivariate upper bound (leave-one-day-out L2 logistic on
     standardized day features) AUC < .75.
Gated days = broad gate of HK1 (any row with pm_h >= .8 and a1 >= 1.0).
Day features = hc0_day_features.csv (all inputs: mean / night / day / zero-share), full
day (non-causal UPPER BOUND; a causal version only if this bound is useful)."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

O = pd.read_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv"))
O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
d = O.a1 - O.pm
sg = np.where(O.a1.notna() & (d.abs() <= .30), O.p + .5 * d, O.p)
both = O.a1.notna() & O.a2.notna() & (O.a1 >= 1) & (O.a2 >= 1) & (O.pm >= .9)
hg = np.where(both, O.p + .5 * ((O.a1 + O.a2) / 2 - O.pm), sg)
orc = np.where(both & (O.dm >= 1), hg, sg)
print("STEP 1 oracle (vs SG2, DIAG10 R3S rows):")
for nm, m in (("all", O.dm.notna()), ("pass-2", O.day >= 179), ("high", O.dm >= 1), ("normal", O.dm < 1)):
    e0 = r(sg[m] - O.sub_ec[m])
    print("  %-7s SG2 %.4f | HGB %.4f (%+.1f%%) | HGB on true highs only %.4f (%+.1f%%)" % (
        nm, e0, r(hg[m] - O.sub_ec[m]), 100 * (r(hg[m] - O.sub_ec[m]) / e0 - 1), r(orc[m] - O.sub_ec[m]), 100 * (r(orc[m] - O.sub_ec[m]) / e0 - 1)))

g = O.a1.notna() & (O.pm >= .8) & (O.a1 >= 1.0)
G = O[g].groupby(["farm", "day"]).dm.first().reset_index()
G["false"] = (G.dm < 1).astype(int)
H = pd.read_csv(os.path.join(env.LOCAL, "hc0_day_features.csv"))
drop = {"farm", "day", "ec", "role", "dong", "hi", "is_B", "is_second", "late", "recday"}
F = [c for c in H.columns if c not in drop and pd.api.types.is_numeric_dtype(H[c])]
G = G.merge(H[["farm", "day", "role", "dong"] + F], on=["farm", "day"], how="left")
print("\nSTEP 2 gated days %d: true %d, false %d" % (len(G), (G.false == 0).sum(), G.false.sum()))

# H2 sibling check
Y = H.set_index(["farm", "day"]).ec
R = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv")).set_index(["farm", "day"])
def sib(f, d):
    ro = R.role.get((f, d))
    s = d + 1 if ro == "first" else (d - 1 if ro == "second" else None)
    return Y.get((f, s), np.nan) if s is not None else np.nan
G["sib"] = [sib(f, d) for f, d in zip(G.farm, G.day)]
print("H2 pair sibling label: true highs (n with sibling %d) median %.2f; false days (n %d) median %.2f" % (
    G[(G.false == 0)].sib.notna().sum(), G[G.false == 0].sib.median(), G[G.false == 1].sib.notna().sum(), G[G.false == 1].sib.median()))
print(G[G.sib.notna()][["farm", "day", "dm", "role", "sib", "false"]].round(2).to_string(index=False))

# H1 / H3 univariate
X = G[F].astype(float); X = X.fillna(X.median())
A = {c: roc_auc_score(G.false, X[c]) for c in F if X[c].nunique() > 1}
rk = sorted(A, key=lambda c: -abs(A[c] - .5))[:10]
print("\nH1/H3 univariate AUC (false vs true), top 10 (farm-wise AUC):")
for c in rk:
    fa = []
    for f in ("F13", "F47"):
        m = (G.farm == f).values
        fa.append(roc_auc_score(G.false[m], X[c][m]) if G.false[m].nunique() == 2 else np.nan)
    print("  %-20s AUC %.2f  F13 %.2f  F47 %.2f  median true %.2f false %.2f" % (c, A[c], fa[0], fa[1], X[c][G.false == 0].median(), X[c][G.false == 1].median()))
obs = max(abs(v - .5) for v in A.values())
rng = np.random.default_rng(20261005); cnt = 0
for _ in range(5000):
    y = G.false.values.copy()
    for f in ("F13", "F47"):
        m = (G.farm == f).values; y[m] = rng.permutation(y[m])
    cnt += max(abs(roc_auc_score(y, X[c]) - .5) for c in A) >= obs
print("family-wise permutation p (max |AUC-.5| over %d features): %.4f" % (len(A), (cnt + 1) / 5001))

# H3 multivariate upper bound (leave-one-day-out)
Z = StandardScaler().fit_transform(X[list(A)])
pr = np.zeros(len(G))
for i in range(len(G)):
    m = np.arange(len(G)) != i
    sc = StandardScaler().fit(X[list(A)].values[m])
    clf = LogisticRegression(C=.1, max_iter=5000).fit(sc.transform(X[list(A)].values[m]), G.false.values[m])
    pr[i] = clf.predict_proba(sc.transform(X[list(A)].values[i:i + 1]))[0, 1]
print("multivariate leave-one-day-out L2 logistic AUC (non-causal upper bound): %.2f" % roc_auc_score(G.false, pr))
print("\nFZ0 reading: H1 needs best AUC >= .85, same direction both farms, family-wise p < .05; H3 = none of that and multivariate AUC < .75")
