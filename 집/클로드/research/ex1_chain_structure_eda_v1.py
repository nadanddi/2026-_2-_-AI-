# -*- coding: utf-8 -*-
"""EX1 (exploration, 2026-10-06 집 클로드; data-exploration skill: number + code + threat for each finding).
Uses CH2 label chains (local/ch2_label_links_v1.csv: training records linked by EC midnight continuity +
outdoor / indoor continuity) as a better 'same source' structure than earlier pair roles.
Q1 persistence: day-mean EC of consecutive chain members (same source, next date) vs the other record of
   the same date (other source): |difference| and Spearman.
Q2 source predictability: per farm, chains are split into 2 groups by chain-mean EC (high / low source;
   group = chain mean >= farm median of chain means, weighted by length).  Can the group be predicted
   from the hour-h causal signature (13 SG2 values over hours 0..h)?  Logistic regression, leave-one-
   CHAIN-out CV; accuracy at h = 0, 5, 11, 23; compare with the old pair-role 동 labels' agreement.
Q3 residual persistence: DIAG10 R3S day residual (label - prediction) of consecutive chain members:
   Spearman lag-1 within chain vs consecutive record order.
Threats: chains built WITH labels (Q1 is partly by construction near midnight -> report day means and
also hour-12 values); groups by chain mean are a proxy for 동."""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(HERE, "ec3_SG2_reference_knn_level_v1.py"))
sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
R, WV, hrs, SIG = sg2.prepare_structure()
L = pd.read_csv(os.path.join(env.LOCAL, "ch2_label_links_v1.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
dm = Y.groupby(["farm", "day"]).sub_ec.mean(); h12 = Y[Y.hour == 12].set_index(["farm", "day"]).sub_ec
# chain ids
cid = {}
for f in ("F13", "F47"):
    Lf = L[L.farm == f]; succ = dict(zip(Lf.a, Lf.b)); pred = set(Lf.b)
    days = sorted(d for ff, d in dm.index if ff == f)
    k = 0
    for s in [d for d in days if d not in pred]:
        c = s
        while True:
            cid[(f, c)] = "%s_%d" % (f, k)
            if c not in succ:
                break
            c = succ[c]
        k += 1
C = pd.DataFrame([(f, d, c) for (f, d), c in cid.items()], columns=["farm", "day", "chain"])
C["ec"] = [dm[(f, d)] for f, d in zip(C.farm, C.day)]
# Q1
pairs = []
for f, a, b in zip(L.farm, L.a, L.b):
    pairs.append(("same", dm[(f, a)], dm[(f, b)], h12.get((f, a)), h12.get((f, b))))
role = R.set_index(["farm", "day"]).role
for (f, d) in dm.index:
    if role.get((f, d)) == "first" and (f, d + 1) in dm.index:
        pairs.append(("other_same_date", dm[(f, d)], dm[(f, d + 1)], h12.get((f, d)), h12.get((f, d + 1))))
Pq = pd.DataFrame(pairs, columns=["kind", "a", "b", "a12", "b12"])
print("Q1 day-mean EC, consecutive same-chain (next date) vs the two 동 of one date:")
for k, g in Pq.groupby("kind"):
    print("  %-16s n %3d  |diff| median %.3f  mean %.3f  Spearman %.2f  | hour-12 |diff| median %.3f" % (
        k, len(g), (g.a - g.b).abs().median(), (g.a - g.b).abs().mean(), spearmanr(g.a, g.b).correlation, (g.a12 - g.b12).abs().median()))
# Q2
cm = C.groupby(["farm", "chain"]).agg(mean=("ec", "mean"), n=("ec", "size")).reset_index()
grp = {}
for f in ("F13", "F47"):
    g = cm[cm.farm == f].sort_values("mean"); cum = g.n.cumsum() / g.n.sum()
    for ch, q in zip(g.chain, cum):
        grp[ch] = int(q > .5)
C["grp"] = C.chain.map(grp)
dg = R.set_index(["farm", "day"]).dong
C["old"] = [dg.get((f, d)) for f, d in zip(C.farm, C.day)]
print("\nQ2 chain groups: high-group mean EC %.2f vs low %.2f; agreement of high-group with old 동B label %.2f" % (
    C[C.grp == 1].ec.mean(), C[C.grp == 0].ec.mean(), np.mean((C.grp == 1) == (C.old == "B"))))
for h in (0, 5, 11, 23):
    S = SIG[h].astype(float)
    acc = []
    for f in ("F13", "F47"):
        Cf = C[C.farm == f].reset_index(drop=True)
        Xf = S.loc[[(f, d) for d in Cf.day]].values; Xf = Xf[:, ~np.isnan(Xf).all(axis=0)]; Xf = np.where(np.isnan(Xf), np.nanmean(Xf, axis=0), Xf)
        pr = np.zeros(len(Cf))
        for ch in Cf.chain.unique():
            te = (Cf.chain == ch).values
            if Cf.grp[~te].nunique() < 2:
                continue
            sc = StandardScaler().fit(Xf[~te]); m = LogisticRegression(C=1, max_iter=3000).fit(sc.transform(Xf[~te]), Cf.grp[~te])
            pr[te] = m.predict_proba(sc.transform(Xf[te]))[:, 1]
        acc.append(((pr >= .5) == (Cf.grp == 1)).mean())
    print("  hour %2d leave-one-chain-out accuracy F13 %.2f F47 %.2f" % (h, acc[0], acc[1]))
# Q3
O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"]
O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
res = (O.groupby(["farm", "day"]).sub_ec.mean() - O.groupby(["farm", "day"]).p.mean())
a = [(res.get((f, x)), res.get((f, y))) for f, x, y in zip(L.farm, L.a, L.b)]
a = np.array([z for z in a if None not in z and np.isfinite(z).all()])
b = [(res.get((f, d)), res.get((f, d + 1))) for (f, d) in res.index if (f, d + 1) in res.index]
b = np.array(b)
print("\nQ3 day residual lag-1 Spearman: within chain %.2f (n %d) vs record order %.2f (n %d)" % (
    spearmanr(a[:, 0], a[:, 1]).correlation, len(a), spearmanr(b[:, 0], b[:, 1]).correlation, len(b)))
C.to_csv(os.path.join(env.LOCAL, "ex1_chain_days_v1.csv"), index=False)
