# -*- coding: utf-8 -*-
"""ST6 (structure audit; 2026-10-03 집 클로드).  Assign a 동 (source) to every
record and re-measure EC state persistence with it.
Step 1: label pair records first = A, second = B (C6.205: order ~ fixed 동).
Step 2: classifier A/B from INPUTS only (day means and 0-6h / 12-17h means of 10
indoor+actuator channels, standardized per farm), logistic regression (C=1),
leave-one-pair-out CV AUC on pair records.  Also the within-pair test: does the
classifier rank the second above the first (share of pairs).
Step 3: predict singletons (P(B)).  Report P(B) along the record sequence in pass 2
(alternation?), and parity agreement.
Step 4: EC label day mean persistence: Spearman of each labelled record's EC with
the previous labelled record of the SAME assigned 동 (<= 4 records back) vs of the
OTHER 동, per farm x pass; compare with the parity rule.
Inputs from train_X + test_X for structure only; labels only for step 4 (train)."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

C = ["in_temp", "in_hum", "in_co2", "act_heating", "act_vent", "act_thermal", "act_shade", "act_circfan", "act_fog", "act_co2"]
A = pd.concat([pd.read_csv(os.path.join(env.DATA, f), usecols=["row_id"] + C) for f in ("train_X.csv", "test_X.csv")])
A = A[A.row_id.str[:3].isin(["F13", "F47"])].copy()
A["farm"], A["day"], A["hour"] = A.row_id.str[:3], A.row_id.str[4:7].astype(int), A.row_id.str[8:10].astype(int)
roles = pd.read_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
EC = Y.groupby(["farm", "day"]).sub_ec.mean()
feats = []
for nm, sel in (("d", slice(0, 24)), ("n", slice(0, 7)), ("a", slice(12, 18))):
    F = A[(A.hour >= sel.start) & (A.hour < sel.stop)].groupby(["farm", "day"])[C].mean()
    F.columns = [nm + "_" + c for c in F.columns]
    feats.append(F)
F = pd.concat(feats, axis=1).reset_index().merge(roles, on=["farm", "day"])
out = []
for f in ("F13", "F47"):
    G = F[F.farm == f].sort_values("day").copy()
    X = G.drop(columns=["farm", "day", "role"])
    X = ((X - X.mean()) / X.std()).fillna(0).values
    pr = G.role.isin(["first", "second"]).values
    yb = (G.role == "second").astype(int).values
    pid = np.where(G.role == "second", G.day - 1, G.day).astype(int)
    oof = np.full(len(G), np.nan)
    for p in np.unique(pid[pr]):
        te = pr & (pid == p); tr = pr & (pid != p)
        oof[te] = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], yb[tr]).predict_proba(X[te])[:, 1]
    auc = roc_auc_score(yb[pr], oof[pr])
    P = pd.DataFrame({"day": G.day.values[pr], "p": oof[pr], "role": G.role.values[pr]})
    s2 = P[P.role == "second"].set_index(P[P.role == "second"].day - 1).p
    s1 = P[P.role == "first"].set_index("day").p
    within = (s2.reindex(s1.index) > s1).mean()
    full = LogisticRegression(C=1.0, max_iter=2000).fit(X[pr], yb[pr])
    pb = full.predict_proba(X)[:, 1]
    G["pB"] = np.where(pr, yb, pb)
    G["dong"] = np.where(G.pB >= 0.5, "B", "A")
    print("\n%s: pair records %d, LOPO AUC %.3f, second ranked above first in %.0f%% of pairs" % (f, pr.sum(), auc, 100 * within))
    L = G[G.day >= 179]
    print("  pass-2 sequence (day:role:P(B)):")
    s = ["%d%s%.1f" % (d, {"first": "[", "second": "]", "single": "."}[r], p) for d, r, p in zip(L.day, L.role, L.pB)]
    for k in range(0, len(s), 10):
        print("    " + "  ".join(s[k:k + 10]))
    sg = G[G.role == "single"]
    for ps, H in sg.groupby(sg.day >= 179):
        print("  singletons %s: n %d, mean P(B) %.2f, agreement dong==parity(odd=B) %.2f" % (
            "late" if ps else "early", len(H), H.pB.mean(), ((H.dong == "B") == (H.day % 2 == 1)).mean()))
    out.append(G[["farm", "day", "role", "pB", "dong"]])
    # step 4: persistence
    G = G.set_index("day")
    lab = [d for d in G.index if (f, d) in EC.index]
    for ps in (0, 1):
        same_x, same_y, oth_x, oth_y, par_x, par_y = [], [], [], [], [], []
        for d in lab:
            if (d >= 179) != ps:
                continue
            prev = [q for q in lab if d - 4 <= q < d and (q >= 179) == ps]
            sm = [q for q in prev if G.loc[q, "dong"] == G.loc[d, "dong"]]
            ot = [q for q in prev if G.loc[q, "dong"] != G.loc[d, "dong"]]
            pp = [q for q in prev if (d - q) % 2 == 0]
            if sm: same_x.append(EC[(f, max(sm))]); same_y.append(EC[(f, d)])
            if ot: oth_x.append(EC[(f, max(ot))]); oth_y.append(EC[(f, d)])
            if pp: par_x.append(EC[(f, max(pp))]); par_y.append(EC[(f, d)])
        rr = lambda x, y: spearmanr(x, y).correlation if len(x) > 5 else np.nan
        print("  EC persistence %s: same-dong rho %.2f (n %d) | other-dong %.2f (n %d) | same-parity %.2f (n %d)" % (
            "late" if ps else "early", rr(same_x, same_y), len(same_x), rr(oth_x, oth_y), len(oth_x), rr(par_x, par_y), len(par_x)))
pd.concat(out).to_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv"), index=False)
