# -*- coding: utf-8 -*-
"""SA-P2 (2026-10-12 집 클로드; decisive test proposed by result critic, ec0h/critic_result_e1sa.md).  Fixed before running:
  data    : DIAG10 EC OOF (tag CUR, column sg) seeds 47 / 1414 / 6464; labels sub_ec.
  targets : records t with number >= 179 having a v7m choice p: same ID (xf False), smaller number, p >= .8, labelled or not
            (p's LABEL IS IGNORED; only p's OOF prediction is used — evaluation-like).
  c_h     : 0 h residual carry on pass-1 records that are NOT targets/prevs of the EV-SA set (DIAG10, seed mean).
  w       : ONE constant fitted on EV pass-1 SA targets (e1_chain_anchor_sameid_v1 arm SA definition), LS of (y_t0 - EA2_t0) on
            (EA2_p23 - EA2_t0), pooled over EV seeds, clipped [0, 1]; never refitted on pass 2.
  rule    : pred_h = q_t,h + c_h * w * (q_p,23 - q_t,0)
  PASS iff all 3 seeds improve all-hour RMSE on targets AND record bootstrap P(worse) < .025 AND max single-record share of net
           gain <= 50%.  If FAIL: record as 'pass-2 transfer not demonstrated' (n ~ 16, low power).
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
H = os.path.join(env.ROOT, u"집", u"클로드", "research")
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]
Q = {sd: G.groupby(["rec", "hour"]).sg.mean().unstack() for sd, G in O.groupby("seed")}
Y = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
# --- w from EV pass-1 SA targets (definition of e1_chain_anchor_sameid_v1, arm SA) ---
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
F = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); V = F[F.role == "val"].set_index("rec")
anch = R.groupby("rec").anchored.first(); sa = []
for t in V.index:
    if t not in L.index or bool(anch.get(t, True)) or int(t[4:7]) >= 179: continue
    pr = L.loc[t]
    if pr.prev_is_labelled or pr.xf or pr.p < .8: continue
    p = pr.prev
    if p in V.index and V.at[p, "fold"] == V.at[t, "fold"] and p[:3] == t[:3] and int(p[4:7]) < int(t[4:7]) and not bool(anch.get(p, False)): sa.append((t, p))
xs, ys = [], []
for sd, G in R.groupby("seed"):
    P = G.pivot_table(index="rec", columns="hour", values="EA2"); Yv = G.groupby(["rec", "hour"]).y.first().unstack()
    for t, p in sa: xs.append(P.at[p, 23] - P.at[t, 0]); ys.append(Yv.at[t, 0] - P.at[t, 0])
xs, ys = np.array(xs), np.array(ys); w = float(np.clip((xs * ys).sum() / (xs * xs).sum(), 0, 1))
print("EV pass-1 SA pairs %d -> fixed w %.3f" % (len(sa), w))
# --- c_h on pass-1 non-target records ---
excl = {x for pair in sa for x in pair}
p1 = [r for r in Y.index if int(r[4:7]) < 179 and r not in excl]
Qm = sum(Q.values()) / len(Q); e = (Y.loc[p1] - Qm.loc[p1]).values
c = np.array([(e[:, 0] * e[:, h]).sum() / (e[:, 0] ** 2).sum() for h in range(24)])
# --- pass-2 targets ---
T = [(t, L.at[t, "prev"]) for t in L.index if int(t[4:7]) >= 179 and not bool(L.at[t, "xf"]) and L.at[t, "p"] >= .8
     and L.at[t, "prev"][:3] == t[:3] and int(L.at[t, "prev"][4:7]) < int(t[4:7]) and t in Y.index and L.at[t, "prev"] in Y.index]
print("pass-2 targets %d: %s" % (len(T), ", ".join("%s<-%s" % x for x in T)))
cells, per = [], []
for sd, Qs in Q.items():
    new = Qs.copy()
    for t, p in T: new.loc[t] = Qs.loc[t].values + c * w * (Qs.at[p, 23] - Qs.at[t, 0])
    recs = [t for t, _ in T]
    cells.append(rm((new.loc[recs] - Y.loc[recs]).values) / rm((Qs.loc[recs] - Y.loc[recs]).values) - 1)
    per.append(((new.loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((Qs.loc[recs] - Y.loc[recs]) ** 2).sum(1))
P = pd.concat(per, axis=1).mean(1); rng = np.random.default_rng(20261219)
idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P.values[idx].sum(1) >= 0).mean())
share = float(P.min() / P.sum()) if P.sum() < 0 else np.nan
print("cells %s | improved %d/%d | P(worse) %.4f | top-record share %s" % (" ".join("%+.2f%%" % (100 * x) for x in cells), int((P < 0).sum()), len(P), pw, "%.0f%%" % (100 * share) if share == share else "n/a"))
print(P.sort_values().round(3).to_string())
ok = all(x < 0 for x in cells) and pw < .025 and share == share and share <= .5
print("VERDICT SA-P2:", "PASS" if ok else "FAIL (pass-2 transfer not demonstrated)")
