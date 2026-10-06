# -*- coding: utf-8 -*-
"""ER1 (exploration, 2026-10-07 집 클로드).  Largest-error days of the latest EC configuration
(submission_14 = clip(shrink(0.8 R3_DP1 + 0.2 PFN)) + SG2 on pass-2 rows), on out-of-fold predictions
stored by WT1 (seeds 47 / 1414 / 6464 averaged; DIAG10 all days, P2LOO pass-2 days).
Per day: label mean, prediction mean, residual (label - prediction), day SSE share, sealed share of hours,
pass, test-matched (TM1 set) membership; top days by SSE; error concentration; class counts."""
import env  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(HERE, "ec3_SG2_reference_knn_level_v1.py"))
sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
S = (47, 1414, 6464)
O = pd.read_csv(os.path.join(env.LOCAL, "ec3_WT1_all.csv"))
r3 = np.mean([.6 * O["et_%d" % s] + .3 * O["lgb_%d" % s] + .1 * O["mlp_%d" % s] for s in S], axis=0)
O["cur"] = np.clip(.8 * r3 + .2 * O.pfn, O.lo, O.hi)
raw, full, lab, lock, signatures, fds = sg2.p3.prepare()
R, WV, hrs, SIG = sg2.prepare_structure()
LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
O["s14"] = O.cur
for (v, k), G in O.groupby(["validator", "validation_fold"]):
    if not (G.day >= 179).any():
        continue
    vd = set(zip(G.farm, G.day)); ref = {x for x in labset if x not in vd}
    cal = sg2.ref_calendar(R, WV, ref)
    Gs = G.sort_values(["farm", "day", "hour"])
    out = sg2.correction(Gs, "cur", vd, lockd, ec, R, WV, hrs, SIG, ref, cal)
    O.loc[Gs.index, "s14"] = np.clip(out, Gs.lo, Gs.hi)
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "act_vent"])
O = O.merge(X, on="row_id", how="left")
TM = pd.read_csv(os.path.join(env.LOCAL, "tm1_set_v1.csv")); TMs = set(zip(TM.farm, TM.day))
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
for v in ("DIAG10", "P2LOO"):
    G = O[O.validator == v] if v == "DIAG10" else O[(O.validator == v) & (O.day >= 179)]
    D = G.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("s14", "mean"), sealed=("act_vent", lambda s: (s == 0).mean()),
                                       sse=("s14", lambda s: 0)).reset_index()
    G = G.assign(se=(G.s14 - G.sub_ec) ** 2)
    D["sse"] = G.groupby(["farm", "day"]).se.sum().values
    D["res"] = D.y - D.p; D["pass"] = np.where(D.day >= 179, 2, 1); D["TM"] = [(f, d) in TMs for f, d in zip(D.farm, D.day)]
    D["cls"] = np.where(D.res > .15, "under", np.where(D.res < -.15, "over", "ok")); D["hi"] = D.y >= 1
    tot = D.sse.sum(); D["share%"] = 100 * D.sse / tot
    print("\n=== %s: %d days, row RMSE %.4f" % (v, len(D), r(G.s14 - G.sub_ec)))
    Ds = D.sort_values("sse", ascending=False)
    print("SSE concentration: top 5 days %.0f%%, top 10 %.0f%%, top 20 %.0f%%" % tuple(Ds["share%"].head(k).sum() for k in (5, 10, 20)))
    print("classes:", D.cls.value_counts().to_dict(), "| SSE share: high days %.0f%%, sealed>=.8 days %.0f%%, TM days %.0f%%" % (
        D[D.hi].sse.sum() / tot * 100, D[D.sealed >= .8].sse.sum() / tot * 100, D[D.TM].sse.sum() / tot * 100))
    print(D.groupby("cls").agg(days=("y", "size"), y=("y", "mean"), p=("p", "mean"), sealed=("sealed", "mean"),
                               hi=("hi", "mean"), sse_share=("share%", "sum")).round(2).to_string())
    print("top 15 days:")
    print(Ds.head(15)[["farm", "day", "pass", "y", "p", "res", "cls", "sealed", "TM", "share%"]].round(2).to_string(index=False))
    D.to_csv(os.path.join(env.LOCAL, "er1_%s_days_v1.csv" % v), index=False)
