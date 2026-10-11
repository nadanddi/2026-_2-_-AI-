# -*- coding: utf-8 -*-
"""agentY y2: global per-cell mean residual and hour-block offset (pass-1 estimate, pass-2 judge) for temp DIAG10 and EC DIAG10/EL1/P2LOO."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); LOC = os.path.join(HERE, "local")


def report(lab, D, y, preds):
    p = np.mean(preds, axis=0); r = y - p; p2 = D.day.values >= 179; fm = D.farm.values; rec = (D.farm + D.day.astype(str)).values
    line = []
    for f in ("F13", "F47"):
        for ps, m in (("p1", ~p2), ("p2", p2)):
            mm = m & (fm == f)
            if mm.sum() == 0:
                continue
            g = pd.Series(r[mm]).groupby(rec[mm]).mean(); line.append("%s%s %+.3f (z %+.1f, rec %d)" % (f, ps, r[mm].mean(), g.mean() / (g.std() / np.sqrt(len(g))), len(g)))
    print("%-14s mean resid: %s" % (lab, " | ".join(line)))
    blk = D.hour.values // 4
    for mode in ("global", "hourblock"):
        adj = p.copy(); out = []
        if p2.sum() == 0 or (~p2).sum() == 0:
            continue
        for b in range(6):
            m = (blk == b) if mode == "hourblock" else np.ones(len(p), bool)
            d = r[m & ~p2].mean(); adj[m & p2] = p[m & p2] + d
            if mode == "global":
                break
        for f in ("F13", "F47"):
            mm = p2 & (fm == f); out.append("%s %+.2f%%" % (f, 100 * (np.sqrt(np.mean((y - adj)[mm] ** 2)) / np.sqrt(np.mean(r[mm] ** 2)) - 1)))
        print("    add pass-1 %-9s offset -> pass-2 RMSE %s" % (mode, ", ".join(out)))


M = pd.read_csv(os.path.join(LOC, "temp_DIAG10.csv"))
report("TEMP DIAG10", M, M.sub_temp.values, [M["p%d" % i].values for i in range(4)])
for L in ("G", "J", "L", "F"):
    M = pd.read_csv(os.path.join(LOC, "temp_%s.csv" % L)); report("TEMP " + L, M, M.sub_temp.values, [M["p%d" % i].values for i in range(4)])
E = pd.read_csv(r"C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv"); E = E[E.tag == "CUR"]
for V in ("DIAG10", "EL1", "P2LOO"):
    e = E[E.validator == V]; W = e.pivot_table(index="row_id", columns="seed", values="sg")
    D = e.groupby("row_id")[["farm", "day", "hour", "sub_ec"]].first().loc[W.index]
    report("EC " + V, D, D.sub_ec.values, [W[c].values for c in W.columns])
