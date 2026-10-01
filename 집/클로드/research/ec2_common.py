# -*- coding: utf-8 -*-
"""Shared helpers for the EC stage-2 experiments (P1, ST1) on Codex phase-3 OOF.
Rules: ec2_P1_ST1_사전고정_2026-10-02.md."""
import os

import numpy as np
import pandas as pd

import env

OOF = os.path.join(env.ROOT, u"집", u"코덱스", "local", "ec_restart_phase3_20261001_v1", "oof_predictions.csv")
SEEDS = (7, 101, 2024)
VALS = ("DIAG10", "A", "B", "EXT10", "EXT12")
GRP = ["validator", "validation_fold", "farm", "day"]


def load():
    o = pd.read_csv(OOF, encoding="utf-8-sig")
    tx = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "act_circfan", "act_vent"])
    o = o.merge(tx, on="row_id", how="left")
    o = o.sort_values(GRP + ["hour"]).reset_index(drop=True)
    d = o.groupby(["farm", "day"])
    o["sealed"] = (d.act_circfan.transform("mean") < 10) & (d.act_vent.transform(lambda s: (s == 0).mean()) > 0.85)
    return o


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def judge(o, col_of_seed, k, name, rng_seed=20261002):
    """col_of_seed(s) -> candidate column; baseline v2_s.  Prints the 15 cells,
    DIAG10 bootstrap per seed and the decision."""
    rng = np.random.default_rng(rng_seed)
    allbetter, noext = True, True
    print("\n[%s] pooled RMSE v2 -> candidate (relative %%)" % name)
    for v in VALS:
        g = o[o.validator == v]
        cells = []
        for s in SEEDS:
            a, b = rmse(g["v2_%d" % s] - g.sub_ec), rmse(g[col_of_seed(s)] - g.sub_ec)
            allbetter &= b < a
            if v in ("DIAG10", "A", "B"):
                noext &= b < a
            cells.append("s%d %.4f->%.4f (%+.2f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = o[o.validator == "DIAG10"].copy()
    D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D[col_of_seed(s)] - D.sub_ec) ** 2 - (D["v2_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"])
        sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    thr = 0.025 / k
    ok = allbetter and all(p < thr for p in ps)
    ok3 = noext and all(p < thr for p in ps)
    print("  DIAG10 P(worse) by seed %s (threshold %.4f)" % ([round(p, 4) for p in ps], thr))
    for nm, m in (("DIAG10 late>=179", D.day >= 179), ("DIAG10 sealed", D.sealed), ("DIAG10 early", D.day < 179)):
        g = D[m]
        print("  %-17s v2 %.4f -> %.4f" % (nm, np.mean([rmse(g["v2_%d" % s] - g.sub_ec) for s in SEEDS]),
                                          np.mean([rmse(g[col_of_seed(s)] - g.sub_ec) for s in SEEDS])))
    print("  -> rule (15 cells + P): %s | descriptive 3-validator view (no EXT): %s"
          % ("PASS" if ok else "FAIL", "pass" if ok3 else "fail"))
    return ok
