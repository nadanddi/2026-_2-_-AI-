# -*- coding: utf-8 -*-
"""EC stage-2 ST1: non-negative restack of existing members (k = 1).  Weights
for each validation fold are fitted by NNLS on DIAG10 OOF rows outside that
fold's validation days +-1 day (target-fold labels never used).  Rules fixed
in ec2_P1_ST1_사전고정_2026-10-02.md before running.  2026-10-02 집 클로드.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_ST1_stack_v1.py"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ec2_common import SEEDS, judge, load

MEM = ["r3", "S", "full_et", "raw_et", "ridge", "farm_mean"]


def main():
    o = load()
    lo, hi = o.sub_ec.min(), o.sub_ec.max()
    for s in SEEDS:
        o["S_%d" % s] = (o["v2_%d" % s] - 0.8 * o["r3_%d" % s]) / 0.2
    D = o[o.validator == "DIAG10"]
    wlog = []
    for s in SEEDS:
        cols = ["r3_%d" % s, "S_%d" % s, "full_et_%d" % s, "raw_et_%d" % s, "ridge", "farm_mean"]
        out = np.full(len(o), np.nan)
        for (v, k), g in o.groupby(["validator", "validation_fold"]):
            vd = {(f, d + j) for f, d in g[["farm", "day"]].drop_duplicates().itertuples(index=False) for j in (-1, 0, 1)}
            tr = D[[(f, d) not in vd for f, d in zip(D.farm, D.day)]]
            w, _ = nnls(tr[cols].values, tr.sub_ec.values)
            out[g.index] = np.clip(g[cols].values @ w, lo, hi)
            wlog.append(dict(seed=s, validator=v, fold=k, **dict(zip(MEM, np.round(w, 3)))))
        o["st_%d" % s] = out
    W = pd.DataFrame(wlog)
    print("NNLS weights (mean / min / max over folds, all seeds):")
    print(W[MEM].agg(["mean", "min", "max"]).round(3).to_string())
    ok = judge(o, lambda s: "st_%d" % s, 1, "ST1 restack")
    print("\nST1 decision:", ok)


if __name__ == "__main__":
    main()
