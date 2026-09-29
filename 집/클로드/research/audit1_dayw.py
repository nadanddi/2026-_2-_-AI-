# -*- coding: utf-8 -*-
"""Audit 1 (methodology): recompute the day-weight screen with saved OOFs and
random-day controls.

Variants (temperature, round-3 blend 0.65/0.25/0.10, member code of screen_v6):
  plain, w02 (167 flagged rows 0.2), d02 (w02 + Q4 days 0.2),
  rK (w02 + a random set of days at 0.2, same per-farm count as Q4), K=0..4
Validators: EXT10 (single fold), A (5), B (5).
Saves local/audit1_dayw_oof.npz.  Analysis in audit1_dayw_eval.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import time

import env  # noqa: F401
import numpy as np
import pandas as pd

import cold_v5
cold_v5.DET = dict(deterministic=True, force_col_wise=True, n_jobs=2, verbose=-1)
from common import TARGET_FARMS
from harness import load, views, folds
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from cold_v5 import THRESH
from screen_v6 import temp_members, collect
from cleanw_v6 import weights


def main():
    t0 = time.time()
    panel, lab0, _ = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    lab = (lab0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
               .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left")).copy()
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    ct = f93 + F4.names(sg) + F4.names(fp)
    phc = F4.names(ph)

    ds = pd.read_csv(env.LOCAL + "/anal_q3f_dayscore.csv")
    q4 = ds[ds.q == "Q4 train-like"]
    key = list(zip(lab.farm.values, lab.day.values))
    q4set = set(map(tuple, q4[["farm", "day"]].values))
    mq4 = np.array([k in q4set for k in key])
    Wf = weights(lab, 0, 0.2)
    V = {"plain": None, "w02": Wf, "d02": Wf * np.where(mq4, 0.2, 1.0)}
    rng = np.random.RandomState(12345)
    rsets = {}
    for k in range(5):
        pick = set()
        for f in TARGET_FARMS:
            pool = ds[ds.farm == f][["farm", "day"]].values
            n = int((q4.farm == f).sum())
            idx = rng.choice(len(pool), n, replace=False)
            pick |= set(map(tuple, pool[idx]))
        m = np.array([kk in pick for kk in key])
        V["r%d" % k] = Wf * np.where(m, 0.2, 1.0)
        rsets["r%d" % k] = m

    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    cd = dmin[dmin < THRESH]
    ext = [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]
    tsets = [("EXT10", ext), ("gA", folds("A")), ("gB", folds("B"))]
    out = {"row_id": lab.row_id.values, "mq4": mq4}
    for k, m in rsets.items():
        out["mask_" + k] = m
    for vn, W in V.items():
        for s, fds in tsets:
            M = collect(lab, fds, lambda tr, va, m: temp_members(tr, va, ct, phc, None if W is None else W[m]))
            for mem in ("res", "ridge", "nys"):
                out["%s|%s|%s" % (vn, s, mem)] = M[mem]
        print("  %s done  %.0fs" % (vn, time.time() - t0), flush=True)
        np.savez(env.LOCAL + "/audit1_dayw_oof.npz", **out)
    print("saved")


if __name__ == "__main__":
    main()
