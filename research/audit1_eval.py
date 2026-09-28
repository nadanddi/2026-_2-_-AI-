# -*- coding: utf-8 -*-
"""Audit 1 (methodology): analysis of audit1_calib_oof.npz and audit1_dayw_oof.npz.

Part A  validator "calibration": ratios R2/R1, R3/R2, R4/R3 on every candidate
        validator and scoring rule vs the real 0.895 / 0.844 / 1.013, with
        day-block bootstrap intervals for EXT10; absolute "test-like" levels
        of R1..R4 vs the real 0.7450 / 0.6666 / 0.5624 / 0.5697.
Part B  day weights: d02 vs w02 vs plain vs random-day controls under several
        scoring rules, including Q4 sets re-derived with other seeds/folds and
        importance weighting instead of hard exclusion.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import env  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb

import common
from harness import load
from cleanw_v6 import weights
from anal_q3d_adversarial import shape_features
from anal_q3e_matched import SHAPE_ONLY

REAL = {"R2/R1": 0.6666 / 0.7450, "R3/R2": 0.5624 / 0.6666, "R4/R3": 0.5697 / 0.5624}
REAL_ABS = {"R1": 0.7450, "R2": 0.6666, "R3": 0.5624, "R4": 0.5697}


def rmse_w(p, y, w=None):
    g = ~np.isnan(p)
    if w is None:
        return float(np.sqrt(np.mean((p[g] - y[g]) ** 2)))
    return float(np.sqrt(np.sum(w[g] * (p[g] - y[g]) ** 2) / np.sum(w[g])))


def cluster_boot(y, P, unit, fn, n=3000, seed=0):
    """Bootstrap a statistic fn(dict name->rmse) over clusters."""
    u, inv = np.unique(unit, return_inverse=True)
    k = len(u)
    S = {nm: np.bincount(inv, (y - p) ** 2) for nm, p in P.items()}
    nn = np.bincount(inv).astype(float)
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        c = np.bincount(rng.randint(0, k, k), minlength=k)
        r = {nm: np.sqrt(c @ s / (c @ nn)) for nm, s in S.items()}
        out.append(fn(r))
    return np.array(out)


def q4_reps(n_rep=6):
    tX, ty, sX = common.load_raw()
    a = pd.concat([tX.assign(split="train"), sX.assign(split="test")], ignore_index=True)
    a = a.query("farm in ['F13','F47']").reset_index(drop=True)
    F = shape_features(a)
    a = a.loc[F.index].reset_index(drop=True)
    F = F.reset_index(drop=True)[SHAPE_ONLY]
    yy = (a.split == "train").astype(int).values
    groups = (a.farm + "_" + a.day.astype(str)).values
    bins = np.floor(a.in_temp.clip(-2, 32)).values
    te_h = pd.Series(bins[yy == 0]).value_counts(normalize=True)
    tr_h = pd.Series(bins[yy == 1]).value_counts(normalize=True)
    sw = np.where(yy == 1, pd.Series(bins).map(te_h / tr_h).fillna(0).values, 1.0)
    sw = np.where(yy == 1, sw * (yy == 0).sum() / sw[yy == 1].sum(), sw)
    ug = np.unique(groups)
    reps = []
    for rep in range(1, n_rep):
        rs = np.random.RandomState(rep)
        fa = dict(zip(ug, rs.randint(0, 5, len(ug))))
        fid = np.array([fa[g] for g in groups])
        p = np.zeros(len(a))
        for k in range(5):
            tr, te = np.where(fid != k)[0], np.where(fid == k)[0]
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                                   n_jobs=2, random_state=rep)
            p[te] = m.fit(F.iloc[tr], yy[tr], sample_weight=sw[tr]).predict_proba(F.iloc[te])[:, 1]
        a["p"] = p
        reps.append(a[a.split == "train"].groupby(["farm", "day"]).p.mean())
    return reps


def main():
    panel, lab, _ = load()
    y = lab.sub_temp.values
    clean = weights(lab, 3, 0.0) >= 1
    ds = pd.read_csv(env.LOCAL + "/anal_q3f_dayscore.csv")
    key = pd.MultiIndex.from_arrays([lab.farm.values, lab.day.values])
    q4s = set(map(tuple, ds[ds.q == "Q4 train-like"][["farm", "day"]].values))
    mq4 = np.array([k in q4s for k in key])
    testlike = clean & ~mq4
    unit = (lab.farm + "_" + lab.day.astype(str)).values
    dmin = lab.groupby(["farm", "day"]).in_temp.transform("min").values

    # ---------------------------------------------------------------- part A
    zc = np.load(env.LOCAL + "/audit1_calib_oof.npz", allow_pickle=True)
    assert (zc["row_id"] == lab.row_id.values).all()
    sets = sorted({k.split("|")[0] for k in zc.files if "|" in k})
    print("== A. candidate validators: ratios vs real (R2/R1 %.3f, R3/R2 %.3f, R4/R3 %.3f) =="
          % tuple(REAL.values()))
    print("%-22s %6s | %7s %7s %7s %7s | %6s %6s %6s | %s" % ("validator/scoring", "n", "R1", "R2", "R3", "R4",
                                                            "R2/R1", "R3/R2", "R4/R3", "err(2 pts) dir3"))
    rows = []
    second = lab.day.values >= 179
    for s in sets:
        P = {r: zc["%s|%s" % (s, r)] for r in ("R1", "R2", "R3", "HA")}
        P["R4"] = 0.5 * P["R3"] + 0.5 * P["HA"]
        base = ~np.isnan(P["R1"])
        scorings = {"all": base, "clean": base & clean, "testlike": base & testlike}
        if s == "DIAG":
            scorings.update({"2nd pass all": base & second, "2nd pass clean": base & second & clean,
                             "1st pass all": base & ~second, "daymin<10 all": base & (dmin < 10),
                             "daymin<12 all": base & (dmin < 12), "2nd pass testlike": base & second & testlike})
        for sc, m in scorings.items():
            r = {k: rmse_w(P[k][m], y[m]) for k in ("R1", "R2", "R3", "R4")}
            q = {"R2/R1": r["R2"] / r["R1"], "R3/R2": r["R3"] / r["R2"], "R4/R3": r["R4"] / r["R3"]}
            err = abs(q["R2/R1"] - REAL["R2/R1"]) + abs(q["R3/R2"] - REAL["R3/R2"])
            d3 = "ok" if q["R4/R3"] > 1 else "WRONG"
            rows.append((s, sc, err))
            print("%-22s %6d | %7.4f %7.4f %7.4f %7.4f | %6.3f %6.3f %6.3f | %.3f %s"
                  % (s + " " + sc, m.sum(), r["R1"], r["R2"], r["R3"], r["R4"], q["R2/R1"], q["R3/R2"],
                     q["R4/R3"], err, d3))
    # bootstrap intervals for EXT10 ratios
    for s in ("EXT10", "gA", "gB"):
        P = {r: zc["%s|%s" % (s, r)] for r in ("R1", "R2", "R3", "HA")}
        P["R4"] = 0.5 * P["R3"] + 0.5 * P["HA"]
        m = ~np.isnan(P["R1"])
        B = cluster_boot(y[m], {k: v[m] for k, v in P.items()}, unit[m],
                         lambda r: (r["R2"] / r["R1"], r["R3"] / r["R2"], r["R4"] / r["R3"]))
        lo, hi = np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)
        print("%s ratio 95%% day-block CI: R2/R1 [%.3f,%.3f]  R3/R2 [%.3f,%.3f]  R4/R3 [%.3f,%.3f]  P(R4 worse) %.2f"
              % (s, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], (B[:, 2] > 1).mean()))

    # ---------------------------------------------------------------- part B
    zd = np.load(env.LOCAL + "/audit1_dayw_oof.npz", allow_pickle=True)
    assert (zd["row_id"] == lab.row_id.values).all()
    assert (zd["mq4"] == mq4).all()
    variants = sorted({k.split("|")[0] for k in zd.files if "|" in k},
                      key=lambda v: ["plain", "w02", "d02"].index(v) if v in ("plain", "w02", "d02") else 9)

    def bl(v, s):
        return 0.65 * zd["%s|%s|res" % (v, s)] + 0.25 * zd["%s|%s|ridge" % (v, s)] + 0.10 * zd["%s|%s|nys" % (v, s)]

    reps = q4_reps()
    alt = []
    for pr in reps:
        thr = pr.quantile(0.75)
        st = set(pr[pr > thr].index)
        alt.append(np.array([k in st for k in key]))
    pday = ds.set_index(["farm", "day"]).p_oof
    prow = pd.Series(list(key)).map(pday).values.astype(float)
    iw = np.where(np.isnan(prow), 0, (1 - np.nan_to_num(prow, nan=0.5)) / np.nan_to_num(prow, nan=0.5))

    print("\n== B. day weights (blend 0.65/0.25/0.10) ==")
    for s in ("EXT10", "gA", "gB"):
        print("\n-- %s --" % s)
        print("%-6s %8s %8s %8s %8s %8s %8s %8s" % ("", "all", "clean", "testlk", "Q4clean", "tl_alt*", "own_tl",
                                                  "IWclean"))
        for v in variants:
            p = bl(v, s)
            own = clean & ~zd["mask_" + v] if ("mask_" + v) in zd.files else testlike
            tla = np.mean([rmse_w(p[clean & ~m], y[clean & ~m]) for m in alt])
            print("%-6s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f"
                  % (v, rmse_w(p, y), rmse_w(p[clean], y[clean]), rmse_w(p[testlike], y[testlike]),
                     rmse_w(p[clean & mq4], y[clean & mq4]), tla, rmse_w(p[own], y[own]),
                     rmse_w(p[clean], y[clean], iw[clean])))
        # paired day-block bootstraps, incremental d02 vs w02
        a, b, c = bl("plain", s), bl("w02", s), bl("d02", s)
        for nm, m in (("all", np.ones(len(y), bool)), ("clean", clean), ("testlike", testlike),
                      ("Q4 clean", clean & mq4)) + tuple(("alt-Q4 testlike #%d" % i, clean & ~mm)
                                                         for i, mm in enumerate(alt[:3])):
            g = m & ~np.isnan(a)
            B = cluster_boot(y[g], {"a": a[g], "b": b[g], "c": c[g]}, unit[g],
                             lambda r: (r["c"] - r["b"], r["c"] - r["a"]), n=2000)
            pt1 = rmse_w(c[g], y[g]) - rmse_w(b[g], y[g])
            pt2 = rmse_w(c[g], y[g]) - rmse_w(a[g], y[g])
            print("   %-20s d02-w02 %+.4f [%+.4f,%+.4f] Pw %.3f | d02-plain %+.4f [%+.4f,%+.4f]"
                  % (nm, pt1, *np.percentile(B[:, 0], [2.5, 97.5]), (B[:, 0] > 0).mean(),
                     pt2, *np.percentile(B[:, 1], [2.5, 97.5])))
        # random controls: incremental gain over w02 on the SAME testlike mask and on their own mask
        gains_tl, gains_own, gains_clean = [], [], []
        for v in variants:
            if not v.startswith("r"):
                continue
            p = bl(v, s)
            own = clean & ~zd["mask_" + v]
            gains_tl.append(rmse_w(p[testlike], y[testlike]) - rmse_w(b[testlike], y[testlike]))
            gains_own.append(rmse_w(p[own], y[own]) - rmse_w(b[own], y[own]))
            gains_clean.append(rmse_w(p[clean], y[clean]) - rmse_w(b[clean], y[clean]))
        print("   random-day controls vs w02: on Q4-testlike mask %s | on own-excluded mask %s | clean %s"
              % (np.round(gains_tl, 4), np.round(gains_own, 4), np.round(gains_clean, 4)))
        print("   d02 vs w02:                 on Q4-testlike mask %+.4f | clean %+.4f"
              % (rmse_w(c[testlike], y[testlike]) - rmse_w(b[testlike], y[testlike]),
                 rmse_w(c[clean], y[clean]) - rmse_w(b[clean], y[clean])))
    # agreement with saved oof_temp_r3
    zr = np.load(env.LOCAL + "/oof_temp_r3.npz", allow_pickle=True)
    for s, s2 in (("EXT10", "EXT10"), ("gA", "geomA"), ("gB", "geomB")):
        p, q = bl("plain", s), zr[s2]
        g = ~np.isnan(p) & ~np.isnan(q)
        print("reproducibility %s: max |plain - saved| %.2e, RMSE %.4f vs %.4f"
              % (s, np.max(np.abs(p[g] - q[g])), rmse_w(p[g], y[g]), rmse_w(q[g], y[g])))


if __name__ == "__main__":
    main()
