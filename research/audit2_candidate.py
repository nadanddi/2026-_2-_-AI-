# -*- coding: utf-8 -*-
"""Audit 2 (inspector): does the next candidate's day weighting depend on
LATER evaluation inputs, and is there a rule-safe alternative?

A. Reproduce anal_q3f_days.py's day score (same classifier, same features,
   same season weights) and compare Q4 with local/anal_q3f_dayscore.csv.
B. Re-run it with the test_X rows of the LAST evaluation blocks removed
   (days >= 218 F47 / >= 220 F13).  If the Q4 set changes, the training
   weights -- hence every prediction, including rows of the FIRST block --
   depend on evaluation inputs that come later in time.
C. Same, dropping the future-looking *_ad1_next columns.
D. Vany rule: which flags use later inputs (V5 = centred run length)?
   Compare with a backward-only run length.
E. A train-only alternative: day roughness from backward differences only,
   thresholded inside the training data (no test_X at all); overlap with Q4.
Writes local/audit2_candidate.json.
"""
import json

import env  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

import common
from anal_q3d_adversarial import shape_features
from anal_q3e_matched import SHAPE_ONLY


def day_score(tX, sX, cols):
    a = pd.concat([tX.assign(split="train"), sX.assign(split="test")], ignore_index=True)
    a = a.query("farm in ['F13','F47']").reset_index(drop=True)
    F = shape_features(a)
    a = a.loc[F.index].reset_index(drop=True)
    F = F.reset_index(drop=True)[cols]
    y = (a.split == "train").astype(int).values
    groups = (a.farm + "_" + a.day.astype(str)).values
    bins = np.floor(a.in_temp.clip(-2, 32)).values
    te_h = pd.Series(bins[y == 0]).value_counts(normalize=True)
    tr_h = pd.Series(bins[y == 1]).value_counts(normalize=True)
    sw = np.where(y == 1, pd.Series(bins).map(te_h / tr_h).fillna(0).values, 1.0)
    sw = np.where(y == 1, sw * (y == 0).sum() / sw[y == 1].sum(), sw)
    p = np.zeros(len(a))
    for tr, te in GroupKFold(5).split(F, y, groups):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                               n_jobs=2, random_state=0)
        p[te] = m.fit(F.iloc[tr], y[tr], sample_weight=sw[tr]).predict_proba(F.iloc[te])[:, 1]
    a["p"] = p
    return a[a.split == "train"].groupby(["farm", "day"]).p.mean()


def q4(score, days):
    s = score.reindex(days).dropna()
    thr = s.quantile(0.75)
    return set(s[s >= thr].index), s   # qcut top quartile (ties negligible)


def main():
    tX, ty, sX = common.load_raw()
    ref = pd.read_csv(env.LOCAL + "/anal_q3f_dayscore.csv")
    days = pd.MultiIndex.from_frame(ref[["farm", "day"]])
    refq4 = set(map(tuple, ref[ref.q == "Q4 train-like"][["farm", "day"]].values))
    out = {"n_days": len(ref), "ref_q4": len(refq4)}

    sA = day_score(tX, sX, SHAPE_ONLY)
    qA, _ = q4(sA, days)
    out["A_reproduce_q4_overlap"] = len(qA & refq4)
    print("A reproduce: Q4 %d days, overlap with saved %d / %d" % (len(qA), len(qA & refq4), len(refq4)))

    late = ((sX.farm == "F47") & (sX.day >= 218)) | ((sX.farm == "F13") & (sX.day >= 220))
    sB = day_score(tX, sX[~late], SHAPE_ONLY)
    qB, _ = q4(sB, days)
    out["B_drop_late_test_changed"] = len(qA ^ qB) // 2
    print("B without later test blocks (%d test rows removed): %d of %d Q4 days change"
          % (late.sum(), len(qA - qB), len(qA)))

    noC = [c for c in SHAPE_ONLY if not c.endswith("_next")]
    sC = day_score(tX, sX, noC)
    qC, _ = q4(sC, days)
    out["C_no_next_cols_changed"] = len(qA - qC)
    print("C without *_ad1_next (future) columns: %d of %d Q4 days change" % (len(qA - qC), len(qA)))

    # ---- D: Vany --------------------------------------------------------------
    G = pd.read_pickle(env.LOCAL + "/eda_forensic_7_G.pkl").sort_values(["farm", "t"]).reset_index(drop=True)
    fl = pd.read_csv(env.LOCAL + "/eda_forensic_11_flags.csv").set_index("row_id")
    G = G.join(fl[["V1", "V3", "V4", "V5", "V6", "V7", "Vany"]], on="row_id", rsuffix="_f")
    tr = G[G.set == "train"].copy()
    other = tr[["V1", "V3", "V4", "V6", "V7"]].astype(bool).any(axis=1)
    only5 = tr.Vany.astype(bool) & ~other
    # backward-only run length of identical in_temp (uses t and earlier only)
    bw = np.ones(len(G), int)
    v, t, f = G.in_temp.values, G.t.values, G.farm.values
    for i in range(1, len(G)):
        if f[i] == f[i - 1] and t[i] == t[i - 1] + 1 and v[i] == v[i - 1]:
            bw[i] = bw[i - 1] + 1
    G["bw_run"] = bw
    tr = G[G.set == "train"]
    v5c = tr.bw_run >= 6
    vany_causal = other | v5c
    out["D"] = dict(vany=int(tr.Vany.astype(bool).sum()), only_by_V5=int(only5.sum()),
                    V5=int(tr.V5.astype(bool).sum()), V5_backward=int(v5c.sum()),
                    vany_causal=int(vany_causal.sum()),
                    vany_symdiff=int((vany_causal != tr.Vany.astype(bool)).sum()))
    print("D Vany %d rows (train); flagged only by V5 (centred run) %d; V5 %d vs backward-run %d; "
          "causal Vany %d, rows that differ %d" % tuple(out["D"][k] for k in
                                                     ("vany", "only_by_V5", "V5", "V5_backward",
                                                      "vany_causal", "vany_symdiff")))
    print("   V2 (label-based) in Vany? %s" % ("V2" in ["V1", "V3", "V4", "V5", "V6", "V7"]))

    # ---- E: train-only roughness score ------------------------------------------
    a = tX[tX.farm.isin(["F13", "F47"])].sort_values(["farm", "t"]).copy()
    g = a.groupby("farm")
    ok = g.t.diff() == 1
    for c in ("in_temp", "in_co2", "in_hum"):
        d1 = g[c].diff().where(ok)
        a[c + "_ad1"] = d1.abs()
        a[c + "_ad2"] = (d1 - d1.groupby(a.farm).shift(1)).abs().where(ok & ok.groupby(a.farm).shift(1, fill_value=False))
    night = ~a.hour.between(7, 18)
    a["co2_ns"] = a.in_co2_ad1.where(night & (a.act_co2.fillna(0) == 0) & (a.act_vent.fillna(0) == 0))
    D = a.groupby(["farm", "day"]).agg(t2=("in_temp_ad2", "median"), c1=("co2_ns", "median"),
                                       c2=("in_co2_ad2", "median"), h2=("in_hum_ad2", "median"))
    D = D.reindex(days)
    z = (D - D.median()) / (D.quantile(0.75) - D.quantile(0.25) + 1e-9)
    rough = z.mean(axis=1)
    thr = rough.quantile(0.75)
    qE = set(rough[rough >= thr].index)
    out["E_trainonly_overlap"] = len(qE & refq4)
    base = 0.25 * len(refq4)
    print("E train-only roughness top quartile: overlap with classifier Q4 %d / %d (chance ~%.0f)"
          % (len(qE & refq4), len(refq4), base))
    with open(env.LOCAL + "/audit2_candidate.json", "w") as fh:
        json.dump(out, fh, indent=1, default=int)


if __name__ == "__main__":
    main()
