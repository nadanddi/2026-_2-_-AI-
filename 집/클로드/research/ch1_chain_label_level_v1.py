# -*- coding: utf-8 -*-
"""CH1: if we knew each day's source (동) - proxied by the calendar-order input
continuity chains of deep_cal_11 (chain column, built from inputs only) - how
well would that chain's labelled days predict the daily EC level?
2026-10-02 집 클로드.  Upper-bound-style diagnostic: the chain ids use
calendar-adjacent (record-future) inputs, so they are NOT a legal feature as
is.  A GO here only licenses building a causal chain identifier.

Why: SF1 (sealed high days are high from 0 h), domain critic M1/M4, and
m1_season_vs_treatment_posthoc (same calendar date: 0 pairs both >= 1.2, 20
pairs one high) -> the level is a per-source state; F47 chain 31 = 1.65-2.71,
chain 32 = 0.40-0.51 on the same dates.  Earlier label tests used record
parity (LI1, 51% same source) or the greenhouse-wide mean (6.141).

Design (fixed before running)
  DIAG10 OOF days of EC v2 (Codex phase 3).  For a held-out day (f, d) of fold
  k, anchors = labelled days of the same farm and same chain that are not in
  fold k and not locked.  Predictors of the daily mean:
    CH_near  anchor nearest in calendar position (tie: mean)
    CH_prev  nearest anchor that is EARLIER in record order (d' < d) - the
             label direction allowed by default
    CH_lin   linear interpolation in calendar between nearest anchors before
             and after in calendar (else nearest)
  Blends with v2 daily mean: 0.5*v2 + 0.5*CH_*  (fixed weight, no tuning).
  Days without an anchor fall back to v2.
Decision (GO = build a causal chain identifier):
  GO if CH_prev or CH_near blend (pre-named best of the two by DIAG10 is NOT
  allowed - both are reported, k = 2 Bonferroni) beats v2 at the day level
  (a) on all days with P(worse) < 0.0125 (farm 5-day cluster bootstrap) and
  (b) on sealed days and on late (>= 179) days (direction only).
Row-level RMSE with v2's hourly shape kept is reported alongside.

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ch1_chain_label_level_v1.py
"""
import env  # noqa: F401
import json
import os

import numpy as np
import pandas as pd

ROOT = env.ROOT
RNG = np.random.default_rng(20261002)
LOCK = os.path.join(ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
OOF = os.path.join(ROOT, u"집", u"코덱스", "local", "ec_restart_phase3_20261001_v1", "oof_predictions.csv")


def main():
    lock = {(s["farm"], int(s["day"])) for s in json.load(open(LOCK, encoding="utf-8"))["selected"]}
    C = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_11_days.csv"))[["farm", "day", "cal", "chain", "ec_m", "is_test"]]
    C = C[[(f, d) not in lock for f, d in zip(C.farm, C.day)] | C.is_test]
    C.loc[[(f, d) in lock for f, d in zip(C.farm, C.day)], "ec_m"] = np.nan
    sd = pd.read_csv(os.path.join(env.LOCAL, "sf1_daytable.csv"))[["farm", "day", "sealed"]]
    o = pd.read_csv(OOF, encoding="utf-8-sig")
    o = o[o.validator == "DIAG10"]
    k = ["farm", "day"]
    day = o.groupby(k).agg(fold=("validation_fold", "first"), pm=("v2", "mean"), y=("sub_ec", "mean")).reset_index()
    day = day.merge(C, on=k, how="left").merge(sd, on=k, how="left")
    lab = C[C.ec_m.notna()]
    fold_of = dict(zip(zip(day.farm, day.day), day.fold))
    rows = []
    for r in day.itertuples(index=False):
        A = lab[(lab.farm == r.farm) & (lab.chain == r.chain) & (lab.day != r.day)]
        A = A[[fold_of.get((f, d), -1) != r.fold for f, d in zip(A.farm, A.day)]]
        near = prev = lin = np.nan
        if len(A):
            dist = (A.cal - r.cal).abs()
            near = A.ec_m[dist == dist.min()].mean()
            P = A[A.day < r.day]
            if len(P):
                dp = (P.cal - r.cal).abs()
                prev = P.ec_m[dp == dp.min()].mean()
            lo, hi = A[A.cal <= r.cal], A[A.cal >= r.cal]
            if len(lo) and len(hi):
                a, b = lo.loc[lo.cal.idxmax()], hi.loc[hi.cal.idxmin()]
                lin = a.ec_m if b.cal == a.cal else a.ec_m + (b.ec_m - a.ec_m) * (r.cal - a.cal) / (b.cal - a.cal)
            else:
                lin = near
        rows.append(dict(near=near, prev=prev, lin=lin))
    day = pd.concat([day, pd.DataFrame(rows)], axis=1)
    for c in ("near", "prev", "lin"):
        day["has_" + c] = day[c].notna()
        day[c] = day[c].fillna(day.pm)
        day["b_" + c] = 0.5 * day.pm + 0.5 * day[c]
    day["late"] = day.day >= 179
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    print("days %d; anchor available near %d prev %d lin %d"
          % (len(day), day.has_near.sum(), day.has_prev.sum(), day.has_lin.sum()))
    cols = ["pm", "near", "prev", "lin", "b_near", "b_prev", "b_lin"]
    print("\nday-level RMSE")
    print("%-14s %4s " % ("segment", "n") + " ".join("%7s" % c for c in cols))
    for nm, m in (("all", day.y.notna()), ("sealed", day.sealed == True), ("not sealed", day.sealed != True),
                  ("late>=179", day.late), ("early", ~day.late), ("F13", day.farm == "F13"), ("F47", day.farm == "F47"),
                  ("high>=1.2", day.y >= 1.2)):
        g = day[m]
        print("%-14s %4d " % (nm, len(g)) + " ".join("%7.4f" % r(g[c] - g.y) for c in cols))

    # row level, v2 shape kept
    o2 = o.merge(day[k + ["pm"] + cols[1:]], on=k)
    print("\nrow-level RMSE (v2 hourly shape + level): v2 %.4f" % r(o2.v2 - o2.sub_ec) +
          "".join("  %s %.4f" % (c, r(o2.v2 - o2.pm + o2[c] - o2.sub_ec)) for c in cols[1:]))

    day["cl"] = day.farm + "_" + (day.day // 5).astype(str)
    go = False
    for c in ("b_near", "b_prev"):
        d = (day[c] - day.y) ** 2 - (day.pm - day.y) ** 2
        cl = d.groupby(day.cl).agg(["sum", "count"])
        s, n = cl["sum"].values, cl["count"].values
        bs = np.array([s[i].sum() / n[i].sum() for i in (RNG.integers(0, len(s), len(s)) for _ in range(5000))])
        p = float((bs >= 0).mean())
        seg_ok = all(r(day[c][m] - day.y[m]) < r(day.pm[m] - day.y[m]) for m in (day.sealed == True, day.late))
        ok = p < 0.0125 and seg_ok
        go |= ok
        print("%s: dMSE %+.5f 95%% [%+.5f, %+.5f] P(worse) %.4f  sealed&late better %s -> %s"
              % (c, d.mean(), np.quantile(bs, .025), np.quantile(bs, .975), p, seg_ok, "PASS" if ok else "FAIL"))
    print("\nCH1 decision:", "GO (build causal chain identifier)" if go else "STOP")
    day.to_csv(os.path.join(env.LOCAL, "ch1_daytable.csv"), index=False)


if __name__ == "__main__":
    main()
