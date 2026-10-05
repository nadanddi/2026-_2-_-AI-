# -*- coding: utf-8 -*-
"""CH4 (diagnostic, fixed before running; 2026-10-06 집 클로드; user: "일반날만 이 모델로 바꿔볼 순 없나").
CH3: picking THE predecessor is near random (36-39 %), but if all predecessor candidates agree, the pick
does not matter.  Apply the label-continuity level only on rows the model calls NORMAL and only when the
candidates agree:
  candidates = labelled reference records (outside the DIAG10 fold) with outdoor date_cost(p -> q) <= 3
               (query hours 0-1 only);  v_p = EC_p(23);
  gate (row at hour h): model's today-so-far mean pm_h < .9  AND  >= 2 candidates  AND
               spread (max - min of v_p) <= S;   level L = median(v_p);
  CH4 pred = m_h + 0.5 (L + (m_h - m_0) - m_h) = m_h + 0.5 (L - m_0)   (half-way, like SG2), on gated rows;
  else the model.  S in {.05, .10, .20} (all reported; .10 is the pre-named main setting).
Baseline = R3S seed mean (and SG2 for reference), pass-2 rows of DIAG10.  Clue (main S = .10): beats
the model on pass-2 rows and normal days, high days not worse by > 1 %, gate covers >= 20 % of normal rows."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch3_place_into_chain_gaps_v1.py"), encoding="utf-8").read()
exec(src.split("# full pair-cost matrices")[0])
O = pd.read_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv"))
Fd = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"), usecols=["farm", "day", "validator", "validation_fold"])
Fd = Fd[Fd.validator == "DIAG10"].drop_duplicates(["farm", "day"]).set_index(["farm", "day"]).validation_fold
O = O[O.day >= 179].sort_values(["farm", "day", "hour"]).copy()
O["m0"] = O.groupby(["farm", "day"]).p.transform("first")
O["pm"] = O.groupby(["farm", "day"]).p.transform(lambda z: z.expanding().mean())
O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
d0 = O.a1 - O.pm; O["sg"] = np.where(O.a1.notna() & (d0.abs() <= .3), O.p + .5 * d0, O.p)
info = {}
for (f, d) in O.groupby(["farm", "day"]).groups:
    k = Fd[(f, d)]
    ref = [e for e in P.loc[f].index if Fd.get((f, e), -1) != k and e != d]
    cs = [p for p in ref if np.isfinite(dcost(f, p, d)) and dcost(f, p, d) <= 3]
    v = np.array([P.loc[(f, p), ("sub_ec", 23)] for p in cs])
    info[(f, d)] = (len(v), (v.max() - v.min()) if len(v) else np.nan, np.median(v) if len(v) else np.nan)
O["nc"] = [info[(f, d)][0] for f, d in zip(O.farm, O.day)]
O["spread"] = [info[(f, d)][1] for f, d in zip(O.farm, O.day)]
O["L"] = [info[(f, d)][2] for f, d in zip(O.farm, O.day)]
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
D = O.groupby(["farm", "day"]).agg(y=("dm", "first"), nc=("nc", "first"), spread=("spread", "first"), L=("L", "first"), y0=("sub_ec", "first"))
print("days: candidates median %d; spread median %.2f; days with spread <= .10: %d of %d (true normal %d); |median - true 0h| on those: %.3f" % (
    D.nc.median(), D.spread.median(), (D.spread <= .1).sum(), len(D), ((D.spread <= .1) & (D.y < 1)).sum(),
    (D[D.spread <= .1].L - D[D.spread <= .1].y0).abs().median()))
clue = False
for S in (.05, .10, .20):
    g = (O.pm < .9) & (O.nc >= 2) & (O.spread <= S)
    new = np.where(g, O.p + .5 * (O.L - O.m0), O.p)
    seg = {nm: (r(O.p[m] - O.sub_ec[m]), r(new[m] - O.sub_ec[m]), r(O.sg[m] - O.sub_ec[m])) for nm, m in
           (("pass-2", O.dm.notna()), ("normal", O.dm < 1), ("high", O.dm >= 1))}
    cov = g[O.dm < 1].mean()
    print("S %.2f: gated %.0f%% of normal rows, %.0f%% of high rows | pass-2 %.4f->%.4f (SG2 %.4f) | normal %.4f->%.4f (SG2 %.4f) | high %.4f->%.4f" % (
        S, 100 * cov, 100 * g[O.dm >= 1].mean(), *seg["pass-2"][:2], seg["pass-2"][2], *seg["normal"][:2], seg["normal"][2], *seg["high"][:2]))
    if S == .10:
        clue = seg["pass-2"][1] < seg["pass-2"][0] and seg["normal"][1] < seg["normal"][0] and seg["high"][1] <= 1.01 * seg["high"][0] and cov >= .2
print("CH4 clue (S = .10):", clue)
