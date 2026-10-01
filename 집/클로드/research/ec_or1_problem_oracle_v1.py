# -*- coding: utf-8 -*-
"""EC-OR1: which single EC problem, if solved, lowers the score most?
Oracle table on EC v2 DIAG10 OOF (Codex phase 3, 360 days, lock excluded).
2026-10-02 집 클로드.

For each problem two fixes are scored:
  ORACLE  replace v2's daily mean by the true daily mean on those days
          (shape kept) - upper bound of "knowing the level"
  BIAS    subtract that segment's mean residual (one number per segment) -
          the realistic "only a systematic offset" fix
Segments: high-EC days (true daily mean >= 1.2), sealed days (inputs: daily
circfan mean < 10 and act_vent == 0 share > 0.85, catalog 6.11), late
segment (day >= 179, test is all late), cold midnight days, F13/F47, and the
hourly shape everywhere.  Also: eval-like reweight (sealed 33%, all late
impossible -> reported for late only).

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_or1_problem_oracle_v1.py
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd

o = pd.read_csv(os.path.join(env.ROOT, u"집", u"코덱스", "local", "ec_restart_phase3_20261001_v1",
                             "oof_predictions.csv"), encoding="utf-8-sig")
o = o[o.validator == "DIAG10"].copy()
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
X = X[X.row_id.isin(o.row_id)]
o = o.merge(X[["row_id", "act_circfan", "act_vent", "in_temp"]], on="row_id", how="left")
k = ["farm", "day"]
o["e"] = o.v2 - o.sub_ec
o["tm"] = o.groupby(k).sub_ec.transform("mean")
o["pm"] = o.groupby(k).v2.transform("mean")
dfan = o.groupby(k).act_circfan.transform("mean")
dv0 = o.groupby(k).act_vent.transform(lambda s: (s == 0).mean())
o["sealed"] = (dfan < 10) & (dv0 > 0.85)
o["high"] = o.tm >= 1.2
o["late"] = o.day >= 179
o["coldmid"] = o.groupby(k).in_temp.transform(lambda s: s.iloc[0] if len(s) else np.nan) <= 10
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
base = r(o.e)
print("v2 DIAG10 RMSE %.4f   days %d   sealed days %d  high days %d  late days %d"
      % (base, o.groupby(k).ngroups, o[o.sealed].groupby(k).ngroups, o[o.high].groupby(k).ngroups,
         o[o.late].groupby(k).ngroups))
print("%-22s %6s %8s %8s %9s %8s %8s" % ("problem", "days", "sq-err%", "seg RMSE", "ORACLE", "gain", "BIAS"))
seg = {"all days level": np.ones(len(o), bool), "high-EC days": o.high.values, "sealed days": o.sealed.values,
       "sealed & high": (o.sealed & o.high).values, "sealed not high": (o.sealed & ~o.high).values,
       "late (>=179)": o.late.values, "cold midnight": o.coldmid.values, "F47": (o.farm == "F47").values,
       "normal days": (~o.sealed & ~o.high).values}
for nm, m in seg.items():
    orc = np.where(m, o.v2 - o.pm + o.tm, o.v2)
    bias = np.where(m, o.v2 - o.e[m].mean(), o.v2)
    print("%-22s %6d %7.1f%% %8.4f %9.4f %7.1f%% %8.4f"
          % (nm, o[m].groupby(k).ngroups, 100 * (o.e[m] ** 2).sum() / (o.e ** 2).sum(), r(o.e[m]),
             r(orc - o.sub_ec), 100 * (r(orc - o.sub_ec) / base - 1), r(bias - o.sub_ec)))
shape = o.pm + (o.sub_ec - o.tm)
print("%-22s %6s %8s %8s %9.4f %7.1f%%" % ("hourly shape (all)", "", "", "", r(shape - o.sub_ec), 100 * (r(shape - o.sub_ec) / base - 1)))
print("\nlevel residual (pred - true daily mean) by segment: mean / median")
for nm in ("high-EC days", "sealed days", "sealed not high", "late (>=179)", "normal days"):
    d = o[seg[nm]].groupby(k).agg(pm=("pm", "first"), tm=("tm", "first"))
    print("   %-16s %+.3f / %+.3f   corr(pred, true) %.2f" % (nm, (d.pm - d.tm).mean(), (d.pm - d.tm).median(),
                                                             np.corrcoef(d.pm, d.tm)[0, 1]))
# eval-like mix: sealed share 33%
w = np.where(o.sealed, 0.33 / o.sealed.mean(), 0.67 / (~o.sealed).mean())
print("\neval-like weights (sealed 33%%): v2 %.4f, sealed-level oracle %.4f, high-day oracle %.4f"
      % (np.sqrt(np.average(o.e ** 2, weights=w)),
         np.sqrt(np.average((np.where(o.sealed, o.v2 - o.pm + o.tm, o.v2) - o.sub_ec) ** 2, weights=w)),
         np.sqrt(np.average((np.where(o.high, o.v2 - o.pm + o.tm, o.v2) - o.sub_ec) ** 2, weights=w))))
