# -*- coding: utf-8 -*-
"""EC stage-2 DC1 (diagnostic, fixed before running; 2026-10-02 집 클로드).
Finding (ec2_DAYCAL1_bias_posthoc.log): on DIAG10 late days (record day >= 179)
whose weather-matched calendar position is early (cal < 70, 18 days) EC v2
over-predicts by +0.276 on average, while early-segment days at the same
calendar position are unbiased (+-0.004).  26 of the 60 test days are late &
cal < 70.  Hypothesis: the tree models read the record index `day` as season;
in the 2nd segment that index points to the high-EC epoch of days 120-178.

Test with Codex's phase-3 full ExtraTrees (core.et, core.FULL features, same
DIAG10 folds and lock purge, same shrink + clip post-processing), seed 7:
  BASE   core.FULL (with day)                      - reproduces full_et_7
  NODAY  FULL - day
  CAL    FULL - day + cal (deep_cal_9; NON-CAUSAL upper bound, not a feature)
  TOUT   FULL - day + legal season proxies from the same greenhouse's inputs:
         mean out_temp of the previous record day, mean out_temp over the
         previous 7 record days, running mean out_temp of the current day up
         to the hour (current / earlier inputs only)
Reported: DIAG10 overall RMSE, late cal<70 mean residual and RMSE, late, early.
GO (-> full validators x seeds x R3/v2 in Codex's pipeline) for a LEGAL variant
if it cuts the late cal<70 mean residual by >= 50% and DIAG10 overall RMSE is
not worse than BASE by more than 1%.
Codex's code is imported read-only.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_DC1_day_vs_season_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

P3 = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "ec_restart_phase3_20261001_v1")
sys.path.insert(0, P3)
spec = importlib.util.spec_from_file_location("p3_readonly", os.path.join(P3, "run_benchmark.py"))
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)
core = p3.core


def season_feats(raw):
    a = core.identify(raw).sort_values(["farm", "day", "hour"]).reset_index(drop=True)
    dm = a.groupby(["farm", "day"]).out_temp.mean().rename("dm").reset_index()
    dm["tout_prev1"] = dm.groupby("farm").dm.shift(1)
    dm["tout_prev7"] = dm.groupby("farm").dm.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    a = a.merge(dm[["farm", "day", "tout_prev1", "tout_prev7"]], on=["farm", "day"], how="left")
    a["tout_run"] = a.groupby(["farm", "day"]).out_temp.transform(lambda s: s.expanding().mean())
    return a[["row_id", "tout_prev1", "tout_prev7", "tout_run"]]


def main():
    raw, full, lab, lock, signatures, fds = p3.prepare()
    cal = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_9_days.csv"))[["farm", "day", "cal"]]
    lab = lab.merge(cal, on=["farm", "day"], how="left").merge(season_feats(raw), on="row_id", how="left")
    F = list(core.FULL)
    V = {"BASE": F, "NODAY": [c for c in F if c != "day"],
         "CAL": [c for c in F if c != "day"] + ["cal"],
         "TOUT": [c for c in F if c != "day"] + ["tout_prev1", "tout_prev7", "tout_run"]}
    pred = {k: pd.Series(np.nan, index=lab.index) for k in V}
    for name, i, vd in fds:
        if name != "DIAG10":
            continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m], lab[va_m]
        for k, cols in V.items():
            m = core.et(7)
            m.fit(tr[cols], tr.sub_ec.to_numpy(float))
            pred[k][va.index] = p3.final(m.predict(va[cols]), tr, va)
        print("fold %d done" % i, flush=True)
    D = lab[pred["BASE"].notna()].copy()
    for k in V:
        D[k] = pred[k][D.index]
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    day = D.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), cal=("cal", "first"), **{k: (k, "mean") for k in V})
    day = day.reset_index()
    late_e = (day.day >= 179) & (day.cal < 70)
    print("\n%-6s %8s %10s %12s %10s %10s" % ("var", "DIAG10", "lateE res", "lateE RMSE", "late", "early"))
    base_res = None
    out = {}
    for k in V:
        res = (day[k] - day.y)[late_e].mean()
        out[k] = (r(D[k] - D.sub_ec), res)
        if k == "BASE":
            base_res = res
        print("%-6s %8.4f %+10.3f %12.4f %10.4f %10.4f" % (
            k, r(D[k] - D.sub_ec), res, r((day[k] - day.y)[late_e]),
            r((D[k] - D.sub_ec)[D.day >= 179]), r((D[k] - D.sub_ec)[D.day < 179])))
    print("late&cal<70 days: %d" % late_e.sum())
    for k in ("NODAY", "TOUT"):
        go = (abs(out[k][1]) <= 0.5 * abs(base_res)) and (out[k][0] <= 1.01 * out["BASE"][0])
        print("%s legal GO: %s" % (k, go))
    D[["row_id"] + list(V)].to_csv(os.path.join(env.LOCAL, "ec2_DC1_oof.csv"), index=False)


if __name__ == "__main__":
    main()
