# -*- coding: utf-8 -*-
"""SE0 (diagnostic, fixed before running; 2026-10-03 집 클로드).
Validation / evaluation days get their season only by record-day interpolation
(DC4).  How far is that from the season their own weather points to, and does the
gap explain the remaining R3S day error?
For each DIAG10 validation day in pass 2 (record day >= 179): season_interp as in
DC5 (season_index with that fold's training days); season_twin = mean pass-1 record
day of the SAME farm's pass-1 training days whose z-RMSE (DC4 z-scaling) <= .05,
else the nearest one (flag).  Also season_twin_any = DC4's own rule (both farms'
pass-1 days, record days averaged) to measure the cross-farm mixing (ST2: F13/F47
record days of one date differ by -8..+8).
Evaluation (test) days: same comparison with all labelled non-lock days as training
(test_X weather used for structure description only).
Reading (fixed): clue if on DIAG10 late days Spearman(season_interp - season_twin,
R3S day residual) has |rho| >= .30 with n >= 30, or median |interp - twin| >= 8
record days on test days.
"""
import env  # noqa: F401
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc5)
dc4, p3 = dc5.dc4, dc5.p3
W = dc4.W


def twin_season(f, d, p1, ZZ, same_farm=True):
    P = p1[p1.farm == f] if same_farm else p1
    A = ZZ.reindex(list(zip(P.farm, P.day))).values
    b = ZZ.reindex([(f, d)]).values
    dist = np.sqrt(np.nanmean((A - b) ** 2, axis=1))
    if np.nanmin(dist) <= 0.05:
        return float(P.day.values[dist <= 0.05].mean()), True, float(np.nanmin(dist))
    return float(P.day.values[np.nanargmin(dist)]), False, float(np.nanmin(dist))


def zscaled(wv, p1):
    Z = {}
    for v in W:
        blk = wv[v]
        vals = blk.reindex(list(zip(p1.farm, p1.day))).values
        Z[v] = (blk - np.nanmean(vals)) / np.nanstd(vals)
    return pd.concat(Z, axis=1)


def main():
    raw, full, lab, lock, signatures, fds = p3.prepare()
    te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"] + W)
    te = te[te.row_id.str[:3].isin(["F13", "F47"])]
    wv_all = dc4.weather_vectors(pd.concat([full[["row_id"] + W], te]))
    wv = dc4.weather_vectors(full)
    O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))
    O = O[O.validator == "DIAG10"].copy()
    O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
    rows = []
    for k, G in O.groupby("validation_fold"):
        vd = set(zip(G.farm, G.day))
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tdays = lab[[(f, d) not in forb for f, d in zip(lab.farm, lab.day)]][["farm", "day"]].drop_duplicates()
        late = sorted({(f, d) for f, d in vd if d >= 179})
        if not late:
            continue
        q = pd.DataFrame(late, columns=["farm", "day"])
        _, sq = dc4.season_index(tdays, q, wv)
        p1 = tdays[tdays.day < 179]
        ZZ = zscaled(wv, p1)
        for (f, d), si in zip(late, sq):
            st, exact, dist = twin_season(f, d, p1, ZZ, True)
            sa, exa, _ = twin_season(f, d, p1, ZZ, False)
            g = G[(G.farm == f) & (G.day == d)]
            rows.append(dict(farm=f, day=d, fold=k, s_interp=si, s_twin=st, exact=exact, dist=dist, s_twin_any=sa,
                             resid=(g.sub_ec - g.p).mean(), y=g.sub_ec.mean()))
    R = pd.DataFrame(rows)
    R["gap"] = R.s_interp - R.s_twin
    print("DIAG10 late validation days %d (exact same-farm twin %d)" % (len(R), R.exact.sum()))
    print("  |interp - twin| median %.1f, mean %.1f, max %.1f record days" % (R.gap.abs().median(), R.gap.abs().mean(), R.gap.abs().max()))
    print("  twin same-farm vs any-farm (DC4 rule) |diff| median %.1f, max %.1f" % ((R.s_twin - R.s_twin_any).abs().median(), (R.s_twin - R.s_twin_any).abs().max()))
    rho = spearmanr(R.gap, R.resid).correlation
    rho_abs = spearmanr(R.gap.abs(), R.resid.abs()).correlation
    rho_e = spearmanr(R.gap[R.exact], R.resid[R.exact]).correlation if R.exact.sum() > 5 else np.nan
    print("  Spearman(gap, resid) %.2f | (|gap|, |resid|) %.2f | exact twins only %.2f (n %d)" % (rho, rho_abs, rho_e, R.exact.sum()))
    print(R.sort_values("gap").round(2).to_string(index=False))
    # test days
    lab_days = lab[[(f, d) not in {(a, b + j) for a, b in lock for j in (-1, 0, 1)} for f, d in zip(lab.farm, lab.day)]][["farm", "day"]].drop_duplicates()
    te["farm"], te["day"] = te.row_id.str[:3], te.row_id.str[4:7].astype(int)
    q = te[["farm", "day"]].drop_duplicates().sort_values(["farm", "day"]).reset_index(drop=True)
    _, sq = dc4.season_index(lab_days, q, wv)
    p1 = lab_days[lab_days.day < 179]
    ZZ = zscaled(wv_all, p1)
    T = []
    for (f, d), si in zip(zip(q.farm, q.day), sq):
        st, exact, dist = twin_season(f, d, p1, ZZ, True)
        T.append(dict(farm=f, day=d, s_interp=si, s_twin=st, exact=exact, dist=dist))
    T = pd.DataFrame(T); T["gap"] = T.s_interp - T.s_twin
    print("\nTEST days %d (exact same-farm twin %d): |interp - twin| median %.1f, mean %.1f, max %.1f" % (
        len(T), T.exact.sum(), T.gap.abs().median(), T.gap.abs().mean(), T.gap.abs().max()))
    print(T.round(2).to_string(index=False))
    T.to_csv(os.path.join(env.LOCAL, "se0_test_season_gap.csv"), index=False)
    R.to_csv(os.path.join(env.LOCAL, "se0_diag_season_gap.csv"), index=False)
    clue = (abs(rho) >= .30 and len(R) >= 30) or T.gap.abs().median() >= 8
    print("\nSE0 clue:", clue)


if __name__ == "__main__":
    main()
