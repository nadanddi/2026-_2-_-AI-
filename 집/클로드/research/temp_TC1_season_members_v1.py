# -*- coding: utf-8 -*-
"""Temperature TC1 (diagnostic, fixed before running; 2026-10-03 집 클로드).
User: build an improved temperature model the way the EC season fix was found.
Hypothesis: temperature members read the record index `day` as season (MASK
base74 has `day`; Codex member FEATURE_COLUMNS has `day`), so in pass 2
(record day >= 179, the whole test) the season is mis-read, as in EC (C6.157).
Evidence before running: G_C2 DIAG10 late RMSE .58-.80 vs early .45-.48, late
bias by calendar -0.15 / +0.13 C (temp_daycal_bias_check_posthoc.log).

Season index = EC DC4 definition (pass 1: own day; pass-2 training days:
exact pass-1 weather twins (z-RMSE <= .05) as anchors, isotonic, record-day
interpolation; held-out days use only their own record day), built per fold
from that fold's TRAINING days.
Members (MASK world, round-5 weights, as temp_mask_v1.py):
  MASK = 0.65 res + 0.25 ridge + 0.10 nys (temp_members, seed 7)
  CODEX = codex_fit_predict (seed 726)
  variants BASE (with day) vs SEAS (day -> season in both feature lists)
Validators: DIAG10 (5-day chunks, round robin), EXT10, EXT12 (temp_mask_v1).
GO (to TC2 = full W30G blend under the user rule) if, for BOTH members,
DIAG10 late (>= 179) RMSE improves by >= 3% AND DIAG10 overall RMSE is not
worse than BASE by more than 0.5%.  EXT reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u temp_TC1_season_members_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

import common
import harness
import cold_v5
import temp_mask_v1 as TM
from common import split_mask, TARGET_FARMS
from screen_v6 import temp_members
from anal_q1_errors import diag_folds
import train_flags_v6 as TF

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc4", os.path.join(HERE, "ec2_DC4_exact_twin_anchor_v1.py"))
dc4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc4)


def main():
    common.load_raw = TM.masked_loader
    try:
        labM, ct, phc = TM.build_world()
    finally:
        common.load_raw = TM.ORIG
        harness._CACHE.clear()
    tX, ty, sX = TM.ORIG()
    CF = TM.build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    for c in TM.FEATURE_COLUMNS:
        if c not in labM.columns:
            labM[c] = CF.loc[labM.row_id, c].values
    w = TF.row_weights(labM, 0.2, w_noisy=0.2)
    full = tX[tX.row_id.str[:3].isin(["F13", "F47"])]
    wv = dc4.weather_vectors(full)
    cal = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_9_days.csv"))[["farm", "day", "cal"]]
    calm = cal.set_index(["farm", "day"]).cal
    labF, _, _ = TM.build_world()  # FULL world only for EXT day definitions, as in temp_mask_v1
    dmin = labF.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(labM))]
    for th in (10.0, 12.0):
        cd = dmin[dmin < th]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))
    ctS = [c if c != "day" else "season" for c in ct]
    cxB = [c for c in TM.FEATURE_COLUMNS if c in labM.columns]
    cxS = [c if c != "day" else "season" for c in cxB]
    assert "day" in ct and "day" in cxB
    cold_v5.SEED = 7
    out = []
    for s, fds in sets:
        for k, fd in enumerate(fds):
            trm, vam = split_mask(labM, fd)
            if not vam.sum():
                continue
            tr, va = labM[trm].copy(), labM[vam].copy()
            tdays = tr[["farm", "day"]].drop_duplicates()
            vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
            season, vq = dc4.season_index(tdays, vdays, wv)
            tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
            vmap = dict(zip(zip(vdays.farm, vdays.day), vq))
            va["season"] = [vmap[(f, d)] for f, d in zip(va.farm, va.day)]
            vr = va.reset_index(drop=True)
            frame = vr[["row_id", "farm", "day", "hour", "sub_temp"]].copy()
            frame["validator"], frame["fold"] = s, k
            for tag, cols, cxc in (("base", ct, cxB), ("seas", ctS, cxS)):
                M = temp_members(tr, vr, cols, phc, w[trm])
                frame["mask_" + tag] = 0.65 * M["res"] + 0.25 * M["ridge"] + 0.10 * M["nys"]
                frame["codex_" + tag] = TM.codex_fit_predict(tr.assign(day=tr["day"]) if tag == "base" else tr, vr, w[trm], 726) \
                    if tag == "base" else TM.codex_fit_predict(tr.assign(**{"day": tr["season"]}), vr.assign(**{"day": vr["season"]}), w[trm], 726)
            out.append(frame)
            print("%s fold %d done" % (s, k), flush=True)
    O = pd.concat(out, ignore_index=True)
    O["cal"] = [calm.get((f, d), np.nan) for f, d in zip(O.farm, O.day)]
    O.to_csv(os.path.join(env.LOCAL, "temp_TC1_oof.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    print("\n%-7s %-6s %8s %8s | %8s %8s | %8s %8s | %8s %8s" % ("set", "member", "all_b", "all_s", "late_b", "late_s", "lateE_b", "lateE_s", "early_b", "early_s"))
    ok = True
    for s in ("DIAG10", "EXT10", "EXT12"):
        g = O[O.validator == s]
        for mem in ("mask", "codex"):
            L, E, LE = g.day >= 179, g.day < 179, (g.day >= 179) & (g.cal < 70)
            vals = [r(g[mem + "_base"] - g.sub_temp), r(g[mem + "_seas"] - g.sub_temp),
                    r((g[mem + "_base"] - g.sub_temp)[L]), r((g[mem + "_seas"] - g.sub_temp)[L]),
                    r((g[mem + "_base"] - g.sub_temp)[LE]), r((g[mem + "_seas"] - g.sub_temp)[LE]),
                    r((g[mem + "_base"] - g.sub_temp)[E]), r((g[mem + "_seas"] - g.sub_temp)[E])]
            print("%-7s %-6s %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f" % ((s, mem) + tuple(vals)))
            if s == "DIAG10":
                ok &= (vals[3] <= 0.97 * vals[2]) and (vals[1] <= 1.005 * vals[0])
    print("\nTC1 decision:", "GO" if ok else "STOP")


if __name__ == "__main__":
    main()
