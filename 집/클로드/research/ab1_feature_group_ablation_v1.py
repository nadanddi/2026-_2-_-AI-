# -*- coding: utf-8 -*-
"""AB1 (explanation, 2026-10-07 집 클로드).  Does EC prediction need inputs other than actuators?  Retrain
ExtraTrees (seed 47, as the R3 ET member with DP1) on DIAG10 folds with feature groups removed / alone.
Groups as FI1: INDOOR (in_* now + hour 0), ACT (actuators now + hour-0 / today-so-far + DP1 ops), SEASON
(outdoor-derived calendar index), TIME.  Day-level split: high (label day mean >= 1) / normal."""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dp1", os.path.join(HERE, "ec3_DP1_daily_operation_pattern_v1.py"))
dp1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp1)
dc4, p3, core = dp1.dc4, dp1.p3, dp1.core
raw, full, lab, lock, signatures, fds = p3.prepare()
lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
wv = dc4.weather_vectors(full)
FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
IN = ["in_temp", "in_hum", "in_co2", "in_temp_h0", "in_hum_h0", "in_co2_h0"]; TIME = ["hr_sin", "hr_cos", "midnight"]
ACT = [c for c in FS if c not in IN + TIME + ["season"]]
SETS = {"ALL": FS, "no INDOOR": [c for c in FS if c not in IN], "no ACT": [c for c in FS if c not in ACT],
        "no SEASON": [c for c in FS if c != "season"], "INDOOR+SEASON+TIME": IN + ["season"] + TIME,
        "ACT+TIME only": ACT + TIME, "ACT+SEASON+TIME": ACT + ["season"] + TIME, "SEASON+TIME only": ["season"] + TIME}
P = {k: [] for k in SETS}; Y = []; HI = []
for name, i, vd in [x for x in fds if x[0] == "DIAG10"]:
    va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
    forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
    tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
    tr, va = lab[tr_m].copy(), lab[va_m].copy()
    tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
    season, vq = dc4.season_index(tdays, vdays, wv)
    tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
    va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
    for k, cols in SETS.items():
        P[k].append(p3.final(core.predict_model(core.et(47), tr, va, cols), tr, va))
    Y.append(va.sub_ec.values); HI.append((va.groupby(["farm", "day"]).sub_ec.transform("mean") >= 1).values)
    print("fold %s done" % i, flush=True)
Y = np.concatenate(Y); HI = np.concatenate(HI)
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
print("\nET (seed 47) DIAG10 row RMSE by feature set:  all / normal days / high days")
base = {}
for k in SETS:
    p = np.concatenate(P[k]); v = (r(p - Y), r((p - Y)[~HI]), r((p - Y)[HI]))
    base.setdefault("ALL", v)
    print("  %-20s %.4f / %.4f / %.4f   (vs ALL %+.0f%% / %+.0f%% / %+.0f%%)" % (k, *v, *[100 * (a / b - 1) for a, b in zip(v, base["ALL"])]))
