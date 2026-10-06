# -*- coding: utf-8 -*-
"""FI1 (explanation, 2026-10-07 집 클로드).  Does the round-9 EC model (submission_14 core: season v2 + DP1
features in ET / LGB) rely on actuators for high-EC days and on indoor climate for normal days?
Permutation importance of feature GROUPS, measured separately on high-EC rows (label day mean >= 1) and
normal rows, over all 10 DIAG10 folds (ET seed 47 and LGB-tweedie seed 47, the two largest R3 members).
Groups: INDOOR (in_temp, in_hum, in_co2 + their hour-0 values), ACT_NOW (7 actuators now),
ACT_HIST (actuator hour-0 values, today-so-far means and zero shares), OPS (DP1 9 features),
TIME (hr_sin, hr_cos, midnight), SEASON.  Outdoor weather is not a model input (only through SEASON).
Importance = RMSE increase (%) when the group's columns are jointly permuted across validation rows
(same permutation for all columns of the group; 3 repeats averaged)."""
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
BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
ACTS = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
G = {"INDOOR": ["in_temp", "in_hum", "in_co2", "in_temp_h0", "in_hum_h0", "in_co2_h0"],
     "ACT_NOW": ACTS,
     "ACT_HIST": [a + s for a in ACTS for s in ("_h0", "_tdm", "_tdz")],
     "OPS": list(dp1.NEW), "TIME": ["hr_sin", "hr_cos", "midnight"], "SEASON": ["season"]}
rng = np.random.default_rng(0)
res = []
for name, i, vd in [x for x in fds if x[0] == "DIAG10"]:
    va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
    forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
    tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
    tr, va = lab[tr_m].copy(), lab[va_m].copy()
    tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
    season, vq = dc4.season_index(tdays, vdays, wv)
    tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
    va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
    hi = (va.groupby(["farm", "day"]).sub_ec.transform("mean") >= 1).values
    for mname, model, cols in (("ET", core.et(47), FS), ("LGB", core.lg(47, "tweedie"), BS)):
        model.fit(tr[cols], tr.sub_ec.to_numpy())
        if hasattr(model, "steps"):
            model.steps[-1][1].n_jobs = 2
        base = model.predict(va[cols]); y = va.sub_ec.to_numpy()
        for g, gc in G.items():
            gc = [c for c in gc if c in cols]
            if not gc:
                continue
            se = np.zeros(len(va))
            for _ in range(3):
                P = va[cols].copy(); perm = rng.permutation(len(P))
                P[gc] = P[gc].values[perm]
                se += (model.predict(P) - y) ** 2 / 3
            for seg, m in (("high", hi), ("normal", ~hi)):
                if m.sum():
                    res.append(dict(model=mname, fold=i, group=g, seg=seg, base=((base - y) ** 2)[m].sum(), perm=se[m].sum(), n=m.sum()))
    print("fold %s done" % i, flush=True)
R = pd.DataFrame(res)
T = R.groupby(["model", "seg", "group"])[["base", "perm", "n"]].sum()
T["imp%"] = 100 * (np.sqrt(T.perm / T.n) / np.sqrt(T.base / T.n) - 1)
print("\nRMSE increase (%) when a feature group is permuted, by day type")
print(T["imp%"].unstack("seg").round(1).to_string())
R.to_csv(os.path.join(env.LOCAL, "fi1_group_importance_v1.csv"), index=False)
