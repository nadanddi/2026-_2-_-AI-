# -*- coding: utf-8 -*-
"""VA1 validator audit (descriptive; 2026-10-04 집 클로드).  User request: are some
validators meaningless, redundant or misleading?
(1) Representativeness: held-out day sets of DIAG10 / A / B / EXT10 / EXT12 / EL1 vs
    the 60 evaluation days: pass-2 share, season position (record day -> DC4-like
    calendar via C6.205 date index rank inside its pass), date distance to the nearest
    labelled training day of the same farm, sealed share (vent zero >= .8), mean
    in_temp, cold share (in_temp day min < 10), pair-second share.  Inputs only
    (test_X inputs used for description only).
(2) Noise: held-out days, R3S RMSE seed spread (max-min over seeds 7/101/2024 in %).
(3) Agreement: for every saved experiment OOF (candidate vs R3S on the same rows),
    the seed-mean % change per validator; Spearman correlation of these changes
    across experiments between validators, and sign agreement with EL1 and with
    DIAG10-late (pass-2 rows of DIAG10)."""
import env  # noqa: F401
import importlib.util, json, os, sys, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc5)
p3 = dc5.p3
SEEDS = (7, 101, 2024)
raw, full, lab, lock, signatures, fds = p3.prepare()
el = []
for f in ("F13", "F47"):
    days = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
    for k in range(0, len(days), 5):
        el.append(("EL1", len(el), {(f, int(d)) for d in days[k:k + 5]}))
sets = {}
for name, i, vd in list(fds) + el:
    sets.setdefault(name, []).append(vd)
te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"]); te = te[te.row_id.str[:3].isin(["F13", "F47"])]
sets["TEST"] = [set(zip(te.row_id.str[:3], te.row_id.str[4:7].astype(int)))]
# day descriptors
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f), usecols=["row_id", "in_temp", "act_vent"]) for f in ("train_X.csv", "test_X.csv")])
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy(); X["farm"], X["day"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int)
DX = X.groupby(["farm", "day"]).agg(tmean=("in_temp", "mean"), tmin=("in_temp", "min"), vz=("act_vent", lambda s: (s == 0).mean()))
R = pd.read_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv")).sort_values(["farm", "day"])
R["date"] = R.groupby("farm").role.transform(lambda s: np.cumsum(s.values != "second") - 1)
R = R.set_index(["farm", "day"])
labelled = {(f, int(d)) for f, d in zip(lab.farm, lab.day)}
rows = []
for name, folds in sets.items():
    vals = []
    for vd in folds:
        lk = {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        avail = {k for k in labelled if k not in vd and k not in lk and (k[0], k[1] - 1) not in vd and (k[0], k[1] + 1) not in vd} if name != "TEST" else (labelled - lock)
        for f, d in vd:
            if (f, d) not in R.index: continue
            dt = R.loc[(f, d), "date"]
            ad = [abs(R.loc[k, "date"] - dt) for k in avail if k[0] == f and k in R.index and (k[1] >= 179) == (d >= 179)]
            vals.append(dict(late=d >= 179, gap=min(ad) if ad else np.nan, second=R.loc[(f, d), "role"] == "second",
                             tmean=DX.tmean.get((f, d), np.nan), cold=DX.tmin.get((f, d), np.nan) < 10, sealed=DX.vz.get((f, d), np.nan) >= .8))
    V = pd.DataFrame(vals)
    rows.append(dict(validator=name, folds=len(folds), days=len(V), late=V.late.mean(), gap_med=V.gap.median(), gap_mean=V.gap.mean(),
                     second=V.second.mean(), tmean=V.tmean.mean(), cold=V.cold.mean(), sealed=V.sealed.mean()))
print("(1) REPRESENTATIVENESS (held-out day sets vs TEST)")
print(pd.DataFrame(rows).round(3).to_string(index=False))
# (2) noise
d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))
e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
B = pd.concat([d5[["row_id", "farm", "day", "sub_ec", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]],
               e1[["row_id", "farm", "day", "sub_ec", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]])
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
print("\n(2) NOISE: R3S RMSE by seed and spread")
for v, G in B.groupby("validator"):
    sc = [r(G["r3s_%d" % s] - G.sub_ec) for s in SEEDS]
    print("  %-6s days %3d  RMSE %s  seed spread %.2f%%" % (v, G.groupby(["farm", "day"]).ngroups, " ".join("%.4f" % x for x in sc), 100 * (max(sc) - min(sc)) / np.mean(sc)))
# (3) agreement across experiments
exp = {"DI1": ("ec3_DI1_all.csv", "di"), "TS1": ("ec3_TS1_all.csv", "ts"), "LG1": ("ec3_LG1_all.csv", "lg"), "LR1": ("ec3_LR1_all.csv", "lr"),
       "MW1": ("ec3_MW1_all.csv", "mw"), "TS2": ("ec3_TS2_all.csv", "ts"), "ST8": ("ec3_ST8_all.csv", "st"), "SE3": ("ec3_SE3_all.csv", "se"),
       "DC6": ("ec3_DC6_all.csv", "dc6"), "ST5": ("ec3_ST5_all.csv", "st5"), "SB1": ("ec3_SB1_all.csv", "sb"), "HG1": ("ec3_HG1_all.csv", "hg")}
VS = ["DIAG10", "A", "B", "EXT10", "EXT12", "EL1"]
res = {}
for nm, (fn, pre) in exp.items():
    p = os.path.join(env.LOCAL, fn)
    if not os.path.exists(p): continue
    O = pd.read_csv(p)
    if "validator" not in O: continue
    out = {}
    for v in VS:
        G = O[O.validator == v]
        if not len(G): continue
        out[v] = np.mean([100 * (r(G["%s_%d" % (pre, s)] - G.sub_ec) / r(G["r3s_%d" % s] - G.sub_ec) - 1) for s in SEEDS])
        if v == "DIAG10":
            L = G[G.day >= 179]
            out["DIAG10_late"] = np.mean([100 * (r(L["%s_%d" % (pre, s)] - L.sub_ec) / r(L["r3s_%d" % s] - L.sub_ec) - 1) for s in SEEDS])
    res[nm] = out
# season effect (DC5: r3 day-version -> r3s) as an extra experiment
out = {}
for v in VS:
    G = B[B.validator == v] if v != "EL1" else None
for v in VS:
    if v == "EL1":
        G = e1; a = "r3"
    else:
        G = d5[d5.validator == v]; a = "r3"
    out[v] = np.mean([100 * (r(G["r3s_%d" % s] - G.sub_ec) / r(G["%s_%d" % (a, s)] - G.sub_ec) - 1) for s in SEEDS])
L = d5[(d5.validator == "DIAG10") & (d5.day >= 179)]
out["DIAG10_late"] = np.mean([100 * (r(L["r3s_%d" % s] - L.sub_ec) / r(L["r3_%d" % s] - L.sub_ec) - 1) for s in SEEDS])
res["SEASON(DC5)"] = out
T = pd.DataFrame(res).T[VS + ["DIAG10_late"]]
print("\n(3) seed-mean %% change per validator, by experiment")
print(T.round(1).to_string())
print("\n  Spearman correlation of changes across %d experiments" % len(T))
print(T.corr(method="spearman").round(2).to_string())
for ref in ("EL1", "DIAG10_late"):
    agree = {v: float((np.sign(T[v]) == np.sign(T[ref])).mean()) for v in T.columns if v != ref}
    print("  sign agreement with %s: %s" % (ref, {k: round(x, 2) for k, x in agree.items()}))
T.to_csv(os.path.join(env.LOCAL, "va1_experiment_changes.csv"))
