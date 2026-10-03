# -*- coding: utf-8 -*-
"""EC stage-3 TS2: one-row-per-day level models, one per cutoff hour (fixed before
running; 2026-10-03 집 클로드).
Basis: 87% of EC error is day level; high-EC days are ranked (AUC .989) but their
size is under-predicted (1.55 vs 1.23, 6.200).  TS1 (6.182) fitted the day-mean
target on hourly rows (each day repeated 24x with drifting features); Codex 6.199
noted a true one-row-per-day model was never tried.  Here, for each cutoff hour h,
a separate ExtraTrees is fitted on ONE row per training day: the day features as
known at hour h (legal: same greenhouse, inputs of hours 0..h only), target = the
day mean label.  ~300 rows per model, so min_samples_leaf = 3 (fixed now).
Features at cutoff h (14 inputs: 4 outdoor + 3 indoor + 7 actuators): value at h,
expanding mean over 0..h, value at hour 0; plus season (DC4) -> 43 columns.
DM_h = model_h(day features at h).  Candidate per seed:
  clip(R3S + 0.5 * (DM_h - cm_h(R3S)), train range)   (same form as TS1)
cm_h = expanding mean of R3S over hours 0..h of the same day.
Baseline R3S = stored DC5 / EL1 OOF.  Rule (user's): all 3 seeds x 5 validators
better and DIAG10 P(worse) < .025; EL1 better for all seeds also required.
High-EC days (label day mean >= 1) mean label / R3S / candidate reported.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_TS2_day_row_models_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc5)
dc4, p3, core = dc5.dc4, dc5.p3, dc5.core
SEEDS = (7, 101, 2024)
CK = os.path.join(env.LOCAL, "ts2_ckpt")
LAM = 0.5
V14 = ["out_temp", "out_hum", "out_rad", "out_wspd"] + core.INDOOR + core.ACTS


def add_cutoff_features(lab):
    lab = lab.sort_values(["farm", "day", "hour"]).copy()
    assert (lab.groupby(["farm", "day"]).hour.min() == 0).all()
    g = lab.groupby(["farm", "day"])
    cols = []
    for v in V14:
        lab["c_" + v] = lab[v]
        lab["cm_" + v] = g[v].transform(lambda s: s.expanding().mean())
        lab["c0_" + v] = g[v].transform("first")   # hour 0 = first row of the day
        cols += ["c_" + v, "cm_" + v, "c0_" + v]
    return lab, cols


def et(seed):
    return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(
        n_estimators=600, max_features=1.0, min_samples_leaf=3, n_jobs=4, random_state=seed))


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    lab, CC = add_cutoff_features(lab)
    FC = CC + ["season"]
    el = []
    for f in ("F13", "F47"):
        days = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(days), 5):
            el.append(("EL1", len(el), {(f, int(d)) for d in days[k:k + 5]}))
    for name, i, vd in list(fds) + el:
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates()
        vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
        vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        tr["ym"] = tr.groupby(["farm", "day"]).sub_ec.transform("mean")
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        frame["lo"], frame["hi"] = tr.sub_ec.min(), tr.sub_ec.max()
        for s in SEEDS:
            out = np.full(len(va), np.nan)
            for h in range(24):
                th, vh = tr[tr.hour == h], (va.hour == h).values
                if not vh.any():
                    continue
                m = et(s).fit(th[FC], th.ym.values)
                out[vh] = m.predict(va.loc[vh, FC])
            frame["dm_%d" % s] = out
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
    base = pd.concat([d5, e1[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]], ignore_index=True)
    O = P.merge(base, on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(O) == len(P), (len(O), len(P))
    O = O.sort_values(["validator", "validation_fold", "farm", "day", "hour"]).reset_index(drop=True)
    g = O.groupby(["validator", "validation_fold", "farm", "day"])
    for s in SEEDS:
        cm = g["r3s_%d" % s].transform(lambda x: x.expanding().mean())
        O["ts_%d" % s] = np.clip(O["r3s_%d" % s] + LAM * (O["dm_%d" % s] - cm), O.lo, O.hi)
    O.to_csv(os.path.join(env.LOCAL, "ec3_TS2_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261003)
    allb = True
    print("\npooled RMSE R3S -> R3S + one-row-per-day cutoff models (lambda .5)")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        G = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(G["r3s_%d" % s] - G.sub_ec), r(G["ts_%d" % s] - G.sub_ec)
            if v != "EL1":
                allb &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["ts_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    E = O[O.validator == "EL1"]
    elok = all(r(E["ts_%d" % s] - E.sub_ec) < r(E["r3s_%d" % s] - E.sub_ec) for s in SEEDS)
    L = D.day >= 179
    hi = D.groupby(["farm", "day"]).sub_ec.transform("mean") >= 1
    print("  DIAG10 late R3S %.4f -> TS2 %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), np.mean([r((D["ts_%d" % s] - D.sub_ec)[L]) for s in SEEDS])))
    print("  DM alone DIAG10:", [round(r(D["dm_%d" % s] - D.sub_ec), 4) for s in SEEDS])
    print("  DIAG10 high-EC days (rows %d): label %.3f  R3S %.3f  DM %.3f  TS2 %.3f" % (
        hi.sum(), D.sub_ec[hi].mean(), D.r3s_7[hi].mean(), D.dm_7[hi].mean(), D.ts_7[hi].mean()))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("\nTS2 decision:", "PASS" if allb and all(p < 0.025 for p in ps) and elok else "FAIL",
          "(rule %s, EL1 %s)" % (allb and all(p < 0.025 for p in ps), elok))


if __name__ == "__main__":
    main()
