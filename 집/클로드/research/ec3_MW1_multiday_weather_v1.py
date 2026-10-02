# -*- coding: utf-8 -*-
"""EC stage-3 MW1: multi-day past-weather features on top of R3S (season), fixed
before running; 2026-10-03 집 클로드.
Literature (lit_survey_2026-10-03 A/B): irrigation happens only in daytime and is
cut on dark/cloudy days (solar-proportional or manual), so runs of low radiation
can raise substrate EC; timer irrigation can do the opposite.  MW0 (descriptive):
on DIAG10 late days the R3S day residual vs share of low-radiation days among the
previous 6 record days rho = -.38 (n=46), F13/F47 opposite overall, min p .0014
over 64 tests (not Bonferroni-significant).
Features (same greenhouse, PREVIOUS record days of the same pass only, train_X
inputs only -> legal; at test time previous test inputs of the same greenhouse
would also be allowed):
  rad_prev4, rad_prev6 (mean daily radiation sum), lowrad_prev6 (share of days
  below the pass-wise 30% radiation quantile of train days), tout_prev4, heat_prev4
Model: DC5 R3S recipe + these 5 columns in ET and LGB/MLP inputs; baseline R3S
(DC5 / EL1 stored OOF, same recipe).  Seeds 7/101/2024.
Rule (user's): all seeds x 5 validators better than R3S and DIAG10 P(worse) <
.025; also required: EL1 eval-like late blocks better for all seeds.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_MW1_multiday_weather_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc5)
dc4, p3, core = dc5.dc4, dc5.p3, dc5.core
SEEDS = (7, 101, 2024)
CK = os.path.join(env.LOCAL, "mw1_ckpt")
MW = ["rad_prev4", "rad_prev6", "lowrad_prev6", "tout_prev4", "heat_prev4"]


def mw_table():
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
    p = X.row_id.str.split("_", expand=True); X["farm"], X["day"] = p[0], p[1].astype(int)
    X = X[X.farm.isin(["F13", "F47"])]
    d = X.groupby(["farm", "day"]).agg(rad=("out_rad", "sum"), tout=("out_temp", "mean"), heat=("act_heating", "mean")).reset_index()
    d = d.sort_values(["farm", "day"]); d["pas"] = (d.day >= 179).astype(int)
    q30 = d.groupby("pas").rad.quantile(0.3).to_dict()
    d["low"] = (d.rad < d.pas.map(q30)).astype(float)
    g = d.groupby(["farm", "pas"])
    d["rad_prev4"] = g.rad.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    d["rad_prev6"] = g.rad.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
    d["lowrad_prev6"] = g.low.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
    d["tout_prev4"] = g.tout.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    d["heat_prev4"] = g.heat.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    return d[["farm", "day"] + MW]


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    lab = lab.merge(mw_table(), on=["farm", "day"], how="left")
    FS = [c for c in core.FULL if c != "day"] + ["season"]
    BS = [c for c in core.BASE if c != "day"] + ["season"]
    FM, BM = FS + MW, BS + MW
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
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["mw_%d" % s] = dc5.r3(tr, va, s, FM, BM)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
    base = pd.concat([d5, e1[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]], ignore_index=True)
    O = P.merge(base, on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(O) == len(P), (len(O), len(P))
    O.to_csv(os.path.join(env.LOCAL, "ec3_MW1_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261003)
    allb = True
    print("\npooled RMSE R3S -> R3S + multi-day weather")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        g = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(g["r3s_%d" % s] - g.sub_ec), r(g["mw_%d" % s] - g.sub_ec)
            if v != "EL1":
                allb &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["mw_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    E = O[O.validator == "EL1"]
    elok = all(r(E["mw_%d" % s] - E.sub_ec) < r(E["r3s_%d" % s] - E.sub_ec) for s in SEEDS)
    L = D.day >= 179
    print("  DIAG10 late R3S %.4f -> MW %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), np.mean([r((D["mw_%d" % s] - D.sub_ec)[L]) for s in SEEDS])))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("\nMW1 decision:", "PASS" if allb and all(p < 0.025 for p in ps) and elok else "FAIL",
          "(rule %s, EL1 %s)" % (allb and all(p < 0.025 for p in ps), elok))


if __name__ == "__main__":
    main()
