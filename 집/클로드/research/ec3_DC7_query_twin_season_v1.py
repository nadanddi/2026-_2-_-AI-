# -*- coding: utf-8 -*-
"""EC stage-3 DC7: evaluation/validation-day season from the day's OWN weather twin,
causal per hour (fixed before running; 2026-10-04 집 클로드).
Basis: NX0/SE0 - query days get their season only by record-day interpolation; at the
edge of pass 2 it clamps (F13/241,243: used 121, exact own-weather twin 148-149, both
under-predicted by .40/.68), and on evaluation days exact twins disagree with the
interpolation by up to 10-15 record days (F47/204-206, F13/239).
Season of a query row (farm f, record d, hour h):
  if h >= 5 and the outdoor weather (4 channels, DC4 z-scaling) of hours 0..h matches
  some pass-1 TRAINING day (any farm, as DC4 anchors) exactly (z-RMSE over hours 0..h
  <= .05): season = mean pass-1 record day of the matching days;
  else: DC4 interpolation (unchanged).  Pass-1 query days keep DC4 (own record day).
Legal: hours 0..h of the same greenhouse's current record + training inputs only.
Training rows unchanged (DC4).  Model: DC5 R3 recipe, seeds 7/101/2024.
Judge (EC rule 2026-10-04, 6.247): every seed x DIAG10/A/B better, DIAG10 P(worse) <
.025; guard: EL1 or DIAG10 pass-2 rows worse by >= 2 % -> HOLD; EXT reported.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_DC7_query_twin_season_v1.py
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
dc5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc5)
dc4, p3, core = dc5.dc4, dc5.p3, dc5.core
SEEDS = (7, 101, 2024)
CK = os.path.join(env.LOCAL, "dc7_ckpt")
HMIN, THR = 5, 0.05


def zscale(wv, p1):
    Z = {}
    for v in dc4.W:
        blk = wv[v]; vals = blk.reindex(list(zip(p1.farm, p1.day))).values
        m, s = np.nanmean(vals), np.nanstd(vals)
        Z[v] = (blk - m) / (s if s > 0 else 1)
    return pd.concat(Z, axis=1)


def row_season(va, vq_day, tdays, wv):
    """va: rows (farm, day, hour); vq_day: dict (farm,day)->DC4 interpolated season."""
    p1 = tdays[tdays.day < 179]
    ZZ = zscale(wv, p1)
    A = ZZ.reindex(list(zip(p1.farm, p1.day)))
    hours = A.columns.get_level_values(1)
    Av = A.values
    out = np.array([vq_day[(f, d)] for f, d in zip(va.farm, va.day)], dtype=float)
    for (f, d), idx in va.groupby(["farm", "day"]).groups.items():
        if d < 179 or (f, d) not in ZZ.index:
            continue
        b = ZZ.loc[(f, d)].values
        for ii in idx:
            h = int(va.loc[ii, "hour"])
            if h < HMIN:
                continue
            cols = np.asarray(hours <= h)
            diff = (Av[:, cols] - b[cols]) ** 2
            dist = np.sqrt(np.nanmean(diff, axis=1))
            ok = dist <= THR
            if ok.any():
                out[va.index.get_loc(ii)] = float(p1.day.values[ok].mean())
    return out


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"] + dc4.W)
    te = te[te.row_id.str[:3].isin(["F13", "F47"])]
    wv = dc4.weather_vectors(pd.concat([full[["row_id"] + dc4.W], te]))   # test weather only for test-day lookups
    FS = [c for c in core.FULL if c != "day"] + ["season"]
    BS = [c for c in core.BASE if c != "day"] + ["season"]
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
        vq_day = dict(zip(zip(vdays.farm, vdays.day), vq))
        va["season_dc4"] = [vq_day[(f, d)] for f, d in zip(va.farm, va.day)]
        va["season"] = row_season(va, vq_day, tdays, wv)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec", "season_dc4", "season"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["dc7_%d" % s] = dc5.r3(tr, va, s, FS, BS)
        frame.to_csv(path, index=False)
        print("%s/%d done (rows with twin season %.2f)" % (name, i, (frame.season != frame.season_dc4).mean()), flush=True)
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
    base = pd.concat([d5, e1[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]], ignore_index=True)
    O = P.merge(base, on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(O) == len(P), (len(O), len(P))
    O.to_csv(os.path.join(env.LOCAL, "ec3_DC7_all.csv"), index=False)
    print("rows with twin season by validator:", O.assign(ch=O.season != O.season_dc4).groupby("validator").ch.mean().round(3).to_dict())
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261004)
    judge = True
    print("\npooled RMSE R3S -> DC7 (judge: DIAG10/A/B; others reported)")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        G = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(G["r3s_%d" % s] - G.sub_ec), r(G["dc7_%d" % s] - G.sub_ec)
            if v in ("DIAG10", "A", "B"):
                judge &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["dc7_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    L = D.day >= 179
    E = O[O.validator == "EL1"]
    guard = []
    for s in SEEDS:
        guard.append(100 * (r(E["dc7_%d" % s] - E.sub_ec) / r(E["r3s_%d" % s] - E.sub_ec) - 1))
        guard.append(100 * (r((D["dc7_%d" % s] - D.sub_ec)[L]) / r((D["r3s_%d" % s] - D.sub_ec)[L]) - 1))
    print("  DIAG10 late R3S %.4f -> DC7 %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), np.mean([r((D["dc7_%d" % s] - D.sub_ec)[L]) for s in SEEDS])))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("  guard changes (EL1, DIAG10-late per seed, %%): %s" % [round(x, 2) for x in guard])
    ok = judge and all(p < 0.025 for p in ps)
    hold = any(x >= 2.0 for x in guard)
    print("\nDC7 decision:", ("HOLD (guard)" if hold else "PASS") if ok else "FAIL", "(judge %s, guard trip %s)" % (ok, hold))


if __name__ == "__main__":
    main()
