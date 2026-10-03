# -*- coding: utf-8 -*-
"""EC stage-3 DC6: season index with SAME-FARM twin anchors (fixed before running;
2026-10-03 집 클로드).
Defect found in SE0 (6.211): DC4 anchors a pass-2 training day to the mean pass-1
record day of exact weather twins of BOTH farms, but F13/F47 record days of one
date differ by -8..+8 (C6.205, records advance per date with 1-2 records each).
DC6 = DC4 with the twin search (and the nearest-match fallback) restricted to the
same farm's pass-1 training days; z-scaling, threshold .05, isotonic, record-day
interpolation for query days unchanged.  On the full training set this moves test
days' season by median 3.4 (max 12.2) record days (dc6_same_farm_anchor_check_v1).
Model: DC5 R3 recipe (FS = FULL - day + season, BS = BASE - day + season), seeds
7/101/2024, all validators + EL1.  Baseline R3S = stored DC5 / EL1 OOF.
Rule (user's): all 3 seeds x 5 validators better and DIAG10 P(worse) < .025; EL1
better for all seeds also required.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_DC6_same_farm_anchor_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc5)
dc4, p3, core = dc5.dc4, dc5.p3, dc5.core
SEEDS = (7, 101, 2024)
CK = os.path.join(env.LOCAL, "dc6_ckpt")


def season_index_same_farm(train_days, query_days, wv):
    tr = train_days.copy()
    p1, p2 = tr[tr.day < 179], tr[tr.day >= 179]
    Z = {}
    for v in dc4.W:
        blk = wv[v]
        vals = blk.reindex(list(zip(p1.farm, p1.day))).values
        m, s = np.nanmean(vals), np.nanstd(vals)
        Z[v] = (blk - m) / (s if s > 0 else 1)
    ZZ = pd.concat(Z, axis=1)
    season = {(f, d): float(d) for f, d in zip(p1.farm, p1.day)}
    fit = {}
    for f in ("F13", "F47"):
        P1 = p1[p1.farm == f]
        A = ZZ.reindex(list(zip(P1.farm, P1.day))).values
        q = p2[p2.farm == f].sort_values("day")
        ax, ay = [], []
        for d in q.day:
            dist = np.sqrt(np.nanmean((A - ZZ.reindex([(f, d)]).values) ** 2, axis=1))
            if np.nanmin(dist) <= 0.05:
                ax.append(d); ay.append(float(P1.day.values[dist <= 0.05].mean()))
        ax, ay = np.array(ax), np.array(ay)
        if len(ax) < 2:
            ax = q.day.values
            ay = np.array([float(P1.day.values[np.nanargmin(np.sqrt(np.nanmean((A - ZZ.reindex([(f, d)]).values) ** 2, axis=1)))]) for d in ax])
            print("   %s: <2 same-farm exact-twin anchors -> nearest same-farm match" % f, flush=True)
        sm = IsotonicRegression(increasing=True).fit(ax, ay).predict(ax)
        for d in q.day:
            season[(f, d)] = float(np.interp(d, ax, sm))
        fit[f] = (ax, sm)
    out = np.full(len(query_days), np.nan)
    for i, (f, d) in enumerate(zip(query_days.farm, query_days.day)):
        if d < 179:
            t = p1[p1.farm == f].sort_values("day")
            out[i] = np.interp(d, t.day.values, t.day.values.astype(float))
        else:
            out[i] = np.interp(d, *fit[f])
    return season, out


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
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
        season, vq = season_index_same_farm(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
        vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["dc6_%d" % s] = dc5.r3(tr, va, s, FS, BS)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
    base = pd.concat([d5, e1[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]], ignore_index=True)
    O = P.merge(base, on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(O) == len(P), (len(O), len(P))
    O.to_csv(os.path.join(env.LOCAL, "ec3_DC6_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261003)
    allb = True
    print("\npooled RMSE R3S (DC4 anchors) -> DC6 (same-farm anchors)")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        G = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(G["r3s_%d" % s] - G.sub_ec), r(G["dc6_%d" % s] - G.sub_ec)
            if v != "EL1":
                allb &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["dc6_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    E = O[O.validator == "EL1"]
    elok = all(r(E["dc6_%d" % s] - E.sub_ec) < r(E["r3s_%d" % s] - E.sub_ec) for s in SEEDS)
    L = D.day >= 179
    print("  DIAG10 late R3S %.4f -> DC6 %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), np.mean([r((D["dc6_%d" % s] - D.sub_ec)[L]) for s in SEEDS])))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("\nDC6 decision:", "PASS" if allb and all(p < 0.025 for p in ps) and elok else "FAIL",
          "(rule %s, EL1 %s)" % (allb and all(p < 0.025 for p in ps), elok))


if __name__ == "__main__":
    main()
