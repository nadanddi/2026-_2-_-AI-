# -*- coding: utf-8 -*-
"""EC stage-2 DC3: train-only, explainable version of DC2 (fixed before running;
2026-10-02 집 클로드).  DC2 PASSED with a day->calendar map whose calendar came
from deep_cal_9 (built with test inputs in the weather matching).  Here the
season index is rebuilt from TRAINING inputs of the fold only:
  1. pass 1 (record day < 179) is in calendar order -> season = own record day
  2. each pass-2 training day (>= 179) is matched to the pass-1 training day
     (either greenhouse; training data) with the closest 24-h outdoor weather
     vector (out_temp, out_hum, out_rad, out_wspd, z-scored, RMSE) -> season =
     that day's record day
  3. pass 2 is also in calendar order -> isotonic (non-decreasing) fit of the
     pass-2 training days' season over record day, per greenhouse
  4. validation days (and later test days) never use their own inputs: season
     = linear interpolation of the fitted training values by record day within
     the same pass and greenhouse.
Model / folds / seeds / rule exactly as DC2: Codex phase-3 full ET, core.FULL
(with day) vs FULL - day + season; 3 seeds x 5 validators all better and
DIAG10 P(worse) < .025 per seed.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_DC3_season_index_trainonly_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

P3 = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "ec_restart_phase3_20261001_v1")
sys.path.insert(0, P3)
spec = importlib.util.spec_from_file_location("p3_readonly", os.path.join(P3, "run_benchmark.py"))
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)
core = p3.core
SEEDS = (7, 101, 2024)
W = ["out_temp", "out_hum", "out_rad", "out_wspd"]


def weather_vectors(full):
    a = full.copy()
    p = a.row_id.str.split("_", expand=True)
    a["farm"], a["day"], a["hour"] = p[0], p[1].astype(int), p[2].astype(int)
    piv = a.pivot_table(index=["farm", "day"], columns="hour", values=W)
    return piv


def season_index(train_days, query_days, wv):
    """train_days / query_days: DataFrames farm, day.  Returns season for train rows (dict) and query (array)."""
    tr = train_days.copy()
    p1 = tr[tr.day < 179]
    p2 = tr[tr.day >= 179]
    X1 = wv.reindex(list(zip(p1.farm, p1.day)))
    mu, sd = X1.stack(future_stack=True).groupby(level=1).mean() if False else (None, None)
    # z-score each weather variable over pass-1 training days
    Z = {}
    for v in W:
        blk = wv[v]
        m, s = np.nanmean(blk.reindex(list(zip(p1.farm, p1.day))).values), np.nanstd(blk.reindex(list(zip(p1.farm, p1.day))).values)
        Z[v] = (blk - m) / (s if s > 0 else 1)
    ZZ = pd.concat(Z, axis=1)
    A = ZZ.reindex(list(zip(p1.farm, p1.day))).values
    season = {(f, d): float(d) for f, d in zip(p1.farm, p1.day)}
    fit = {}
    for f in ("F13", "F47"):
        q = p2[p2.farm == f].sort_values("day")
        raw = []
        for d in q.day:
            b = ZZ.reindex([(f, d)]).values
            dist = np.sqrt(np.nanmean((A - b) ** 2, axis=1))
            raw.append(float(p1.day.values[np.nanargmin(dist)]))
        if len(q):
            iso = IsotonicRegression(increasing=True).fit(q.day.values, raw)
            sm = iso.predict(q.day.values)
            for d, v in zip(q.day, sm):
                season[(f, d)] = float(v)
            fit[f] = (q.day.values, sm)
    out = np.full(len(query_days), np.nan)
    for i, (f, d) in enumerate(zip(query_days.farm, query_days.day)):
        if d < 179:
            t = p1[p1.farm == f].sort_values("day")
            out[i] = np.interp(d, t.day.values, t.day.values.astype(float))
        else:
            x, y = fit[f]
            out[i] = np.interp(d, x, y)
    return season, out


def main():
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = weather_vectors(full)
    cal = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_9_days.csv"))[["farm", "day", "cal"]]
    lab = lab.merge(cal, on=["farm", "day"], how="left")
    F = list(core.FULL)
    FS = [c for c in F if c != "day"] + ["season"]
    rec = []
    for name, i, vd in fds:
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates()
        vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
        vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec", "cal", "season"]].copy()
        frame["validator"], frame["fold"] = name, i
        for s in SEEDS:
            for tag, cols in (("base", F), ("seas", FS)):
                m = core.et(s)
                m.fit(tr[cols], tr.sub_ec.to_numpy(float))
                frame["%s_%d" % (tag, s)] = p3.final(m.predict(va[cols]), tr, va)
        rec.append(frame)
        vd_ = va.drop_duplicates(["farm", "day"])
        print("%s/%d done (val days %d, corr(season, deep_cal) %.3f)" % (name, i, len(vdays),
              np.corrcoef(vd_.season, vd_.cal)[0, 1]), flush=True)
    O = pd.concat(rec, ignore_index=True)
    O.to_csv(os.path.join(env.LOCAL, "ec2_DC3_oof.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261002)
    allbetter = True
    print("\npooled RMSE base -> season index (train-only)")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12"):
        g = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(g["base_%d" % s] - g.sub_ec), r(g["seas_%d" % s] - g.sub_ec)
            allbetter &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy()
    D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["seas_%d" % s] - D.sub_ec) ** 2 - (D["base_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    for nm, m in (("late>=179", D.day >= 179), ("early", D.day < 179), ("late&cal<70", (D.day >= 179) & (D.cal < 70))):
        g = D[m]
        print("  DIAG10 %-12s base %.4f season %.4f" % (nm, np.mean([r(g["base_%d" % s] - g.sub_ec) for s in SEEDS]),
                                                       np.mean([r(g["seas_%d" % s] - g.sub_ec) for s in SEEDS])))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("\nDC3 decision:", "PASS" if allbetter and all(p < 0.025 for p in ps) else "FAIL")


if __name__ == "__main__":
    main()
