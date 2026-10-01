# -*- coding: utf-8 -*-
"""EC stage-2 DC2: replace the record index `day` by a calendar position read
from a day->calendar MAP (fixed before running; 2026-10-02 집 클로드).

DC1 (ec2_DC1_day_vs_season_v1.log): full ET with the weather-matched calendar
instead of `day` -> DIAG10 .2285 -> .1813, late .403 -> .222, late&early-cal
bias +.324 -> +.004; but that calendar is not a legal feature as such.

Legal form tested here: the record visits the calendar twice (catalog 1.12),
each pass in calendar order.  The map is built ONLY from training days of the
fold: for each greenhouse and each pass (day < 179 / >= 179), (record day ->
calendar) pairs of training days, linearly interpolated in record day.
Validation days get the INTERPOLATED value (their own calendar never used);
training rows use their own calendar.  At test time each test row would use
only its own id (day) through the map built from all training days.
Calendar values come from deep_cal_9 (built from inputs only).  Caveat: that
table was built with test inputs included in the weather matching; a final
version must rebuild the training-day calendar from training inputs only.

Model: Codex phase-3 full ET (core.et, core.FULL, same folds / lock purge /
shrink + clip), seeds 7, 101, 2024.  Variants: BASE (FULL with day) vs
CALMAP (FULL - day + calmap).  Rule (user's): CALMAP better on all 3 seeds x
5 validators and DIAG10 P(worse) < .025 per seed (farm x 5-day blocks,
20,000).  PASS -> hand to Codex for R3/TabPFN (v2-type) integration.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_DC2_calmap_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

P3 = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "ec_restart_phase3_20261001_v1")
sys.path.insert(0, P3)
spec = importlib.util.spec_from_file_location("p3_readonly", os.path.join(P3, "run_benchmark.py"))
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)
core = p3.core
SEEDS = (7, 101, 2024)


def calmap(train_days, query_days):
    """train_days: DataFrame farm, day, cal (training days only).
    query_days: DataFrame farm, day -> interpolated cal within the same pass."""
    out = np.full(len(query_days), np.nan)
    for i, (f, d) in enumerate(zip(query_days.farm, query_days.day)):
        late = d >= 179
        t = train_days[(train_days.farm == f) & ((train_days.day >= 179) == late)].sort_values("day")
        out[i] = np.interp(d, t.day.values, t.cal.values)
    return out


def main():
    raw, full, lab, lock, signatures, fds = p3.prepare()
    cal = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_9_days.csv"))[["farm", "day", "cal"]]
    lab = lab.merge(cal, on=["farm", "day"], how="left")
    F = list(core.FULL)
    FC = [c for c in F if c != "day"] + ["calm"]
    rec = []
    for name, i, vd in fds:
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day", "cal"]].drop_duplicates()
        vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        vdays["calm"] = calmap(tdays, vdays)
        tr["calm"] = tr.cal
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec", "cal", "calm"]].copy()
        frame["validator"], frame["fold"] = name, i
        for s in SEEDS:
            for tag, cols in (("base", F), ("calm", FC)):
                m = core.et(s)
                m.fit(tr[cols], tr.sub_ec.to_numpy(float))
                frame["%s_%d" % (tag, s)] = p3.final(m.predict(va[cols]), tr, va)
        rec.append(frame)
        print("%s/%d done (val days %d, map abs err %.1f)" % (name, i, len(vdays),
              np.nanmean(np.abs(va.drop_duplicates(["farm", "day"]).calm - va.drop_duplicates(["farm", "day"]).cal))), flush=True)
    O = pd.concat(rec, ignore_index=True)
    O.to_csv(os.path.join(env.LOCAL, "ec2_DC2_oof.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261002)
    allbetter = True
    print("\npooled RMSE base -> calmap")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12"):
        g = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(g["base_%d" % s] - g.sub_ec), r(g["calm_%d" % s] - g.sub_ec)
            allbetter &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy()
    D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["calm_%d" % s] - D.sub_ec) ** 2 - (D["base_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    for nm, m in (("late>=179", D.day >= 179), ("early", D.day < 179), ("late&cal<70", (D.day >= 179) & (D.cal < 70))):
        g = D[m]
        print("  DIAG10 %-12s base %.4f calmap %.4f" % (nm, np.mean([r(g["base_%d" % s] - g.sub_ec) for s in SEEDS]),
                                                       np.mean([r(g["calm_%d" % s] - g.sub_ec) for s in SEEDS])))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("\nDC2 decision:", "PASS" if allbetter and all(p < 0.025 for p in ps) else "FAIL")


if __name__ == "__main__":
    main()
