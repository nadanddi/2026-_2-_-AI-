# -*- coding: utf-8 -*-
"""EC stage-2 PAR1: same-parity previous labelled EC as a feature on top of R3S
(fixed before running; 2026-10-02 집 클로드).

Basis: after the season fix the remaining late error is the high-EC source
state (5 days = 74% of late squared error).  C3: within pass 2 record parity
separates the two sources 15/15.  PAR0 (labels only): in pass 2 a day's EC
correlates .65 with the previous SAME-parity labelled day (<= 6 days) and .01
with the opposite-parity one.  Round 7's state feature (parity-blind past EC)
lost 35% on the leaderboard - hence a strict test.
Features (allowed: earlier train labels as features, user 10-01):
  par_ec  = daily mean EC of the nearest EARLIER labelled training day of the
            same greenhouse, same pass (<179 / >=179) and same record-day
            parity, at most 12 record days back (NaN otherwise)
  par_gap = its record-day distance (NaN otherwise)
Training rows use other training days only; validation rows use the fold's
training days only (held-out days and +-1 never feed the feature).
Model: R3S recipe (DC5: core.FULL/BASE with season instead of day) + the two
features in the ET and LGB/MLP inputs.  Baseline: R3S.  Seeds 7/101/2024.
Rule (user's): all 3 seeds x 5 validators better than R3S and DIAG10 P(worse)
< .025 per seed.  Also required for a positive conclusion (stated up front,
because round 7 showed label-state features can fail on the test):
EL1 eval-like late blocks better for all 3 seeds.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_PAR1_parity_prev_label_v1.py
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


def par_feats(days, labels):
    """days: iterable of (farm, day); labels: dict (farm, day) -> daily mean EC of usable training days."""
    out = []
    for f, d in days:
        best = None
        for g in range(2, 13, 2):
            x = d - g
            if (x >= 179) != (d >= 179):
                break
            if (f, x) in labels:
                best = (labels[(f, x)], g)
                break
        out.append(best if best else (np.nan, np.nan))
    return np.array(out, dtype=float)


def add_feats(tr, va, wv):
    tdays = tr[["farm", "day"]].drop_duplicates()
    vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
    season, vq = dc4.season_index(tdays, vdays, wv)
    tr = tr.copy(); va = va.copy()
    tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
    vdays["season"] = vq
    lab = tr.groupby(["farm", "day"]).sub_ec.mean().to_dict()
    tk = list(zip(tdays.farm, tdays.day))
    tf = par_feats(tk, lab)
    tmap = {k: v for k, v in zip(tk, tf)}
    tr["par_ec"] = [tmap[(f, d)][0] for f, d in zip(tr.farm, tr.day)]
    tr["par_gap"] = [tmap[(f, d)][1] for f, d in zip(tr.farm, tr.day)]
    vf = par_feats(list(zip(vdays.farm, vdays.day)), lab)
    vdays["par_ec"], vdays["par_gap"] = vf[:, 0], vf[:, 1]
    va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
    return tr, va


def run(folds, lab, lock, wv, FS, BS, FP, BP, tag):
    rec = []
    for name, i, vd in folds:
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = add_feats(lab[tr_m], lab[va_m], wv)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec", "cal", "par_gap"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["r3s_%d" % s] = dc5.r3(tr, va, s, FS, BS)
            frame["par_%d" % s] = dc5.r3(tr, va, s, FP, BP)
        rec.append(frame)
        print("%s %s/%d done (val days with par feature %.0f%%)" % (tag, name, i, 100 * frame.groupby(["farm", "day"]).par_gap.first().notna().mean()), flush=True)
    return pd.concat(rec, ignore_index=True)


def main():
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    cal = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_9_days.csv"))[["farm", "day", "cal"]]
    lab = lab.merge(cal, on=["farm", "day"], how="left")
    FS = [c for c in core.FULL if c != "day"] + ["season"]
    BS = [c for c in core.BASE if c != "day"] + ["season"]
    FP, BP = FS + ["par_ec", "par_gap"], BS + ["par_ec", "par_gap"]
    # eval-like folds (as EL1)
    el = []
    for f in ("F13", "F47"):
        days = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(days), 5):
            el.append(("EL1", len(el), {(f, int(d)) for d in days[k:k + 5]}))
    O = run(fds, lab, lock, wv, FS, BS, FP, BP, "main")
    E = run(el, lab, lock, wv, FS, BS, FP, BP, "EL1")
    O.to_csv(os.path.join(env.LOCAL, "ec2_PAR1_oof.csv"), index=False)
    E.to_csv(os.path.join(env.LOCAL, "ec2_PAR1_el1.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261002)
    allbetter = True
    print("\npooled RMSE R3S -> R3S + parity prev label")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12"):
        g = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(g["r3s_%d" % s] - g.sub_ec), r(g["par_%d" % s] - g.sub_ec)
            allbetter &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy()
    D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["par_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    elok = True
    for s in SEEDS:
        a, b = r(E["r3s_%d" % s] - E.sub_ec), r(E["par_%d" % s] - E.sub_ec)
        elok &= b < a
        print("  EL1 seed %d: R3S %.4f -> PAR %.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
    for nm, m in (("DIAG10 late", D.day >= 179), ("DIAG10 early", D.day < 179)):
        g = D[m]
        print("  %-13s R3S %.4f PAR %.4f" % (nm, np.mean([r(g["r3s_%d" % s] - g.sub_ec) for s in SEEDS]), np.mean([r(g["par_%d" % s] - g.sub_ec) for s in SEEDS])))
    ok = allbetter and all(p < 0.025 for p in ps) and elok
    print("\nPAR1 decision:", "PASS" if ok else "FAIL", "(rule %s, EL1 %s)" % (allbetter and all(p < 0.025 for p in ps), elok))


if __name__ == "__main__":
    main()
