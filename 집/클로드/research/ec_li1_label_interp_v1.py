# -*- coding: utf-8 -*-
"""EC-LI1: do labelled train days on BOTH sides of a test block, joined along
the same-source (alternate-day) chain, recover the daily EC level better than
EC v2?  Diagnostic, 2026-10-01 집 클로드 (user: "온도 말고 EC로 틀어보자").

Why (data-generation forensics)
-------------------------------
* EC v2 error is 86.8% day level (catalog 6.139/6.140); removing the day level
  leaves ~0.075 (6.126).  The EC leader's 0.0594 is below that, so the leader
  must know the daily level almost exactly.
* Every test block (5 or 10 days) has labelled days on both sides at record
  distance 2 (one unlabelled day between; LS1 / this script's geometry print).
* EC is a slowly drifting state within one source (6.3) and the record
  alternates two sources (6.2, 6.4, 1.14).  Labels just before and after a block
  show it plainly (e.g. F47 212..216 = .52 .70 .53 .71 .56, after the block
  229..231 = 1.79 .46 2.18).
* User (2026-10-01): public train_y may be used as features for test rows;
  earlier-day labels by default, later-day (future-direction) use flagged.
  This script uses both directions -> it is flagged as such.
* Not tried before: 6.49 PAR2 used d+-2 inside 5-day DIAG blocks (mostly
  hidden), 6.141 used past-only public EC mean, 6b.28/6b.38 were temperature.

Design (fixed before running)
-----------------------------
Daily mean EC of F13/F47 from train_y.  Codex's 40 final-lock days are treated
as UNLABELLED everywhere (never scored, never anchors) to keep that lock sealed.
Pseudo test blocks copy the test geometry: scored days [a, b], length L in
{5, 10}; days a-1 and b+1 also hidden (the unlabelled gap); anchors = labelled,
non-locked days outside [a-1, b+1].  Every a with all of [a, b] labelled and
non-locked is used.

Predictors of day d's mean EC:
  V2   EC v2 DIAG10 OOF daily mean (Codex phase 3, oof_predictions.csv)
  I1   parity interpolation: nearest anchor <= a-2 and nearest >= b+2 with
       (anchor - d) even, linear in day index; one side only -> that value
  I0   same without parity (nearest anchors either side)
  I2   parity, flat mean of the two anchors
Also the row-level score proxy: v2 hourly shape kept, level swapped:
  row = v2_row - V2_daymean + I1   (vs plain v2 rows).

Decision rule (GO = worth building a pre-registered model on it):
  GO if, for I1 vs V2 day-level squared error on the same scored days,
  * I1 better in F13 and in F47, for L=5 and for L=10 (4 cells), and
  * pooled paired 5-day-cluster bootstrap (5000) P(I1 worse) < 0.025.
  Otherwise STOP for this representation.  Late segment (day >= 179) and the
  row-level proxy are reported descriptively (few late pseudo blocks).
This is a diagnostic; it adds 0 to the Bonferroni hypothesis count and
produces no submission.

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_li1_label_interp_v1.py
"""
import env  # noqa: F401
import json
import os

import numpy as np
import pandas as pd

D = env.DATA
LOCK = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
OOF = os.path.join(env.ROOT, u"집", u"코덱스", "local", "ec_restart_phase3_20261001_v1", "oof_predictions.csv")
FARMS = ("F13", "F47")
RNG = np.random.default_rng(20261001)


def split_id(df):
    p = df.row_id.str.split("_", expand=True)
    df["farm"] = p[0]
    df["day"] = p[1].astype(int)
    df["hour"] = p[2].astype(int)
    return df


def interp(anch, d, a, b, parity):
    """anch: dict day->ec of usable anchors. Returns (I value, flat mean)."""
    left = [x for x in anch if x <= a - 2 and (not parity or (x - d) % 2 == 0)]
    right = [x for x in anch if x >= b + 2 and (not parity or (x - d) % 2 == 0)]
    L = max(left) if left else None
    R = min(right) if right else None
    if L is None and R is None:
        return np.nan, np.nan
    if L is None:
        return anch[R], anch[R]
    if R is None:
        return anch[L], anch[L]
    w = (d - L) / (R - L)
    return (1 - w) * anch[L] + w * anch[R], 0.5 * (anch[L] + anch[R])


def main():
    y = split_id(pd.read_csv(os.path.join(D, "train_y.csv")))
    y = y[y.farm.isin(FARMS) & y.sub_ec.notna()]
    lock = {(s["farm"], int(s["day"])) for s in json.load(open(LOCK, encoding="utf-8"))["selected"]}
    print("locked days excluded:", len(lock))
    ec = y.groupby(["farm", "day"]).sub_ec.mean()
    oof = pd.read_csv(OOF, encoding="utf-8-sig")
    oof = oof[oof.validator == "DIAG10"]
    v2d = oof.groupby(["farm", "day"]).v2.mean()

    # test geometry print
    t = split_id(pd.read_csv(os.path.join(D, "test_X.csv")))
    for f in FARMS:
        td = sorted(t[t.farm == f].day.unique())
        print(f, "test days", td[0], "..", td[-1], "n", len(td))

    rec = []
    for f in FARMS:
        lab = {d: v for (ff, d), v in ec.items() if ff == f and (ff, d) not in lock}
        for L in (5, 10):
            for a in range(min(lab), max(lab) + 1):
                b = a + L - 1
                if not all(d in lab for d in range(a, b + 1)):
                    continue
                anch = {d: v for d, v in lab.items() if d < a - 1 or d > b + 1}
                for d in range(a, b + 1):
                    if (f, d) not in v2d.index:
                        continue
                    i1, i2 = interp(anch, d, a, b, True)
                    i0, _ = interp(anch, d, a, b, False)
                    rec.append(dict(farm=f, L=L, a=a, day=d, y=lab[d], V2=v2d[(f, d)],
                                    I1=i1, I0=i0, I2=i2))
    R = pd.DataFrame(rec).dropna()
    R["late"] = R.day >= 179
    print("pseudo (block, day) rows:", len(R), " distinct days:", R[["farm", "day"]].drop_duplicates().shape[0])

    def rm(s):
        return float(np.sqrt(np.mean(s ** 2)))

    print("\nday-level RMSE  (n = block-day occurrences)")
    print("%-14s %6s %8s %8s %8s %8s" % ("cell", "n", "V2", "I1", "I0", "I2"))
    cells = []
    for f in FARMS:
        for L in (5, 10):
            g = R[(R.farm == f) & (R.L == L)]
            r = [rm(g[c] - g.y) for c in ("V2", "I1", "I0", "I2")]
            cells.append(r[1] < r[0])
            print("%-14s %6d %8.4f %8.4f %8.4f %8.4f" % ("%s L=%d" % (f, L), len(g), *r))
    for nm, g in (("all", R), ("early<179", R[~R.late]), ("late>=179", R[R.late])):
        if len(g):
            print("%-14s %6d %8.4f %8.4f %8.4f %8.4f" % (nm, len(g), *[rm(g[c] - g.y) for c in ("V2", "I1", "I0", "I2")]))

    # paired cluster bootstrap on per-(farm, day) mean squared-error difference
    R["d"] = (R.I1 - R.y) ** 2 - (R.V2 - R.y) ** 2
    per = R.groupby(["farm", "day"]).d.mean().reset_index()
    per["cl"] = per.farm + "_" + (per.day // 5).astype(str)
    cl = per.groupby("cl").d.agg(["sum", "count"])
    s, n = cl["sum"].values, cl["count"].values
    bs = []
    for _ in range(5000):
        k = RNG.integers(0, len(s), len(s))
        bs.append(s[k].sum() / n[k].sum())
    bs = np.array(bs)
    p_worse = float((bs >= 0).mean())
    print("\nmean dMSE (I1 - V2) per day: %.5f  95%% [%.5f, %.5f]  P(I1 worse) %.4f  clusters %d"
          % (per.d.mean(), np.quantile(bs, .025), np.quantile(bs, .975), p_worse, len(s)))

    # row-level proxy: keep v2 hourly shape, swap level
    o = oof.merge(R.groupby(["farm", "day"])[["I1"]].mean().reset_index(), on=["farm", "day"])
    o["v2d"] = o.groupby(["farm", "day"]).v2.transform("mean")
    o["sw"] = o.v2 - o.v2d + o.I1
    o["orc"] = o.v2 - o.v2d + o.groupby(["farm", "day"]).sub_ec.transform("mean")
    print("row-level on those days: v2 %.4f  v2-shape+I1-level %.4f  v2-shape+true-level %.4f  (rows %d)"
          % (rm(o.v2 - o.sub_ec), rm(o.sw - o.sub_ec), rm(o.orc - o.sub_ec), len(o)))
    for nm, g in (("early", o[o.day < 179]), ("late", o[o.day >= 179])):
        if len(g):
            print("   %-5s v2 %.4f  swap %.4f  oracle %.4f  rows %d"
                  % (nm, rm(g.v2 - g.sub_ec), rm(g.sw - g.sub_ec), rm(g.orc - g.sub_ec), len(g)))

    go = all(cells) and p_worse < 0.025
    print("\ncells I1<V2:", cells, " -> %s" % ("GO" if go else "STOP"))


if __name__ == "__main__":
    main()
