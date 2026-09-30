# -*- coding: utf-8 -*-
"""Source-level "substrate minus air" gap: can operating habits / early-morning inputs
anticipate it?  (lab Claude, 2026-09-30)

Already known (not repeated): catalog 6b.3 (round-3 model: day offset not explained
by source, date, neighbour days or 18 same-day input summaries, R2 -0.03..-0.05) and
logs/day_offset48.log (48 continuous greenhouses: median R2 +0.002).  New here:

A. BETWEEN greenhouses (48 continuous records, training labels): is a greenhouse's
   mean daily gap (mean sub_temp - mean in_temp) related to its operating habits at
   night (hours 0-6 means of heating, thermal screen, vent, CO2 dosing, in_temp,
   in_hum, out_temp)?  Spearman over greenhouses; plus the same WITHIN greenhouses
   (daily anomalies from each greenhouse's mean, pooled).  Descriptive.

B'. F13/F47 with the CURRENT model: can the day-mean residual of G_C2 (out-of-fold,
   DIAG10 and EXT12) be predicted from that day's hours 0-6 inputs (means of sensors
   and actuators, in-out difference, the 23h->0h in_temp jump) and the source
   fingerprint at 06h (fp_features running means)?  Legal inputs only: same record,
   current day up to 06h and earlier.  Models: Ridge (primary) and a small LightGBM,
   cross-validated over days with the diagnostic 5-day-chunk folds.

PRE-SET RULE for B' (fixed before running): the direction is WORTH A CANDIDATE only
  if, for at least one of the two models, the out-of-fold R2 of the day residual has
  a day-bootstrap CI (Bonferroni for 2 models: 1.25-98.75 percentile) above 0 on BOTH
  DIAG10 and EXT12, AND the point R2 is > 0 separately for F13 and F47 on both.
  Otherwise the direction is CLOSED.  No correction is applied or scored here.

Output: logs/gap_source_v1.log
Run:  PYTHONPATH="" <python> -u gap_source_v1.py   (from 연구실/클로드/code)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "집", "클로드", "research"))
import env  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402

import common  # noqa: E402
from harness import load  # noqa: E402
from anal_q1_errors import diag_folds  # noqa: E402
import features_v4 as F4  # noqa: E402

NIGHT = ["act_heating", "act_thermal", "act_vent", "act_co2", "in_temp", "in_hum", "out_temp"]
EARLY = ["in_temp", "in_hum", "in_co2", "out_temp", "out_hum", "act_heating", "act_thermal", "act_vent",
         "act_circfan", "act_co2", "act_fog", "act_shade"]


def part_a():
    tX, ty, _ = common.load_raw()
    df = tX.merge(ty[["row_id", "sub_temp"]], on="row_id")
    df = df[~df.farm.isin(["F13", "F47"])]
    day = df.groupby(["farm", "day"]).agg(n=("sub_temp", "count"), y=("sub_temp", "mean"), t=("in_temp", "mean"))
    day = day[day.n >= 20]
    day["gap"] = day.y - day.t
    night = df[df.hour <= 6].groupby(["farm", "day"])[NIGHT].mean()
    day = day.join(night, how="inner")
    fm = day.groupby(level=0).mean()
    print("A. %d greenhouses, %d days.  Greenhouse mean gap: median %+.2f, IQR %+.2f..%+.2f, range %+.2f..%+.2f"
          % (len(fm), len(day), fm.gap.median(), fm.gap.quantile(.25), fm.gap.quantile(.75), fm.gap.min(), fm.gap.max()))
    print("   between greenhouses (Spearman, n = greenhouses with the column) | within greenhouses (daily anomalies)")
    for c in NIGHT:
        m = fm[[c, "gap"]].dropna()
        m = m[m[c].abs() > 0] if c.startswith("act_") else m
        rb = spearmanr(m[c], m.gap).correlation if len(m) > 8 else np.nan
        an = day[[c, "gap"]].dropna()
        an = an - an.groupby(level=0).transform("mean")
        rw = spearmanr(an[c], an.gap).correlation
        print("   night %-12s  between %+.2f (n=%2d) | within %+.2f (days %d)" % (c, rb, len(m), rw, len(an)))
    # how much of the day-to-day gap variance is between greenhouses?
    tot = ((day.gap - day.gap.mean()) ** 2).sum()
    btw = (day.gap.groupby(level=0).transform("mean") - day.gap.mean()).pow(2).sum()
    print("   share of daily-gap variance that is between greenhouses: %.0f%%" % (100 * btw / tot))


def early_features(lab):
    lab = lab.sort_values(["farm", "t"]).copy()
    lab["jump0"] = np.where(lab.hour == 0, lab.groupby("farm").in_temp.diff(), np.nan)
    e = lab[lab.hour <= 6].groupby(["farm", "day"])
    X = e[EARLY].mean().add_prefix("e_")
    X["e_in_out"] = X.e_in_temp - X.e_out_temp
    X["jump0"] = e.jump0.first()
    fp = F4.fp_features()
    fpc = [c for c in fp.columns if c.endswith("_tdm") or c.endswith("_h0")]
    f6 = lab[["row_id", "farm", "day", "hour"]].merge(fp[["row_id"] + fpc], on="row_id")
    X = X.join(f6[f6.hour == 6].set_index(["farm", "day"])[fpc])
    return X


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def part_b():
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (z["row_id"] == lab.row_id.values).all()
    t = lab.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    X = early_features(lab)
    folds = diag_folds(lab)
    fold_of = {(f, d): i for i, fd in enumerate(folds) for f in fd for d in fd[f]}
    verdict = {"ridge": [], "lgb": []}
    for split in ("DIAG10", "EXT12"):
        base = np.mean([z["%s__MASK__7" % split], z["%s__MASK__101" % split]], axis=0)
        cx = np.mean([z["%s__CODEX__726" % split], z["%s__CODEX__727" % split]], axis=0)
        pfn = np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % split).mean(0)
        pred = (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn
        d = lab.assign(e=pred - lab.sub_temp.values)[~np.isnan(pred)]
        yb = d.groupby(["farm", "day"]).e.mean()
        D = X.join(yb.rename("bias"), how="inner")
        fid = np.array([fold_of[k] for k in D.index])
        cols = [c for c in X.columns]
        print("B'. %s: %d days, day-residual SD %.3f" % (split, len(D), D.bias.std()))
        for name in ("ridge", "lgb"):
            oof = np.full(len(D), np.nan)
            for k in np.unique(fid):
                tr, va = fid != k, fid == k
                if name == "ridge":
                    m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                      Ridge(alpha=10.0))
                else:
                    m = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=7,
                                      verbose=-1, n_jobs=4)
                m.fit(D[cols][tr], D.bias[tr])
                oof[va] = m.predict(D[cols][va])
            y = D.bias.values
            rng = np.random.default_rng(0)
            bs = [r2(y[i], oof[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
            lo, hi = np.percentile(bs, [1.25, 98.75])
            farms = D.index.get_level_values(0)
            pf = {f: r2(y[farms == f], oof[farms == f]) for f in ("F13", "F47")}
            ok = lo > 0 and pf["F13"] > 0 and pf["F47"] > 0
            verdict[name].append(ok)
            print("   %-5s OOF R2 %+.3f  CI97.5 [%+.3f, %+.3f] | F13 %+.3f  F47 %+.3f | pass %s"
                  % (name, r2(y, oof), lo, hi, pf["F13"], pf["F47"], ok))
    worth = any(all(v) for v in verdict.values())
    print("B' verdict: %s" % ("WORTH A CANDIDATE" if worth else "CLOSED"))


if __name__ == "__main__":
    part_a()
    print("=" * 100)
    part_b()
