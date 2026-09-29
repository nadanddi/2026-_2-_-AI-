# -*- coding: utf-8 -*-
"""Held-out permutation importance on the geometry folds.

Usage:  PYTHONPATH="" python feat_perm.py temp|ec

Train-set importances (LightGBM `gain`, ExtraTrees impurity) are not used
anywhere in this study: they reward whatever the model happened to split on,
including columns that only memorise the training days.  Here the model is fit
on the fold's training rows and every column is shuffled *inside the held-out
rows only*, so the number reported is "how much held-out RMSE gets worse when
this column stops being informative".

Two shuffle schemes are reported, because the rows are hourly and heavily
autocorrelated:
  row  -- shuffle the column across held-out rows (breaks everything)
  day  -- shuffle whole greenhouse-days of the column (keeps the within-day
          shape, breaks only the day-to-day level).  For sub_ec, where ~95% of
          the variance is the daily level, `day` is the meaningful one.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

import env  # noqa: F401  MUST be first
from harness import load, folds, views
from common import split_mask, rmse
import feat_lib as L

SEED = 7
REPEATS = 3


def _perm_row(rng, x):
    return rng.permutation(x)


def _perm_day(rng, x, daykey):
    """Shuffle the column between greenhouse-days, keeping each day's shape."""
    out = np.array(x, float, copy=True)
    uniq = np.unique(daykey)
    order = rng.permutation(len(uniq))
    pos = {d: np.where(daykey == d)[0] for d in uniq}
    for i, d in enumerate(uniq):
        src = pos[uniq[order[i]]]
        dst = pos[d]
        n = min(len(src), len(dst))
        out[dst[:n]] = np.asarray(x, float)[src[:n]]
    return out


def run(target):
    panel, lab_t, lab_e = load()
    v = views(panel)
    if target == "temp":
        lab, tgt, cols, model = lab_t, "sub_temp", v["temp"], L.TEMP_MODEL
    else:
        lab, tgt, cols, model = lab_e, "sub_ec", v["ec"], L.EC_MODEL

    acc = {k: {c: [] for c in cols} for k in ("row", "day")}
    basel = []
    for kind in ("A", "B"):
        for fi, fd in enumerate(folds(kind)):
            t0 = time.time()
            trm, vam = split_mask(lab, fd)
            tr, va = lab[trm], lab[vam]
            est = model(SEED)
            est.fit(tr[cols], tr[tgt].values)
            Xv = va[cols].reset_index(drop=True)
            yv = va[tgt].values
            b = rmse(est.predict(Xv), yv)
            basel.append(b)
            dk = (va.farm.astype(str) + "_" + va.day.astype(str)).values
            for c in cols:
                orig = Xv[c].values.copy()
                for scheme in ("row", "day"):
                    ds = []
                    for r in range(REPEATS):
                        rng = np.random.RandomState(1000 * fi + 7 * r
                                                    + (0 if kind == "A" else 5))
                        Xv[c] = (_perm_row(rng, orig) if scheme == "row"
                                 else _perm_day(rng, orig, dk))
                        ds.append(rmse(est.predict(Xv), yv) - b)
                    acc[scheme][c].append(float(np.mean(ds)))
                Xv[c] = orig
            print("  %s fold%d base %.4f  (%.0fs)" % (kind, fi, b, time.time() - t0),
                  flush=True)

    rec = []
    for c in cols:
        rec.append(dict(col=c, group=L.group_of(c),
                        imp_row=float(np.mean(acc["row"][c])),
                        imp_row_sd=float(np.std(acc["row"][c])),
                        imp_day=float(np.mean(acc["day"][c])),
                        imp_day_sd=float(np.std(acc["day"][c])),
                        pos_frac=float(np.mean(np.array(acc["day"][c]) > 0))))
    df = pd.DataFrame(rec).sort_values("imp_day", ascending=False)
    path = env.LOCAL + "/perm_%s.csv" % target
    df.to_csv(path, index=False)
    print("\nmean fold baseline RMSE %.4f over %d fits" % (np.mean(basel), len(basel)))
    print("wrote", path)

    pd.set_option("display.width", 200)
    print("\n--- top 25 by day-shuffle importance ---")
    print(df.head(25).to_string(index=False, float_format=lambda x: "%+.4f" % x))
    print("\n--- bottom 20 (negative = model does better without the real values) ---")
    print(df.tail(20).to_string(index=False, float_format=lambda x: "%+.4f" % x))
    print("\n--- by group (sum / mean of day-shuffle importance) ---")
    gg = df.groupby("group").imp_day.agg(["sum", "mean", "count"]).sort_values(
        "sum", ascending=False)
    print(gg.to_string(float_format=lambda x: "%+.4f" % x))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "ec")
