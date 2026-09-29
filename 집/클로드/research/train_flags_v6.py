# -*- coding: utf-8 -*-
"""Training-row weights for round 5, rebuilt after the three audits
(research/감사결과_2026-09-26.md).

Only TRAINING rows of F13/F47 get weights.  Nothing here reads test_X (the
classifier-based day score used test inputs, including the next hour, so
earlier test predictions depended on later test inputs -- inspector 2).  All
differences look backwards (t minus t-1); nothing uses a later hour.
Computed straight from the raw CSVs, no local/ pickle chain (inspector 2).

1. Restored-row flags (inspector 3): keep only the rules that mark real
   restoration, drop the midnight-seam detectors.
     V1  night (19-06 h) in_temp < out_temp - 3
     V5  in_temp unchanged for >= 6 consecutive hours, counted BACKWARDS
         (the run length up to and including this hour)
     V4  night, not 00 h: |in_co2 - in_co2(t-1)| > 150 with act_co2 == 0
   V6 / V7 fired at 00 h in 27/28 and 22/22 cases: they detected the source
   switch at midnight, which the test has too.

2. Noisy-day score (inspector 3's signature of injected noise): per training
   greenhouse-day,
     ac1   lag-1 autocorrelation of the hourly CO2 difference
           (noisy days 0.08, clean training days 0.54, test days 0.46)
     d2    median |second difference| of CO2
   noise_score = rank(-ac1) + rank(d2), ranked among TRAINING days only.
   Top quarter = noisy.  Days that contain rows with 3-h air ewm < 8 C are
   exempted (they hold a quarter of the scarce cold training rows).
"""
import numpy as np
import pandas as pd

import common
from common import TARGET_FARMS


def _train_frame():
    tX, ty, sX = common.load_raw()
    a = tX[tX.farm.isin(TARGET_FARMS)].sort_values(["farm", "t"]).reset_index(drop=True)
    return a


def restored_flags():
    a = _train_frame()
    g = a.groupby("farm")
    night = (a.hour >= 19) | (a.hour <= 6)
    contiguous = g.t.diff() == 1
    d1_co2 = g.in_co2.diff().where(contiguous)
    d1_temp = g.in_temp.diff().where(contiguous)
    same = ((d1_temp == 0) & contiguous).astype(int)
    run_back = same.groupby([a.farm, (same == 0).cumsum()]).cumsum() + 1   # hours of the run so far
    v1 = night & (a.in_temp < a.out_temp - 3)
    v5 = run_back >= 6
    v4 = night & (a.hour != 0) & (d1_co2.abs() > 150) & (a.act_co2.fillna(0) == 0)
    out = pd.DataFrame({"row_id": a.row_id, "farm": a.farm, "day": a.day, "t": a.t,
                        "V1": v1.fillna(False), "V5b": v5.fillna(False), "V4nm": v4.fillna(False)})
    out["flag"] = out[["V1", "V5b", "V4nm"]].any(axis=1)
    return out


def noisy_days():
    a = _train_frame()
    a["ewm3"] = a.groupby("farm").in_temp.transform(lambda s: s.ewm(halflife=3, ignore_na=True).mean())
    rows = []
    for (f, d), g in a.groupby(["farm", "day"]):
        g = g.sort_values("t")
        d1 = g.in_co2.diff()
        d1 = d1[g.t.diff() == 1]
        ac1 = d1.autocorr(lag=1) if d1.notna().sum() >= 8 else np.nan
        d2 = d1.diff().abs().median()
        rows.append(dict(farm=f, day=d, ac1=ac1, d2=d2, cold=bool((g.ewm3 < 8).any())))
    D = pd.DataFrame(rows)
    D["noise_score"] = (-D.ac1).rank(pct=True) + D.d2.rank(pct=True)
    thr = D.noise_score.quantile(0.75)
    D["noisy"] = (D.noise_score >= thr) & ~D.cold
    return D


def row_weights(lab, w_flag=0.2, w_noisy=None, radius=0):
    """Weights aligned to `lab` (rows of the labelled temperature frame)."""
    fl = restored_flags()
    bad = fl[fl.flag]
    w = np.ones(len(lab))
    key = lab[["farm", "t"]]
    for farm, g in bad.groupby("farm"):
        m = (key.farm.values == farm) & (np.abs(key.t.values[:, None] - g.t.values[None, :]) <= radius).any(1)
        w[m] = np.minimum(w[m], w_flag)
    if w_noisy is not None:
        nd = noisy_days()
        s = set(map(tuple, nd[nd.noisy][["farm", "day"]].values))
        m = np.array([(f, d) in s for f, d in zip(lab.farm.values, lab.day.values)])
        w[m] = w[m] * w_noisy
    return w


if __name__ == "__main__":
    fl = restored_flags()
    print("restored flags (training rows): V1 %d | V5b %d | V4nm %d | any %d | at 00 h %d"
          % (fl.V1.sum(), fl.V5b.sum(), fl.V4nm.sum(), fl.flag.sum(), int((fl.flag & (fl.t % 24 == 0)).sum())))
    nd = noisy_days()
    print("noisy days: %d of %d (cold-exempt %d) | ac1 median noisy %.2f vs rest %.2f"
          % (nd.noisy.sum(), len(nd), int(((nd.noise_score >= nd.noise_score.quantile(.75)) & nd.cold).sum()),
             nd[nd.noisy].ac1.median(), nd[~nd.noisy].ac1.median()))
