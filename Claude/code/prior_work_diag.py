# -*- coding: utf-8 -*-
"""Check whether findings from the earlier repo (v1-v5) apply to the current model.

Findings under test:
  (a) "EC>1 rows are 17% of rows but 84% of squared error" -- if that holds for
      the current OOF, the high-EC regime is the whole game;
  (b) exact-duplicate outside-weather days inside F13/F47 -- the earlier CV
      purged these; the current CV does not, so quantify the exposure.
"""
import numpy as np
import pandas as pd

from common import make_folds, rmse, TARGET_FARMS
from model_v2 import get_panel, N_FOLDS
from model_v3 import run
from model_v4 import E_HUB, MODEL_SEEDS
from submit_v2 import slim


def main():
    panel, tX, ty, sX = get_panel()
    days = {f: sorted(panel[(panel.farm == f) & (~panel.is_test)].day.unique())
            for f in TARGET_FARMS}
    ve = slim(panel, "sub_ec")

    # ---- (a) error concentration by EC level on current OOF -----------------
    folds = make_folds(days, n_folds=N_FOLDS, seed=0)
    lab, o1 = run(panel, folds, "sub_ec", ve, E_HUB, seeds=MODEL_SEEDS)
    _, o2 = run(panel, folds, "sub_ec", ve, E_HUB, seeds=MODEL_SEEDS, recency_tau=120)
    oof = 0.5 * (o1 + o2)
    got = ~np.isnan(oof)
    y = lab.sub_ec.values
    se = (oof - y) ** 2
    print("=== (a) EC squared-error concentration (current final model, partition 0) ===")
    print("  overall RMSE %.4f" % rmse(oof[got], y[got]))
    for thr in (0.5, 0.8, 1.0, 1.5):
        hi = got & (y > thr)
        print("  EC>%.1f : %5.1f%% of rows -> %5.1f%% of SSE | RMSE inside %.4f, outside %.4f"
              % (thr, 100 * hi.sum() / got.sum(), 100 * se[hi].sum() / se[got].sum(),
                 rmse(oof[hi], y[hi]), rmse(oof[got & ~hi], y[got & ~hi])))
    bias_hi = (oof - y)[got & (y > 1.0)].mean()
    print("  mean bias on EC>1 rows: %+.4f (negative = under-prediction)" % bias_hi)
    late = got & (lab.day.values >= 183)
    print("  late period: EC>1 share of rows %.1f%%, of SSE %.1f%%"
          % (100 * (late & (y > 1)).sum() / late.sum(),
             100 * se[late & (y > 1)].sum() / se[late].sum()))

    # ---- (b) duplicate outside-weather days within F13/F47 -----------------
    print("\n=== (b) exact-duplicate 24h outside-weather signatures ===")
    x = pd.concat([tX, sX])
    x = x[x.farm.isin(TARGET_FARMS)]
    for farm in TARGET_FARMS:
        g = x[x.farm == farm]
        p = g.pivot(index="day", columns="hour",
                    values=["out_temp", "out_hum", "out_rad", "out_wspd"])
        full = p.dropna()
        h = pd.util.hash_pandas_object(full, index=False)
        h.index = full.index
        vc = h.value_counts()
        dup_days = h[h.isin(vc[vc > 1].index)]
        test_days = set(sX[sX.farm == farm].day)
        tr_days = set(tX[tX.farm == farm].day)
        dup_test = [d for d in dup_days.index if d in test_days]
        # test days whose signature also appears on a train day
        sig_train = set(h[h.index.isin(tr_days)])
        test_with_train_twin = [d for d in test_days
                                if d in h.index and h[d] in sig_train]
        print("  %s: %d complete days, %d in duplicate groups (%d groups); "
              "test days w/ a train twin: %d / %d"
              % (farm, len(full), len(dup_days), (vc > 1).sum(),
                 len(test_with_train_twin), len(test_days)))
        # do labels agree on twin days?  (if yes -> real leakage channel)
        L = ty[ty.farm == farm].groupby("day").agg(ec=("sub_ec", "mean"),
                                                   tp=("sub_temp", "mean"))
        agree_ec, agree_tp, n = [], [], 0
        for sig, grp in h.groupby(h):
            ds = [d for d in grp.index if d in L.index]
            if len(ds) >= 2:
                agree_ec.append(L.loc[ds, "ec"].std())
                agree_tp.append(L.loc[ds, "tp"].std())
                n += 1
        if n:
            print("     twin-day label spread: EC std %.3f (global day-std %.3f) | "
                  "temp std %.2f (global %.2f)"
                  % (np.nanmean(agree_ec), L.ec.std(),
                     np.nanmean(agree_tp), L.tp.std()))


if __name__ == "__main__":
    main()
