# -*- coding: utf-8 -*-
"""Future-invariance audit for the cold-hinge block of features_v5.

Same method as causality_v4.py: perturb and blank every input after a
mid-day cut, rebuild, require the hinge columns at t <= cut to be
bit-identical.  (seg / phys / fp are covered by causality_v4.py.)

Run:  cd research && PYTHONPATH="" <python> causality_v5.py
"""
import sys

import env  # noqa: F401  MUST be the first project import
import numpy as np
import pandas as pd

import common
from common import USABLE, TARGET_FARMS
import features_v5 as F5


def build():
    return F5.hinge_features(F5.phys_features()).set_index("row_id").sort_index()


def main():
    orig = common.load_raw
    tX, ty, sX = orig()
    allx = pd.concat([tX, sX]).set_index("row_id")
    base = build()
    ok = True
    for farm in TARGET_FARMS:
        ts = np.sort(allx[allx.farm == farm].t.unique())
        for frac in (0.4, 0.75):
            T = int(ts[int(len(ts) * frac)] // 24 * 24 + 13)

            def loader(farm=farm, T=T):
                a, b, c = orig()
                rng = np.random.RandomState(2)
                for df in (a, c):
                    m = ((df.farm == farm) & (df.t > T)).values
                    for col in USABLE:
                        df.loc[m, col] = df.loc[m, col] + rng.normal(0, 5.0, int(m.sum()))
                    df.loc[m & (rng.rand(len(df)) < 0.33), USABLE] = np.nan
                return a, b, c

            common.load_raw = loader
            try:
                alt = build()
            finally:
                common.load_raw = orig
            past = allx.index[(allx.farm == farm) & (allx.t <= T)]
            a, b = base.reindex(past), alt.reindex(past)
            same = (a.values == b.values) | (a.isna().values & b.isna().values)
            bad = int((~same).sum())
            ok &= bad == 0
            print("  [%s cut t=%d] hinge %d cols: %s"
                  % (farm, T, a.shape[1], "PASS" if bad == 0 else "FAIL %d cells" % bad))
    print("\nRESULT: %s" % ("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
