# -*- coding: utf-8 -*-
"""Future-invariance audit for the feature blocks of features_v4 (seg, phys, fp).

Perturb and blank every input after a cut time T (placed MID-day so that
same-day blocks are exercised), rebuild, and require every column at t <= T to
be bit-identical.  Two cuts per greenhouse.

Run:  cd research && PYTHONPATH="" <python> causality_v4.py
"""
import sys

import env  # noqa: F401  MUST be the first project import
import numpy as np
import pandas as pd

import common
from common import USABLE, TARGET_FARMS
import features_v4 as F4

BLOCKS = {"seg": F4.seg_features, "phys": F4.phys_features, "fp": F4.fp_features}


def main():
    orig = common.load_raw
    tX, ty, sX = orig()
    allx = pd.concat([tX, sX]).set_index("row_id")
    base = {k: f().set_index("row_id").sort_index() for k, f in BLOCKS.items()}
    ok = True
    for farm in TARGET_FARMS:
        ts = np.sort(allx[allx.farm == farm].t.unique())
        for frac in (0.4, 0.75):
            T = int(ts[int(len(ts) * frac)] // 24 * 24 + 13)     # 13:00 of that day

            def loader(farm=farm, T=T):
                a, b, c = orig()
                rng = np.random.RandomState(1)
                for df in (a, c):
                    m = ((df.farm == farm) & (df.t > T)).values
                    for col in USABLE:
                        df.loc[m, col] = df.loc[m, col] + rng.normal(0, 5.0, int(m.sum()))
                    df.loc[m & (rng.rand(len(df)) < 0.33), USABLE] = np.nan
                return a, b, c

            common.load_raw = loader
            try:
                alt = {k: f().set_index("row_id").sort_index() for k, f in BLOCKS.items()}
            finally:
                common.load_raw = orig
            past = allx.index[(allx.farm == farm) & (allx.t <= T)]
            for k in BLOCKS:
                a = base[k].reindex(past)
                b = alt[k].reindex(past)
                same = (a.values == b.values) | (a.isna().values & b.isna().values)
                bad = int((~same).sum())
                ok &= bad == 0
                print("  [%s cut t=%d] %-4s %d cols: %s"
                      % (farm, T, k, a.shape[1], "PASS" if bad == 0 else "FAIL %d cells" % bad))
    print("\nRESULT: %s" % ("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
