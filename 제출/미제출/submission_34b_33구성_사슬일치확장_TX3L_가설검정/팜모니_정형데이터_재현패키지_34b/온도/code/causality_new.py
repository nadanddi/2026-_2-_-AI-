# -*- coding: utf-8 -*-
"""Future-invariance audit for the NEW derived columns in feat_new.py.

Claude/code/causality_test.py only covers features_v2's 125 columns.  The 47
blocks added by feat_new.py have never been machine-checked.  Competition rule
(PDF section 5): a row may use only inputs of the SAME greenhouse at the
current or an EARLIER interval.

Method (same as causality_test.py): perturb and blank every input after a cut
time T, rebuild, and require the columns at t <= T to be bit-identical.

Usage:  PYTHONPATH="" python causality_new.py
"""
import sys

import env  # noqa: F401  MUST be first project import
import numpy as np
import pandas as pd

import common
from common import USABLE, TARGET_FARMS
import feat_new

_ORIG = common.load_raw


def perturbed_loader(farm, T, seed=0):
    def loader():
        tX, ty, sX = _ORIG()
        rng = np.random.RandomState(seed)
        tX2, sX2 = tX.copy(), sX.copy()
        for df in (tX2, sX2):
            m = ((df.farm == farm) & (df.t > T)).values
            if m.sum() == 0:
                continue
            for c in USABLE:
                df.loc[m, c] = df.loc[m, c] + rng.normal(0, 5.0, int(m.sum()))
            blank = m & (rng.rand(len(df)) < 0.33)
            df.loc[blank, USABLE] = np.nan
        return tX2, ty, sX2
    return loader


def main():
    tX, ty, sX = _ORIG()
    allx = pd.concat([tX, sX])

    base, blocks = feat_new.build_extra()
    base = base.set_index("row_id").sort_index()
    names = list(base.columns)
    print("new columns under test: %d in %d blocks" % (len(names), len(blocks)))

    ok = True
    for farm in TARGET_FARMS:
        ts = np.sort(allx.loc[allx.farm == farm, "t"].unique())
        T = int(ts[len(ts) * 2 // 3])
        common.load_raw = perturbed_loader(farm, T)
        try:
            alt, _ = feat_new.build_extra()
        finally:
            common.load_raw = _ORIG
        alt = alt.set_index("row_id").sort_index()

        rid = allx.set_index("row_id")
        past = [r for r in base.index
                if r in rid.index and rid.loc[r, "farm"] == farm
                and rid.loc[r, "t"] <= T]
        a, b = base.loc[past, names], alt.loc[past, names]
        diff = ~((a.values == b.values) | (a.isna().values & b.isna().values))
        bad = {}
        for j, c in enumerate(names):
            n = int(diff[:, j].sum())
            if n:
                bad[c] = n
        if bad:
            ok = False
            print("  [%s] cut t=%d, %d past rows: FAIL on %d columns"
                  % (farm, T, len(past), len(bad)))
            byblock = {}
            for c, n in bad.items():
                blk = next((k for k, v in blocks.items() if c in v), "?")
                byblock.setdefault(blk, []).append((c, n))
            for blk, items in sorted(byblock.items()):
                worst = max(n for _, n in items)
                print("     block %-6s %2d/%2d columns differ, worst %d rows"
                      % (blk, len(items), len(blocks.get(blk, [])), worst))
        else:
            print("  [%s] cut t=%d, %d past rows: PASS" % (farm, T, len(past)))

    print("\nRESULT: %s" % ("ALL PASS" if ok else "FAIL - see blocks above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
