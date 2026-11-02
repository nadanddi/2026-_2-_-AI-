# -*- coding: utf-8 -*-
"""Causality audit for the post-processing step used by make_submission_v3.

The within-day shrink is the only part of the pipeline that is NOT a feature
of features_v2 or feat_new, so causality_test.py / causality_new.py do not
cover it.  Claim under test: the shrunk prediction at hour h depends only on
raw predictions of hours 0..h of the SAME greenhouse-day.  If that holds, and
each raw prediction is causal (verified separately), the shrink adds no
dependence on later inputs.

Method: perturb the raw predictions after a cut hour inside every
greenhouse-day and require the shrunk values at or before the cut to be
bit-identical.  Also check that greenhouse-days do not leak into each other
and that row order does not matter.

Run:  cd research && PYTHONPATH="" <python> causality_shrink.py
"""
import sys

import env  # noqa: F401  MUST be the first project import
import numpy as np
import pandas as pd

from make_submission_v3 import causal_shrink, SHRINK_L


def main():
    rng = np.random.RandomState(0)
    n_day, n_farm = 6, 2
    rows = []
    for f in range(n_farm):
        for d in range(n_day):
            for h in range(24):
                rows.append(("F%02d" % f, 100 + d, h))
    frame = pd.DataFrame(rows, columns=["farm", "day", "hour"])
    p = rng.rand(len(frame)) * 2.0
    base = causal_shrink(p, frame, SHRINK_L)
    ok = True

    # ---- 1. future invariance inside each greenhouse-day -------------------
    for cut in (0, 5, 11, 17, 22):
        p2 = p.copy()
        m = (frame.hour > cut).values
        p2[m] = p2[m] + rng.normal(0, 3.0, int(m.sum()))
        alt = causal_shrink(p2, frame, SHRINK_L)
        past = (frame.hour <= cut).values
        if not np.array_equal(base[past], alt[past]):
            bad = int((base[past] != alt[past]).sum())
            print("  [1] cut hour %2d: FAIL (%d of %d past rows changed)"
                  % (cut, bad, past.sum()))
            ok = False
        else:
            print("  [1] cut hour %2d: PASS (%d past rows unchanged)"
                  % (cut, past.sum()))

    # ---- 2. greenhouse-day isolation ---------------------------------------
    p3 = p.copy()
    tgt = ((frame.farm == "F00") & (frame.day == 102)).values
    p3[tgt] = p3[tgt] + 5.0
    alt = causal_shrink(p3, frame, SHRINK_L)
    if not np.array_equal(base[~tgt], alt[~tgt]):
        print("  [2] greenhouse-day isolation: FAIL")
        ok = False
    else:
        print("  [2] greenhouse-day isolation: PASS")

    # ---- 3. row-order independence -----------------------------------------
    perm = rng.permutation(len(frame))
    shuf = frame.iloc[perm].reset_index(drop=True)
    alt = causal_shrink(p[perm], shuf, SHRINK_L)
    inv = np.empty(len(perm), int)
    inv[perm] = np.arange(len(perm))
    if not np.allclose(base, alt[inv], rtol=0, atol=0):
        print("  [3] row-order independence: FAIL")
        ok = False
    else:
        print("  [3] row-order independence: PASS")

    # ---- 4. hour 0 is untouched (no information to average yet) ------------
    h0 = (frame.hour == 0).values
    if not np.array_equal(base[h0], p[h0]):
        print("  [4] hour 0 passthrough: FAIL")
        ok = False
    else:
        print("  [4] hour 0 passthrough: PASS")

    print("\nRESULT: %s" % ("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
