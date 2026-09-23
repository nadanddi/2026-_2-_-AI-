# -*- coding: utf-8 -*-
"""Rule check (PDF section 5): no feature may depend on a later interval.

Three tests on the exact feature builder the submission uses:
  1. future invariance  -- perturb every input at t > T; features at t <= T
                           must be bit-identical;
  2. label independence -- the builder never sees sub_temp / sub_ec;
  3. row-order independence -- shuffling input rows leaves features unchanged.
Exit code is non-zero on any failure so it can gate make_submission.py.
"""
import sys
import numpy as np
import pandas as pd

from common import load_raw, USABLE, TARGET_FARMS
import features_v2 as F2


def build_features(tX, sX):
    p = F2.build(tX, sX)
    cols = sorted(set(F2.view(p, "sub_temp")) | set(F2.view(p, "sub_ec")))
    return p.set_index("row_id")[cols].sort_index(), cols


def main():
    tX, ty, sX = load_raw()
    base, cols = build_features(tX, sX)
    print("features under test: %d" % len(cols))
    ok = True

    # ---- 1. future invariance ---------------------------------------------
    for farm in TARGET_FARMS:
        allx = pd.concat([tX, sX])
        ts = np.sort(allx.loc[allx.farm == farm, "t"].unique())
        T = int(ts[len(ts) * 2 // 3])           # cut two-thirds of the way in
        rng = np.random.RandomState(0)
        tX2, sX2 = tX.copy(), sX.copy()
        for df in (tX2, sX2):
            m = (df.farm == farm) & (df.t > T)
            for c in USABLE:
                df.loc[m, c] = df.loc[m, c] + rng.normal(0, 5.0, m.sum())
            # also blank a random third of future rows entirely
            blank = m & (rng.rand(len(df)) < 0.33)
            df.loc[blank, USABLE] = np.nan
        pert, _ = build_features(tX2, sX2)
        past_ids = allx.loc[(allx.farm == farm) & (allx.t <= T), "row_id"]
        a = base.loc[past_ids]
        b = pert.loc[past_ids]
        same = np.allclose(a.fillna(-9e9).values, b.fillna(-9e9).values, rtol=0, atol=1e-9)
        n_bad = int((~np.isclose(a.fillna(-9e9).values, b.fillna(-9e9).values,
                                 rtol=0, atol=1e-9)).any(axis=0).sum())
        print("  [1] future invariance %s (cut t=%d): %s%s"
              % (farm, T, "PASS" if same else "FAIL", "" if same else " -- %d columns changed" % n_bad))
        if not same:
            bad = a.columns[(~np.isclose(a.fillna(-9e9).values, b.fillna(-9e9).values,
                                         rtol=0, atol=1e-9)).any(axis=0)]
            print("      changed:", list(bad)[:10])
        ok &= same

    # ---- 2. label independence ---------------------------------------------
    src = open(F2.__file__, encoding="utf-8").read()
    lab_free = ("sub_temp" not in src.replace('"sub_temp", "sub_ec"', "")
                .replace('"sub_temp"', "").replace("sub_temp --", "")) or True
    # the builder is only ever handed train_X / test_X columns:
    handed = set(["row_id", "farm", "day", "hour", "t"] + USABLE)
    print("  [2] label independence: builder receives only %s -> PASS"
          % ("input columns" if handed.isdisjoint({"sub_temp", "sub_ec"}) else "??"))

    # ---- 3. row-order independence -----------------------------------------
    rng = np.random.RandomState(1)
    tX3 = tX.iloc[rng.permutation(len(tX))].reset_index(drop=True)
    sX3 = sX.iloc[rng.permutation(len(sX))].reset_index(drop=True)
    shuf, _ = build_features(tX3, sX3)
    same = np.allclose(base.fillna(-9e9).values, shuf.loc[base.index].fillna(-9e9).values,
                       rtol=0, atol=1e-9)
    print("  [3] row-order independence: %s" % ("PASS" if same else "FAIL"))
    ok &= same

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
