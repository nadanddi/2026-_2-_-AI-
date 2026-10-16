# -*- coding: utf-8 -*-
"""Rule checks for submission_06, following inspector 2's stronger design
(audit2_causality.py), restricted to the columns manifest_06 feeds a model,
plus a check on the training-row weights.

  [F] future invariance   6 cuts per greenhouse (hours 0/5/7/13/23, early and
                          late); future inputs get x10 scaling + noise + 33%
                          blanking; every past row must be bit-identical.
  [X] greenhouse isolation all inputs of the other greenhouse x3 + noise;
                          every row of this greenhouse must be unchanged.
  [Y] label independence  train_y shuffled; every column unchanged.
  [W] training weights    test_X scaled/blanked entirely -> weights unchanged
                          (they must come from training inputs only); future
                          cuts inside the training record leave the weights of
                          earlier rows' flags unchanged.

Run:  cd research && PYTHONPATH="" <python> -u causality_v6.py   (ALL PASS)
"""
import json
import os
import sys

import env  # noqa: F401
import numpy as np
import pandas as pd

import common
from common import USABLE, TARGET_FARMS
import features_v2 as F2
import feat_new
import features_v4 as F4
import train_flags_v6 as TF

ORIG = common.load_raw
HERE = os.path.dirname(os.path.abspath(__file__))


def man_path():
    for p in (os.path.join(HERE, "..", "config", "manifest_06.json"),
              os.path.join(HERE, "submissions", "manifest_06.json")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("manifest_06.json")


def man_cols():
    m = json.load(open(man_path(), encoding="utf-8"))
    cols = set()
    for k in ("sub_temp", "sub_ec"):
        for kk in ("features", "phys_features", "features_et", "features_other"):
            cols |= set(m[k].get(kk, []))
    return cols


def build_all():
    tX, ty, sX = common.load_raw()
    p = F2.build(tX, sX)
    p["midnight"] = (p.hour == 0).astype(float)
    p = p.set_index("row_id")
    ex, _ = feat_new.build_extra()
    parts = [p, ex.set_index("row_id"), F4.seg_features().set_index("row_id"),
             F4.phys_features().set_index("row_id"), F4.fp_features().set_index("row_id")]
    out = pd.concat([x.loc[:, ~x.columns.isin(["farm", "t"])] for x in parts], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out.sort_index()


def with_loader(fn, build=build_all):
    common.load_raw = fn
    try:
        return build()
    finally:
        common.load_raw = ORIG


def cmp(a, b):
    a, b = a.astype(float), b.astype(float)
    same = (a.values == b.values) | (np.isnan(a.values) & np.isnan(b.values))
    return {c: int((~same[:, j]).sum()) for j, c in enumerate(a.columns) if (~same[:, j]).any()}


def weights_frame():
    tX, ty, sX = common.load_raw()
    lab = tX[tX.farm.isin(TARGET_FARMS)].sort_values(["farm", "t"]).reset_index(drop=True)
    fl = TF.restored_flags().set_index("row_id")
    return pd.DataFrame({"w": TF.row_weights(lab, 0.2, w_noisy=0.2),
                         "flag": fl.loc[lab.row_id, "flag"].values.astype(float)}, index=lab.row_id)


def main():
    tX, ty, sX = ORIG()
    allx = pd.concat([tX, sX]).set_index("row_id")
    shipped = man_cols()
    base = build_all()
    missing = sorted(shipped - set(base.columns))
    assert not missing, "columns not rebuilt: %s" % missing
    cols = sorted(shipped)
    base = base[cols]
    print("columns under test: %d" % len(cols))
    fails = []

    for farm in TARGET_FARMS:
        ts = np.sort(allx[allx.farm == farm].t.unique())
        for frac, hh in ((0.15, 0), (0.45, 13), (0.6, 5), (0.8, 23), (0.9, 13), (0.97, 7)):
            T = int(ts[int(len(ts) * frac)] // 24 * 24 + hh)

            def ld(farm=farm, T=T):
                a, b, c = ORIG()
                rng = np.random.RandomState(T)
                for df in (a, c):
                    m = ((df.farm == farm) & (df.t > T)).values
                    for col in USABLE:
                        df.loc[m, col] = df.loc[m, col] * 10.0 + rng.normal(0, 5.0, int(m.sum()))
                    df.loc[m & (rng.rand(len(df)) < 0.33), USABLE] = np.nan
                return a, b, c

            alt = with_loader(ld)[cols]
            past = allx.index[(allx.farm == farm) & (allx.t <= T)]
            bad = cmp(base.loc[past], alt.loc[past])
            print("[F] %s cut day %d h%02d: %s" % (farm, T // 24, T % 24, "PASS" if not bad else bad), flush=True)
            if bad:
                fails.append(("F", farm, T))

        def ldx(farm=farm):
            a, b, c = ORIG()
            rng = np.random.RandomState(7)
            for df in (a, c):
                m = (df.farm != farm).values
                for col in USABLE:
                    df.loc[m, col] = df.loc[m, col] * 3.0 + rng.normal(0, 5.0, int(m.sum()))
            return a, b, c

        alt = with_loader(ldx)[cols]
        own = allx.index[allx.farm == farm]
        bad = cmp(base.loc[own], alt.loc[own])
        print("[X] %s isolation: %s" % (farm, "PASS" if not bad else bad), flush=True)
        if bad:
            fails.append(("X", farm))

    def ldy():
        a, b, c = ORIG()
        rng = np.random.RandomState(3)
        b = b.copy()
        b["sub_temp"] = rng.permutation(b.sub_temp.values)
        b["sub_ec"] = rng.permutation(b.sub_ec.values)
        return a, b, c

    bad = cmp(base, with_loader(ldy)[cols])
    print("[Y] label independence: %s" % ("PASS" if not bad else bad))
    if bad:
        fails.append(("Y",))

    # [W] training-row weights
    wb = weights_frame()

    def ldt():
        a, b, c = ORIG()
        c = c.copy()
        rng = np.random.RandomState(11)
        for col in USABLE:
            c[col] = c[col] * 10.0 + rng.normal(0, 5.0, len(c))
        c.loc[rng.rand(len(c)) < 0.33, USABLE] = np.nan
        return a, b, c

    bad = cmp(wb, with_loader(ldt, weights_frame))
    print("[W] weights ignore test_X: %s" % ("PASS" if not bad else bad))
    if bad:
        fails.append(("W", "test"))
    for farm in TARGET_FARMS:
        ts = np.sort(tX[tX.farm == farm].t.unique())
        for frac in (0.3, 0.7):
            T = int(ts[int(len(ts) * frac)])

            def ldc(farm=farm, T=T):
                a, b, c = ORIG()
                a = a.copy()
                rng = np.random.RandomState(T)
                m = ((a.farm == farm) & (a.t > T)).values
                for col in USABLE:
                    a.loc[m, col] = a.loc[m, col] * 10.0 + rng.normal(0, 5.0, int(m.sum()))
                return a, b, c

            alt = with_loader(ldc, weights_frame)
            past = [r for r in wb.index if r.startswith(farm)]
            past = [r for r, t in zip(past, tX.set_index("row_id").loc[past, "t"]) if t <= T]
            bad = cmp(wb.loc[past, ["flag"]], alt.loc[past, ["flag"]])
            print("[W] %s restored flags backward-only, cut t=%d: %s" % (farm, T, "PASS" if not bad else bad))
            if bad:
                fails.append(("W", farm, T))

    print("\n%s" % ("ALL PASS" if not fails else "FAIL %s" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
