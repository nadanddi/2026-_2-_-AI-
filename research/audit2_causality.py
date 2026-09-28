# -*- coding: utf-8 -*-
"""Audit 2 (inspector): stronger rule checks on EVERY column the 2nd-4th
submissions feed to a model, built by the exact shipped builders.

  [F] future invariance   many cuts per greenhouse (hours 0/5/13/23, early
                          and late in the record); future inputs get noise
                          + x10 scaling + 33% blanking (a scale change is
                          what exposes whole-series statistics); past rows
                          must be bit-identical.
  [X] greenhouse isolation all inputs of every OTHER greenhouse perturbed;
                          every row of this greenhouse must be unchanged.
  [Y] label independence  train_y shuffled; every column unchanged.

Columns under test = manifest feature lists of 03/04/05 (+ phys + hinge),
plus feat_new's rep24 block (not shipped; tested to show the known weakness).
Writes local/audit2_causality.json.  Read-only on project files.
"""
import json
import sys

import env  # noqa: F401
import numpy as np
import pandas as pd

import common
from common import USABLE, TARGET_FARMS
import features_v2 as F2
import feat_new
import features_v4 as F4
import features_v5 as F5

ORIG = common.load_raw


def man_cols():
    cols = set()
    for n in ("03", "04", "05"):
        m = json.load(open(env.os.path.join(env.os.path.dirname(__file__), "submissions",
                                            "manifest_%s.json" % n), encoding="utf-8"))
        for k in ("sub_temp", "sub_ec"):
            for kk in ("features", "phys_features", "hinge_features", "features_et", "features_other"):
                cols |= set(m[k].get(kk, []))
    return cols


def build_all():
    tX, ty, sX = common.load_raw()
    p = F2.build(tX, sX)
    p["midnight"] = (p.hour == 0).astype(float)
    p = p.set_index("row_id")
    ex, _ = feat_new.build_extra()
    ph = F4.phys_features()
    parts = [p, ex.set_index("row_id"), F4.seg_features().set_index("row_id"),
             ph.set_index("row_id"), F4.fp_features().set_index("row_id"),
             F5.hinge_features(ph).set_index("row_id")]
    out = pd.concat([x.loc[:, ~x.columns.isin(["farm", "t"])] for x in parts], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out.sort_index()


def with_loader(fn):
    common.load_raw = fn
    try:
        return build_all()
    finally:
        common.load_raw = ORIG


def cmp(a, b):
    a, b = a.astype(float), b.astype(float)
    same = (a.values == b.values) | (np.isnan(a.values) & np.isnan(b.values))
    return {c: int((~same[:, j]).sum()) for j, c in enumerate(a.columns) if (~same[:, j]).any()}


def main():
    tX, ty, sX = ORIG()
    allx = pd.concat([tX, sX]).set_index("row_id")
    shipped = man_cols()
    base = build_all()
    missing = sorted(shipped - set(base.columns))
    print("shipped columns %d, not rebuilt here: %s" % (len(shipped), missing))
    test_cols = sorted((shipped & set(base.columns)) - {"day", "hour"}) + ["day", "hour"]
    rep = [c for c in base.columns if c.endswith("_rep24")]
    print("rep24 shipped?", bool(set(rep) & shipped))
    cols = test_cols + rep
    base = base[cols]
    res = dict(F=[], X=[], Y=None, shipped_n=len(shipped), leaked_labels=sorted(shipped & {"sub_temp", "sub_ec"}))

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
            bship = {k: v for k, v in bad.items() if k in shipped}
            brep = {k: v for k, v in bad.items() if k in rep}
            print("[F] %s cut t=%d (day %d h%02d): shipped bad %d cols %s | rep24 bad %s"
                  % (farm, T, T // 24, T % 24, len(bship), list(bship)[:5], brep), flush=True)
            res["F"].append(dict(farm=farm, T=T, shipped_bad=bship, rep24_bad=brep))

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
        res["X"].append(dict(farm=farm, bad=bad))

    def ldy():
        a, b, c = ORIG()
        rng = np.random.RandomState(3)
        b = b.copy()
        b["sub_temp"] = rng.permutation(b.sub_temp.values)
        b["sub_ec"] = rng.permutation(b.sub_ec.values)
        return a, b, c

    alt = with_loader(ldy)[cols]
    bad = cmp(base, alt)
    print("[Y] label independence: %s" % ("PASS" if not bad else bad))
    res["Y"] = bad
    nF = sum(len(r["shipped_bad"]) for r in res["F"])
    print("\nSHIPPED: future %s | isolation %s | labels %s"
          % ("PASS" if nF == 0 else "FAIL", "PASS" if all(not r["bad"] for r in res["X"]) else "FAIL",
             "PASS" if not res["Y"] else "FAIL"))
    with open(env.LOCAL + "/audit2_causality.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
