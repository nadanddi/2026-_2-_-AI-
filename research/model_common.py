# -*- coding: utf-8 -*-
"""Shared scaffolding for the model-family search.

Everything here sits on top of harness.score so that every candidate is
measured on the same geometry folds (kind A and kind B) as the submitted
models.  The only thing added is a per-segment diagnostic: overall RMSE is
not enough for sub_ec, where 8% of the rows (y > 1) carry 63% of the squared
error and are under-predicted by ~0.54 on average.

Import order matters: `import env` must precede numpy/sklearn/lightgbm.
"""
import time

import env  # noqa: F401  MUST be first project import (DLL path + sys.path)
import numpy as np

from harness import load, score, views, blend_factory  # noqa: F401

HI = 1.0  # sub_ec high-value cut


def seg_stats(y, p, cut=HI):
    """RMSE and mean bias (pred - true) on the high-value segment."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = y > cut
    if m.sum() == 0:
        return dict(n=0, rmse=float("nan"), bias=float("nan"), share=float("nan"))
    sse_hi = float(np.sum((p[m] - y[m]) ** 2))
    sse_all = float(np.sum((p - y) ** 2))
    return dict(n=int(m.sum()),
                rmse=float(np.sqrt(sse_hi / m.sum())),
                bias=float(np.mean(p[m] - y[m])),
                share=sse_hi / sse_all if sse_all > 0 else float("nan"))


def evaluate(lab, target, cols, factory, seeds=(7, 101, 2024), segment=False):
    """Score a candidate on both fold sets.  Returns a dict."""
    out = {}
    for kind in ("A", "B"):
        (r, sd, per), oof = score(lab, target, cols, factory, kind=kind,
                                  seeds=seeds, return_oof=True)
        got = ~np.isnan(oof)
        out[kind] = dict(rmse=r, std=sd, per=per)
        if segment:
            out[kind]["seg"] = seg_stats(lab[target].values[got], oof[got])
        out[kind]["oof"] = oof
    return out


def run_table(lab, target, cands, segment=False, tag=""):
    """cands: list of (name, cols, factory[, seeds]).  Prints a table."""
    print("\n===== %s %s =====" % (target, tag))
    if segment:
        hdr = ("%-40s %7s %7s %7s %7s | %7s %7s %7s %6s" %
               ("candidate", "A", "sdA", "B", "sdB",
                "hiRMSE", "hiBias", "hiShr", "sec"))
    else:
        hdr = ("%-40s %7s %7s %7s %7s %6s" %
               ("candidate", "A", "sdA", "B", "sdB", "sec"))
    print(hdr)
    print("-" * len(hdr))
    res = {}
    for c in cands:
        nm, cols, fac = c[0], c[1], c[2]
        seeds = c[3] if len(c) > 3 else (7, 101, 2024)
        t0 = time.time()
        try:
            r = evaluate(lab, target, cols, fac, seeds=seeds, segment=segment)
        except Exception as e:  # keep the table going
            print("%-40s  FAILED: %s" % (nm, str(e)[:80]))
            continue
        dt = time.time() - t0
        if segment:
            # average the two fold sets' segment numbers for the display
            sa, sb = r["A"]["seg"], r["B"]["seg"]
            print("%-40s %7.4f %7.4f %7.4f %7.4f | %7.4f %+7.4f %7.3f %6.0f" %
                  (nm, r["A"]["rmse"], r["A"]["std"], r["B"]["rmse"], r["B"]["std"],
                   0.5 * (sa["rmse"] + sb["rmse"]), 0.5 * (sa["bias"] + sb["bias"]),
                   0.5 * (sa["share"] + sb["share"]), dt))
        else:
            print("%-40s %7.4f %7.4f %7.4f %7.4f %6.0f" %
                  (nm, r["A"]["rmse"], r["A"]["std"], r["B"]["rmse"], r["B"]["std"], dt))
        res[nm] = r
    return res
