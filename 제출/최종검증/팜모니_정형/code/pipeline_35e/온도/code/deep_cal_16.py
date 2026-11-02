# -*- coding: utf-8 -*-
"""Distance from held-out days to the nearest labelled day (same record, by day index)
for fold sets A, B, SP, SP2 (4-day chunks of the second pass) and the real test."""
import env  # noqa
import numpy as np
from harness import load, folds
from common import split_mask
from deep_cal_eval import sp_folds, SP_BLOCKS
import common


def sp2_folds(chunk=4):
    out = []
    for f in SP_BLOCKS:
        pass
    ch = {f: [] for f in SP_BLOCKS}
    for f, bl in SP_BLOCKS.items():
        for a, b in bl:
            ds = list(range(a, b + 1))
            for i in range(0, len(ds), chunk):
                ch[f].append(ds[i:i + chunk])
    n = max(len(v) for v in ch.values())
    for i in range(n):
        out.append({f: set(ch[f][i]) if i < len(ch[f]) else set() for f in ch})
    return out


def gaps(lab, fds):
    g = []
    for fd in fds:
        trm, _ = split_mask(lab, fd)
        for f, vd in fd.items():
            tr = np.array(sorted(lab[trm & (lab.farm == f).values].day.unique()))
            g += [np.abs(tr - d).min() for d in vd]
    g = np.array(g)
    return "mean %.1f median %.0f max %d" % (g.mean(), np.median(g), g.max())


if __name__ == "__main__":
    panel, lab_t, lab_e = load()
    for k in ("A", "B"):
        print(k, gaps(lab_e, folds(k)))
    print("SP", gaps(lab_e, sp_folds()))
    print("SP2 (%d folds)" % len(sp2_folds()), gaps(lab_e, sp2_folds()))
    tX, ty, sX = common.load_raw()
    g = []
    for f in ["F13", "F47"]:
        tr = np.array(sorted(ty[ty.farm == f].day.unique()))
        g += [np.abs(tr - d).min() for d in sX[sX.farm == f].day.unique()]
    print("TEST mean %.1f median %.0f max %d" % (np.mean(g), np.median(g), np.max(g)))
