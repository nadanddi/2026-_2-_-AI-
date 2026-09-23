# -*- coding: utf-8 -*-
"""Are the duplicate-weather days adjacent pairs?  (interleaved-series test)"""
import numpy as np
import pandas as pd
from collections import Counter

from common import load_raw, TARGET_FARMS


def main():
    tX, ty, sX = load_raw()
    x = pd.concat([tX, sX])
    for farm in TARGET_FARMS:
        g = x[x.farm == farm]
        p = g.pivot(index="day", columns="hour",
                    values=["out_temp", "out_hum", "out_rad", "out_wspd"]).dropna()
        h = pd.util.hash_pandas_object(p, index=False)
        h.index = p.index
        gaps, parity = Counter(), Counter()
        for sig, grp in h.groupby(h):
            ds = sorted(grp.index)
            if len(ds) < 2:
                continue
            for a, b in zip(ds[:-1], ds[1:]):
                gaps[b - a] += 1
                parity[(a % 2, b % 2)] += 1
        print("%s  twin gap distribution: %s" % (farm, dict(sorted(gaps.items()))))
        print("      parity of (first, second): %s" % dict(parity))
        # label agreement as a function of gap
        L = ty[ty.farm == farm].groupby("day").agg(ec=("sub_ec", "mean"),
                                                   tp=("sub_temp", "mean"))
        rows = []
        for sig, grp in h.groupby(h):
            ds = sorted(d for d in grp.index if d in L.index)
            for a, b in zip(ds[:-1], ds[1:]):
                rows.append((b - a, abs(L.ec[a] - L.ec[b]), abs(L.tp[a] - L.tp[b])))
        r = pd.DataFrame(rows, columns=["gap", "dEC", "dT"])
        print("      |dEC| by gap:", r.groupby("gap").dEC.mean().round(3).to_dict())
        print("      |dT|  by gap:", r.groupby("gap").dT.mean().round(2).to_dict())
        # compare with plain adjacent (non-twin) days
        Ls = L.sort_index()
        adj = Ls.index[1:][np.diff(Ls.index) == 1]
        dEC1 = np.abs(Ls.ec.diff().loc[adj]).mean()
        dEC2 = np.abs(Ls.ec.diff(2).dropna()).mean()
        print("      reference |dEC|: any adjacent day %.3f, 2 days apart %.3f"
              % (dEC1, dEC2))


if __name__ == "__main__":
    main()
