# -*- coding: utf-8 -*-
"""Leaderboard pair analysis (segment level, allowed under our rule-5 reading:
only public scores of our own submissions are used, no hidden label is read).

For two submissions A, B with LB RMSE ra, rb and prediction change d = pB - pA:
    rb^2 - ra^2 = mean(d^2) + 2 * mean(d * eA),     eA = pA - truth
so  mean(d * eA) = (rb^2 - ra^2 - mean(d^2)) / 2      (exact, over 1440 rows)
If d is confined to a segment S, mean_S(d*eA) = that / share(S): the average
error of A on S weighted by the change.  With d of one sign on S, its sign
says whether A was too high or too low there.

Pairs:  temperature 04 -> 05 (cold hinge only),  04 -> 06 (weights),
        EC 04 -> 06 (ExtraTrees `day` removed).
Run:  cd research && PYTHONPATH="" <python> -u anal_lb_pairs.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

import common

LB = {"03": (0.6666, 0.2287), "04": (0.5624, 0.2055), "05": (0.5697, 0.2055), "06": (0.5456, 0.2134)}


def load(n):
    return pd.read_csv(env.HERE + "/submissions/submission_%s.csv" % n if hasattr(env, "HERE") else
                       "submissions/submission_%s.csv" % n).set_index("row_id")


def main():
    _, _, sX = common.load_raw()
    sX = sX.set_index("row_id")
    S = {n: pd.read_csv("submissions/submission_%s.csv" % n).set_index("row_id").loc[sX.index] for n in LB}
    for tgt, j in (("sub_temp", 0), ("sub_ec", 1)):
        for a, b in (("04", "05"), ("04", "06"), ("03", "04")):
            d = (S[b][tgt] - S[a][tgt]).values
            if np.abs(d).max() < 1e-12:
                continue
            ra, rb = LB[a][j], LB[b][j]
            cov = (rb ** 2 - ra ** 2 - np.mean(d ** 2)) / 2
            m = np.abs(d) > 1e-9
            print("%s %s->%s | LB %.4f->%.4f | rows changed %d (%.0f%%) | mean d on changed %+.3f | "
                  "mean(d*e_%s) on changed %+.4f" % (tgt, a, b, ra, rb, m.sum(), 100 * m.mean(), d[m].mean(), a,
                                                     cov / m.mean()), flush=True)
            if tgt == "sub_temp":
                band = pd.cut(sX.in_temp.values, [-99, 6, 8, 10, 12, 15, 99])
                t = pd.DataFrame({"band": band, "d": d}).groupby("band", observed=False).d.agg(["mean", "count"])
                print("   change by test in_temp band:", {str(k): round(v, 3) for k, v in t["mean"].items()})
            if a == "04" and b == "05":
                # d one-signed? then implied mean error of 04 on changed rows
                if (d[m] < 0).all() or (d[m] > 0).all():
                    print("   d one-signed -> implied change-weighted mean error of 04 on those rows: %+.3f"
                          % (cov / m.mean() / d[m].mean()))


if __name__ == "__main__":
    main()
