# -*- coding: utf-8 -*-
"""EC-SH1: bias-variance - shrink EC v2 predictions toward the training mean.
2026-10-01 집 클로드 (user relayed a professor's advice: "if the error is
large, accept more bias to cut variance").

Boosting (LightGBM) and bagging (ExtraTrees) are already inside EC v2 (R3
members) and extra RF / HistGB blends were rejected (6.121), so the untested
part of the advice is explicit shrinkage:
    p' = m + lam * (v2 - m),  m = farm training mean of the fold (farm_mean)
lam < 1 adds bias / removes spread; lam > 1 is the opposite direction,
reported so the answer is not one-sided.

Data: Codex phase-3 OOF (집/코덱스/local/ec_restart_phase3_20261001_v1/
oof_predictions.csv), validators DIAG10, A, B, EXT10, EXT12 (pooled
occurrence RMSE as in Codex's report).  Final-lock 40 days are not in it.

Decision rule (fixed before running), variants lam in {0.8, 0.9, 1.1, 1.2}
(k = 4): a variant passes only if RMSE improves on all five validators AND
DIAG10 paired 5-day farm-block bootstrap (5000) P(worse) < 0.025/4.
Also reported (descriptive): late segment (day >= 179) and high-EC days.

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_shrink1_v1.py
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd

OOF = os.path.join(env.ROOT, u"집", u"코덱스", "local", "ec_restart_phase3_20261001_v1", "oof_predictions.csv")
LAMS = (0.8, 0.9, 1.1, 1.2)
VALS = ("DIAG10", "A", "B", "EXT10", "EXT12")
RNG = np.random.default_rng(20261001)


def rm(e):
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    o = pd.read_csv(OOF, encoding="utf-8-sig")
    for lam in LAMS:
        o["s%.1f" % lam] = o.farm_mean + lam * (o.v2 - o.farm_mean)
    print("RMSE by validator")
    print("%-8s %6s %8s" % ("val", "rows", "v2") + "".join("  lam%.1f" % l for l in LAMS))
    better = {l: [] for l in LAMS}
    for v in VALS:
        g = o[o.validator == v]
        base = rm(g.v2 - g.sub_ec)
        r = [rm(g["s%.1f" % l] - g.sub_ec) for l in LAMS]
        for l, x in zip(LAMS, r):
            better[l].append(x < base)
        print("%-8s %6d %8.4f" % (v, len(g), base) + "".join(" %7.4f" % x for x in r))

    g = o[o.validator == "DIAG10"].copy()
    for nm, s in (("late>=179", g.day >= 179), ("early", g.day < 179)):
        h = g[s]
        print("DIAG10 %-9s v2 %.4f " % (nm, rm(h.v2 - h.sub_ec)) +
              " ".join("lam%.1f %.4f" % (l, rm(h["s%.1f" % l] - h.sub_ec)) for l in LAMS))
    dm = g.groupby(["farm", "day"]).sub_ec.transform("mean")
    for nm, s in (("highEC>=1.2", dm >= 1.2), ("rest", dm < 1.2)):
        h = g[s]
        print("DIAG10 %-11s v2 %.4f " % (nm, rm(h.v2 - h.sub_ec)) +
              " ".join("lam%.1f %.4f" % (l, rm(h["s%.1f" % l] - h.sub_ec)) for l in LAMS))

    g["cl"] = g.farm + "_" + (g.day // 5).astype(str)
    print("\nDIAG10 bootstrap (5-day farm blocks), threshold %.5f" % (0.025 / len(LAMS)))
    for l in LAMS:
        d = (g["s%.1f" % l] - g.sub_ec) ** 2 - (g.v2 - g.sub_ec) ** 2
        cl = d.groupby(g.cl).agg(["sum", "count"])
        s, n = cl["sum"].values, cl["count"].values
        bs = np.array([s[k].sum() / n[k].sum() for k in (RNG.integers(0, len(s), len(s)) for _ in range(5000))])
        p = float((bs >= 0).mean())
        ok = all(better[l]) and p < 0.025 / len(LAMS)
        print("  lam %.1f  dMSE %+.5f  95%% [%+.5f, %+.5f]  P(worse) %.4f  all-5-better %s  -> %s"
              % (l, d.mean(), np.quantile(bs, .025), np.quantile(bs, .975), p, all(better[l]),
                 "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
