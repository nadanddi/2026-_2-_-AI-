# -*- coding: utf-8 -*-
"""Analysis Q2: are the day-level temperature errors a property of the SOURCE?

anal_q1_errors.py: 58% of the remaining error is day-level, the worst days are
whole-day offsets (F47 191: -3.26 C all day), and the 2nd pass (all test days)
is much harder (0.864 vs 0.609; F47 2nd pass 1.051).  Earlier the day-level
error looked uncorrelated across days (lag-3 autocorrelation 0.05-0.09), but
that was measured along the day INDEX, which interleaves sources.

Here, with the calendar agent's analysis-only source chains
(local/deep_cal_11_days.csv: cal = calendar date, chain = source chain;
local/deep_cal_10_links.csv: consecutive-date links within a source), the
day-level residual of the diagnostic OOF (local/oof_temp_diag.npz, clean rows)
is compared across
  * same-source consecutive dates (chain links)
  * the other source on the same date
  * day-index neighbours d+1, d+2
and the share of day-level variance explained by chain identity is computed.

Analysis only: chain ids come from a global reconstruction and may not be
used as features (rule 5).

Run:  cd research && PYTHONPATH="" <python> anal_q2_sources.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

from harness import load
from cleanw_v6 import weights


def corr(pairs):
    a = np.array(pairs, float)
    a = a[~np.isnan(a).any(1)]
    return (float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), len(a)) if len(a) > 5 else (np.nan, len(a))


def main():
    panel, lab, _ = load()
    z = np.load(env.LOCAL + "/oof_temp_diag.npz", allow_pickle=True)
    assert (z["row_id"] == lab.row_id.values).all()
    clean = weights(lab, 3, 0.0) >= 1
    y = lab.sub_temp.values
    days = pd.read_csv(env.LOCAL + "/deep_cal_11_days.csv")[["farm", "day", "cal", "chain", "is_test"]]
    links = pd.read_csv(env.LOCAL + "/deep_cal_10_links.csv")

    for tag in ("r3", "w02"):
        e = z[tag] - y
        g = ~np.isnan(e) & clean
        dres = (pd.DataFrame({"farm": lab.farm.values[g], "day": lab.day.values[g], "e": e[g]})
                  .groupby(["farm", "day"]).e.agg(["mean", "size"]).reset_index())
        dres = dres[dres["size"] >= 12]
        R = dres.set_index(["farm", "day"])["mean"]
        D = days.merge(dres, on=["farm", "day"], how="left")
        print("\n######## %s: day-level residual (clean rows, days with >=12 scored hours: %d) ########"
              % (tag, int(D["mean"].notna().sum())))

        link_pairs = [(R.get((r.farm, r.d_from), np.nan), R.get((r.farm, r.d_to), np.nan))
                      for r in links.itertuples()]
        same_date = []
        for (f, c), grp in D.groupby(["farm", "cal"]):
            v = grp["mean"].dropna().values
            if len(v) >= 2:
                same_date.append((v[0], v[1]))
        idx1 = [(R.get((f, d), np.nan), R.get((f, d + 1), np.nan)) for (f, d) in R.index]
        idx2 = [(R.get((f, d), np.nan), R.get((f, d + 2), np.nan)) for (f, d) in R.index]
        for nm, pr in (("same source, next calendar date", link_pairs),
                       ("other source, same calendar date", same_date),
                       ("day index d vs d+1", idx1), ("day index d vs d+2", idx2)):
            c, n = corr(pr)
            print("  corr %-34s %+.3f  (n=%d)" % (nm, c, n))

        v = D.dropna(subset=["mean"])
        tot = float(((v["mean"] - v["mean"].mean()) ** 2).sum())
        cm = v.groupby("chain")["mean"].transform("mean")
        big = v.groupby("chain")["mean"].transform("size") >= 3
        within = float(((v["mean"] - cm) ** 2)[big].sum())
        between_share = 1 - within / float(((v["mean"] - v["mean"].mean()) ** 2)[big].sum())
        print("  chain identity explains %.1f%% of day-level residual variance (chains with >=3 scored days)"
              % (100 * between_share))

        cs = v.groupby(["farm", "chain"]).agg(n=("mean", "size"), bias=("mean", "mean"),
                                              sd=("mean", "std")).reset_index()
        tst = days[days.is_test].groupby(["farm", "chain"]).size().rename("test_days").reset_index()
        cs = cs.merge(tst, on=["farm", "chain"], how="left").fillna({"test_days": 0})
        cs = cs[cs.n >= 3].sort_values("bias")
        if tag == "r3":
            print("\n  chains with >=3 scored days, sorted by mean day-level bias (pred - true):")
            print(cs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
