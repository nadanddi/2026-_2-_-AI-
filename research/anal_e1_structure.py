# -*- coding: utf-8 -*-
"""EC gap analysis, step 1: where does the round-3 EC error come from?

  1. label structure: between-day vs within-day variance, first vs second
     pass (all test days are second pass), level over the record
  2. round-3 EC pipeline (ET 14f+fp .60, LGB tweedie .30, MLP .10, causal
     shrink, clip; seed 7) out-of-fold on the non-overlapping diagnostic folds
  3. error decomposition: day-level offset share, EC level bands, pass,
     noisy training days / restored rows (train_flags_v6), hour of day
Saves local/oof_ec_diag.npz.

Run:  cd research && PYTHONPATH="" <python> -u anal_e1_structure.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load
import features_v4 as F4
from anal_q1_errors import diag_folds
from ec_v6 import pipeline
import train_flags_v6 as TF


def share(d, key):
    tot = float((d.err ** 2).sum())
    g = d.groupby(key, observed=True)
    out = pd.DataFrame({"n": g.size(), "rmse": g.err.apply(lambda e: np.sqrt((e ** 2).mean())),
                        "bias": g.err.mean(), "ec_mean": g.y.mean(),
                        "SSE%": g.err.apply(lambda e: 100 * (e ** 2).sum() / tot)})
    print(out.round(3).to_string())


def main():
    panel, _, lab0 = load()
    print("USABLE:", list(USABLE))
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left")
    fpc = F4.names(fp)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]

    # 1. label structure
    print("\n== label structure ==")
    for f in TARGET_FARMS:
        g = lab[lab.farm == f]
        dm = g.groupby("day").sub_ec.transform("mean")
        vb, vw = float(dm.var()), float((g.sub_ec - dm).var())
        print("%s: total var %.4f | between-day %.4f (%.0f%%) | within-day %.4f"
              % (f, g.sub_ec.var(), vb, 100 * vb / (vb + vw), vw))
    lab["pass2"] = lab.day >= 179
    print(lab.groupby(["farm", "pass2"]).sub_ec.agg(["size", "mean", "std", "median"]).round(3).to_string())
    lab["dbin"] = pd.cut(lab.day, [0, 60, 120, 178, 210, 250])
    print(lab.groupby(["farm", "dbin"], observed=True).sub_ec.agg(["size", "mean", "std"]).round(3).to_string())
    # day-to-day persistence of the daily mean
    for f in TARGET_FARMS:
        s = lab[lab.farm == f].groupby("day").sub_ec.mean()
        s = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
        print("%s daily-mean autocorr lag1 %.2f lag2 %.2f lag7 %.2f"
              % (f, s.autocorr(1), s.autocorr(2), s.autocorr(7)))

    # 2. OOF
    fds = diag_folds(lab)
    fn = pipeline(f14 + fpc, f14)
    oof = np.full(len(lab), np.nan)
    for i, fd in enumerate(fds):
        trm, vam = split_mask(lab, fd)
        oof[np.where(vam)[0]] = fn(lab[trm], lab[vam].reset_index(drop=True))
        print("  fold %d" % i, flush=True)
    np.savez(env.LOCAL + "/oof_ec_diag.npz", row_id=lab.row_id.values, oof=oof)
    y = lab.sub_ec.values
    print("\nOOF RMSE all %.4f | F13 %.4f | F47 %.4f" % (rmse(oof, y), rmse(oof[lab.farm == "F13"], y[lab.farm == "F13"]),
                                                       rmse(oof[lab.farm == "F47"], y[lab.farm == "F47"])))

    # 3. decomposition
    d = lab[["farm", "day", "hour", "pass2"]].copy()
    d["y"], d["p"] = y, oof
    d["err"] = d.p - d.y
    dmean = d.groupby(["farm", "day"]).err.transform("mean")
    print("\nday-offset share of SSE: %.1f%%  (within-day %.1f%%)"
          % (100 * (dmean ** 2).sum() / (d.err ** 2).sum(), 100 * ((d.err - dmean) ** 2).sum() / (d.err ** 2).sum()))
    ymean = d.groupby(["farm", "day"]).y.transform("mean")
    pmean = d.groupby(["farm", "day"]).p.transform("mean")
    print("within-day shape corr (demeaned p vs y): %.3f" % np.corrcoef(d.p - pmean, d.y - ymean)[0, 1])
    print("\n-- by pass --"); share(d, "pass2")
    d["band"] = pd.cut(d.y, [0, 0.25, 0.5, 0.75, 1.0, 1.5, 4])
    print("\n-- by true EC band --"); share(d, "band")
    d["hb"] = pd.cut(d.hour, [-1, 2, 5, 9, 13, 17, 20, 23])
    print("\n-- by hour --"); share(d, "hb")
    fl = TF.restored_flags().set_index("row_id")
    d["restored"] = fl.loc[lab.row_id, "flag"].values
    nd = TF.noisy_days()
    ns = set(map(tuple, nd[nd.noisy][["farm", "day"]].values))
    d["noisy"] = [(f, dd) in ns for f, dd in zip(d.farm, d.day)]
    print("\n-- restored rows --"); share(d, "restored")
    print("\n-- noisy days --"); share(d, "noisy")
    # worst days
    dd = d.groupby(["farm", "day"]).agg(bias=("err", "mean"), rmse=("err", lambda e: np.sqrt((e ** 2).mean())),
                                         y=("y", "mean"), p=("p", "mean"))
    dd["sse%"] = 100 * d.groupby(["farm", "day"]).err.apply(lambda e: (e ** 2).sum()) / (d.err ** 2).sum()
    dd = dd.sort_values("sse%", ascending=False)
    print("\nworst 15 days (%.1f%% of SSE in top 15, %.1f%% in top 40):"
          % (dd["sse%"].head(15).sum(), dd["sse%"].head(40).sum()))
    print(dd.head(15).round(3).to_string())


if __name__ == "__main__":
    main()
