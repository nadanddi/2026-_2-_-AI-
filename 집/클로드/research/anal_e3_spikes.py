# -*- coding: utf-8 -*-
"""EC gap analysis, step 3: can the inputs tell a high-EC day?

High-EC rows (>1.5) are 5% of rows and 63% of the round-3 OOF squared error
(step 1).  They sit in the late-season stretches (days 120-178 and 210-250),
which is where test blocks 3 and 4 lie.

  1. day-level input summaries (mean / min / max / 00 h value of each input,
     share of zero actuator hours, CO2-difference autocorrelation)
  2. inside the late-season stretches only (season held fixed): AUC of each
     summary for spike days (daily mean EC > 1.2), and Spearman correlation
     with the OOF daily residual (what the model misses)
  3. test blocks 3/4: where their summaries fall relative to training spike /
     non-spike days (inputs only; no test label exists)

Run:  cd research && PYTHONPATH="" <python> -u anal_e3_spikes.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import USABLE, TARGET_FARMS
from harness import load

ACT = [c for c in USABLE if c.startswith("act_")]


def day_table(df):
    rows = []
    for (f, d), g in df.groupby(["farm", "day"]):
        g = g.sort_values("hour")
        r = dict(farm=f, day=int(d))
        for c in USABLE:
            r[c + "_mean"] = g[c].mean()
            r[c + "_max"] = g[c].max()
            r[c + "_min"] = g[c].min()
            h0 = g[g.hour == 0][c]
            r[c + "_h0"] = h0.iloc[0] if len(h0) else np.nan
        for c in ACT:
            r[c + "_zero"] = (g[c].fillna(0) == 0).mean()
        d1 = g.in_co2.diff()
        r["co2_ac1"] = d1.autocorr(1) if d1.notna().sum() >= 8 else np.nan
        r["dT_in_out"] = (g.in_temp - g.out_temp).mean()
        r["day_night_T"] = g[g.hour.between(10, 15)].in_temp.mean() - g[(g.hour <= 4)].in_temp.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    panel, _, lab = load()
    z = np.load(env.LOCAL + "/oof_ec_diag.npz", allow_pickle=True)
    oof = pd.Series(z["oof"], index=z["row_id"])
    lab = lab.copy()
    lab["p"] = oof.loc[lab.row_id].values
    tr = day_table(lab)
    lv = lab.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("p", "mean"), ymax=("sub_ec", "max")).reset_index()
    tr = tr.merge(lv, on=["farm", "day"])
    tr["res"] = tr.y - tr.p
    tr["spike"] = tr.y > 1.2
    late = tr[tr.day.between(120, 178) | tr.day.between(210, 250)].copy()
    print("late-season labelled days: %d, spike days %d" % (len(late), int(late.spike.sum())))
    feats = [c for c in tr.columns if c not in ("farm", "day", "y", "p", "ymax", "res", "spike")]
    rows = []
    for c in feats:
        s = late[[c, "spike", "res"]].dropna()
        if s[c].nunique() < 3 or s.spike.nunique() < 2:
            continue
        auc = roc_auc_score(s.spike, s[c])
        rho = spearmanr(s[c], s.res).correlation
        rows.append(dict(feat=c, auc=auc, auc_dev=abs(auc - 0.5), rho_res=rho))
    R = pd.DataFrame(rows)
    print("\n-- top 15 summaries separating spike days (late season) --")
    print(R.sort_values("auc_dev", ascending=False).head(15).round(3).to_string(index=False))
    print("\n-- top 12 summaries correlated with what the model misses (daily residual) --")
    R["arho"] = R.rho_res.abs()
    print(R.sort_values("arho", ascending=False).head(12).round(3).to_string(index=False))
    # permutation null for the best AUC (how big would the max be by chance?)
    rng = np.random.RandomState(0)
    mx = []
    X = late[feats]
    for _ in range(200):
        sp = rng.permutation(late.spike.values)
        best = 0
        for c in feats:
            s = X[c].values
            m = ~np.isnan(s)
            if len(np.unique(s[m])) < 3:
                continue
            best = max(best, abs(roc_auc_score(sp[m], s[m]) - 0.5))
        mx.append(best)
    print("\nnull max |AUC-0.5| over %d summaries: 95th pct %.3f" % (len(feats), np.percentile(mx, 95)))

    te = day_table(panel[panel.is_test])
    top = R.sort_values("auc_dev", ascending=False).feat.head(6).tolist()
    print("\n-- test days vs late-season training days (medians) --")
    te["blk"] = np.where(te.day >= 215, "test blk3/4", "test blk1/2")
    comp = pd.concat([late.assign(grp=np.where(late.spike, "train spike", "train late non-spike"))[["grp"] + top],
                      te.rename(columns={"blk": "grp"})[["grp"] + top]])
    print(comp.groupby("grp")[top].median().round(3).T.to_string())


if __name__ == "__main__":
    main()
