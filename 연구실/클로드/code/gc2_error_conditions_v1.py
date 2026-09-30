# -*- coding: utf-8 -*-
"""Step 2: common conditions of the large G_C2 errors (lab Claude, 2026-09-30).

Builds on gc2_error_anatomy_v1.py (same G_C2 out-of-fold rebuild, DIAG10 main,
EXT12 cold check).  Exploratory diagnosis only - nothing is selected or tuned.

  A. flagged (restored / noisy, weight 0.2) vs clean rows: how much of the
     validation error is on days that barely exist in the test period
  B. clean days only: day bias vs day-level conditions (Spearman), worst-decile
     days vs the rest (medians)
  C. within-day error (error minus its day mean): by hour, vs in_temp rate of
     change, vs outside radiation, at the midnight seam
  D. range compression: day-mean prediction vs day-mean label slope
  E. F47 second pass (the test period's part of the record)

Outputs: logs/gc2_error_conditions_v1.log
Run:  PYTHONPATH="" <python> -u gc2_error_conditions_v1.py   (from 연구실/클로드/code)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "집", "클로드", "research"))
import env  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from harness import load  # noqa: E402
import train_flags_v6 as TF  # noqa: E402


def rmse(e):
    return float(np.sqrt(np.nanmean(np.asarray(e) ** 2)))


def build(lab, z, split, g):
    base = np.nanmean([z["%s__MASK__7" % split], z["%s__MASK__101" % split]], axis=0)
    cx = np.nanmean([z["%s__CODEX__726" % split], z["%s__CODEX__727" % split]], axis=0)
    pfn = np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % split).mean(0)
    pred = (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn
    d = lab.assign(pred=pred)
    d = d[~np.isnan(pred)].copy()
    d["e"] = d.pred - d.sub_temp
    d["e_day"] = d.groupby(["farm", "day"]).e.transform("mean")
    d["e_in"] = d.e - d.e_day
    return d


def day_table(d):
    d = d.sort_values(["farm", "t"])
    d["dT"] = d.groupby("farm").in_temp.diff()                       # hourly change of air temp
    agg = d.groupby(["farm", "day"]).agg(
        n=("e", "size"), sse=("e", lambda x: (x ** 2).sum()), bias=("e", "mean"),
        in_rmse=("e_in", lambda x: np.sqrt((x ** 2).mean())),
        y_mean=("sub_temp", "mean"), p_mean=("pred", "mean"), t_mean=("in_temp", "mean"),
        y_rng=("sub_temp", lambda x: x.max() - x.min()), t_rng=("in_temp", lambda x: x.max() - x.min()),
        p_rng=("pred", lambda x: x.max() - x.min()),
        out_mean=("out_temp", "mean"), rad=("out_rad", "mean"), heat=("act_heating", "mean"),
        vent=("act_vent", "mean"), thermal=("act_thermal", "mean"), shade=("act_shade", "mean"),
        hum=("in_hum", "mean"), co2=("in_co2", "mean"), flag=("flag", "mean"),
        t0=("in_temp", "first"), y0=("sub_temp", "first"))
    agg["gap"] = agg.y_mean - agg.t_mean
    agg["rmse"] = np.sqrt(agg.sse / agg.n)
    # neighbours in the stitched record (same farm, previous labelled row block)
    agg = agg.reset_index().sort_values(["farm", "day"])
    prev = agg.groupby("farm")
    agg["dt_mean_prev"] = agg.t_mean - prev.t_mean.shift(1)            # air level jump vs previous day
    agg["gap_prev"] = prev.gap.shift(1)
    agg["prev_is_adjacent"] = (agg.day - prev.day.shift(1)) == 1
    return agg.set_index(["farm", "day"])


def main():
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (lab.row_id.values == z["row_id"]).all()
    t = lab.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    lab = lab.assign(flag=(w < 1).astype(int))

    for split in ("DIAG10", "EXT12"):
        d = build(lab, z, split, g)
        print("=" * 100)
        print(split)
        # ---------------- A. flagged vs clean
        sse = (d.e ** 2).sum()
        for k, grp in d.groupby("flag"):
            print("  A. flag=%d rows %5d (%.0f%%)  RMSE %.3f  SSE share %.1f%%  day-offset share of its SSE %.0f%%"
                  % (k, len(grp), 100 * len(grp) / len(d), rmse(grp.e), 100 * (grp.e ** 2).sum() / sse,
                     100 * (grp.e_day ** 2).sum() / (grp.e ** 2).sum()))
        clean = d[d.flag == 0]
        print("  A. clean-row RMSE by farm/pass: " + " | ".join(
            "%s %s n%d %.3f" % (f, p, len(gg), rmse(gg.e))
            for (f, p), gg in clean.groupby([clean.farm, np.where(clean.day < 179, "pass1", "pass2")])))

        # ---------------- B. clean days: conditions
        dt = day_table(d)
        cd = dt[dt.flag < 0.5].copy()
        print("  B. clean days %d (of %d); day-offset share of clean SSE %.0f%%"
              % (len(cd), len(dt), 100 * (clean.e_day ** 2).sum() / (clean.e ** 2).sum()))
        feats = ["gap", "t_mean", "y_mean", "out_mean", "rad", "heat", "vent", "thermal", "shade", "hum", "co2",
                 "t_rng", "y_rng", "dt_mean_prev", "gap_prev"]
        rows = []
        for f in feats:
            m = cd[[f, "bias"]].dropna()
            r_b = spearmanr(m[f], m.bias).correlation
            m2 = cd[[f, "rmse"]].dropna()
            r_r = spearmanr(m2[f], m2.rmse).correlation
            rows.append((f, r_b, r_r))
        print("     Spearman with day bias | with day RMSE")
        for f, rb, rr in sorted(rows, key=lambda x: -abs(x[1])):
            print("     %-13s %+.2f | %+.2f" % (f, rb, rr))
        q = cd.rmse.quantile(0.9)
        top, rest = cd[cd.rmse >= q], cd[cd.rmse < q]
        print("     worst 10%% clean days (n=%d, RMSE>=%.2f) vs rest: median" % (len(top), q))
        for f in ["bias", "in_rmse", "gap", "t_mean", "y_mean", "out_mean", "heat", "vent", "thermal", "t_rng",
                  "y_rng", "p_rng", "dt_mean_prev"]:
            print("       %-13s %7.2f  vs %7.2f   |abs| %6.2f vs %6.2f"
                  % (f, top[f].median(), rest[f].median(), top[f].abs().median(), rest[f].abs().median()))
        # gap anomaly = today's gap minus the farm's typical gap: is |bias| explained by it?
        cd["gap_anom"] = cd.gap - cd.groupby(level=0).gap.transform("median")
        rr = spearmanr(cd.gap_anom, cd.bias).correlation
        lin = np.polyfit(cd.gap_anom, cd.bias, 1)
        print("     gap anomaly vs bias: Spearman %+.2f, slope %+.2f (bias ~ %.2f*gap_anom)" % (rr, lin[0], lin[0]))

        # ---------------- C. within-day error (clean rows)
        c = clean.sort_values(["farm", "t"]).copy()
        c["dT"] = c.groupby("farm").in_temp.diff()
        hr = c.groupby("hour").e_in.agg(lambda x: np.sqrt((x ** 2).mean()))
        print("  C. clean within-day RMSE by hour: " + " ".join("%d:%.2f" % kv for kv in hr.items()))
        print("     within-day bias by hour:        " + " ".join("%d:%+.2f" % kv for kv in c.groupby("hour").e_in.mean().items()))
        for f in ["dT", "out_rad", "in_temp", "act_vent", "act_heating", "act_thermal"]:
            m = c[[f, "e_in"]].dropna()
            print("     within-day e vs %-11s Spearman %+.2f" % (f, spearmanr(m[f], m.e_in).correlation))
        # dT bins
        m = c.dropna(subset=["dT"])
        b = pd.cut(m.dT, [-99, -2, -1, -0.3, 0.3, 1, 2, 99])
        print("     within-day by dT(°C/h): " + " | ".join("%s n%d %+.2f/%.2f" % (k, len(v), v.mean(), rmse(v))
                                                        for k, v in m.groupby(b, observed=True).e_in))
        # midnight seam: hour 0 error vs air jump from previous hour
        h0 = c[c.hour == 0]
        m = h0.dropna(subset=["dT"])
        print("     hour 0: RMSE %.2f (all-hour %.2f) | Spearman(e, air jump 23h->0h) %+.2f | |jump|>2°C rows %d RMSE %.2f"
              % (rmse(h0.e), rmse(c.e), spearmanr(m.dT, m.e).correlation, int((m.dT.abs() > 2).sum()),
                 rmse(m.e[m.dT.abs() > 2])))

        # ---------------- D. range compression
        k1 = np.polyfit(cd.y_mean, cd.p_mean, 1)[0]
        k2 = np.polyfit(cd.y_rng, cd.p_rng, 1)[0]
        print("  D. day-mean pred vs label slope %.3f (1 = no compression) | day range slope %.3f | "
              "median range: label %.2f pred %.2f air %.2f" % (k1, k2, cd.y_rng.median(), cd.p_rng.median(),
                                                             cd.t_rng.median()))

        # ---------------- E. F47 second pass
        f47 = dt.loc["F47"]
        f47 = f47[f47.index >= 179].sort_values("sse", ascending=False)
        print("  E. F47 pass2 days %d, RMSE %.3f; worst 8:" % (len(f47), np.sqrt(f47.sse.sum() / f47.n.sum())))
        print(f47.head(8)[["rmse", "bias", "in_rmse", "y_mean", "t_mean", "gap", "out_mean", "heat", "vent", "flag"]]
              .round(2).to_string())


if __name__ == "__main__":
    main()
