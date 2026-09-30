# -*- coding: utf-8 -*-
"""Where does the current temperature model (G_C2, round 6, 0.5251) still fail?

Step 1 of "big errors -> common conditions -> hypotheses" (lab Claude, 2026-09-30).
Exploratory diagnosis only: no model is selected or tuned here.

G_C2 out-of-fold prediction rebuilt from saved members (catalog 6.70):
  g = clip((in_temp - 8)/2, 0, 1)
  pred = (0.6 - 0.2(1-g)) MASK base + (0.2 + 0.4(1-g)) Codex + 0.2 g TabPFN
  base/Codex: mean of seed pairs (7,726) and (101,727); TabPFN: v2 samples 1-8.
Validators: DIAG10 (non-overlapping, covers all 400 labelled farm-days) and
EXT12 (days colder than 12 C held out whole).

Outputs: logs/gc2_error_anatomy_v1.log, results/gc2_error_days_v1.csv
Run:  PYTHONPATH="" <python> -u gc2_error_anatomy_v1.py   (from 연구실/클로드/code)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "집", "클로드", "research"))
import env  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import load  # noqa: E402
import train_flags_v6 as TF  # noqa: E402

OUT_RES = os.path.join(HERE, "..", "results")


def gc2_oof(z, split):
    base = np.nanmean([z["%s__MASK__7" % split], z["%s__MASK__101" % split]], axis=0)
    cx = np.nanmean([z["%s__CODEX__726" % split], z["%s__CODEX__727" % split]], axis=0)
    pfn = np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % split).mean(0)
    return base, cx, pfn


def rmse(e):
    return float(np.sqrt(np.nanmean(e ** 2)))


def main():
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (lab.row_id.values == z["row_id"]).all()
    y = lab.sub_temp.values
    t = lab.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    w = TF.row_weights(lab, 0.2, w_noisy=0.2)
    lab = lab.assign(flag=(w < 1).astype(int))

    for split in ("DIAG10", "EXT12"):
        base, cx, pfn = gc2_oof(z, split)
        pred = (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn
        ok = ~np.isnan(pred)
        d = lab.loc[ok].copy()
        d["pred"], d["base"], d["cx"], d["pfn"] = pred[ok], base[ok], cx[ok], pfn[ok]
        d["e"] = d.pred - d.sub_temp
        print("=" * 100)
        print("%s  rows %d  days %d  G_C2 RMSE %.4f | base %.4f codex %.4f tabpfn %.4f"
              % (split, len(d), d.groupby(["farm", "day"]).ngroups, rmse(d.e),
                 rmse(d.base - d.sub_temp), rmse(d.cx - d.sub_temp), rmse(d.pfn - d.sub_temp)))

        # --- day offset vs within-day shape ---
        d["e_day"] = d.groupby(["farm", "day"]).e.transform("mean")
        d["e_in"] = d.e - d.e_day
        sse = (d.e ** 2).sum()
        print("  SSE share: day offset %.1f%% | within-day %.1f%%"
              % (100 * (d.e_day ** 2).sum() / sse, 100 * (d.e_in ** 2).sum() / sse))

        # --- farm / pass ---
        d["pass"] = np.where(d.day < 179, "pass1", "pass2")
        for k, grp in d.groupby(["farm", "pass"]):
            print("  %-14s rows %5d  RMSE %.3f  bias %+.3f" % ("%s %s" % k, len(grp), rmse(grp.e), grp.e.mean()))

        # --- rows: how concentrated ---
        e2 = np.sort((d.e ** 2).values)[::-1]
        print("  top 1%% rows hold %.1f%% of SSE, top 5%% hold %.1f%%, top 10%% hold %.1f%%"
              % tuple(100 * e2[:int(len(e2) * q)].sum() / e2.sum() for q in (0.01, 0.05, 0.10)))

        # --- days ---
        day = d.groupby(["farm", "day"]).agg(
            n=("e", "size"), sse=("e", lambda x: (x ** 2).sum()), bias=("e", "mean"),
            in_rmse=("e_in", lambda x: np.sqrt((x ** 2).mean())), y_mean=("sub_temp", "mean"),
            t_mean=("in_temp", "mean"), t_min=("in_temp", "min"), t_max=("in_temp", "max"),
            out_mean=("out_temp", "mean"), heat=("act_heating", "mean"), vent=("act_vent", "mean"),
            thermal=("act_thermal", "mean"), rad=("out_rad", "mean"), hum=("in_hum", "mean"),
            co2=("in_co2", "mean"), flag=("flag", "mean"))
        day["rmse"] = np.sqrt(day.sse / day.n)
        day["gap"] = day.y_mean - day.t_mean          # root-zone minus air, daily mean
        day = day.sort_values("sse", ascending=False)
        tot = day.sse.sum()
        print("  days: top 10 hold %.1f%% of SSE, top 20 %.1f%%, top 40 (10%%) %.1f%%"
              % tuple(100 * day.sse.head(k).sum() / tot for k in (10, 20, 40)))
        cols = ["n", "rmse", "bias", "in_rmse", "y_mean", "t_mean", "t_min", "out_mean", "gap", "heat", "vent", "flag"]
        print("  worst 20 farm-days:")
        print(day.head(20)[cols].round(2).to_string())
        day.to_csv(os.path.join(OUT_RES, "gc2_error_days_v1_%s.csv" % split), encoding="utf-8-sig")

        # --- hour ---
        hr = d.groupby("hour").e.agg(lambda x: np.sqrt((x ** 2).mean()))
        hb = d.groupby("hour").e.mean()
        print("  RMSE by hour: " + " ".join("%d:%.2f" % (h, v) for h, v in hr.items()))
        print("  bias by hour: " + " ".join("%d:%+.2f" % (h, v) for h, v in hb.items()))

        # --- bands: current air temp and true root-zone temp ---
        for name, col, bins in (("in_temp", "in_temp", [-99, 6, 8, 10, 12, 15, 20, 25, 99]),
                                ("label", "sub_temp", [-99, 8, 10, 12, 15, 20, 25, 99])):
            c = pd.cut(d[col], bins)
            gb = d.groupby(c, observed=True).e
            print("  by %-7s " % name + " | ".join("%s n%d %.2f/%+.2f" % (k, len(v), rmse(v.values), v.mean())
                                                  for k, v in gb))


if __name__ == "__main__":
    main()
