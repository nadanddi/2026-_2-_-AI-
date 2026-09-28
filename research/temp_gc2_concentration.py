# -*- coding: utf-8 -*-
"""Is the temperature candidate's gain spread out or carried by a few days?
(Codex raised this for EC: top-5 days held 58-76% of the gain.)

G_C2 (catalog 6.70) vs the pre-TabPFN reference 0.8*MASK base + 0.2*Codex,
DIAG10 / EXT10 / EXT12, base 7 / Codex 726, members samples 1-8 and 17-24.
Per (farm, day): SSE reduction.  Reported: share of days improved, share of
the total positive reduction held by the top-5 / top-10 days, the gain with
the top-5 days removed, and the gain per farm and per record half
(day < 179 first pass, >= 179 second pass - test days are all second pass).
Analysis only.

Run:  cd research && PYTHONPATH="" <python> -u temp_gc2_concentration.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

from common import rmse
from harness import load


def main():
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (lab.row_id.values == z["row_id"]).all()
    y = lab.sub_temp.values
    t = lab.in_temp.values
    g8 = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    for s in ("DIAG10", "EXT10", "EXT12"):
        base, cx = z["%s__MASK__7" % s], z["%s__CODEX__726" % s]
        for name, m in (("1-8", np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s).mean(0)),
                        ("17-24", np.load(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s)[:8].mean(0))):
            g = ~np.isnan(base) & ~np.isnan(m)
            ref = 0.8 * base + 0.2 * cx
            cand = (0.6 - 0.2 * (1 - g8)) * base + (0.2 + 0.4 * (1 - g8)) * cx + 0.2 * g8 * m
            d = pd.DataFrame({"farm": lab.farm.values[g], "day": lab.day.values[g],
                              "red": ((ref - y) ** 2 - (cand - y) ** 2)[g], "e_ref": ((ref - y) ** 2)[g]})
            per = d.groupby(["farm", "day"]).agg(red=("red", "sum"), e_ref=("e_ref", "sum"))
            pos = per.red.clip(lower=0).sort_values(ascending=False)
            top5, top10 = pos.head(5).sum() / pos.sum(), pos.head(10).sum() / pos.sum()
            drop5 = per.drop(pos.head(5).index)
            gain_all = 100 * (1 - np.sqrt((per.e_ref.sum() - per.red.sum()) / per.e_ref.sum()))
            gain_d5 = 100 * (1 - np.sqrt((drop5.e_ref.sum() - drop5.red.sum()) / drop5.e_ref.sum()))
            by = {}
            for key, grp in (("F13", per.loc["F13"]), ("F47", per.loc["F47"])):
                by[key] = 100 * (1 - np.sqrt((grp.e_ref.sum() - grp.red.sum()) / grp.e_ref.sum()))
            idx_day = per.index.get_level_values(1)
            for key, m2 in (("pass1", idx_day < 179), ("pass2", idx_day >= 179)):
                grp = per[m2]
                by[key] = 100 * (1 - np.sqrt((grp.e_ref.sum() - grp.red.sum()) / grp.e_ref.sum())) if len(grp) else np.nan
            print("%-6s m %-5s | days %d, improved %.0f%% | top5 %.0f%% top10 %.0f%% of gain | RMSE gain %.2f%% (without top5 %.2f%%) | %s"
                  % (s, name, len(per), 100 * (per.red > 0).mean(), 100 * top5, 100 * top10, gain_all, gain_d5,
                     " ".join("%s %.2f%%" % kv for kv in by.items())), flush=True)


if __name__ == "__main__":
    main()
