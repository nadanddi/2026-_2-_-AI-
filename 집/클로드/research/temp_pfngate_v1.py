# -*- coding: utf-8 -*-
"""Protect cold rows from TabPFN extrapolation (temp_cold_band_check.py: on
EXT10/EXT12 rows with in_temp <= 8 C the 8-sample TabPFN member scores
1.4-2.5 with bias +1..+1.9 C, and W10 / W20_8 are WORSE than no TabPFN
there; 15% of test rows are <= 8 C).

Gate (POST-HOC thresholds, set after seeing that check): TabPFN share
multiplied by g = clip((in_temp - 8) / 2, 0, 1) of the row's CURRENT indoor
temperature (a current input, rule-compliant); the removed share goes to
  G_B : the MASK base          G_C : the Codex member (best in the cold)
Applied to W20_8 (0.6 base / 0.2 Codex / 0.2 TabPFN-8).
Rule (fixed before running, protection change): vs ungated W20_8,
  EXT10 and EXT12 better in all cells, DIAG10 not worse than +0.10%,
cells = 2 seed pairs (7/726, 101/727) x 2 members (samples 1-8, 17-24).
Also shown: each arm vs the pre-TabPFN reference and vs W10.

Run:  cd research && PYTHONPATH="" <python> -u temp_pfngate_v1.py
"""
import env  # noqa: F401
import numpy as np

from common import rmse
from harness import load


def main():
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (lab.row_id.values == z["row_id"]).all()
    y = lab.sub_temp.values
    gate = np.clip((lab.in_temp.values - 8.0) / 2.0, 0.0, 1.0)
    gate = np.where(np.isnan(gate), 1.0, gate)
    ok = {"G_B": True, "G_C": True}
    for s in ("DIAG10", "EXT10", "EXT12"):
        mems = {"1-8": (np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s).mean(0),
                        np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s)[0]),
                "17-24": (np.load(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s)[:8].mean(0),
                          np.load(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s)[:4].mean(0))}
        for bs, cs in ((7, 726), (101, 727)):
            base, cx = z["%s__MASK__%d" % (s, bs)], z["%s__CODEX__%d" % (s, cs)]
            for k, (m8, m4) in mems.items():
                g = ~np.isnan(base) & ~np.isnan(m8)
                ref = 0.6 * base + 0.2 * cx + 0.2 * m8
                old = 0.8 * base + 0.2 * cx
                w10 = 0.7 * base + 0.2 * cx + 0.1 * m4
                p = 0.2 * gate
                arms = {"G_B": (0.8 - p) * base + 0.2 * cx + p * m8,
                        "G_C": 0.6 * base + (0.4 - p) * cx + p * m8}
                txt = []
                for a, c in arms.items():
                    d = rmse(c[g], y[g]) / rmse(ref[g], y[g]) - 1
                    ok[a] = ok[a] and (d < 0 if s != "DIAG10" else d <= 0.001)
                    txt.append("%s %+.2f%% vs W20_8, %+.2f%% vs noPFN, %+.2f%% vs W10"
                               % (a, 100 * d, 100 * (rmse(c[g], y[g]) / rmse(old[g], y[g]) - 1),
                                  100 * (rmse(c[g], y[g]) / rmse(w10[g], y[g]) - 1)))
                print("%-6s seeds %3d/%d m %-5s | W20_8 %.5f | %s" % (s, bs, cs, k, rmse(ref[g], y[g]), " | ".join(txt)),
                      flush=True)
    for a, v in ok.items():
        print("%s PRE-SET RULE VERDICT: %s" % (a, "ADOPT" if v else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
