# -*- coding: utf-8 -*-
"""With TabPFN at 0.2 (W20_8, catalog 6.62), is the base/Codex split still
right?  Saved OOFs only.  Two independent 8-sample members: samples 1-8
(mean of the v2 bags) and 17-24 (v6 rows 0-7).  Base seeds 7/101 paired with
Codex seeds 726/727.

Arms (fixed), reference W20_8 = 0.6*base + 0.2*Codex + 0.2*member8:
  C30 : 0.5*base + 0.3*Codex + 0.2*member8
  C10 : 0.7*base + 0.1*Codex + 0.2*member8
Cells DIAG10/EXT10/EXT12 x 2 seed pairs x 2 members.  Rule: better in all
12 cells, DIAG10 p_worse < 0.025/2.

Run:  cd research && PYTHONPATH="" <python> -u temp_rebalance_v1.py
"""
import env  # noqa: F401
import numpy as np

from common import rmse
from harness import load
from screen_v6 import boot

ARMS = {"C30": (0.5, 0.3), "C10": (0.7, 0.1)}


def main():
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (lab.row_id.values == z["row_id"]).all()
    y = lab.sub_temp.values
    ok = {a: True for a in ARMS}
    for s in ("DIAG10", "EXT10", "EXT12"):
        mems = {"1-8": np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s).mean(0),
                "17-24": np.load(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s)[:8].mean(0)}
        for bs, cs in ((7, 726), (101, 727)):
            base, cx = z["%s__MASK__%d" % (s, bs)], z["%s__CODEX__%d" % (s, cs)]
            for k, m in mems.items():
                g = ~np.isnan(base) & ~np.isnan(m)
                ref = 0.6 * base + 0.2 * cx + 0.2 * m
                txt = []
                for a, (wb, wc) in ARMS.items():
                    c = wb * base + wc * cx + 0.2 * m
                    d = rmse(c[g], y[g]) / rmse(ref[g], y[g]) - 1
                    pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], c[g])[3] if s == "DIAG10" else np.nan
                    ok[a] = ok[a] and d < 0 and (s != "DIAG10" or pw < 0.0125)
                    txt.append("%s %+.2f%% (p_worse %.4f)" % (a, 100 * d, pw))
                print("%-6s seeds %3d/%d member %-5s | W20_8 %.5f | %s"
                      % (s, bs, cs, k, rmse(ref[g], y[g]), " | ".join(txt)), flush=True)
    for a, v in ok.items():
        print("%s PRE-SET RULE VERDICT: %s" % (a, "ADOPT" if v else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
