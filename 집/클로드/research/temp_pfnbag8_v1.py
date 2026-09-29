# -*- coding: utf-8 -*-
"""web_tabpfn_v5_gpu.py rejected W20: on cold EXT10 days the 4-sample bag is
still sample-sensitive (member 1.010 vs 0.925 between two fresh bags).
Does an 8-sample bag stabilise it?  Saved OOFs already hold 16 samples in
four 4-sample bags: v2 {1-4},{5-8}; v5 {9-12},{13-16}.  Two INDEPENDENT
8-sample members:  M_A = mean(v2 bags)  (samples 1-8),
                   M_B = mean(v5 bags)  (samples 9-16).

Arms (fixed): with an 8-sample member
  W10_8 : 0.7*base + 0.2*Codex + 0.1*M        vs candidate W10 (4-sample)
  W20_8 : 0.6*base + 0.2*Codex + 0.2*M        vs candidate W10 (4-sample)
The 4-sample reference pairs: M_A with v2 bag 1, M_B with v5 bag 1 (the
candidate as it would be built with the first 4 samples of each series).
Cells DIAG10/EXT10/EXT12 x base 7/101 x {A,B}.  Rule: better in all 12
cells, DIAG10 p_worse < 0.025/2.

Run:  cd research && PYTHONPATH="" <python> -u temp_pfnbag8_v1.py
"""
import env  # noqa: F401
import numpy as np

from common import rmse
from harness import load
from screen_v6 import boot


def main():
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (lab.row_id.values == z["row_id"]).all()
    y = lab.sub_temp.values
    ok = {"W10_8": True, "W20_8": True}
    for s in ("DIAG10", "EXT10", "EXT12"):
        v2 = np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s)
        v5 = np.load(env.LOCAL + "/web_tabpfn_v5_temp_%s.npy" % s)
        series = {"A": (v2.mean(0), v2[0]), "B": (v5.mean(0), v5[0])}
        cx = z["%s__CODEX__726" % s]
        for bs in (7, 101):
            base = z["%s__MASK__%d" % (s, bs)]
            for k, (m8, m4) in series.items():
                g = ~np.isnan(base) & ~np.isnan(m8)
                ref = 0.7 * base + 0.2 * cx + 0.1 * m4
                txt = []
                for a, (wb, wp) in (("W10_8", (0.7, 0.1)), ("W20_8", (0.6, 0.2))):
                    c = wb * base + 0.2 * cx + wp * m8
                    d = rmse(c[g], y[g]) / rmse(ref[g], y[g]) - 1
                    pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], c[g])[3] if s == "DIAG10" else np.nan
                    ok[a] = ok[a] and d < 0 and (s != "DIAG10" or pw < 0.0125)
                    txt.append("%s %+.2f%% (p_worse %.4f)" % (a, 100 * d, pw))
                print("%-6s base %3d series %s | member8 %.5f member4 %.5f | W10 %.5f | %s"
                      % (s, bs, k, rmse(m8[g], y[g]), rmse(m4[g], y[g]), rmse(ref[g], y[g]), " | ".join(txt)), flush=True)
    for a, v in ok.items():
        print("%s PRE-SET RULE VERDICT: %s" % (a, "ADOPT" if v else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
