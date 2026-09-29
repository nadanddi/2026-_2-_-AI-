# -*- coding: utf-8 -*-
"""After W20_8 was confirmed (web_tabpfn_v6_gpu.py): do 16 samples help, and
does a steadier member tolerate weight 0.3?  Saved OOFs give two independent
16-sample members:  S1 = samples 1-16 (mean of the four v2/v5 4-sample bags),
S2 = samples 17-32 (mean of the 16 per-sample v6 rows).

Arms (fixed), reference = W20_8 of the same series (first 8 samples:
S1 -> mean of v2 bags, S2 -> v6 samples 17-24):
  W20_16 : 0.6*base + 0.2*Codex + 0.2*member16
  W30_16 : 0.5*base + 0.2*Codex + 0.3*member16
Cells DIAG10/EXT10/EXT12 x base 7/101 x {S1,S2}.  Rule (2026-09-27): better
in all 12 cells, DIAG10 p_worse < 0.025/2.

Run:  cd research && PYTHONPATH="" <python> -u temp_pfnbag16_v1.py
"""
import env  # noqa: F401
import numpy as np

from common import rmse
from harness import load
from screen_v6 import boot

ARMS = {"W20_16": (0.6, 0.2), "W30_16": (0.5, 0.3)}


def main():
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (lab.row_id.values == z["row_id"]).all()
    y = lab.sub_temp.values
    ok = {a: True for a in ARMS}
    for s in ("DIAG10", "EXT10", "EXT12"):
        v2 = np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s)
        v5 = np.load(env.LOCAL + "/web_tabpfn_v5_temp_%s.npy" % s)
        v6 = np.load(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s)
        series = {"S1": (np.vstack([v2, v5]).mean(0), v2.mean(0)),
                  "S2": (v6.mean(0), v6[:8].mean(0))}
        cx = z["%s__CODEX__726" % s]
        for bs in (7, 101):
            base = z["%s__MASK__%d" % (s, bs)]
            for k, (m16, m8) in series.items():
                g = ~np.isnan(base) & ~np.isnan(m16)
                ref = 0.6 * base + 0.2 * cx + 0.2 * m8
                txt = []
                for a, (wb, wp) in ARMS.items():
                    c = wb * base + 0.2 * cx + wp * m16
                    d = rmse(c[g], y[g]) / rmse(ref[g], y[g]) - 1
                    pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], c[g])[3] if s == "DIAG10" else np.nan
                    ok[a] = ok[a] and d < 0 and (s != "DIAG10" or pw < 0.0125)
                    txt.append("%s %+.2f%% (p_worse %.4f)" % (a, 100 * d, pw))
                print("%-6s base %3d %s | member16 %.5f member8 %.5f | W20_8 %.5f | %s"
                      % (s, bs, k, rmse(m16[g], y[g]), rmse(m8[g], y[g]), rmse(ref[g], y[g]), " | ".join(txt)), flush=True)
    for a, v in ok.items():
        print("%s PRE-SET RULE VERDICT: %s" % (a, "ADOPT" if v else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
