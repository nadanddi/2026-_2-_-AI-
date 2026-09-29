# -*- coding: utf-8 -*-
"""Temperature: is the TabPFN weight 0.1 too small?  (Bagged member alone is
0.668 on DIAG10 vs 0.688 for the reference, but worse on cold EXT days.)

Uses the saved GPU bag OOFs of web_tabpfn_v2_gpu.py (bags {1..4} and {5..8})
- no refit.  Arms (fixed): the TabPFN share taken from the MASK base
  W20 : 0.6*base + 0.2*Codex + 0.2*bag
  W30 : 0.5*base + 0.2*Codex + 0.3*bag
Reference = the candidate W10 (0.7/0.2/0.1).  Validators DIAG10, EXT10,
EXT12; base seeds 7/101 x bags 1/2.
Rule (2026-09-27): better than W10 in all 12 cells, DIAG10 p_worse <
0.025/2 in all 4 DIAG10 cells.

Run:  cd research && PYTHONPATH="" <python> -u temp_pfnweight_v1.py
"""
import env  # noqa: F401
import numpy as np

from common import rmse
from harness import load
from screen_v6 import boot

ARMS = {"W20": (0.6, 0.2), "W30": (0.5, 0.3)}


def main():
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (lab.row_id.values == z["row_id"]).all()
    y = lab.sub_temp.values
    ok = {a: True for a in ARMS}
    for s in ("DIAG10", "EXT10", "EXT12"):
        bags = np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s)
        cx = z["%s__CODEX__726" % s]
        for bs in (7, 101):
            base = z["%s__MASK__%d" % (s, bs)]
            for b in (0, 1):
                g = ~np.isnan(base) & ~np.isnan(bags[b])
                ref = 0.7 * base + 0.2 * cx + 0.1 * bags[b]
                txt = []
                for a, (wb, wp) in ARMS.items():
                    cand = wb * base + 0.2 * cx + wp * bags[b]
                    d = rmse(cand[g], y[g]) / rmse(ref[g], y[g]) - 1
                    pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], cand[g])[3] if s == "DIAG10" else np.nan
                    ok[a] = ok[a] and d < 0 and (s != "DIAG10" or pw < 0.025 / len(ARMS))
                    txt.append("%s %+.2f%% (p_worse %.4f)" % (a, 100 * d, pw))
                print("%-6s base %3d bag %d | W10 %.5f | %s" % (s, bs, b + 1, rmse(ref[g], y[g]), " | ".join(txt)),
                      flush=True)
    for a, v in ok.items():
        print("%s PRE-SET RULE VERDICT: %s" % (a, "ADOPT" if v else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
