# -*- coding: utf-8 -*-
"""Seed averaging of the tree members in G_C2: MASK base averaged over seeds
7/101 and Codex over 726/727, instead of one seed each.  Saved OOFs.

Arm (fixed): SAVG = G_C2 with base = mean(base 7, base 101), Codex = mean(726, 727).
Reference: G_C2 with single seeds, both pairings (7/726 and 101/727).
Cells DIAG10/EXT10/EXT12 x 2 pairings x 2 TabPFN members (1-8, 17-24).
Rule: better in all 12 cells, DIAG10 p_worse < 0.025.

Run:  cd research && PYTHONPATH="" <python> -u temp_seedavg_v1.py
"""
import env  # noqa: F401
import numpy as np

from common import rmse
from harness import load
from screen_v6 import boot


def gc2(base, cx, m, g8):
    return (0.6 - 0.2 * (1 - g8)) * base + (0.2 + 0.4 * (1 - g8)) * cx + 0.2 * g8 * m


def main():
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    y = lab.sub_temp.values
    t = lab.in_temp.values
    g8 = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    ok = True
    for s in ("DIAG10", "EXT10", "EXT12"):
        bavg = 0.5 * (z["%s__MASK__7" % s] + z["%s__MASK__101" % s])
        cavg = 0.5 * (z["%s__CODEX__726" % s] + z["%s__CODEX__727" % s])
        for name, m in (("1-8", np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s).mean(0)),
                        ("17-24", np.load(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s)[:8].mean(0))):
            cand = gc2(bavg, cavg, m, g8)
            for bs, cs in ((7, 726), (101, 727)):
                ref = gc2(z["%s__MASK__%d" % (s, bs)], z["%s__CODEX__%d" % (s, cs)], m, g8)
                g = ~np.isnan(ref) & ~np.isnan(cand)
                d = rmse(cand[g], y[g]) / rmse(ref[g], y[g]) - 1
                pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], cand[g])[3] if s == "DIAG10" else np.nan
                ok = ok and d < 0 and (s != "DIAG10" or pw < 0.025)
                print("%-6s m %-5s vs seeds %3d/%d | G_C2 %.5f SAVG %.5f (%+.3f%%, p_worse %.4f)"
                      % (s, name, bs, cs, rmse(ref[g], y[g]), rmse(cand[g], y[g]), 100 * d, pw), flush=True)
    print("SAVG PRE-SET RULE VERDICT:", "ADOPT" if ok else "REJECT")


if __name__ == "__main__":
    main()
