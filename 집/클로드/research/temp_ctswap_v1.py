# -*- coding: utf-8 -*-
"""Replace (not add) the Codex-view TabPFN member in G_C2 by the ct-view
member (web_tabpfn_ct_gpu.py: ct member alone is better, 0.652-0.658 vs
0.665 on DIAG10 and better on EXT, but adding it hurt DIAG10).  Saved OOFs.
POST-HOC arm (after seeing the ct member numbers):
  G_C2_ct : G_C2 weights, TabPFN share 0.2g given to the ct-view member.
Rule (same as web_tabpfn_ct_gpu.py): better than G_C2 in all 12 cells,
DIAG10 p_worse < 0.025.
"""
import env  # noqa: F401
import numpy as np
from common import rmse
from harness import load
from screen_v6 import boot

_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
y = lab.sub_temp.values
t = lab.in_temp.values
g8 = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
ok = True
for s in ("DIAG10", "EXT10", "EXT12"):
    cxv = {"A": np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s).mean(0),
           "B": np.load(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s)[:8].mean(0)}
    ctv = dict(zip("AB", np.load(env.LOCAL + "/web_tabpfn_ct_temp_%s.npy" % s)))
    for bs, cs in ((7, 726), (101, 727)):
        base, cx = z["%s__MASK__%d" % (s, bs)], z["%s__CODEX__%d" % (s, cs)]
        for k in "AB":
            g = ~np.isnan(base) & ~np.isnan(cxv[k]) & ~np.isnan(ctv[k])
            wb, wc = 0.6 - 0.2 * (1 - g8), 0.2 + 0.4 * (1 - g8)
            ref = wb * base + wc * cx + 0.2 * g8 * cxv[k]
            cand = wb * base + wc * cx + 0.2 * g8 * ctv[k]
            d = rmse(cand[g], y[g]) / rmse(ref[g], y[g]) - 1
            pw = boot(lab[g].reset_index(drop=True), "sub_temp", ref[g], cand[g])[3] if s == "DIAG10" else np.nan
            ok = ok and d < 0 and (s != "DIAG10" or pw < 0.025)
            print("%-6s seeds %3d/%d series %s | G_C2 %.5f G_C2_ct %.5f (%+.2f%%, p_worse %.4f)"
                  % (s, bs, cs, k, rmse(ref[g], y[g]), rmse(cand[g], y[g]), 100 * d, pw), flush=True)
print("G_C2_ct PRE-SET RULE VERDICT:", "ADOPT" if ok else "REJECT")
