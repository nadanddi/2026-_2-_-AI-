# -*- coding: utf-8 -*-
"""Sensitivity of the post-hoc gate thresholds of temp_pfngate_v1.py (G_C):
ramp (lo, lo+2) for lo = 6, 7, 8, 9, 10, and hard cuts at 8 and 10.
Change vs ungated W20_8 per validator (mean over 2 seed pairs x 2 members,
plus the worst cell).  Not a decision rule - checks the result is not a knife edge.
"""
import env  # noqa: F401
import numpy as np
from common import rmse
from harness import load

_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
y = lab.sub_temp.values
t = lab.in_temp.values
gates = {"ramp%g-%g" % (lo, lo + 2): (lambda lo=lo: np.clip((t - lo) / 2.0, 0, 1)) for lo in (6, 7, 8, 9, 10)}
gates["cut8"] = lambda: (t > 8).astype(float)
gates["cut10"] = lambda: (t > 10).astype(float)
for gname, gf in gates.items():
    gate = np.where(np.isnan(t), 1.0, gf())
    out = []
    for s in ("DIAG10", "EXT10", "EXT12"):
        v2 = np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s)
        v6 = np.load(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s)
        ds = []
        for bs, cs in ((7, 726), (101, 727)):
            base, cx = z["%s__MASK__%d" % (s, bs)], z["%s__CODEX__%d" % (s, cs)]
            for m8 in (v2.mean(0), v6[:8].mean(0)):
                g = ~np.isnan(base) & ~np.isnan(m8)
                ref = 0.6 * base + 0.2 * cx + 0.2 * m8
                p = 0.2 * gate
                c = 0.6 * base + (0.4 - p) * cx + p * m8
                ds.append(100 * (rmse(c[g], y[g]) / rmse(ref[g], y[g]) - 1))
        out.append("%s mean %+.2f%% worst %+.2f%%" % (s, np.mean(ds), np.max(ds)))
    print("%-10s | %s" % (gname, " | ".join(out)), flush=True)
