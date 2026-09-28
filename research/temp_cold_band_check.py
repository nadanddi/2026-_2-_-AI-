# -*- coding: utf-8 -*-
"""Cold-band check (analysis): RMSE and bias on rows with in_temp <= 8 (and
8-10) for the pre-TabPFN reference, W10 and W20_8, DIAG10/EXT10/EXT12,
both 8-sample series, base 7 / Codex 726.  Test has 15% of rows <= 8 C.
"""
import env  # noqa: F401
import numpy as np
from common import rmse
from harness import load

_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
y = lab.sub_temp.values
t = lab.in_temp.values
for s in ("DIAG10", "EXT10", "EXT12"):
    v2 = np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s)
    v6 = np.load(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s)
    base, cx = z["%s__MASK__7" % s], z["%s__CODEX__726" % s]
    for name, m8, m4 in (("1-8", v2.mean(0), v2[0]), ("17-24", v6[:8].mean(0), v6[:4].mean(0))):
        P = {"noPFN": 0.8 * base + 0.2 * cx, "W10": 0.7 * base + 0.2 * cx + 0.1 * m4,
             "W20_8": 0.6 * base + 0.2 * cx + 0.2 * m8, "PFN8": m8, "base": base, "codex": cx}
        g = ~np.isnan(base) & ~np.isnan(m8)
        for lo, hi in ((-99, 8), (8, 10)):
            m = g & (t > lo) & (t <= hi)
            print("%-6s %s in_temp (%g,%g] n=%d | %s" % (s, name, lo, hi, m.sum(), " | ".join(
                "%s %.3f (bias %+.2f)" % (k, rmse(p[m], y[m]), (p - y)[m].mean()) for k, p in P.items())), flush=True)
