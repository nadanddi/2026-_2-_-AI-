# -*- coding: utf-8 -*-
"""Segment check of W20_8 vs W10 (analysis): by greenhouse and by indoor
temperature band, DIAG10 and EXT12, both 8-sample series (1-8, 17-24),
base 7 / Codex 726.  No decision rule - looks for a segment that gets worse.
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from common import rmse
from harness import load

_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
y = lab.sub_temp.values
band = np.asarray(pd.cut(lab.in_temp.values, [-99, 8, 10, 12, 15, 99]).astype(str), dtype=object)
for s in ("DIAG10", "EXT12"):
    v2 = np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s)
    v6 = np.load(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s)
    base, cx = z["%s__MASK__7" % s], z["%s__CODEX__726" % s]
    for name, m8, m4 in (("1-8", v2.mean(0), v2[0]), ("17-24", v6[:8].mean(0), v6[:4].mean(0))):
        w10 = 0.7 * base + 0.2 * cx + 0.1 * m4
        w20 = 0.6 * base + 0.2 * cx + 0.2 * m8
        g = ~np.isnan(base) & ~np.isnan(m8)
        rows = []
        for key, lab_k in (("farm", lab.farm.values), ("band", band)):
            for v in sorted(set(map(str, lab_k[g]))):
                m = g & (np.asarray(lab_k).astype(str) == v)
                rows.append("%s=%s n=%d %+.2f%%" % (key, v, m.sum(), 100 * (rmse(w20[m], y[m]) / rmse(w10[m], y[m]) - 1)))
        print("%s series %s | %s" % (s, name, " | ".join(rows)), flush=True)
