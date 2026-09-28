# -*- coding: utf-8 -*-
"""Inspector 4-1: round-5 change (plain->F60ND) and Codex blend split by farm x pass x cold on DIAG10. stdout only."""
import env  # noqa: F401
import numpy as np
from harness import load
from features_v4 import phys_features
_, lab, _ = load()
z = np.load(env.LOCAL + "/eval_v6_oof.npz", allow_pickle=True)
n = np.load(env.LOCAL + "/verify_codex_resid_reset_diag.npz")["new"]
y = lab.sub_temp.values
ph = phys_features().set_index("row_id").loc[lab.row_id]
cold = ph.groupby([lab.farm.values, lab.day.values]).ph_in_temp_3.transform("min").values < 10
p, q = z["plain__DIAG10"], z["F60ND__DIAG10"]; r = 0.8 * q + 0.2 * n
for f in ("F13", "F47"):
    for p2 in (False, True):
        for c in (False, True):
            m = (lab.farm.values == f) & ((lab.day.values >= 179) == p2) & (cold == c)
            if m.sum() == 0: continue
            e = lambda a: np.sqrt(np.mean((a[m] - y[m]) ** 2))
            print("%s pass%d cold%d days %3d | plain %.3f F60ND %.3f (%+.1f%%) blend %.3f (%+.1f%%)" % (f, 2 if p2 else 1, c,
                  len(set(lab.day.values[m])), e(p), e(q), 100 * (e(q) / e(p) - 1), e(r), 100 * (e(r) / e(q) - 1)))
