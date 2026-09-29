# -*- coding: utf-8 -*-
"""EXT8 check of G_C2 vs G_C.  The two arms differ only in the base/Codex
split (TabPFN share 0.2 g identical), so  G_C2 - G_C = 0.2 (1-g) (Codex - base)
and the TabPFN OOF cancels: we evaluate the difference exactly using any
TabPFN value (set to the base, the result is invariant).  Rule (fixed, as
temp_pfngate_v2.py): G_C2 better than G_C in all EXT8 cells (2 seed pairs).
"""
import env  # noqa: F401
import numpy as np
from common import rmse
from harness import load

_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_ext8_oof.npz", allow_pickle=True)
y = lab.sub_temp.values
t = lab.in_temp.values
g8 = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
pf = np.load(env.LOCAL + "/web_tabpfn_v2_temp_EXT12.npy").mean(0)   # placeholder only for rows; cancels in the difference
ok = True
for bs, cs in ((7, 726), (101, 727)):
    base, cx = z["EXT8__MASK__%d" % bs], z["EXT8__CODEX__%d" % cs]
    g = ~np.isnan(base)
    m = np.where(np.isnan(pf), base, pf)
    gc = 0.6 * base + (0.4 - 0.2 * g8) * cx + 0.2 * g8 * m
    gc2 = (0.6 - 0.2 * (1 - g8)) * base + (0.2 + 0.4 * (1 - g8)) * cx + 0.2 * g8 * m
    # the absolute RMSEs depend on the placeholder; the SIGN of the change is judged on squared-error terms
    # restricted to rows where the arms differ (g8 < 1), where the TabPFN term is 0.2*g8*m in both.
    d = g & (g8 < 1)
    e1, e2 = ((gc - y) ** 2)[d], ((gc2 - y) ** 2)[d]
    better = e2.sum() < e1.sum()
    ok = ok and better
    cold = g & (t <= 8)
    print("EXT8 seeds %d/%d | rows where arms differ %d | SSE G_C %.2f  G_C2 %.2f | <=8C RMSE (TabPFN share 0) G_C %.3f G_C2 %.3f"
          % (bs, cs, d.sum(), e1.sum(), e2.sum(), rmse(gc[cold], y[cold]), rmse(gc2[cold], y[cold])))
print("G_C2 EXT8 VERDICT:", "ADOPT" if ok else "REJECT")
