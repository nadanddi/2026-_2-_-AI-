# -*- coding: utf-8 -*-
"""Extend the temperature gate (catalog 6.67/6.69): in the cold the Codex
member beats even the MASK base (EXT10 <= 8 C: Codex 0.713, base 0.938).
POST-HOC arm (thresholds from the same observations):
  G_C2 : base 0.6 - 0.2(1-g),  Codex 0.2 + 0.4(1-g),  TabPFN 0.2 g
         (g = clip((in_temp - 8)/2, 0, 1); warm rows identical to G_C,
          rows <= 8 C get 0.4 base / 0.6 Codex)
Reference G_C.  Validators DIAG10/EXT10/EXT12 (saved OOFs) and EXT8
(temp_ext8_oof.npz + TabPFN recomputed? no - EXT8 TabPFN OOFs were not saved,
so EXT8 uses only rows <= 8 C where TabPFN share is 0 for both arms... not
available; EXT8 is skipped here and re-run on GPU if G_C2 passes).
Rule (fixed): vs G_C, EXT10 and EXT12 better in all 8 cells, DIAG10 not
worse than +0.10%.

Run:  cd research && PYTHONPATH="" <python> -u temp_pfngate_v2.py
"""
import env  # noqa: F401
import numpy as np

from common import rmse
from harness import load


def main():
    _, lab, _ = load()
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (lab.row_id.values == z["row_id"]).all()
    y = lab.sub_temp.values
    t = lab.in_temp.values
    g8 = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    ok = True
    for s in ("DIAG10", "EXT10", "EXT12"):
        mems = {"1-8": np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s).mean(0),
                "17-24": np.load(env.LOCAL + "/web_tabpfn_v6_temp_%s.npy" % s)[:8].mean(0)}
        for bs, cs in ((7, 726), (101, 727)):
            base, cx = z["%s__MASK__%d" % (s, bs)], z["%s__CODEX__%d" % (s, cs)]
            for k, m in mems.items():
                g = ~np.isnan(base) & ~np.isnan(m)
                gc = 0.6 * base + (0.4 - 0.2 * g8) * cx + 0.2 * g8 * m
                gc2 = (0.6 - 0.2 * (1 - g8)) * base + (0.2 + 0.4 * (1 - g8) - 0.2 * g8 + 0.2 * g8) * cx + 0.2 * g8 * m
                # codex share: 0.2 + 0.4(1-g) ; check weights sum to 1
                wsum = (0.6 - 0.2 * (1 - g8)) + (0.2 + 0.4 * (1 - g8)) + 0.2 * g8
                assert np.allclose(wsum, 1.0)
                gc2 = (0.6 - 0.2 * (1 - g8)) * base + (0.2 + 0.4 * (1 - g8)) * cx + 0.2 * g8 * m
                d = rmse(gc2[g], y[g]) / rmse(gc[g], y[g]) - 1
                ok = ok and (d < 0 if s != "DIAG10" else d <= 0.001)
                cold = g & (t <= 8)
                print("%-6s seeds %3d/%d m %-5s | G_C %.5f G_C2 %.5f (%+.2f%%) | <=8C G_C %.3f G_C2 %.3f"
                      % (s, bs, cs, k, rmse(gc[g], y[g]), rmse(gc2[g], y[g]), 100 * d,
                         rmse(gc[cold], y[cold]), rmse(gc2[cold], y[cold])), flush=True)
    print("G_C2 PRE-SET RULE VERDICT (DIAG10/EXT10/EXT12):", "ADOPT" if ok else "REJECT", flush=True)


if __name__ == "__main__":
    main()
