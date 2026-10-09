# -*- coding: utf-8 -*-
"""TT2: two no-refit temperature candidates on the stored W40G-S members (TT1 checkpoints = TK2-identical members).
(2026-10-10 집 클로드, user: improved temperature model).  Fixed before running.
Reference W40G-S (per BASE seed s in {7,101}, PFN family F in {A: contexts 1-8, B: 17-24}):
  0.4 BASE + (0.2 + 0.4 (1 - g)) CODEX + 0.4 g PFN_F,  g = clip((in_temp - 8)/2, 0, 1)
  TA  PFN = mean of ALL 16 contexts (A and B) instead of one family of 8 (variance reduction only).
  TB  weights (wb, wc, wp) re-chosen: blend = wb BASE + (wc + (1 - wc - wb) ... ) -- concretely
      wb BASE + (1 - wb - wp g) CODEX + wp g PFN_F with wb in {.2,.3,.4,.5,.6}, wp in {.2,.3,.4,.5,.6} (wb + wp <= .9),
      selected ONCE by the lowest mean RMSE over both seeds and families on DIAG10 PASS-1 rows (day < 179) only.
      (W40G-S itself is wb .4, wp .4: 0.4 BASE + (0.6 - 0.4 g) CODEX + 0.4 g PFN.)
RULE (k = 2, alpha .0125): candidate better than W40G-S in all 12 cells (seed x family x DIAG10/EXT10/EXT12) and
DIAG10 farm x 5-day block bootstrap P(worse) < .0125 in every combo.  EL1 and pass-2 reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt2_temp_pfn16_reweight_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
CK = os.path.join(env.LOCAL, "tt1_ckpt")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
SEEDS = (7, 101); FAMS = ("A", "B")


def blend(X, s, pf, wb=.4, wp=.4):
    g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1))
    return wb * X["base_REF_%d" % s] + (1 - wb - wp * g) * X["codex_REF"] + wp * g * pf


def main():
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    G["pfn_AB"] = (G.pfn_A + G.pfn_B) / 2
    # TB selection on DIAG10 pass-1 rows only
    D1 = G[(G.validator == "DIAG10") & (G.day < 179)]
    grid = [(wb, wp) for wb in (.2, .3, .4, .5, .6) for wp in (.2, .3, .4, .5, .6) if wb + wp <= .9 + 1e-9]
    sc = {w: np.mean([r(blend(D1, s, D1["pfn_%s" % f], *w) - D1.sub_temp) for s in SEEDS for f in FAMS]) for w in grid}
    wsel = min(sc, key=sc.get)
    print("TB selected (wb, wp) on DIAG10 pass-1: %s (score %.4f; W40G-S (.4,.4) %.4f)" % (wsel, sc[wsel], sc[(.4, .4)]))
    rng = np.random.default_rng(20261010)
    for cand in ("TA", "TB"):
        print("\n==== %s" % cand); ok, cells = True, 0
        for s in SEEDS:
            for f in FAMS:
                line = "seed %3d PFN %s |" % (s, f)
                for v in ("DIAG10", "EXT10", "EXT12", "EL1"):
                    X = G[G.validator == v]
                    ref = blend(X, s, X["pfn_%s" % f])
                    new = blend(X, s, X.pfn_AB) if cand == "TA" else blend(X, s, X["pfn_%s" % f], *wsel)
                    a, b = r(ref - X.sub_temp), r(new - X.sub_temp); L = (X.day >= 179).values
                    line += " %s %+.2f%% (p2 %+.2f%%)" % (v, 100 * (b / a - 1), 100 * (r((new - X.sub_temp)[L]) / r((ref - X.sub_temp)[L]) - 1))
                    if v != "EL1":
                        cells += b < a; ok &= b < a
                    if v == "DIAG10":
                        cl = (X.farm + "_" + (X.day // 5).astype(str)).values
                        dd = pd.Series(((new - X.sub_temp) ** 2 - (ref - X.sub_temp) ** 2).values).groupby(cl).agg(["sum", "count"])
                        sm, n = dd["sum"].values, dd["count"].values
                        idx = rng.integers(0, len(sm), (20000, len(sm)))
                        p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()); ok &= p < .0125
                        line += " P %.4f |" % p
                print(line)
        print("VERDICT %s: cells %d/12 -> %s" % (cand, cells, "PASS" if ok and cells == 12 else "FAIL"))


if __name__ == "__main__":
    main()
