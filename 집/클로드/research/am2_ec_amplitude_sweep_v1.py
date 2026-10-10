# -*- coding: utf-8 -*-
"""AM2 (2026-10-12 집 클로드; researcher re-investigation of AM1, which tried only two values and failed on P .26/.33 while
improving all 9 cells).  Why: high-EC days have ~4x the within-day swing (6.544) while the model's deviations are shrunk; the
widening should scale with the predicted level.  Form (causal): p' = cm + (1 + b * a) (p - cm),  a = clip((cm - t0) / tw, 0, 1),
cm = running mean of the prediction over hours 0..h.
Grid: b {.25, .5, .75, 1, 1.5} x t0 {.6, .8, 1.0} x tw {.2, .4, .8} -> 45.  Base: stored RV1 OOF (CUR + SG2, column sg).
Pass-1 and pass-2 differ (6.584), so selection is done INSIDE pass-2 across farms: select on F13 pass-2 rows (DIAG10 seed-mean),
judge on F47 pass-2 rows, and vice versa.  PASS iff in BOTH directions the selected setting beats no-widening on the held-out
farm in all 3 seeds of DIAG10, P2LOO and EL1, and record-level bootstrap P(worse) on DIAG10 < .025 (pooled over both directions).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u am2_ec_amplitude_sweep_v1.py
"""
import os, itertools
import env  # noqa: F401
import numpy as np, pandas as pd
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))


def widen(O, b, t0, tw):
    a = np.clip((O.cm - t0) / tw, 0, 1)
    return (O.cm + (1 + b * a) * (O.sg - O.cm)).values


def main():
    O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.day >= 179)].copy()
    O["rec"] = O.row_id.str[:7]; O = O.sort_values(["validator", "fold", "seed", "rec", "hour"]).reset_index(drop=True)
    O["cm"] = O.groupby(["validator", "fold", "seed", "rec"]).sg.transform(lambda s: s.expanding().mean())
    grid = list(itertools.product((.25, .5, .75, 1.0, 1.5), (.6, .8, 1.0), (.2, .4, .8)))
    D = O[O.validator == "DIAG10"]; rows = []
    for g in grid:
        pr = widen(D, *g); r = {}
        for f in ("F13", "F47"):
            m = (D.farm == f).values; r[f] = np.mean([rm((pr - D.sub_ec)[m & (D.seed == s).values]) / rm((D.sg - D.sub_ec)[m & (D.seed == s).values]) - 1 for s in D.seed.unique()])
        rows.append(dict(b=g[0], t0=g[1], tw=g[2], F13=100 * r["F13"], F47=100 * r["F47"]))
    G = pd.DataFrame(rows); G.to_csv(os.path.join(env.LOCAL, "am2_grid.csv"), index=False)
    for fac in ("b", "t0", "tw"):
        print("  main effect %-2s:" % fac, " ".join("%s: F13 %+.2f%% F47 %+.2f%%" % (v, gg.F13.mean(), gg.F47.mean()) for v, gg in G.groupby(fac)))
    rng = np.random.default_rng(20261206); ok = True; per_all = []
    for sel_f, held in (("F13", "F47"), ("F47", "F13")):
        best = G.sort_values(sel_f).iloc[0]; g = (best.b, best.t0, best.tw)
        print("select on %s: b %.2f t0 %.1f tw %.1f (%s %+.2f%%) -> held-out %s %+.2f%% (rank %d of %d on held-out)" % (
            sel_f, *g, sel_f, best[sel_f], held, best[held], int((G[held] < best[held]).sum()) + 1, len(G)))
        for v in ("DIAG10", "P2LOO", "EL1"):
            X = O[(O.validator == v) & (O.farm == held)]; pr = widen(X, *g)
            cells = [rm((pr - X.sub_ec)[(X.seed == s).values]) / rm((X.sg - X.sub_ec)[(X.seed == s).values]) - 1 for s in X.seed.unique()]
            ok &= all(c < 0 for c in cells); print("    %s on %s: cells %s" % (v, held, " ".join("%+.2f%%" % (100 * c) for c in cells)))
            if v == "DIAG10":
                per_all.append(pd.Series((pr - X.sub_ec.values) ** 2 - (X.sg - X.sub_ec).values ** 2).groupby([X.seed.values, X.rec.values]).sum().groupby(level=1).mean())
    P = pd.concat(per_all); P = P[P.abs() > 1e-12]; idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P.values[idx].sum(1) >= 0).mean())
    ok &= pw < .025
    print("pooled held-out record bootstrap P(worse) %.4f (%d records, improved %d) -> VERDICT AM2: %s" % (pw, len(P), int((P < 0).sum()), "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
