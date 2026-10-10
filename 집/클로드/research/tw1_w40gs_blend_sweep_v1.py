# -*- coding: utf-8 -*-
"""TW1 (2026-10-12 집 클로드; researcher sweep of the temperature base blend W40G-S, whose constants were set by a few LB steps):
  pred = wb * BASE + (1 - wb - wp*g) * CODEX + wp * g * PFN,   g = clip((in_temp - gc) / gw + .5, 0, 1)   (gate centre gc, width gw)
  submitted: wb .4, wp .4, gate 8 -> 10 C (gc 9, gw 2).
Why: BASE (physics + trees) extrapolates to cold, PFN is best inside the training range; the gate hands weight from PFN to
CODEX as it gets cold.  Grid: wb {.2,.3,.4,.5} x wp {.2,.3,.4,.5,.6} x gc {7,8,9,10,11} x gw {1,2,4} -> 300 (wb + wp <= 1).
Selection on PASS-1 rows of DIAG10 (no chain / anchor corrections: blend-only), judgement on PASS-2 rows of DIAG10, F..L layouts
(stored members), EXT10/EXT12; record-level bootstrap on DIAG10.  Main effects reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tw1_w40gs_blend_sweep_v1.py
"""
import os, sys, itertools, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("tc1", os.path.join(HERE, "tc1_chain_history_early_hours_v1.py")); T = importlib.util.module_from_spec(s); s.loader.exec_module(T)
T.LAY["L"] = ("tt27_ckpt", "L_", ("base_R_1414", "base_R_1515"), "codex_R")
rm = T.rm


def blend(M, b, cx, fam, wb, wp, gc, gw):
    g = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - gc) / gw + .5, 0, 1))
    return (wb * M[b] + (1 - wb - wp * g) * M[cx] + wp * g * M["pfn_" + fam]).values


def main():
    CT = T.chain_table(); lays = {L: T.layout_frame(L, CT) for L in T.LAY}
    grid = [g for g in itertools.product((.2, .3, .4, .5), (.2, .3, .4, .5, .6), (7, 8, 9, 10, 11), (1, 2, 4)) if g[0] + g[1] <= 1]
    M = lays["DIAG10"]; d, pre, bs, cx = T.LAY["DIAG10"]; y = M.sub_temp.values; p1 = M.day.values < 179
    rows = []
    for g in grid:
        r1 = np.mean([rm((blend(M, b, cx, f, *g) - y)[p1]) for b in bs for f in "AB"]); r2 = np.mean([rm((blend(M, b, cx, f, *g) - y)[~p1]) for b in bs for f in "AB"])
        rows.append(dict(wb=g[0], wp=g[1], gc=g[2], gw=g[3], pass1=r1, pass2=r2))
    G = pd.DataFrame(rows); base = G[np.isclose(G.wb, .4) & np.isclose(G.wp, .4) & (G.gc == 9) & (G.gw == 2)].iloc[0]
    G["d1"] = 100 * (G.pass1 / base.pass1 - 1); G["d2"] = 100 * (G.pass2 / base.pass2 - 1); G.to_csv(os.path.join(env.LOCAL, "tw1_grid.csv"), index=False)
    for fac in ("wb", "wp", "gc", "gw"):
        print("  main effect %-3s:" % fac, " ".join("%s: p1 %+.2f%% p2 %+.2f%%" % (v, gg.d1.mean(), gg.d2.mean()) for v, gg in G.groupby(fac)))
    best = G.sort_values("pass1").iloc[0]; sel = (best.wb, best.wp, best.gc, best.gw)
    print("selected on pass-1:", sel, "| pass1 %+.2f%% pass2 %+.2f%%; pass-2 rank %d of %d" % (best.d1, best.d2, int((G.pass2 < best.pass2).sum()) + 1, len(G)))
    print(G.sort_values("pass1").head(6)[["wb", "wp", "gc", "gw", "d1", "d2"]].round(2).to_string())
    rng = np.random.default_rng(20261205)
    for L, M in lays.items():
        d, pre, bs, cx = T.LAY[L]; y = M.sub_temp.values; m2 = M.day.values >= 179; key = (M.farm + "_" + M.day.astype(str)).values; rel = []; per = []
        for b in bs:
            for f in "AB":
                a = blend(M, b, cx, f, *sel); c = blend(M, b, cx, f, .4, .4, 9, 2)
                rel.append(rm((a - y)[m2]) / rm((c - y)[m2]) - 1); per.append(pd.Series(((a - y) ** 2 - (c - y) ** 2)[m2]).groupby(key[m2]).sum())
        P = pd.concat(per, axis=1).mean(axis=1); P = P[P.abs() > 1e-12]
        line = "  %-6s selected vs submitted, pass-2 cells %s" % (L, " ".join("%+.2f%%" % (100 * x) for x in rel))
        if L == "DIAG10" and len(P):
            idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P.values[idx].sum(1) >= 0).mean())
            line += " | records %d improved %d | record P(worse) %.4f -> %s" % (len(P), int((P < 0).sum()), pw, "PASS" if all(x < 0 for x in rel) and pw < .025 else "FAIL")
        print(line, flush=True)


if __name__ == "__main__":
    main()
