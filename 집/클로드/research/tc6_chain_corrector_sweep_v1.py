# -*- coding: utf-8 -*-
"""TC6 (2026-10-12 집 클로드; researcher sweep of the submitted chain corrector, which was set once by convention:
shrink .5, 100 trees, min_child 100).  Why each knob: shrink = how much of the learned early-hour residual is trusted (stacking
on OOF residuals overfits -> < 1 expected); n_estimators / min_child = capacity vs. the ~3,000 early-hour training rows.
Grid: shrink {.25, .5, .75, 1.0, 1.25} x trees {50, 100, 200, 400} x min_child {50, 100, 200}  -> 60 settings (lr .03, 15 leaves).
Selection on PASS-1 rows of DIAG10 (all-hour RMSE, seed x family mean); judgement of the selected setting vs the submitted
setting on PASS-2 rows of DIAG10 / G / EL1 / EXT10 / EXT12 (cells + record-level bootstrap on DIAG10).  Main effects reported.
Base predictions: W40G-S (stored members); corrector training source DIAG10 OOF with fold purge, as TC1c.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tc6_chain_corrector_sweep_v1.py
"""
import os, sys, itertools, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("tc1", os.path.join(HERE, "tc1_chain_history_early_hours_v1.py")); T = importlib.util.module_from_spec(s); s.loader.exec_module(T)
s = importlib.util.spec_from_file_location("tc1b", os.path.join(HERE, "tc1b_chain_history_common_train_v1.py")); B = importlib.util.module_from_spec(s); s.loader.exec_module(B)
rm = T.rm; H = 7


def corr_preds(M, p, D, bi, fam, trees, mchild):
    """raw (unshrunk) corrector predictions for early rows of M, cross-fitted as TC1c."""
    pD = B.w40(D, B.DB[bi], "codex_REF", fam)
    D = D.assign(p=pD, p_air=pD - D.in_temp.values); M = M.assign(p=p, p_air=p - M.in_temp.values)
    out = np.zeros(len(M)); De = (D.hour <= H).values; Me = (M.hour <= H).values
    for k in M.fold.unique():
        te = Me & (M.fold == k).values
        if not te.any():
            continue
        vd = set(zip(M.farm[M.fold == k], M.day[M.fold == k]))
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {("F47" if f == "F13" else "F13", d + j) for f, d in vd for j in range(-5, 6)}
        tr = De & np.array([(f, d) not in forb for f, d in zip(D.farm, D.day)])
        mdl = T.lgb.LGBMRegressor(n_estimators=trees, learning_rate=.03, num_leaves=15, min_child_samples=mchild, subsample=.8, subsample_freq=1,
                                  colsample_bytree=.8, random_state=1, verbose=-1, deterministic=True, force_col_wise=True, n_jobs=2)
        mdl.fit(D.loc[tr, T.FEAT], (D.sub_temp - D.p).values[tr]); out[te] = mdl.predict(M.loc[te, T.FEAT])
    return out


def main():
    CT = T.chain_table(); D = T.layout_frame("DIAG10", CT)
    lays = {L: T.layout_frame(L, CT) for L in ("DIAG10", "G", "EL1", "EXT10", "EXT12")}
    cap = {}
    for trees, mc in itertools.product((50, 100, 200, 400), (50, 100, 200)):
        for L, M in lays.items():
            d, pre, bs, cx = T.LAY[L]
            for bi, b in enumerate(bs):
                for fam in "AB":
                    p0 = B.w40(M, b, cx, fam); cap[(trees, mc, L, bi, fam)] = (p0, corr_preds(M, p0, D, bi, fam, trees, mc))
        print("  fitted trees %d min_child %d" % (trees, mc), flush=True)
    rows = []
    for sh, trees, mc in itertools.product((.25, .5, .75, 1.0, 1.25), (50, 100, 200, 400), (50, 100, 200)):
        M = lays["DIAG10"]; y = M.sub_temp.values; p1 = M.day.values < 179; p2 = ~p1
        r1, r2 = [], []
        for bi in (0, 1):
            for fam in "AB":
                p0, c = cap[(trees, mc, "DIAG10", bi, fam)]; pr = p0 + sh * c
                r1.append(rm((pr - y)[p1])); r2.append(rm((pr - y)[p2]))
        rows.append(dict(shrink=sh, trees=trees, min_child=mc, pass1=np.mean(r1), pass2=np.mean(r2)))
    G = pd.DataFrame(rows); base = G[np.isclose(G.shrink, .5) & (G.trees == 100) & (G.min_child == 100)].iloc[0]
    G["d1"] = 100 * (G.pass1 / base.pass1 - 1); G["d2"] = 100 * (G.pass2 / base.pass2 - 1)
    G.to_csv(os.path.join(env.LOCAL, "tc6_grid.csv"), index=False)
    for fac in ("shrink", "trees", "min_child"):
        print("  main effect %-9s:" % fac, " ".join("%s: p1 %+.2f%% p2 %+.2f%%" % (v, g.d1.mean(), g.d2.mean()) for v, g in G.groupby(fac)))
    best = G.sort_values("pass1").iloc[0]
    print("selected on pass-1: shrink %.2f trees %d min_child %d | pass1 %+.2f%% pass2 %+.2f%% vs submitted; pass-2 rank %d of %d" % (
        best.shrink, best.trees, best.min_child, best.d1, best.d2, int((G.pass2 < best.pass2).sum()) + 1, len(G)))
    rng = np.random.default_rng(20261204)
    for L, M in lays.items():
        d, pre, bs, cx = T.LAY[L]; y = M.sub_temp.values; m2 = M.day.values >= 179; rel = []; per = []
        key = (M.farm + "_" + M.day.astype(str)).values
        for bi in (0, 1):
            for fam in "AB":
                p0, c = cap[(int(best.trees), int(best.min_child), L, bi, fam)]; a = p0 + best.shrink * c
                p0b, cb = cap[(100, 100, L, bi, fam)]; b = p0b + .5 * cb
                rel.append(rm((a - y)[m2]) / rm((b - y)[m2]) - 1); per.append(pd.Series(((a - y) ** 2 - (b - y) ** 2)[m2]).groupby(key[m2]).sum())
        P = pd.concat(per, axis=1).mean(axis=1); P = P[P.abs() > 1e-12]
        line = "  %-6s selected vs submitted, pass-2 cells %s | records %d improved %d" % (L, " ".join("%+.2f%%" % (100 * x) for x in rel), len(P), int((P < 0).sum()))
        if L == "DIAG10" and len(P):
            idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P.values[idx].sum(1) >= 0).mean())
            line += " | record P(worse) %.4f -> %s" % (pw, "PASS" if all(x < 0 for x in rel) and pw < .025 else "FAIL")
        print(line, flush=True)


if __name__ == "__main__":
    main()
