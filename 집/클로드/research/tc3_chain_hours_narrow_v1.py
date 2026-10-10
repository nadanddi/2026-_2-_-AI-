# -*- coding: utf-8 -*-
"""TC3 (SCREEN, 2026-10-11 집 클로드, after round 11: temperature .5127 -> .5089 with the chain-aware early-hour correction).
Pre-registered follow-up (설명자료_18): TC2 showed widening hurts (6.555); TC3 tests NARROWER windows.  Fixed before running.
Baseline: TC1c CHAIN corrector on hours 0-7 (as submitted).  Candidates (k = 2, alpha .0125):
  H3: same recipe, corrector trained and applied on hours 0-3
  H5: same recipe, hours 0-5
Recipe otherwise identical (DIAG10 OOF training source with fold purge, LightGBM 100/.03/15/min_child 100, seed 1, shrink .5,
TC1 chain features).  Layouts: DIAG10, EL1, F, G, H, I, J, K, L (tt27; L already used for TC1 -> screen only), EXT10/EXT12
reported.  PASS (screen) iff better than the 0-7 baseline in every cell of DIAG10, EL1, F..L AND pooled P(worse) < .0125
(per-block differences averaged over DIAG10 + F..L first).  A pass needs a fresh layout with refitted members.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tc3_chain_hours_narrow_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("tc1", os.path.join(HERE, "tc1_chain_history_early_hours_v1.py")); T = importlib.util.module_from_spec(s); s.loader.exec_module(T)
s = importlib.util.spec_from_file_location("tc1b", os.path.join(HERE, "tc1b_chain_history_common_train_v1.py")); B = importlib.util.module_from_spec(s); s.loader.exec_module(B)
T.LAY["L"] = ("tt27_ckpt", "L_", ("base_R_1414", "base_R_1515"), "codex_R")
ALPHA = .0125; rm = T.rm


def corrected(M, p, D, bi, fam, H):
    pD = B.w40(D, B.DB[bi], "codex_REF", fam)
    D = D.assign(p=pD, p_air=pD - D.in_temp.values); M = M.assign(p=p, p_air=p - M.in_temp.values)
    out = p.copy(); De = (D.hour <= H).values; Me = (M.hour <= H).values
    for k in M.fold.unique():
        te = Me & (M.fold == k).values
        if not te.any():
            continue
        vd = set(zip(M.farm[M.fold == k], M.day[M.fold == k]))
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {("F47" if f == "F13" else "F13", d + j) for f, d in vd for j in range(-5, 6)}
        tr = De & np.array([(f, d) not in forb for f, d in zip(D.farm, D.day)])
        mdl = T.lgb.LGBMRegressor(n_estimators=100, learning_rate=.03, num_leaves=15, min_child_samples=100, subsample=.8, subsample_freq=1,
                                  colsample_bytree=.8, random_state=1, verbose=-1, deterministic=True, force_col_wise=True, n_jobs=2)
        mdl.fit(D.loc[tr, T.FEAT], (D.sub_temp - D.p).values[tr])
        out[te] = p[te] + .5 * mdl.predict(M.loc[te, T.FEAT])
    return out


def main():
    CT = T.chain_table(); D = T.layout_frame("DIAG10", CT)
    rng = np.random.default_rng(20261121); ok = {"H3": True, "H5": True}; diffs = {"H3": [], "H5": []}
    for L, (d, pre, bs, cx) in T.LAY.items():
        M = T.layout_frame(L, CT); y = M.sub_temp.values; cells = {"H3": [], "H5": []}; base_rel = []
        for bi, b in enumerate(bs):
            for fam in "AB":
                p0 = B.w40(M, b, cx, fam); c7 = corrected(M, p0, D, bi, fam, 7)
                base_rel.append(rm(c7 - y) / rm(p0 - y) - 1)
                for name, H in (("H3", 3), ("H5", 5)):
                    cH = corrected(M, p0, D, bi, fam, H); cells[name].append(rm(cH - y) / rm(c7 - y) - 1)
                    if bi == 0 and fam == "A" and L not in ("EL1", "EXT10", "EXT12"):
                        key = (M.farm + "_" + (M.day // 5).astype(str)).values
                        diffs[name].append(pd.Series((cH - y) ** 2 - (c7 - y) ** 2).groupby(key).mean())
        print("  %-6s 0-7 vs W40G-S %s | H3 vs 0-7 %s | H5 vs 0-7 %s" % (L, " ".join("%+.2f%%" % (100 * x) for x in base_rel),
              " ".join("%+.2f%%" % (100 * x) for x in cells["H3"]), " ".join("%+.2f%%" % (100 * x) for x in cells["H5"])), flush=True)
        if L not in ("EXT10", "EXT12"):
            for n in ok:
                ok[n] &= all(x < 0 for x in cells[n])
    for n in ("H3", "H5"):
        S = pd.concat(diffs[n], axis=1).mean(axis=1).values; idx = rng.integers(0, len(S), (20000, len(S))); p = float((S[idx].mean(1) >= 0).mean())
        print("%s vs 0-7: all cells %s, pooled P(worse) %.4f -> %s" % (n, ok[n], p, "PASS" if ok[n] and p < ALPHA else "FAIL"))


if __name__ == "__main__":
    main()
