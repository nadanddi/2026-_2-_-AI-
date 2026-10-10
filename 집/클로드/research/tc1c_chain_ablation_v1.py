# -*- coding: utf-8 -*-
"""TC1c (ABLATION, 2026-10-11 집 클로드; critic of TC1/TC1b).  TC1b passed the screen, but the corrector also sees hour /
in_temp / p / p - in_temp, so the gain may be a generic early-hour bias correction of W40G-S rather than the chain history.
Fixed before running (critic's design):
  (a) NOCHAIN: features hour, in_temp, p, p_air only
  (b) CHAIN  : TC1 features (adds lit1/lit2/ch_in23/ch_eve/ch_cost/jump)
Both: TC1b training source (DIAG10 OOF, purge same farm +-1 / other farm +-5), LightGBM 100 trees, lr .03, 15 leaves,
min_child 100 (stronger regularisation, fixed now, no tuning), seed 1, hours 0-7, shrink 0.5.
Layouts DIAG10, EL1, EXT10, EXT12, F..K.  Pooled bootstrap: per (farm, 5-day block) the squared-error difference is first
AVERAGED over layouts DIAG10 + F..K (same days, not counted six times), then blocks are resampled.  k = 2 -> alpha .0125.
Verdicts:  CHAIN-GAIN iff (b) beats (a) in every cell of DIAG10, EL1, F..K and pooled P((b) worse than (a)) < .0125.
           BIAS-GAIN  iff (a) beats W40G-S in every cell of DIAG10, EL1, F..K and pooled P((a) worse than W40G-S) < .0125.
EXT10/EXT12 reported (cold safety).  Either verdict still needs a never-used layout with refitted members.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tc1c_chain_ablation_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("tc1", os.path.join(HERE, "tc1_chain_history_early_hours_v1.py")); T = importlib.util.module_from_spec(s); s.loader.exec_module(T)
s = importlib.util.spec_from_file_location("tc1b", os.path.join(HERE, "tc1b_chain_history_common_train_v1.py")); B = importlib.util.module_from_spec(s); s.loader.exec_module(B)
ALPHA = .0125; H = 7; rm = T.rm
FA = ["hour", "in_temp", "p", "p_air"]; FB = T.FEAT


def corrected(M, p, D, bi, fam, feats):
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
                                  colsample_bytree=.8, random_state=1, verbose=-1, deterministic=True, force_col_wise=True)
        mdl.fit(D.loc[tr, feats], (D.sub_temp - D.p).values[tr])
        out[te] = p[te] + .5 * mdl.predict(M.loc[te, feats])
    return out


def main():
    CT = T.chain_table(); D = T.layout_frame("DIAG10", CT)
    rng = np.random.default_rng(20261116)
    okA = okB = True; diffs = {"ab": [], "a0": []}
    for L, (d, pre, bs, cx) in T.LAY.items():
        M = T.layout_frame(L, CT); y = M.sub_temp.values; ca, cb, cba = [], [], []
        for bi, b in enumerate(bs):
            for fam in "AB":
                p0 = B.w40(M, b, cx, fam); pa = corrected(M, p0, D, bi, fam, FA); pb = corrected(M, p0, D, bi, fam, FB)
                ca.append(rm(pa - y) / rm(p0 - y) - 1); cb.append(rm(pb - y) / rm(p0 - y) - 1); cba.append(rm(pb - y) / rm(pa - y) - 1)
                if bi == 0 and fam == "A" and L not in ("EL1", "EXT10", "EXT12"):
                    key = M.farm + "_" + (M.day // 5).astype(str)
                    diffs["ab"].append(pd.Series((pb - y) ** 2 - (pa - y) ** 2).groupby(key.values).mean())
                    diffs["a0"].append(pd.Series((pa - y) ** 2 - (p0 - y) ** 2).groupby(key.values).mean())
        print("  %-6s NOCHAIN vs W40G-S %s | CHAIN vs W40G-S %s | CHAIN vs NOCHAIN %s" % (
            L, " ".join("%+.2f%%" % (100 * x) for x in ca), " ".join("%+.2f%%" % (100 * x) for x in cb), " ".join("%+.2f%%" % (100 * x) for x in cba)), flush=True)
        if L not in ("EXT10", "EXT12"):
            okA &= all(x < 0 for x in ca); okB &= all(x < 0 for x in cba)
    for name, ok, key in (("CHAIN-GAIN (b vs a)", okB, "ab"), ("BIAS-GAIN (a vs W40G-S)", okA, "a0")):
        S = pd.concat(diffs[key], axis=1).mean(axis=1).values          # per block, averaged over layouts
        idx = rng.integers(0, len(S), (20000, len(S))); p = float((S[idx].mean(1) >= 0).mean())
        print("%s: all cells %s, pooled P(worse) %.4f (%d blocks) -> %s" % (name, ok, p, len(S), "PASS" if ok and p < ALPHA else "FAIL"))


if __name__ == "__main__":
    main()
