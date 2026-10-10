# -*- coding: utf-8 -*-
"""TC1b (SCREEN v2, 2026-10-11 집 클로드, user: "v2 진행해봐").  TC1 (380db53) improved every cell of DIAG10 and F..K
(-0.4..-1.3 %, hours 0-7 -1..-3 %, pooled P .0000) but EL1 got worse (+3.1..+3.4 %): EL1 holds only the 46 pass-2 days, so
its within-layout cross-fit trained the corrector on ~380 early rows (others ~2,600).  v2 removes that artefact: for EVERY
layout and fold the corrector trains on the same source - stored DIAG10 OOF rows (all 400 labelled days, hours 0-7) minus
the evaluated fold's days with the usual purge (same farm +-1, other farm +-5).  Training target = y - W40G-S built from the
DIAG10 members with the SAME BASE-seed index and PFN family as the evaluated cell.  Everything else identical to TC1
(chain features, LightGBM 200/.03/15/40 seed 1, hours 0-7, shrink 0.5, MASK world).
Known caveat (fixed now): DIAG10 OOF residuals of training days come from member models that saw the evaluated fold's days
(stacking leak, mild); the decisive test on a never-used layout will need refits.
Fixed before running.  PROMOTE iff every BASE-seed x PFN-family cell better on DIAG10, EL1, F, G, H, I, J, K AND pooled
(DIAG10, F..K) farm x 5-day block bootstrap P(worse) < .025 AND neither EXT10 nor EXT12 significantly worse (seed-mean worse
with block-bootstrap share(better) < .025 -> fail).  Hours 0-7 and pass-2 reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tc1b_chain_history_common_train_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("tc1", os.path.join(HERE, "tc1_chain_history_early_hours_v1.py")); T = importlib.util.module_from_spec(s); s.loader.exec_module(T)
ALPHA = .025; H = 7; FEAT = T.FEAT; rm = T.rm
DB = ("base_REF_7", "base_REF_101")


def w40(M, b, cx, fam):
    g = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1))
    return (.4 * M[b] + (.2 + .4 * (1 - g)) * M[cx] + .4 * g * M["pfn_" + fam]).values


def corrected(M, p, D, bi, fam):
    """M: evaluated layout frame, p: its W40G-S; D: DIAG10 frame (training source)."""
    pD = w40(D, DB[bi], "codex_REF", fam)
    D = D.assign(p=pD, p_air=pD - D.in_temp.values); M = M.assign(p=p, p_air=p - M.in_temp.values)
    out = p.copy(); De = (D.hour <= H).values; Me = (M.hour <= H).values
    for k in M.fold.unique():
        te = Me & (M.fold == k).values
        if not te.any():
            continue
        vd = set(zip(M.farm[M.fold == k], M.day[M.fold == k]))
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {("F47" if f == "F13" else "F13", d + j) for f, d in vd for j in range(-5, 6)}
        tr = De & np.array([(f, d) not in forb for f, d in zip(D.farm, D.day)])
        mdl = T.lgb.LGBMRegressor(n_estimators=200, learning_rate=.03, num_leaves=15, min_child_samples=40, subsample=.8, subsample_freq=1,
                                  colsample_bytree=.8, random_state=1, verbose=-1, deterministic=True, force_col_wise=True)
        mdl.fit(D.loc[tr, FEAT], (D.sub_temp - D.p).values[tr])
        out[te] = p[te] + .5 * mdl.predict(M.loc[te, FEAT])
    return out


def block_p(M, a, c, rng):
    cl = (M.farm + "_" + (M.day // 5).astype(str)).values
    dd = pd.Series((c - M.sub_temp.values) ** 2 - (a - M.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd))); v = dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)
    return float((v >= 0).mean())


def main():
    CT = T.chain_table(); D = T.layout_frame("DIAG10", CT)
    rng = np.random.default_rng(20261115); ok = True; pool = []
    for L, (d, pre, bs, cx) in T.LAY.items():
        M = T.layout_frame(L, CT); y = M.sub_temp.values; cells, e7, p2, pw = [], [], [], []
        for bi, b in enumerate(bs):
            for fam in "AB":
                a = w40(M, b, cx, fam); c = corrected(M, a, D, bi, fam)
                cells.append(rm(c - y) / rm(a - y) - 1); m7 = M.hour.values <= H; m2 = M.day.values >= 179
                e7.append(rm((c - y)[m7]) / rm((a - y)[m7]) - 1); p2.append(rm((c - y)[m2]) / rm((a - y)[m2]) - 1)
                pw.append(block_p(M, a, c, rng))
                if bi == 0 and fam == "A" and L not in ("EL1", "EXT10", "EXT12"):
                    pool.append((M, a, c))
        line = "  %-6s cells %s | hours0-7 %+.2f%% | pass-2 %+.2f%% | P(worse) max %.3f" % (L, " ".join("%+.2f%%" % (100 * x) for x in cells), 100 * np.mean(e7), 100 * np.mean(p2), max(pw))
        if L in ("EXT10", "EXT12"):
            bad = np.mean(cells) > 0 and (1 - min(pw)) < ALPHA; ok &= not bad; line += " -> %s" % ("SIGNIFICANTLY WORSE" if bad else "ok")
        else:
            ok &= all(x < 0 for x in cells)
        print(line, flush=True)
    sm, n = [], []
    for M, a, c in pool:
        cl = (M.farm + "_" + (M.day // 5).astype(str)).values
        dd = pd.Series((c - M.sub_temp.values) ** 2 - (a - M.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"]); sm.append(dd["sum"].values); n.append(dd["count"].values)
    sm, n = np.concatenate(sm), np.concatenate(n); idx = rng.integers(0, len(sm), (20000, len(sm)))
    p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    print("pooled P(worse) %.4f -> %s" % (p, "PROMOTE" if ok and p < ALPHA else "stop"))


if __name__ == "__main__":
    main()
