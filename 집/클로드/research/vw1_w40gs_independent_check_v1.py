# -*- coding: utf-8 -*-
"""VW1 (independent check of the teammate's candidate W40G-S; fixed before running;
2026-10-03 집 클로드).
W40G   = 0.4 BASE + (0.2 + 0.4(1-g)) CODEX + 0.4 g PFN,  g = clip((in_temp-8)/2, 0, 1)
W40G-S = same, CODEX and PFN with `day` -> season (BASE keeps `day`), as described in
         설명자료_12 / make_submission_v13_temp.py.
Members from Codex TK2 (local/temp_tk_season_20261003_v1): fold-wise season built from
that fold's TRAINING days only (validation/eval days interpolated by record day, as at
test time); members.csv (BASE seeds 7/101, CODEX base/season), pfn_<val>_<fold>_<ctx>.npz
(base/season), PFN = mean over contexts 1-8 (family A) or 17-24 (family B).
Cells: BASE seed {7,101} x PFN family {A,B} = 4 combos x DIAG10/EXT10/EXT12, + EL1.
Reading (fixed, user's rule): adopt-level if W40G-S better in all 12 cells and DIAG10
block bootstrap (farm x 5-day, 20000) P(worse) < .025 for every combo; EL1 and pass-2
(record day >= 179) rows reported."""
import env  # noqa: F401
import glob
import os
import numpy as np
import pandas as pd

TK = os.path.join(env.ROOT, "집", "코덱스", "local", "temp_tk_season_20261003_v1")
M = pd.read_csv(os.path.join(TK, "members.csv"))
fam = {"A": range(1, 9), "B": range(17, 25)}
pf = []
for (v, k), G in M.groupby(["validator", "fold"]):
    for fn, ctxs in fam.items():
        b, s = [], []
        for c in ctxs:
            z = np.load(os.path.join(TK, "pfn_%s_%d_%d.npz" % (v, k, c)), allow_pickle=True)
            b.append(pd.Series(z["base"], index=z["row_id"])); s.append(pd.Series(z["season"], index=z["row_id"]))
        bb, ss = pd.concat(b, axis=1).mean(axis=1), pd.concat(s, axis=1).mean(axis=1)
        pf.append(pd.DataFrame({"row_id": bb.index, "validator": v, "fold": k, "fam": fn, "pfn_base": bb.values, "pfn_season": ss.reindex(bb.index).values}))
PF = pd.concat(pf)
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
rng = np.random.default_rng(20261003)
ok = True
rows = []
for sd in (7, 101):
    for fn in ("A", "B"):
        X = M.merge(PF[PF.fam == fn], on=["row_id", "validator", "fold"], how="inner")
        g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1))
        X["w40"] = 0.4 * X["mask_base_%d" % sd] + (0.2 + 0.4 * (1 - g)) * X.codex_base + 0.4 * g * X.pfn_base
        X["w40s"] = 0.4 * X["mask_base_%d" % sd] + (0.2 + 0.4 * (1 - g)) * X.codex_season + 0.4 * g * X.pfn_season
        for v in ("DIAG10", "EXT10", "EXT12", "EL1"):
            G = X[X.validator == v]
            if not len(G):
                continue
            L = G.day >= 179
            a, b = r(G.w40 - G.sub_temp), r(G.w40s - G.sub_temp)
            la, lb = (r(G.w40[L] - G.sub_temp[L]), r(G.w40s[L] - G.sub_temp[L])) if L.any() else (np.nan, np.nan)
            p = np.nan
            if v == "DIAG10":
                G = G.copy(); G["cl"] = G.farm + "_" + (G.day // 5).astype(str)
                dd = (G.w40s - G.sub_temp) ** 2 - (G.w40 - G.sub_temp) ** 2
                cl = dd.groupby(G.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
                idx = rng.integers(0, len(sm), (20000, len(sm)))
                p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
                ok &= p < .025
            if v != "EL1":
                ok &= b < a
            rows.append(dict(seed=sd, pfn=fn, validator=v, n=len(G), W40G=a, W40GS=b, pct=100 * (b / a - 1), late_W40G=la, late_W40GS=lb,
                             late_pct=100 * (lb / la - 1) if la == la else np.nan, p_worse=p))
R = pd.DataFrame(rows)
print(R.round(4).to_string(index=False))
print("\nVW1 adopt-level (user rule):", ok)
