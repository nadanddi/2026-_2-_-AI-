# -*- coding: utf-8 -*-
"""TC1 (SCREEN, 2026-10-11 집 클로드, user: "온도도 더 분석해보자 ... 창의적으로"): CHAIN-AWARE evening history for the
early hours.  6b.13: hours 0-2 are 12.5 % of rows but 19-23 % of squared error ("information gap right after the midnight
seam").  Records are day pieces of different houses; the literal previous record is often another house, so the model either
resets at midnight or reads the wrong house's evening.  TC1 picks, among the previous 1-3 records of the same farm, the one
whose 23 h state best continues this record's 0 h state (in_temp / in_hum / in_co2 / heating / thermal / vent / circfan,
z-scaled mean |difference|) and uses ITS evening as history.  Causal: earlier records' inputs + this record's hour 0..h.
Training-row features in the MASK world (test_X inputs absent -> earlier evaluation records are not candidates there).
Corrector (stacking on stored OOF, no member refit): LightGBM (200 trees, lr .03, 15 leaves, min_child 40, seed 1), target
y - W40G-S, features: hour, in_temp, p, p - in_temp, lit1_in23, lit2_in23 (literal previous / second-previous record's 23 h
in_temp), ch_in23, ch_eve (chain predecessor's 23 h and 18-23 h mean in_temp), ch_cost, jump = in_temp - ch_in23.
Applied to hours 0-7 only, shrink 0.5.  Cross-fitted inside each layout (train on the other folds' rows of hours 0-7).
Fixed before running.  Layouts with stored members: DIAG10, EL1, F, G, H, I, J, K (all rows, rough days excluded).
PROMOTE iff better in every BASE-seed x PFN-family cell of all 8 layouts AND pooled (DIAG10, F..K) farm x 5-day block
bootstrap P(worse) < .025; EXT10 / EXT12 reported (cold safety); hours 0-7 and pass-2 reported.  A promotion needs a new
layout with new seeds.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tc1_chain_history_early_hours_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
import lightgbm as lgb
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
ALPHA = .025; H = 7
LAY = {"DIAG10": ("tt1_ckpt", "DIAG10_", ("base_REF_7", "base_REF_101"), "codex_REF"),
       "EL1": ("tt1_ckpt", "EL1_", ("base_REF_7", "base_REF_101"), "codex_REF"),
       "EXT10": ("tt1_ckpt", "EXT10_", ("base_REF_7", "base_REF_101"), "codex_REF"),
       "EXT12": ("tt1_ckpt", "EXT12_", ("base_REF_7", "base_REF_101"), "codex_REF"),
       "F": ("tt12_ckpt", "fresh_", ("base_202", "base_303"), "codex_R"),
       "G": ("tt13_ckpt", "G_", ("base_R_404", "base_R_505"), "codex_R"),
       "H": ("tt17_ckpt", "H_", ("base_R_606", "base_R_707"), "codex_R"),
       "I": ("tt23_ckpt", "I_", ("base_R_808", "base_R_909"), "codex_R"),
       "J": ("tt24_ckpt", "J_", ("base_R_1010", "base_R_1111"), "codex_R"),
       "K": ("tt26_ckpt", "K_", ("base_R_1212", "base_R_1313"), "codex_R")}
V = ["in_temp", "in_hum", "in_co2", "act_heating", "act_thermal", "act_vent", "act_circfan"]
FEAT = ["hour", "in_temp", "p", "p_air", "lit1_in23", "lit2_in23", "ch_in23", "ch_eve", "ch_cost", "jump"]
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))


def chain_table():
    """MASK world: training inputs only.  Per (farm, day): literal / chain predecessor evening features."""
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
    sd = X[V].std()
    h0 = X[X.hour == 0].set_index(["farm", "day"])[V]; h23 = X[X.hour == 23].set_index(["farm", "day"])[V]
    eve = X[X.hour >= 18].groupby(["farm", "day"]).in_temp.mean()
    allx = pd.concat([pd.read_csv(os.path.join(env.DATA, f), usecols=["row_id"]) for f in ("train_X.csv", "test_X.csv")])
    allx = allx[allx.row_id.str[:3].isin(["F13", "F47"])]
    rec = sorted({(r[:3], int(r[4:7])) for r in allx.row_id})          # record order incl. evaluation records (MASKed)
    rows = []
    for f in ("F13", "F47"):
        days = [d for ff, d in rec if ff == f]
        for i, d in enumerate(days):
            prev = days[max(0, i - 3):i][::-1]          # nearest first
            lit = [h23.in_temp.get((f, p), np.nan) for p in prev[:2]] + [np.nan, np.nan]
            best, bc = None, np.inf
            if (f, d) in h0.index:
                for p in prev:
                    if (f, p) in h23.index:
                        c = np.nanmean(np.abs((h0.loc[(f, d)] - h23.loc[(f, p)]) / sd).values)
                        if np.isfinite(c) and c < bc:
                            best, bc = p, c
            rows.append(dict(farm=f, day=d, lit1_in23=lit[0], lit2_in23=lit[1],
                             ch_in23=h23.in_temp.get((f, best), np.nan) if best is not None else np.nan,
                             ch_eve=eve.get((f, best), np.nan) if best is not None else np.nan, ch_cost=bc if best is not None else np.nan))
    return pd.DataFrame(rows)


def layout_frame(L, CT):
    d, pre, bs, cx = LAY[L]; D = os.path.join(env.LOCAL, d)
    parts = []
    for k, fn in enumerate(sorted(f for f in os.listdir(D) if f.startswith(pre) and f.endswith(".csv"))):
        M = pd.read_csv(os.path.join(D, fn)); M["fold"] = k; parts.append(M)
    M = pd.concat(parts, ignore_index=True).merge(CT, on=["farm", "day"], how="left")
    M["jump"] = M.in_temp - M.ch_in23
    return M[np.array([(f, x) not in RD for f, x in zip(M.farm, M.day)])].reset_index(drop=True)


def corrected(M, p):
    M = M.assign(p=p, p_air=p - M.in_temp.values); out = p.copy(); early = (M.hour <= H).values
    for k in M.fold.unique():
        tr = early & (M.fold != k).values; te = early & (M.fold == k).values
        if not te.any() or tr.sum() < 200:
            continue
        mdl = lgb.LGBMRegressor(n_estimators=200, learning_rate=.03, num_leaves=15, min_child_samples=40, subsample=.8, subsample_freq=1,
                                colsample_bytree=.8, random_state=1, verbose=-1, deterministic=True, force_col_wise=True)
        mdl.fit(M.loc[tr, FEAT], (M.sub_temp - M.p).values[tr])
        out[te] = p[te] + .5 * mdl.predict(M.loc[te, FEAT])
    return out


def main():
    CT = chain_table()
    print("chain predecessor found for %.0f%% of records (MASK world)" % (100 * CT.ch_in23.notna().mean()))
    rng = np.random.default_rng(20261114); ok = True; pool = []
    for L, (d, pre, bs, cx) in LAY.items():
        M = layout_frame(L, CT); y = M.sub_temp.values
        g = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1)); cells, e7, p2 = [], [], []
        for b in bs:
            for fam in "AB":
                a = (.4 * M[b] + (.2 + .4 * (1 - g)) * M[cx] + .4 * g * M["pfn_" + fam]).values; c = corrected(M, a)
                cells.append(rm(c - y) / rm(a - y) - 1); m7 = M.hour.values <= H; m2 = M.day.values >= 179
                e7.append(rm((c - y)[m7]) / rm((a - y)[m7]) - 1); p2.append(rm((c - y)[m2]) / rm((a - y)[m2]) - 1)
                if b == bs[0] and fam == "A" and L not in ("EL1", "EXT10", "EXT12"):
                    pool.append((M, a, c))
        print("  %-6s cells %s | hours0-7 %+.2f%% | pass-2 %+.2f%%" % (L, " ".join("%+.2f%%" % (100 * x) for x in cells), 100 * np.mean(e7), 100 * np.mean(p2)), flush=True)
        if L not in ("EXT10", "EXT12"):
            ok &= all(x < 0 for x in cells)
    sm, n = [], []
    for M, a, c in pool:
        cl = (M.farm + "_" + (M.day // 5).astype(str)).values
        dd = pd.Series((c - M.sub_temp.values) ** 2 - (a - M.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"]); sm.append(dd["sum"].values); n.append(dd["count"].values)
    sm, n = np.concatenate(sm), np.concatenate(n); idx = rng.integers(0, len(sm), (20000, len(sm)))
    p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    print("pooled P(worse) %.4f -> %s" % (p, "PROMOTE" if ok and p < ALPHA else "stop"))


if __name__ == "__main__":
    main()
