# -*- coding: utf-8 -*-
"""Is the day-level offset unique to F13/F47, or intrinsic to the task?

Catalogue 6b.2-6b.3: 58% of the temperature error is a whole-day offset, and
nothing available explains it (day-level CV R2 of -0.03 to -0.05).  That was
measured on F13/F47, whose record concatenates several source greenhouses.

The other 48 greenhouses are continuous records.  Running the same
decomposition there separates two very different worlds:

  * if the day offset is unexplained everywhere -> it is intrinsic, every team
    hits it, and the gap to the leaders is somewhere else;
  * if it is explained in continuous records but not in F13/F47 -> the
    concatenation destroys something specific, and we can name it.

All greenhouses are scored on the three columns they share (in_temp, in_hum,
in_co2) so the comparison is like-for-like; F13/F47 are also run on their full
feature set for reference.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import load_raw, rmse

CORE = ["in_temp", "in_hum", "in_co2"]
LGBP = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=40, random_state=7, n_jobs=4, verbose=-1)


def feats(g):
    g = g.sort_values("t")
    p = g.set_index("t").reindex(np.arange(g.t.min(), g.t.max() + 1))
    f = {}
    for c in CORE:
        s = p[c]
        for hl in (1, 3, 8, 24):
            f["%s_ewm%d" % (c, hl)] = s.ewm(halflife=hl, ignore_na=True).mean()
        for L in (1, 2, 3, 6):
            f["%s_lag%d" % (c, L)] = s.shift(L)
        f[c + "_d1"] = s - s.shift(1)
        f[c + "_r24"] = s.rolling(24, min_periods=6).mean()
        f[c] = s
    f["hour"] = p.index % 24
    f["hr_sin"] = np.sin(2 * np.pi * f["hour"] / 24)
    f["hr_cos"] = np.cos(2 * np.pi * f["hour"] / 24)
    out = pd.DataFrame(f, index=p.index)
    out["sub_temp"] = p.sub_temp
    out["day"] = p.index // 24
    return out.dropna(subset=["sub_temp"])


def block_oof(F, cols, n_folds=5, block=10):
    """10-day blocks round-robin to folds, 1-day buffer -- same shape as ours."""
    days = np.sort(F.day.unique())
    blk = (days - days.min()) // block % n_folds
    fold = pd.Series(blk, index=days)
    oof = np.full(len(F), np.nan)
    for k in range(n_folds):
        vd = set(days[fold.values == k])
        near = {d + o for d in vd for o in (-1, 0, 1)}
        va = F.day.isin(vd).values
        tr = ~F.day.isin(near).values
        if va.sum() < 50 or tr.sum() < 500:
            continue
        m = lgb.LGBMRegressor(**LGBP).fit(F.loc[tr, cols], F.loc[tr, "sub_temp"])
        oof[va] = m.predict(F.loc[va, cols])
    return oof


def decompose(F, oof):
    got = ~np.isnan(oof)
    d = F[got].copy()
    d["res"] = oof[got] - d.sub_temp
    day_off = d.groupby("day").res.mean()
    within = d.res - d.day.map(day_off)
    total = float(np.sqrt(np.mean(d.res ** 2)))
    lvl = float(np.sqrt(np.mean(d.day.map(day_off) ** 2)))
    wit = float(np.sqrt(np.mean(within ** 2)))
    return total, lvl, wit, day_off, d


def explain_offset(d, day_off):
    """Can that day's own input summary predict its offset?  Leave-one-day-out."""
    agg = d.groupby("day").agg(
        t_mean=("in_temp", "mean"), t_min=("in_temp", "min"), t_max=("in_temp", "max"),
        h_mean=("in_hum", "mean"), c_mean=("in_co2", "mean"),
        t_rng=("in_temp", lambda s: s.max() - s.min()),
        t_d1=("in_temp_d1", "std"))
    agg = agg.join(day_off.rename("y")).dropna()
    if len(agg) < 40:
        return np.nan
    X = agg.drop(columns="y").values
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = agg.y.values
    pred = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        A = np.c_[np.ones(m.sum()), X[m]]
        w = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ y[m])
        pred[i] = np.r_[1, X[i]] @ w
    return 1 - np.mean((pred - y) ** 2) / np.var(y)


def main():
    tX, ty, sX = load_raw()
    d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    farms = sorted(d.farm.unique())
    rows = []
    for f in farms:
        F = feats(d[d.farm == f].copy())
        if len(F) < 2000:
            continue
        cols = [c for c in F.columns if c not in ("sub_temp", "day")]
        oof = block_oof(F, cols)
        if np.isnan(oof).all():
            continue
        tot, lvl, wit, day_off, dd = decompose(F, oof)
        r2 = explain_offset(dd, day_off)
        rows.append(dict(farm=f, rmse=tot, day_lvl=lvl, within=wit,
                         share=lvl ** 2 / tot ** 2, r2=r2, n_days=len(day_off)))
        print("  %s rmse %.3f  하루오프셋 %.3f (%.0f%%)  일내 %.3f  오프셋설명 R2 %+.3f"
              % (f, tot, lvl, 100 * lvl ** 2 / tot ** 2, wit, r2), flush=True)
    t = pd.DataFrame(rows).set_index("farm")
    t.to_csv("day_offset48.csv")

    tgt = [f for f in ("F13", "F47") if f in t.index]
    oth = [f for f in t.index if f not in tgt + ["F32"]]
    print("\n=== 하루 오프셋이 오차에서 차지하는 비중 ===")
    print("  F13/F47   : %s" % ", ".join("%s %.0f%%" % (f, 100 * t.loc[f, "share"]) for f in tgt))
    print("  다른 48곳 : 중앙 %.0f%%  (25~75%%: %.0f~%.0f%%)"
          % (100 * t.loc[oth, "share"].median(),
             100 * t.loc[oth, "share"].quantile(.25), 100 * t.loc[oth, "share"].quantile(.75)))

    print("\n=== 그날 입력으로 하루 오프셋을 설명할 수 있나 (leave-one-day-out R2) ===")
    print("  F13/F47   : %s" % ", ".join("%s %+.3f" % (f, t.loc[f, "r2"]) for f in tgt))
    s = t.loc[oth, "r2"].dropna()
    print("  다른 48곳 : 중앙 %+.3f,  양수인 곳 %d/%d,  상위 %+.3f"
          % (s.median(), int((s > 0).sum()), len(s), s.max()))
    print("\n  * R2가 0 이하 = 그날 입력으로는 그날의 어긋남을 못 맞힘")


if __name__ == "__main__":
    main()
