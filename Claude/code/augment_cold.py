# -*- coding: utf-8 -*-
"""Can cold rows borrowed from other greenhouses strengthen cold prediction?

Rejecting whole greenhouses is not the only option: we can keep the rows we
lack (cold ones) and discard the rest.  The obstacle is that each greenhouse
sits at its own substrate-air level, so donor rows are re-levelled first -- and
the correction is computed ONLY from warm-band statistics, which F13/F47 also
have, so anything that works here is applicable to them.

Leave-one-greenhouse-out over the 40 testbed greenhouses:
    train on X's rows above 8 degC (+ donors' cold rows)  ->  predict X's <= 6.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from common import load_raw

CORE = ["in_temp", "in_hum", "in_co2"]
WARM, COLD, DONOR_COLD = 8.0, 6.0, 7.0
MIN_COLD, MIN_WARM = 20, 1000
LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
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
    out["sub_temp"] = p["sub_temp"]
    out["farm"] = g.farm.iloc[0]
    return out.dropna(subset=["sub_temp"])


def mix_fit_predict(X, y, Xt, w=None):
    lin = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, y, ridge__sample_weight=w)
    res = lgb.LGBMRegressor(**LGB).fit(X, y - lin.predict(X), sample_weight=w)
    tree = lgb.LGBMRegressor(**LGB).fit(X, y, sample_weight=w)
    return 0.4 * tree.predict(Xt) + 0.6 * (lin.predict(Xt) + res.predict(Xt))


def main():
    tX, ty, sX = load_raw()
    d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    farms = sorted(f for f in d.farm.unique() if f not in ("F13", "F47", "F32"))

    print("피처 생성 중 ...", flush=True)
    F = {f: feats(d[d.farm == f].copy()) for f in farms}
    cols = [c for c in next(iter(F.values())).columns if c not in ("sub_temp", "farm")]

    # warm-band level of each greenhouse: median(substrate - smoothed air)
    level = {f: float((g.loc[g.in_temp > WARM, "sub_temp"]
                       - g.loc[g.in_temp > WARM, "in_temp_ewm3"]).median())
             for f, g in F.items()}

    usable = [f for f in farms
              if (F[f].in_temp <= COLD).sum() >= MIN_COLD
              and (F[f].in_temp > WARM).sum() >= MIN_WARM]
    print("대상 %d곳, 기증 가능 %d곳\n" % (len(usable), len(farms)))

    rows = []
    for i, X in enumerate(usable, 1):
        g = F[X]
        own = g[g.in_temp > WARM]
        te = g[g.in_temp <= COLD]
        med = own[cols].median()
        Xt, yt = te[cols].fillna(med), te.sub_temp.values

        donors = [f for f in farms if f != X]
        sim = sorted(donors, key=lambda f: abs(level[f] - level[X]))
        pools = {}
        for nm, dl in [("all", donors), ("top10", sim[:10])]:
            parts = [F[f][F[f].in_temp <= DONOR_COLD] for f in dl]
            pools[nm] = pd.concat(parts) if parts else None

        def build(pool, calib, weight_half):
            if pool is None or not len(pool):
                return None
            p = pool.copy()
            if calib:
                p["sub_temp"] = p.sub_temp + p.farm.map(lambda f: level[X] - level[f])
            Xa = pd.concat([own[cols], p[cols]]).fillna(med)
            ya = pd.concat([own.sub_temp, p.sub_temp])
            w = None
            if weight_half:
                w = np.r_[np.ones(len(own)), np.full(len(p), 0.5 * len(own) / len(p))]
            return mix_fit_predict(Xa, ya, Xt, w)

        r = {"farm": X, "n_cold": len(te)}
        r["own"] = mix_fit_predict(own[cols].fillna(med), own.sub_temp, Xt)
        r["raw_all"] = build(pools["all"], False, False)
        r["cal_all"] = build(pools["all"], True, False)
        r["cal_top10"] = build(pools["top10"], True, False)
        r["cal_all_w"] = build(pools["all"], True, True)
        out = {"farm": X, "n_cold": len(te)}
        for k in ("own", "raw_all", "cal_all", "cal_top10", "cal_all_w"):
            p = r[k]
            out[k + "_rmse"] = float(np.sqrt(np.mean((p - yt) ** 2))) if p is not None else np.nan
            out[k + "_bias"] = float(np.mean(p - yt)) if p is not None else np.nan
        rows.append(out)
        print("  [%2d/%2d] %s  own %.3f  cal_all %.3f  cal_top10 %.3f"
              % (i, len(usable), X, out["own_rmse"], out["cal_all_rmse"],
                 out["cal_top10_rmse"]), flush=True)

    t = pd.DataFrame(rows).set_index("farm")
    t.to_csv("augment_cold.csv")

    print("\n=== 추운 구간 예측 성적 (대상 %d곳) ===" % len(t))
    names = {"own": "자기 온실만 (기준)", "raw_all": "+ 타 온실 추운행, 보정 없음",
             "cal_all": "+ 타 온실 추운행, 수준 보정", "cal_top10": "+ 닮은 10곳만, 수준 보정",
             "cal_all_w": "+ 전체 보정, 가중치 50%"}
    print("  %-30s %9s %9s %9s" % ("구성", "RMSE중앙", "편향중앙", "기준보다 나은 곳"))
    for k, nm in names.items():
        win = int((t[k + "_rmse"] < t["own_rmse"]).sum()) if k != "own" else 0
        print("  %-30s %9.3f %9.3f %13s"
              % (nm, t[k + "_rmse"].median(), t[k + "_bias"].median(),
                 "-" if k == "own" else "%d/%d" % (win, len(t))))

    print("\n=== 짝지은 차이 (음수 = 개선), 온실별 중앙값과 부트스트랩 95% ===")
    rng = np.random.default_rng(0)
    for k, nm in names.items():
        if k == "own":
            continue
        dd = (t[k + "_rmse"] - t["own_rmse"]).dropna().values
        bs = [np.median(rng.choice(dd, len(dd))) for _ in range(2000)]
        print("  %-30s %+.4f  [%+.4f, %+.4f]"
              % (nm, np.median(dd), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))


if __name__ == "__main__":
    main()
