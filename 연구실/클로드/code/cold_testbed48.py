# -*- coding: utf-8 -*-
"""Cold-extrapolation testbed on ALL 48 other greenhouses.

Rounding was just shown to be free (rounding_cost.log), so integer-resolution
greenhouses are valid test instances.  Each greenhouse is trained on rows above
8 degC only and asked to predict its own rows at or below 6 degC -- the same
situation F13/F47 face on the evaluation set.  Four methods are compared, one
of which is the cold-row mix the current temperature candidate uses.

This re-runs catalogue 6.75 without the unexplained 32-greenhouse subset.
Only in_temp / in_hum / in_co2 are used, because those are the columns these
greenhouses share with F13/F47.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from common import load_raw

CORE = ["in_temp", "in_hum", "in_co2"]
WARM, COLD = 8.0, 6.0
MIN_COLD, MIN_WARM = 20, 1000


def feats(g):
    g = g.sort_values("t")
    full = np.arange(g.t.min(), g.t.max() + 1)
    p = g.set_index("t").reindex(full)
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
    out["row_id"] = p["row_id"]
    return out.dropna(subset=["row_id", "sub_temp"])


def run_farm(df):
    F = feats(df)
    cols = [c for c in F.columns if c not in ("sub_temp", "row_id")]
    tr = F[F.in_temp > WARM]
    te = F[F.in_temp <= COLD]
    if len(te) < MIN_COLD or len(tr) < MIN_WARM:
        return None
    X, y = tr[cols].fillna(tr[cols].median()), tr.sub_temp
    Xt = te[cols].fillna(tr[cols].median())
    yt = te.sub_temp.values

    tree = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             min_child_samples=40, random_state=7, n_jobs=4,
                             verbose=-1).fit(X, y).predict(Xt)
    lin_m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, y)
    lin = lin_m.predict(Xt)
    res = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                            min_child_samples=40, random_state=7, n_jobs=4,
                            verbose=-1).fit(X, y - lin_m.predict(X)).predict(Xt)
    linres = lin + res
    mix = 0.4 * tree + 0.6 * linres

    def sc(p):
        return float(np.sqrt(np.mean((p - yt) ** 2))), float(np.mean(p - yt))
    return dict(n_cold=len(te), n_warm=len(tr),
                **{k: v for k, (r, b) in
                   [("tree", sc(tree)), ("lin", sc(lin)),
                    ("linres", sc(linres)), ("mix", sc(mix))]
                   for k, v in [(k + "_rmse", r), (k + "_bias", b)]})


def main():
    tX, ty, sX = load_raw()
    d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    farms = sorted(f for f in d.farm.unique() if f not in ("F13", "F47", "F32"))
    rows = []
    for f in farms:
        r = run_farm(d[d.farm == f].copy())
        if r:
            r["farm"] = f
            rows.append(r)
        print("  %s %s" % (f, "ok" if r else "skip (추운 행 또는 따뜻한 행 부족)"), flush=True)
    t = pd.DataFrame(rows).set_index("farm")
    t.to_csv("cold_testbed48.csv")

    print("\n=== 대상 온실 %d곳 / 48곳 ===" % len(t))
    skipped = [f for f in farms if f not in t.index]
    if skipped:
        print("  제외:", ", ".join(skipped))

    print("\n=== 방법별 한랭 외삽 성적 (각 온실 >8℃로 학습 → ≤6℃ 예측) ===")
    print("  %-26s %9s %9s %9s %9s" % ("방법", "RMSE중앙", "RMSE평균", "편향중앙", "최선인곳"))
    best = t[["tree_rmse", "lin_rmse", "linres_rmse", "mix_rmse"]].idxmin(axis=1)
    names = {"tree": "트리만", "lin": "선형만", "linres": "선형+트리잔차",
             "mix": "0.4트리+0.6선형잔차 (G_C2 비율)"}
    for k, nm in names.items():
        print("  %-26s %9.3f %9.3f %9.3f %7d곳"
              % (nm, t[k + "_rmse"].median(), t[k + "_rmse"].mean(),
                 t[k + "_bias"].median(), int((best == k + "_rmse").sum())))

    print("\n=== 짝지은 비교: mix(G_C2 비율) vs 각 방법 ===")
    for k in ("tree", "lin", "linres"):
        d_ = t["mix_rmse"] - t[k + "_rmse"]
        win = int((d_ < 0).sum())
        print("  vs %-14s mix가 이긴 곳 %2d/%d,  차이 중앙값 %+.3f"
              % (names[k], win, len(t), d_.median()))

    print("\n=== 참고: 카탈로그 6.75는 32곳으로 28곳 우세 보고 ===")
    w = int((t["mix_rmse"] < t["linres_rmse"]).sum())
    print("  48곳 전수 재현: mix가 linres보다 나은 곳 %d/%d (%.0f%%)" % (w, len(t), 100 * w / len(t)))
    print("  상세: cold_testbed48.csv")


if __name__ == "__main__":
    main()
