# -*- coding: utf-8 -*-
"""TD9: on cold days, using ONLY the inputs and the sub_temp label (no model residuals), which inputs relate to the
substrate-temperature level?  (2026-10-10 집 클로드, user: "추운 날 주어지는 데이터와 그 데이터의 온도 정답만 봤을 때,
배지 온도의 크기와 연관된 데이터들이 있는지 확인해봐")  Descriptive diagnostic, F13/F47 only.  Fixed before running:
 Day sets: COLD11 = days whose ewm3(in_temp, halflife 3h, causal per farm) < 8 at any hour (11 days);
           COLD85 = the EXT10 cold-side holdout days (85 days, from TT1 checkpoints).
 Step A  raw: Spearman(sub_temp, x) per farm for every input (current hour).
 Step B  'beyond air': AIR = leave-one-day-out OLS of sub_temp on [in_temp, ewm1, ewm3, ewm6, ewm12, sin/cos hour]
         fitted within the day set.  Residual = what air temperature history does not explain.
         For every input x (current hour value and causal 6 h rolling mean) report Spearman(residual, x) per farm.
         'related' iff same sign in F13 and F47 and min |rho| >= .20.
 Step C  incremental value: LODO R^2 of the residual for AIR + x (each related x alone) and AIR + all inputs
         (ridge alpha 10 standardized; LightGBM 200 trees depth 3 as a nonlinear check).  Gain counted only if
         R^2 > .05 and day-bootstrap 95% CI lower > 0.
 Step D  day level: daily mean (sub - in_temp) vs daily means of inputs, Spearman per farm, same rule as B.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u td9_cold_sub_level_relations_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
import common
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LinearRegression
import lightgbm as lgb
CK = os.path.join(env.LOCAL, "tt1_ckpt")
INP = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_hum", "in_co2", "in_rad", "act_vent", "act_side", "act_shade",
       "act_thermal", "act_valve", "act_heating", "act_circfan", "act_co2", "act_fog", "act_cool", "act_pump"]
AIR = ["in_temp", "ewm1", "ewm3", "ewm6", "ewm12", "hs", "hc"]


def prep():
    tX, ty, _ = common.load_raw()
    a = tX.merge(ty[["row_id", "sub_temp"]], on="row_id")
    a = a[a.farm.isin(["F13", "F47"])].sort_values(["farm", "t"]).reset_index(drop=True)
    for h in (1, 3, 6, 12):
        a["ewm%d" % h] = a.groupby("farm").in_temp.transform(lambda s: s.ewm(halflife=h, ignore_na=True).mean())
    a["hs"], a["hc"] = np.sin(2 * np.pi * a.hour / 24), np.cos(2 * np.pi * a.hour / 24)
    for c in INP:
        a[c + "_r6"] = a.groupby("farm")[c].transform(lambda s: s.rolling(6, min_periods=1).mean())
    a["in_temp"] = a.in_temp.fillna(a.ewm1)
    return a


def lodo(X, y, days, model):
    p = np.zeros(len(y))
    for d in np.unique(days):
        m = days != d
        p[~m] = model().fit(X[m], y[m]).predict(X[~m])
    return p


def r2ci(y, p, days, n=2000):
    R = lambda yy, pp: 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    u = np.unique(days); idx = {d: np.where(days == d)[0] for d in u}; rng = np.random.default_rng(0); bs = []
    for _ in range(n):
        j = np.concatenate([idx[d] for d in rng.choice(u, len(u))]); bs.append(R(y[j], p[j]))
    return R(y, p), *np.percentile(bs, [2.5, 97.5])


def rho2(x, c, ycol):
    out = []
    for f in ("F13", "F47"):
        z = x[x.farm == f][[c, ycol]].dropna()
        out.append(spearmanr(z[c], z[ycol])[0] if z[c].nunique() > 2 and len(z) > 10 else np.nan)
    return out


def main():
    a = prep()
    cold11 = a.groupby(["farm", "day"]).ewm3.transform("min") < 8
    G = pd.concat([pd.read_csv(os.path.join(CK, f), usecols=["validator", "farm", "day"]) for f in sorted(os.listdir(CK))])
    s85 = set(map(tuple, G[G.validator == "EXT10"][["farm", "day"]].drop_duplicates().values))
    sets = {"COLD11": a[cold11].copy(), "COLD85": a[[(f, d) in s85 for f, d in zip(a.farm, a.day)]].copy()}
    for nm, x in sets.items():
        x = x.dropna(subset=["sub_temp"]).reset_index(drop=True)
        days = (x.farm + "_" + x.day.astype(str)).values
        print("\n################ %s: %d일, %d행, 배지 평균 %.2f (최저 %.2f), 실내 평균 %.2f" % (nm, len(np.unique(days)), len(x), x.sub_temp.mean(), x.sub_temp.min(), x.in_temp.mean()))
        print("== A 원값 Spearman(배지온도, 입력)  F13 / F47   (참고: in_temp %+.2f / %+.2f)" % tuple(rho2(x, "in_temp", "sub_temp")))
        for c in INP:
            v = rho2(x, c, "sub_temp")
            if np.nanmax(np.abs(v)) >= .2:
                print("   %-12s %+.2f / %+.2f" % (c, *v))
        x["air"] = lodo(x[AIR].values, x.sub_temp.values, days, LinearRegression)
        x["res"] = x.sub_temp - x.air
        R, lo, hi = r2ci(x.sub_temp.values, x.air.values, days)
        print("== B 공기 이력만으로 배지 설명: R² %.3f [%.3f, %.3f], 남은 오차 RMSE %.3f" % (R, lo, hi, np.sqrt((x.res ** 2).mean())))
        rel = []
        for c in INP:
            for cc in (c, c + "_r6"):
                v = rho2(x, cc, "res")
                if not np.isnan(v).any() and np.sign(v[0]) == np.sign(v[1]) and min(map(abs, v)) >= .2:
                    rel.append(cc); print("   연관 %-16s %+.2f / %+.2f" % (cc, *v))
        if not rel:
            print("   연관(두 온실 같은 방향, |rho|>=.2) 없음")
        print("== C 공기 이력 뒤에 남은 부분을 입력으로 더 맞히나 (날 하나 빼기 R², 95% CI)")
        y = x.res.values
        for cc in rel:
            Xc = x[[cc]].fillna(0).values
            R, lo, hi = r2ci(y, lodo(Xc, y, days, LinearRegression), days); print("   %-16s R² %+.3f [%+.3f, %+.3f] %s" % (cc, R, lo, hi, "이득" if R > .05 and lo > 0 else ""))
        cols = INP + [c + "_r6" for c in INP]
        Xa = x[cols].fillna(0).values; Xa = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-9)
        R, lo, hi = r2ci(y, lodo(Xa, y, days, lambda: Ridge(alpha=10)), days); print("   전체 입력 Ridge     R² %+.3f [%+.3f, %+.3f]" % (R, lo, hi))
        R, lo, hi = r2ci(y, lodo(Xa, y, days, lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=.05, verbose=-1)), days); print("   전체 입력 LightGBM  R² %+.3f [%+.3f, %+.3f]" % (R, lo, hi))
        k = x.groupby(days).res.transform("mean")
        print("   남은 부분 중 하루 수준 몫 %.0f%%" % (100 * (k ** 2).sum() / (x.res ** 2).sum()))
        print("== D 날 단위: 하루 평균(배지-실내) vs 하루 평균 입력  F13 / F47")
        dd = x.assign(gap=x.sub_temp - x.in_temp).groupby(["farm", "day"])[["gap", "in_temp"] + INP].mean().reset_index()
        hit = 0
        for c in ["in_temp"] + INP:
            v = rho2(dd, c, "gap")
            if not np.isnan(v).any() and np.sign(v[0]) == np.sign(v[1]) and min(map(abs, v)) >= .2:
                hit += 1; print("   %-12s %+.2f / %+.2f" % (c, *v))
        if not hit:
            print("   없음")


if __name__ == "__main__":
    main()
