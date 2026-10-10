# -*- coding: utf-8 -*-
"""TD26 (step 0-2): pick 'similar greenhouse' TESTBEDS for cold extrapolation, using TRAINING data only
(2026-10-10 집 클로드, user: "0-1, 0-2단계 먼저 진행해봐").  F13/F47 have only 20 training rows with in_temp < 6, so cold
extrapolation cannot be measured on them; other greenhouses with many cold rows and F13/F47-like substrate physics can
serve as a test bench.  Selection uses labels+inputs of the public training data only (no test_X statistics).
Per greenhouse (training rows): band means of (sub - in_temp) for in_temp bands 6-8, 8-10, 10-12, 12-15, >=15;
  best lag (0..6 h) of corr(sub, in_temp shifted); daily amplitude ratio (sub range / in_temp range, median);
  cold slope = OLS slope of (sub - in) on in_temp for rows in_temp < 12; rows < 6 and < 4; midnight jump ratio.
Similarity to the F13/F47 average (fixed before running):
  S = RMS over bands 8-10, 10-12, 12-15, >=15 of (band mean - F13/F47 band mean)
      + 0.3 * |lag - 3| + 2 * |amp ratio - F13/F47 amp| + 1 * |cold slope - F13/F47 cold slope|
TESTBED = S <= 1.0 and >= 200 training rows with in_temp < 6.  Report the full ranking (top 15) and the testbeds.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u td26_similar_farm_testbed_v1.py
"""
import env  # noqa: F401
import numpy as np, pandas as pd
import common
BANDS = [(6, 8), (8, 10), (10, 12), (12, 15), (15, 99)]


def profile(x):
    x = x.sort_values("t"); d = (x.sub_temp - x.in_temp)
    out = {}
    for lo, hi in BANDS:
        m = (x.in_temp >= lo) & (x.in_temp < hi)
        out["b%d" % lo] = d[m].mean() if m.sum() >= 30 else np.nan
    cont = x.t.diff() == 1
    best, bl = -2, 0
    for L in range(0, 7):
        s = x.in_temp.shift(L).where((x.t - x.t.shift(L)) == L)
        m = s.notna() & x.sub_temp.notna()
        c = np.corrcoef(x.sub_temp[m], s[m])[0, 1] if m.sum() > 100 else -2
        if c > best:
            best, bl = c, L
    out["lag"] = bl
    dd = x.groupby("day").agg(sr=("sub_temp", lambda v: v.max() - v.min()), ir=("in_temp", lambda v: v.max() - v.min()))
    out["amp"] = (dd.sr / dd.ir.where(dd.ir > 2)).median()
    c = x[x.in_temp < 12].dropna(subset=["sub_temp", "in_temp"])
    out["cold_slope"] = np.polyfit(c.in_temp, c.sub_temp - c.in_temp, 1)[0] if len(c) > 100 else np.nan
    out["n_lt6"] = int((x.in_temp < 6).sum()); out["n_lt4"] = int((x.in_temp < 4).sum())
    mj = x.sub_temp.diff()[cont]; h = x.hour[cont]
    out["mid_jump"] = mj[h == 0].abs().mean() / mj[h != 0].abs().mean()
    out["rows"] = len(x)
    return pd.Series(out)


def main():
    tX, ty, _ = common.load_raw()
    a = tX.merge(ty[["row_id", "sub_temp"]], on="row_id").dropna(subset=["sub_temp", "in_temp"])
    P = pd.DataFrame([profile(x).rename(f) for f, x in a.groupby("farm")])
    ref = P.loc[["F13", "F47"]].mean()
    print("F13/F47 기준:", ref.round(2).to_dict())
    bd = ["b8", "b10", "b12", "b15"]
    P["S"] = np.sqrt(((P[bd] - ref[bd]) ** 2).mean(1)) + .3 * (P.lag - 3).abs() + 2 * (P.amp - ref.amp).abs() + (P.cold_slope - ref.cold_slope).abs()
    P["testbed"] = (P.S <= 1.0) & (P.n_lt6 >= 200) & ~P.index.isin(["F13", "F47"])
    pd.set_option("display.width", 220)
    cols = ["S", "b6", "b8", "b10", "b12", "b15", "lag", "amp", "cold_slope", "n_lt6", "n_lt4", "mid_jump", "rows", "testbed"]
    print("\n닮은 정도 순위 (S 작을수록 F13·F47과 비슷; 상위 15 + F13·F47·F32)")
    show = pd.concat([P.loc[["F13", "F47", "F32"]], P.drop(["F13", "F47", "F32"]).sort_values("S").head(15)])
    print(show[cols].round(2).to_string())
    print("\n시험대(S≤1.0, 6℃ 미만 행≥200): %s | 6℃ 미만 행 합계 %d, 4℃ 미만 %d" % (
        list(P[P.testbed].index), int(P[P.testbed].n_lt6.sum()), int(P[P.testbed].n_lt4.sum())))
    print("참고: S≤1.5로 넓히면 %s" % list(P[(P.S <= 1.5) & (P.n_lt6 >= 200) & ~P.index.isin(["F13", "F47"])].index))
    P.to_csv(env.LOCAL + "/td26_farm_profiles_v1.csv")


if __name__ == "__main__":
    main()
