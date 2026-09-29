# -*- coding: utf-8 -*-
"""Is F32 unusable, or just offset?  Separate level from dynamics.

The catalogue rejected ONE way of using F32 (adding its rows to a member's
training set).  That test disagreed across validators: worse on DIAG10, better
on the cold extrapolation EXT10.  Since the evaluation period is colder than
training, this asks the prior question: does F32 carry cold rows we lack, and
does its substrate respond to air the same way once the level is removed?
"""
import numpy as np
import pandas as pd

from common import load_raw

FARMS = ["F13", "F47", "F32"]
BANDS = [(-99, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 99)]


def smooth(s, hl):
    return s.ewm(halflife=hl, ignore_na=True).mean()


def panel(df, farm):
    g = df[df.farm == farm].sort_values("t")
    full = np.arange(g.t.min(), g.t.max() + 1)
    p = g.set_index("t").reindex(full)
    p["in_s3"] = smooth(p.in_temp, 3)
    return p


def main():
    tX, ty, sX = load_raw()
    d = tX.merge(ty[["row_id", "sub_temp", "sub_ec"]], on="row_id", how="left")
    test = sX.copy()

    print("=== 1. 기본 제원 ===")
    for f in FARMS:
        g = d[d.farm == f]
        cols = [c for c in tX.columns if c not in ("row_id", "farm", "day", "hour", "t")]
        have = [c for c in cols if g[c].notna().any()]
        print("  %s  행 %5d  라벨 %5d  일차 %3d~%3d  입력 %2d개  라벨 소수비율 %.2f"
              % (f, len(g), int(g.sub_temp.notna().sum()), g.day.min(), g.day.max(),
                 len(have), 1 - np.isclose(g.sub_temp.dropna() % 1, 0).mean()))

    print("\n=== 2. 추운 행을 실제로 갖고 있는가 (라벨 있는 행 기준) ===")
    print("  %-22s %8s %8s %8s %8s" % ("", "<6℃", "6~8℃", "8~10℃", "10℃미만계"))
    rows = {}
    for f in FARMS:
        g = d[(d.farm == f) & d.sub_temp.notna()]
        rows[f] = g
        n = len(g)
        c = [(g.in_temp < 6).sum(), ((g.in_temp >= 6) & (g.in_temp < 8)).sum(),
             ((g.in_temp >= 8) & (g.in_temp < 10)).sum()]
        print("  %-22s %8d %8d %8d %8d" % (f + " 학습라벨", c[0], c[1], c[2], sum(c)))
    tt = test[test.farm.isin(["F13", "F47"])]
    c = [(tt.in_temp < 6).sum(), ((tt.in_temp >= 6) & (tt.in_temp < 8)).sum(),
         ((tt.in_temp >= 8) & (tt.in_temp < 10)).sum()]
    print("  %-22s %8d %8d %8d %8d  (평가, 라벨 없음)" % ("F13+F47 평가입력", c[0], c[1], c[2], sum(c)))

    print("\n=== 3. 배지 − 공기(3시간 평활) 온도대별 ===")
    print("  %-8s %s" % ("", "".join("%12s" % ("%g~%g" % (a, b) if b < 99 else "%g+" % a)
                                     for a, b in BANDS)))
    prof = {}
    for f in FARMS:
        p = panel(d, f)
        p = p[p.sub_temp.notna()]
        diff = p.sub_temp - p.in_s3
        row, prof[f] = "", []
        for a, b in BANDS:
            m = (p.in_s3 >= a) & (p.in_s3 < b)
            v = diff[m].mean() if m.sum() >= 5 else np.nan
            prof[f].append(v)
            row += "%12s" % ("%+.2f(%d)" % (v, m.sum()) if m.sum() >= 5 else "-")
        print("  %-8s %s" % (f, row))

    print("\n  * 각 온실 평균을 뺀 '모양'만 비교 (수준 차이 제거 후에도 다른가)")
    for f in FARMS:
        v = np.array(prof[f], float)
        c = v - np.nanmean(v)
        print("  %-8s %s" % (f, "".join("%12s" % ("%+.2f" % x if np.isfinite(x) else "-") for x in c)))

    print("\n=== 4. 동역학: 배지가 공기를 얼마나/언제 따라가나 ===")
    print("  %-6s %10s %10s %10s %10s" % ("", "최적지연", "그때상관", "기울기", "야간기울기"))
    for f in FARMS:
        p = panel(d, f)
        p = p[p.sub_temp.notna()]
        best = max(range(0, 9), key=lambda L: p.sub_temp.corr(p.in_temp.shift(L)))
        r = p.sub_temp.corr(p.in_temp.shift(best))
        z = p.dropna(subset=["in_s3", "sub_temp"])
        k = np.polyfit(z.in_s3, z.sub_temp, 1)[0]
        night = z[(z.index % 24 <= 5) | (z.index % 24 >= 21)]
        kn = np.polyfit(night.in_s3, night.sub_temp, 1)[0] if len(night) > 50 else np.nan
        print("  %-6s %9dh %10.3f %10.3f %10.3f" % (f, best, r, k, kn))

    print("\n=== 5. F32의 고유 컬럼이 F13/F47에 없는 것 ===")
    cols = [c for c in tX.columns if c not in ("row_id", "farm", "day", "hour", "t")]
    only32 = [c for c in cols if d[d.farm == "F32"][c].notna().any()
              and not d[d.farm.isin(["F13", "F47"])][c].notna().any()]
    shared = [c for c in cols if d[d.farm == "F32"][c].notna().any()
              and d[d.farm.isin(["F13", "F47"])][c].notna().any()]
    print("  F32에만: %s" % ", ".join(only32))
    print("  공유(%d개): %s" % (len(shared), ", ".join(shared)))

    print("\n=== 6. F32는 이어붙임이 있는가 (자정 점프) ===")
    for f in FARMS:
        p = panel(d, f)
        dd = p.sub_temp.diff().abs()
        mid = dd[p.index % 24 == 0].mean()
        oth = dd[p.index % 24 != 0].mean()
        print("  %-6s 자정 |Δ| %.3f / 그 외 %.3f = %.2f배" % (f, mid, oth, mid / oth))


if __name__ == "__main__":
    main()
