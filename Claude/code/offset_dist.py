# -*- coding: utf-8 -*-
"""How does substrate-minus-air behave across all 51 greenhouses, by temperature?

Checks whether "other greenhouses keep the substrate warmer when it is cold"
is universal or merely typical, and where F13/F47 actually sit -- noting that
catalogue 2.5b withdrew an earlier claim about their cold-band behaviour for
resting on 5 rows.
"""
import numpy as np
import pandas as pd
from common import load_raw

BANDS = [("<=6", -99, 6), ("6~8", 6, 8), ("8~10", 8, 10),
         ("10~15", 10, 15), (">15", 15, 99)]


def main():
    tX, ty, sX = load_raw()
    d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    rows = []
    for f, g in d.groupby("farm"):
        g = g.sort_values("t")
        p = g.set_index("t").reindex(np.arange(g.t.min(), g.t.max() + 1))
        p["air"] = p.in_temp.ewm(halflife=3, ignore_na=True).mean()
        p = p.dropna(subset=["sub_temp", "air"])
        diff = p.sub_temp - p.air
        r = {"farm": f}
        for nm, a, b in BANDS:
            m = (p.air >= a) & (p.air < b)
            r[nm] = diff[m].median() if m.sum() >= 5 else np.nan
            r["n" + nm] = int(m.sum())
        rows.append(r)
    t = pd.DataFrame(rows).set_index("farm")

    tgt = ["F13", "F47"]
    oth = [f for f in t.index if f not in tgt + ["F32"]]

    print("=== 배지 − 공기(3시간 평활) 중앙값, 온도대별 ===")
    print("  %-6s %s" % ("", "".join("%10s" % nm for nm, _, _ in BANDS)))
    for f in tgt + ["F32"]:
        print("  %-6s %s" % (f, "".join(
            "%10s" % ("%+.2f(%d)" % (t.loc[f, nm], t.loc[f, "n" + nm])
                      if np.isfinite(t.loc[f, nm]) else "-") for nm, _, _ in BANDS)))
    print("  %-6s %s" % ("48곳중앙", "".join("%10.2f" % t.loc[oth, nm].median() for nm, _, _ in BANDS)))

    print("\n=== 48개 온실의 분포 (온도대별) ===")
    print("  %-8s %8s %8s %8s %8s %10s %12s" %
          ("온도대", "최소", "25%", "중앙", "75%", "최대", "음수인 곳"))
    for nm, _, _ in BANDS:
        s = t.loc[oth, nm].dropna()
        print("  %-8s %8.2f %8.2f %8.2f %8.2f %10.2f %9d/%d"
              % (nm, s.min(), s.quantile(.25), s.median(), s.quantile(.75),
                 s.max(), int((s < 0).sum()), len(s)))

    print("\n=== 6℃ 이하 구간에서 F13/F47과 같은 방향(음수)인 온실 ===")
    s = t.loc[oth, "<=6"].dropna().sort_values()
    neg = s[s < 0]
    print("  음수 %d곳: %s" % (len(neg), ", ".join("%s %+.2f" % (f, v) for f, v in neg.items())))
    print("  가장 낮은 5곳:", ", ".join("%s %+.2f" % (f, v) for f, v in s.head(5).items()))
    print("  F13 %+.2f (%d행), F47 %+.2f (%d행)"
          % (t.loc["F13", "<=6"], t.loc["F13", "n<=6"],
             t.loc["F47", "<=6"], t.loc["F47", "n<=6"])
          if np.isfinite(t.loc["F47", "<=6"]) else
          "  F13 %+.2f (%d행), F47 표본부족(%d행)"
          % (t.loc["F13", "<=6"], t.loc["F13", "n<=6"], t.loc["F47", "n<=6"]))

    print("\n=== F13/F47 추운 구간 근거의 빈약함 (카탈로그 2.5b 관련) ===")
    for f in tgt:
        print("  %s: <=6℃ %d행, 6~8℃ %d행  ← 이 표본으로 결론 내기 어려움"
              % (f, t.loc[f, "n<=6"], t.loc[f, "n6~8"]))
    print("  참고: 48곳 중앙값은 <=6℃에서 %d행" % int(t.loc[oth, "n<=6"].median()))

    print("\n=== 온도가 내려갈 때 방향 (10~15℃ 대비 <=6℃ 변화) ===")
    delta = (t["<=6"] - t["10~15"]).dropna()
    o = delta[[f for f in delta.index if f in oth]]
    print("  48곳: 중앙 %+.2f, 양수(추울수록 따뜻해짐) %d/%d"
          % (o.median(), int((o > 0).sum()), len(o)))
    for f in tgt + ["F32"]:
        if f in delta.index:
            print("  %s: %+.2f" % (f, delta[f]))
    t.to_csv("offset_dist.csv")


if __name__ == "__main__":
    main()
