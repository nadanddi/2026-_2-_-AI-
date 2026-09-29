# -*- coding: utf-8 -*-
"""Is the day-level offset persistent from day to day, or day-to-day noise?

day_offset48 showed the offset is unexplained by same-day inputs everywhere.
That leaves two very different worlds:

  * offset is noise             -> autocorrelation ~0 in every greenhouse,
                                   nothing to recover, the limit is real;
  * offset persists across days -> continuous records could carry it forward,
                                   and F13/F47 lose it at each midnight source
                                   switch, which names exactly what the
                                   concatenation costs.

Catalogue reports 0.05-0.09 for F13/F47.  The question is what the 48
continuous records show.
"""
import numpy as np
import pandas as pd

from common import rmse
from day_offset48 import feats, block_oof, decompose
from common import load_raw


def main():
    tX, ty, sX = load_raw()
    d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    rows = []
    for f in sorted(d.farm.unique()):
        F = feats(d[d.farm == f].copy())
        if len(F) < 2000:
            continue
        cols = [c for c in F.columns if c not in ("sub_temp", "day")]
        oof = block_oof(F, cols)
        if np.isnan(oof).all():
            continue
        tot, lvl, wit, day_off, dd = decompose(F, oof)
        s = day_off.reindex(np.arange(day_off.index.min(), day_off.index.max() + 1))
        r = {"farm": f, "sd": float(s.std())}
        for L in (1, 2, 3, 7):
            r["ac%d" % L] = float(s.autocorr(L))
        # how much would knowing yesterday's offset help?
        j = pd.concat([s.rename("y"), s.shift(1).rename("x")], axis=1).dropna()
        r["gain1"] = (1 - np.mean((j.y - j.x) ** 2) / np.mean(j.y ** 2)) if len(j) > 30 else np.nan
        j2 = pd.concat([s.rename("y"), s.shift(2).rename("x")], axis=1).dropna()
        r["gain2"] = (1 - np.mean((j2.y - j2.x) ** 2) / np.mean(j2.y ** 2)) if len(j2) > 30 else np.nan
        rows.append(r)
        print("  %s  하루오프셋 std %.3f  자기상관 1일 %+.3f 2일 %+.3f 7일 %+.3f"
              % (f, r["sd"], r["ac1"], r["ac2"], r["ac7"]), flush=True)
    t = pd.DataFrame(rows).set_index("farm")
    t.to_csv("offset_persist.csv")

    tgt = [f for f in ("F13", "F47") if f in t.index]
    oth = [f for f in t.index if f not in tgt + ["F32"]]
    print("\n=== 하루 오프셋의 자기상관 ===")
    print("  %-12s %8s %8s %8s %8s" % ("", "1일", "2일", "3일", "7일"))
    for f in tgt + ["F32"]:
        if f in t.index:
            print("  %-12s %8.3f %8.3f %8.3f %8.3f"
                  % (f, t.loc[f, "ac1"], t.loc[f, "ac2"], t.loc[f, "ac3"], t.loc[f, "ac7"]))
    print("  %-12s %8.3f %8.3f %8.3f %8.3f  (48곳 중앙)"
          % ("다른 온실", t.loc[oth, "ac1"].median(), t.loc[oth, "ac2"].median(),
             t.loc[oth, "ac3"].median(), t.loc[oth, "ac7"].median()))
    print("  %-12s %8s %8s" % ("", "양수 곳수", "0.3 넘는 곳"))
    print("  %-12s %8d/%d %8d" % ("1일 자기상관", int((t.loc[oth, "ac1"] > 0).sum()),
                                  len(oth), int((t.loc[oth, "ac1"] > 0.3).sum())))

    print("\n=== 어제 오프셋을 그대로 쓰면 오차가 줄어드나 (1 - MSE비) ===")
    print("  양수 = 도움이 됨")
    for f in tgt:
        if f in t.index:
            print("  %-10s 어제 %+.3f   그저께 %+.3f" % (f, t.loc[f, "gain1"], t.loc[f, "gain2"]))
    g = t.loc[oth, "gain1"].dropna()
    print("  %-10s 어제 중앙 %+.3f,  양수 %d/%d,  최대 %+.3f"
          % ("다른 48곳", g.median(), int((g > 0).sum()), len(g), g.max()))
    print("\n  * 모두 0 근처 = 하루 오프셋은 날마다 새로 생기는 잡음")


if __name__ == "__main__":
    main()
