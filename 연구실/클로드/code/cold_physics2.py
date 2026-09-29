# -*- coding: utf-8 -*-
"""Step 2 -- is the cold problem really a HEATING problem?

If the substrate lags air mainly when the heater is driving the air, then the
evaluation set is hard not because it visits unseen temperatures but because it
visits heating levels that training also contains.  That converts an
extrapolation problem into an interpolation one, which is testable.

Checks:
  1. shape of the heating -> gap relationship, per temperature band
  2. whether evaluation's heating x temperature cells exist in training
  3. whether the relationship is about heating or merely about temperature
  4. whether the current model already accounts for it (residual vs heating)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import load_raw, split_mask, rmse
from geometry_cv import geometry_folds
from make_submission import T_HUB
import features_v2 as F2

TARGET = ["F13", "F47"]
HB = [(0, 1), (1, 25), (25, 60), (60, 90), (90, 101)]
TB = [("<8", -99, 8), ("8~10", 8, 10), ("10~12", 10, 12), ("12~15", 12, 15), (">15", 15, 99)]


def build(g):
    g = g.sort_values("t")
    p = g.set_index("t").reindex(np.arange(g.t.min(), g.t.max() + 1))
    p["a3"] = p.in_temp.ewm(halflife=3, ignore_na=True).mean()
    if "sub_temp" not in p:          # test_X has inputs only
        p["sub_temp"] = np.nan
    p["gap"] = p.sub_temp - p.a3
    p["night"] = ((p.index % 24 <= 5) | (p.index % 24 >= 21)).astype(int)
    return p


def main():
    tX, ty, sX = load_raw()
    d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    P = {f: build(d[d.farm == f].copy()) for f in TARGET}
    T = {f: build(sX[sX.farm == f].copy()) for f in TARGET}

    print("=== 1. 난방 세기별 배지−공기 격차 (학습, 정답 있는 행) ===")
    print("  %-5s %-8s %s" % ("온실", "온도대", "".join("%14s" % ("난방 %d~%d" % h) for h in HB)))
    for f in TARGET:
        p = P[f].dropna(subset=["gap", "act_heating"])
        for nm, a, b in TB:
            m = (p.a3 >= a) & (p.a3 < b)
            row = ""
            for lo, hi in HB:
                s = p[m & (p.act_heating >= lo) & (p.act_heating < hi)]
                row += "%14s" % ("%+.2f(%d)" % (s.gap.median(), len(s)) if len(s) >= 20 else "-")
            print("  %-5s %-8s %s" % (f, nm, row))

    print("\n=== 2. 평가의 (온도 x 난방) 조합이 학습에 있나 ===")
    print("  %-5s %-8s %s" % ("온실", "온도대", "".join("%14s" % ("난방 %d~%d" % h) for h in HB)))
    for f in TARGET:
        tr = P[f].dropna(subset=["gap"])
        te = T[f]
        for nm, a, b in TB:
            row = ""
            for lo, hi in HB:
                nte = int(((te.a3 >= a) & (te.a3 < b) & (te.act_heating >= lo)
                           & (te.act_heating < hi)).sum())
                ntr = int(((tr.a3 >= a) & (tr.a3 < b) & (tr.act_heating >= lo)
                           & (tr.act_heating < hi)).sum())
                row += "%14s" % ("평%d/학%d" % (nte, ntr) if nte else "·")
            print("  %-5s %-8s %s" % (f, nm, row))

    print("\n=== 3. 난방 효과인가 온도 효과인가 (같은 온도대 안에서 난방만 변할 때) ===")
    for f in TARGET:
        p = P[f].dropna(subset=["gap", "act_heating"])
        print("  [%s]" % f)
        for nm, a, b in TB[1:4]:
            s = p[(p.a3 >= a) & (p.a3 < b)]
            if len(s) < 200:
                continue
            lo = s[s.act_heating < 1]
            hi = s[s.act_heating >= 60]
            if len(lo) >= 30 and len(hi) >= 30:
                print("    %-7s 난방 0%%: gap %+.2f (n=%d) vs 난방 60%%+: gap %+.2f (n=%d)  차이 %+.2f"
                      % (nm, lo.gap.median(), len(lo), hi.gap.median(), len(hi),
                         hi.gap.median() - lo.gap.median()))

    print("\n=== 4. 현재 모델이 이미 반영하고 있나 (블록 CV 잔차 vs 난방) ===")
    panel = F2.build(tX, sX).merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    panel["is_test"] = panel.row_id.isin(set(sX.row_id))
    v = F2.view(panel, "sub_temp")
    lab = panel[(~panel.is_test) & panel.sub_temp.notna()].reset_index(drop=True)
    oof = np.full(len(lab), np.nan)
    for fd in geometry_folds():
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        m = lgb.LGBMRegressor(random_state=7, n_jobs=4, verbose=-1, **T_HUB).fit(tr[v], tr.sub_temp)
        oof[np.where(vam)[0]] = m.predict(va[v])
    got = ~np.isnan(oof)
    lab = lab[got].copy()
    lab["resid"] = oof[got] - lab.sub_temp.values
    print("  %-14s %10s %10s %10s" % ("난방 구간", "행 수", "평균 잔차", "RMSE"))
    for lo, hi in HB:
        s = lab[(lab.act_heating >= lo) & (lab.act_heating < hi)]
        if len(s) >= 50:
            print("  %-14s %10d %+10.3f %10.3f"
                  % ("%d~%d" % (lo, hi), len(s), s.resid.mean(),
                     float(np.sqrt((s.resid ** 2).mean()))))
    print("\n  * 평균 잔차가 난방에 따라 기울면 = 모델이 난방 효과를 덜 반영")
    r = np.corrcoef(lab.act_heating.fillna(0), lab.resid)[0, 1]
    print("  잔차와 난방의 상관: %+.3f" % r)
    night = lab[lab.hour.isin(list(range(0, 6)) + list(range(21, 24)))]
    print("  야간만: %+.3f" % np.corrcoef(night.act_heating.fillna(0), night.resid)[0, 1])


if __name__ == "__main__":
    main()
