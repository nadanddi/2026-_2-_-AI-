# -*- coding: utf-8 -*-
"""TD10: heating switch-on events - does outside cooling come first, does the substrate lag the air, and does it
warm slowly?  (2026-10-10 집 클로드, user hypothesis: "외부가 먼저 추워져 배지를 낮추면 난방이 돌고, 배지가 오를 때까지
시간이 걸리며 천천히 오를 것")  Descriptive, F13/F47 train (inputs + sub_temp labels).  Fixed before running:
 Event = act_heating goes 0 -> >0 (previous hour 0, current > 0) within the same record day.
 Window -4..+8 h, kept only inside the same day (records are concatenated by dong at midnight, 1.3) -> report n per lag.
 (a) mean profile of out_temp, in_temp, sub_temp, sub-in, act_heating relative to the event hour (each series minus
     its value at -1 h).
 (b) which drops first before the event: hour of minimum slope in -4..0 for out/in/sub.
 (c) after the event: hours until in_temp and sub_temp first exceed their -1 h value (median, IQR).
 (d) 'follow rate' a from  sub(t+1)-sub(t) = a*(in(t)-sub(t)) + b  (OLS) for heating-on rows vs heating-off rows,
     cold-side (in_temp<14) only; a small = slow follow.  Time constant ~ 1/a hours.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u td10_heating_event_lag_v1.py
"""
import env  # noqa: F401
import numpy as np, pandas as pd
import common


def main():
    tX, ty, _ = common.load_raw()
    a = tX.merge(ty[["row_id", "sub_temp"]], on="row_id")
    a = a[a.farm.isin(["F13", "F47"])].sort_values(["farm", "t"]).reset_index(drop=True)
    a["gap"] = a.sub_temp - a.in_temp
    for f in ("F13", "F47", "both"):
        x = a if f == "both" else a[a.farm == f]
        x = x.reset_index(drop=True)
        same = x.day.eq(x.day.shift(1)) & x.farm.eq(x.farm.shift(1))
        ev = x.index[(x.act_heating > 0) & (x.act_heating.shift(1) == 0) & same]
        print("\n######## %s: 난방 켜짐 %d회, 켜진 시각 분포(중앙 %d시, 사분위 %d~%d시)" % (f, len(ev), np.median(x.hour[ev]), *np.percentile(x.hour[ev], [25, 75])))
        rows = []
        for i in ev:
            if i - 1 < 0:
                continue
            base = x.loc[i - 1]
            for k in range(-4, 9):
                j = i + k
                if 0 <= j < len(x) and x.day[j] == x.day[i] and x.farm[j] == x.farm[i]:
                    rows.append(dict(ev=i, k=k, out=x.out_temp[j] - base.out_temp, inn=x.in_temp[j] - base.in_temp, sub=x.sub_temp[j] - base.sub_temp,
                                     gap=x.gap[j], heat=x.act_heating[j]))
        P = pd.DataFrame(rows)
        T = P.groupby("k").agg(n=("ev", "size"), 외기=("out", "mean"), 실내=("inn", "mean"), 배지=("sub", "mean"), 배지_실내=("gap", "mean"), 난방=("heat", "mean"))
        print("(a) 난방 켜진 시각(0) 기준, -1시 대비 변화(℃)"); print(T.round(2).to_string())
        sl = T[["외기", "실내", "배지"]].diff().loc[-3:0]
        print("(b) 켜지기 전 가장 빨리 떨어진 시각:", {c: int(sl[c].idxmin()) for c in sl}, " 켜지기 직전 1시간 변화:", sl.loc[0].round(2).to_dict())
        up = {"실내": [], "배지": []}
        for e, g in P.groupby("ev"):
            g = g.set_index("k")
            for c, cc in (("실내", "inn"), ("배지", "sub")):
                pos = [k for k in range(0, 9) if k in g.index and g.loc[k, cc] > 0]
                up[c].append(pos[0] if pos else np.nan)
        for c in up:
            v = np.array(up[c], float)
            print("(c) %s이 켜지기 전 값을 넘을 때까지: 중앙 %.0f시간 (사분위 %.0f~%.0f), 8시간 안에 못 넘음 %.0f%%" % (c, np.nanmedian(v), *np.nanpercentile(v, [25, 75]), 100 * np.isnan(v).mean()))
        y = (x.sub_temp.shift(-1) - x.sub_temp)[same.shift(-1, fill_value=False)]
        d = (x.in_temp - x.sub_temp)[y.index]; h = x.act_heating[y.index] > 0; c = x.in_temp[y.index] < 14
        for nm, m in (("난방 켜짐", h & c), ("난방 꺼짐", ~h & c)):
            z = pd.DataFrame({"y": y[m], "d": d[m]}).dropna()
            A = np.polyfit(z.d, z.y, 1)
            print("(d) %s(실내<14℃, %d행): 배지가 공기 쪽으로 따라가는 비율 a=%.3f/시간 → 시정수 약 %.1f시간, 절편 %+.3f" % (nm, len(z), A[0], 1 / A[0] if A[0] > 0 else np.nan, A[1]))


if __name__ == "__main__":
    main()
