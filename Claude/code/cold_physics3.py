# -*- coding: utf-8 -*-
"""Step 3 -- is the slow ("ground") state broken by the concatenation?

Every one of the 48 other greenhouses shows the reservoir effect: when the slow
air state sits above current air, the substrate is held above air too.  F13/F47
show it only weakly.  The physical reading is not that they have no floor, but
that their record switches source greenhouse at midnight, so a 72-hour average
of "their" air mixes several greenhouses.

Test: rebuild the slow state along SOURCE CHAINS instead of along the record.
Source is identified from hour-00 actuator values only (causal), following the
fingerprint idea already validated at 91-92% in the catalogue.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from common import load_raw

TARGET = ["F13", "F47"]
FP = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan",
      "act_co2", "act_fog", "in_co2", "in_hum"]


def day_fingerprint(p):
    """Hour-00 values per day -- known at the start of the day, so causal."""
    h0 = p[p.index % 24 == 0]
    fp = h0[FP].copy()
    fp.index = h0.index // 24
    return fp.dropna()


def assign_source(fp, k):
    X = StandardScaler().fit_transform(fp.values)
    km = KMeans(n_clusters=k, n_init=10, random_state=7).fit(X)
    return pd.Series(km.labels_, index=fp.index)


def chain_slow(p, src, halflife_days):
    """Slow air state accumulated along each source's own chain of days.

    For day d of source s, the state is an EWM over the DAILY MEAN air of that
    source's earlier days only -- strictly past information.
    """
    daily = p.groupby(p.index // 24).in_temp.mean()
    out = {}
    for s in src.unique():
        days = sorted(src.index[src == s])
        v = daily.reindex(days)
        ew = v.shift(1).ewm(halflife=halflife_days, ignore_na=True).mean()
        for dd, val in ew.items():
            out[dd] = val
    return pd.Series(out)


def build(g):
    g = g.sort_values("t")
    p = g.set_index("t").reindex(np.arange(g.t.min(), g.t.max() + 1))
    p["a3"] = p.in_temp.ewm(halflife=3, ignore_na=True).mean()
    p["a72"] = p.in_temp.ewm(halflife=72, ignore_na=True).mean()
    p["gap"] = p.sub_temp - p.a3
    p["day"] = p.index // 24
    return p


def partial(df, x, y="gap", ctrl=("a3",)):
    d = df[[x, y] + list(ctrl)].dropna()
    if len(d) < 200:
        return np.nan, 0
    A = np.ones((len(d), 1))
    for c in ctrl:
        v = d[c].values
        A = np.c_[A, v, v ** 2, v ** 3]
    f = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    rx, ry = f(d[x].values), f(d[y].values)
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return np.nan, len(d)
    return float(np.corrcoef(rx, ry)[0, 1]), len(d)


def main():
    tX, ty, sX = load_raw()
    d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")

    print("=== 기준: 다른 48곳의 저장고 효과 (기록 기반이 정상 작동) ===")
    ref = []
    for f in sorted(set(d.farm) - set(TARGET) - {"F32"}):
        p = build(d[d.farm == f].copy())
        p["res"] = p.a72 - p.a3
        r, _ = partial(p[(p.a3 >= 8) & (p.a3 < 14)], "res")
        if np.isfinite(r):
            ref.append(r)
    ref = np.array(ref)
    print("  중앙 %+.3f,  양수 %d/%d" % (np.median(ref), int((ref > 0).sum()), len(ref)))

    print("\n=== F13/F47: 기록 기반 vs 출처사슬 기반 저장고 ===")
    print("  %-5s %-16s %10s %10s %10s" % ("온실", "저장고 정의", "8~14℃", "야간", "전체"))
    for f in TARGET:
        p = build(d[d.farm == f].copy())
        fp = day_fingerprint(p)
        p["res_rec"] = p.a72 - p.a3
        rows = [("기록 기반 72h", "res_rec")]
        for k in (2, 3, 4, 6):
            src = assign_source(fp, k)
            for hd in (2, 5):
                cs = chain_slow(p, src, hd)
                col = "res_k%d_h%d" % (k, hd)
                p[col] = p.day.map(cs) - p.a3
                rows.append(("출처%d개·%d일평활" % (k, hd), col))
        for nm, col in rows:
            band = p[(p.a3 >= 8) & (p.a3 < 14)]
            night = p[(p.index % 24 <= 5) | (p.index % 24 >= 21)]
            r1, n1 = partial(band, col)
            r2, _ = partial(night, col)
            r3, _ = partial(p, col)
            print("  %-5s %-16s %10s %10s %10s"
                  % (f, nm, "%+.3f" % r1, "%+.3f" % r2, "%+.3f" % r3))

    print("\n=== 출처 군집이 실제로 운전 체계를 가르는가 (검증) ===")
    for f in TARGET:
        p = build(d[d.farm == f].copy())
        fp = day_fingerprint(p)
        src = assign_source(fp, 4)
        dm = p.groupby("day").agg(ec=("sub_ec", "mean") if "sub_ec" in p else ("in_temp", "mean"),
                                  heat=("act_heating", "mean"), gap=("gap", "mean"))
        dm["src"] = src
        dm = dm.dropna(subset=["src"])
        g = dm.groupby("src").agg(n=("gap", "size"), gap=("gap", "median"),
                                  heat=("heat", "median"))
        print("  [%s] 군집별 일수 / 평균 gap / 난방" % f)
        print("   ", g.round(2).to_dict("index"))
        # does the midnight EC/temperature jump shrink within a source?
        same = []
        diff = []
        for dd in sorted(src.index)[1:]:
            if dd - 1 not in src.index:
                continue
            j = abs(p.loc[p.day == dd, "sub_temp"].iloc[0]
                    - p.loc[p.day == dd - 1, "sub_temp"].iloc[-1]) \
                if (p.day == dd).any() and (p.day == dd - 1).any() else np.nan
            if np.isfinite(j):
                (same if src[dd] == src[dd - 1] else diff).append(j)
        print("    자정 |Δ배지| : 같은 군집 %.2f (n=%d) / 다른 군집 %.2f (n=%d)"
              % (np.median(same) if same else np.nan, len(same),
                 np.median(diff) if diff else np.nan, len(diff)))


if __name__ == "__main__":
    main()
