# -*- coding: utf-8 -*-
"""Step 1 -- what physically changes about the substrate as it gets cold?

Hypotheses to separate, all about substrate temperature T_s versus air T_a:

  H1 GROUND    a slab sits on a floor whose temperature is a slow integrator of
               past air.  During a cold snap the floor is still warm, so T_s is
               held ABOVE T_a, and the more so the faster/newer the snap.
               -> gap should grow with (slow air state - current air).
  H2 TAU       the effective time constant lengthens in the cold (heating
               cycles, lower ventilation), so a fixed halflife under-fits.
  H3 HEATING   heated air warms the canopy faster than the slab, so running
               heat widens the gap NEGATIVELY (T_s below T_a).
  H4 RATE      "cold" is confounded with "falling fast"; the gap may be a
               function of dT_a/dt rather than of the level.

Measured per greenhouse so that a mechanism has to hold in many of them, and
separately inside F13/F47's own measurable range (8-15C, thousands of rows).
"""
import numpy as np
import pandas as pd

from common import load_raw

TARGET = ["F13", "F47"]


def build(g):
    g = g.sort_values("t")
    p = g.set_index("t").reindex(np.arange(g.t.min(), g.t.max() + 1))
    a = p.in_temp
    p["a1"] = a.ewm(halflife=1, ignore_na=True).mean()
    p["a3"] = a.ewm(halflife=3, ignore_na=True).mean()
    p["a12"] = a.ewm(halflife=12, ignore_na=True).mean()
    p["a72"] = a.ewm(halflife=72, ignore_na=True).mean()     # ground proxy, ~3 days
    p["a168"] = a.ewm(halflife=168, ignore_na=True).mean()   # ~1 week
    p["gap"] = p.sub_temp - p.a3
    p["drop6"] = p.a3 - a.shift(6)                # negative when falling
    p["drop24"] = p.a3 - a.shift(24)
    p["reservoir"] = p.a72 - p.a3                 # H1: how much warmer the slow state is
    p["res168"] = p.a168 - p.a3
    p["night"] = ((p.index % 24 <= 5) | (p.index % 24 >= 21)).astype(int)
    return p.dropna(subset=["sub_temp", "a3"])


def partial(df, x, y="gap", ctrl=("a3",)):
    """Correlation of x with y after removing a cubic in the control variables."""
    d = df[[x, y] + list(ctrl)].dropna()
    if len(d) < 200:
        return np.nan, 0
    A = np.c_[np.ones(len(d))]
    for c in ctrl:
        v = d[c].values
        A = np.c_[A, v, v ** 2, v ** 3]
    def resid(v):
        w, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ w
    rx, ry = resid(d[x].values), resid(d[y].values)
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return np.nan, len(d)
    return float(np.corrcoef(rx, ry)[0, 1]), len(d)


def best_halflife(df):
    """Which EWM halflife of air best explains substrate, in this subset?"""
    best, bh = -9, None
    for hl in (1, 2, 3, 4, 6, 8, 12, 24):
        col = "a%d" % hl if "a%d" % hl in df else None
        v = df.in_temp.ewm(halflife=hl, ignore_na=True).mean() if col is None else df[col]
        d = pd.concat([df.sub_temp, v], axis=1).dropna()
        if len(d) < 100:
            continue
        r = d.iloc[:, 0].corr(d.iloc[:, 1])
        if r > best:
            best, bh = r, hl
    return bh, best


def main():
    tX, ty, sX = load_raw()
    d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    P = {f: build(g.copy()) for f, g in d.groupby("farm")}

    print("=== A. F13/F47 측정 가능 구간에서 각 가설의 부분상관 (a3 통제) ===")
    print("  gap = 배지 − 공기(3h평활).  양수 = 배지가 공기보다 따뜻함\n")
    print("  %-6s %-10s %10s %10s %10s %10s" %
          ("온실", "구간", "H1 저장고", "H3 난방", "H4 6h낙폭", "H4 24h낙폭"))
    for f in TARGET:
        p = P[f]
        for nm, m in [("8~12℃", (p.a3 >= 8) & (p.a3 < 12)),
                      ("12~18℃", (p.a3 >= 12) & (p.a3 < 18)),
                      ("야간 전체", p.night == 1)]:
            s = p[m]
            r1, n = partial(s, "reservoir")
            r3, _ = partial(s, "act_heating")
            r4, _ = partial(s, "drop6")
            r5, _ = partial(s, "drop24")
            print("  %-6s %-10s %10s %10s %10s %10s  (n=%d)"
                  % (f, nm, "%+.3f" % r1, "%+.3f" % r3, "%+.3f" % r4, "%+.3f" % r5, n))

    print("\n=== B. 같은 관계가 다른 온실에서도 성립하나 (48곳, 8~12℃) ===")
    others = [f for f in P if f not in TARGET + ["F32"]]
    res = {k: [] for k in ("reservoir", "drop6", "drop24")}
    for f in others:
        p = P[f]
        s = p[(p.a3 >= 8) & (p.a3 < 12)]
        for k in res:
            r, n = partial(s, k)
            if np.isfinite(r):
                res[k].append(r)
    print("  %-12s %8s %8s %8s %12s" % ("가설", "중앙", "25%", "75%", "같은부호"))
    for k, v in res.items():
        v = np.array(v)
        same = max((v > 0).sum(), (v < 0).sum())
        print("  %-12s %8.3f %8.3f %8.3f %8d/%d"
              % (k, np.median(v), np.percentile(v, 25), np.percentile(v, 75), same, len(v)))

    print("\n=== C. H2: 최적 시정수가 추울수록 길어지나 ===")
    print("  %-6s %12s %12s %12s" % ("온실", "8℃미만", "8~12℃", "12℃이상"))
    rows = []
    for f in TARGET + others[:12]:
        p = P[f]
        out = []
        for a, b in [(-99, 8), (8, 12), (12, 99)]:
            s = p[(p.a3 >= a) & (p.a3 < b)]
            hl, r = best_halflife(s) if len(s) > 200 else (None, None)
            out.append("%s" % (("%dh (r%.2f)" % (hl, r)) if hl else "-"))
        print("  %-6s %12s %12s %12s" % (f, *out))
        rows.append((f, out))

    print("\n=== D. 추운 구간에서 실제로 무엇이 다른가 (온실 평균, 8~12℃ vs 12~18℃) ===")
    print("  %-6s %10s %10s %10s %10s" % ("온실", "gap 8~12", "gap 12~18", "난방 8~12", "저장고 8~12"))
    for f in TARGET + ["F32"] + others[:8]:
        p = P[f]
        c = p[(p.a3 >= 8) & (p.a3 < 12)]
        w = p[(p.a3 >= 12) & (p.a3 < 18)]
        print("  %-6s %10.2f %10.2f %10s %10.2f"
              % (f, c.gap.median(), w.gap.median(),
                 "%.0f" % c.act_heating.median() if c.act_heating.notna().any() else "-",
                 c.reservoir.median()))


if __name__ == "__main__":
    main()
