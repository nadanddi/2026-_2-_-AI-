# -*- coding: utf-8 -*-
"""How much of sub_temp is a first-order thermal response to the air?

The slab is physically a low-pass filter on in_temp (plus radiation and pipe
heat).  features_v2 only exposes fixed half-lives 2/6/24 h.  Here the time
constant is scanned, and small linear models on tuned filters are scored on
the same geometry folds as everything else.  If a two- or three-parameter
physical model lands near the leaderboard leader (0.4985), the gap is
structure, not tuning.

Run:  cd research && PYTHONPATH="" <python> -u struct_temp.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

import common
from common import TARGET_FARMS
from harness import load, folds
from common import split_mask, rmse


def series(tX, sX):
    cols = ["row_id", "farm", "t", "in_temp", "out_temp", "out_rad",
            "act_shade", "act_thermal", "act_heating", "in_hum"]
    a = pd.concat([tX[cols], sX[cols]], ignore_index=True)
    a = a[a.farm.isin(TARGET_FARMS)]
    out = {}
    for f, g in a.groupby("farm"):
        g = g.sort_values("t").set_index("t")
        g = g.reindex(np.arange(g.index.min(), g.index.max() + 1))
        g["farm"] = f
        out[f] = g
    return out


def ewm(s, hl):
    return s.ewm(halflife=hl, ignore_na=True).mean()


def build(S, spec):
    """spec: list of (column, halflife or None)."""
    parts = []
    for f, g in S.items():
        d = pd.DataFrame({"row_id": g.row_id.values})
        rad = g.out_rad * (g.act_shade.fillna(100) / 100) * (g.act_thermal.fillna(100) / 100)
        base = {"in_temp": g.in_temp, "out_temp": g.out_temp, "rad": rad,
                "heat": g.act_heating}
        for c, hl in spec:
            s = base[c]
            d["%s_%s" % (c, hl)] = (s if hl is None else ewm(s, hl)).values
        d["farm_id"] = float(f == "F47")
        parts.append(d.dropna(subset=["row_id"]))
    return pd.concat(parts, ignore_index=True)


def cv(lab, X, cols, model, kind):
    Z = lab[["row_id", "farm", "day", "sub_temp"]].merge(X, on="row_id", how="left")
    y = Z.sub_temp.values
    oof = np.full(len(Z), np.nan)
    for fd in folds(kind):
        tr, va = split_mask(Z, fd)
        A = Z.loc[tr, cols].fillna(Z.loc[tr, cols].median())
        B = Z.loc[va, cols].fillna(Z.loc[tr, cols].median())
        m = model().fit(A.values, y[tr])
        oof[va] = m.predict(B.values)
    g = ~np.isnan(oof)
    return rmse(oof[g], y[g])


def main():
    panel, lab_t, lab_e = load()
    tX, ty, sX = common.load_raw()
    S = series(tX, sX)
    OLS = LinearRegression

    print("=== 1. in_temp 단일 1차 필터, 반감기 스캔 (OLS, 온실 더미 포함) ===")
    best = None
    for hl in [None, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12]:
        X = build(S, [("in_temp", hl)])
        cols = [c for c in X.columns if c != "row_id"]
        ra, rb = cv(lab_t, X, cols, OLS, "A"), cv(lab_t, X, cols, OLS, "B")
        tag = "원값" if hl is None else "hl=%s" % hl
        print("  %-8s A %.4f  B %.4f" % (tag, ra, rb))
        if best is None or ra + rb < best[1]:
            best = (hl, ra + rb)
    print("  -> 최적 반감기 %s" % best[0])

    print("\n=== 2. 다중 시정수 + 일사 + 난방 (선형) ===")
    specs = {
        "in_temp 원값+1+3+8": [("in_temp", None), ("in_temp", 1), ("in_temp", 3), ("in_temp", 8)],
        "+ rad 원값/2/6": [("in_temp", None), ("in_temp", 1), ("in_temp", 3), ("in_temp", 8),
                           ("rad", None), ("rad", 2), ("rad", 6)],
        "+ heat 1/4 + out 6": [("in_temp", None), ("in_temp", 1), ("in_temp", 3), ("in_temp", 8),
                               ("rad", None), ("rad", 2), ("rad", 6),
                               ("heat", 1), ("heat", 4), ("out_temp", 6)],
        "+ 24/72h 느린 성분": [("in_temp", None), ("in_temp", 1), ("in_temp", 3), ("in_temp", 8),
                               ("in_temp", 24), ("in_temp", 72),
                               ("rad", None), ("rad", 2), ("rad", 6),
                               ("heat", 1), ("heat", 4), ("out_temp", 6), ("out_temp", 48)],
    }
    for nm, sp in specs.items():
        X = build(S, sp)
        cols = [c for c in X.columns if c != "row_id"]
        ra, rb = cv(lab_t, X, cols, OLS, "A"), cv(lab_t, X, cols, OLS, "B")
        print("  %-22s (%2d열) A %.4f  B %.4f" % (nm, len(cols), ra, rb))

    print("\n참고: 현재 제출 구성 교차검증 A 0.8212 / B 0.7671, 실제 0.6666")


if __name__ == "__main__":
    main()
