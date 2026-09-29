# -*- coding: utf-8 -*-
"""Step 5 -- the cold hypothesis needs a cold validation set.

The geometry CV places its folds in days 70-180, which are warm: it holds fewer
than 20 rows below 8 degC, so step 4 could not test the cold claim at all.
Here the coldest days are removed from training wholesale and predicted, which
is the situation the evaluation set actually poses.  Several hold-out sizes are
used so the answer does not hinge on one threshold.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import load_raw, rmse, TARGET_FARMS
from make_submission import T_HUB, SEEDS
from cold_physics4 import source_features
import features_v2 as F2


def cold_days(panel, n_days):
    """The n coldest (farm, day) pairs by daily minimum smoothed air."""
    lab = panel[(~panel.is_test) & panel.sub_temp.notna()]
    dm = lab.groupby(["farm", "day"]).in_temp_ewm6.min().sort_values()
    return set(dm.index[:n_days])


def run(panel, cols, held, seeds=SEEDS):
    lab = panel[(~panel.is_test) & panel.sub_temp.notna()].copy()
    key = list(zip(lab.farm, lab.day))
    is_va = np.array([k in held for k in key])
    # 1-day buffer around held-out days, as always
    near = set()
    for f, dd in held:
        for o in (-1, 0, 1):
            near.add((f, dd + o))
    is_buf = np.array([k in near for k in key])
    tr, va = lab[~is_buf], lab[is_va]
    ps = [lgb.LGBMRegressor(random_state=s, n_jobs=4, verbose=-1, **T_HUB)
          .fit(tr[cols], tr.sub_temp).predict(va[cols]) for s in seeds]
    p = np.mean(ps, axis=0)
    a = va.in_temp_ewm6.fillna(va.in_temp).values
    out = {"rmse": rmse(p, va.sub_temp), "bias": float(np.mean(p - va.sub_temp)),
           "n": len(va), "n_tr": len(tr)}
    m = a <= 8
    out["rmse_le8"] = rmse(p[m], va.sub_temp.values[m]) if m.sum() >= 20 else np.nan
    out["n_le8"] = int(m.sum())
    return out


def main():
    tX, ty, sX = load_raw()
    panel = F2.build(tX, sX).merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    panel["is_test"] = panel.row_id.isin(set(sX.row_id))
    panel = panel.merge(source_features(tX, sX), on="row_id", how="left")

    base = F2.view(panel, "sub_temp")
    res = [c for c in panel.columns if c.startswith("src_res")]
    allsrc = [c for c in panel.columns if c.startswith("src_")]

    sets = {"기준 (%d개)" % len(base): base,
            "+ 저장고 2개": base + res,
            "+ 출처 전체 %d개" % len(allsrc): base + allsrc}

    print("=== 가장 추운 날들을 통째로 빼고 예측 ===")
    for n in (10, 20, 30, 40):
        held = cold_days(panel, n)
        print("\n  [가장 추운 %d일 보류]" % n)
        first = None
        for nm, cols in sets.items():
            r = run(panel, cols, held)
            if first is None:
                first = r["rmse"]
                delta = ""
            else:
                delta = "  (%+.2f%%)" % (100 * (r["rmse"] / first - 1))
            print("    %-22s RMSE %.4f%s  편향 %+.3f  |  ≤8℃ %s (n=%d)  [평가행 %d]"
                  % (nm, r["rmse"], delta, r["bias"],
                     "%.4f" % r["rmse_le8"] if np.isfinite(r["rmse_le8"]) else "-",
                     r["n_le8"], r["n"]))

    print("\n=== 참고: 보류된 날들이 실제로 추운가 ===")
    lab = panel[(~panel.is_test) & panel.sub_temp.notna()]
    dm = lab.groupby(["farm", "day"]).in_temp_ewm6.min().sort_values()
    print("  가장 추운 40일의 일 최저 실내온도: %.1f ~ %.1f℃" % (dm.iloc[0], dm.iloc[39]))
    te = panel[panel.is_test]
    tmin = te.groupby(["farm", "day"]).in_temp_ewm6.min()
    print("  평가 60일의 일 최저: %.1f ~ %.1f℃ (중앙 %.1f)"
          % (tmin.min(), tmin.max(), tmin.median()))
    print("  평가일 중 학습 최저 40일 범위 안에 드는 날: %d일"
          % int((tmin <= dm.iloc[39]).sum()))


if __name__ == "__main__":
    main()
