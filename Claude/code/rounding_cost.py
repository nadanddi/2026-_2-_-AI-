# -*- coding: utf-8 -*-
"""How much does 1-degree input resolution actually cost?

The other 48 greenhouses store integers.  Rather than argue about whether that
makes them unusable, this rounds F13/F47's own inputs to the same resolution
and measures the damage on the same CV.  If the damage is small, resolution is
not what blocks transfer -- something else is.
"""
import numpy as np
import pandas as pd

from common import load_raw, rmse, split_mask, TARGET_FARMS, USABLE
from geometry_cv import geometry_folds, score, lgb_fp
from make_submission import T_HUB
import features_v2 as F2


def resolution_report(tX, ty):
    print("=== 1. 해상도 실태 (정수 비율) ===")
    d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    d["grp"] = np.where(d.farm.isin(["F13", "F47"]), "F13/F47",
                        np.where(d.farm == "F32", "F32", "나머지 48곳"))
    cols = ["in_temp", "in_hum", "in_co2", "in_rad", "sub_temp"]
    print("  %-14s %s" % ("", "".join("%12s" % c for c in cols)))
    for g, sub in d.groupby("grp"):
        row = ""
        for c in cols:
            s = sub[c].dropna()
            row += "%12s" % ("%.0f%%" % (100 * np.isclose(s % 1, 0).mean()) if len(s) else "-")
        print("  %-14s %s" % (g, row))
    print("\n  → in_hum·in_co2는 어디서나 정수. 차이가 나는 건 in_temp와 sub_temp뿐.")
    s = d.loc[d.grp == "F13/F47", "in_temp"].dropna()
    print("  → F13/F47 in_temp 최소 간격 %.2f, 고유값 %d개"
          % (np.diff(np.sort(s.unique())).min(), s.nunique()))
    print("  → 1℃ 반올림이 더하는 잡음 std = %.3f℃ (균등분포 1/sqrt(12))" % (1 / np.sqrt(12)))


def main():
    tX, ty, sX = load_raw()
    resolution_report(tX, ty)

    print("\n=== 2. F13/F47 입력을 정수로 반올림하면 온도 모델이 얼마나 나빠지나 ===")
    variants = {
        "원본 (in_temp 0.1℃)": None,
        "in_temp만 1℃ 반올림": ["in_temp"],
        "in_temp+외부기온 반올림": ["in_temp", "out_temp"],
        "실내외 온도 모두 반올림": ["in_temp", "out_temp", "in_hum", "out_hum"],
    }
    base = None
    for name, round_cols in variants.items():
        t2 = tX.copy()
        s2 = sX.copy()
        if round_cols:
            for c in round_cols:
                for df in (t2, s2):
                    m = df.farm.isin(TARGET_FARMS)
                    df.loc[m, c] = df.loc[m, c].round()
        panel = F2.build(t2, s2)
        panel = panel.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
        panel["is_test"] = panel.row_id.isin(set(s2.row_id))
        v = F2.view(panel, "sub_temp")
        lab = panel[(~panel.is_test) & panel.sub_temp.notna()].reset_index(drop=True)
        r = score(lab, geometry_folds(), "sub_temp", lgb_fp(v, T_HUB))[0]
        if base is None:
            base = r
        print("  %-26s %.4f   (%+.2f%%)" % (name, r, 100 * (r / base - 1)))

    print("\n=== 3. 라벨까지 반올림하면 (다른 온실과 같은 조건) ===")
    ty2 = ty.copy()
    m = ty2.row_id.str[:3].isin(TARGET_FARMS)
    ty2.loc[m, "sub_temp"] = ty2.loc[m, "sub_temp"].round()
    t2, s2 = tX.copy(), sX.copy()
    for c in ["in_temp"]:
        for df in (t2, s2):
            mm = df.farm.isin(TARGET_FARMS)
            df.loc[mm, c] = df.loc[mm, c].round()
    panel = F2.build(t2, s2).merge(ty2[["row_id", "sub_temp"]], on="row_id", how="left")
    panel["is_test"] = panel.row_id.isin(set(s2.row_id))
    v = F2.view(panel, "sub_temp")
    lab = panel[(~panel.is_test) & panel.sub_temp.notna()].reset_index(drop=True)
    # score against the ROUNDED labels it was trained on, and against the true ones
    r_round = score(lab, geometry_folds(), "sub_temp", lgb_fp(v, T_HUB))[0]
    print("  입력+라벨 모두 반올림, 반올림 라벨로 채점 : %.4f" % r_round)
    true = ty.set_index("row_id").sub_temp
    oof = np.full(len(lab), np.nan)
    for fd in geometry_folds():
        trm, vam = split_mask(lab, fd)
        oof[np.where(vam)[0]] = lgb_fp(v, T_HUB)(lab[trm], lab[vam], "sub_temp")
    got = ~np.isnan(oof)
    print("  같은 모델을 '진짜(소수) 라벨'로 채점        : %.4f  (%+.2f%% vs 원본)"
          % (rmse(oof[got], true.loc[lab.row_id[got]].values),
             100 * (rmse(oof[got], true.loc[lab.row_id[got]].values) / base - 1)))


if __name__ == "__main__":
    main()
