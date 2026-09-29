# -*- coding: utf-8 -*-
"""Step 4 -- does the source-chain reservoir actually improve prediction?

Steps 1-3 established: the reservoir effect is universal (48/48), it is damaged
for F13/F47 by the day-level concatenation, and rebuilding the slow state along
source chains recovers most of it.  This turns that into features and scores
them on the test-geometry CV, with a breakdown by temperature band because the
mechanism is supposed to matter most when it is cold.

Everything uses hour-00 values of the current day and earlier days only.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from common import load_raw, split_mask, rmse, TARGET_FARMS
from geometry_cv import geometry_folds
from make_submission import T_HUB, SEEDS
import features_v2 as F2

FP = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan",
      "act_co2", "act_fog", "in_co2", "in_hum"]
BANDS = [("<8", -99, 8), ("8~10", 8, 10), ("10~12", 10, 12),
         ("12~15", 12, 15), (">15", 15, 99)]


def source_features(tX, sX, k=4, halflives=(2, 5)):
    """Per-row source id and source-chain slow air state (causal)."""
    cols = ["row_id", "farm", "day", "hour", "t", "in_temp", "act_heating"] + FP
    allx = pd.concat([tX, sX], ignore_index=True)
    allx = allx[allx.farm.isin(TARGET_FARMS)]
    out = []
    for farm, g in allx.groupby("farm"):
        g = g.sort_values("t")
        p = g.set_index("t").reindex(np.arange(g.t.min(), g.t.max() + 1))
        p["day"] = p.index // 24
        h0 = p[p.index % 24 == 0][FP].copy()
        h0.index = h0.index // 24
        h0 = h0.dropna()
        lab = pd.Series(
            KMeans(n_clusters=k, n_init=10, random_state=7)
            .fit_predict(StandardScaler().fit_transform(h0.values)), index=h0.index)
        daily_air = p.groupby("day").in_temp.mean()
        daily_heat = p.groupby("day").act_heating.mean()
        feat = {"src_id": p.day.map(lab)}
        for hl in halflives:
            slow, prev_air, prev_heat, gapd = {}, {}, {}, {}
            for s in lab.unique():
                days = sorted(lab.index[lab == s])
                v = daily_air.reindex(days)
                h = daily_heat.reindex(days)
                ew = v.shift(1).ewm(halflife=hl, ignore_na=True).mean()
                hw = h.shift(1).ewm(halflife=hl, ignore_na=True).mean()
                pv = v.shift(1)
                for i, dd in enumerate(days):
                    slow[dd] = ew.get(dd, np.nan)
                    prev_air[dd] = pv.get(dd, np.nan)
                    prev_heat[dd] = hw.get(dd, np.nan)
                    gapd[dd] = dd - days[i - 1] if i else np.nan
            feat["src_slow%d" % hl] = p.day.map(pd.Series(slow))
            feat["src_prevair%d" % hl] = p.day.map(pd.Series(prev_air))
            feat["src_prevheat%d" % hl] = p.day.map(pd.Series(prev_heat))
            feat["src_gapdays%d" % hl] = p.day.map(pd.Series(gapd))
        f = pd.DataFrame(feat, index=p.index)
        a3 = p.in_temp.ewm(halflife=3, ignore_na=True).mean()
        for hl in halflives:
            # the reservoir itself: how much warmer the source's slow state is
            f["src_res%d" % hl] = f["src_slow%d" % hl] - a3
        f["row_id"] = p["row_id"]
        out.append(f.dropna(subset=["row_id"]))
    return pd.concat(out, ignore_index=True)


def evaluate(panel, cols, label):
    lab = panel[(~panel.is_test) & panel.sub_temp.notna()].reset_index(drop=True)
    oof = np.full(len(lab), np.nan)
    for fd in geometry_folds():
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        ps = [lgb.LGBMRegressor(random_state=s, n_jobs=4, verbose=-1, **T_HUB)
              .fit(tr[cols], tr.sub_temp).predict(va[cols]) for s in SEEDS]
        oof[np.where(vam)[0]] = np.mean(ps, axis=0)
    got = ~np.isnan(oof)
    lab = lab[got].copy()
    lab["p"] = oof[got]
    a3 = lab.in_temp_ewm6.fillna(lab.in_temp)
    print("  %-34s 전체 %.4f" % (label, rmse(lab.p, lab.sub_temp)), end="")
    for nm, a, b in BANDS:
        m = (a3 >= a) & (a3 < b)
        print("  %s %s" % (nm, "%.3f(%d)" % (rmse(lab.p[m], lab.sub_temp[m]), int(m.sum()))
                           if m.sum() >= 20 else "-"), end="")
    print()
    return lab


def main():
    tX, ty, sX = load_raw()
    panel = F2.build(tX, sX).merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    panel["is_test"] = panel.row_id.isin(set(sX.row_id))
    src = source_features(tX, sX)
    panel = panel.merge(src, on="row_id", how="left")

    base = F2.view(panel, "sub_temp")
    res_cols = [c for c in src.columns if c.startswith("src_res")]
    slow_cols = [c for c in src.columns if c.startswith("src_slow") or c.startswith("src_prev")]
    all_src = [c for c in src.columns if c != "row_id"]

    print("=== 온도 예측, 구간별 (test 배치 CV, 3시드) ===")
    print("  %-34s %s" % ("구성", "전체" + "".join("   %s" % nm for nm, _, _ in BANDS)))
    evaluate(panel, base, "기준 (%d개)" % len(base))
    evaluate(panel, base + res_cols, "+ 저장고 2개")
    evaluate(panel, base + res_cols + ["src_id"], "+ 저장고 + 출처ID")
    evaluate(panel, base + all_src, "+ 출처 전체 %d개" % len(all_src))


if __name__ == "__main__":
    main()
