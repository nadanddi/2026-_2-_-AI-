# -*- coding: utf-8 -*-
"""Fit the chosen v2 models on every labelled row and write the submission.

Final configuration and why each piece is there:
  sub_temp : curated thermal view (96 features) + huber + 3 model seeds.
             late-period block CV 0.8954 +-0.003 over 5 partitions.
  sub_ec   : curated balance view (102 features), blend of huber and
             huber-with-recency-weighting, 3 model seeds each.
             late-period block CV 0.3142 +-0.018.  The two components swap
             places between partition counts, so blending them is a hedge
             against noise rather than a bet on the luckier one.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import TARGET_FARMS, rmse
from model_v2 import get_panel
from model_v4 import T_HUB, E_HUB, MODEL_SEEDS
from ablation import is_domain
import features_v2 as F2


def slim(panel, target):
    """Final view: the curated view minus the literature-derived block.

    The ablation and then the full-ensemble comparison both put the slim view
    ahead (late 0.8944 vs 0.8963 for sub_temp, 0.3037 vs 0.3086 for sub_ec),
    and it is 30% smaller, which is the tie-breaker under overfitting pressure.
    """
    return [c for c in F2.view(panel, target) if not is_domain(c)]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "submission.csv")
PREV = os.path.join(os.path.dirname(HERE), "submission_v1.csv")


def fit_full(lab, test, fcols, target, params, recency_tau=None):
    w = None
    if recency_tau:
        dmax = lab.day.max()
        w = np.exp(-(dmax - lab.day.values) / float(recency_tau))
    ps = []
    for sd in MODEL_SEEDS:
        m = lgb.LGBMRegressor(random_state=sd, n_jobs=4, verbose=-1, **params)
        m.fit(lab[fcols], lab[target], sample_weight=w)
        ps.append(m.predict(test[fcols]))
    return np.mean(ps, axis=0)


def main():
    panel, tX, ty, sX = get_panel()
    test = panel[panel.is_test]
    vt = slim(panel, "sub_temp")
    ve = slim(panel, "sub_ec")
    print("features: sub_temp %d, sub_ec %d" % (len(vt), len(ve)))

    lab_t = panel[(~panel.is_test) & panel.sub_temp.notna()]
    lab_e = panel[(~panel.is_test) & panel.sub_ec.notna()]
    print("training rows: sub_temp %d, sub_ec %d" % (len(lab_t), len(lab_e)))

    p_temp = fit_full(lab_t, test, vt, "sub_temp", T_HUB)
    p_ec = np.mean([
        fit_full(lab_e, test, ve, "sub_ec", E_HUB),
        fit_full(lab_e, test, ve, "sub_ec", E_HUB, recency_tau=120),
    ], axis=0)

    sub = pd.DataFrame({"row_id": test.row_id.values,
                        "sub_temp": p_temp, "sub_ec": p_ec})
    sub = sX[["row_id"]].merge(sub, on="row_id", how="left")

    # clip EC to the range these two greenhouses have actually shown
    lo, hi = ty.sub_ec.min(), ty.sub_ec.max()
    sub["sub_ec"] = sub.sub_ec.clip(lo, hi)

    assert len(sub) == 1440
    assert sub.row_id.tolist() == sX.row_id.tolist()
    assert sub[["sub_temp", "sub_ec"]].notna().all().all()
    assert np.isfinite(sub[["sub_temp", "sub_ec"]].values).all()

    if os.path.exists(OUT) and not os.path.exists(PREV):
        os.rename(OUT, PREV)
    sub.to_csv(OUT, index=False, encoding="utf-8")
    print("wrote %s" % OUT)

    print("\n=== prediction summary ===")
    obs = ty[ty.row_id.str[:3].isin(TARGET_FARMS)]
    for t in ["sub_temp", "sub_ec"]:
        o = obs[t].dropna()
        print("  %-9s pred [%.3f, %.3f] mean %.3f | train [%.3f, %.3f] mean %.3f"
              % (t, sub[t].min(), sub[t].max(), sub[t].mean(),
                 o.min(), o.max(), o.mean()))

    if os.path.exists(PREV):
        old = pd.read_csv(PREV)
        j = old.merge(sub, on="row_id", suffixes=("_v1", "_v2"))
        print("\n=== movement vs the v1 submission ===")
        for t in ["sub_temp", "sub_ec"]:
            d = j["%s_v2" % t] - j["%s_v1" % t]
            print("  %-9s mean shift %+.4f | rms change %.4f | max |d| %.3f"
                  % (t, d.mean(), np.sqrt((d ** 2).mean()), d.abs().max()))

    # per-block levels against neighbouring labelled days
    sub["farm"] = sub.row_id.str[:3]
    sub["day"] = sub.row_id.str[4:7].astype(int)
    lab = ty.copy()
    lab["farm"] = lab.row_id.str[:3]
    lab["day"] = lab.row_id.str[4:7].astype(int)
    print("\n=== block level vs neighbouring train days ===")
    for farm in TARGET_FARMS:
        s = sub[sub.farm == farm]
        dl = lab[(lab.farm == farm) & lab.sub_ec.notna()].groupby("day").sub_ec.mean()
        dt = lab[lab.farm == farm].groupby("day").sub_temp.mean()
        ds = np.sort(s.day.unique())
        for b in np.split(ds, np.where(np.diff(ds) > 1)[0] + 1):
            near_e = dl[(dl.index >= b[0] - 6) & (dl.index <= b[-1] + 6)]
            near_t = dt[(dt.index >= b[0] - 6) & (dt.index <= b[-1] + 6)]
            print("  %s day %3d-%3d  EC %.3f vs nbr %.3f | temp %.2f vs nbr %.2f"
                  % (farm, b[0], b[-1], s[s.day.isin(b)].sub_ec.mean(),
                     near_e.mean(), s[s.day.isin(b)].sub_temp.mean(),
                     near_t.mean()))


if __name__ == "__main__":
    main()
