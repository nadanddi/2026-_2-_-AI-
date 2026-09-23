# -*- coding: utf-8 -*-
"""Final v3: confirm the chosen configurations on 7 partitions, fit, submit.

  sub_temp : full thermal view incl. the corrected-sign domain block (98 feat)
             + huber + 3 seeds.               late CV 0.8827 +-0.013 (5 part.)
  sub_ec   : slim view (68 feat) + ExtraTrees(min_samples_leaf=2) x 3 seeds,
             a carry-over from the earlier repo's v4 that reproduced here.
                                              late CV 0.3046 +-0.015 (5 part.)
Two EC alternatives (ET + recency weighting, ET/LGB blend) are scored one
last time on 7 partitions; ET alone is kept unless one beats it by more than
its spread.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from common import make_folds, rmse, TARGET_FARMS
from model_v2 import get_panel, N_FOLDS
from model_v4 import T_HUB, E_HUB, MODEL_SEEDS, blend_multi
from model_v5 import run as run_ec, evaluate as eval_ec
from submit_v2 import slim
import features_v2 as F2
import model_v5

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "submission.csv")
PREV = os.path.join(os.path.dirname(HERE), "submission_v2.csv")
FOLD_SEEDS = [0, 1, 2, 3, 4, 5, 6]


def et_model(seed):
    return make_pipeline(SimpleImputer(strategy="median"),
                         ExtraTreesRegressor(n_estimators=300, max_features=1.0,
                                             min_samples_leaf=2, random_state=seed,
                                             n_jobs=4))


def main():
    panel, tX, ty, sX = get_panel()
    days = {f: sorted(panel[(panel.farm == f) & (~panel.is_test)].day.unique())
            for f in TARGET_FARMS}
    vt = F2.view(panel, "sub_temp")
    ve = slim(panel, "sub_ec")
    print("features: sub_temp %d (with domain), sub_ec %d (slim)" % (len(vt), len(ve)))

    print("\n=== sub_temp confirmation (%d partitions) ===" % len(FOLD_SEEDS))
    r = blend_multi(panel, days, "sub_temp",
                    [dict(fcols=vt, params=T_HUB, seeds=MODEL_SEEDS)],
                    fold_seeds=FOLD_SEEDS)
    print("  huber x3, domain view          all %.4f | late %.4f +-%.3f" % r)

    print("\n=== sub_ec confirmation (%d partitions) ===" % len(FOLD_SEEDS))
    model_v5.FOLD_SEEDS = FOLD_SEEDS
    cands = {
        "ET alone": [dict(fcols=ve, kind="et", params=dict(leaf=2))],
        "ET + recency120": [dict(fcols=ve, kind="et", params=dict(leaf=2), recency_tau=120)],
        "ET / LGB-huber blend": [dict(fcols=ve, kind="et", params=dict(leaf=2)),
                                 dict(fcols=ve, params=E_HUB)],
    }
    scores = {nm: eval_ec(panel, days, cf, nm) for nm, cf in cands.items()}
    best = min(scores, key=scores.get)
    chosen = "ET alone"
    if best != chosen and scores[chosen] - scores[best] > 0.015:
        chosen = best
    print("  -> chosen: %s" % chosen)

    # ---------------------------- final fit ---------------------------------
    test = panel[panel.is_test]
    lab_t = panel[(~panel.is_test) & panel.sub_temp.notna()]
    lab_e = panel[(~panel.is_test) & panel.sub_ec.notna()]

    p_temp = np.mean([
        lgb.LGBMRegressor(random_state=sd, n_jobs=4, verbose=-1, **T_HUB)
        .fit(lab_t[vt], lab_t.sub_temp).predict(test[vt]) for sd in MODEL_SEEDS], axis=0)

    def fit_et(w=None):
        ps = []
        for sd in MODEL_SEEDS:
            m = et_model(sd)
            m.fit(lab_e[ve], lab_e.sub_ec, extratreesregressor__sample_weight=w)
            ps.append(m.predict(test[ve]))
        return np.mean(ps, axis=0)

    parts = [fit_et()]
    if chosen == "ET + recency120":
        w = np.exp(-(lab_e.day.max() - lab_e.day.values) / 120.0)
        parts = [fit_et(w)]
    elif chosen == "ET / LGB-huber blend":
        parts.append(np.mean([
            lgb.LGBMRegressor(random_state=sd, n_jobs=4, verbose=-1, **E_HUB)
            .fit(lab_e[ve], lab_e.sub_ec).predict(test[ve]) for sd in MODEL_SEEDS], axis=0))
    p_ec = np.mean(parts, axis=0)

    sub = sX[["row_id"]].merge(
        pd.DataFrame({"row_id": test.row_id.values, "sub_temp": p_temp, "sub_ec": p_ec}),
        on="row_id", how="left")
    sub["sub_ec"] = sub.sub_ec.clip(ty.sub_ec.min(), ty.sub_ec.max())
    assert len(sub) == 1440 and sub.row_id.tolist() == sX.row_id.tolist()
    assert np.isfinite(sub[["sub_temp", "sub_ec"]].values).all()

    if os.path.exists(OUT) and not os.path.exists(PREV):
        os.rename(OUT, PREV)
    sub.to_csv(OUT, index=False, encoding="utf-8")
    print("\nwrote %s" % OUT)
    obs = ty[ty.row_id.str[:3].isin(TARGET_FARMS)]
    for t in ["sub_temp", "sub_ec"]:
        o = obs[t].dropna()
        print("  %-9s pred [%.3f, %.3f] mean %.3f | train [%.3f, %.3f] mean %.3f"
              % (t, sub[t].min(), sub[t].max(), sub[t].mean(), o.min(), o.max(), o.mean()))
    if os.path.exists(PREV):
        old = pd.read_csv(PREV)
        j = old.merge(sub, on="row_id", suffixes=("_v2", "_v3"))
        for t in ["sub_temp", "sub_ec"]:
            d = j["%s_v3" % t] - j["%s_v2" % t]
            print("  %-9s vs v2: mean shift %+.4f | rms change %.4f" % (t, d.mean(), np.sqrt((d ** 2).mean())))


if __name__ == "__main__":
    main()
