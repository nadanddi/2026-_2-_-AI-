# -*- coding: utf-8 -*-
"""Round after reading the earlier repo: corrected curtain sign, pooled temp.

1. Curtain sign.  The earlier report states the PDF definition (curtain values
   are openness) and the data confirms it, so rad_eff / screen terms were
   inverted until now.  Re-run the domain ablation with the corrected block.
2. Pooled temperature model, the earlier repo's v3/v4 design: one LightGBM on
   all 51 greenhouses with NaN kept for absent sensors, farm as categorical,
   target farms weighted x20, blended 50:50 with the target-only model.  This
   is a different design from the stacked prior that failed earlier.
3. Air-residual target for temperature: learn sub_temp - in_temp_ewm6.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import load_raw, make_folds, split_mask, rmse, TARGET_FARMS
from model_v2 import N_FOLDS, score
from model_v3 import run
from model_v4 import T_HUB, MODEL_SEEDS, blend_multi
from ablation import is_domain
import features_v2 as F2

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE2 = os.path.join(HERE, "panel_v2.pkl")
CACHE51 = os.path.join(HERE, "panel_v2_all51.pkl")
FOLD_SEEDS = [0, 1, 2, 3, 4]

POOL = dict(objective="huber", n_estimators=300, learning_rate=0.05,
            num_leaves=23, min_child_samples=20, reg_lambda=10.0,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.6)


def rebuild_target_panel():
    if os.path.exists(CACHE2):
        os.remove(CACHE2)
    tX, ty, sX = load_raw()
    panel = F2.build(tX, sX)
    panel = panel.merge(ty[["row_id", "sub_temp", "sub_ec"]], on="row_id", how="left")
    panel["is_test"] = panel.row_id.isin(set(sX.row_id))
    pd.to_pickle((panel, tX, ty, sX), CACHE2)
    return panel, tX, ty, sX


def all51_panel(tX, ty, sX):
    if os.path.exists(CACHE51):
        return pd.read_pickle(CACHE51)
    farms = sorted(tX.farm.unique())
    p = F2.build(tX, sX, farms=farms)
    p = p.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    p["is_test"] = p.row_id.isin(set(sX.row_id))
    pd.to_pickle(p, CACHE51)
    return p


def run_pooled(p51, folds, fcols, params, weight=20.0, seeds=(7,)):
    """Train on every labelled greenhouse; validate on target-farm blocks."""
    lab = p51[(~p51.is_test) & p51.sub_temp.notna()].reset_index(drop=True)
    tgt = lab.farm.isin(TARGET_FARMS).values
    oof = np.full(len(lab), np.nan)
    use = fcols + ["farm_code"]
    for fd in folds:
        trm, vam = split_mask(lab, fd)          # buffer only touches target farms
        tr, va = lab[trm], lab[vam & tgt]
        if not len(va):
            continue
        w = np.where(tr.farm.isin(TARGET_FARMS), weight, 1.0)
        ps = []
        for sd in seeds:
            m = lgb.LGBMRegressor(random_state=sd, n_jobs=4, verbose=-1, **params)
            m.fit(tr[use], tr.sub_temp, sample_weight=w,
                  categorical_feature=["farm_code"])
            ps.append(m.predict(va[use]))
        oof[np.where(vam & tgt)[0]] = np.mean(ps, axis=0)
    keep = tgt
    return lab[keep].reset_index(drop=True), oof[keep]


def run_residual(panel, folds, fcols, params, seeds=MODEL_SEEDS, base_col="in_temp_ewm6"):
    lab = panel[(~panel.is_test) & panel.sub_temp.notna()].reset_index(drop=True)
    base = lab[base_col].fillna(lab.in_temp).fillna(lab.sub_temp.mean()).values
    oof = np.full(len(lab), np.nan)
    for fd in folds:
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        if not len(va):
            continue
        ps = []
        for sd in seeds:
            m = lgb.LGBMRegressor(random_state=sd, n_jobs=4, verbose=-1, **params)
            m.fit(tr[fcols], tr.sub_temp.values - base[trm])
            ps.append(m.predict(va[fcols]) + base[vam])
        oof[np.where(vam)[0]] = np.mean(ps, axis=0)
    return lab, oof


def main():
    print("rebuilding target panel with corrected curtain sign ...")
    panel, tX, ty, sX = rebuild_target_panel()
    days = {f: sorted(panel[(panel.farm == f) & (~panel.is_test)].day.unique())
            for f in TARGET_FARMS}
    vt = F2.view(panel, "sub_temp")
    ve = F2.view(panel, "sub_ec")
    vt_s = [c for c in vt if not is_domain(c)]
    ve_s = [c for c in ve if not is_domain(c)]

    def rep(n, r):
        print("  %-40s all %.4f | late %.4f +-%.3f" % ((n,) + tuple(r)))

    print("\n===== 1. domain block, curtain sign corrected (5 partitions) =====")
    from model_v4 import E_HUB
    for target, vf, vs, cfgs in [
            ("sub_temp", vt, vt_s, lambda fc: [dict(fcols=fc, params=T_HUB, seeds=MODEL_SEEDS)]),
            ("sub_ec", ve, ve_s, lambda fc: [dict(fcols=fc, params=E_HUB, seeds=MODEL_SEEDS),
                                            dict(fcols=fc, params=E_HUB, seeds=MODEL_SEEDS,
                                                 recency_tau=120)])]:
        rep("%s  with domain (%d)" % (target, len(vf)),
            blend_multi(panel, days, target, cfgs(vf), fold_seeds=FOLD_SEEDS))
        rep("%s  slim (%d)" % (target, len(vs)),
            blend_multi(panel, days, target, cfgs(vs), fold_seeds=FOLD_SEEDS))

    print("\n===== 2. air-residual target for sub_temp =====")
    for fc, nm in [(vt, "with domain"), (vt_s, "slim")]:
        alls, lates = [], []
        for fs in FOLD_SEEDS:
            folds = make_folds(days, n_folds=N_FOLDS, seed=fs)
            lab, oof = run_residual(panel, folds, fc, T_HUB)
            a, l = score(lab, oof, "sub_temp")
            alls.append(a); lates.append(l)
        rep("residual on in_temp_ewm6, %s" % nm, (np.mean(alls), np.mean(lates), np.std(lates)))

    print("\n===== 3. pooled 51-farm temperature (earlier repo v3/v4 design) =====")
    print("building 51-farm panel ...")
    p51 = all51_panel(tX, ty, sX)
    print("  rows %d" % len(p51))
    for fc, nm in [(vt_s, "slim"), (vt, "with domain")]:
        alls, lates, alls_b, lates_b = [], [], [], []
        for fs in FOLD_SEEDS[:3]:
            folds = make_folds(days, n_folds=N_FOLDS, seed=fs)
            labp, oofp = run_pooled(p51, folds, fc, POOL)
            labt, ooft = run(panel, folds, "sub_temp", fc, T_HUB, seeds=(7,))
            # align by row_id
            m = pd.DataFrame({"row_id": labp.row_id, "p": oofp}).merge(
                pd.DataFrame({"row_id": labt.row_id, "t": ooft, "y": labt.sub_temp,
                              "day": labt.day}), on="row_id")
            got = m.p.notna() & m.t.notna()
            late = got & (m.day >= 183)
            alls.append(rmse(m.p[got], m.y[got])); lates.append(rmse(m.p[late], m.y[late]))
            b = 0.5 * (m.p + m.t)
            alls_b.append(rmse(b[got], m.y[got])); lates_b.append(rmse(b[late], m.y[late]))
        rep("pooled x20 alone, %s" % nm, (np.mean(alls), np.mean(lates), np.std(lates)))
        rep("pooled/target 50:50, %s" % nm, (np.mean(alls_b), np.mean(lates_b), np.std(lates_b)))


if __name__ == "__main__":
    main()
