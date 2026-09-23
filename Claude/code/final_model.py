# -*- coding: utf-8 -*-
"""Final model: multi-seed ensembles, honest block CV, and submission file.

Findings that shaped this configuration (see experiments.py output):
  * the auxiliary-greenhouse prior does NOT help sub_temp -> dropped;
  * huber and l2 are complementary for sub_temp -> averaged;
  * log-target and raw-target are complementary for sub_ec -> averaged;
  * post-hoc smoothing of EC (within-day expanding mean, previous-day blend)
    makes it WORSE, so predictions are used as-is;
  * single-model CV noise is ~0.01 dS/m, hence seed averaging everywhere.
"""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import (feature_columns, make_folds, split_mask, rmse,
                    TARGET_FARMS)
from cv import prepare, N_FOLDS

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [7, 101, 2024]

CFG = {
    "sub_temp": dict(
        rounds=1204,
        drop=["aux_temp_prior"],
        variants=[
            dict(objective="regression", learning_rate=0.03, num_leaves=63,
                 min_child_samples=40, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.6, reg_lambda=1.0),
            dict(objective="huber", learning_rate=0.03, num_leaves=63,
                 min_child_samples=40, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.6, reg_lambda=1.0),
        ],
        log=[False, False],
    ),
    "sub_ec": dict(
        rounds=411,
        drop=[],
        variants=[
            dict(objective="regression", learning_rate=0.02, num_leaves=15,
                 min_child_samples=60, subsample=0.7, subsample_freq=1,
                 colsample_bytree=0.4, reg_lambda=5.0),
            dict(objective="regression", learning_rate=0.02, num_leaves=15,
                 min_child_samples=60, subsample=0.7, subsample_freq=1,
                 colsample_bytree=0.4, reg_lambda=5.0),
        ],
        log=[False, True],
    ),
}


def fit_predict(tr, va, fcols, target, cfg, rounds=None):
    """Average over variants x seeds."""
    preds = []
    for var, lg in zip(cfg["variants"], cfg["log"]):
        y = np.log(tr[target]) if lg else tr[target]
        for sd in SEEDS:
            m = lgb.LGBMRegressor(n_estimators=rounds or cfg["rounds"],
                                  random_state=sd, n_jobs=4, verbose=-1, **var)
            m.fit(tr[fcols], y)
            p = m.predict(va[fcols])
            preds.append(np.exp(p) if lg else p)
    return np.mean(preds, axis=0)


def cv_score(panel, folds, target, rounds=None, verbose=True):
    cfg = CFG[target]
    lab = panel[(~panel.is_test) & panel[target].notna()].reset_index(drop=True)
    fcols = feature_columns(panel, extra_drop=["is_test", "day_mod7"] + cfg["drop"])
    oof = np.full(len(lab), np.nan)
    for i, fd in enumerate(folds):
        fd = {f: d for f, d in fd.items() if f in TARGET_FARMS}
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        if len(va) == 0:
            continue
        p = fit_predict(tr, va, fcols, target, cfg, rounds)
        oof[np.where(vam)[0]] = p
        if verbose:
            print("    fold %d  n_va=%4d  rmse=%.4f" % (i, len(va),
                                                        rmse(p, va[target])))
    got = ~np.isnan(oof)
    per = {f: rmse(oof[got & (lab.farm == f).values],
                   lab[target].values[got & (lab.farm == f).values])
           for f in TARGET_FARMS}
    return rmse(oof[got], lab[target].values[got]), per, oof, lab, fcols


def main():
    panel, tX, ty, sX = prepare(use_aux=False)
    days = {f: sorted(panel[(panel.farm == f) & (~panel.is_test)].day.unique())
            for f in TARGET_FARMS}
    folds = make_folds(days, n_folds=N_FOLDS, seed=0)

    results = {}
    for target in ["sub_temp", "sub_ec"]:
        print("\n=== %s : final ensemble (%d variants x %d seeds) ==="
              % (target, len(CFG[target]["variants"]), len(SEEDS)))
        r, per, oof, lab, fcols = cv_score(panel, folds, target)
        print("  OOF RMSE = %.4f   F13 %.4f  F47 %.4f" % (r, per["F13"], per["F47"]))
        got = ~np.isnan(oof)
        late = got & (lab.day.values >= 183)
        print("  LATE (test period, day>=183) = %.4f   n=%d   [early %.4f]"
              % (rmse(oof[late], lab[target].values[late]), int(late.sum()),
                 rmse(oof[got & ~late], lab[target].values[got & ~late])))
        print("  features used: %d" % len(fcols))
        results[target] = (r, per, lab, fcols)

        # sensitivity to the round count (it was picked by early stopping)
        base = CFG[target]["rounds"]
        for mult in (0.5, 1.5):
            rr, _, _, _, _ = cv_score(panel, folds, target,
                                      rounds=int(base * mult), verbose=False)
            print("  rounds x%.1f (%4d) -> %.4f" % (mult, int(base * mult), rr))

    # ---------------- fit on everything, predict the test set ----------------
    print("\n=== fitting final models on all labelled rows ===")
    sub = sX[["row_id"]].copy()
    for target in ["sub_temp", "sub_ec"]:
        cfg = CFG[target]
        fcols = results[target][3]
        lab = panel[(~panel.is_test) & panel[target].notna()]
        test = panel[panel.is_test]
        p = fit_predict(lab, test, fcols, target, cfg)
        pred = pd.DataFrame({"row_id": test.row_id.values, target: p})
        sub = sub.merge(pred, on="row_id", how="left")
        print("  %s: mean %.3f  min %.3f  max %.3f"
              % (target, p.mean(), p.min(), p.max()))

    # clip EC to the physically observed range of these two greenhouses
    lo, hi = ty.sub_ec.min(), ty.sub_ec.max()
    sub["sub_ec"] = sub.sub_ec.clip(lo, hi)
    sub = sub[["row_id", "sub_temp", "sub_ec"]]

    assert len(sub) == 1440
    assert sub.row_id.tolist() == sX.row_id.tolist()
    assert sub[["sub_temp", "sub_ec"]].notna().all().all()
    assert np.isfinite(sub[["sub_temp", "sub_ec"]].values).all()

    out = os.path.join(os.path.dirname(HERE), "submission.csv")
    sub.to_csv(out, index=False, encoding="utf-8")
    print("\nwrote %s" % out)
    print(sub.head(3).to_string(index=False))

    print("\n" + "=" * 52)
    print("%-10s %12s %10s %10s" % ("target", "CV RMSE", "F13", "F47"))
    for t, (r, per, _, _) in results.items():
        print("%-10s %12.4f %10.4f %10.4f" % (t, r, per["F13"], per["F47"]))
    print("=" * 52)


if __name__ == "__main__":
    main()
