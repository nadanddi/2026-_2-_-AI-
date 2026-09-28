# -*- coding: utf-8 -*-
"""Analysis Q3e: is the train-vs-test shape difference real, and does it sit
on the days with the unexplained day-level offsets?

anal_q3d_adversarial.py: shape-only features separate training from test rows
(AUC 0.642) and the new suspect rows behave like the rule-flagged ones, but
they are warm (in_temp mean 17.7) while the test is cold, so the classifier
may be reading the season through level-dependent shapes.

1. Season-matched adversarial validation: resample training rows so their
   in_temp histogram matches the test's (1 C bins), refit the classifier.
   AUC near 0.5 after matching = the difference was the season.
2. Day-level: mean train-likeness per greenhouse-day (matched model) vs the
   day-level residual of the diagnostic OOF.  If the unexplained day offsets
   concentrate on the most train-like (noised / restored) days, part of that
   58% floor is restoration the clean test does not contain.

Run:  cd research && PYTHONPATH="" <python> anal_q3e_matched.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import common
from harness import load
from cleanw_v6 import weights
from anal_q3d_adversarial import shape_features

SHAPE_ONLY = ["in_temp_d1", "in_temp_ad1", "in_temp_d2", "in_temp_ad1_next", "in_temp_run",
              "in_hum_d1", "in_hum_ad1", "in_hum_d2", "in_hum_ad1_next", "in_hum_run",
              "in_co2_d1", "in_co2_ad1", "in_co2_d2", "in_co2_ad1_next", "in_co2_run",
              "rh_vs_temp", "co2_d1_no_supply", "is_night"]


def clf():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                              n_jobs=2, random_state=0, is_unbalance=True)


def cv_auc(F, y, groups):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(F, y, groups):
        p[te] = clf().fit(F.iloc[tr], y[tr]).predict_proba(F.iloc[te])[:, 1]
    return roc_auc_score(y, p), p


def main():
    tX, ty, sX = common.load_raw()
    a = pd.concat([tX.assign(split="train"), sX.assign(split="test")], ignore_index=True)
    a = a.query("farm in ['F13','F47']").reset_index(drop=True)
    F = shape_features(a)
    a = a.loc[F.index].reset_index(drop=True)
    F = F.reset_index(drop=True)
    y = (a.split == "train").astype(int).values
    groups = (a.farm + "_" + a.day.astype(str)).values

    auc0, _ = cv_auc(F, y, groups)
    auc1, _ = cv_auc(F[SHAPE_ONLY], y, groups)
    print("unmatched: all shape features AUC %.3f | without level-dependent ones AUC %.3f" % (auc0, auc1))

    # season matching on in_temp (1 C bins)
    rng = np.random.RandomState(0)
    bins = np.floor(a.in_temp.clip(-2, 32)).values
    te_h = pd.Series(bins[y == 0]).value_counts(normalize=True)
    tr_idx = np.where(y == 1)[0]
    tr_h = pd.Series(bins[tr_idx]).value_counts(normalize=True)
    w = pd.Series(bins[tr_idx]).map(te_h / tr_h).fillna(0).values
    take = rng.choice(tr_idx, size=len(tr_idx) // 2, replace=True, p=w / w.sum())
    sel = np.concatenate([take, np.where(y == 0)[0]])
    for nm, cols in (("all shape features", list(F.columns)), ("without level-dependent", SHAPE_ONLY)):
        auc, _ = cv_auc(F.iloc[sel].reset_index(drop=True)[cols], y[sel], groups[sel])
        print("season-matched (%s): AUC %.3f" % (nm, auc))

    # day-level: train-likeness (level-free model, trained on matched sample) vs day residual
    m = clf().fit(F.iloc[sel][SHAPE_ONLY], y[sel])
    a["p_train"] = m.predict_proba(F[SHAPE_ONLY])[:, 1]
    panel, lab, _ = load()
    z = np.load(env.LOCAL + "/oof_temp_diag.npz", allow_pickle=True)
    e = pd.DataFrame({"row_id": lab.row_id.values, "e": z["r3"] - lab.sub_temp.values})
    t = a[a.split == "train"].merge(e, on="row_id")
    day = t.groupby(["farm", "day"]).agg(p=("p_train", "mean"), res=("e", "mean"),
                                         ares=("e", lambda s: np.sqrt((s ** 2).mean())), n=("e", "size"))
    day = day[day.n >= 12]
    print("\nday-level: corr(train-likeness, |day-level residual|) = %.3f  (n=%d days)"
          % (np.corrcoef(day.p, day.res.abs())[0, 1], len(day)))
    day["q"] = pd.qcut(day.p, 4, labels=["least train-like", "Q2", "Q3", "most train-like"])
    print(day.groupby("q", observed=True).agg(days=("p", "size"), abs_day_offset=("res", lambda s: s.abs().mean()),
                                              day_rmse=("ares", "mean")).round(3).to_string())
    test_p = a[a.split == "test"].groupby(["farm", "day"]).p_train.mean()
    print("\nmean train-likeness: test days %.3f | training days %.3f" % (test_p.mean(), day.p.mean()))


if __name__ == "__main__":
    main()
