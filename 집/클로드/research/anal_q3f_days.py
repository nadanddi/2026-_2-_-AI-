# -*- coding: utf-8 -*-
"""Analysis Q3f: out-of-fold "train-likeness" per greenhouse-day, and what the
most train-like days are.

anal_q3e_matched.py found that the quarter of training days whose input SHAPE
looks least like the (clean, unrestored) test inputs carries double the
day-level offset (0.584 vs ~0.28).  But those scores were in-sample: the
classifier was scored on rows it was trained on.  Here:

  * season matching by importance weights (training rows re-weighted so their
    1 C in_temp histogram matches the test's) instead of resampling;
  * level-free shape features only (SHAPE_ONLY);
  * every row scored out of fold (GroupKFold by greenhouse-day, 5 folds).

Step 1  does the day-level link survive?  corr and quartile table.
Step 2  what are the most train-like days: which shape features differ,
        record section, greenhouse, overlap with the rule-flagged rows.

Saves local/anal_q3f_dayscore.csv (farm, day, p_oof, quartile).

Run:  cd research && PYTHONPATH="" <python> anal_q3f_days.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import common
from harness import load
from anal_q3d_adversarial import shape_features
from anal_q3e_matched import SHAPE_ONLY


def main():
    tX, ty, sX = common.load_raw()
    a = pd.concat([tX.assign(split="train"), sX.assign(split="test")], ignore_index=True)
    a = a.query("farm in ['F13','F47']").reset_index(drop=True)
    F = shape_features(a)
    a = a.loc[F.index].reset_index(drop=True)
    F = F.reset_index(drop=True)[SHAPE_ONLY]
    y = (a.split == "train").astype(int).values
    groups = (a.farm + "_" + a.day.astype(str)).values

    bins = np.floor(a.in_temp.clip(-2, 32)).values
    te_h = pd.Series(bins[y == 0]).value_counts(normalize=True)
    tr_h = pd.Series(bins[y == 1]).value_counts(normalize=True)
    sw = np.where(y == 1, pd.Series(bins).map(te_h / tr_h).fillna(0).values, 1.0)
    sw = np.where(y == 1, sw * (y == 0).sum() / sw[y == 1].sum(), sw)   # balance classes

    p = np.zeros(len(a))
    for tr, te in GroupKFold(5).split(F, y, groups):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                               n_jobs=2, random_state=0)
        p[te] = m.fit(F.iloc[tr], y[tr], sample_weight=sw[tr]).predict_proba(F.iloc[te])[:, 1]
    print("season-weighted, level-free, out-of-fold AUC: %.3f" % roc_auc_score(y, p, sample_weight=sw))
    a["p_oof"] = p

    panel, lab, _ = load()
    z = np.load(env.LOCAL + "/oof_temp_diag.npz", allow_pickle=True)
    e = pd.DataFrame({"row_id": lab.row_id.values, "e": z["r3"] - lab.sub_temp.values})
    t = a[a.split == "train"].merge(e, on="row_id")
    day = t.groupby(["farm", "day"]).agg(p=("p_oof", "mean"), res=("e", "mean"),
                                         ares=("e", lambda s: np.sqrt((s ** 2).mean())),
                                         n=("e", "size")).reset_index()
    day = day[day.n >= 12].copy()
    print("\n== step 1: does the day-level link survive out of fold? ==")
    print("corr(train-likeness, |day offset|) = %.3f   (in-sample earlier: 0.321, n=%d days)"
          % (np.corrcoef(day.p, day.res.abs())[0, 1], len(day)))
    day["q"] = pd.qcut(day.p, 4, labels=["Q1 test-like", "Q2", "Q3", "Q4 train-like"])
    print(day.groupby("q", observed=True).agg(days=("p", "size"), abs_day_offset=("res", lambda s: s.abs().mean()),
                                              day_rmse=("ares", "mean")).round(3).to_string())
    # bootstrap CI for Q4 vs Q1-3 difference in |offset|
    rng = np.random.RandomState(0)
    q4 = day[day.q == "Q4 train-like"].res.abs().values
    rest = day[day.q != "Q4 train-like"].res.abs().values
    diffs = [rng.choice(q4, len(q4)).mean() - rng.choice(rest, len(rest)).mean() for _ in range(4000)]
    print("|offset| Q4 minus rest: %.3f, 95%% CI [%.3f, %.3f]"
          % (q4.mean() - rest.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)))

    print("\n== step 2: what are the most train-like days? ==")
    q4d = day[day.q == "Q4 train-like"]
    print("section: Q4 %s | all %s" % ((q4d.day >= 179).mean().round(3), (day.day >= 179).mean().round(3))
          + "  (share in 2nd pass)")
    print("greenhouse: Q4 %s | all %s" % (q4d.farm.value_counts(normalize=True).round(2).to_dict(),
                                          day.farm.value_counts(normalize=True).round(2).to_dict()))
    flags = pd.read_csv(env.LOCAL + "/eda_forensic_11_flags.csv")
    fl = flags[flags.set == "train"].merge(lab[["row_id", "farm", "day"]], on="row_id")
    fdays = set(map(tuple, fl[fl.Vany.astype(bool)][["farm", "day"]].drop_duplicates().values))
    q4set = set(map(tuple, q4d[["farm", "day"]].values))
    print("days with any rule-flagged row: %d | of those in Q4: %d (%.0f%%)"
          % (len(fdays), len(fdays & q4set), 100 * len(fdays & q4set) / max(1, len(fdays))))
    tt = t.merge(day[["farm", "day", "q"]], on=["farm", "day"])
    Fd = F.loc[a.split == "train"].reset_index(drop=True)
    Fd["farm"], Fd["day"] = a[a.split == "train"].farm.values, a[a.split == "train"].day.values
    Fd = Fd.merge(day[["farm", "day", "q"]], on=["farm", "day"])
    Ft = F.loc[a.split == "test"]
    print("\nshape feature medians of |value|: Q4 train-like / Q1-3 / test")
    for c in ["in_temp_ad1", "in_temp_d2", "in_hum_ad1", "in_hum_d2", "in_co2_ad1", "in_co2_d2", "co2_d1_no_supply", "rh_vs_temp"]:
        q = Fd[Fd.q == "Q4 train-like"][c].abs().median()
        r = Fd[Fd.q != "Q4 train-like"][c].abs().median()
        s = Ft[c].abs().median()
        print("  %-18s %8.3f %8.3f %8.3f" % (c, q, r, s))
    print("\nmost train-like 12 days:")
    print(day.sort_values("p", ascending=False).head(12)[["farm", "day", "p", "res", "ares"]].round(3).to_string(index=False))
    day[["farm", "day", "p", "q"]].rename(columns={"p": "p_oof"}).to_csv(env.LOCAL + "/anal_q3f_dayscore.csv", index=False)
    tp = a[a.split == "test"].groupby(["farm", "day"]).p_oof.mean()
    print("\ntest days' train-likeness: mean %.3f | training days mean %.3f | Q4 threshold %.3f"
          % (tp.mean(), day.p.mean(), day.p.quantile(0.75)))


if __name__ == "__main__":
    main()
