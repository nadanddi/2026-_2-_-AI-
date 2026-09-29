# -*- coding: utf-8 -*-
"""Shape of the remaining sub_temp error.

Pure first-order physics tops out near 1.0 on the geometry folds and the
current GBM blend sits at 0.82, so the leader's 0.4985 is not explained by a
missing low-pass filter.  Which kind of error is left decides which tool can
reach it:

  * day-level, persistent across days -> a slowly drifting slab offset
    (irrigation water temperature, sensor placement).  Inputs may not carry it.
  * within-day, hour-structured       -> dynamics the features still miss.
  * farm-specific                     -> separate models per greenhouse.

Run:  cd research && PYTHONPATH="" <python> -u struct_resid.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb

from harness import load, views, score
import feat_temp74 as T74
import feat_new
from common import rmse

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)


def LGBH(s):
    return lgb.LGBMRegressor(random_state=s, **DET, **T_HUB)


def main():
    panel, lab0, _ = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    lab = lab0.merge(ex, on="row_id", how="left")
    cols = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    y = lab.sub_temp.values

    for kind in ("A", "B"):
        (r, sd, per), oof = score(lab, "sub_temp", cols, LGBH, kind=kind,
                                  seeds=(7,), return_oof=True)
        g = ~np.isnan(oof)
        d = pd.DataFrame({"farm": lab.farm.values[g], "day": lab.day.values[g],
                          "hour": lab.hour.values[g], "y": y[g], "p": oof[g]})
        d["e"] = d.p - d.y
        print("\n######## 배치 %s | LGB 93f seed7 RMSE %.4f ########" % (kind, r))

        # 1. day-level vs within-day
        dm = d.groupby(["farm", "day"]).e.transform("mean")
        lvl = float(np.sqrt((dm ** 2).mean()))
        wit = float(np.sqrt(((d.e - dm) ** 2).mean()))
        print("  일수준 오차 %.4f | 일내 오차 %.4f | 일수준 SSE 비중 %.1f%%"
              % (lvl, wit, 100 * lvl ** 2 / (lvl ** 2 + wit ** 2)))

        # 2. per farm
        for f, s in d.groupby("farm"):
            print("  %s  RMSE %.4f  편향 %+.4f" % (f, np.sqrt((s.e ** 2).mean()), s.e.mean()))

        # 3. hour of day
        h = d.groupby("hour").e.agg(lambda s: float(np.sqrt((s ** 2).mean())))
        print("  시간대 RMSE 최소 %.3f(%d시) 최대 %.3f(%d시)"
              % (h.min(), h.idxmin(), h.max(), h.idxmax()))
        print("  시간대별: " + " ".join("%d:%.2f" % (k, v) for k, v in h.items() if k % 3 == 0))

        # 4. persistence of the day-level error across days
        de = d.groupby(["farm", "day"]).e.mean().reset_index()
        for lag in (1, 2, 3, 5):
            pairs = []
            for f, s in de.groupby("farm"):
                s = s.set_index("day").e
                sh = s.reindex(s.index + lag)
                ok = sh.notna().values
                pairs += list(zip(s.values[ok], sh.values[ok]))
            if len(pairs) > 10:
                a, b = np.array(pairs).T
                print("  일수준 오차 자기상관 lag%d: %.3f (n=%d)"
                      % (lag, float(np.corrcoef(a, b)[0, 1]), len(pairs)))

        # 5. error vs level: bias at the extremes?
        q = pd.qcut(d.y, 5, labels=False)
        bq = d.groupby(q).e.mean()
        print("  정답 5분위별 평균 편향: " + " ".join("%+.3f" % x for x in bq.values))


if __name__ == "__main__":
    main()
