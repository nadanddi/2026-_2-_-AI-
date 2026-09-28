# -*- coding: utf-8 -*-
"""Analysis Q2b: can the day-level temperature error be explained by the day's
own inputs, and how big is the date-shared component?

anal_q2_sources.py: day-level residuals are only weakly tied to the source
chain (13% of variance) and slightly more to the calendar date (other source,
same date: r = 0.20).  If a day's own input summary explains a real part of
its day-level residual, a feature is missing; if not, that part is a floor for
input-only models.

  date component: share of day-level residual variance explained by the
                  calendar date (all sources on that date, both records).
  input model:    ridge + gradient boosting on day-level input summaries,
                  scored by leave-date-out cross-validation (so the date's
                  twins never train the model that scores it).

Analysis only (the target is the residual itself).

Run:  cd research && PYTHONPATH="" <python> anal_q2b_daylevel.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
from harness import load
from cleanw_v6 import weights

ACTS = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]


def main():
    panel, lab, _ = load()
    tX, ty, sX = common.load_raw()
    z = np.load(env.LOCAL + "/oof_temp_diag.npz", allow_pickle=True)
    clean = weights(lab, 3, 0.0) >= 1
    y = lab.sub_temp.values
    e = z["r3"] - y
    g = ~np.isnan(e) & clean
    dres = (pd.DataFrame({"farm": lab.farm.values[g], "day": lab.day.values[g], "e": e[g]})
              .groupby(["farm", "day"]).e.agg(res="mean", n="size").reset_index())
    dres = dres[dres.n >= 12]

    a = pd.concat([tX, sX]).query("farm in ['F13','F47']").sort_values(["farm", "t"])
    grp = a.groupby(["farm", "day"])
    feats = pd.DataFrame({
        "in_temp_mean": grp.in_temp.mean(), "in_temp_min": grp.in_temp.min(), "in_temp_max": grp.in_temp.max(),
        "in_temp_h0": grp.in_temp.first(), "in_hum_mean": grp.in_hum.mean(), "in_co2_mean": grp.in_co2.mean(),
        "in_co2_min": grp.in_co2.min(), "out_temp_mean": grp.out_temp.mean(), "out_rad_sum": grp.out_rad.sum(),
        "out_wspd_mean": grp.out_wspd.mean(), "heat_hours": grp.act_heating.apply(lambda s: float((s > 50).sum())),
    })
    for c in ACTS:
        feats[c + "_mean"] = grp[c].mean()
    last = a.groupby("farm").in_temp.shift(1)
    a["jump"] = (a.in_temp - last).where(a.hour == 0)
    feats["midnight_jump"] = a.groupby(["farm", "day"]).jump.max()
    feats = feats.reset_index()
    cal = pd.read_csv(env.LOCAL + "/deep_cal_11_days.csv")[["farm", "day", "cal"]]
    D = dres.merge(feats, on=["farm", "day"]).merge(cal, on=["farm", "day"], how="left")
    D["f47"] = (D.farm == "F47").astype(float)
    X = D.drop(columns=["farm", "day", "res", "n", "cal"])
    yv = D.res.values
    print("days: %d | day-level residual sd %.3f" % (len(D), yv.std()))

    dm = D.groupby("cal").res.transform("mean")
    multi = D.groupby("cal").res.transform("size") >= 2
    v = D[multi]
    share = 1 - float(((v.res - dm[multi]) ** 2).sum()) / float(((v.res - v.res.mean()) ** 2).sum())
    print("calendar-date component: %.1f%% of day-level variance (dates with >=2 scored days, %d days)"
          % (100 * share, int(multi.sum())))

    gkf = GroupKFold(n_splits=10)
    groups = D.cal.fillna(-1).values
    for nm, mk in (("ridge", lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                   RidgeCV(alphas=np.logspace(-1, 3, 20)))),
                   ("gbm", lambda: make_pipeline(SimpleImputer(strategy="median"),
                                                 GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                                                           learning_rate=0.05, subsample=0.8,
                                                                           random_state=0)))):
        oof = np.zeros(len(D))
        for tr, te in gkf.split(X, yv, groups):
            oof[te] = mk().fit(X.iloc[tr], yv[tr]).predict(X.iloc[te])
        r2 = 1 - float(((yv - oof) ** 2).sum()) / float(((yv - yv.mean()) ** 2).sum())
        print("inputs -> day-level residual, leave-date-out R2 (%s): %.3f" % (nm, r2))
        if nm == "gbm":
            m = mk().fit(X, yv)
            imp = pd.Series(m.steps[-1][1].feature_importances_, index=X.columns).sort_values(ascending=False)
            print("  top inputs:", ", ".join("%s %.2f" % (k, v) for k, v in imp.head(6).items()))
    c = D[X.columns].corrwith(D.res).sort_values(key=np.abs, ascending=False)
    print("strongest single correlations with day-level residual:")
    print("  " + ", ".join("%s %+.2f" % (k, v) for k, v in c.head(8).items()))


if __name__ == "__main__":
    main()
