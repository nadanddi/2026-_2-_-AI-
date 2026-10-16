# -*- coding: utf-8 -*-
"""sub_temp round 4: how far to bend the cold end of the physics baseline.

Real scores confirmed the cold hypothesis: round 3 (physics-linear baseline,
extrapolates into the cold) beat round 2 by 15.6% on temperature while the
geometry CV predicted ~4%.  Training labels say the slab runs further below
the air the colder it gets (slab - ewm3 air: >=10 C -0.6..-0.8, 6-8 C -1.17
n=51, 0-6 C -1.72 n=5), but round 3 still predicts only -0.48 on the coldest
test rows.

1. EXTRAPOLATION fold: hold out every greenhouse-day whose coldest hour
   (ewm3 of in_temp) is below THRESH, all at once, so the model must predict
   below its training range -- the situation of the test.  Calibration: the
   round-2 -> round-3 configuration change should show a large gain here
   (real ratio 0.5624 / 0.6666 = 0.844).
2. Candidate baselines: the round-3 linear baseline plus cold hinge terms
   max(0, k - x) on the smoothed air temperature, and a heating x cold
   interaction.  Trees keep learning the residual.

Run:  cd research && PYTHONPATH="" <python> -u cold_v5.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import split_mask, rmse, TARGET_FARMS
from harness import load, views, folds
import feat_temp74 as T74
import feat_new
import features_v4 as F4

SEED = 7
DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
THRESH = 10.0


def lgbh():
    return lgb.LGBMRegressor(random_state=SEED, **DET, **T_HUB)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))


def nys():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Nystroem(gamma=0.005, n_components=500, random_state=SEED), Ridge(alpha=1.0))


def add_hinges(df):
    x3, x0 = df["ph_in_temp_3"], df["ph_in_temp_None"]
    h = pd.DataFrame(index=df.index)
    for k in (8, 10, 12):
        h["hg_ewm3_%d" % k] = np.maximum(0.0, k - x3)
        h["hg_raw_%d" % k] = np.maximum(0.0, k - x0)
    h["hg_heat_cold"] = df["ph_heat_4"].fillna(0) * np.maximum(0.0, 10 - x3)
    return h


def main():
    panel, lab0, _ = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    lab = (lab0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
               .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left"))
    hg = add_hinges(lab)
    lab = pd.concat([lab, hg], axis=1)
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    ct = f93 + F4.names(sg) + F4.names(fp)
    phc = F4.names(ph)
    hgc = list(hg.columns)
    y_all = lab.sub_temp.values

    def blend_r2(tr, va):
        y = tr.sub_temp.values
        return (0.65 * lgbh().fit(tr[f93], y).predict(va[f93])
                + 0.25 * ridge().fit(tr[f93], y).predict(va[f93])
                + 0.10 * nys().fit(tr[f93], y).predict(va[f93]))

    def resid(bcols, feats=ct):
        def fn(tr, va):
            imp = SimpleImputer(strategy="median").fit(tr[bcols])
            b = LinearRegression().fit(imp.transform(tr[bcols]), tr.sub_temp.values)
            btr, bva = b.predict(imp.transform(tr[bcols])), b.predict(imp.transform(va[bcols]))
            y = tr.sub_temp.values
            r = lgbh().fit(tr[feats], y - btr).predict(va[feats])
            return (0.65 * (bva + r) + 0.25 * ridge().fit(tr[feats], y).predict(va[feats])
                    + 0.10 * nys().fit(tr[feats], y).predict(va[feats]))
        return fn

    cands = [("R2 구성 (2회차)", blend_r2),
             ("R3 구성 (3회차)", resid(phc)),
             ("R3 + 한랭 힌지", resid(phc + hgc[:-1])),
             ("R3 + 한랭 힌지 + 난방교호", resid(phc + hgc))]

    # extrapolation fold: all greenhouse-days whose coldest ewm3 < THRESH
    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    cold_days = dmin[dmin < THRESH]
    ext = [{f: set(int(d) for (ff, d) in cold_days.index if ff == f) for f in TARGET_FARMS}]
    trm, vam = split_mask(lab, ext[0])
    print("EXTRAP fold: %d greenhouse-days held out (coldest hour < %.0f C), %d rows; "
          "training min ewm3 %.2f vs held-out min %.2f"
          % (len(cold_days), THRESH, int(vam.sum()),
             lab.loc[trm, "ph_in_temp_3"].min(), lab.loc[vam, "ph_in_temp_3"].min()))

    res = {}
    for sname, fds in (("geometry A", folds("A")), ("geometry B", folds("B")), ("EXTRAP", ext)):
        for nm, fn in cands:
            o = np.full(len(lab), np.nan)
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                o[np.where(vam)[0]] = fn(lab[trm], lab[vam])
            res[(sname, nm)] = o
            print("  %-10s %-28s done" % (sname, nm), flush=True)

    print("\n%-28s %11s %11s %11s %13s %9s" % ("", "geometry A", "geometry B", "EXTRAP", "EXTRAP x<9C", "bias<9C"))
    cold9 = lab.ph_in_temp_3.values < 9
    for nm, _ in cands:
        row = []
        for sname in ("geometry A", "geometry B", "EXTRAP"):
            o = res[(sname, nm)]
            g = ~np.isnan(o)
            row.append(rmse(o[g], y_all[g]))
        o = res[("EXTRAP", nm)]
        g = ~np.isnan(o) & cold9
        row += [rmse(o[g], y_all[g]), float(np.mean(o[g] - y_all[g]))]
        print("%-28s %11.4f %11.4f %11.4f %13.4f %+9.3f" % (nm, *row))
    r2 = res[("EXTRAP", cands[0][0])]; r3 = res[("EXTRAP", cands[1][0])]
    g = ~np.isnan(r2)
    print("\nEXTRAP R3/R2 = %.3f  (real 0.844)" % (rmse(r3[g], y_all[g]) / rmse(r2[g], y_all[g])))
    for kind in ("geometry A", "geometry B"):
        a, b = res[(kind, cands[0][0])], res[(kind, cands[1][0])]
        g = ~np.isnan(a)
        print("%s R3/R2 = %.3f" % (kind, rmse(b[g], y_all[g]) / rmse(a[g], y_all[g])))


if __name__ == "__main__":
    main()
