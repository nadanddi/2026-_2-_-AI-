# -*- coding: utf-8 -*-
"""sub_temp: the test period is colder than the labelled data.

extrap check: 18.1% of test rows have in_temp_ewm6 below the 1st percentile
of labelled rows (test mean 13.28 vs 15.91).  Trees cannot predict beyond the
training range, so they floor out exactly where the test lives.  This would
explain why submission 2 (which added extrapolating Ridge/Nystroem members)
gained 10.5% for real but only ~4.5% on the geometry CV, whose held-out days
(63-180) are warmer.

1. COLD CV: hold out the coldest 25% labelled greenhouse-days (5 interleaved
   chunks by coldness rank, 1-day buffer).  Calibration target: does it
   reproduce the real submission-1 -> submission-2 ratio 0.895?
2. RESIDUAL TARGET: fit a linear first-order physics baseline inside each
   fold (it extrapolates), let the trees learn only sub_temp - baseline.

Run:  cd research && PYTHONPATH="" <python> -u cold_temp.py
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

import common
from common import split_mask, rmse, TARGET_FARMS
from harness import load, views, folds
import feat_temp74 as T74
import feat_new
from struct_segment import seg_features
from struct_temp import series, build as phys_build

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
PHYS = [("in_temp", None), ("in_temp", 1), ("in_temp", 3), ("in_temp", 8),
        ("in_temp", 24), ("in_temp", 72), ("rad", None), ("rad", 2), ("rad", 6),
        ("heat", 1), ("heat", 4), ("out_temp", 6), ("out_temp", 48)]


def lgbh():
    return lgb.LGBMRegressor(random_state=7, **DET, **T_HUB)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))


def nys():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Nystroem(gamma=0.005, n_components=500, random_state=7),
                         Ridge(alpha=1.0))


def cold_folds(lab, frac=0.25, k=5):
    dm = lab.groupby(["farm", "day"]).in_temp.mean().reset_index()
    dm = dm.sort_values("in_temp").reset_index(drop=True)
    cold = dm.iloc[: int(len(dm) * frac)]
    out = [{f: set() for f in TARGET_FARMS} for _ in range(k)]
    for i, r in enumerate(cold.itertuples()):
        out[i % k][r.farm].add(int(r.day))
    return out


def main():
    panel, lab0, _ = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg = seg_features()
    tX, ty, sX = common.load_raw()
    ph = phys_build(series(tX, sX), PHYS)
    phc = [c for c in ph.columns if c not in ("row_id",)]
    lab = (lab0.merge(ex, on="row_id", how="left")
               .merge(sg, on="row_id", how="left")
               .merge(ph.rename(columns={c: "ph_" + c for c in phc}), on="row_id", how="left"))
    phc = ["ph_" + c for c in phc]
    f98 = v["temp"]
    f93 = T74.base74(f98) + list(blocks["dew"]) + list(blocks["event"])
    seg = [c for c in sg.columns if c != "row_id"]
    fS = f93 + seg
    y = lab.sub_temp.values

    def fit(m, cols, tr, target):
        return m.fit(tr[cols], target)

    def methods(tr, va):
        out = {}
        out["sub1 LGB98"] = fit(lgbh(), f98, tr, tr.sub_temp).predict(va[f98])
        pl = fit(lgbh(), f93, tr, tr.sub_temp).predict(va[f93])
        pr = fit(ridge(), f93, tr, tr.sub_temp).predict(va[f93])
        pn = fit(nys(), f93, tr, tr.sub_temp).predict(va[f93])
        out["sub2 blend93"] = 0.65 * pl + 0.25 * pr + 0.10 * pn
        # physics baseline, fitted in-fold
        imp = SimpleImputer(strategy="median").fit(tr[phc])
        base = LinearRegression().fit(imp.transform(tr[phc]), tr.sub_temp.values)
        b_tr = base.predict(imp.transform(tr[phc]))
        b_va = base.predict(imp.transform(va[phc]))
        out["phys linear"] = b_va
        rl = fit(lgbh(), fS, tr, tr.sub_temp.values - b_tr).predict(va[fS])
        out["resid LGB"] = b_va + rl
        pr2 = fit(ridge(), fS, tr, tr.sub_temp).predict(va[fS])
        pn2 = fit(nys(), fS, tr, tr.sub_temp).predict(va[fS])
        out["resid blend"] = 0.65 * (b_va + rl) + 0.25 * pr2 + 0.10 * pn2
        return out

    sets = [("geometry A", folds("A")), ("geometry B", folds("B")),
            ("cold 25%", cold_folds(lab))]
    names = None
    table = {}
    for sname, fds in sets:
        oof = {}
        for fd in fds:
            trm, vam = split_mask(lab, fd)
            res = methods(lab[trm], lab[vam])
            for k, p in res.items():
                oof.setdefault(k, np.full(len(lab), np.nan))[np.where(vam)[0]] = p
            print("  %s fold done" % sname, flush=True)
        names = list(oof)
        for k, o in oof.items():
            g = ~np.isnan(o)
            table[(sname, k)] = rmse(o[g], y[g])

    print("\n%-14s" % "" + "".join("%13s" % s for s, _ in sets))
    for k in names:
        print("%-14s" % k + "".join("%13.4f" % table[(s, k)] for s, _ in sets))
    print("\n실제 1회차->2회차 비율 0.895 와 비교:")
    for s, _ in sets:
        print("  %-11s sub2/sub1 = %.3f" % (s, table[(s, "sub2 blend93")] / table[(s, "sub1 LGB98")]))


if __name__ == "__main__":
    main()
