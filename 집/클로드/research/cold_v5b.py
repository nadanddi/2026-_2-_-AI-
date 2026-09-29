# -*- coding: utf-8 -*-
"""sub_temp round 4, second pass: fair test of the cold hinge.

cold_v5.py established the EXTRAPOLATION fold as the first validator that
reproduces the leaderboard: round-3/round-2 ratio 0.858 there vs 0.844 real
(geometry CV said 0.956/0.963).  On that fold (all days colder than 10 C held
out) round 3 still over-predicts the cold rows by +0.66, and a hinge baseline
over-corrected to -0.22.  But with every day below 10 C removed, the hinge
terms have no cold rows to learn from, which is not the real situation: the
training set does have ~450 rows at 5-10 C and the test goes down to ~2 C.

Here the held-out threshold is 7 and 8 C, so part of the cold slope stays in
training, and softer variants are compared:
  R3            round-3 baseline
  H8_10         hinges at 8 and 10 C only
  H_all         hinges at 8, 10, 12 C (+ raw)
  half          0.5 * R3 + 0.5 * H_all
  H_ridge       H_all baseline fitted with Ridge (shrinks the hinge slopes)

Run:  cd research && PYTHONPATH="" <python> -u cold_v5b.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import split_mask, rmse, TARGET_FARMS
from harness import load, views, folds
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from cold_v5 import lgbh, ridge, nys, add_hinges


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
    h810 = ["hg_ewm3_8", "hg_ewm3_10", "hg_raw_8", "hg_raw_10"]
    hall = [c for c in hg.columns if c != "hg_heat_cold"]
    y_all = lab.sub_temp.values

    def resid(bcols, base_model=LinearRegression):
        def fn(tr, va):
            imp = SimpleImputer(strategy="median").fit(tr[bcols])
            bm = base_model() if base_model is LinearRegression else base_model
            b = bm.fit(imp.transform(tr[bcols]), tr.sub_temp.values)
            btr, bva = b.predict(imp.transform(tr[bcols])), b.predict(imp.transform(va[bcols]))
            y = tr.sub_temp.values
            r = lgbh().fit(tr[ct], y - btr).predict(va[ct])
            return (0.65 * (bva + r) + 0.25 * ridge().fit(tr[ct], y).predict(va[ct])
                    + 0.10 * nys().fit(tr[ct], y).predict(va[ct]))
        return fn

    def half(tr, va):
        return 0.5 * resid(phc)(tr, va) + 0.5 * resid(phc + hall)(tr, va)

    cands = [("R3", resid(phc)),
             ("H8_10", resid(phc + h810)),
             ("H_all", resid(phc + hall)),
             ("half", half),
             ("H_ridge", resid(phc + hall, make_pipeline(StandardScaler(), Ridge(alpha=30.0))))]

    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = []
    for th in (7.0, 8.0):
        cd = dmin[dmin < th]
        fd = {f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}
        trm, vam = split_mask(lab, fd)
        print("EXTRAP<%.0f: %d days out, %d rows | train min ewm3 %.2f, train rows <9C: %d | held-out min %.2f"
              % (th, len(cd), int(vam.sum()), lab.loc[trm, "ph_in_temp_3"].min(),
                 int((lab.loc[trm, "ph_in_temp_3"] < 9).sum()), lab.loc[vam, "ph_in_temp_3"].min()))
        sets.append(("EXT<%.0f" % th, [fd]))
    sets += [("geom A", folds("A")), ("geom B", folds("B"))]

    cold = lab.ph_in_temp_3.values < 8
    out = {}
    for sname, fds in sets:
        for nm, fn in cands:
            o = np.full(len(lab), np.nan)
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                o[np.where(vam)[0]] = fn(lab[trm], lab[vam])
            out[(sname, nm)] = o
            print("  %-7s %-8s done" % (sname, nm), flush=True)

    print("\n%-8s" % "" + "".join("%12s %9s %8s |" % (s, "cold<8", "bias") for s, _ in sets[:2])
          + "".join("%9s" % s for s, _ in sets[2:]))
    for nm, _ in cands:
        line = "%-8s" % nm
        for sname, _ in sets[:2]:
            o = out[(sname, nm)]
            g = ~np.isnan(o)
            gc = g & cold
            line += "%12.4f %9.4f %+8.3f |" % (rmse(o[g], y_all[g]), rmse(o[gc], y_all[gc]),
                                             float(np.mean(o[gc] - y_all[gc])))
        for sname, _ in sets[2:]:
            o = out[(sname, nm)]
            g = ~np.isnan(o)
            line += "%9.4f" % rmse(o[g], y_all[g])
        print(line)


if __name__ == "__main__":
    main()
