# -*- coding: utf-8 -*-
"""Audit 1 (methodology): out-of-fold predictions of the four submitted
temperature configurations on many candidate validators, so the claimed
"calibration" of EXT10 can be compared with its rivals and given error bars.

  R1  LightGBM huber, 98-column features_v2 view (single seed 7)
  R2  0.65 LGB + 0.25 Ridge + 0.10 Nystroem on 93 columns
  R3  round-3 blend (linear physics baseline + LGB residual, Ridge, Nystroem)
  HA  R3 with cold hinges in the baseline (cold_v5b H_all); R4 = 0.5 R3 + 0.5 HA

Validators: EXT<th for th in 8, 9, 10, 11, 12 (single fold each), geometry A,
geometry B, and the 10-fold diagnostic layout of anal_q1_errors.
Saves local/audit1_calib_oof.npz.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import time

import env  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

import cold_v5
cold_v5.DET = dict(deterministic=True, force_col_wise=True, n_jobs=2, verbose=-1)
from common import split_mask, TARGET_FARMS
from harness import load, views, folds
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from cold_v5 import lgbh, ridge, nys, add_hinges
from anal_q1_errors import diag_folds


def main():
    t0 = time.time()
    panel, lab0, _ = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    lab = (lab0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
               .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left"))
    hg = add_hinges(lab)
    lab = pd.concat([lab, hg], axis=1).copy()
    f98 = v["temp"]
    f93 = T74.base74(f98) + list(blocks["dew"]) + list(blocks["event"])
    ct = f93 + F4.names(sg) + F4.names(fp)
    phc = F4.names(ph)
    hall = [c for c in hg.columns if c != "hg_heat_cold"]

    def fit_fold(tr, va):
        y = tr.sub_temp.values
        o = {}
        o["R1"] = lgbh().fit(tr[f98], y).predict(va[f98])
        o["R2"] = (0.65 * lgbh().fit(tr[f93], y).predict(va[f93])
                   + 0.25 * ridge().fit(tr[f93], y).predict(va[f93])
                   + 0.10 * nys().fit(tr[f93], y).predict(va[f93]))
        rg = ridge().fit(tr[ct], y).predict(va[ct])
        ny = nys().fit(tr[ct], y).predict(va[ct])
        for nm, bc in (("R3", phc), ("HA", phc + hall)):
            imp = SimpleImputer(strategy="median").fit(tr[bc])
            b = LinearRegression().fit(imp.transform(tr[bc]), y)
            btr, bva = b.predict(imp.transform(tr[bc])), b.predict(imp.transform(va[bc]))
            r = lgbh().fit(tr[ct], y - btr).predict(va[ct])
            o[nm] = 0.65 * (bva + r) + 0.25 * rg + 0.10 * ny
        return o

    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = []
    for th in (8.0, 9.0, 10.0, 11.0, 12.0):
        cd = dmin[dmin < th]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))
    sets += [("gA", folds("A")), ("gB", folds("B")), ("DIAG", diag_folds(lab))]
    out = {"row_id": lab.row_id.values}
    for sname, fds in sets:
        acc = {}
        for fd in fds:
            trm, vam = split_mask(lab, fd)
            idx = np.where(vam)[0]
            for k, p in fit_fold(lab[trm], lab[vam]).items():
                acc.setdefault(k, np.full(len(lab), np.nan))[idx] = p
        for k, p in acc.items():
            out["%s|%s" % (sname, k)] = p
        print("  %s done (%d folds) %.0fs" % (sname, len(fds), time.time() - t0), flush=True)
        np.savez(env.LOCAL + "/audit1_calib_oof.npz", **out)
    print("saved")


if __name__ == "__main__":
    main()
