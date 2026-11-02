# -*- coding: utf-8 -*-
"""Q3/Q4 validation: causal calendar / true-yesterday features on light base models.

  python deep_cal_eval.py ec      ExtraTrees member of submission v4 (14 cols + fingerprint)
  python deep_cal_eval.py temp    LGB-huber surrogate (T_FAST) on the 93-col temp view

Fold sets: harness A, B (first-pass days 63..180) and SP = leave-one-second-pass-block-out
(labelled days after 178 in both records, the part of the record that looks like the test
days).  Paired greenhouse-day block bootstrap (feat_lib.paired_block_boot) for every delta.
"""
import json
import sys
import time

import env  # noqa: F401  MUST be first
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from common import USABLE, OUT_COLS, split_mask, rmse
from harness import load, views, folds
import feat_lib as L
import feat_temp74 as T74
import feat_new
import features_v4 as F4
import deep_cal_feats as DF

SP_BLOCKS = {"F13": [(179, 183), (191, 198), (211, 218), (231, 233), (241, 245)],
             "F47": [(178, 181), (189, 196), (209, 216), (229, 231), (239, 243)]}


def sp_folds():
    return [{f: set(range(SP_BLOCKS[f][i][0], SP_BLOCKS[f][i][1] + 1)) for f in SP_BLOCKS}
            for i in range(5)]


def get_folds(kind):
    return sp_folds() if kind == "SP" else folds(kind)


def score(lab, target, cols, factory, kind, seeds):
    oof = np.full(len(lab), np.nan)
    per = []
    for fd in get_folds(kind):
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam]
        ps = [np.asarray(factory(s).fit(tr[cols], tr[target].values).predict(va[cols]), float)
              for s in seeds]
        p = np.mean(ps, axis=0)
        oof[np.where(vam)[0]] = p
        per.append(rmse(p, va[target].values))
    ok = ~np.isnan(oof)
    return rmse(oof[ok], lab[target].values[ok]), per, oof


def et_fac(s):
    return make_pipeline(SimpleImputer(strategy="median"),
                         ExtraTreesRegressor(random_state=s, n_estimators=300, max_features=1.0,
                                             min_samples_leaf=1, n_jobs=2))


TEMP_FAC = L.lgbf(L.T_FAST, L.DET2)


def main(target):
    panel, lab_t0, lab_e0 = load()
    v = views(panel)
    fp = F4.fp_features()
    fpc = F4.names(fp)
    dfe = DF.build2()
    dfe["cday"] = dfe.cal2
    ycols = [c for c in dfe.columns if c.startswith("y_")] + ["pred_gap", "pred_cost", "pred_margin"]
    l1 = [c for c in dfe.columns if c.startswith("l1_")]
    l2 = [c for c in dfe.columns if c.startswith("l2_")]
    calc = ["cday", "has_twin", "second_pass"]
    allc = ycols + l1 + l2 + calc
    if target == "ec":
        lab = lab_e0.merge(fp, on="row_id", how="left")
        lab = lab.merge(DF.to_rows(dfe, lab, allc), on="row_id", how="left")
        f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
        base = f14 + fpc
        nod = [c for c in base if c != "day"]
        cands = [("+yest", base + ycols), ("+d-1", base + l1), ("+d-2", base + l2),
                 ("+cal", base + calc), ("day->cday", nod + calc),
                 ("+yest+cal", base + ycols + calc)]
        fac, seeds, tgt = et_fac, (7, 101), "sub_ec"
    else:
        ex, blocks = feat_new.build_extra()
        lab = lab_t0.merge(ex, on="row_id", how="left").merge(fp, on="row_id", how="left")
        lab = lab.merge(DF.to_rows(dfe, lab, allc), on="row_id", how="left")
        base = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
        cands = [("+yest", base + ycols), ("+d-1", base + l1), ("+d-2", base + l2),
                 ("+cal", base + calc), ("+fp", base + fpc), ("+fp+yest", base + fpc + ycols)]
        fac, seeds, tgt = TEMP_FAC, (7,), "sub_temp"
    if len(sys.argv) > 2:
        keep = sys.argv[2].split(",")
        cands = [c for c in cands if c[0] in keep]
    print("%s base %d cols, n=%d" % (target, len(base), len(lab)), flush=True)
    ref = {}
    for kind in ("A", "B", "SP"):
        r, per, oof = score(lab, tgt, base, fac, kind, seeds)
        ref[kind] = (r, per, oof)
        print("BASE %-2s %.4f  folds %s" % (kind, r, " ".join("%.4f" % x for x in per)), flush=True)
    rows = []
    for name, cs in cands:
        t0 = time.time()
        rec = dict(name=name, n=len(cs))
        line = "%-12s" % name
        for kind in ("A", "B", "SP"):
            r, per, oof = score(lab, tgt, cs, fac, kind, seeds)
            d, lo, hi, pw = L.paired_block_boot(lab, tgt, ref[kind][2], oof, n_boot=1000)
            nneg = int(sum(1 for a, b in zip(per, ref[kind][1]) if a < b))
            rec[kind] = dict(rmse=r, d=d, lo=lo, hi=hi, nneg=nneg, rel=d / ref[kind][0])
            line += " | %s %.4f %+.4f (%+.1f%%) CI[%+.4f,%+.4f] %d/5" % (kind, r, d, 100 * d / ref[kind][0], lo, hi, nneg)
        rows.append(rec)
        print(line + "  (%.0fs)" % (time.time() - t0), flush=True)
    with open(env.LOCAL + "/deep_cal_eval_%s.json" % target, "w") as fh:
        json.dump(dict(base={k: ref[k][0] for k in ref}, cands=rows), fh, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1])
