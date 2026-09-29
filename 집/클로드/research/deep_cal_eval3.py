# -*- coding: utf-8 -*-
"""SP2 folds: second-pass labelled days held out in 4-day chunks (gap to nearest label
mean 4.8 / median 4, close to the real test's 3.6 / 3).  EC ExtraTrees member and
(optionally) the temp LGB surrogate.

  python deep_cal_eval3.py ec|temp [names]
"""
import json
import sys
import time

import env  # noqa: F401
import numpy as np

from common import USABLE, OUT_COLS, split_mask, rmse
from harness import load, views
import feat_lib as L
import feat_temp74 as T74
import feat_new
import features_v4 as F4
import deep_cal_feats as DF
from deep_cal_eval import et_fac, TEMP_FAC
from deep_cal_16 import sp2_folds


def score(lab, target, cols, factory, fds, seeds):
    oof = np.full(len(lab), np.nan); per = []
    for fd in fds:
        trm, vam = split_mask(lab, fd)
        if vam.sum() == 0:
            continue
        tr, va = lab[trm], lab[vam]
        p = np.mean([np.asarray(factory(s).fit(tr[cols], tr[target].values).predict(va[cols]), float)
                     for s in seeds], axis=0)
        oof[np.where(vam)[0]] = p
        per.append(rmse(p, va[target].values))
    ok = ~np.isnan(oof)
    return rmse(oof[ok], lab[target].values[ok]), per, oof


def main(target):
    panel, lab_t0, lab_e0 = load()
    v = views(panel)
    fp = F4.fp_features(); fpc = F4.names(fp)
    dfe = DF.build2(); dfe["cday"] = dfe.cal2
    ycols = [c for c in dfe.columns if c.startswith("y_")] + ["pred_gap", "pred_cost", "pred_margin"]
    ycore = ["y_in_temp_m", "y_in_temp_min", "y_in_hum_m", "y_in_co2_m", "y_heat_m", "y_heat_hrs"]
    calc = ["cday", "has_twin", "second_pass"]
    allc = sorted(set(ycols + calc))
    if target == "ec":
        lab = lab_e0.merge(fp, on="row_id", how="left")
        f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
        base = f14 + fpc
        fac, seeds, tgt = et_fac, (7, 101), "sub_ec"
    else:
        ex, blocks = feat_new.build_extra()
        lab = lab_t0.merge(ex, on="row_id", how="left").merge(fp, on="row_id", how="left")
        base = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"]) + fpc
        fac, seeds, tgt = TEMP_FAC, (7,), "sub_temp"
    lab = lab.merge(DF.to_rows(dfe, lab, allc), on="row_id", how="left")
    nod = [c for c in base if c != "day"]
    cands = [("-day", nod), ("-day+cday", nod + ["cday"]), ("+cday", base + ["cday"]),
             ("+cal", base + calc), ("+yest", base + ycols), ("+ycore", base + ycore),
             ("-day+cday+ycore", nod + ["cday"] + ycore), ("+d-2", base + [c.replace("y_", "l2_") for c in ycore])]
    if len(sys.argv) > 2:
        keep = sys.argv[2].split(",")
        cands = [c for c in cands if c[0] in keep]
    fds = sp2_folds()
    r0, per0, oof0 = score(lab, tgt, base, fac, fds, seeds)
    print("%s BASE SP2 %.4f (%d folds)" % (target, r0, len(per0)), flush=True)
    rows = []
    for name, cs in cands:
        missing = [c for c in cs if c not in lab.columns]
        if missing:
            print("skip", name, missing[:3]); continue
        t0 = time.time()
        r, per, oof = score(lab, tgt, cs, fac, fds, seeds)
        d, lo, hi, pw = L.paired_block_boot(lab, tgt, oof0, oof, n_boot=1000)
        nneg = int(sum(1 for a, b in zip(per, per0) if a < b))
        rows.append(dict(name=name, rmse=r, d=d, lo=lo, hi=hi, nneg=nneg, nf=len(per)))
        print("%-16s SP2 %.4f %+.4f (%+.1f%%) CI[%+.4f,%+.4f] %d/%d  (%.0fs)"
              % (name, r, d, 100 * d / r0, lo, hi, nneg, len(per), time.time() - t0), flush=True)
    with open(env.LOCAL + "/deep_cal_eval3_%s.json" % target, "w") as fh:
        json.dump(dict(base=r0, cands=rows), fh, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1])
