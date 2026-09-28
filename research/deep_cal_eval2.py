# -*- coding: utf-8 -*-
"""EC follow-up: is the SP gain from the yesterday/calendar content, or from diluting `day`?

  python deep_cal_eval2.py [names]
"""
import json
import sys
import time

import env  # noqa: F401
import numpy as np

from common import USABLE, OUT_COLS
from harness import load
import feat_lib as L
import features_v4 as F4
import deep_cal_feats as DF
from deep_cal_eval import score, et_fac


def main():
    panel, lab_t0, lab_e0 = load()
    fp = F4.fp_features(); fpc = F4.names(fp)
    dfe = DF.build2(); dfe["cday"] = dfe.cal2
    ycore = ["y_in_temp_m", "y_in_temp_min", "y_in_hum_m", "y_in_co2_m", "y_heat_m", "y_heat_hrs"]
    ycols = [c for c in dfe.columns if c.startswith("y_")] + ["pred_gap", "pred_cost", "pred_margin"]
    calc = ["cday", "has_twin", "second_pass"]
    allc = sorted(set(ycols + calc))
    lab = lab_e0.merge(fp, on="row_id", how="left")
    lab = lab.merge(DF.to_rows(dfe, lab, allc), on="row_id", how="left")
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    base = f14 + fpc
    nod = [c for c in base if c != "day"]
    cands = [("-day", nod), ("-day+cday", nod + ["cday"]), ("+cday", base + ["cday"]),
             ("+second", base + ["second_pass"]), ("-day+yest", nod + ycols),
             ("+ycore", base + ycore), ("-day+cday+ycore", nod + ["cday"] + ycore)]
    if len(sys.argv) > 1:
        keep = sys.argv[1].split(",")
        cands = [c for c in cands if c[0] in keep]
    tgt, seeds = "sub_ec", (7, 101)
    ref = {}
    for kind in ("A", "B", "SP"):
        r, per, oof = score(lab, tgt, base, et_fac, kind, seeds)
        ref[kind] = (r, per, oof)
        print("BASE %-2s %.4f" % (kind, r), flush=True)
    rows = []
    for name, cs in cands:
        t0 = time.time(); rec = dict(name=name, n=len(cs)); line = "%-16s" % name
        for kind in ("A", "B", "SP"):
            r, per, oof = score(lab, tgt, cs, et_fac, kind, seeds)
            d, lo, hi, pw = L.paired_block_boot(lab, tgt, ref[kind][2], oof, n_boot=1000)
            nneg = int(sum(1 for a, b in zip(per, ref[kind][1]) if a < b))
            rec[kind] = dict(rmse=r, d=d, lo=lo, hi=hi, nneg=nneg, per=per)
            line += " | %s %.4f %+.4f (%+.1f%%) CI[%+.4f,%+.4f] %d/5" % (kind, r, d, 100 * d / ref[kind][0], lo, hi, nneg)
            if kind == "SP":
                line += " sp-folds " + " ".join("%.3f" % x for x in per)
        rows.append(rec)
        print(line + "  (%.0fs)" % (time.time() - t0), flush=True)
    with open(env.LOCAL + "/deep_cal_eval2_ec.json", "w") as fh:
        json.dump(dict(base={k: ref[k][0] for k in ref}, base_sp_per=ref["SP"][1], cands=rows), fh, indent=1, default=float)


if __name__ == "__main__":
    main()
