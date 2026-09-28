# -*- coding: utf-8 -*-
"""Does the EC TabPFN member break under cold EXTRAPOLATION like the
temperature member did (catalog 6.67)?  EC has only been validated on A, B
and DIAG10, where cold days keep warm neighbours in training.

EXT10 / EXT12 for EC (same cold-day sets as temperature: days whose minimum
3-h smoothed indoor temperature is below 10 / 12 C, held out at once).
Round-3 raw (seed 7) + TabPFN bag (2000 x 4, samples 1-4), shrink + clip.
Reported (analysis; any adoption would need its own pre-set test):
  round-3, TabPFN alone, candidate 0.8/0.2, and a gated candidate whose
  TabPFN share is x clip((in_temp - 8)/2, 0, 1) (the temperature gate),
overall and by in_temp band.

Run:  cd research && PYTHONPATH="" <python> -u ec_ext_check_gpu.py
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np
import torch

import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load
import features_v4 as F4
from make_submission_v3 import causal_shrink
from anal_ec_noncausal_tabpfn import tp

assert torch.cuda.is_available()


def main():
    _, _, lab0 = load()
    fp = F4.fp_features()
    ph = F4.phys_features()
    lab = lab0.merge(fp, on="row_id", how="left").merge(ph[["row_id", "ph_in_temp_3"]], on="row_id", how="left")
    lab = lab.reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    y = lab.sub_ec.values
    X = lab[c_et].values.astype(np.float32)
    t = lab.in_temp.values
    gate = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    ec_v6.SEED = 7
    for th in (10, 12):
        cd = dmin[dmin < float(th)]
        fd = {f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam].reset_index(drop=True)
        yt = tr.sub_ec.values
        raw = (0.6 * ec_v6.et().fit(tr[c_et], yt).predict(va[c_et])
               + 0.3 * ec_v6.ltw().fit(tr[f14], yt).predict(va[f14])
               + 0.1 * ec_v6.mlp().fit(tr[f14], yt).predict(va[f14]))
        bag = np.mean([tp(X[trm], y[trm], X[vam], s) for s in (1, 2, 3, 4)], axis=0)
        gv = gate[vam]
        P = {"round3": raw, "cand": 0.8 * raw + 0.2 * bag, "gated": (1 - 0.2 * gv) * raw + 0.2 * gv * bag}
        Q = {k: np.clip(causal_shrink(p, va, 0.5), 0.062, 3.46) for k, p in P.items()}
        Q["PFN_alone"] = bag
        yv, tv = y[vam], t[vam]
        print("EC EXT%d (%d rows) | %s" % (th, len(yv), " | ".join("%s %.4f" % (k, rmse(q, yv)) for k, q in Q.items())),
              flush=True)
        for lo, hi in ((-99, 8), (8, 10), (10, 99)):
            m = (tv > lo) & (tv <= hi)
            print("   in_temp (%g,%g] n=%d | %s" % (lo, hi, m.sum(), " | ".join(
                "%s %.4f (bias %+.3f)" % (k, rmse(q[m], yv[m]), (q - yv)[m].mean()) for k, q in Q.items())), flush=True)


if __name__ == "__main__":
    main()
