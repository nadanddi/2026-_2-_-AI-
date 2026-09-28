# -*- coding: utf-8 -*-
"""EC error by regime, reweighted to the TEST composition (analysis only).

Sealed day (catalog 6.11): daily mean act_circfan < 10 and share of hours
with act_vent == 0 > 0.85 (full-day definition, analysis only).  Test: 20 of
60 days sealed (33%); training 59/400 (15%).  Training sealed days are mostly
noisy (6.13), test sealed days clean.

DIAG10 OOF of the round-3 raw blend (et/ltw/mlp + shrink + clip), seed 7:
  RMSE on sealed / non-sealed days, noisy vs clean sealed days,
  and the overall RMSE re-weighted to the test share of sealed days.
If the reweighted number sits near our LB 0.2134, test difficulty is driven
by sealed days, and sealed-day accuracy is where a rival could win big.

Run:  cd research && PYTHONPATH="" <python> -u anal_ec_regime.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

import common
import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink
import train_flags_v6 as TF


def sealed_days(df):
    g = df.groupby(["farm", "day"])
    s = (g.act_circfan.mean() < 10) & (g.act_vent.apply(lambda v: (v == 0).mean()) > 0.85)
    return s


def main():
    tX, ty, sX = common.load_raw()
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    f13 = [c for c in f14 if c != "day"]
    c_et = f13 + F4.names(fp)
    y = lab.sub_ec.values
    s_tr = sealed_days(tX[tX.farm.isin(TARGET_FARMS)])
    s_te = sealed_days(sX)
    print("sealed days: training %d/%d, test %d/%d" % (s_tr.sum(), len(s_tr), s_te.sum(), len(s_te)), flush=True)
    nd = TF.noisy_days()
    noisy = set(map(tuple, nd[nd.noisy][["farm", "day"]].values))
    key = list(zip(lab.farm, lab.day))
    is_s = np.array([bool(s_tr.get(k, False)) for k in key])
    is_n = np.array([k in noisy for k in key])
    ec_v6.SEED = 7
    o = np.full(len(lab), np.nan)
    for fd in diag_folds(lab):
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam].reset_index(drop=True)
        yt = tr.sub_ec.values
        raw = (0.6 * ec_v6.et().fit(tr[c_et], yt).predict(va[c_et])
               + 0.3 * ec_v6.ltw().fit(tr[f14], yt).predict(va[f14])
               + 0.1 * ec_v6.mlp().fit(tr[f14], yt).predict(va[f14]))
        o[vam] = np.clip(causal_shrink(raw, va, 0.5), 0.062, 3.46)
    e2 = (o - y) ** 2
    ms, mn = e2[is_s].mean(), e2[~is_s].mean()
    print("DIAG10 round-3 | all %.4f | sealed %.4f (n days %d) | non-sealed %.4f"
          % (np.sqrt(e2.mean()), np.sqrt(ms), len({k for k, s in zip(key, is_s) if s}), np.sqrt(mn)))
    print("  sealed & noisy %.4f | sealed & clean %.4f | non-sealed noisy %.4f | non-sealed clean %.4f"
          % tuple(np.sqrt(e2[m].mean()) if m.any() else np.nan
                  for m in (is_s & is_n, is_s & ~is_n, ~is_s & is_n, ~is_s & ~is_n)))
    for share in (s_tr.mean(), s_te.mean()):
        print("  reweighted to sealed share %.2f -> %.4f" % (share, np.sqrt(share * ms + (1 - share) * mn)))
    print("  our LB with this EC model: 0.2134")
    # bias on sealed days
    print("  mean error (pred - truth): sealed %+.3f, non-sealed %+.3f" % ((o - y)[is_s].mean(), (o - y)[~is_s].mean()))


if __name__ == "__main__":
    main()
