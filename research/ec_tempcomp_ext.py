# -*- coding: utf-8 -*-
"""EC temperature compensation (ec_tempcomp_v1.py) on the extrapolation folds:
all days whose 3-h air EWM drops below 8/10/12 C held out at once, the way the
colder test is.  Temperature for the back-transform = eval_v6 F60ND__EXT* OOF
(the round-5 temperature model trained on the same folds).

Run:  cd research && PYTHONPATH="" <python> -u ec_tempcomp_ext.py
"""
import env  # noqa: F401
import numpy as np

from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load
import features_v4 as F4
from screen_v6 import boot
from ec_tempcomp_v1 import ec_fit_predict, finish, A


def main():
    _, _, lab0 = load()
    fp, ph = F4.fp_features(), F4.phys_features()
    lab = lab0.merge(fp, on="row_id", how="left").merge(ph, on="row_id", how="left").copy()
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    lab["ec25"] = lab.sub_ec / (1 + A * (lab.sub_temp - 25))
    z = np.load(env.LOCAL + "/eval_v6_oof.npz", allow_pickle=True)
    assert (z["row_id"] == lab.row_id.values).all()
    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    y = lab.sub_ec.values
    for th in (8.0, 10.0, 12.0):
        cd = dmin[dmin < th]
        fd = {f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam].reset_index(drop=True)
        th_hat = z["F60ND__EXT%d" % th][vam]
        pb = finish(ec_fit_predict(tr, va, c_et, f14, "sub_ec"), va)
        p25 = ec_fit_predict(tr, va, c_et, f14, "ec25")
        pc = finish(p25 * (1 + A * (th_hat - 25)), va)
        yy = y[vam]
        pr, lo, hi, pw = boot(va, "sub_ec", pb, pc)
        print("EXT%-3d days %3d | base %.4f (bias %+.3f) | comp %.4f (bias %+.3f) | %+.1f%% [%+.4f, %+.4f] P(worse)=%.3f"
              % (th, len(cd), rmse(pb, yy), (pb - yy).mean(), rmse(pc, yy), (pc - yy).mean(),
                 100 * (rmse(pc, yy) / rmse(pb, yy) - 1), lo, hi, pw), flush=True)


if __name__ == "__main__":
    main()
