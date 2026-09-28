# -*- coding: utf-8 -*-
"""EC: the label looks like UNCOMPENSATED conductivity.

Within a day, log(sub_ec) moves with sub_temp at 0.0204 per C (the textbook
~2 %/C of uncompensated EC); per-day correlation median 0.82, positive on 92%
of days.  The test period is colder, so the measured EC should be lower by
~2 % per degree - something a tree model cannot extrapolate.

Recipe
  training target  ec25 = sub_ec / (1 + a (sub_temp - 25))    (a = 0.02; the
                   training rows' own labels, a transformation of the target)
  model            round-3 EC pipeline on ec25 (same features, shrink, clip
                   applied on the measured scale at the end)
  back-transform   ec = ec25_hat * (1 + a (T_hat - 25)), T_hat = the
                   temperature model's prediction for the row (out-of-fold
                   here; the submitted temperature at test time), i.e. built
                   from inputs only.
Variants: a in {0.02}; T_hat = temperature OOF vs true sub_temp (oracle, to
see how much the temperature error costs).

Folds: DIAG10 (temperature OOF = eval_v6 F60ND__DIAG10) and geometry A (the EC
validator; temperature OOF computed here on the same A folds with the
round-5 temperature members and weights).  A folds overlap: scored per fold.

Run:  cd research && PYTHONPATH="" <python> -u ec_tempcomp_v1.py
"""
import env  # noqa: F401
import numpy as np

from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load, folds, views
import features_v4 as F4
import feat_temp74 as T74
import feat_new
from anal_q1_errors import diag_folds
from ec_v6 import et, ltw, mlp
from make_submission_v3 import causal_shrink
from screen_v6 import temp_members, boot
import train_flags_v6 as TF

A = 0.02


def ec_fit_predict(tr, va, c_et, c_rest, target):
    y = tr[target].values
    return (0.60 * et().fit(tr[c_et], y).predict(va[c_et])
            + 0.30 * ltw().fit(tr[c_rest], y).predict(va[c_rest])
            + 0.10 * mlp().fit(tr[c_rest], y).predict(va[c_rest]))


def finish(p, va):
    return np.clip(causal_shrink(p, va, 0.5), 0.062, 3.46)


def main():
    panel, lab_t0, lab_e0 = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    fpc = F4.names(fp)
    lab = (lab_e0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
                 .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left")).copy()
    assert (lab.row_id.values == lab_t0.row_id.values).all()
    ct = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"]) + F4.names(sg) + fpc
    phc = F4.names(ph)
    wT = TF.row_weights(lab, 0.2, w_noisy=0.2)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + fpc
    lab["ec25"] = lab.sub_ec / (1 + A * (lab.sub_temp - 25))
    y = lab.sub_ec.values

    z = np.load(env.LOCAL + "/eval_v6_oof.npz", allow_pickle=True)
    tdiag = z["F60ND__DIAG10"]
    zb = np.load(env.LOCAL + "/oof_ec_diag.npz", allow_pickle=True)
    assert (zb["row_id"] == lab.row_id.values).all()

    for vset, fds in (("DIAG10", diag_folds(lab)), ("A", folds("A"))):
        per = []
        pooled = {k: np.full(len(lab), np.nan) for k in ("base", "comp", "oracle")}
        for i, fd in enumerate(fds):
            trm, vam = split_mask(lab, fd)
            tr, va = lab[trm], lab[vam].reset_index(drop=True)
            idx = np.where(vam)[0]
            if vset == "DIAG10":
                th = tdiag[idx]
                pb = zb["oof"][idx]
            else:
                M = temp_members(tr, va, ct, phc, wT[trm])
                th = 0.65 * M["res"] + 0.25 * M["ridge"] + 0.10 * M["nys"]
                pb = finish(ec_fit_predict(tr, va, c_et, f14, "sub_ec"), va)
            p25 = ec_fit_predict(tr, va, c_et, f14, "ec25")
            pc = finish(p25 * (1 + A * (th - 25)), va)
            po = finish(p25 * (1 + A * (va.sub_temp.values - 25)), va)
            yy = y[idx]
            per.append((rmse(pb, yy), rmse(pc, yy), rmse(po, yy)))
            pooled["base"][idx], pooled["comp"][idx], pooled["oracle"][idx] = pb, pc, po
            print("  %s fold %d: base %.4f | comp %.4f | oracle-T %.4f" % ((vset, i) + per[-1]), flush=True)
        P = np.array(per)
        print("\n== %s ==  mean of folds: base %.4f | comp %.4f (%+.1f%%, better %d/%d) | oracle-T %.4f"
              % (vset, P[:, 0].mean(), P[:, 1].mean(), 100 * (P[:, 1].mean() / P[:, 0].mean() - 1),
                 int((P[:, 1] < P[:, 0]).sum()), len(P), P[:, 2].mean()))
        if vset == "DIAG10":
            g = ~np.isnan(pooled["comp"])
            pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_ec", pooled["base"][g], pooled["comp"][g])
            print("   pooled delta %+.4f [%+.4f, %+.4f] P(worse)=%.3f" % (pr, lo, hi, pw))
            cold = (lab.groupby(["farm", "day"]).ph_in_temp_3.transform("min") < 10).values & g
            for nm, m in (("cold days (<10C)", cold), ("pass 2", (lab.day >= 179).values & g)):
                print("   %-18s base %.4f comp %.4f" % (nm, rmse(pooled["base"][m], y[m]), rmse(pooled["comp"][m], y[m])))
            np.savez(env.LOCAL + "/ec_tempcomp_v1_diag.npz", row_id=lab.row_id.values, **pooled)


if __name__ == "__main__":
    main()
