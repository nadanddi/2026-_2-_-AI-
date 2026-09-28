# -*- coding: utf-8 -*-
"""sub_ec: does dropping `day` survive the full submission pipeline?

Calendar agent (deep_cal_*): every record has two passes.  Days <= 178 lay
the calendar out in order with two sources per date; days >= 179 re-lay the
leftover days, again in calendar order.  ALL test days are in the second pass,
and their real calendar position is much earlier than `day` says.  On folds
cut from labelled second-pass days (SP2: 4-day chunks, nearest-label gap 4.8 d
vs 3.6 d in the test) removing `day` from the ExtraTrees member cut RMSE by
31.5% (8/9 folds, CI [-0.149,-0.083]); on the usual A/B folds (first pass
only) it made it 17-21% worse.  Same pattern as temperature, where the
validator that resembled the test was right and the block CV was wrong.

That was ExtraTrees alone.  Here: the real round-3 EC pipeline (ET + LGB
tweedie + MLP, within-day shrink, clip), with `day` removed from ET only or
from all members, on SP2, SP, A and B.

Run:  cd research && PYTHONPATH="" <python> -u ec_v6.py
"""
import env  # noqa: F401
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load, folds
import fp_features as FP
from deep_cal_16 import sp2_folds
from deep_cal_eval import sp_folds
from make_submission_v3 import causal_shrink
from feat_lib import paired_block_boot

SEED = 7
DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
ET1 = dict(n_estimators=600, max_features=1.0, min_samples_leaf=1, n_jobs=4)
LGBP = dict(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)


def et():
    return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(random_state=SEED, **ET1))


def ltw():
    return lgb.LGBMRegressor(random_state=SEED, objective="tweedie", tweedie_variance_power=1.5, **DET, **LGBP)


def mlp():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3,
                                      max_iter=800, early_stopping=True, n_iter_no_change=25,
                                      validation_fraction=0.12, random_state=SEED))


def pipeline(c_et, c_rest):
    def fn(tr, va):
        y = tr.sub_ec.values
        p = (0.60 * et().fit(tr[c_et], y).predict(va[c_et])
             + 0.30 * ltw().fit(tr[c_rest], y).predict(va[c_rest])
             + 0.10 * mlp().fit(tr[c_rest], y).predict(va[c_rest]))
        return np.clip(causal_shrink(p, va, 0.5), 0.062, 3.46)
    return fn


def main():
    panel, _, lab0 = load()
    fp = FP.build()
    fpc = FP.names(fp)
    lab = lab0.merge(fp, on="row_id", how="left")
    y = lab.sub_ec.values
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    f13 = [c for c in f14 if c != "day"]

    cands = [("R3 (3회차)", pipeline(f14 + fpc, f14)),
             ("day 제거: ET만", pipeline(f13 + fpc, f14)),
             ("day 제거: 전 구성원", pipeline(f13 + fpc, f13))]
    sets = [("SP2", sp2_folds()), ("SP", sp_folds()), ("A", folds("A")), ("B", folds("B"))]

    oof = {}
    for sname, fds in sets:
        for nm, fn in cands:
            o = np.full(len(lab), np.nan)
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                if vam.sum() == 0:
                    continue
                o[np.where(vam)[0]] = fn(lab[trm], lab[vam].reset_index(drop=True))
            oof[(sname, nm)] = o
            print("  %-4s %-18s done" % (sname, nm), flush=True)

    print("\n%-20s" % "" + "".join("%10s" % s for s, _ in sets))
    for nm, _ in cands:
        line = "%-20s" % nm
        for sname, _ in sets:
            o = oof[(sname, nm)]
            g = ~np.isnan(o)
            line += "%10.4f" % rmse(o[g], y[g])
        print(line)

    print("\n== paired vs R3 ==")
    for sname, fds in sets:
        ref = oof[(sname, cands[0][0])]
        idx = [np.where(split_mask(lab, fd)[1])[0] for fd in fds]
        idx = [i for i in idx if len(i)]
        for nm, _ in cands[1:]:
            alt = oof[(sname, nm)]
            g = ~np.isnan(ref) & ~np.isnan(alt)
            d = [rmse(alt[i], y[i]) - rmse(ref[i], y[i]) for i in idx]
            sub = lab[g].reset_index(drop=True)
            pr, lo, hi, pw = paired_block_boot(sub, "sub_ec", ref[g], alt[g], n_boot=2000, seed=0, level="row")
            print("  %-4s %-18s %+6.1f%% | 폴드 %d/%d | CI [%+.4f,%+.4f] P(worse)=%.3f"
                  % (sname, nm, 100 * (rmse(alt[g], y[g]) / rmse(ref[g], y[g]) - 1),
                     sum(x < 0 for x in d), len(d), lo, hi, pw))


if __name__ == "__main__":
    main()
