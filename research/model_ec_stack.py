# -*- coding: utf-8 -*-
"""sub_ec: honest stacking across capability groups.

Fixed-weight blending is only one way to combine families.  Here the weights
are fitted by non-negative least squares on ONE fold set's out-of-fold
predictions and evaluated on the OTHER fold set (and vice versa), so the
weights never see the rows they are scored on and never see the placement
they are scored on either.  If the best honest stack does not beat the best
single model by more than a fold standard deviation, the extra capabilities
add nothing.

Also prints the pairwise error correlations, which say whether a partner is
even capable of helping.

Run:  cd research && PYTHONPATH="" <python> -u model_ec_stack.py
"""
import env  # noqa: F401  MUST be first project import
import numpy as np
import lightgbm as lgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.optimize import nnls

from model_common import load, views, seg_stats
from harness import score
from common import rmse

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
ET2 = dict(n_estimators=300, max_features=1.0, min_samples_leaf=2, n_jobs=4)
LGBP = dict(n_estimators=800, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0)


def imp(f):
    return lambda s: make_pipeline(SimpleImputer(strategy="median"), f(s))


def sc(f):
    return lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                   StandardScaler(), f(s))


def main():
    panel, lab_t, lab_e = load()
    v = views(panel)
    v5, ec = v["v5"], v["ec"]
    y = lab_e["sub_ec"].values.astype(float)

    REF = imp(lambda s: ExtraTreesRegressor(random_state=s, **ET2))
    members = [
        ("T  ET300/2 v5", v5, REF, (7, 101, 2024)),
        ("P  LGBgamma v5", v5,
         lambda s: lgb.LGBMRegressor(random_state=s, objective="gamma",
                                     **DET, **LGBP), (7, 101, 2024)),
        ("TP logET300/2 v5", v5,
         lambda s: TransformedTargetRegressor(regressor=REF(s), func=np.log,
                                              inverse_func=np.exp),
         (7, 101, 2024)),
        ("H  HGBgamma v5", v5,
         lambda s: HistGradientBoostingRegressor(
             random_state=s, loss="gamma", max_iter=400, learning_rate=0.05,
             max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
             early_stopping=False), (7, 101, 2024)),
        ("XS MLP128-64 v5", v5,
         sc(lambda s: MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                   learning_rate_init=1e-3, max_iter=600,
                                   early_stopping=True, n_iter_no_change=25,
                                   validation_fraction=0.12, random_state=s)),
         (7, 101, 2024)),
        ("S  Nystroem .2/800 v5", v5,
         sc(lambda s: make_pipeline(Nystroem(gamma=0.2, n_components=800,
                                             random_state=s),
                                    Ridge(alpha=0.01))), (7, 101, 2024)),
        ("S  SVR C=30 v5", v5,
         sc(lambda s: SVR(C=30.0, epsilon=0.02, gamma="scale",
                          cache_size=700)), (7,)),
        ("XS Ridge10 v5", v5, sc(lambda s: Ridge(alpha=10.0)), (7,)),
        ("T  ET100/8 ec68", ec,
         imp(lambda s: ExtraTreesRegressor(random_state=s, n_estimators=100,
                                           max_features=1.0,
                                           min_samples_leaf=8, n_jobs=4)),
         (7, 101, 2024)),
    ]

    O = {"A": {}, "B": {}}
    print("%-24s %8s %8s | %8s %8s" % ("member", "A rmse", "B rmse",
                                       "hiBiasA", "hiBiasB"))
    for nm, cols, fac, seeds in members:
        row = [nm]
        for kind in ("A", "B"):
            (r, sd, per), oof = score(lab_e, "sub_ec", cols, fac, kind=kind,
                                      seeds=seeds, return_oof=True)
            O[kind][nm] = oof
            row.append((r, sd, oof))
        a, b = row[1], row[2]
        ga, gb = ~np.isnan(a[2]), ~np.isnan(b[2])
        sa = seg_stats(y[ga], a[2][ga])
        sb = seg_stats(y[gb], b[2][gb])
        print("%-24s %8.4f %8.4f | %+8.4f %+8.4f   (sd %.3f/%.3f)"
              % (nm, a[0], b[0], sa["bias"], sb["bias"], a[1], b[1]))

    names = [m[0] for m in members]
    print("\nerror correlation with the reference (folds A):")
    ga = ~np.isnan(O["A"][names[0]])
    e0 = O["A"][names[0]][ga] - y[ga]
    for nm in names[1:]:
        e = O["A"][nm][ga] - y[ga]
        print("  %-24s r=%.3f" % (nm, float(np.corrcoef(e0, e)[0, 1])))

    print("\n--- honest NNLS stack (weights fitted on one fold set, "
          "scored on the other) ---")
    for src, dst in (("A", "B"), ("B", "A")):
        gs = ~np.isnan(O[src][names[0]])
        gd = ~np.isnan(O[dst][names[0]])
        Xs = np.column_stack([O[src][n][gs] for n in names])
        Xd = np.column_stack([O[dst][n][gd] for n in names])
        w, _ = nnls(Xs, y[gs])
        p = Xd @ w
        s0 = seg_stats(y[gd], O[dst][names[0]][gd])
        s1 = seg_stats(y[gd], p)
        print("  fit %s -> score %s : reference %.4f | stack %.4f | "
              "hi-bias %+.4f -> %+.4f" %
              (src, dst, rmse(O[dst][names[0]][gd], y[gd]), rmse(p, y[gd]),
               s0["bias"], s1["bias"]))
        print("      weights " + "  ".join("%s=%.2f" % (n.split()[1], wi)
                                           for n, wi in zip(names, w) if wi > 1e-3))


if __name__ == "__main__":
    main()
