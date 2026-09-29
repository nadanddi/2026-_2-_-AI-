# -*- coding: utf-8 -*-
"""sub_ec final config: 14 columns, 3-way blend, then causal within-day shrink.

Member definitions are taken VERBATIM from model_ec_mlp_v5.py (ET1 / LGBP /
MLP_BEST).  An earlier draft of this file guessed the boosting parameters from
the 68-column huber model (num_leaves 7, min_child_samples 240) and the blend
came out WORSE than ExtraTrees alone; the agent's LGBP is a much larger tree
(800 rounds, 31 leaves).  Do not re-guess these.

The shrinkage was validated on the 18-column blend (A -0.0047, B -0.0081).
The submission uses 14 columns, so it is re-measured here.

Run:  cd research && PYTHONPATH="" <python> -u ec_final.py
"""
import env  # noqa: F401  MUST be first project import
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness import load, score, views
from common import USABLE, OUT_COLS, rmse
from feat_lib import paired_block_boot

DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
ET1 = dict(n_estimators=600, max_features=1.0, min_samples_leaf=1, n_jobs=4)
LGBP = dict(n_estimators=800, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0)
SEEDS = (7, 101, 2024)


def ET(s):
    return make_pipeline(SimpleImputer(strategy="median"),
                         ExtraTreesRegressor(random_state=s, **ET1))


def LGBTW(s):
    return lgb.LGBMRegressor(random_state=s, objective="tweedie",
                             tweedie_variance_power=1.5, **DET, **LGBP)


def MLP(s):
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2,
                     learning_rate_init=1e-3, max_iter=800,
                     early_stopping=True, n_iter_no_change=25,
                     validation_fraction=0.12, random_state=s))


def causal_shrink(p, lab, L):
    d = pd.DataFrame({"f": lab.farm.values, "d": lab.day.values,
                      "h": lab.hour.values, "p": p}).sort_values(["f", "d", "h"])
    em = d.groupby(["f", "d"]).p.transform(lambda s: s.expanding().mean())
    out = np.empty(len(p))
    out[d.index.values] = (em + L * (d.p - em)).values
    return out


def main():
    panel, lab_t, lab_e = load()
    v5 = list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]
    cols14 = [c for c in v5 if c not in OUT_COLS]
    print("v5 %d cols -> 14f %d cols (dropped %s)"
          % (len(v5), len(cols14), ",".join(OUT_COLS)))
    y = lab_e["sub_ec"].values.astype(float)

    for W in ({"ET": 0.60, "LGBTW": 0.30, "MLP": 0.10},
              {"ET": 0.45, "LGBTW": 0.40, "MLP": 0.15}):
        print("\n######## weights %s ########" % W)
        for kind in ("A", "B"):
            mem = {}
            for nm, fac in (("ET", ET), ("LGBTW", LGBTW), ("MLP", MLP)):
                (r, sd, per), o = score(lab_e, "sub_ec", cols14, fac, kind=kind,
                                        seeds=SEEDS, return_oof=True)
                mem[nm] = o
                print("  [%s] %-6s rmse %.4f" % (kind, nm, r))
            blend = sum(W[k] * mem[k] for k in W)
            g = ~np.isnan(blend)
            sub = lab_e[g].reset_index(drop=True)
            et_r = rmse(mem["ET"][g], y[g])
            print("  [%s] ET alone %.4f -> blend %.4f" % (kind, et_r, rmse(blend[g], y[g])))
            pr, lo, hi, pw = paired_block_boot(sub, "sub_ec", mem["ET"][g], blend[g],
                                               n_boot=2000, seed=0, level="row")
            print("        blend vs ET : %+.4f CI [%+.4f,%+.4f] P(worse)=%.3f"
                  % (pr, lo, hi, pw))
            for L in (0.4, 0.5, 0.6):
                alt = causal_shrink(blend[g], sub, L)
                p2, l2, h2, w2 = paired_block_boot(sub, "sub_ec", blend[g], alt,
                                                   n_boot=2000, seed=0, level="row")
                print("        +shrink L=%.1f rmse %.4f | %+.4f CI [%+.4f,%+.4f] P(worse)=%.3f"
                      % (L, rmse(alt, y[g]), p2, l2, h2, w2))


if __name__ == "__main__":
    main()
