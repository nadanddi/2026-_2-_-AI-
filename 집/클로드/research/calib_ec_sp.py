# -*- coding: utf-8 -*-
"""Is the second-pass validator (SP / SP2) calibrated for sub_ec?

ec_v6.py: dropping `day` from the EC pipeline is -20% (ET only) / -33% (all
members) on the second-pass folds and +7..+27% on the first-pass A/B folds.
One of the two validator families is wrong.  Check which one reproduces the
two real EC score changes we have:

  round 1 -> round 2: 0.2442 -> 0.2287  ratio 0.937
  round 2 -> round 3: 0.2287 -> 0.2055  ratio 0.899

Round configurations (EC only):
  R1  68 features, 0.5 ExtraTrees(100, leaf 8) + 0.5 LGB huber          no shrink
  R2  14 features, 0.6 ET600/1 + 0.3 LGB tweedie + 0.1 MLP, shrink 0.5, clip
  R3  R2 with the source fingerprint added to the ET member

Run:  cd research && PYTHONPATH="" <python> -u calib_ec_sp.py
"""
import env  # noqa: F401
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load, views, folds, DOMAIN_MARKS
import fp_features as FP
from deep_cal_16 import sp2_folds
from deep_cal_eval import sp_folds
from ec_v6 import SEED, DET, pipeline

E_HUB = dict(objective="huber", n_estimators=400, learning_rate=0.02, num_leaves=7,
             min_child_samples=240, subsample=0.7, subsample_freq=1, colsample_bytree=0.4,
             reg_lambda=5.0)
ET8 = dict(n_estimators=100, max_features=1.0, min_samples_leaf=8, n_jobs=4)
REAL = {"R2/R1": 0.2287 / 0.2442, "R3/R2": 0.2055 / 0.2287}


def r1_pipeline(cols):
    def fn(tr, va):
        y = tr.sub_ec.values
        a = make_pipeline(SimpleImputer(strategy="median"),
                          ExtraTreesRegressor(random_state=SEED, **ET8)).fit(tr[cols], y).predict(va[cols])
        b = lgb.LGBMRegressor(random_state=SEED, **DET, **E_HUB).fit(tr[cols], y).predict(va[cols])
        return 0.5 * a + 0.5 * b
    return fn


def main():
    panel, _, lab0 = load()
    v = views(panel)
    fp = FP.build()
    fpc = FP.names(fp)
    lab = lab0.merge(fp, on="row_id", how="left")
    y = lab.sub_ec.values
    f68 = [c for c in v["ec_full"] if not any(m in c for m in DOMAIN_MARKS)]
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    cfg = {"R1": r1_pipeline(f68), "R2": pipeline(f14, f14), "R3": pipeline(f14 + fpc, f14)}
    sets = [("SP2", sp2_folds()), ("SP", sp_folds()), ("A", folds("A")), ("B", folds("B"))]

    score = {}
    for sname, fds in sets:
        for k, fn in cfg.items():
            o = np.full(len(lab), np.nan)
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                if vam.sum():
                    o[np.where(vam)[0]] = fn(lab[trm], lab[vam].reset_index(drop=True))
            g = ~np.isnan(o)
            score[(sname, k)] = rmse(o[g], y[g])
            print("  %-4s %s %.4f" % (sname, k, score[(sname, k)]), flush=True)

    print("\n%-6s %8s %8s %8s | %8s %8s   (real R2/R1 %.3f, R3/R2 %.3f)"
          % ("", "R1", "R2", "R3", "R2/R1", "R3/R2", REAL["R2/R1"], REAL["R3/R2"]))
    for sname, _ in sets:
        r1, r2, r3 = (score[(sname, k)] for k in ("R1", "R2", "R3"))
        err = abs(r2 / r1 - REAL["R2/R1"]) + abs(r3 / r2 - REAL["R3/R2"])
        print("%-6s %8.4f %8.4f %8.4f | %8.3f %8.3f   |error| sum %.3f"
              % (sname, r1, r2, r3, r2 / r1, r3 / r2, err))


if __name__ == "__main__":
    main()
