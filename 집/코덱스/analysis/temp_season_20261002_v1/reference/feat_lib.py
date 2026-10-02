# -*- coding: utf-8 -*-
"""Shared pieces for the feature-triage study.

Three things live here that the individual feat_* scripts all need:

1. `GROUPS` -- a partition of the temp/EC column sets by PHYSICAL ROLE, not by
   name-pattern convenience.  Ablation is done group-wise because the columns
   are massively collinear (74 of the 98 temp columns have VIF > 10); dropping
   one column at a time measures nothing but the redundancy of its neighbours.

2. Uncertainty that respects the autocorrelation.  Rows are hourly and sub_ec
   is ~95% explained by the daily level, so the effective sample is ~400
   greenhouse-days, not 9,600 rows.  Every delta reported anywhere in this
   study is accompanied by a PAIRED block bootstrap over greenhouse-days
   (resample whole (farm, day) blocks, recompute both RMSEs on the same
   resample, take the difference).  Row-level standard errors / p-values are
   invalid here and are never computed.

3. Day-level scoring for EC: RMSE of the greenhouse-day mean prediction
   against the greenhouse-day mean label.  If a feature only moves the hourly
   RMSE but not the daily one it is polishing noise.
"""
import numpy as np
import pandas as pd

import env  # noqa: F401  MUST be the first project import
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from harness import blend_factory

# ---------------------------------------------------------------- estimators
DET = dict(deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
T_HUB = dict(objective="huber", n_estimators=1200, learning_rate=0.03,
             num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)
E_HUB = dict(objective="huber", n_estimators=400, learning_rate=0.02,
             num_leaves=7, min_child_samples=240, subsample=0.7,
             subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0)
ET8 = dict(n_estimators=100, max_features=1.0, min_samples_leaf=8, n_jobs=4)

# Cheap screening surrogate for sub_temp.  The submitted temp model is 1200
# trees x 3 seeds and costs ~530 s per candidate over both fold sets, which is
# unaffordable for a 15-way ablation on a machine that is also running five
# other experiment processes.  This keeps the loss, the leaf size and the
# column subsampling, and buys speed with 1/5 the trees at 4x the learning
# rate, one seed, and two threads so it does not thrash against the others.
# Its ABSOLUTE RMSE is not comparable to the submitted model's; only deltas
# against its own baseline are ever quoted, and the groups that screen large
# are re-measured at the real setting.
DET2 = dict(deterministic=True, force_col_wise=True, n_jobs=2, verbose=-1)
T_FAST = dict(objective="huber", n_estimators=250, learning_rate=0.12,
              num_leaves=31, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0)


def lgbf(p, det=None):
    d = DET if det is None else det
    return lambda s: lgb.LGBMRegressor(random_state=s, **d, **p)


def etf(p):
    return lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                   ExtraTreesRegressor(random_state=s, **p))


TEMP_MODEL = lgbf(T_HUB)                                   # submitted temp model
TEMP_FAST = lgbf(T_FAST, DET2)                             # screening surrogate
EC_MODEL = blend_factory([etf(ET8), lgbf(E_HUB)])          # submitted EC model


# -------------------------------------------------------------- group tables
# Each entry: (group name, predicate on the column name).  Order matters: the
# first matching rule wins, so specific rules come before generic ones.
RAW_IN = {"in_temp", "in_hum", "in_co2"}
RAW_OUT = {"out_temp", "out_hum", "out_rad", "out_wspd"}
RAW_ACT = {"act_vent", "act_shade", "act_thermal", "act_heating",
           "act_circfan", "act_co2", "act_fog"}
PHYS_NOW = {"dt_in_out", "vpd_in", "vpd_out", "rad_eff", "transp_pm",
            "root_dh", "heat_input", "screen_ins", "heat_screen"}

_RULES = [
    # --- zero-history identity / position -----------------------------------
    ("t_index",    lambda c: c in ("day", "day_par")),
    ("clock",      lambda c: c in ("hour", "hr_sin", "hr_cos", "midnight")),
    ("farm",       lambda c: c == "farm_id"),
    # --- current interval ----------------------------------------------------
    ("raw_in",     lambda c: c in RAW_IN),
    ("raw_out",    lambda c: c in RAW_OUT),
    ("raw_act",    lambda c: c in RAW_ACT),
    ("phys_now",   lambda c: c in PHYS_NOW),
    # --- history, by timescale ----------------------------------------------
    ("prev_week",  lambda c: c.endswith("_pd7m")),
    ("prev_day",   lambda c: "_pd1" in c or "_pd2" in c),
    ("today",      lambda c: c.endswith("_tdmean") or c.endswith("_tdsum")),
    ("season_cum", lambda c: c.endswith("_cum") or c.endswith("_cumrate")),
    ("rtr",        lambda c: c.startswith("rtr_") or c.startswith("transp_per_rad")),
    ("mem_long",   lambda c: c.endswith("_ewm24") or c.endswith("_dev24")),
    ("mem_short",  lambda c: c.endswith("_ewm2") or c.endswith("_ewm6")
                             or c.endswith("_dev6")),
    ("short_dyn",  lambda c: "_lag" in c or c.endswith("_d1") or c.endswith("_d3")),
    ("roll24",     lambda c: "_r24m" in c or "_r24s" in c),
    ("roll_multi", lambda c: "_r72m" in c or "_r168m" in c),
    # --- anything added later by feat_new -----------------------------------
    ("new_hinge",  lambda c: c.startswith("day_hinge")),
    ("new_rep24",  lambda c: c.endswith("_rep24")),
    ("new_event",  lambda c: "_hsince" in c),
    ("new_dew",    lambda c: c.startswith("dew")),
    ("new_duty",   lambda c: "_duty" in c),
]


def group_of(col):
    for name, pred in _RULES:
        if pred(col):
            return name
    return "UNGROUPED"


def grouping(cols):
    """{group: [cols]} preserving a stable order; raises if anything is left out."""
    g = {}
    for c in cols:
        g.setdefault(group_of(c), []).append(c)
    if "UNGROUPED" in g:
        raise RuntimeError("no physical group for: %s" % g["UNGROUPED"])
    return {k: sorted(v) for k, v in sorted(g.items())}


# ------------------------------------------------------- day-level + bootstrap
def day_frame(lab, target, oof):
    """Greenhouse-day means of (prediction, label) over the scored rows."""
    ok = ~np.isnan(oof)
    d = pd.DataFrame({"farm": lab.farm.values[ok], "day": lab.day.values[ok],
                      "y": lab[target].values[ok], "p": oof[ok]})
    return d.groupby(["farm", "day"], as_index=False)[["y", "p"]].mean()


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def day_rmse(lab, target, oof):
    d = day_frame(lab, target, oof)
    return _rmse(d.y, d.p)


def paired_block_boot(lab, target, oof_ref, oof_alt, n_boot=2000, seed=0,
                      level="row"):
    """Paired greenhouse-day block bootstrap of RMSE(alt) - RMSE(ref).

    Blocks are whole (farm, day) units -- the unit that actually carries
    independent information in this dataset.  `level="row"` scores the hourly
    rows inside the resampled days; `level="day"` scores the day means.
    Returns (delta_point, lo95, hi95, p_worse) where p_worse is the bootstrap
    fraction of resamples in which the alternative is worse.
    """
    ok = ~np.isnan(oof_ref) & ~np.isnan(oof_alt)
    df = pd.DataFrame({"farm": lab.farm.values[ok], "day": lab.day.values[ok],
                       "y": lab[target].values[ok],
                       "a": oof_ref[ok], "b": oof_alt[ok]})
    if level == "day":
        df = df.groupby(["farm", "day"], as_index=False)[["y", "a", "b"]].mean()
    keys = df.groupby(["farm", "day"]).indices
    blocks = [np.asarray(v) for v in keys.values()]
    y, a, b = df.y.values, df.a.values, df.b.values
    point = _rmse(y, b) - _rmse(y, a)
    rng = np.random.RandomState(seed)
    nb = len(blocks)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.randint(0, nb, nb)
        idx = np.concatenate([blocks[j] for j in pick])
        out[i] = _rmse(y[idx], b[idx]) - _rmse(y[idx], a[idx])
    return (point, float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5)), float(np.mean(out > 0)))


def fmt_delta(d):
    return "%+.4f" % d
