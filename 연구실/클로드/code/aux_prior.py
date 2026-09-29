# -*- coding: utf-8 -*-
"""Substrate-temperature prior learned from the 49 auxiliary greenhouses.

Those greenhouses only carry in_temp / in_hum / in_co2 and their sub_temp is
quantised to 1 degC, so they cannot be mixed into the main model directly.
Instead we train a generic "substrate temperature given indoor climate" model
on them (F13/F47 are NEVER used here, so the prior is leak-free for any fold)
and feed its prediction to the main model as a single extra feature.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import TARGET_FARMS

CORE = ["in_temp", "in_hum", "in_co2"]
A_LAGS = [1, 2, 3, 6, 12, 24]
A_ROLLS = [3, 6, 24, 72]


def core_panel(df):
    """Causal lag/rolling features on the 3 universally available columns."""
    out = []
    for farm, g in df.groupby("farm"):
        g = g.sort_values("t")
        full = np.arange(g.t.min(), g.t.max() + 1)
        p = g.set_index("t").reindex(full)
        p["farm"] = farm
        p["day"] = p.index // 24
        p["hour"] = p.index % 24
        feats = {}
        for c in CORE:
            s = p[c]
            for L in A_LAGS:
                feats["%s_lag%d" % (c, L)] = s.shift(L)
            for W in A_ROLLS:
                r = s.rolling(W, min_periods=max(2, W // 4))
                feats["%s_r%dm" % (c, W)] = r.mean()
            feats["%s_d1" % c] = s - s.shift(1)
            feats["%s_d24" % c] = s - s.shift(24)
            feats["%s_dev24" % c] = s - feats["%s_r24m" % c]
        p = pd.concat([p, pd.DataFrame(feats, index=p.index)], axis=1)
        out.append(p.reset_index().rename(columns={"index": "t"}))
    p = pd.concat(out, ignore_index=True)
    p = p.dropna(subset=["row_id"]).reset_index(drop=True)
    p["hr_sin"] = np.sin(2 * np.pi * p.hour / 24)
    p["hr_cos"] = np.cos(2 * np.pi * p.hour / 24)
    return p


def aux_feature_cols(p):
    drop = {"row_id", "farm", "t", "sub_temp", "sub_ec"}
    return [c for c in p.columns if c not in drop]


def fit_aux_prior(tX, ty, seed=0):
    """Train on the 49 auxiliary greenhouses only; return the fitted model."""
    cols = ["row_id", "farm", "day", "hour", "t"] + CORE
    aux = tX.loc[~tX.farm.isin(TARGET_FARMS), cols]
    p = core_panel(aux)
    p = p.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    p = p[p.sub_temp.notna()]
    fc = aux_feature_cols(p)
    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        min_child_samples=100, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, n_jobs=4,
        verbose=-1,
    )
    model.fit(p[fc], p.sub_temp)
    return model, fc


def apply_aux_prior(model, fc, tX, sX):
    """Predict the prior for the two evaluation greenhouses."""
    cols = ["row_id", "farm", "day", "hour", "t"] + CORE
    tgt = pd.concat([tX[cols], sX[cols]], ignore_index=True)
    tgt = tgt[tgt.farm.isin(TARGET_FARMS)]
    p = core_panel(tgt)
    for c in fc:
        if c not in p.columns:
            p[c] = np.nan
    return pd.DataFrame({"row_id": p.row_id.values,
                         "aux_temp_prior": model.predict(p[fc])})
