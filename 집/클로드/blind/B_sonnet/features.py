# -*- coding: utf-8 -*-
"""Shared feature engineering utilities. Causal-only (no future leakage)."""
import numpy as np
import pandas as pd


def parse_rid(df):
    parts = df["row_id"].str.split("_", expand=True)
    df = df.copy()
    df["gh"] = parts[0]
    df["day"] = parts[1].astype(int)
    df["hour"] = parts[2].astype(int)
    df["t"] = df["day"] * 24 + df["hour"]
    return df


def add_time_features(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


class CausalLagBuilder:
    """For a given greenhouse+target, given a set of KNOWN (t, value) label points,
    build strictly-causal (t' < t) lag features for any query t array:
      - last_val, last_dt (hours since last known value)
      - prev2_val, prev2_dt
      - slope estimated from last two known points
      - trend_extrap = last_val + slope * last_dt
      - roll_mean_5 = mean of last up to 5 known values before t
    Query rows may themselves be in the known set (e.g. when building training
    features) -- we always exclude any known point with t' >= query t.
    """

    def __init__(self, known_t, known_val):
        order = np.argsort(known_t)
        self.t = np.asarray(known_t)[order]
        self.v = np.asarray(known_val)[order]

    def build(self, query_t):
        query_t = np.asarray(query_t)
        n = len(query_t)
        last_val = np.full(n, np.nan)
        last_dt = np.full(n, np.nan)
        prev2_val = np.full(n, np.nan)
        prev2_dt = np.full(n, np.nan)
        roll_mean5 = np.full(n, np.nan)
        if len(self.t) == 0:
            slope = np.full(n, np.nan)
            trend_extrap = np.full(n, np.nan)
            return pd.DataFrame({
                "lag_last_val": last_val, "lag_last_dt": last_dt,
                "lag_prev2_val": prev2_val, "lag_prev2_dt": prev2_dt,
                "lag_slope": slope, "lag_trend_extrap": trend_extrap,
                "lag_roll_mean5": roll_mean5,
            })
        # idx = number of known points strictly before query_t (searchsorted 'left'
        # on strict < means: position where known_t < query_t ends)
        idx = np.searchsorted(self.t, query_t, side="left")  # count of t' < query_t (since query not necessarily in known)
        for i in range(n):
            k = idx[i]  # number of known points with self.t < query_t[i]
            if k >= 1:
                last_val[i] = self.v[k - 1]
                last_dt[i] = query_t[i] - self.t[k - 1]
            if k >= 2:
                prev2_val[i] = self.v[k - 2]
                prev2_dt[i] = query_t[i] - self.t[k - 2]
            if k >= 1:
                lo = max(0, k - 5)
                roll_mean5[i] = self.v[lo:k].mean()
        slope = np.where(
            (~np.isnan(prev2_val)) & ((last_dt - prev2_dt) != 0)
            if False else (~np.isnan(prev2_val)),
            np.divide(last_val - prev2_val, (prev2_dt - last_dt),
                      out=np.full(n, np.nan), where=(~np.isnan(prev2_val)) & ((prev2_dt - last_dt) != 0)),
            np.nan,
        )
        trend_extrap = last_val + slope * last_dt
        # fallback: where slope is nan, trend_extrap = last_val
        trend_extrap = np.where(np.isnan(trend_extrap), last_val, trend_extrap)
        return pd.DataFrame({
            "lag_last_val": last_val, "lag_last_dt": last_dt,
            "lag_prev2_val": prev2_val, "lag_prev2_dt": prev2_dt,
            "lag_slope": slope, "lag_trend_extrap": trend_extrap,
            "lag_roll_mean5": roll_mean5,
        })


def build_lag_features_for_target(df_query, df_labels_source, target_col, gh_col="gh", t_col="t"):
    """df_labels_source must have columns [gh, t, target_col] (non-null rows = known points).
    Returns a DataFrame aligned with df_query index containing lag_* columns.
    """
    out_frames = []
    src = df_labels_source[df_labels_source[target_col].notna()]
    for gh, grp_q in df_query.groupby(gh_col):
        src_gh = src[src[gh_col] == gh]
        builder = CausalLagBuilder(src_gh[t_col].values, src_gh[target_col].values)
        feats = builder.build(grp_q[t_col].values)
        feats.index = grp_q.index
        out_frames.append(feats)
    result = pd.concat(out_frames).sort_index()
    return result
