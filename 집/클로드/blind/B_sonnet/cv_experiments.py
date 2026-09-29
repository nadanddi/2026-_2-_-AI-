# -*- coding: utf-8 -*-
"""Cross-validation experiments mimicking the real test's day-block-with-gap
structure, evaluated purely on historical (fully-labeled) portions of F13/F47
so we have real ground truth to score against.

Template (relative day offsets 0..54, discovered from the real test/train
layout for both F13 and F47):
  0-4    TEST   (5d)
  5      GAP
  6-13   INTERIOR (visible/known, 8d)
  14     GAP
  15-24  TEST   (10d)
  25     GAP
  26-33  INTERIOR (8d)
  34     GAP
  35-44  TEST   (10d)
  45     GAP
  46-48  INTERIOR (3d)
  49     GAP
  50-54  TEST   (5d)
"""
import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")
import boot
import numpy as np
import pandas as pd
import os
import lightgbm as lgb

from features import parse_rid, add_time_features, build_lag_features_for_target

pd.set_option("display.width", 220)

RNG_SEEDS = [0, 1, 2]

TEST_OFFSETS = set(range(0, 5)) | set(range(15, 25)) | set(range(35, 45)) | set(range(50, 55))
TEMPLATE_LEN = 55

INPUT_COLS_COMMON = ["in_temp", "in_hum", "in_co2", "in_rad"]
INPUT_COLS_F1347 = ["out_temp", "out_hum", "out_rad", "out_wspd",
                    "act_vent", "act_shade", "act_thermal", "act_heating",
                    "act_circfan", "act_co2", "act_fog"]

DATA = boot.DATA
train_X = pd.read_csv(os.path.join(DATA, "train_X.csv"))
train_y = pd.read_csv(os.path.join(DATA, "train_y.csv"))

train_X = parse_rid(train_X)
train_y = parse_rid(train_y)
full = train_X.merge(train_y[["row_id", "sub_temp", "sub_ec"]], on="row_id", how="left")
full = add_time_features(full)

log = []
def p(s=""):
    print(s)
    log.append(str(s))


def make_synthetic_masks(gh_df, anchor):
    """Return boolean arrays (aligned to gh_df index) for TEST rows given anchor day."""
    offset = gh_df["day"] - anchor
    in_window = (offset >= 0) & (offset < TEMPLATE_LEN)
    test_mask = in_window & offset.isin(TEST_OFFSETS)
    return test_mask


def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_fold(target, gh, anchor, other_gh_data=None, use_pooled=False, seed=0, sibling_gh=None):
    """Simulate one CV fold for `target` on greenhouse `gh` with given anchor.
    Returns dict of method -> rmse, plus n_test.
    """
    gdf = full[full.gh == gh].copy().sort_values("t").reset_index(drop=True)
    test_mask = make_synthetic_masks(gdf, anchor)
    n_test = test_mask.sum()
    if n_test == 0:
        return None

    # visible labels: hide the synthetic test rows' target
    visible = gdf.copy()
    visible.loc[test_mask, target] = np.nan
    # rows usable for training (own greenhouse): all EXCEPT synthetic test rows,
    # and must have non-null target
    own_train_rows = visible[visible[target].notna()].copy()

    # lag features must be built using ONLY visible (masked) labels source,
    # for both test rows and own_train_rows (mimics real hidden-test opacity)
    label_source = visible[["gh", "t", target]]

    query_all = gdf[["gh", "t"]].copy()
    lag_feats_all = build_lag_features_for_target(query_all, label_source, target)
    gdf_feat = pd.concat([gdf.reset_index(drop=True), lag_feats_all.reset_index(drop=True)], axis=1)

    test_df = gdf_feat[test_mask.values]
    train_df_own = gdf_feat[(~test_mask.values) & gdf_feat[target].notna()]

    if len(train_df_own) < 20 or len(test_df) == 0:
        return None

    y_true = test_df[target].values

    results = {}

    # BASE1: per-gh mean of visible labels
    base_mean = train_df_own[target].mean()
    results["base_mean"] = rmse(y_true, np.full(len(y_true), base_mean))

    # BASE2: persistence (last known value)
    pers = test_df["lag_last_val"].values
    pers_filled = np.where(np.isnan(pers), base_mean, pers)
    results["base_persistence"] = rmse(y_true, pers_filled)

    # BASE3: trend extrapolation
    trend = test_df["lag_trend_extrap"].values
    trend_filled = np.where(np.isnan(trend), pers_filled, trend)
    results["base_trend_extrap"] = rmse(y_true, trend_filled)

    # feature sets
    avail_cols = [c for c in INPUT_COLS_COMMON if c in gdf_feat.columns]
    if gh in ("F13", "F47"):
        avail_cols = avail_cols + [c for c in INPUT_COLS_F1347 if c in gdf_feat.columns]
    lag_cols = ["lag_last_val", "lag_last_dt", "lag_prev2_val", "lag_prev2_dt", "lag_slope", "lag_trend_extrap", "lag_roll_mean5"]
    time_cols = ["hour_sin", "hour_cos"]

    def fit_predict(feature_cols, extra_train=None, seed=0):
        tr = train_df_own
        if extra_train is not None and len(extra_train) > 0:
            tr = pd.concat([tr, extra_train], axis=0, ignore_index=True)
        X_tr = tr[feature_cols]
        y_tr = tr[target]
        X_te = test_df[feature_cols]
        model = lgb.LGBMRegressor(
            n_estimators=300, num_leaves=15, learning_rate=0.05,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=-1,
        )
        model.fit(X_tr, y_tr)
        return model.predict(X_te)

    # ML1: input features + time only (no lag)
    pred1 = fit_predict(avail_cols + time_cols, seed=seed)
    results["ml_input_only"] = rmse(y_true, pred1)

    # ML2: input features + lag features
    pred2 = fit_predict(avail_cols + time_cols + lag_cols, seed=seed)
    results["ml_input_plus_lag"] = rmse(y_true, pred2)

    # ML3: lag features only
    pred3 = fit_predict(lag_cols, seed=seed)
    results["ml_lag_only"] = rmse(y_true, pred3)

    if use_pooled and other_gh_data is not None:
        pred4 = fit_predict(INPUT_COLS_COMMON + time_cols + lag_cols,
                             extra_train=other_gh_data, seed=seed)
        results["ml_pooled_input_lag"] = rmse(y_true, pred4)
        pred5 = fit_predict(avail_cols + time_cols, extra_train=other_gh_data, seed=seed)
        results["ml_pooled_input_only"] = rmse(y_true, pred5)

    # blend: input_only model + persistence (only meaningful for ec where both are decent)
    blend = 0.7 * pred1 + 0.3 * pers_filled
    results["blend_input70_pers30"] = rmse(y_true, blend)

    if sibling_gh is not None:
        sib_df = full[full.gh == sibling_gh].copy().sort_values("t").reset_index(drop=True)
        sib_label_source = sib_df[["gh", "t", target]]
        sib_query = sib_df[["gh", "t"]].copy()
        sib_lag = build_lag_features_for_target(sib_query, sib_label_source, target)
        sib_feat = pd.concat([sib_df.reset_index(drop=True), sib_lag.reset_index(drop=True)], axis=1)
        sib_feat = sib_feat[sib_feat[target].notna()]
        pred_joint = fit_predict(avail_cols + time_cols, extra_train=sib_feat, seed=seed)
        results["ml_joint_2gh_input_only"] = rmse(y_true, pred_joint)
        pred_joint_lag = fit_predict(avail_cols + time_cols + lag_cols, extra_train=sib_feat, seed=seed)
        results["ml_joint_2gh_input_lag"] = rmse(y_true, pred_joint_lag)

    return {"n_test": int(n_test), **results}


def main():
    anchors_by_gh = {
        "F13": [25, 65, 105],
        "F47": [25, 65, 105],
    }

    # prepare "other greenhouses" pooled data for sub_temp pooling experiment
    other_gh_full = full[~full.gh.isin(["F13", "F47"])].copy()
    other_gh_full_feat_cache = {}

    all_results = []

    for target in ["sub_temp", "sub_ec"]:
        p(f"\n{'='*70}\nTARGET = {target}\n{'='*70}")
        for gh in ["F13", "F47"]:
            for anchor in anchors_by_gh[gh]:
                for seed in RNG_SEEDS:
                    use_pooled = (target == "sub_temp")
                    extra = None
                    if use_pooled:
                        key = target
                        if key not in other_gh_full_feat_cache:
                            oq = other_gh_full[["gh", "t"]].copy()
                            olab = other_gh_full[["gh", "t", target]]
                            olag = build_lag_features_for_target(oq, olab, target)
                            ofeat = pd.concat([other_gh_full.reset_index(drop=True), olag.reset_index(drop=True)], axis=1)
                            ofeat = ofeat[ofeat[target].notna()]
                            other_gh_full_feat_cache[key] = ofeat
                        extra = other_gh_full_feat_cache[key]
                    sibling = "F47" if gh == "F13" else "F13"
                    res = run_fold(target, gh, anchor, other_gh_data=extra, use_pooled=use_pooled, seed=seed, sibling_gh=sibling)
                    if res is None:
                        continue
                    res.update({"target": target, "gh": gh, "anchor": anchor, "seed": seed})
                    all_results.append(res)
                    p(f"target={target} gh={gh} anchor={anchor} seed={seed} n_test={res['n_test']} "
                      f"mean={res['base_mean']:.4f} pers={res['base_persistence']:.4f} trend={res['base_trend_extrap']:.4f} "
                      f"ml_in={res['ml_input_only']:.4f} ml_in_lag={res['ml_input_plus_lag']:.4f} ml_lag={res['ml_lag_only']:.4f}"
                      + (f" ml_pool={res.get('ml_pooled_input_lag', float('nan')):.4f}" if "ml_pooled_input_lag" in res else ""))

    res_df = pd.DataFrame(all_results)
    res_df.to_csv("cv_results_raw.csv", index=False)

    p("\n\n=== SUMMARY (mean +/- std across gh/anchor/seed folds) ===")
    method_cols = ["base_mean", "base_persistence", "base_trend_extrap", "ml_input_only", "ml_input_plus_lag", "ml_lag_only",
                   "ml_pooled_input_lag", "ml_pooled_input_only", "blend_input70_pers30", "ml_joint_2gh_input_only", "ml_joint_2gh_input_lag"]
    for target in ["sub_temp", "sub_ec"]:
        sub = res_df[res_df.target == target]
        p(f"\n--- {target} (n_folds={len(sub)}) ---")
        for m in method_cols:
            if m not in sub.columns:
                continue
            vals = sub[m].dropna()
            if len(vals) == 0:
                continue
            p(f"  {m:25s}: mean={vals.mean():.4f}  std={vals.std():.4f}  min={vals.min():.4f}  max={vals.max():.4f}  n={len(vals)}")

    with open("cv_experiments_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print("ALL DONE")


if __name__ == "__main__":
    main()
