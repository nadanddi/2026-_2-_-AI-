# -*- coding: utf-8 -*-
"""Final model: joint F13+F47 LightGBM, contemporaneous input features + hour
cyclic encoding + greenhouse categorical indicator. No lag/label features, no
pooling with the other 49 greenhouses -- CV showed both of those hurt.
Trains one model per target using ALL available F13+F47 train_y rows and
predicts on the real test_X (F13+F47, 1440 rows), matching sample_submission
row_id order.
"""
import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")
import boot
import numpy as np
import pandas as pd
import os
import lightgbm as lgb

from features import parse_rid, add_time_features

DATA = boot.DATA
OUT_DIR = r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind\B_sonnet"

train_X = pd.read_csv(os.path.join(DATA, "train_X.csv"))
train_y = pd.read_csv(os.path.join(DATA, "train_y.csv"))
test_X = pd.read_csv(os.path.join(DATA, "test_X.csv"))
sample_sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))

train_X = parse_rid(train_X)
test_X = parse_rid(test_X)
train_y = parse_rid(train_y)

train_full = train_X.merge(train_y[["row_id", "sub_temp", "sub_ec"]], on="row_id", how="left")
train_full = train_full[train_full.gh.isin(["F13", "F47"])].copy()
train_full = add_time_features(train_full)
test_full = add_time_features(test_X[test_X.gh.isin(["F13", "F47"])].copy())

INPUT_COLS = ["in_temp", "in_hum", "in_co2", "in_rad",
              "out_temp", "out_hum", "out_rad", "out_wspd",
              "act_vent", "act_shade", "act_thermal", "act_heating",
              "act_circfan", "act_co2", "act_fog"]
TIME_COLS = ["hour_sin", "hour_cos"]

train_full["gh_cat"] = train_full["gh"].astype("category")
test_full["gh_cat"] = test_full["gh"].astype("category").cat.set_categories(train_full["gh_cat"].cat.categories)

FEATURE_COLS = INPUT_COLS + TIME_COLS + ["gh_cat"]

preds = {}
for target in ["sub_temp", "sub_ec"]:
    tr = train_full[train_full[target].notna()]
    X_tr = tr[FEATURE_COLS]
    y_tr = tr[target]
    X_te = test_full[FEATURE_COLS]

    # small ensemble over 3 seeds for stability (matches CV protocol)
    seed_preds = []
    for seed in [0, 1, 2]:
        model = lgb.LGBMRegressor(
            n_estimators=300, num_leaves=15, learning_rate=0.05,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=-1,
        )
        model.fit(X_tr, y_tr, categorical_feature=["gh_cat"])
        seed_preds.append(model.predict(X_te))
    preds[target] = np.mean(seed_preds, axis=0)

result = test_full[["row_id"]].copy()
result["sub_temp"] = preds["sub_temp"]
result["sub_ec"] = preds["sub_ec"]

# reorder to match sample_submission row_id order exactly
result = sample_sub[["row_id"]].merge(result, on="row_id", how="left")
assert result["sub_temp"].notna().all() and result["sub_ec"].notna().all(), "missing predictions!"
assert list(result["row_id"]) == list(sample_sub["row_id"]), "row_id order mismatch!"

out_path = os.path.join(OUT_DIR, "pred_test.csv")
result.to_csv(out_path, index=False)
print(f"Wrote {out_path}")
print(result.describe())
print(result.head())
