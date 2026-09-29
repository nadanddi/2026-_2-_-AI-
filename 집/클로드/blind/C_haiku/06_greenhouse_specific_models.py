import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")

import boot
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

data_path = r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\온라인대회자료\정형데이터\참가자_배포"

# Load data
train_X = pd.read_csv(f"{data_path}/train_X.csv")
train_y = pd.read_csv(f"{data_path}/train_y.csv")
test_X = pd.read_csv(f"{data_path}/test_X.csv")

def parse_row_id(row_id):
    parts = row_id.split('_')
    return parts[0], int(parts[1]), int(parts[2])

train_X['gh'] = train_X['row_id'].apply(lambda x: parse_row_id(x)[0])
train_X['day'] = train_X['row_id'].apply(lambda x: parse_row_id(x)[1])
train_X['hour'] = train_X['row_id'].apply(lambda x: parse_row_id(x)[2])

train_y['gh'] = train_y['row_id'].apply(lambda x: parse_row_id(x)[0])
train_y['day'] = train_y['row_id'].apply(lambda x: parse_row_id(x)[1])

test_X['gh'] = test_X['row_id'].apply(lambda x: parse_row_id(x)[0])
test_X['day'] = test_X['row_id'].apply(lambda x: parse_row_id(x)[1])

merged = train_X.merge(train_y[['row_id', 'sub_temp', 'sub_ec']], on='row_id', how='left')

input_cols = [c for c in train_X.columns if c not in ['row_id', 'gh', 'day', 'hour']]

print("=== GREENHOUSE-SPECIFIC MODELS ===\n")

# Train separate models for each test greenhouse
test_ghs = ['F13', 'F47']

# ==================== SUB_TEMP MODELS ====================
print("=== SUB_TEMP: Training greenhouse-specific models ===")

temp_models_by_gh = {}
val_results_temp = {}

for test_gh in test_ghs:
    print(f"\n{test_gh}:")

    # Get all temp data including from other greenhouses
    temp_all = merged[merged['sub_temp'].notna()].copy()
    X_all = temp_all[input_cols].fillna(0)
    y_all = temp_all['sub_temp'].values

    # Train full model
    model_full = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=4,
        subsample=0.85,
        random_state=42
    )
    model_full.fit(X_all, y_all)
    temp_models_by_gh[test_gh] = model_full

    # Validate on this greenhouse
    gh_data = temp_all[temp_all['gh'] == test_gh].sort_values('day')
    train_d = gh_data[gh_data['day'] < 180]
    val_d = gh_data[gh_data['day'] >= 180]

    if len(train_d) > 0 and len(val_d) > 0:
        X_tr = train_d[input_cols].fillna(0)
        y_tr = train_d['sub_temp'].values
        X_vl = val_d[input_cols].fillna(0)
        y_vl = val_d['sub_temp'].values

        m = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            subsample=0.85,
            random_state=42
        )
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_vl)
        rmse = np.sqrt(np.mean((y_pred - y_vl) ** 2))
        val_results_temp[test_gh] = rmse
        print(f"  Validation RMSE: {rmse:.4f}")

# ==================== SUB_EC MODELS ====================
print("\n=== SUB_EC: Training greenhouse-specific models ===")

ec_models_by_gh = {}
val_results_ec = {}

for test_gh in test_ghs:
    print(f"\n{test_gh}:")

    # EC data only for F13 and F47
    ec_all = merged[(merged['sub_ec'].notna()) & (merged['gh'].isin(['F13', 'F47']))].copy()
    X_all = ec_all[input_cols].fillna(0)
    y_all = ec_all['sub_ec'].values

    # Train model
    model_full = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=4,
        subsample=0.85,
        random_state=42
    )
    model_full.fit(X_all, y_all)
    ec_models_by_gh[test_gh] = model_full

    # Validate
    gh_data = ec_all[ec_all['gh'] == test_gh].sort_values('day')
    train_d = gh_data[gh_data['day'] < 180]
    val_d = gh_data[gh_data['day'] >= 180]

    if len(train_d) > 0 and len(val_d) > 0:
        X_tr = train_d[input_cols].fillna(0)
        y_tr = train_d['sub_ec'].values
        X_vl = val_d[input_cols].fillna(0)
        y_vl = val_d['sub_ec'].values

        m = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            subsample=0.85,
            random_state=42
        )
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_vl)
        rmse = np.sqrt(np.mean((y_pred - y_vl) ** 2))
        val_results_ec[test_gh] = rmse
        print(f"  Validation RMSE: {rmse:.4f}")

# ==================== PREDICTIONS ====================
print("\n=== GENERATING PREDICTIONS ===")

pred_dict = {'row_id': [], 'sub_temp': [], 'sub_ec': []}

for idx, row in test_X.iterrows():
    row_id = row['row_id']
    gh = row['gh']

    X_row = row[input_cols].fillna(0).values.reshape(1, -1)

    # Use model for this greenhouse
    temp_pred = temp_models_by_gh[gh].predict(X_row)[0]
    ec_pred = ec_models_by_gh[gh].predict(X_row)[0]

    pred_dict['row_id'].append(row_id)
    pred_dict['sub_temp'].append(temp_pred)
    pred_dict['sub_ec'].append(ec_pred)

submission = pd.DataFrame(pred_dict)
submission = submission.sort_values('row_id').reset_index(drop=True)

pred_path = r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind\C_haiku\pred_test.csv"
submission.to_csv(pred_path, index=False)
print(f"✓ Saved predictions")

print(f"\nValidation RMSE:")
print(f"  Temp: {list(val_results_temp.values())}")
print(f"  EC: {list(val_results_ec.values())}")

print(f"\nPredictions stats:")
print(f"  sub_temp: mean={submission['sub_temp'].mean():.3f}, std={submission['sub_temp'].std():.3f}")
print(f"  sub_ec: mean={submission['sub_ec'].mean():.3f}, std={submission['sub_ec'].std():.3f}")
print(f"\nSample:")
print(submission.head(10).to_string(index=False))
