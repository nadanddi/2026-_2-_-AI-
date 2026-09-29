import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")

import boot
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

data_path = r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\온라인대회자료\정형데이터\참가자_배포"

# Load data
train_X = pd.read_csv(f"{data_path}/train_X.csv")
train_y = pd.read_csv(f"{data_path}/train_y.csv")
test_X = pd.read_csv(f"{data_path}/test_X.csv")

def parse_row_id(row_id):
    parts = row_id.split('_')
    gh_id = parts[0]
    day = int(parts[1])
    hour = int(parts[2])
    return gh_id, day, hour

# Add greenhouse, day, hour
train_X['gh'] = train_X['row_id'].apply(lambda x: parse_row_id(x)[0])
train_X['day'] = train_X['row_id'].apply(lambda x: parse_row_id(x)[1])
train_X['hour'] = train_X['row_id'].apply(lambda x: parse_row_id(x)[2])

train_y['gh'] = train_y['row_id'].apply(lambda x: parse_row_id(x)[0])
train_y['day'] = train_y['row_id'].apply(lambda x: parse_row_id(x)[1])

test_X['gh'] = test_X['row_id'].apply(lambda x: parse_row_id(x)[0])
test_X['day'] = test_X['row_id'].apply(lambda x: parse_row_id(x)[1])

# Merge
merged = train_X.merge(train_y[['row_id', 'sub_temp', 'sub_ec']], on='row_id', how='left')

input_cols = [c for c in train_X.columns if c not in ['row_id', 'gh', 'day', 'hour']]

print("=== ENSEMBLE APPROACH WITH MULTIPLE SEEDS ===\n")

# ==================== SUB_TEMP ENSEMBLE ====================
print("=== SUB_TEMP: Temporal Validation with Multiple Seeds ===")

temp_data = merged[merged['sub_temp'].notna()].copy()
X_temp = temp_data[input_cols].fillna(0)
y_temp = temp_data['sub_temp'].values

# Test on F13 and F47
val_results_temp = {test_gh: [] for test_gh in ['F13', 'F47']}
seeds = [42, 123, 456, 789, 999]

for seed in seeds:
    print(f"\nSeed {seed}:")
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=seed
    )
    model.fit(X_temp, y_temp)

    for test_gh in ['F13', 'F47']:
        gh_data = temp_data[temp_data['gh'] == test_gh].sort_values('day')
        train_d = gh_data[gh_data['day'] < 180]
        val_d = gh_data[gh_data['day'] >= 180]

        if len(train_d) > 0 and len(val_d) > 0:
            X_tr = train_d[input_cols].fillna(0)
            y_tr = train_d['sub_temp'].values
            X_vl = val_d[input_cols].fillna(0)
            y_vl = val_d['sub_temp'].values

            m = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=seed
            )
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_vl)
            rmse = np.sqrt(np.mean((y_pred - y_vl) ** 2))
            val_results_temp[test_gh].append(rmse)
            print(f"  {test_gh}: RMSE = {rmse:.4f}")

print("\n=== SUB_TEMP VALIDATION SUMMARY ===")
for test_gh in ['F13', 'F47']:
    rmses = val_results_temp[test_gh]
    print(f"{test_gh}: mean RMSE = {np.mean(rmses):.4f}, std = {np.std(rmses):.4f}, min = {np.min(rmses):.4f}, max = {np.max(rmses):.4f}")

# ==================== SUB_EC ENSEMBLE ====================
print("\n=== SUB_EC: Temporal Validation with Multiple Seeds ===")

ec_data = merged[(merged['sub_ec'].notna()) & (merged['gh'].isin(['F13', 'F47']))].copy()
X_ec = ec_data[input_cols].fillna(0)
y_ec = ec_data['sub_ec'].values

val_results_ec = {test_gh: [] for test_gh in ['F13', 'F47']}

for seed in seeds:
    print(f"\nSeed {seed}:")
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=seed
    )
    model.fit(X_ec, y_ec)

    for test_gh in ['F13', 'F47']:
        gh_data = ec_data[ec_data['gh'] == test_gh].sort_values('day')
        train_d = gh_data[gh_data['day'] < 180]
        val_d = gh_data[gh_data['day'] >= 180]

        if len(train_d) > 0 and len(val_d) > 0:
            X_tr = train_d[input_cols].fillna(0)
            y_tr = train_d['sub_ec'].values
            X_vl = val_d[input_cols].fillna(0)
            y_vl = val_d['sub_ec'].values

            m = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=seed
            )
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_vl)
            rmse = np.sqrt(np.mean((y_pred - y_vl) ** 2))
            val_results_ec[test_gh].append(rmse)
            print(f"  {test_gh}: RMSE = {rmse:.4f}")

print("\n=== SUB_EC VALIDATION SUMMARY ===")
for test_gh in ['F13', 'F47']:
    rmses = val_results_ec[test_gh]
    print(f"{test_gh}: mean RMSE = {np.mean(rmses):.4f}, std = {np.std(rmses):.4f}, min = {np.min(rmses):.4f}, max = {np.max(rmses):.4f}")

# ==================== FINAL ENSEMBLE PREDICTIONS ====================
print("\n=== GENERATING FINAL ENSEMBLE PREDICTIONS ===")

all_temp_models = []
for seed in seeds:
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=seed
    )
    model.fit(X_temp, y_temp)
    all_temp_models.append(model)

all_ec_models = []
for seed in seeds:
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=seed
    )
    model.fit(X_ec, y_ec)
    all_ec_models.append(model)

X_test = test_X[input_cols].fillna(0)

# Ensemble predictions
temp_preds = np.array([m.predict(X_test) for m in all_temp_models])
pred_temp = np.mean(temp_preds, axis=0)

ec_preds = np.array([m.predict(X_test) for m in all_ec_models])
pred_ec = np.mean(ec_preds, axis=0)

# Create submission
submission = pd.DataFrame({
    'row_id': test_X['row_id'].values,
    'sub_temp': pred_temp,
    'sub_ec': pred_ec
})

submission = submission.sort_values('row_id').reset_index(drop=True)

pred_path = r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind\C_haiku\pred_test.csv"
submission.to_csv(pred_path, index=False)
print(f"\n✓ Saved ensemble predictions to pred_test.csv")

print(f"\nEnsemble predictions stats:")
print(f"  sub_temp: mean={pred_temp.mean():.3f}, std={pred_temp.std():.3f}, min={pred_temp.min():.3f}, max={pred_temp.max():.3f}")
print(f"  sub_ec: mean={pred_ec.mean():.3f}, std={pred_ec.std():.3f}, min={pred_ec.min():.3f}, max={pred_ec.max():.3f}")

print("\n=== PREDICTIONS SAMPLES ===")
print(submission.head(10).to_string(index=False))
