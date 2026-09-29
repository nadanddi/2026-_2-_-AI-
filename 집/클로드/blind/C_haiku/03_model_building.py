import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")

import boot
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

data_path = r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\온라인대회자료\정형데이터\참가자_배포"

# Load data
train_X = pd.read_csv(f"{data_path}/train_X.csv")
train_y = pd.read_csv(f"{data_path}/train_y.csv")
test_X = pd.read_csv(f"{data_path}/test_X.csv")
sample_sub = pd.read_csv(f"{data_path}/sample_submission.csv")

def parse_row_id(row_id):
    parts = row_id.split('_')
    gh_id = parts[0]
    day = int(parts[1])
    hour = int(parts[2])
    return gh_id, day, hour

# Add greenhouse, day, hour to all dataframes
train_X['gh'] = train_X['row_id'].apply(lambda x: parse_row_id(x)[0])
train_X['day'] = train_X['row_id'].apply(lambda x: parse_row_id(x)[1])
train_X['hour'] = train_X['row_id'].apply(lambda x: parse_row_id(x)[2])

train_y['gh'] = train_y['row_id'].apply(lambda x: parse_row_id(x)[0])
train_y['day'] = train_y['row_id'].apply(lambda x: parse_row_id(x)[1])
train_y['hour'] = train_y['row_id'].apply(lambda x: parse_row_id(x)[2])

test_X['gh'] = test_X['row_id'].apply(lambda x: parse_row_id(x)[0])
test_X['day'] = test_X['row_id'].apply(lambda x: parse_row_id(x)[1])
test_X['hour'] = test_X['row_id'].apply(lambda x: parse_row_id(x)[2])

# Merge train_X and train_y
merged = train_X.merge(train_y[['row_id', 'sub_temp', 'sub_ec']], on='row_id', how='left')

# Input columns (exclude row_id and derived columns)
input_cols = [c for c in train_X.columns if c not in ['row_id', 'gh', 'day', 'hour']]

print("=== VALIDATION STRATEGY ===")
print("1. For sub_temp: Temporal split by day within each greenhouse")
print("2. For sub_ec: Only use F13 and F47 (the only greenhouses with EC data)")
print("")

# ==================== SUB_TEMP MODEL ====================
print("=== TRAINING SUB_TEMP MODEL ===")

# Get rows with sub_temp labels
temp_data = merged[merged['sub_temp'].notna()].copy()
print(f"Total training samples for temp: {len(temp_data)}")

# Handle missing values - use 0 for missing (as a placeholder, or forward fill)
X_temp = temp_data[input_cols].fillna(0)
y_temp = temp_data['sub_temp'].values

# Train on all available data (for final submission)
print("Training temp model on all available data...")
temp_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)
temp_model.fit(X_temp, y_temp)

# Validate with temporal split on test greenhouses
print("\nTemporal validation on test greenhouses (F13, F47)...")
for test_gh in ['F13', 'F47']:
    gh_temp_data = temp_data[temp_data['gh'] == test_gh].sort_values('day')

    # Use days up to 180 for training, 181+ for validation
    train_data = gh_temp_data[gh_temp_data['day'] < 180]
    val_data = gh_temp_data[gh_temp_data['day'] >= 180]

    if len(train_data) > 0 and len(val_data) > 0:
        X_train_val = train_data[input_cols].fillna(0)
        y_train_val = train_data['sub_temp'].values

        X_val = val_data[input_cols].fillna(0)
        y_val = val_data['sub_temp'].values

        model_val = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        model_val.fit(X_train_val, y_train_val)

        y_pred = model_val.predict(X_val)
        rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
        print(f"  {test_gh}: RMSE = {rmse:.4f} (n_val={len(val_data)})")
    else:
        print(f"  {test_gh}: Not enough data for validation")

# ==================== SUB_EC MODEL ====================
print("\n=== TRAINING SUB_EC MODEL ===")

# EC data only for F13 and F47
ec_data = merged[(merged['sub_ec'].notna()) & (merged['gh'].isin(['F13', 'F47']))].copy()
print(f"Total training samples for EC: {len(ec_data)}")

if len(ec_data) > 0:
    X_ec = ec_data[input_cols].fillna(0)
    y_ec = ec_data['sub_ec'].values

    # Train on all available EC data
    print("Training EC model on all available data...")
    ec_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    ec_model.fit(X_ec, y_ec)

    # Temporal validation
    print("\nTemporal validation on EC data...")
    for test_gh in ['F13', 'F47']:
        gh_ec_data = ec_data[ec_data['gh'] == test_gh].sort_values('day')

        # Use days up to 180 for training, 181+ for validation
        train_data = gh_ec_data[gh_ec_data['day'] < 180]
        val_data = gh_ec_data[gh_ec_data['day'] >= 180]

        if len(train_data) > 0 and len(val_data) > 0:
            X_train_val = train_data[input_cols].fillna(0)
            y_train_val = train_data['sub_ec'].values

            X_val = val_data[input_cols].fillna(0)
            y_val = val_data['sub_ec'].values

            model_val = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            )
            model_val.fit(X_train_val, y_train_val)

            y_pred = model_val.predict(X_val)
            rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
            print(f"  {test_gh}: RMSE = {rmse:.4f} (n_val={len(val_data)})")

# ==================== GENERATE PREDICTIONS ====================
print("\n=== GENERATING TEST PREDICTIONS ===")

X_test = test_X[input_cols].fillna(0)

pred_temp = temp_model.predict(X_test)
pred_ec = ec_model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'row_id': test_X['row_id'].values,
    'sub_temp': pred_temp,
    'sub_ec': pred_ec
})

# Sort by row_id to match sample_submission order
submission = submission.sort_values('row_id').reset_index(drop=True)

# Save predictions
pred_path = r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind\C_haiku\pred_test.csv"
submission.to_csv(pred_path, index=False)
print(f"✓ Saved predictions to pred_test.csv")

print("\nSample predictions:")
print(submission.head(10))
print(f"\nPredictions stats:")
print(f"  sub_temp: mean={pred_temp.mean():.3f}, std={pred_temp.std():.3f}")
print(f"  sub_ec: mean={pred_ec.mean():.3f}, std={pred_ec.std():.3f}")
