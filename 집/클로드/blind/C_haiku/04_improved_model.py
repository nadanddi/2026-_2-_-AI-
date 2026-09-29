import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")

import boot
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
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
train_y['hour'] = train_y['row_id'].apply(lambda x: parse_row_id(x)[2])

test_X['gh'] = test_X['row_id'].apply(lambda x: parse_row_id(x)[0])
test_X['day'] = test_X['row_id'].apply(lambda x: parse_row_id(x)[1])
test_X['hour'] = test_X['row_id'].apply(lambda x: parse_row_id(x)[2])

# Merge
merged = train_X.merge(train_y[['row_id', 'sub_temp', 'sub_ec']], on='row_id', how='left')

# Base input columns
base_input_cols = [c for c in train_X.columns if c not in ['row_id', 'gh', 'day', 'hour']]

# Add time-based features
merged['hour_sin'] = np.sin(2 * np.pi * merged['hour'] / 24)
merged['hour_cos'] = np.cos(2 * np.pi * merged['hour'] / 24)
merged['day_sin'] = np.sin(2 * np.pi * merged['day'] / 365)
merged['day_cos'] = np.cos(2 * np.pi * merged['day'] / 365)

# Add day-of-week-like feature
merged['hour_category'] = pd.cut(merged['hour'], bins=[0, 6, 12, 18, 24], labels=False)

test_X['hour_sin'] = np.sin(2 * np.pi * test_X['hour'] / 24)
test_X['hour_cos'] = np.cos(2 * np.pi * test_X['hour'] / 24)
test_X['day_sin'] = np.sin(2 * np.pi * test_X['day'] / 365)
test_X['day_cos'] = np.cos(2 * np.pi * test_X['day'] / 365)
test_X['hour_category'] = pd.cut(test_X['hour'], bins=[0, 6, 12, 18, 24], labels=False)

all_input_cols = base_input_cols + ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'hour_category']

print("=== IMPROVED MODEL WITH FEATURE ENGINEERING ===")
print(f"Input features: {len(all_input_cols)}")

def fill_missing_smart(df, cols):
    """Fill missing values by greenhouse and hour pattern"""
    df = df.copy()
    for col in cols:
        if df[col].isna().sum() > 0:
            # Fill with greenhouse-hour mean
            fill_values = df.groupby(['gh', 'hour'])[col].transform('mean')
            mask = df[col].isna()
            df.loc[mask, col] = fill_values[mask]

            # Fill remaining with greenhouse mean
            fill_values = df.groupby('gh')[col].transform('mean')
            mask = df[col].isna()
            df.loc[mask, col] = fill_values[mask]

            # Fill remaining with overall mean
            df[col].fillna(df[col].mean(), inplace=True)
    return df

# ==================== SUB_TEMP MODEL ====================
print("\n=== TRAINING IMPROVED SUB_TEMP MODEL ===")

temp_data = merged[merged['sub_temp'].notna()].copy()
print(f"Training samples: {len(temp_data)}")

# Smart fill for train
temp_data = fill_missing_smart(temp_data, base_input_cols)

X_temp = temp_data[all_input_cols].fillna(0)
y_temp = temp_data['sub_temp'].values

print("Training with LightGBM...")
temp_model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
temp_model.fit(X_temp, y_temp)

# Temporal validation
print("\nTemporal validation (F13, F47)...")
for test_gh in ['F13', 'F47']:
    gh_data = temp_data[temp_data['gh'] == test_gh].sort_values('day')
    train_d = gh_data[gh_data['day'] < 180]
    val_d = gh_data[gh_data['day'] >= 180]

    if len(train_d) > 0 and len(val_d) > 0:
        X_tr = train_d[all_input_cols].fillna(0)
        y_tr = train_d['sub_temp'].values
        X_vl = val_d[all_input_cols].fillna(0)
        y_vl = val_d['sub_temp'].values

        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_vl)
        rmse = np.sqrt(np.mean((y_pred - y_vl) ** 2))
        print(f"  {test_gh}: RMSE = {rmse:.4f}")

# ==================== SUB_EC MODEL ====================
print("\n=== TRAINING IMPROVED SUB_EC MODEL ===")

ec_data = merged[(merged['sub_ec'].notna()) & (merged['gh'].isin(['F13', 'F47']))].copy()
print(f"Training samples: {len(ec_data)}")

# Smart fill for EC
ec_data = fill_missing_smart(ec_data, base_input_cols)

X_ec = ec_data[all_input_cols].fillna(0)
y_ec = ec_data['sub_ec'].values

print("Training with LightGBM...")
ec_model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
ec_model.fit(X_ec, y_ec)

# Temporal validation
print("\nTemporal validation (EC data)...")
for test_gh in ['F13', 'F47']:
    gh_data = ec_data[ec_data['gh'] == test_gh].sort_values('day')
    train_d = gh_data[gh_data['day'] < 180]
    val_d = gh_data[gh_data['day'] >= 180]

    if len(train_d) > 0 and len(val_d) > 0:
        X_tr = train_d[all_input_cols].fillna(0)
        y_tr = train_d['sub_ec'].values
        X_vl = val_d[all_input_cols].fillna(0)
        y_vl = val_d['sub_ec'].values

        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_vl)
        rmse = np.sqrt(np.mean((y_pred - y_vl) ** 2))
        print(f"  {test_gh}: RMSE = {rmse:.4f}")

# ==================== GENERATE PREDICTIONS ====================
print("\n=== GENERATING TEST PREDICTIONS ===")

# Prepare test data
test_prep = test_X.copy()
test_prep = fill_missing_smart(test_prep, base_input_cols)
X_test = test_prep[all_input_cols].fillna(0)

pred_temp = temp_model.predict(X_test)
pred_ec = ec_model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'row_id': test_X['row_id'].values,
    'sub_temp': pred_temp,
    'sub_ec': pred_ec
})

submission = submission.sort_values('row_id').reset_index(drop=True)

pred_path = r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind\C_haiku\pred_test.csv"
submission.to_csv(pred_path, index=False)
print(f"✓ Saved predictions to pred_test.csv")

print(f"\nPredictions stats:")
print(f"  sub_temp: mean={pred_temp.mean():.3f}, std={pred_temp.std():.3f}, min={pred_temp.min():.3f}, max={pred_temp.max():.3f}")
print(f"  sub_ec: mean={pred_ec.mean():.3f}, std={pred_ec.std():.3f}, min={pred_ec.min():.3f}, max={pred_ec.max():.3f}")

# Feature importance
print("\n=== TOP 10 IMPORTANT FEATURES (TEMP) ===")
feature_imp = pd.DataFrame({
    'feature': all_input_cols,
    'importance': temp_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_imp.head(10).to_string(index=False))

print("\n=== TOP 10 IMPORTANT FEATURES (EC) ===")
feature_imp_ec = pd.DataFrame({
    'feature': all_input_cols,
    'importance': ec_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_imp_ec.head(10).to_string(index=False))
