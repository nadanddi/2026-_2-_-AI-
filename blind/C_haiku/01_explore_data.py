import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")

import boot
import pandas as pd
import numpy as np
from collections import defaultdict

data_path = r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\온라인대회자료\정형데이터\참가자_배포"

# Load data
train_X = pd.read_csv(f"{data_path}/train_X.csv")
train_y = pd.read_csv(f"{data_path}/train_y.csv")
test_X = pd.read_csv(f"{data_path}/test_X.csv")
sample_sub = pd.read_csv(f"{data_path}/sample_submission.csv")

print("=== DATA SHAPE ===")
print(f"train_X: {train_X.shape}")
print(f"train_y: {train_y.shape}")
print(f"test_X: {test_X.shape}")
print(f"sample_submission: {sample_sub.shape}")

print("\n=== TRAIN_X COLUMNS ===")
print(train_X.columns.tolist())

print("\n=== TRAIN_Y COLUMNS ===")
print(train_y.columns.tolist())

# Analyze row_id structure
print("\n=== ROW_ID STRUCTURE ===")
print("Sample row_ids from train_X:")
print(train_X['row_id'].head(10).tolist())

# Parse row_id to extract greenhouse, day, hour
def parse_row_id(row_id):
    parts = row_id.split('_')
    gh_id = parts[0]
    day = int(parts[1])
    hour = int(parts[2])
    return gh_id, day, hour

# Analyze greenhouses
gh_info = defaultdict(lambda: {'min_day': float('inf'), 'max_day': 0, 'count': 0})
for row_id in train_X['row_id']:
    gh, day, hour = parse_row_id(row_id)
    gh_info[gh]['min_day'] = min(gh_info[gh]['min_day'], day)
    gh_info[gh]['max_day'] = max(gh_info[gh]['max_day'], day)
    gh_info[gh]['count'] += 1

print("\nGreenhouses in train_X:")
for gh in sorted(gh_info.keys()):
    info = gh_info[gh]
    print(f"  {gh}: {info['count']:6d} rows, days {info['min_day']:3d}-{info['max_day']:3d}")

# Analyze test data greenhouses
test_gh_info = defaultdict(lambda: {'min_day': float('inf'), 'max_day': 0, 'count': 0})
for row_id in test_X['row_id']:
    gh, day, hour = parse_row_id(row_id)
    test_gh_info[gh]['min_day'] = min(test_gh_info[gh]['min_day'], day)
    test_gh_info[gh]['max_day'] = max(test_gh_info[gh]['max_day'], day)
    test_gh_info[gh]['count'] += 1

print("\nGreenhouses in test_X:")
for gh in sorted(test_gh_info.keys()):
    info = test_gh_info[gh]
    print(f"  {gh}: {info['count']:6d} rows, days {info['min_day']:3d}-{info['max_day']:3d}")

# Analyze missing values
print("\n=== MISSING VALUES ===")
print("train_X missing counts:")
missing_counts = train_X.isnull().sum()
print(missing_counts[missing_counts > 0].to_dict())

print("\ntrain_y missing counts:")
print(f"  sub_temp: {train_y['sub_temp'].isnull().sum()}")
print(f"  sub_ec: {train_y['sub_ec'].isnull().sum()}")

# Merge to check alignment
merged = train_X.merge(train_y, on='row_id', how='left')
print(f"\nAfter merging train_X and train_y on row_id:")
print(f"  Total rows: {len(merged)}")
print(f"  Rows with both temp and ec: {(~merged['sub_temp'].isnull() & ~merged['sub_ec'].isnull()).sum()}")
print(f"  Rows with only temp: {(~merged['sub_temp'].isnull() & merged['sub_ec'].isnull()).sum()}")
print(f"  Rows with only ec: {(merged['sub_temp'].isnull() & ~merged['sub_ec'].isnull()).sum()}")
print(f"  Rows with neither: {(merged['sub_temp'].isnull() & merged['sub_ec'].isnull()).sum()}")

# Target statistics
print("\n=== TARGET STATISTICS ===")
valid_mask = ~merged['sub_temp'].isnull()
print(f"sub_temp (n={valid_mask.sum()}):")
print(f"  mean: {merged.loc[valid_mask, 'sub_temp'].mean():.3f}")
print(f"  std:  {merged.loc[valid_mask, 'sub_temp'].std():.3f}")
print(f"  min:  {merged.loc[valid_mask, 'sub_temp'].min():.3f}")
print(f"  max:  {merged.loc[valid_mask, 'sub_temp'].max():.3f}")

valid_ec = ~merged['sub_ec'].isnull()
print(f"\nsub_ec (n={valid_ec.sum()}):")
print(f"  mean: {merged.loc[valid_ec, 'sub_ec'].mean():.3f}")
print(f"  std:  {merged.loc[valid_ec, 'sub_ec'].std():.3f}")
print(f"  min:  {merged.loc[valid_ec, 'sub_ec'].min():.3f}")
print(f"  max:  {merged.loc[valid_ec, 'sub_ec'].max():.3f}")

# Input statistics
print("\n=== INPUT STATISTICS ===")
input_cols = [c for c in train_X.columns if c != 'row_id']
for col in input_cols:
    non_null = train_X[col].notna().sum()
    if non_null > 0:
        print(f"{col:15s}: mean={train_X[col].mean():8.2f}, std={train_X[col].std():8.2f}, miss={train_X[col].isna().sum():6d}")

# Save merged data for further analysis
merged.to_csv(r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind\C_haiku\merged_data.csv", index=False)
print("\n✓ Saved merged_data.csv")
