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

# Parse row_id
def parse_row_id(row_id):
    parts = row_id.split('_')
    gh_id = parts[0]
    day = int(parts[1])
    hour = int(parts[2])
    return gh_id, day, hour

# Check which greenhouses have EC data
train_y['gh'] = train_y['row_id'].apply(lambda x: parse_row_id(x)[0])
train_y['day'] = train_y['row_id'].apply(lambda x: parse_row_id(x)[1])
train_y['hour'] = train_y['row_id'].apply(lambda x: parse_row_id(x)[2])

ec_by_gh = train_y.groupby('gh').agg({
    'sub_ec': lambda x: x.notna().sum(),
    'sub_temp': lambda x: x.notna().sum()
}).reset_index()
ec_by_gh.columns = ['gh', 'ec_count', 'temp_count']
ec_by_gh = ec_by_gh.sort_values('ec_count', ascending=False)

print("=== EC DATA BY GREENHOUSE ===")
print(ec_by_gh.to_string(index=False))

# Find greenhouses with EC data
gh_with_ec = ec_by_gh[ec_by_gh['ec_count'] > 0]['gh'].tolist()
print(f"\nGreenhouses with EC data: {gh_with_ec}")
print(f"Total greenhouses with EC: {len(gh_with_ec)} out of {len(ec_by_gh)}")

# Check day ranges for EC data
print("\n=== EC DATA RANGES ===")
for gh in gh_with_ec[:10]:  # First 10
    gh_data = train_y[train_y['gh'] == gh]
    ec_data = gh_data[gh_data['sub_ec'].notna()]
    if len(ec_data) > 0:
        print(f"{gh}: days {ec_data['day'].min()}-{ec_data['day'].max()}, {len(ec_data)} samples")

# Check test greenhouses
print("\n=== TEST GREENHOUSES ===")
test_X['gh'] = test_X['row_id'].apply(lambda x: parse_row_id(x)[0])
test_X['day'] = test_X['row_id'].apply(lambda x: parse_row_id(x)[1])
test_X['hour'] = test_X['row_id'].apply(lambda x: parse_row_id(x)[2])

for gh in test_X['gh'].unique():
    test_gh = test_X[test_X['gh'] == gh]
    print(f"{gh}: days {test_gh['day'].min()}-{test_gh['day'].max()}, {len(test_gh)} samples")

# Check if test greenhouses have EC training data
print("\n=== EC DATA IN TEST GREENHOUSES ===")
for gh in ['F13', 'F47']:
    if gh in gh_with_ec:
        gh_ec = train_y[(train_y['gh'] == gh) & (train_y['sub_ec'].notna())]
        print(f"{gh}: {len(gh_ec)} EC samples, days {gh_ec['day'].min()}-{gh_ec['day'].max()}")
    else:
        print(f"{gh}: NO EC DATA in training set")

# Check F47 EC data specifically
print("\n=== F47 DETAILED ANALYSIS ===")
f47_ec = train_y[(train_y['gh'] == 'F47') & (train_y['sub_ec'].notna())].sort_values('day')
print(f"Total F47 EC samples: {len(f47_ec)}")
if len(f47_ec) > 0:
    print(f"Day range: {f47_ec['day'].min()}-{f47_ec['day'].max()}")
    # Count by day
    by_day = f47_ec.groupby('day').size()
    print(f"Samples per day: min={by_day.min()}, max={by_day.max()}, mean={by_day.mean():.1f}")
    print(f"Days with EC data: {len(by_day)} out of {f47_ec['day'].max() - f47_ec['day'].min() + 1}")

# Check F13 EC data
print("\n=== F13 DETAILED ANALYSIS ===")
f13_ec = train_y[(train_y['gh'] == 'F13') & (train_y['sub_ec'].notna())].sort_values('day')
print(f"Total F13 EC samples: {len(f13_ec)}")
if len(f13_ec) > 0:
    print(f"Day range: {f13_ec['day'].min()}-{f13_ec['day'].max()}")
    by_day = f13_ec.groupby('day').size()
    print(f"Samples per day: min={by_day.min()}, max={by_day.max()}, mean={by_day.mean():.1f}")
    print(f"Days with EC data: {len(by_day)} out of {f13_ec['day'].max() - f13_ec['day'].min() + 1}")

# Check what hours have EC data
print("\n=== EC DATA BY HOUR (F47) ===")
if len(f47_ec) > 0:
    by_hour = f47_ec.groupby('hour').size().sort_index()
    print(by_hour.to_dict())

print("\n=== EC DATA BY HOUR (F13) ===")
if len(f13_ec) > 0:
    by_hour = f13_ec.groupby('hour').size().sort_index()
    print(by_hour.to_dict())
