import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")

import boot
import pandas as pd
import numpy as np

data_path = r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\온라인대회자료\정형데이터\참가자_배포"

# Load data
train_X = pd.read_csv(f"{data_path}/train_X.csv")
train_y = pd.read_csv(f"{data_path}/train_y.csv")
test_X = pd.read_csv(f"{data_path}/test_X.csv")
pred = pd.read_csv(r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind\C_haiku\pred_test.csv")

def parse_row_id(row_id):
    parts = row_id.split('_')
    return parts[0], int(parts[1]), int(parts[2])

# Statistics
print("=== 최종 데이터 통계 ===\n")

print("1. 훈련 데이터:")
print(f"   - train_X: {len(train_X):,}행, {len([c for c in train_X.columns if c != 'row_id']):,}개 입력")
print(f"   - train_y: {len(train_y):,}행, 온도 {train_y['sub_temp'].notna().sum():,}개, EC {train_y['sub_ec'].notna().sum():,}개")

print("\n2. 평가 데이터:")
print(f"   - test_X: {len(test_X):,}행 (F13: {len(test_X[test_X['row_id'].str.startswith('F13')])}, F47: {len(test_X[test_X['row_id'].str.startswith('F47')])})")

print("\n3. 목표 변수 분포:")
print(f"   - sub_temp (n={train_y['sub_temp'].notna().sum()}):")
print(f"     mean: {train_y['sub_temp'].mean():.2f}°C, std: {train_y['sub_temp'].std():.2f}°C")
print(f"     min: {train_y['sub_temp'].min():.2f}°C, max: {train_y['sub_temp'].max():.2f}°C")

print(f"\n   - sub_ec (n={train_y['sub_ec'].notna().sum()}):")
print(f"     mean: {train_y['sub_ec'].mean():.3f} dS/m, std: {train_y['sub_ec'].std():.3f} dS/m")
print(f"     min: {train_y['sub_ec'].min():.3f} dS/m, max: {train_y['sub_ec'].max():.3f} dS/m")

print("\n4. 예측값 분포:")
print(f"   - sub_temp (predicted):")
print(f"     mean: {pred['sub_temp'].mean():.2f}°C, std: {pred['sub_temp'].std():.2f}°C")
print(f"     min: {pred['sub_temp'].min():.2f}°C, max: {pred['sub_temp'].max():.2f}°C")

print(f"\n   - sub_ec (predicted):")
print(f"     mean: {pred['sub_ec'].mean():.3f} dS/m, std: {pred['sub_ec'].std():.3f} dS/m")
print(f"     min: {pred['sub_ec'].min():.3f} dS/m, max: {pred['sub_ec'].max():.3f} dS/m")

print("\n=== EC 데이터 특이성 ===")
train_y['gh'] = train_y['row_id'].apply(lambda x: parse_row_id(x)[0])
ec_by_gh = train_y.groupby('gh')['sub_ec'].apply(lambda x: x.notna().sum())
print(f"EC 데이터가 있는 온실: {(ec_by_gh > 0).sum()}개 중 51개 온실")
print(f"EC 데이터 있는 온실: {ec_by_gh[ec_by_gh > 0].index.tolist()}")
print(f"각 온실당 EC 샘플: {ec_by_gh[ec_by_gh > 0].values}")

print("\n=== 파일 목록 ===")
import os
work_dir = r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind\C_haiku"
files = os.listdir(work_dir)
for f in sorted(files):
    if not f.startswith('.'):
        file_path = os.path.join(work_dir, f)
        size = os.path.getsize(file_path)
        print(f"  {f}: {size:,} bytes")
