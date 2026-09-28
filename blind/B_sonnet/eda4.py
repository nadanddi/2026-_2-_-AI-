# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:\Users\aozks\OneDrive\바탕 화면\2026_2학기_농업AI경진대회\blind")
import boot
import pandas as pd
import numpy as np
import os

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 400)

train_X = pd.read_csv(os.path.join(boot.DATA, "train_X.csv"))
train_y = pd.read_csv(os.path.join(boot.DATA, "train_y.csv"))
test_X = pd.read_csv(os.path.join(boot.DATA, "test_X.csv"))

def parse_rid(df):
    parts = df["row_id"].str.split("_", expand=True)
    df = df.copy()
    df["gh"] = parts[0]
    df["day"] = parts[1].astype(int)
    df["hour"] = parts[2].astype(int)
    df["t"] = df["day"] * 24 + df["hour"]
    return df

train_X_p = parse_rid(train_X)
test_X_p = parse_rid(test_X)
train_y_p = parse_rid(train_y)
merged = train_X_p.merge(train_y_p[['row_id','sub_temp','sub_ec']], on='row_id', how='left')

out = []
def p(s=""):
    out.append(str(s))

cols = [c for c in train_X.columns if c not in ("row_id",)]

# 1. per-greenhouse row counts and sensor availability (non-missing fraction)
avail = train_X_p.groupby('gh')[cols].apply(lambda d: d.notna().mean())
p("=== Sensor availability (fraction non-missing) per greenhouse ===")
p(avail.round(2).to_string())
p()

# how many greenhouses have each sensor mostly available (>0.5)
p("Number of greenhouses with each sensor >50% available:")
p((avail > 0.5).sum().to_string())
p()

# 2. correlation of in_temp with sub_temp for F13, F47, and pooled
for gh in ["F13","F47"]:
    sub = merged[(merged.gh==gh) & merged.sub_temp.notna()]
    corr = sub[['in_temp','out_temp','out_rad','act_shade','act_thermal','act_heating','act_vent','in_hum','in_co2','sub_temp']].corr()['sub_temp']
    p(f"=== {gh} correlations with sub_temp (n={len(sub)}) ===")
    p(corr.round(3).to_string())
    # simple diff
    diff = sub['sub_temp'] - sub['in_temp']
    p(f"sub_temp - in_temp: mean={diff.mean():.3f}, std={diff.std():.3f}")
    p()

# 3. correlation with sub_ec
for gh in ["F13","F47"]:
    sub = merged[(merged.gh==gh) & merged.sub_ec.notna()]
    corr = sub[['in_temp','out_temp','out_rad','act_shade','act_thermal','act_heating','act_vent','in_hum','in_co2','hour','sub_ec']].corr()['sub_ec']
    p(f"=== {gh} correlations with sub_ec (n={len(sub)}) ===")
    p(corr.round(3).to_string())
    p()

# 4. hour-of-day pattern in sub_ec (mean by hour) to see diurnal effect size vs noise
for gh in ["F13","F47"]:
    sub = merged[(merged.gh==gh) & merged.sub_ec.notna()]
    by_hour = sub.groupby('hour')['sub_ec'].mean()
    p(f"{gh} sub_ec mean by hour: range {by_hour.max()-by_hour.min():.4f} (overall std {sub['sub_ec'].std():.4f})")

p()
# 5. persistence baseline check: how much does sub_ec/sub_temp change hour to hour vs day to day
for gh in ["F13","F47"]:
    sub = merged[(merged.gh==gh) & merged.sub_ec.notna()].sort_values('t')
    sub = sub.set_index('t')
    diffs_1h = sub['sub_ec'].diff()
    p(f"{gh} sub_ec hour-to-hour diff (where consecutive): std={diffs_1h.std():.4f}, mean abs={diffs_1h.abs().mean():.4f}")

with open("eda4_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("done")
