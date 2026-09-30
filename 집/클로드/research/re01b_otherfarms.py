# -*- coding: utf-8 -*-
"""재분석 1b (2026-10-01, 집 클로드): 다른 49개 온실의 라벨·입력 가용성과 저온 구간 커버리지."""
import env  # noqa
import os, numpy as np, pandas as pd
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
T = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
D = X.merge(Y, on="row_id")
D["farm"] = D.row_id.str[:3]
o = D[~D.farm.isin(["F13", "F47"])]
f = D[D.farm.isin(["F13", "F47"])]
print("다른 온실 라벨행", len(o), " sub_temp 결측", o.sub_temp.isna().sum())
print("다른 온실 중 외기 입력 있는 행:", o.out_temp.notna().sum(), " 온실:", sorted(o.loc[o.out_temp.notna(), "farm"].unique()))
print("다른 온실 중 act_heating 있는 행:", o.act_heating.notna().sum())
print("in_temp 구간별 행 수 (F13F47 학습 / 다른온실 / test)")
bins = [-99, 4, 6, 8, 10, 12, 99]
for lo, hi in zip(bins[:-1], bins[1:]):
    a = f.in_temp.between(lo, hi, inclusive="left").sum()
    b = o.in_temp.between(lo, hi, inclusive="left").sum()
    c = T.in_temp.between(lo, hi, inclusive="left").sum()
    print(f"  [{lo},{hi}): {a:6d} {b:7d} {c:5d}")
print("다른 온실 in_temp<6 행의 온실 분포:", dict(o.loc[o.in_temp < 6, "farm"].value_counts().head(12)))
bad = o[(o.sub_temp < 0) | (o.sub_temp > 40)]
print("sub_temp 불가능값 25행 온실:", dict(bad.farm.value_counts()), bad.sub_temp.describe().round(2).to_dict())
print("in_temp 범위밖 57행 온실:", dict(o.loc[(o.in_temp < 0) | (o.in_temp > 45), "farm"].value_counts()))
r = o.groupby("farm").apply(lambda g: g[["in_temp", "sub_temp"]].corr().iloc[0, 1])
print("다른 온실 in_temp-sub_temp 상관 분포:", r.describe().round(3).to_dict())
print("F13/F47 in_temp-sub_temp 상관:", f.groupby("farm").apply(lambda g: round(g[["in_temp","sub_temp"]].corr().iloc[0,1],3)).to_dict())
d = (o.sub_temp - o.in_temp); df_ = (f.sub_temp - f.in_temp)
print("sub-in 차 (다른온실 / F13F47) in_temp<8:", round(d[o.in_temp < 8].mean(), 2), round(df_[f.in_temp < 8].mean(), 2),
      " 전체:", round(d.mean(), 2), round(df_.mean(), 2))
