# -*- coding: utf-8 -*-
"""DFC1 (2026-10-16 집 클로드; 사용자 '데이터 형태부터 볼까? F13·47 과 나머지 온실 흐름이 많이 다른가? 상황별 및 일반적인 날에도').  서술(판정 없음).  온도.
학습 정답(train_X in_temp + train_y sub_temp)으로 하루(0~23시) 흐름 비교.  집단: F13, F47, 닮은 4곳(F30·F38·F45·F50, 6.520), 나머지 45곳(F32 강한 가온 제외).
날 상황(공통 입력 in_temp 만으로): 추운 날 = 하루 최저 <= 8도 | 보통 날 = 최저 8~15 & 최고 <= 30 | 더운 날 = 최고 > 30 (그 밖은 '기타').
지표(온실·상황별 중앙값 → 집단 중앙값): 시각별 실내·배지·차(배지−실내), 하루 진폭(최고−최저) 비 = 배지/실내, 지연(실내 대비 배지 시각별 상관 최대 지연 0~6h), 배지 하루 평균−실내 하루 평균.
산출: dfc1_daily_flow_v1.png (상황 x [실내·배지 흐름, 차 흐름]), 표 로그.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dfc1_daily_flow_compare_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp"]); Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id", "sub_temp"])
D = X.merge(Y, on="row_id"); D["farm"] = D.row_id.str[:3]; D["day"] = D.row_id.str[4:7].astype(int); D["h"] = D.row_id.str[8:10].astype(int)
D = D[D.in_temp.between(-5, 50) & D.sub_temp.between(-5, 45) & (D.farm != "F32")]
SIM = ["F30", "F38", "F45", "F50"]; D["grp"] = np.where(D.farm.isin(["F13", "F47"]), D.farm, np.where(D.farm.isin(SIM), "닮은 4곳", "나머지 45곳"))
G = D.groupby(["farm", "day"]); full = G.h.transform("size") == 24; D = D[full].copy(); G = D.groupby(["farm", "day"])
dmin, dmax = G.in_temp.transform("min"), G.in_temp.transform("max")
D["sit"] = np.select([dmin <= 8, dmax > 30, (dmin > 8) & (dmin <= 15) & (dmax <= 30)], ["추운 날(최저≤8)", "더운 날(최고>30)", "보통 날"], "기타")
D["gap"] = D.sub_temp - D.in_temp
SITS = ["추운 날(최저≤8)", "보통 날", "더운 날(최고>30)"]; GRPS = ["F13", "F47", "닮은 4곳", "나머지 45곳"]; COL = {"F13": "C3", "F47": "C1", "닮은 4곳": "C2", "나머지 45곳": "C7"}
P = D[D.sit.isin(SITS)].groupby(["grp", "farm", "sit", "h"])[["in_temp", "sub_temp", "gap"]].median().groupby(["grp", "sit", "h"]).median()
def dstats(g):
    a = g.sort_values("h"); amp = (a.sub_temp.max() - a.sub_temp.min()) / max(a.in_temp.max() - a.in_temp.min(), .5)
    c = [np.corrcoef(a.in_temp.values[: 24 - k], a.sub_temp.values[k:])[0, 1] if a.in_temp.std() > 0 and a.sub_temp.std() > 0 else np.nan for k in range(7)]
    return pd.Series(dict(amp_ratio=amp, lag=int(np.nanargmax(c)) if np.isfinite(c).any() else np.nan, mean_gap=a.gap.mean(), in_range=a.in_temp.max() - a.in_temp.min()))
S = D[D.sit.isin(SITS)].groupby(["grp", "farm", "sit", "day"]).apply(dstats).groupby(["grp", "farm", "sit"]).median().groupby(["grp", "sit"]).median()
N = D[D.sit.isin(SITS)].groupby(["grp", "sit"]).apply(lambda g: g.groupby(["farm", "day"]).ngroups).rename("days")
pd.set_option("display.width", 200); print(pd.concat([N, S.round(2)], axis=1).reindex(pd.MultiIndex.from_product([GRPS, SITS])).to_string())
fig, ax = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
for j, s in enumerate(SITS):
    for g in GRPS:
        if (g, s) not in N.index: continue
        q = P.loc[(g, s)]; ax[0, j].plot(q.index, q.sub_temp, "-", color=COL[g], lw=2.2, label=f"{g} 배지"); ax[0, j].plot(q.index, q.in_temp, "--", color=COL[g], lw=1.2, label=f"{g} 실내")
        ax[1, j].plot(q.index, q.gap, "-", color=COL[g], lw=2, label=g)
    ax[0, j].set_title(f"{s}: 실내(점선)·배지(실선)"); ax[1, j].set_title(f"{s}: 배지 − 실내"); ax[1, j].axhline(0, color="k", lw=.6); ax[1, j].set_xlabel("시각")
    ax[0, j].set_ylabel("℃"); ax[1, j].set_ylabel("℃")
ax[0, 0].legend(fontsize=7, ncol=2); ax[1, 0].legend(fontsize=8)
fig.suptitle("하루 흐름 비교: F13·F47 vs 다른 온실 (학습 정답, 온실·상황별 중앙값)", fontsize=13); fig.tight_layout()
out = os.path.join(HERE, "dfc1_daily_flow_v1.png"); fig.savefig(out, dpi=110); print("saved", out)
