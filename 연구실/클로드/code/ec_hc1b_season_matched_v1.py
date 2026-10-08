# -*- coding: utf-8 -*-
"""HC1b — HC1 의 평가·학습 차이가 계절(추위) 때문인지: 하루 평균 외기온 구간을 맞춰 비교 (서술) — 2026-10-08 연구실 클로드
외기온 일평균 구간(<3, 3~6, 6~10, ≥10℃) 안에서 TE vs (TR1+TR2) 하루 평균 |Δ| 중앙값 비, 구간별 날 수로 가중한 층화 비교(van Elteren 대신 구간별 Mann-Whitney 의 Stouffer 결합)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, norm
R = os.path.dirname(os.path.abspath(__file__))
DA = pd.read_csv(os.path.join(R, "..", "results", "ec_hc1_day_change_v1.csv")); DA = DA[DA.grp.isin(["TR1", "TR2", "TE"])]
def load(fn):
    d = pd.read_csv(os.path.join(env.DATA, fn)); d["farm"] = d.row_id.str[:3]; d["day"] = d.row_id.str[4:7].astype(int); return d
X = pd.concat([load("train_X.csv"), load("test_X.csv")]); X = X[X.farm.isin(["F13", "F47"])]
ot = X.groupby(["farm", "day"]).out_temp.mean().rename("ot").reset_index(); DA = DA.merge(ot, on=["farm", "day"])
DA["bin"] = pd.cut(DA.ot, [-99, 3, 6, 10, 99], labels=["<3", "3~6", "6~10", "≥10"])
print("구간별 날 수:", DA.groupby(["bin", "grp"], observed=True).size().unstack().fillna(0).astype(int).to_dict("index"))
V = [c[:-2] for c in DA.columns if c.endswith("_m")]
for v in V:
    zs, ws, cells = [], [], []
    for b, g in DA.groupby("bin", observed=True):
        a = g.loc[g.grp == "TE", v + "_m"].dropna(); t = g.loc[g.grp != "TE", v + "_m"].dropna()
        if len(a) < 4 or len(t) < 4:
            continue
        u = mannwhitneyu(a, t, alternative="two-sided"); n1, n2 = len(a), len(t)
        z = (u.statistic - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12); zs.append(z); ws.append(np.sqrt(n1 * n2 / (n1 + n2)))
        cells.append(f"{b}: TE {a.mean():.2f} 학습 {t.mean():.2f}")
    zc = np.dot(ws, zs) / np.sqrt(np.dot(ws, ws)) if zs else np.nan; p = 2 * norm.sf(abs(zc)) if zs else np.nan
    print(f"{v:12s} 층화 z {zc:+.2f} p {p:.2e}{' **' if p < .05/14 else (' *' if p < .05 else '')} | " + " | ".join(cells))
