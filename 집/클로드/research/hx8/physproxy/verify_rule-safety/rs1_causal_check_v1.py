# -*- coding: utf-8 -*-
"""규정 감사(physproxy): 평가 8기록 AHx 인과판을 test_X 의 그 기록 자기 행 0..h 만으로 독립 재계산.
(1) 시간별 cm[h] 가 h 이후 입력을 바꿔도 불변인지(미래 입력 섞임 검사) (2) PP4 (C) 의 하루 스칼라와 일치하는지
(3) 하루 평균 보정 = 시간별 인과 보정 a+b·cm[h] 의 하루 평균과 같은지 (4) 결측(ffill 영향) 확인. 계수는 PP4 로그 값(학습 40기록) 그대로."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
EV8 = ["F13_221", "F13_223", "F13_225", "F13_237", "F47_221", "F47_223", "F47_225", "F47_227"]
X = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
def es(T): return 0.6108 * np.exp(17.27 * T / (T + 237.3))
def ah(T, RH): return 2165.0 * (RH / 100.0) * es(T) / (T + 273.15)
ac, bc = -0.558, 0.158
S22 = pd.read_csv(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)", "submission_22.csv")); S22["rec"] = S22.row_id.str[:7]
s22 = S22.groupby("rec").sub_ec.mean()
for r in EV8:
    g = X[X.rec == r].sort_values("h").set_index("h").reindex(range(24))
    cols = ["in_temp", "in_hum", "out_temp", "out_hum"]
    nnan = int(g[cols].isna().sum().sum())
    def cm_at(gg, h):
        sub = gg.loc[:h, cols].ffill()          # 자기 기록 0..h 행만
        d = ah(sub.in_temp.values, sub.in_hum.values) - ah(sub.out_temp.values, np.clip(sub.out_hum.values, 1, 100))
        return float(np.nanmean(d)) if np.isfinite(d).any() else np.nan
    cm = np.array([cm_at(g, h) for h in range(24)])
    # 미래 행 교란: h 이후 값을 크게 바꿔도 cm[h] 불변이어야 함
    maxdiff = 0.0
    for h in [0, 5, 11, 17]:
        gp = g.copy(); gp.loc[h + 1:, cols] = gp.loc[h + 1:, cols] * 1.5 + 3
        maxdiff = max(maxdiff, abs(cm_at(gp, h) - cm[h]))
    ahxc = float(np.nanmean(cm))
    hourly_corr_mean = float(np.nanmean(ac + bc * cm))
    print(f"{r}: NaN cells {nnan} | AHxc {ahxc:.3f} | future-perturb max|dcm| {maxdiff:.2e} | dayshift {ac+bc*ahxc:+.4f} vs mean(hourly causal shift) {hourly_corr_mean:+.4f} | ref {s22[r]+ac+bc*ahxc:.3f}")
