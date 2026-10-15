# -*- coding: utf-8 -*-
"""DHM6 (2026-10-16 집 클로드; 사용자 '쌍둥이 날들의 하루 동안 조건이 어떻게 움직였는지도 비교').  서술(판정 없음), 실행 전 커밋.  온도.
DHM4/5 쌍(크게 틀린 기록 ↔ 조건 최근접 맞힌 기록) 19쌍: 입력 14종 시각별 흐름 비교.
(1) 쌍마다 입력별 시각 차 RMS(표준화 단위)를 '맞힌 기록끼리 최근접 쌍' 의 같은 값 분포(기준)와 비교 -> 쌍둥이인데도 유난히 다르게 움직인 입력 찾기
    (틀린-맞힌 쌍의 중앙 / 맞힌-맞힌 쌍의 중앙 비, 그리고 틀린-맞힌 쌍 중 기준 90% 분위를 넘는 비율)
(2) 시각대별(0~5, 6~11, 12~17, 18~23) 어느 시간대에서 다른가
(3) 그림: 대표 6쌍 x 입력 6종(실내 온도·습도·CO2, 외기 온도·일사, 난방·보온커튼) 흐름  -> dhm6_twin_inputs_v1.png
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dhm6_twin_hourly_inputs_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = [("F47_191", "F47_078"), ("F47_178", "F47_049"), ("F13_109", "F13_091"), ("F13_139", "F13_153"), ("F47_123", "F47_120"), ("F47_018", "F47_024"), ("F47_241", "F47_151"),
         ("F47_240", "F47_172"), ("F47_096", "F47_127"), ("F13_107", "F13_103"), ("F47_105", "F47_211"), ("F47_132", "F47_136"), ("F47_095", "F47_083"), ("F13_175", "F13_172"),
         ("F47_145", "F47_120"), ("F13_244", "F13_152"), ("F13_157", "F13_170"), ("F47_239", "F47_150"), ("F47_214", "F47_161")]
COLS = ["in_temp", "in_hum", "in_co2", "out_temp", "out_hum", "out_rad", "out_wspd", "act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog"]
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy(); X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:10].astype(int)
sd = X[COLS].std(); P = {c: X.pivot(index="rec", columns="h", values=c) for c in COLS}
def prof_diff(a, b, c, hrs=range(24)):
    va, vb = P[c].loc[a].reindex(hrs).values, P[c].loc[b].reindex(hrs).values; m = np.isfinite(va) & np.isfinite(vb)
    return float(np.sqrt(np.mean(((va[m] - vb[m]) / sd[c]) ** 2))) if m.sum() >= 3 else np.nan
# baseline: hit-hit nearest pairs from DHM4 feature space
src = open(os.path.join(HERE, "dhm4_miss_vs_nearest_hit_v1.py"), encoding="utf-8").read(); exec(src[src.index("src = open"):src.index("hh = dist")])
hits = G[hit].rec.values; dh = dist[np.ix_(hit, hit)]; BASEP = [(hits[i], hits[int(np.argmin(dh[i]))]) for i in range(len(hits))]
rows = []
for c in COLS:
    tw = np.array([prof_diff(a, b, c) for a, b in PAIRS]); bs = np.array([prof_diff(a, b, c) for a, b in BASEP])
    q90 = np.nanquantile(bs, .9); rows.append(dict(input=c, miss_hit_med=np.nanmedian(tw), hit_hit_med=np.nanmedian(bs), ratio=np.nanmedian(tw) / max(np.nanmedian(bs), 1e-9), share_above_hit_q90=np.nanmean(tw > q90)))
T = pd.DataFrame(rows).sort_values("ratio", ascending=False); pd.set_option("display.width", 200)
print("(1) hourly-profile difference (standardised RMS) of twin pairs vs hit-hit nearest pairs"); print(T.round(2).to_string(index=False))
rows = []
for c in COLS:
    r = dict(input=c)
    for nm, hrs in (("00-05", range(0, 6)), ("06-11", range(6, 12)), ("12-17", range(12, 18)), ("18-23", range(18, 24))):
        tw = np.nanmedian([prof_diff(a, b, c, hrs) for a, b in PAIRS]); bs = np.nanmedian([prof_diff(a, b, c, hrs) for a, b in BASEP]); r[nm] = tw / max(bs, 1e-9)
    rows.append(r)
print("\n(2) ratio (twin / hit-hit) by hour block"); print(pd.DataFrame(rows).round(2).to_string(index=False))
SHOW = [("F47_191", "F47_078"), ("F13_139", "F13_153"), ("F47_240", "F47_172"), ("F47_241", "F47_151"), ("F13_107", "F13_103"), ("F47_123", "F47_120")]
VARS = [("in_temp", "실내 온도"), ("in_hum", "실내 습도"), ("in_co2", "실내 CO2"), ("out_temp", "외기 온도"), ("out_rad", "외기 일사"), ("act_heating", "난방"), ("act_thermal", "보온커튼")]
fig, ax = plt.subplots(len(SHOW), len(VARS), figsize=(22, 17), sharex=True)
for i, (a, b) in enumerate(SHOW):
    for j, (c, lab) in enumerate(VARS):
        x = ax[i, j]; x.plot(range(24), P[c].loc[a].reindex(range(24)), "-", color="C3", lw=1.8, label=f"틀린 {a}"); x.plot(range(24), P[c].loc[b].reindex(range(24)), "-", color="C0", lw=1.8, label=f"맞힌 {b}")
        if i == 0: x.set_title(lab)
        if j == 0: x.set_ylabel(f"{a}\nvs {b}", fontsize=9); x.legend(fontsize=7)
fig.suptitle("쌍둥이 조건 날의 하루 입력 흐름 (빨강 = 크게 틀린 날, 파랑 = 맞힌 날)", fontsize=14); fig.tight_layout()
out = os.path.join(HERE, "dhm6_twin_inputs_v1.png"); fig.savefig(out, dpi=90); print("saved", out)
