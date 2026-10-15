# -*- coding: utf-8 -*-
"""DHM5 (2026-10-16 집 클로드; 사용자 '쌍둥이 조건일 때 답(실제 배지온도)도 다른가?').  서술(판정 없음), 실행 전 커밋.  온도.
DHM4 의 '크게 틀린 기록 ↔ 조건 최근접 맞힌 기록' 쌍마다(배치 평균, CV8·N·CK2·EL1 의 19회차 구성 최종 예측):
  하루 평균 — 실제 배지, 예측 배지, 실내, (실제 배지 − 실내), (예측 − 실내), 0시·12시 실제 배지; 시각별 흐름 그림(대표 쌍 6개).
판단: 두 기록의 실내 하루 평균 차 vs 실제 배지 하루 평균 차 vs 예측 하루 평균 차 비교 — '입력 같고 정답만 다름' 이면 예측 차 ≈ 실내 차, 정답 차가 큼.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dhm5_twin_pairs_truth_v1.py
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
R = pd.read_csv(os.path.join(HERE, "dgx5_rows_v1.csv.gz"))
H = R.groupby(["rec", "h"]).agg(y=("y", "mean"), fin=("fin", "mean"), t=("t", "mean")).reset_index()
d = H.groupby("rec").agg(y=("y", "mean"), fin=("fin", "mean"), t=("t", "mean")); d["gap_true"] = d.y - d.t; d["gap_pred"] = d.fin - d.t
h0 = H[H.h == 0].set_index("rec").y; h12 = H[H.h == 12].set_index("rec").y
rows = []
for a, b in PAIRS:
    if a not in d.index or b not in d.index: continue
    A, Bb = d.loc[a], d.loc[b]
    rows.append(dict(miss=a, hit=b, in_mean=f"{A.t:.1f}|{Bb.t:.1f}", true_sub=f"{A.y:.1f}|{Bb.y:.1f}", pred_sub=f"{A.fin:.1f}|{Bb.fin:.1f}",
                     gap_true=f"{A.gap_true:+.1f}|{Bb.gap_true:+.1f}", gap_pred=f"{A.gap_pred:+.1f}|{Bb.gap_pred:+.1f}",
                     d_in=round(A.t - Bb.t, 2), d_true=round(A.y - Bb.y, 2), d_pred=round(A.fin - Bb.fin, 2), true0=f"{h0.get(a, np.nan):.1f}|{h0.get(b, np.nan):.1f}"))
T = pd.DataFrame(rows); pd.set_option("display.width", 230); print(T.to_string(index=False))
print(f"\nmedian |indoor diff| {T.d_in.abs().median():.2f} | median |true substrate diff| {T.d_true.abs().median():.2f} | median |pred diff| {T.d_pred.abs().median():.2f}")
print(f"share of pairs where |true diff - indoor diff| > 1C: {np.mean(np.abs(T.d_true - T.d_in) > 1):.2f} | where |pred diff - indoor diff| > 1C: {np.mean(np.abs(T.d_pred - T.d_in) > 1):.2f}")
SHOW = [("F13_107", "F13_103"), ("F13_139", "F13_153"), ("F47_241", "F47_151"), ("F47_240", "F47_172"), ("F13_109", "F13_091"), ("F47_191", "F47_078")]
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
for k, (a, b) in enumerate(SHOW):
    x = ax.flat[k]
    for r, c in ((a, "C3"), (b, "C0")):
        q = H[H.rec == r].sort_values("h")
        x.plot(q.h, q.y, "-", color=c, lw=2.2, label=f"{r} 실제 배지"); x.plot(q.h, q.fin, ":", color=c, lw=1.8, label=f"{r} 예측 배지"); x.plot(q.h, q.t, "--", color=c, lw=1, alpha=.7, label=f"{r} 실내")
    x.set_title(f"틀린 {a} (빨강) vs 맞힌 {b} (파랑)"); x.set_xlabel("시각"); x.set_ylabel("℃"); x.legend(fontsize=7)
fig.suptitle("쌍둥이 조건 날: 실선 = 실제 배지, 점선 = 예측 배지, 파선 = 실내 공기", fontsize=13); fig.tight_layout()
out = os.path.join(HERE, "dhm5_twin_pairs_v1.png"); fig.savefig(out, dpi=105); print("saved", out)
