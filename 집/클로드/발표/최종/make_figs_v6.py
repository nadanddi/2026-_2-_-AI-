# 집 최종 v6 그림(PNG) 만들기 (2026-11-04, 집 클로드). 한쇼·PowerPoint에서 똑같이 보이도록 차트 대신 그림 파일로 넣는다.
# 자료: data/example_day_v1.json(dlv1d), data/cold_bins_v1.json(dlv1e), 제출 모델 β_h, 제출/성능_정리.md 회차별 온도 공개 점수
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "fig")
os.makedirs(OUT, exist_ok=True)
for f in ("malgun.ttf", "malgunbd.ttf"):
    p = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", f)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8A8FA6",
                     "axes.labelcolor": "#3A3F57", "xtick.color": "#3A3F57", "ytick.color": "#3A3F57"})
INK, ACC, ACC2, MUTE, SAND = "#1F2452", "#6B69CF", "#2F59AA", "#8A8FA6", "#CFCFEA"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


# 1) 실제 하루: 정답과 예측 두 선만
ex = json.load(open(os.path.join(D, "example_day_v1.json"), encoding="utf-8"))
fig, ax = plt.subplots(figsize=(6.2, 2.9))
ax.plot(ex["hour"], ex["truth"], color=INK, lw=2.6, label="정답")
ax.plot(ex["hour"], ex["pred"], color=ACC, lw=2.6, label="기준 모델 예측")
ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xlim(0, 23)
ax.set_ylabel("배지 온도(℃)"); ax.set_xlabel("시각")
ax.annotate("", xy=(5, ex["pred"][5]), xytext=(5, ex["truth"][5]), arrowprops=dict(arrowstyle="<->", color=MUTE, lw=1.6))
ax.text(0.6, 21.3, f"하루 내내 약 {abs(ex['mean_error']):.1f}℃ 밀림", va="center", color=INK, fontsize=12)
ax.legend(frameon=False, loc="lower right", fontsize=11)
ax.grid(axis="y", color="#E6E6EF", lw=0.8)
save(fig, "fig_example_day.png")

# 2) 실내 온도별 오차 크기
cb = json.load(open(os.path.join(D, "cold_bins_v1.json"), encoding="utf-8"))
fig, ax = plt.subplots(figsize=(5.6, 3.1))
cols = [ACC if i < 4 else SAND for i in range(len(cb["bins"]))]
b = ax.bar(cb["bins"], cb["rmse"], color=cols, width=0.62)
for r, v in zip(b, cb["rmse"]):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=11, color=INK)
ax.set_ylabel("오차 RMSE(℃)"); ax.set_xlabel("실내 온도(℃)"); ax.set_ylim(0, 1.4)
ax.grid(axis="y", color="#E6E6EF", lw=0.8); ax.set_axisbelow(True)
save(fig, "fig_cold_rmse.png")

# 3) 고EC 날과 보통 날의 치우침(정답 − 예측)
fig, ax = plt.subplots(figsize=(5.6, 3.1))
x = range(len(cb["bins"]))
hi = [v if v is not None else float("nan") for v in cb["hiec_mean"]]
ax.axhline(0, color=MUTE, lw=1)
ax.plot(x, hi, color=ACC2, lw=2.6, marker="o", label="EC가 높은 날")
ax.plot(x, cb["low_mean"], color=MUTE, lw=2.0, marker="o", label="보통 날")
ax.set_xticks(list(x)); ax.set_xticklabels(cb["bins"])
ax.set_ylabel("정답 - 예측(℃)"); ax.set_xlabel("실내 온도(℃)"); ax.set_ylim(-0.7, 0.7)
ax.text(1, 0.55, "낮게 예측", color=ACC2, fontsize=11)
ax.legend(frameon=False, loc="lower left", fontsize=11)
ax.grid(axis="y", color="#E6E6EF", lw=0.8); ax.set_axisbelow(True)
save(fig, "fig_cold_hiec.png")

# 4) 전날 실측 보정 반영률(제출 모델 값), 9시 이후 '측정 흔들림' 띠
B = [0.7648, 0.6232, 0.5525, 0.4844, 0.4091, 0.3453, 0.3025, 0.2576, 0.1866, 0.1264, 0.157, 0.1661, 0.1655, 0.1566, 0.1999, 0.2032, 0.2233, 0.2263, 0.2145, 0.2008, 0.1897, 0.1721, 0.1991, 0.2355]
fig, ax = plt.subplots(figsize=(5.4, 3.3))
ax.axvspan(9, 23, color="#EFEFF4")
ax.text(16, 70, "측정 흔들림", ha="center", color=MUTE, fontsize=12)
ax.plot(range(24), [v * 100 for v in B], color=ACC, lw=2.6, marker="o", ms=4)
ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xlim(0, 23); ax.set_ylim(0, 80)
ax.set_ylabel("반영률(%)"); ax.set_xlabel("오늘 시각")
ax.grid(axis="y", color="#E6E6EF", lw=0.8); ax.set_axisbelow(True)
save(fig, "fig_beta.png")

# 5) 회차별 온도 공개 점수(1~24회차), 중요 지점·실패·판정 색
S = {1: .7450, 2: .6666, 3: .5624, 4: .5697, 5: .5456, 6: .5251, 7: .5193, 8: .5160, 9: .5127, 10: .5126, 11: .5089, 12: .5012,
     13: .4883, 14: .4864, 15: .4853, 16: .4861, 17: .4817, 18: .4922, 19: .4706, 20: .4754, 21: .4685, 22: .5000, 23: .4665, 24: .4608}
r = sorted(S); y = [S[k] for k in r]
fig, ax = plt.subplots(figsize=(11.2, 3.9))
ax.plot(r, y, color=SAND, lw=2, zorder=1)
pas, fail, lb = {19, 21}, {20, 22}, {23, 24}
for k in r:
    c = ACC if k in pas else (MUTE if k in fail else (ACC2 if k in lb else "#B8BAD6"))
    ax.scatter(k, S[k], s=55 if k in pas | fail | lb else 30, color=c, zorder=3)
# 이름표 자리(텍스트 x, y, 정렬, 연결선)를 손으로 잡아 겹치지 않게
LAB = [(1, "부스팅 트리 하나로 시작", 1.4, 0.745, "left", False),
       (3, "배지가 공기를 늦게 따라가는 식", 3.2, 0.580, "left", False),
       (6, "실내 온도별 세 모델 혼합", 5.8, 0.556, "left", False),
       (12, "전날 23시 실측으로 새벽 보정", 6.6, 0.477, "left", True),
       (17, "대회 표식 빼고 흐름으로 전날 찾기", 10.8, 0.452, "left", True),
       (21, "전날 실측 없는 날은 전날 예측값으로", 15.2, 0.425, "left", True),
       (20, "기록을 전날 순서로 다시 연결", 19.9, 0.533, "right", True),
       (22, "추운 날 보정 강화", 22.0, 0.513, "center", False),
       (24, "최종", 24.1, 0.482, "center", False)]
for k, t, tx, ty, ha, line in LAB:
    ax.annotate(t, (k, S[k]), xytext=(tx, ty), fontsize=11, color=MUTE if k in fail else INK, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color="#C9CBDA", lw=0.8) if line else None)
ax.set_xticks(r); ax.set_xlim(0.5, 24.8); ax.set_ylim(0.41, 0.76)
ax.set_ylabel("배지 온도 RMSE(℃)"); ax.set_xlabel("제출 회차")
ax.grid(axis="y", color="#E6E6EF", lw=0.8); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=ACC, label="사전 기준 통과"), Line2D([], [], marker="o", ls="", color=MUTE, label="사전 기준 미달"),
     Line2D([], [], marker="o", ls="", color=ACC2, label="공개 점수로 선택")]
ax.legend(handles=h, frameon=False, loc="upper right", fontsize=11)
save(fig, "fig_rounds.png")
print("ok", sorted(os.listdir(OUT)))
