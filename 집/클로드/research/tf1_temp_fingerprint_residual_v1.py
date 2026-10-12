# -*- coding: utf-8 -*-
"""TF1 (2026-10-12 집 클로드; 사용자: '2번 온도 실험' = 운영 일정 지문으로 같은 동을 찾아 온도 하루 오프셋 빌리기).  1단계 = 서술(신호 확인).
질문: 온도 하루 잔차 e_r = mean_h(y − q) (q = DIAG10 OOF, 4칸 평균)가 '일정 지문이 닮은 다른 날 학습 기록'에서 예측되는가.
  (a) 기준: 진단용 연결(truth a1)로 같은 동 k 일 전(k = 1..5) 기록의 잔차와의 상관 — '같은 동이면 얼마나 이어지나' 상한.
  (b) 지문 이웃: 기록마다 구동기 7종 x 24시 일정 벡터(학습 전체로 표준화)로 가까운 학습 기록 K = 1/3/5개 (같은 농장 ±2일, 다른 농장 ±2일 제외 —
      바로 옆날 연속성은 앵커가 이미 쓰는 정보라 빼고 '먼 날짜' 정보만 봄) 의 잔차 평균과의 상관. 전체 하루 일정 사용(비인과 상한).
  (c) 같은 (b)를 0..7시 일정만으로 (새벽에 쓸 수 있는 인과 버전).
  (d) 무작위 이웃(같은 제외) 기준.
보고: 상관 r, 그리고 e_r 을 이웃 평균으로 최소제곱 회귀했을 때 하루 잔차 제곱합 감소율(설명 비율). 1·2차별.
판단(서술): (b) 설명 비율이 5% 미만이면 2단계(사전 고정 검증)로 넘어가지 않음.
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta2", os.path.join(HERE, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s); s.loader.exec_module(A2)
R = A2.A1.frames(); M, cells = R["DIAG10"]
M = M.reset_index(drop=True); M["rec"] = M.farm + "_" + M.day.map("{:03d}".format)
q = np.mean(cells, 0); M["res"] = M.sub_temp.values - q
E = M.groupby("rec").res.mean()
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
X = X[X.rec.isin(E.index)]
ACT = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog"]
recs = [r for r in E.index if np.isfinite(DAY.get(r, np.nan))]
V = np.stack([X.pivot(index="rec", columns="h", values=c).reindex(index=recs, columns=range(24)).values for c in ACT], 2)
V = np.nan_to_num((V - np.nanmean(V, (0, 1))) / (np.nanstd(V, (0, 1)) + 1e-9))
e = E.loc[recs].values; farm = np.array([r[:3] for r in recs]); day = np.array([DAY[r] for r in recs]); p2 = np.array([int(r[4:7]) >= 179 for r in recs])
rng = np.random.default_rng(5)
def report(name, nb, mask=None):
    m = np.isfinite(nb) if mask is None else (np.isfinite(nb) & mask)
    for lab, mm in (("all", m), ("pass-1", m & ~p2), ("pass-2", m & p2)):
        if mm.sum() < 8: continue
        x, y = nb[mm], e[mm]; r = np.corrcoef(x, y)[0, 1]; b = (x * y).sum() / (x * x).sum()
        print(f"  {name:34s} {lab:6s} n {mm.sum():3d} | r {r:+.2f} | explained {1 - ((y - b * x) ** 2).sum() / (y ** 2).sum():+.1%} (beta {b:.2f})")
print(f"day-level residual: records {len(recs)}, sd {e.std():.3f}")
TL = pd.read_pickle(r"C:\work\farmai\집\클로드\research\local\linker_v3\truth.pkl")
from linker_v4.feat4 import FARM, NUM  # noqa
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; pre = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}
Ei = dict(zip(recs, e))
print("(a) same house via truth link, k days back:")
for k in range(1, 6):
    nb = np.full(len(recs), np.nan)
    for i, r in enumerate(recs):
        x = r
        for _ in range(k): x = pre.get(x) if x else None
        if x in Ei and DAY.get(x) == day[i] - k: nb[i] = Ei[x]
    report(f"k={k}", nb)
excl = lambda i: ~((np.abs(day - day[i]) <= 2)) & (np.arange(len(recs)) != i)
for name, hrs in (("(b) fingerprint full day", range(24)), ("(c) fingerprint hours 0-7", range(8))):
    W = V[:, list(hrs), :].reshape(len(recs), -1)
    print(name)
    for K in (1, 3, 5):
        nb = np.full(len(recs), np.nan)
        for i in range(len(recs)):
            ok = np.where(excl(i))[0]; d = ((W[ok] - W[i]) ** 2).sum(1); nb[i] = e[ok[np.argsort(d)[:K]]].mean()
        report(f"K={K}", nb)
print("(d) random neighbours (same exclusion), K=5:")
nb = np.array([e[rng.choice(np.where(excl(i))[0], 5, replace=False)].mean() for i in range(len(recs))]); report("random K=5", nb)
