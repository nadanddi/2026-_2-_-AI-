# -*- coding: utf-8 -*-
"""HX9 / history family — v1d (2026-10-13 집 클로드). 사후(post-hoc) 강건성 점검. hist_v1*.py 는 고치지 않는다.

배경: v1c K5 에서 'VPD·습도 이력 묶음(vpd_p1, vpd_p3, vpd_p7, hum_p3)' 만 넣은 GBM(leaf 20)이 고EC 잔차 MSE 를
0.161 → 0.137 로 줄였고 31개 날짜 이동 위약 모두보다 좋았다(P .000). 전체 MSE 도 0.0259 → 0.0247 로 유일하게 줄었다.
단, K5 는 '서술용, 선택에 쓰지 않음' 으로 정했고 3개 묶음 중 하나를 결과를 보고 고른 것이므로 이 점검은 확증이 아니라
'우연·설정 의존인지' 를 보는 진단이다 (같은 자료 재사용 → 통과해도 채택 근거 아님, 다중비교: 묶음 3개 × leaf 3개).

계산 전에 적은 판정 규칙:
  V1. leaf 10/20/40 세 값 모두에서 hi10 MSE < 선형 sg 기준이고 62개 위약 대비 P <= .05 여야 '설정 무관' 이라 부른다.
  V2. 보통 기록 MSE 와 전체 MSE 가 선형 기준보다 나빠지지 않아야 한다 (고EC 이득이 일반 손해로 상쇄되면 쓸모 없음).
  V3. 이득(기록별 |오차| 감소)이 한 에피소드에 몰리지 않아야 한다: 가장 큰 기여 에피소드 하나를 빼도 hi10 MSE 이득 > 0.
  V4. 1차(num<179)/2차 기록 각각에서 hi10 이득 방향이 같아야 한다 (2차 hi10 은 5개 → 서술).
  V5. 사용 가능한 DMT3e 판(평가에서 계산 가능한 판)도 같은 방향이어야 한다 (위약 불가, 서술).
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
HX9 = os.path.dirname(HERE)
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
V = pd.read_csv(os.path.join(HERE, "vars_v1.csv")).set_index("rec")
src = open(os.path.join(HERE, "hist_v1.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(HERE, "hist_v1.py"), "__name__": "hist_v1_defs"}
exec(compile(src[:src.index("VD = {}")], "hist_v1_defs", "exec"), ns)
T_11, date_vars = ns["T_11"], ns["date_vars"]
VG = ["vpd_p1", "vpd_p3", "vpd_p7", "hum_p3"]
T11v = {}
for d_ in range(1, 127):
    def g(k, d_=d_):
        dd = d_ - k
        if dd < 1 or dd > 126 or not np.isfinite(T_11.loc[dd, "rad"]): return None
        return T_11.loc[dd]
    T11v[d_] = date_vars(g)
T11v = pd.DataFrame.from_dict(T11v, orient="index")

A = R[R.has_sg == 1].join(V)
A["blk"] = [f"b{int(d // 14)}" if np.isfinite(d) else f"r{r}" for r, d in zip(A.index, A.dm11_day)]
yv = A.resid_mean.values; sg = A.sg_mean.values; hi = A.hi10.values == 1; no = ~hi
p2 = A.pass2.values == 1; ep = A.ep_id.values
blks = A.blk.values
splits = [(np.where(blks != b)[0], np.where(blks == b)[0]) for b in np.unique(blks)]

def shifted(cols, s):
    dd = A.dm11_day.values
    return np.array([[T11v.at[int(d + s), v] if (np.isfinite(d) and 1 <= d + s <= 126) else np.nan for v in cols] for d in dd])

def fit(Xd, leaf):
    Xf = np.c_[sg, Xd]
    p = np.zeros(len(A)); pb = np.zeros(len(A))
    for tr, te in splits:
        w = np.linalg.lstsq(np.c_[np.ones(len(tr)), sg[tr]], yv[tr], rcond=None)[0]; pb[te] = np.c_[np.ones(len(te)), sg[te]] @ w
        p[te] = HistGradientBoostingRegressor(max_depth=2, max_iter=100, learning_rate=0.05, min_samples_leaf=leaf,
                                              random_state=0).fit(Xf[tr], yv[tr]).predict(Xf[te])
    return p, pb

def mse(e, m): return float(np.mean(e[m] ** 2)) if m.sum() else np.nan

SH = [s for s in range(-35, 36) if abs(s) >= 5]
for leaf in (10, 20, 40):
    p, pb = fit(shifted(VG, 0), leaf)
    e, eb = yv - p, yv - pb
    pl = []
    for s in SH:
        q, _ = fit(shifted(VG, s), leaf); pl.append(((yv - q)[hi] ** 2).mean())
    pl = np.array(pl)
    print(f"leaf {leaf}: hi MSE base {mse(eb, hi):.4f} -> {mse(e, hi):.4f} | normal {mse(eb, no):.4f} -> {mse(e, no):.4f} | "
          f"all {mse(eb, no | hi):.4f} -> {mse(e, no | hi):.4f} | hi improved {int(np.sum(np.abs(e[hi]) < np.abs(eb[hi])))}/{int(hi.sum())} | "
          f"placebo hi median {np.median(pl):.4f}, P(<= actual) {np.mean(pl <= mse(e, hi)):.3f} (62 shifts)")
    # V3 episode concentration
    d = (eb ** 2 - e ** 2)
    contrib = pd.Series(d[hi]).groupby(ep[hi]).sum().sort_values(ascending=False)
    top = contrib.index[0]
    m_wo = hi & (ep != top)
    print(f"     gain by episode (sum of sq-err reduction): {dict(contrib.round(3))}")
    print(f"     drop top episode {top}: hi MSE base {mse(eb, m_wo):.4f} -> {mse(e, m_wo):.4f}")
    # V4 pass split
    print(f"     pass-1 hi: {mse(eb, hi & ~p2):.4f} -> {mse(e, hi & ~p2):.4f} (n {int((hi & ~p2).sum())}) | pass-2 hi: "
          f"{mse(eb, hi & p2):.4f} -> {mse(e, hi & p2):.4f} (n {int((hi & p2).sum())}) | pass-2 normal {mse(eb, no & p2):.4f} -> {mse(e, no & p2):.4f}")
    # V5 usable DMT3e version
    pu, _ = fit(A[VG].values, leaf); eu = yv - pu
    print(f"     usable DMT3e: hi {mse(eb, hi):.4f} -> {mse(eu, hi):.4f} | normal {mse(eb, no):.4f} -> {mse(eu, no):.4f} | "
          f"all {mse(eb, no | hi):.4f} -> {mse(eu, no | hi):.4f} | hi improved {int(np.sum(np.abs(eu[hi]) < np.abs(eb[hi])))}/{int(hi.sum())}")
