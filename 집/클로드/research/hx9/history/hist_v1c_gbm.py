# -*- coding: utf-8 -*-
"""HX9 / history family — v1c (2026-10-13 집 클로드). v1b C2 에서 본 'GBM 이 고EC 잔차 MSE 를 0.161 → 0.134 로 줄이고
12개 날짜 이동 위약 모두보다 좋다' 를 더 엄하게 확인한다. hist_v1*.py 는 고치지 않는다.

계산 전에 적은 판정 규칙:
  K1. 위약을 |s| = 5..35 일 전부(62개)로 늘려 P(위약 hi10 MSE <= 실제) 를 다시 잰다. < .05 이어야 '날짜 식별표 이상' 이다.
  K2. sg 만 넣은 GBM (날씨 없음) 의 hi10 MSE 를 같이 보고, 이득 중 sg 비선형 몫을 뺀다.
  K3. 보통 기록(hi10=0) MSE 도 같이 본다. 고EC 이득이 보통 기록 손해로 상쇄되면 '쓸 수 있는 신호' 가 아니다.
  K4. min_samples_leaf 10/20/40 세 값에서 방향이 같아야 한다.
  K5. 변수 묶음(일사·ET, 온도, 습도·VPD, 일사 무관 시간 변수)별로 따로 넣어 어디서 오는지 서술 (선택에 쓰지 않음).
  K6. 사용 가능한 DMT3e 판(결측은 GBM 이 그대로 처리)도 같은 블록 CV 로 잰다 (위약 불가, 서술).
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
PRIMARY_U = ["rad_p1", "rad_p3", "rad_p5", "rad_p7", "rad_p14", "et0_p1", "et0_p3", "et0_p7", "vpd_p1", "vpd_p3", "vpd_p7",
             "tout_p1", "tout_p3", "tout_p7", "tmin_p3", "hum_p3", "ttrend5", "tchg1", "radchg1", "dsince_hirad", "lowrad_run"]
DV = PRIMARY_U + ["rad_d0", "et0_d0"]
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
blks = A.blk.values
splits = [(np.where(blks != b)[0], np.where(blks == b)[0]) for b in np.unique(blks)]
def shifted(cols, s):
    dd = A.dm11_day.values
    return np.array([[T11v.at[int(d + s), v] if (np.isfinite(d) and 1 <= d + s <= 126) else np.nan for v in cols] for d in dd])
def gbm(Xd, leaf=20):
    Xf = sg[:, None] if Xd is None else np.c_[sg, Xd]
    p = np.zeros(len(A)); pb = np.zeros(len(A))
    for tr, te in splits:
        w = np.linalg.lstsq(np.c_[np.ones(len(tr)), sg[tr]], yv[tr], rcond=None)[0]; pb[te] = np.c_[np.ones(len(te)), sg[te]] @ w
        p[te] = HistGradientBoostingRegressor(max_depth=2, max_iter=100, learning_rate=0.05, min_samples_leaf=leaf, random_state=0).fit(Xf[tr], yv[tr]).predict(Xf[te])
    return dict(hi=np.mean((yv - p)[hi] ** 2), no=np.mean((yv - p)[no] ** 2), all=np.mean((yv - p) ** 2),
                hi_b=np.mean((yv - pb)[hi] ** 2), no_b=np.mean((yv - pb)[no] ** 2), all_b=np.mean((yv - pb) ** 2),
                imp=int(np.sum(np.abs(yv - p)[hi] < np.abs(yv - pb)[hi])))
SH = [s for s in range(-35, 36) if abs(s) >= 5]
for leaf in (10, 20, 40):
    a = gbm(shifted(DV, 0), leaf); s0 = gbm(None, leaf)
    pl = [gbm(shifted(DV, s), leaf) for s in SH]
    ph = np.array([x["hi"] for x in pl]); pn = np.array([x["no"] for x in pl]); pa = np.array([x["all"] for x in pl])
    print(f"leaf {leaf}: linear-sg base MSE hi {a['hi_b']:.4f} normal {a['no_b']:.4f} all {a['all_b']:.4f} | GBM sg-only hi {s0['hi']:.4f} "
          f"normal {s0['no']:.4f} all {s0['all']:.4f} | GBM sg+history hi {a['hi']:.4f} (improves {a['imp']}/{int(hi.sum())}) normal {a['no']:.4f} "
          f"all {a['all']:.4f}")
    print(f"     placebo (62 shifts) hi median {np.median(ph):.4f} min {ph.min():.4f}; P(placebo hi <= actual) {np.mean(ph <= a['hi']):.3f} | "
          f"normal median {np.median(pn):.4f}, P(placebo normal <= actual) {np.mean(pn <= a['no']):.3f} | all median {np.median(pa):.4f}, "
          f"P(placebo all <= actual) {np.mean(pa <= a['all']):.3f}")
print("\nK5 variable groups (leaf 20), actual vs placebo hi10 MSE:")
GR = {"rad/ET": ["rad_p1", "rad_p3", "rad_p5", "rad_p7", "rad_p14", "et0_p1", "et0_p3", "et0_p7", "radchg1", "dsince_hirad", "lowrad_run", "rad_d0", "et0_d0"],
      "temperature": ["tout_p1", "tout_p3", "tout_p7", "tmin_p3", "ttrend5", "tchg1"],
      "VPD/humidity": ["vpd_p1", "vpd_p3", "vpd_p7", "hum_p3"]}
for k, cols in GR.items():
    a = gbm(shifted(cols, 0)); pl = np.array([gbm(shifted(cols, s))["hi"] for s in SH[::2]])
    print(f"  {k:13s} hi {a['hi']:.4f} normal {a['no']:.4f} all {a['all']:.4f} | placebo hi median {np.median(pl):.4f}, P(<= actual) {np.mean(pl <= a['hi']):.3f} (31 shifts)")
print("\nK6 usable DMT3e version (NaN where the training calendar has no previous day), same block CV:")
for leaf in (10, 20, 40):
    a = gbm(A[DV].values, leaf)
    print(f"  leaf {leaf}: hi {a['hi']:.4f} (improves {a['imp']}/{int(hi.sum())}) normal {a['no']:.4f} all {a['all']:.4f} "
          f"(linear base hi {a['hi_b']:.4f} normal {a['no_b']:.4f} all {a['all_b']:.4f})")
