# -*- coding: utf-8 -*-
"""EIA1 (2026-10-13 집 클로드; 사용자: 'EC 정보원 찾아봐'). 서술 전용(DM11·뒷날 입력 사용 — 규정상 모델엔 못 씀, 정보가 '있기는 한지' 지도).
평가 60 기록마다 정답 없이(DM11 달력 ±1일 + 실내 자정 이음 + 운영 지문 거리) 같은 동 사슬을 앞(전날)·뒤(다음날)로 따라가
  처음 만나는 정답 기록까지 단계 수 k_b, k_f 와 그 값(앞: 23시, 뒤: 0시).  학습 대응표(NB3 6.688)로 단계별 0시/23시 오차 σ(k) 를 주고
  '앞뒤 정답을 최선으로 쓴다면' 기록별 기대 수준 오차 → 하루 모양 오차(Z0 잔여)와 합쳐 기대 RMSE 를 셈.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X, IN_, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
DAY = np.array([D11.get(rid(i), np.nan) for i in range(N)])
sdi = np.nanstd(IN_[TRAIN].reshape(-1, IN_.shape[2]), 0)
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
tr = TRAIN; FPA = (FPA - np.nanmean(FPA[tr], (0, 1))) / (np.nanstd(FPA[tr], (0, 1)) + 1e-9)
fpd = lambda a, b: float(np.nanmean(np.minimum(np.abs(FPA[a] - FPA[b]), 4)))
jn = lambda a, b: float(np.nanmean(np.abs(IN_[a, 23] - IN_[b, 0]) / sdi))   # a = earlier day, b = later day
def step(i, dirn):
    c = np.where(DAY == DAY[i] + dirn)[0]
    if len(c) == 0: return None
    s = [jn(k, i) + fpd(k, i) if dirn < 0 else jn(i, k) + fpd(k, i) for k in c]
    return c[int(np.argmin(s))]
def walk(i, dirn, kmax=12):
    cur = i
    for k in range(1, kmax + 1):
        nx = step(cur, dirn)
        if nx is None: return None, None, None
        if TRAIN[nx] and np.isfinite(EC[nx, 0]): return k, rid(nx), EC[nx, 23] if dirn < 0 else EC[nx, 0]
        cur = nx
    return None, None, None
# step-error table sigma(k) from training: walk on labelled pass-2 & pass-1 targets, compare anchor to own 0h (back) / 23h (fwd)
rows = []
for i in np.where(TRAIN & np.isfinite(DAY))[0]:
    for dirn in (-1, 1):
        # force at least: first step only to a labelled neighbour = k=1; to estimate k>=2 skip labelled ones at intermediate steps is not possible -> use chain as found
        k, r, v = walk(i, dirn)
        if k: rows.append(dict(dirn=dirn, k=k, err=v - (EC[i, 0] if dirn < 0 else EC[i, 23]), hi=np.nanmean(EC[i]) >= 1.0))
T = pd.DataFrame(rows); T["ae"] = T.err.abs()
print("training label-free chain anchor error by direction & steps (to own 0h for back, own 23h for fwd):")
print(T.groupby(["dirn", "k"]).agg(n=("ae", "size"), rmse=("err", lambda s: float(np.sqrt(np.mean(s ** 2)))), med=("ae", "median"), within10=("ae", lambda s: (s < .1).mean())).round(3).to_string())
sig = T.groupby(["dirn", "k"]).err.apply(lambda s: float(np.sqrt(np.mean(s ** 2)))).to_dict()
P12 = pd.read_csv(os.path.join(env.ROOT, "제출", "12회차_2026-10-11(나)", "submission_19.csv")); P12["rec"] = P12.row_id.str[:7]; pm = P12.groupby("rec").sub_ec.mean()
E = []
for i in [i for i in range(N) if not TRAIN[i] and FARM[i] in ("F13", "F47")]:
    kb, rb, vb = walk(i, -1); kf, rf, vf = walk(i, 1)
    sb = sig.get((-1, kb), .30) if kb else np.inf; sf = sig.get((1, kf), .30) if kf else np.inf
    s = 1 / np.sqrt((1 / sb ** 2 if np.isfinite(sb) else 0) + (1 / sf ** 2 if np.isfinite(sf) else 0)) if (np.isfinite(sb) or np.isfinite(sf)) else np.nan
    E.append(dict(rec=rid(i), day=DAY[i], kb=kb, back=rb, vb=vb, kf=kf, fwd=rf, vf=vf, sig=s, p12=pm[rid(i)]))
E = pd.DataFrame(E); pd.set_option("display.width", 250)
print("\neval availability:"); print(E.round(3).to_string())
print("\ncounts: back k", E.kb.value_counts(dropna=False).to_dict(), "| fwd k", E.kf.value_counts(dropna=False).to_dict(), "| neither", int((E.kb.isna() & E.kf.isna()).sum()))
for shape in (.03, .05):
    for none_sig in (.15, .25, .40):
        s2 = np.where(E.sig.notna(), E.sig ** 2, none_sig ** 2) + shape ** 2
        print(f"expected eval RMSE if best use of label neighbours: shape {shape}, no-info sigma {none_sig} -> {np.sqrt(s2.mean()):.4f}")
E.to_csv(os.path.join(env.LOCAL, "eia1_eval_availability.csv"), index=False)
