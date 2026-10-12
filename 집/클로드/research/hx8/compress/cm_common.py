# -*- coding: utf-8 -*-
"""hx8/compress 공용 로더 (2026-10-13 집 클로드).
- r: 460 기록(base.pkl 순서 = feat4 순서), rid = F13_217 형식
- A: 460 x 24 x 14 입력 (train_X + test_X 원값)
- Ytr: 학습 정답 460 x 24 (평가 NaN)
- oof(validator): 기록별 OOF(CUR, 시드 평균) 시간별 sg / nosg  -> dict rid -> (24,)
- links: 진단용 같은 동 전날 연결(truth.pkl cert, 학습끼리) prev[b]=a
- cal: 학습 전용 달력(dmt3e_E0_days.csv) rid -> day
- sub22: 13회차 평가 EC 예측 rid -> (24,)
"""
import sys, os, pickle, warnings
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
S3 = os.path.join(env.ROOT, u"집", u"클로드", "research", "local", "linker_v3")
R = pickle.load(open(os.path.join(S3, "base.pkl"), "rb"))
r = R["recs"].reset_index(drop=True)
r["rid"] = r.farm + "_" + r.day.map(lambda d: f"{d:03d}")
FARM = r.farm.values; NUM = r.day.values.astype(int); TRAIN = (r.split == "train").values; N = len(r)
rid2i = dict(zip(r.rid, r.index))
CH = ["in_temp", "in_hum", "in_co2", "out_temp", "out_hum", "out_rad", "out_wspd",
      "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
Xtr = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); Xte = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
Yt = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
XX = pd.concat([Xtr, Xte], ignore_index=True)
XX["rid"] = XX.row_id.str[:7]; XX["h"] = XX.row_id.str[-2:].astype(int)
A = np.full((N, 24, len(CH)), np.nan)
for rid, g in XX.sort_values(["rid", "h"]).groupby("rid"):
    if rid in rid2i and len(g) == 24:
        A[rid2i[rid]] = g[CH].values
Yt["rid"] = Yt.row_id.str[:7]; Yt["h"] = Yt.row_id.str[-2:].astype(int)
Ytr = np.full((N, 24), np.nan)
for rid, g in Yt.sort_values(["rid", "h"]).groupby("rid"):
    if rid in rid2i:
        Ytr[rid2i[rid]] = g.sub_ec.values
ydm = np.nanmean(Ytr, 1)  # 학습 하루 평균 (평가 NaN)

T = pd.read_pickle(os.path.join(S3, "truth.pkl"))
Tc = T[T.cert].copy()
prev = dict(zip(Tc.b.astype(int), Tc.a1.astype(int)))   # b 의 같은 동 전날 = a1 (학습끼리, 진단용)
nxt = {a: b for b, a in prev.items()}

D = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv"))
cal = dict(zip(D.rec, D.day))

O = pd.read_csv(r"C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv")
O = O[O.tag == "CUR"].copy()
O["rid"] = O.row_id.str[:7]


def oof(validator, col="sg", seed=None):
    o = O[O.validator == validator]
    if seed is not None:
        o = o[o.seed == seed]
    g = o.groupby(["rid", "hour"])[col].mean().unstack()
    fold = o.groupby("rid").fold.first().to_dict()
    return {k: g.loc[k].values for k in g.index}, fold


SEEDS = sorted(O.seed.unique())
S22 = pd.read_csv(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)", "submission_22.csv"))
S22["rid"] = S22.row_id.str[:7]; S22["h"] = S22.row_id.str[-2:].astype(int)
sub22 = {k: g.sort_values("h").sub_ec.values for k, g in S22.groupby("rid")}

EVAL8 = ["F13_221", "F13_223", "F13_225", "F13_237", "F47_221", "F47_223", "F47_225", "F47_227"]


def episodes_true(thr=1.0):
    """진단 연결로 고EC(하루 평균 >= thr) 연속 사슬 → 기록별 (episode id, 경과일 E, 사슬 위치)."""
    hi = {i for i in range(N) if TRAIN[i] and ydm[i] >= thr}
    ep, E = {}, {}
    for i in sorted(hi):
        # 거슬러 올라가 시작점 찾기
        j, k = i, 0
        while j in prev and prev[j] in hi:
            j = prev[j]; k += 1
            if k > 60: break
        ep[i] = j; E[i] = k
    return hi, ep, E


def rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2)))
