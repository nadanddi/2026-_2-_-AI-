# -*- coding: utf-8 -*-
"""ORC1 (2026-10-13 집 클로드; 사용자: 0.05권이 규정 안이라면 → '사슬(같은 동 여러 단계) 식별이 완벽하면 EC 가 얼마까지 가나' 오라클). 서술(채택 판정 아님).
진짜 전날 = truth.pkl a1 중 DM11 달력 차 1 (DM11 은 평가 입력 사용 → 서술 전용).  검증 기하마다 대상 기록 r 에서 진짜 전날을 따라 거슬러 올라가
  처음 만나는 '보이는 정답 기록'(학습 & 그 폴드에서 숨김 아님) 까지 k 단계; 사슬이 끊기면(진짜 전날 없음) 앵커 없음.
안: K1 = k=1 만 앵커(현재 앵커의 완벽판), K2 = k ≤ 2, K3 = k ≤ 3, KA = 모든 k; 값 A = 조상 23시 EC, pred = clip(sg + (A − sg 0..h 누적평균)).
  참고 Z0 = 대상 자기 0시 정답을 A 로(0시 완벽 상한), DAY = 하루 평균 정답 그대로.
보고: 2차 행 RMSE, NONE(≈12회차 .1271) 대비 비율로 공개 환산 = .1271 × 비율;  k 분포(2차 기록), k 별 기록 RMSE.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
TL = pd.read_pickle(os.path.join(S3, "truth.pkl"))
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
PREV = {}
for a, b in zip(TL.a1, TL.b):
    a, b = int(a), int(b)
    if D11.get(rid(b), np.nan) - D11.get(rid(a), np.nan) == 1: PREV[b] = a
print(f"true prev links: {len(PREV)} of {len(TL)} records; pass-2 targets with true prev: {sum(NUM[b] >= 179 for b in PREV)}")
def ancestor(i, vis):
    k = 0; cur = i; seen = set()
    while cur in PREV and cur not in seen:
        seen.add(cur); cur = PREV[cur]; k += 1
        if vis[cur] and np.isfinite(EC[cur, 23]): return cur, k
    return None, None
ARMS = {"K1": 1, "K2": 2, "K3": 3, "KA": 99}
res = []; krec = []
def run(Gv, y, sg, p2, hours, cm, recrows, hsf, tag, sd):
    P = {a: sg.copy() for a in list(ARMS) + ["Z0", "DAY"]}
    for r, ix in recrows.items():
        i = idx[r]; hs = hsf(r); vis = TRAIN & ~np.array([rid(k) in hs for k in range(N)]); vis[i] = False
        anc, k = ancestor(i, vis)
        if anc is not None:
            for a, K in ARMS.items():
                if k <= K: P[a][ix] = np.clip(sg[ix] + (EC[anc, 23] - cm[ix]), lo_, hi_)
        P["Z0"][ix] = np.clip(sg[ix] + (y[ix][0] - cm[ix]), lo_, hi_); P["DAY"][ix] = y[ix].mean()
        if NUM[i] >= 179: krec.append(dict(val=tag, seed=str(sd), rec=r, k=k if k else 0, e_none=rm(sg[ix] - y[ix]), e_or=rm(P["KA"][ix] - y[ix]), y=y[ix].mean()))
    base = rm((sg - y)[p2])
    for a, pr in P.items(): res.append(dict(val=tag, seed=str(sd), arm=a, p2_rmse=rm((pr - y)[p2]), ratio=rm((pr - y)[p2]) / base))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict(); hsf = lambda r: hidden.get(fold_of[r], set())
    run(Gv, y, sg, p2, Gv.hour.values, cm, {r: g.index.values for r, g in Gv.groupby("rec")}, hsf, v, sd)
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.y.values; sg = Gv.SG2.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    fo = {r: int(Gv.fold.values[ix[0]]) for r, ix in recrows.items()}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, lambda r: hid[fo[r]], "EV", sd)
S = pd.DataFrame(res); K = pd.DataFrame(krec); pd.set_option("display.width", 240)
T = S.groupby(["arm", "val"]).ratio.mean().unstack(); print("pass-2 RMSE ratio vs NONE (seed mean):"); print(T.round(3).to_string())
print("\npublic-equivalent (.1271 x ratio):"); print((T * .1271).round(4).to_string())
print("\npass-2 records by chain length k (0 = no true chain to a visible label), seed-mean record RMSE:")
print(K.groupby(["val", "k"]).agg(n=("rec", "nunique"), e_none=("e_none", "mean"), e_oracle=("e_or", "mean"), y=("y", "mean")).round(3).to_string())
print("\nSSE share of pass-2 NONE error by k (seed mean):")
K["sse"] = K.e_none ** 2 * 24; print((K.groupby(["val", "k"]).sse.sum() / K.groupby("val").sse.sum()).round(3).unstack().to_string())
