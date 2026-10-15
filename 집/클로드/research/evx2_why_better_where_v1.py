# -*- coding: utf-8 -*-
"""EVX2 (2026-10-15 집 클로드; 사용자: 'EVM·EVY 가 왜 더 정확한지, 함께 썼을 때 장점만 따로 뺄 수 없을지 분석').  서술(사후, 판정 아님).  EC.
저장 예측(local/evx_preds_v1.pkl: EV3L·EVM·EVY, ec_rv1 CUR 배치) + 기본 sg 로:
 (1) 오차 분해: 하루 수준(하루 평균 오차) vs 하루 안 모양 — 어느 쪽이 줄었나.
 (2) 상황별 단독 RMSE: 어제 EC 보임 여부(링커 짝이 정답 있는 학습 기록이고 숨김 아님), 2차 보통/고EC, 시각대.
 (3) 섞을 때 이득이 줄어드는 이유: sg 와 오차 상관, 상황별 '최적 섞기 비율'(사후, 최소제곱) 과 그때 이득.
 (4) 장점만 따로: 상황별로 X 가 sg 보다 나은 칸 / 고정 비율 대신 상황별 비율이면 얼마나 (사후 상한, 채택 근거 아님).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u evx2_why_better_where_v1.py
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
P = pickle.load(open(os.path.join(env.LOCAL, "evx_preds_v1.pkl"), "rb"))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & O.validator.isin(["DIAG10", "EL1", "P2LOO"])].copy(); O["rec"] = O.row_id.str[:7]
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv")); LINK = {r: p for r, p, q in zip(L.rec, L.prev0, L.p0) if isinstance(p, str) and q >= .5 and p in idx}
LAB = {rid(i) for i in range(N) if TRAIN[i] and np.isfinite(EC[i]).any()}
rows = []
for (v, k), G in O.groupby(["validator", "fold"]):
    hide = set(G.rec)
    if v == "EL1": hide |= {r for r in LAB for t in G.rec.unique() if r[:3] == t[:3] and abs(int(r[4:]) - int(t[4:])) <= 1}
    G = G.copy(); G["yvis"] = [LINK.get(r) in LAB and LINK.get(r) not in hide for r in G.rec]; G["linked"] = [r in LINK for r in G.rec]; rows.append(G)
O = pd.concat(rows); O = O[O.seed == O.seed.min()].copy()
for m in ("EV3L", "EVM", "EVY"): O[m] = [P[m][(v, r)][h] for v, r, h in zip(O.validator, O.rec, O.hour)]
O["y"] = O.sub_ec; O["p2"] = O.day >= 179; O["dm"] = O.groupby(["validator", "rec"]).y.transform("mean"); O["hi"] = O.dm >= 1.0
M = ["sg", "EV3L", "EVM", "EVY"]; pd.set_option("display.width", 220)
print("(1) error split: day-level (RMS of record-mean error) / within-day (RMS of deviation from record-mean error), pass-2 rows")
X = O[O.p2]
for m in M:
    e = X[m] - X.y; dl = e.groupby([X.validator, X.rec]).transform("mean"); print(f"  {m:5s} total {rm(e):.4f} | day-level {rm(dl):.4f} | within-day {rm(e - dl):.4f}")
print("\n(2) standalone RMSE by situation (pass-2 rows, all validators)")
seg = {"yesterday EC visible": X.yvis, "linked, EC not visible": X.linked & ~X.yvis, "no link": ~X.linked, "normal day": ~X.hi, "high-EC day": X.hi,
       "hours 0-7": X.hour <= 7, "hours 8-15": X.hour.between(8, 15), "hours 16-23": X.hour >= 16}
T = pd.DataFrame({nm: {m: rm((X[m] - X.y)[msk.values]) for m in M} | {"n_rec": int(X[msk.values].groupby(["validator", "rec"]).ngroups)} for nm, msk in seg.items()}).T
print(T.round(4).to_string())
print("\n(3) why the 10% blend gains little: corr(error of X, error of sg) and best blend weight w* (least squares, post-hoc) by situation")
out = []
for nm, msk in [("all pass-2", np.ones(len(X), bool))] + list(seg.items()):
    S = X[np.asarray(msk)]; es = (S.sg - S.y).values
    for m in ("EV3L", "EVM", "EVY"):
        ex = (S[m] - S.y).values; dlt = ex - es; w = float(np.clip(-np.dot(es, dlt) / max(np.dot(dlt, dlt), 1e-12), 0, 1))
        out.append(dict(seg=nm, model=m, corr=float(np.corrcoef(es, ex)[0, 1]), w_star=w, gain_w10=100 * (rm(es + .1 * dlt) / rm(es) - 1), gain_wstar=100 * (rm(es + w * dlt) / rm(es) - 1)))
B = pd.DataFrame(out); print(B.round(3).to_string(index=False))
print("\n(4) per-situation weight (post-hoc upper bound, NOT for adoption): EVY with w* by {yesterday visible / not} vs fixed 10%, pass-2 rows by validator")
for v, S in X.groupby("validator"):
    es = (S.sg - S.y).values; d = (S.EVY - S.sg).values; base = rm(es)
    w = np.where(S.yvis.values, B[(B.seg == "yesterday EC visible") & (B.model == "EVY")].w_star.iloc[0], B[(B.seg == "linked, EC not visible") & (B.model == "EVY")].w_star.iloc[0])
    w = np.where(~S.linked.values, B[(B.seg == "no link") & (B.model == "EVY")].w_star.iloc[0], w)
    print(f"  {v}: sg {base:.4f} | 10% {100 * (rm(es + .1 * d) / base - 1):+.2f}% | situation w* {100 * (rm(es + w * d) / base - 1):+.2f}%")
