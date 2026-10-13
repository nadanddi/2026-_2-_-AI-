# -*- coding: utf-8 -*-
"""R2C1 (2026-10-14 집 클로드; 사용자: '번호 −2 구조로 CH2 다시 해봐').  1단계 서술(판정 전 점검).  EC.
6.549: 순수 2차 창(F13 216~238, F47 211~231)에서 같은 ID 가 연속 날짜에 2기록씩이면 '번호 −2 = 같은 동 전날'(91~94%), 섞인 날짜에선 우연 수준; 덩어리 첫 기록(r−1 지움)은 불확실.
(1) 규칙 정확도(학습 자료): 정답 기록 r 의 r−2 가 정답 기록일 때 r−2 == cert 진짜 전날 비율 — 순수 창 안/밖, 1차/2차.  학습 달력 날짜 차(r − (r−2)) == 1 비율.
(2) 평가 적용(서술, 입력·학습 자료만): 평가 기록 R 에서 같은 ID 사슬 R ← R−2 ← R−4 … 를 따라(지워진 번호면 멈춤), 처음 만나는 정답 학습 기록까지 단계 k,
    그 기록의 23시 EC, 학습 달력 날짜 차(대상 날짜는 0시 외기 일치 최빈), 하루 운영 지문 거리(같은 동 확인용), 14회차 예측 하루 평균.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
T = pd.read_csv(os.path.join(HERE, "cp2_tracks_v1.csv")).set_index("rec")
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CERT = {rid(int(b)): rid(int(a)) for a, b, c in zip(TL.a1, TL.b, TL.cert) if c}
PURE = {"F13": (216, 238), "F47": (211, 231)}
inpure = lambda r: PURE[r[:3]][0] <= int(r[4:]) <= PURE[r[:3]][1]
key = lambda f, n: f"{f}_{n:03d}"
rows = []
for r in T.index:
    if not T.at[r, "train"] or r not in CERT: continue
    m2 = key(r[:3], int(r[4:]) - 2)
    if m2 not in T.index or not T.at[m2, "train"]: continue
    dd = T.at[r, "day"] - T.at[m2, "day"] if np.isfinite(T.at[r, "day"]) and np.isfinite(T.at[m2, "day"]) else np.nan
    rows.append(dict(rec=r, pure=inpure(r), p2=int(r[4:]) >= 179, ok=CERT[r] == m2, day1=dd == 1))
R = pd.DataFrame(rows); pd.set_option("display.width", 200)
print("(1) rule accuracy on labelled records with cert truth and labelled r-2:")
print(R.groupby(["pure", "p2"]).agg(n=("rec", "size"), r2_is_true_prev=("ok", "mean"), calendar_gap1=("day1", "mean")).round(3).to_string())
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
FP = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FP = (FP - np.nanmean(FP[TRAIN], (0, 1))) / (np.nanstd(FP[TRAIN], (0, 1)) + 1e-9); fpd = lambda a, b: float(np.nanmean(np.minimum(np.abs(FP[idx[a]] - FP[idx[b]]), 4)))
P14 = pd.read_csv(os.path.join(env.ROOT, "제출", "14회차_2026-10-13(나)", "submission_23.csv")); P14["rec"] = P14.row_id.str[:7]; pm = P14.groupby("rec").sub_ec.mean()
out = []
for r in sorted(T[~T.train & ~T.lock].index):
    if r not in pm.index: continue
    cur, k, path = r, 0, []
    while True:
        nx = key(r[:3], int(cur[4:]) - 2); k += 1
        if nx not in T.index: anc = None; break
        path.append(nx)
        if T.at[nx, "train"]: anc = nx; break
        cur = nx
        if k > 6: anc = None; break
    dday = (T.at[r, "day"] - T.at[anc, "day"]) if (anc and np.isfinite(T.at[r, "day"]) and np.isfinite(T.at[anc, "day"])) else np.nan
    out.append(dict(rec=r, pure=inpure(r), pred14=round(float(pm[r]), 3), steps=k if anc else np.nan, ancestor=anc, anc_EC23=round(float(EC[idx[anc], 23]), 3) if anc else np.nan,
                    cal_gap=dday, fp_dist=round(fpd(r, anc), 3) if anc else np.nan, path=">".join(path)))
E = pd.DataFrame(out)
print("\n(2) eval records: number-2 chain to a labelled ancestor"); print(E.to_string(index=False))
print("\nreach labelled ancestor:", int(E.ancestor.notna().sum()), "/", len(E), "| steps:", E.steps.value_counts().sort_index().to_dict(), "| in pure window:", int(E.pure.sum()))
E.to_csv(os.path.join(HERE, "r2c1_eval_chains_v1.csv"), index=False)
