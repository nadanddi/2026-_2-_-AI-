# -*- coding: utf-8 -*-
"""FC1 (2026-10-13 집 클로드; 사용자: '가짜 고EC 연쇄 끊을 방법 찾아봐').  서술·탐색(판정 아님).  EC.
(a) 전제: 학습 전용 달력(DMT3e: rec → 구간 seg·날짜 day)에서 같은 날짜 정답 기록 중 고EC(하루 평균 ≥ 1.0) 가 몇 개인가 — '하루에 고EC 동 하나' 인가.
(b) DIAG10 OOF 예측 하루 평균 ≥ 1.0 인 기록(진짜/가짜 고EC)마다, 그 폴드에서 보이는 정보만으로 끊는 단서:
   D1 같은 날짜(같은 seg)의 다른 정답 기록 중 고EC 있음 / 보통만 있음 / 없음;
   D2 깨끗한 사슬(진단용, 상한)을 최대 6단계 거슬러 처음 만나는 '보이는 정답' 기록의 상태와 거리;
   D3 사슬 안(최대 3단계) 어느 날이든 B5 꼴 쌍둥이 고EC 표시(z).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
XR = np.round(X, 1)
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
CPREV = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAY = {r: (s, d) for r, s, d in zip(C.rec, C.seg, C.day)}
lab = [rid(i) for i in range(N) if TRAIN[i] and np.isfinite(DM[i]) and rid(i) in DAY]
T = pd.DataFrame(dict(rec=lab, key=[DAY[r] for r in lab], hi=[DM[idx[r]] >= 1.0 for r in lab], farm=[r[:3] for r in lab]))
g = T.groupby("key").agg(n=("rec", "size"), nh=("hi", "sum"))
print(f"(a) labelled records with training-calendar date: {len(T)} on {len(g)} dates")
print("    dates by (#records, #high):", g.value_counts().sort_index().to_dict())
print("    dates with a high record: share where it is the ONLY high:", round(float((g[g.nh >= 1].nh == 1).mean()), 3), "| dates with >= 2 high:", int((g.nh >= 2).sum()))
def zflag(i, vis):
    c = np.where(vis)[0]; c = c[c != i]; t = c[np.all(np.abs(XR[c] - XR[i]) < 1e-6, axis=(1, 2))]; t = t[np.isfinite(DM[t])]
    return bool((DM[t] >= 1.2).any())
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]; O["rec"] = O.row_id.str[:7]
pred = O.groupby("rec").sg.mean(); fold = O.groupby("rec").fold.first(); hidden = O.groupby("fold").rec.apply(set).to_dict()
rows = []
for r in pred.index[pred >= 1.0]:
    hs = hidden[fold[r]]; vis = TRAIN & ~np.array([rid(k) in hs for k in range(N)])
    d1 = "no date"
    if r in DAY:
        same = T[(T.key == DAY[r]) & (T.rec != r) & ~T.rec.isin(hs)]
        d1 = "other HIGH" if same.hi.any() else ("normal only" if len(same) else "none visible")
    cur, k, anc, dist = r, 0, "none", np.nan
    while k < 6 and cur in CPREV:
        cur = CPREV[cur]; k += 1
        if TRAIN[idx[cur]] and cur not in hs and np.isfinite(DM[idx[cur]]): anc, dist = ("HIGH" if DM[idx[cur]] >= 1.0 else "normal"), k; break
    ch = [r]; cur = r
    for _ in range(3):
        if cur not in CPREV: break
        cur = CPREV[cur]; ch.append(cur)
    rows.append(dict(rec=r, trueH=DM[idx[r]] >= 1.0, y=round(float(DM[idx[r]]), 3), pred=round(float(pred[r]), 3), D1_sameday=d1, D2_ancestor=anc, D2_dist=dist,
                     D3_chain_z=any(zflag(idx[q], vis) for q in ch)))
R = pd.DataFrame(rows).sort_values(["trueH", "pred"]); pd.set_option("display.width", 220)
print("\n(b) predicted-high records (DIAG10):"); print(R.to_string(index=False))
for c in ("D1_sameday", "D2_ancestor", "D3_chain_z"):
    print(f"\n{c}:"); print(pd.crosstab(R[c], R.trueH.map({True: "true high", False: "FALSE high"})).to_string())
R.to_csv(os.path.join(HERE, "fc1_rows_v1.csv"), index=False)
