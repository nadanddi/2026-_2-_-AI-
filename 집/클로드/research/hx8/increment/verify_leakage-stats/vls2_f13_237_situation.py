# -*- coding: utf-8 -*-
"""verify_leakage-stats vls2: F13_237 상황이 INC4 k2 학습 사례와 닮았는지.
(a) 같은 날짜 쌍둥이(0시 외기 4변수가 같은 학습 기록)의 EC 평균: F13_237 vs INC4 k2 사례 표적
(b) 사슬 식별: F13_229~239 학습 기록의 진단 연결 선행(a1)과 학습 전용 달력 날짜 — '번호-2 = 같은 동 전날'이 이 구간에서 성립하나
(c) INC4 k2 사례에서 쌍둥이 평균이 H 잔차를 설명하나(관찰, 표본 작음)
평가 입력은 F13_237 의 0시 외기만 읽어 쌍둥이를 찾는 데 쓰고(같은 ID 자기 입력, 서술용), 계수 추정에는 쓰지 않음.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import env
from feat4 import X, EC, FARM, NUM, N, TRAIN, S3
import numpy as np, pandas as pd
pd.set_option("display.width", 220)
inc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
T1 = pd.read_csv(os.path.join(inc, "inc1_targets_v1.csv"), index_col=0); y, comp = T1.y, T1.comp
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); pre = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")).set_index("rec")
key0 = {i: tuple(np.round(X[i, 0, :], 4)) for i in range(N)}
def twins(r):
    i = idx[r]; return [rid(j) for j in range(N) if j != i and TRAIN[j] and key0[j] == key0[i]]
def twin_mean(r):
    tw = twins(r); return (np.mean([y[t] for t in tw]) if tw else np.nan), tw
m, tw = twin_mean("F13_237"); print(f"(a) F13_237 twins {tw} -> EC mean {m:.3f}; days {[CAL.day.get(t, np.nan) for t in tw]}")
for r in ["F13_233", "F13_231", "F13_217", "F47_229", "F47_215"]:
    m2, tw2 = twin_mean(r); print(f"    {r}: day {CAL.day.get(r, np.nan)}, y {y.get(r, np.nan):.3f}, twins {tw2} mean {m2:.3f}")
print("\n(b) F13 229..245 / F47 225..235 training records: truth predecessor a1, day")
for r in [f"F13_{n:03d}" for n in range(226, 246)] + [f"F47_{n:03d}" for n in range(224, 236)]:
    if r in idx and TRAIN[idx[r]]:
        print(f"    {r}: day {CAL.day.get(r, np.nan)}, seg {CAL.seg.get(r, np.nan)}, y {y[r]:.3f}, pred(a1) {pre.get(r)}, pred day {CAL.day.get(pre.get(r), np.nan)}")
# INC4 k2 rebuild
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]; O["rec"] = O.row_id.str[:7]
Mm = O.groupby(["seed", "rec"]).sg.mean().unstack(0).mean(axis=1)
def consec(a, b): return a in CAL.index and b in CAL.index and CAL.seg[a] == CAL.seg[b] and CAL.day[b] == CAL.day[a] + 1
succ = {}
for a, b in zip(TL.a1, TL.b):
    a, b = rid(int(a)), rid(int(b))
    if a in y.index and b in y.index and consec(a, b): succ.setdefault(a, []).append(b)
rows = []
for a in y.index:
    if y[a] >= 1: continue
    for q in succ.get(a, []):
        for t in succ.get(q, []):
            if t in Mm.index and Mm[t] >= .9:
                tm, tws = twin_mean(t)
                rows.append(dict(a=a, t=t, anc=EC[idx[a], 23], yt=y[t], m=Mm[t], twin=tm, ntw=len(tws), ep=comp[a]))
G = pd.DataFrame(rows).drop_duplicates(["a", "t"])
G["H"] = (G.m + G.anc) / 2; G["eH"] = G.yt - G.H; G["gain"] = (G.m - G.yt) ** 2 - (G.H - G.yt) ** 2
print("\n(c) INC4 k2 cases with same-date twin EC mean")
print(G.round(3).to_string(index=False))
ok = G.twin.notna()
print(f"    corr(twin, y) {np.corrcoef(G.twin[ok], G.yt[ok])[0,1]:+.2f}, corr(twin, H residual) {np.corrcoef(G.twin[ok], G.eH[ok])[0,1]:+.2f}, n {ok.sum()}")
print(f"    cases with twin mean >= 1.2: n {(G.twin >= 1.2).sum()}, their mean y {G.yt[G.twin>=1.2].mean():.3f}, mean H gain {G.gain[G.twin>=1.2].mean():+.3f}; twin<1.2 mean H gain {G.gain[G.twin<1.2].mean():+.3f}")
