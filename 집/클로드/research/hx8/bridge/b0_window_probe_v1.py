# -*- coding: utf-8 -*-
"""B0 (2026-10-13 집 클로드, hx8/bridge) — 순수 2차 창 구조 확인(서술).
기록마다: split, 학습 전용 달력(dmt3e_E0) day, DM11 day(서술 대조만), 하루 평균/0시/23시 EC(학습), OOF sg 평균(DIAG10/EL1/P2LOO),
13회차 예측 하루 평균(평가), 진단 연결(truth a1, cert).  '번호 −2 = 같은 동 전날' 이 학습끼리 성립하는지 확인.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
E0 = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); D0 = dict(zip(E0.rec, E0.day))
CAL = pd.read_csv(os.path.join(env.ROOT, u"집", u"클로드", "research", "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
TL = pd.read_pickle(S3 + r"\truth.pkl"); pre = {rid(int(b)): (rid(int(a)), float(c)) for a, b, c in zip(TL.a1, TL.b, TL.cert)}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
Q = O.groupby(["validator", "rec"]).sg.mean().unstack(0)
S = pd.read_csv(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)", "submission_22.csv"))
print(S.columns.tolist()[:6]); idcol = S.columns[0]
S["rec"] = S[idcol].str[:7]; P13 = S.groupby("rec").sub_ec.mean()
print("N", N, "train", TRAIN.sum())
rows = []
for f, lo, hi in (("F13", 200, 250), ("F47", 195, 250)):
    for n in range(lo, hi):
        r = f"{f}_{n:03d}"
        if r not in idx:
            rows.append(dict(rec=r, split="MISSING")); continue
        i = idx[r]
        d = dict(rec=r, split="train" if TRAIN[i] else "test", d0=D0.get(r, np.nan), d11=D11.get(r, np.nan))
        if TRAIN[i]:
            d.update(y=np.nanmean(EC[i]), e0=EC[i, 0], e23=EC[i, 23])
            for v in ("DIAG10", "EL1", "P2LOO"):
                d["q_" + v] = Q.loc[r, v] if r in Q.index else np.nan
            if r in pre: d.update(pre=pre[r][0], cert=pre[r][1])
        else:
            d["p13"] = P13.get(r, np.nan)
        # outdoor mean (for 'same date' check)
        d["xo0"] = X[i, 0, 0]
        rows.append(d)
T = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(T.round(3).to_string())
# check 'num-2 = same house previous day' among training pairs in the window
print("\n'번호-2' 학습 쌍(창 안): r-2 23시 vs r 0시")
for f, lo, hi in (("F13", 210, 245), ("F47", 205, 245)):
    for n in range(lo, hi):
        a, b = f"{f}_{n-2:03d}", f"{f}_{n:03d}"
        if a in idx and b in idx and TRAIN[idx[a]] and TRAIN[idx[b]]:
            ia, ib = idx[a], idx[b]
            print(f"  {a}->{b}: d0 {D0.get(a)}->{D0.get(b)} d11 {D11.get(a)}->{D11.get(b)} | e23 {EC[ia,23]:.3f} e0 {EC[ib,0]:.3f} diff {EC[ib,0]-EC[ia,23]:+.3f} | truth pre(b)={pre.get(b)}")
