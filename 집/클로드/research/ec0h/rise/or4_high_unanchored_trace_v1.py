# -*- coding: utf-8 -*-
"""OR4 (OR3 다음 단, 서술): 2차 고EC 기록(0시 정답 >= 1.0)의 진실 사슬(truth a1, EC 연속으로 만든 진단용 연결)을 앞으로 5단계 따라가
같은 동 앞날들의 EC 수준·정답 유무·번호를 나열 — '앵커 없는 고EC 날'의 정보가 어디 있는지(앞날이 고EC였나, 정답 있는 날이었나)."""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
import os
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
TL = pd.read_pickle(S3 + r"\truth.pkl"); pre = {int(b): (int(a), bool(c), float(c1)) for a, b, c, c1 in zip(TL.a1, TL.b, TL.cert, TL.c1)}
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
Q = R.groupby(["rec", "hour"]).EA2.mean().unstack(); anc = R.groupby("rec").anchored.first()
for i in np.where(TRAIN & (NUM >= 179))[0]:
    if EC[i, 0] < 1.0: continue
    r = rid(i); s = f"{r} y0 {EC[i,0]:.2f} pred0 {Q.at[r,0]:.2f} anchored {bool(anc.get(r, False))} <- "
    j = i
    for k in range(5):
        if j not in pre: s += "(no link)"; break
        a, cert, c1 = pre[j]
        s += f"{rid(a)}[{'L' if TRAIN[a] else 'E'}{'c' if cert else ''} 23h {EC[a,23] if TRAIN[a] else float('nan'):.2f}] <- "
        j = a
        if not TRAIN[a]: break
    print(s)
