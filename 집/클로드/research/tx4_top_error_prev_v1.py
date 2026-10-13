import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); os.chdir(r"C:\work\farmai\집\클로드\research")
import env, numpy as np, pandas as pd
sys.path.insert(0, "linker_v4"); from feat4 import ST, IN_, X, FARM, NUM, N, TRAIN, S3
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CAL = pd.read_csv(os.path.join("ec0h","rise","dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
PREV = {int(b): int(a) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv")).set_index("rec")
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").set_index("rec")
for r in ["F47_191", "F47_241", "F47_239", "F13_244", "F13_109", "F47_123", "F47_178", "F47_145", "F47_141", "F47_132"]:
    b = idx[r]; a = PREV.get(b)
    s = f"{r}: ST0 {ST[b,0]:.1f} STmean {np.nanmean(ST[b]):.1f} in0 {IN_[b,0,0]:.1f} inMean {np.nanmean(IN_[b,:,0]):.1f} ST-in(day) {np.nanmean(ST[b]-IN_[b,:,0]):+.2f}"
    if a is not None: s += f" | clean prev {rid(a)} ST23 {ST[a,23]:.1f} ST-in(day) {np.nanmean(ST[a]-IN_[a,:,0]):+.2f} labelled {bool(TRAIN[a])}"
    else: s += " | no clean prev"
    if r in PM.index: s += f" | v15 {PM.at[r,'prev0']} p0 {PM.at[r,'p0']:.2f}"
    if r in L7.index: s += f" | v7 {L7.at[r,'prev']} p {L7.at[r,'p']:.2f} dT {L7.at[r,'dT_air']:+.1f}"
    print(s)
