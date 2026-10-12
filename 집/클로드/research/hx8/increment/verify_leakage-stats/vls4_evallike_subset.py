# -*- coding: utf-8 -*-
"""verify_leakage-stats vls4: 평가와 같은 '모델 학습 상태' 부분집합.
평가: 13회차 모델은 앵커(F47_229, F13_217, F13_233)를 학습에 썼고, 사이 날(평가 기록)은 학습에 없음.
가짜 사례 M(DIAG10 OOF): 대상과 같은 폴드에 앵커가 있으면 모델이 앵커를 못 봄 -> 평가보다 M 이 불리.
부분집합 EL = 앵커가 대상과 다른 폴드(모델이 앵커를 학습) AND 사이 날 모두 대상과 같은 폴드(모델이 사이 날을 못 봄).
또 F13_217·F47_229 등 평가 근처 앵커와 가짜 사례 대상의 ec_rv1 day(1차/2차) 비교.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import env
from feat4 import EC, FARM, NUM, N, TRAIN, S3
import numpy as np, pandas as pd
pd.set_option("display.width", 220)
inc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
T1 = pd.read_csv(os.path.join(inc, "inc1_targets_v1.csv"), index_col=0); y, comp = T1.y, T1.comp
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
fold = O.groupby("rec").fold.first(); day = O.groupby("rec").day.first()
Mm = O.groupby("rec").sg.mean()
print("ec_rv1 day of eval-near anchors:", {r: int(day[r]) if r in day else None for r in ["F13_217", "F47_229", "F47_231", "F13_233", "F13_231", "F47_215"]})
TL = pd.read_pickle(os.path.join(S3, "truth.pkl"))
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")).set_index("rec")
def consec(a, b): return a in CAL.index and b in CAL.index and CAL.seg[a] == CAL.seg[b] and CAL.day[b] == CAL.day[a] + 1
succ, prev = {}, {}
for a, b in zip(TL.a1, TL.b):
    a, b = rid(int(a)), rid(int(b))
    if a in y.index and b in y.index and consec(a, b): succ.setdefault(a, []).append(b); prev.setdefault(b, []).append(a)
def paths(s, k, nxt):
    out = [[s]]
    for _ in range(k): out = [p + [q] for p in out for q in nxt.get(p[-1], [])]
    return out
rm = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
rows = []
for d, nxt in (("fwd", succ), ("bwd", prev)):
    for a in y.index:
        if y[a] < 1: continue
        for k in range(1, 5):
            for p in paths(a, k, nxt):
                t = p[-1]
                if t not in Mm.index or Mm[t] < .9: continue
                mids = p[1:-1]
                ev = (a in fold.index) and fold.get(a) != fold[t] and all((m_ in fold.index) and fold[m_] == fold[t] for m_ in mids)
                anc = EC[idx[a], 23] if d == "fwd" else EC[idx[a], 0]
                rows.append(dict(dir=d, k=k, a=a, t=t, anc=anc, yt=y[t], m=Mm[t], ep=comp[a], evlike=ev, dt=day.get(t, np.nan)))
C = pd.DataFrame(rows)
rng = np.random.default_rng(3)
print("\nP (anchor as is) vs M, overall vs eval-like subset (anchor seen by model, intermediates not)")
for (d, k), g in C.groupby(["dir", "k"]):
    e = g[g.evlike]
    line = f" {d} k{k}: all n {len(g)} M {rm(g.m, g.yt):.3f} P {rm(g.anc, g.yt):.3f} | eval-like n {len(e)} eps {e.ep.nunique()}"
    if len(e):
        dif = rm(e.anc, e.yt) - rm(e.m, e.yt); bs = []
        eps = e.ep.unique()
        for _ in range(2000):
            gg = pd.concat([e[e.ep == x] for x in rng.choice(eps, len(eps))]); bs.append(rm(gg.anc, gg.yt) - rm(gg.m, gg.yt))
        line += f" M {rm(e.m, e.yt):.3f} P {rm(e.anc, e.yt):.3f} diff {dif:+.3f} CI [{np.quantile(bs,.025):+.3f},{np.quantile(bs,.975):+.3f}] win {np.mean(np.abs(e.anc-e.yt) < np.abs(e.m-e.yt)):.2f}"
    print(line)
print("\nfake-case target ec_rv1 day: share day>=179:", f"{np.mean(C.dt >= 179):.2f}", "| range", C.dt.min(), C.dt.max())
# INC4-type (normal anchor -> model high, fwd k2) eval-like subset
rows = []
for a in y.index:
    if y[a] >= 1: continue
    for p in paths(a, 2, succ):
        t = p[-1]
        if t not in Mm.index or Mm[t] < .9: continue
        m_ = p[1]
        ev = (a in fold.index) and fold.get(a) != fold[t] and (m_ in fold.index) and fold[m_] == fold[t]
        rows.append(dict(a=a, t=t, anc=EC[idx[a], 23], yt=y[t], m=Mm[t], ep=comp[a], evlike=ev, fa=fold.get(a), fm=fold.get(m_), ft=fold[t]))
G = pd.DataFrame(rows).drop_duplicates(["a", "t"]); G["H"] = (G.m + G.anc) / 2
print("\nINC4 k2 cases with fold info (eval-like = model saw anchor, not the middle day)")
print(G.round(3).to_string(index=False))
e = G[G.evlike]
if len(e): print(f" eval-like n {len(e)}: RMSE M {rm(e.m, e.yt):.3f} H {rm(e.H, e.yt):.3f} P {rm(e.anc, e.yt):.3f} | rest n {(~G.evlike).sum()}: M {rm(G.m[~G.evlike], G.yt[~G.evlike]):.3f} H {rm(G.H[~G.evlike], G.yt[~G.evlike]):.3f}")
