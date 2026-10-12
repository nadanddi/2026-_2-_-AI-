# -*- coding: utf-8 -*-
"""verify_leakage-stats vls6: INC4 k2 H 의 (a) 에피소드 안 순열, (b) '모델이 앵커를 본' 느슨한 부분집합(n7) 부트스트랩,
(c) 2차(기록 번호>=179) 사례만. 학습 자료만 사용."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import env
from feat4 import EC, FARM, NUM, N, S3
import numpy as np, pandas as pd
inc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
T1 = pd.read_csv(os.path.join(inc, "inc1_targets_v1.csv"), index_col=0); y, comp = T1.y, T1.comp
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
fold = O.groupby("rec").fold.first(); Mm = O.groupby("rec").sg.mean()
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")).set_index("rec")
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
                rows.append(dict(a=a, t=t, anc=EC[idx[a], 23], yt=y[t], m=Mm[t], ep=comp[a], seen=(a in fold.index) and fold.get(a) != fold[t], num=int(t[4:])))
G = pd.DataFrame(rows).drop_duplicates(["a", "t"]).reset_index(drop=True)
rm = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
H = lambda m, a: (m + a) / 2
rng = np.random.default_rng(11)
obs = rm(H(G.m, G.anc), G.yt) - rm(G.m, G.yt); ds = []
for _ in range(5000):
    a = G.anc.values.copy()
    for e in G.ep.unique():
        ii = np.where(G.ep.values == e)[0]; a[ii] = rng.permutation(a[ii])
    ds.append(rm(H(G.m.values, a), G.yt) - rm(G.m, G.yt))
ds = np.array(ds); print(f"(a) k2 H within-episode permutation: obs {obs:+.3f}, perm mean {ds.mean():+.3f}, p(perm<=obs) {np.mean(ds <= obs + 1e-12):.3f}")
s = G[G.seen]; eps = s.ep.unique(); bs = []
for _ in range(2000):
    gg = pd.concat([s[s.ep == e] for e in rng.choice(eps, len(eps))]); bs.append(rm(H(gg.m, gg.anc), gg.yt) - rm(gg.m, gg.yt))
print(f"(b) anchor seen by model (looser): n {len(s)}, eps {len(eps)}, RMSE M {rm(s.m, s.yt):.3f} H {rm(H(s.m, s.anc), s.yt):.3f}, diff CI [{np.quantile(bs,.025):+.3f},{np.quantile(bs,.975):+.3f}], win {np.mean(np.abs(H(s.m,s.anc)-s.yt) < np.abs(s.m-s.yt)):.2f}")
z = G[G.num >= 179]; print(f"(c) 2차 targets (num>=179): n {len(z)}" + (f", RMSE M {rm(z.m, z.yt):.3f} H {rm(H(z.m, z.anc), z.yt):.3f}" if len(z) else ""))
# how often in all train is the next-next day after a normal day (y<1) truly high when model says >= 1.2 (F13_237 13th = 1.256)?
hi = G[G.m >= 1.15]; print(f"(d) cases with M >= 1.15 (near F13_237's 1.256): n {len(hi)}, y {hi.yt.round(3).tolist()}, M {hi.m.round(3).tolist()}, anchor {hi.anc.round(3).tolist()}")
