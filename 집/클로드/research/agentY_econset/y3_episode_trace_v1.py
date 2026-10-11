# -*- coding: utf-8 -*-
"""agentY_econset y3: 고EC 날에서 같은 동 연결을 거꾸로 따라가 '에피소드 시작(낮은 EC -> 높은 EC 상승)'이
학습 자료 안에 관측되는지 확인. 연결 = truth a1 (EC+배지온도 연속 비용 c1) 중 c1<=CMAX, 같은 a1 중복 사용은 비용 작은 쪽만."""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env, os, pickle, numpy as np, pandas as pd
OUTD = os.path.dirname(os.path.abspath(__file__))
R = pickle.load(open(r"C:\work\farmai\집\클로드\research\local\linker_v3\base.pkl", "rb"))
r = R["recs"].reset_index(drop=True); EC = R["EC"]
T = pd.read_pickle(r"C:\work\farmai\집\클로드\research\local\linker_v3\truth.pkl")
ecm = np.nanmean(EC, 1)
rid = lambda i: f"{r.farm[i]}_{r.day[i]:03d}"
for CMAX in [1.0, 2.0]:
    t = T[T.c1 <= CMAX].sort_values("c1").drop_duplicates("a1")
    prev = dict(zip(t.b, t.a1)); nxt = {a: b for b, a in prev.items()}
    print(f"\n#### CMAX={CMAX}: links {len(prev)}")
    # chains: heads = records with no prev
    heads = [i for i in np.where(r.split == "train")[0] if i not in prev]
    chains = []
    for h in heads:
        c = [h]; seen = {h}
        while c[-1] in nxt and nxt[c[-1]] not in seen: c.append(nxt[c[-1]]); seen.add(c[-1])
        chains.append(c)
    print("chains", len(chains), "len dist", pd.Series([len(c) for c in chains]).value_counts().sort_index().to_dict())
    nrise = 0; nstart_high = 0
    for c in chains:
        e = ecm[c]
        if np.nanmax(e) < 1.0: continue
        s = " ".join(f"{rid(i)}:{ecm[i]:.2f}" for i in c)
        first_hi = int(np.argmax(e >= 1.0))
        tag = "STARTS-HIGH" if first_hi == 0 else ("rise-from %.2f" % e[0])
        if first_hi == 0: nstart_high += 1
        else: nrise += 1
        print(f"  [{tag}] len{len(c)} min{np.nanmin(e):.2f} | {s}")
    print("high-containing chains: starts high", nstart_high, "| has rise in chain", nrise)
