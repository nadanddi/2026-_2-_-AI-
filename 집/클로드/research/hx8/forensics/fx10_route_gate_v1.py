# -*- coding: utf-8 -*-
# hx8/forensics fx10 v1: fx9 E1 FAIL(DIAG10 발동 3 중 2 오짝 — F13_146→F47_149, F13_217→F47_214: 진짜 다음날 기록이 묶음에 없는데
#   남은 다른 동을 고름) 뒤의 점검. 사후 동기임을 밝힘(결과 보고 게이트를 떠올림) → 판정은 같은 기준, 결과는 '참고'로만.
# 사전 고정(이 스크립트 실행 전): E1g = E1 + '고른 기록 지문 거리 acd ≤ q' 일 때만 발동. q = 학습 진짜 다음날 쌍(c1 ≤ 1) acd 의 90% 분위
#   (발동 사례 결과와 무관하게 분포에서 결정). F13_237 경로(쌍둥이 F47_230·231 → 후속 묶음 → 지문)도 계산해 q 와 비교.
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd
from feat4 import X, IN_, AC, EC, ST, FARM, NUM, N, TRAIN, S3

OUT = os.path.dirname(os.path.abspath(__file__))
rid = np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)]); ix = {r: i for i, r in enumerate(rid)}
tr = np.where(TRAIN)[0]
sdA = np.nanstd(AC[tr].reshape(-1, 5), 0) + 1e-6
def acd(x, y): return np.nanmean(np.abs(AC[x] - AC[y]) / sdA)
t = pd.read_pickle(S3 + r"\truth.pkl")
t = t[t.a1.map(lambda a: TRAIN[a]) & t.b.map(lambda b: TRAIN[b])].copy()
tp = t[t.c1 <= 1.0]
dt = np.array([acd(int(r.a1), int(r.b)) for _, r in tp.iterrows()])
q = float(np.quantile(dt, .9))
print(f"true next-day pair acd: n {len(dt)} median {np.median(dt):.3f} q75 {np.quantile(dt, .75):.3f} q90 {q:.3f}")
A = pd.read_csv(os.path.join(OUT, "fx9_cases_v1.csv"))
A["acd_pick"] = [acd(ix[r], ix[d]) for r, d in zip(A.rec, A.d)]
print("acd of route pick: correct", A[A.d_true].acd_pick.median().round(3), "| wrong", A[~A.d_true].acd_pick.median().round(3),
      "| share wrong among acd<=q:", round((~A[A.acd_pick <= q].d_true).mean(), 3), "vs among acd>q:", round((~A[A.acd_pick > q].d_true).mean(), 3))
out = []
for (v, s), g in A.groupby(["validator", "seed"]):
    h = g[(g.sgm >= 1.0) & (g.d0 < .9)]; hg = h[h.acd_pick <= q]
    out.append(dict(validator=v, seed=s, e1_fire=len(h), e1g_fire=len(hg), recs=" ".join(hg.rec),
                    e1g_day_model=np.sqrt(((hg.sgm - hg.y) ** 2).mean()) if len(hg) else np.nan,
                    e1g_day_anchor=np.sqrt(((hg.d0 - hg.y) ** 2).mean()) if len(hg) else np.nan,
                    dropped=" ".join(f"{r}({a:.2f})" for r, a in zip(h[h.acd_pick > q].rec, h[h.acd_pick > q].acd_pick))))
print(pd.DataFrame(out).round(3).to_string(index=False))
# F13_237 route (eval record: twins by own outdoor 24h; at hour 0 the 0h outdoor alone gives the same twins — see fx6)
c = ix["F13_237"]
tw = [k for k in tr if (np.abs(X[k] - X[c]) < 1e-6).all()]
t2 = t.copy(); t2["marg"] = t2.c2 / t2.c1.clip(lower=.05)
good = t2[(t2.c1 <= 2.0) & (t2.marg >= 10)]
S = [int(r.b) for _, r in good.iterrows() if int(r.a1) in tw]
key = {}
for i in tr: key.setdefault(np.round(X[i], 1).tobytes(), []).append(i)
bund = {i: v for v in key.values() for i in v}
B = bund[S[0]]; rest = [k for k in B if k not in S]
print("\nF13_237 twins", list(rid[tw]), "| distinctive twin successors", list(rid[S]), "| bundle", list(rid[B]))
print("  acd F13_237 -> rest:", {rid[k]: round(acd(c, k), 3) for k in rest}, f"(q90 {q:.3f})")
chain = ["F13_231", "F13_233", "F13_128", "F13_132", "F13_129", "F13_232", "F47_231", "F47_132", "F47_230"]
print("  acd among 'shade-53' dong chain & others:")
M = pd.DataFrame([[acd(ix[a], ix[b]) for b in chain] for a in chain], index=chain, columns=chain)
print(M.round(2).to_string())
