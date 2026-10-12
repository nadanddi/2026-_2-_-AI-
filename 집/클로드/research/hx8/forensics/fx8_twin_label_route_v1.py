# -*- coding: utf-8 -*-
# hx8/forensics fx8 v1: 다음날 묶음을 외기 연속성(fx7, 여유 3 이상에서도 .68) 대신 '쌍둥이(같은 날짜 학습 기록)의 정답 연속 후속'으로 찾는 길.
# F13_237: 쌍둥이 F47_231(23시 2.750) → F47_132(0시 2.796) 진단 연결 c1 1.5, 2위 c2 51(여유 34배) → F47_132 묶음 {F13_128, F13_129, F47_132}.
# 학습 검산(진단): 학습 기록 c(다음날 진짜 b, c1 ≤ 1) 의 쌍둥이 t 의 후속 s_t(진단 연결, 여유 c2/c1 ≥ k)가 있을 때 b ∈ 묶음(s_t) 인 비율.
# 그리고 그 묶음 안에서 '쌍둥이 후속 빼고 c 지문 최근접'이 b 인 비율. 서술.
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd
from feat4 import X, IN_, AC, EC, ST, FARM, NUM, N, TRAIN, S3

rid = np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)]); ix = {r: i for i, r in enumerate(rid)}
tr = np.where(TRAIN)[0]
sdA = np.nanstd(AC[tr].reshape(-1, 5), 0) + 1e-6
def acd(x, y): return np.nanmean(np.abs(AC[x] - AC[y]) / sdA)
key = {}
for i in tr: key.setdefault(np.round(X[i], 1).tobytes(), []).append(i)
bund = {i: v for v in key.values() for i in v}
t = pd.read_pickle(S3 + r"\truth.pkl")
t = t[t.a1.map(lambda a: TRAIN[a]) & t.b.map(lambda b: TRAIN[b])].copy()
t["marg"] = t.c2 / t.c1.clip(lower=.05)
succ_any = {}
for _, r in t.iterrows(): succ_any.setdefault(int(r.a1), []).append((int(r.b), r.c1, r.marg))
true_next = {int(r.a1): int(r.b) for _, r in t[t.c1 <= 1.0].iterrows()}
rows = []
for c, b in true_next.items():
    for tw in bund.get(c, []):
        if tw == c: continue
        for (s, c1, mg) in succ_any.get(tw, []):
            if s == b: continue  # twin's successor must be a different dong
            B = bund.get(s, [s])
            rest = [k for k in B if k != s]
            pick = min(rest, key=lambda k: acd(c, k)) if rest else None
            rows.append(dict(c=rid[c], b=rid[b], twin=rid[tw], s=rid[s], c1=c1, marg=mg, in_bundle=b in B,
                             pick_ok=pick == b, nrest=len(rest), seg2=NUM[c] >= 179))
D = pd.DataFrame(rows)
for k in (1, 5, 10):
    g = D[(D.c1 <= 2.0) & (D.marg >= k)]
    print(f"twin successor c1<=2 & margin>={k}: n {len(g)}, b in successor's bundle {g.in_bundle.mean():.3f}, fingerprint pick == b {g.pick_ok.mean():.3f}; seg2 n {int(g.seg2.sum())} in_bundle {g[g.seg2].in_bundle.mean():.3f} pick {g[g.seg2].pick_ok.mean():.3f}")
c = ix["F13_237"]
for tw in ("F47_230", "F47_231"):
    print(tw, "successors:", [(rid[s], round(c1, 3), round(mg, 1)) for s, c1, mg in succ_any.get(ix[tw], [])])
print("F47_132 bundle:", [rid[k] for k in bund[ix["F47_132"]]], "| F13_237 acd:", {rid[k]: round(acd(c, k), 2) for k in bund[ix["F47_132"]]})
