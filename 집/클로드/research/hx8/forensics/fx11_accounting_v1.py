# -*- coding: utf-8 -*-
# hx8/forensics fx11 v1: '동 장부' 규칙 — 다음날 묶음에 c 의 동이 반드시 있는지 셈으로 보장될 때만 쓰기(fx9 오짝 원인 = 진짜 다음날 기록 부재).
# 사전 고정(실행 전):
#   D = 학습 F13/F47 날짜 묶음 크기 최댓값(학습만으로 '동 수' 추정). 대상 c 의 쌍둥이 T(학습), 다음날 묶음 B'(쌍둥이의 뚜렷한 정답 후속,
#   c1 ≤ 2·여유 ≥ 10). 쌍둥이 후속 배정: 뚜렷한 후속이면 그것, 아니면 B' 안 지문 최근접 acd ≤ q90(.623, fx10)일 때만 배정.
#   남은 R = B' − 배정. 보장 조건: |R| > D − |T| − 1 (c 가 아닌 미지 동 수보다 R 이 많음 → c 의 동이 R 안에 있음).
#   E1a: 모델 OOF ≥ 1.0 & 보장 & R 전원 0시 < .9 → c 를 R 중 c 지문 최근접의 0시 값으로. 판정 기준 fx9 와 같음(서로 다른 기록 < 10 이면 판정 불가).
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
key = {}
for i in tr: key.setdefault(np.round(X[i], 1).tobytes(), []).append(i)
bund = {i: v for v in key.values() for i in v}
sizes = pd.Series([len(v) for v in key.values()])
D = int(sizes.max()); print("train bundle sizes:", sizes.value_counts().sort_index().to_dict(), "-> D =", D)
t = pd.read_pickle(S3 + r"\truth.pkl")
t = t[t.a1.map(lambda a: TRAIN[a]) & t.b.map(lambda b: TRAIN[b])].copy(); t["marg"] = t.c2 / t.c1.clip(lower=.05)
good = t[(t.c1 <= 2.0) & (t.marg >= 10)]
succ = {}
for _, r in good.iterrows(): succ.setdefault(int(r.a1), []).append(int(r.b))
truth_next = {int(r.a1): int(r.b) for _, r in t[t.c1 <= 1.0].iterrows()}
Q90 = 0.623
def ledger(c, twins):
    S = [s for k in twins for s in succ.get(k, [])]
    if not S: return None
    B = [k for k in bund.get(S[0], [S[0]]) if k != c]
    assigned = set()
    for k in twins:
        s = [x for x in succ.get(k, []) if x in B]
        if s: assigned.add(s[0]); continue
        rest = [x for x in B if x not in assigned]
        if rest:
            j = min(rest, key=lambda x: acd(k, x))
            if acd(k, j) <= Q90: assigned.add(j)
    R = [x for x in B if x not in assigned]
    guaranteed = len(R) > D - len(twins) - 1
    pick = min(R, key=lambda x: acd(c, x)) if R else None
    return dict(R=R, guaranteed=guaranteed, pick=pick, nB=len(B), nT=len(twins), nA=len(assigned))
P = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
P = P[P.tag == "CUR"].copy(); P["rec"] = P.row_id.str[:7]; P["h"] = P.row_id.str[8:].astype(int)
LG = {c: ledger(c, [k for k in bund.get(c, []) if k != c]) for c in tr}
rows = []
for (v, s), S in P.groupby(["validator", "seed"]):
    for r, g in S.groupby("rec"):
        c = ix[r]; L = LG.get(c)
        if L is None or len(g) != 24 or L["pick"] is None: continue
        g = g.sort_values("h"); sg = g.sg.values; y = EC[c]
        rN = all(EC[x, 0] < .9 for x in L["R"])
        rows.append(dict(validator=v, seed=s, rec=r, sgm=sg.mean(), y=y.mean(), guaranteed=L["guaranteed"], R=" ".join(rid[L["R"]]),
                         allnormal=rN, pick=rid[L["pick"]], d0=EC[L["pick"], 0], pick_true=truth_next.get(c) == L["pick"],
                         true_in_R=truth_next.get(c) in L["R"], sse_m=((sg - y) ** 2).sum(), sse_a=((EC[L["pick"], 0] - y) ** 2).sum()))
A = pd.DataFrame(rows); A.to_csv(os.path.join(OUT, "fx11_cases_v1.csv"), index=False)
g47 = A[(A.validator == "DIAG10") & (A.seed == 47)]
print("DIAG10 s47 records with ledger:", len(g47), "| guaranteed:", int(g47.guaranteed.sum()),
      "| true next in R when guaranteed:", round(g47[g47.guaranteed].true_in_R.mean(), 3), "vs not guaranteed:", round(g47[~g47.guaranteed].true_in_R.mean(), 3))
for (v, s), g in A.groupby(["validator", "seed"]):
    f = g[(g.sgm >= 1.0) & g.guaranteed & g.allnormal]
    print(f"{v} s{s}: E1a fired {len(f)} {list(f.rec)} | day RMSE model {np.sqrt(((f.sgm - f.y) ** 2).mean()) if len(f) else np.nan:.3f} vs anchor {np.sqrt(((f.d0 - f.y) ** 2).mean()) if len(f) else np.nan:.3f}")
# looser: model-high & guaranteed (any R) — how often is truth near the guaranteed-R pick?
f = A[(A.validator == "DIAG10") & (A.sgm >= 1.0) & A.guaranteed].groupby("rec").agg(y=("y", "mean"), sg=("sgm", "mean"), d0=("d0", "first"), pick=("pick", "first"), ok=("pick_true", "first"), R=("R", "first"))
print("\nDIAG10 model-high & guaranteed (any R level):"); print(f.round(3).to_string())
c = ix["F13_237"]; tw = [k for k in tr if (np.abs(X[k] - X[c]) < 1e-6).all()]
L = ledger(c, tw)
print("\nF13_237 ledger:", {k: (list(rid[v]) if k == "R" else (rid[v] if k == "pick" else v)) for k, v in L.items()}, "| R 0h:", {rid[x]: round(float(EC[x, 0]), 3) for x in L["R"]})
