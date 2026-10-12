# -*- coding: utf-8 -*-
# verify_rule-safety vr1: forensics 결과 규정 감사용 재계산(읽기 전용, 원 파일 미수정).
# (a) 참조 기록의 학습/평가 여부 (b) F13_237 시각별(0..h) 쌍둥이 (c) fx11 장부의 시각 인과판 pick
# (d) q90 배정 문턱 없을 때 / 미배정 쌍둥이를 셈에 넣을 때 보장 여부 (e) F47_133 묶음 소속 (f) fx6 v2 구조 변형 재현
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd
from feat4 import X, IN_, AC, EC, ST, FARM, NUM, N, TRAIN, S3

rid = np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)]); ix = {r: i for i, r in enumerate(rid)}
tr = np.where(TRAIN)[0]
sdA = np.nanstd(AC[tr].reshape(-1, 5), 0) + 1e-6
def acd(x, y, h=23): return np.nanmean(np.abs(AC[x, :h + 1] - AC[y, :h + 1]) / sdA)
refs = ["F13_237", "F13_235", "F13_233", "F13_231", "F13_232", "F47_230", "F47_231", "F47_132", "F47_133", "F13_128", "F13_129",
        "F13_102", "F13_103", "F47_106", "F47_229", "F13_217", "F47_227"]
print("(a) TRAIN flags:", {r: bool(TRAIN[ix[r]]) for r in refs if r in ix}, "| missing:", [r for r in refs if r not in ix])
key = {}
for i in tr: key.setdefault(np.round(X[i], 1).tobytes(), []).append(i)
bund = {i: v for v in key.values() for i in v}
D = max(len(v) for v in key.values())
t = pd.read_pickle(S3 + r"\truth.pkl")
t = t[t.a1.map(lambda a: TRAIN[a]) & t.b.map(lambda b: TRAIN[b])].copy(); t["marg"] = t.c2 / t.c1.clip(lower=.05)
good = t[(t.c1 <= 2.0) & (t.marg >= 10)]
succ = {}
for _, r in good.iterrows(): succ.setdefault(int(r.a1), []).append(int(r.b))
c = ix["F13_237"]
tw_sets = set(); picks = []
for h in range(24):
    tw = [k for k in tr if (np.abs(X[k, :h + 1] - X[c, :h + 1]) < 1e-6).all()]
    tw_sets.add(tuple(rid[tw]))
    S = [s for k in tw for s in succ.get(k, [])]
    B = [k for k in bund.get(S[0], [S[0]]) if k != c]
    assigned = set()
    for k in tw:
        s = [x for x in succ.get(k, []) if x in B]
        if s: assigned.add(s[0]); continue
        rest = [x for x in B if x not in assigned]
        if rest:
            j = min(rest, key=lambda x: acd(k, x))
            if acd(k, j) <= 0.623: assigned.add(j)
    R = [x for x in B if x not in assigned]
    picks.append((h, rid[min(R, key=lambda x: acd(c, x, h))], {rid[x]: round(acd(c, x, h), 3) for x in R}))
print("(b) distinct causal twin sets over h=0..23:", tw_sets)
print("(c) causal ledger pick per h (first 4, last 2):", picks[:4], picks[-2:], "| all F13_128:", all(p[1] == "F13_128" for p in picks))
# (d) ledger variants for full-day twins
tw = [k for k in tr if (np.abs(X[k] - X[c]) < 1e-6).all()]
S = [s for k in tw for s in succ.get(k, [])]; B = [k for k in bund.get(S[0], [S[0]]) if k != c]
print("(d) twins", list(rid[tw]), "| distinctive succ", list(rid[S]), "| B", list(rid[B]), "| D", D)
for gate in (0.623, np.inf):
    assigned = set(); unassigned_tw = 0
    for k in tw:
        s = [x for x in succ.get(k, []) if x in B]
        if s: assigned.add(s[0]); continue
        rest = [x for x in B if x not in assigned]
        j = min(rest, key=lambda x: acd(k, x)) if rest else None
        if j is not None: print(f"    twin {rid[k]} nearest in B {rid[j]} acd {acd(k, j):.3f} (gate {gate})")
        if j is not None and acd(k, j) <= gate: assigned.add(j)
        else: unassigned_tw += 1
    R = [x for x in B if x not in assigned]
    g_fx11 = len(R) > D - len(tw) - 1
    g_fix = len(R) > (D - len(tw) - 1) + unassigned_tw
    print(f"  gate {gate}: R {list(rid[R])} | fx11 formula guaranteed {g_fx11} | counting unassigned twins guaranteed {g_fix}")
# (e) where is F47_230's diagnostic successor F47_133?
for nm in ("F47_133", "F47_132"):
    print("(e)", nm, "bundle:", list(rid[bund.get(ix[nm], [])]))
print("    all links from twins:", [(rid[int(r.a1)], rid[int(r.b)], round(r.c1, 3), round(r.marg, 1)) for _, r in t[t.a1.isin(tw)].iterrows()])
# (f) fx6 v2 structure variant reproduction (causal in h; only training labels)
a, d = ix["F13_233"], ix["F13_128"]
fb = np.array([EC[a, 23] + (EC[d, 0] - EC[a, 23]) * (48 + h - 23) / (72 - 23) for h in range(24)])
sub = pd.read_csv(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)", "submission_22.csv"))
p13 = sub[sub.row_id.str.startswith("F13_237")].sort_values("row_id").sub_ec.values
rms = np.sqrt(np.mean((fb - p13) ** 2))
print(f"(f) fb mean {fb.mean():.3f}, RMS vs 13th {rms:.3f}")
# (g) scenario arithmetic base: what EC column RMSE R0 reproduces .105 / .146 (1440 eval rows)
dm = np.sum((fb - p13) ** 2) / 1440
for R0 in (0.1279, 0.1271):
    print(f"(g) base {R0}: right -> {np.sqrt(R0**2 - dm):.4f}, wrong -> {np.sqrt(R0**2 + dm):.4f}")
