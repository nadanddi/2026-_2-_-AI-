# -*- coding: utf-8 -*-
"""VERIFY3 c (2026-10-12 집 클로드; addendum v2 item 3, rule fixed before running):
ELIMINATION by training labels.  Target t (unlabelled). Own date S0 = visible records with exactly equal 0 h outdoor.
Previous date D1 = date (>=90% equal outdoor, visible records) of the a-side of label-certain training links into S0 (most frequent).
A labelled D1 member j is 'taken' if its EC/ST midnight continuation (feat4 label_successor cost, extrapolated) matches a
labelled S0 member k with cost <= 2 and j and k are mutual best.  If exactly ONE D1 member is left untaken (visible, labelled or not),
predict it as t's previous day.  Calibration: every label-certain training record b treated as unlabelled in a leave-one-out world
(all other training records labelled; evaluation records follow the evaluation visibility rule of b's ID); links touching b excluded.
Accuracy vs truth, split by #visible D1 members and #untaken.  Then applied to evaluation records (visible = training + same-ID earlier).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
OUT = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLa[TLa.cert]; truth = {int(b): int(a) for a, b in zip(TLc.a1, TLc.b)}
links = [(int(a), int(b)) for a, b in zip(TLc.a1, TLc.b) if TRAIN[int(a)] and TRAIN[int(b)]]
Xf = X.reshape(N, -1)
pe = lambda A, i: A[i, 23] + 0.5 * (A[i, 23] - A[i, 22]); pb = lambda A, j: A[j, 0] - 0.5 * (A[j, 1] - A[j, 0])
cost = lambda j, k: abs(pe(EC, j) - pb(EC, k)) / .01 + abs(pe(ST, j) - pb(ST, k)) / .3
def run(t, vis, lab):
    vi = np.where(vis)[0]
    S0 = [j for j in vi if np.all(np.abs(X[j, 0] - X[t, 0]) < 1e-6)]
    lk = [(a, b) for a, b in links if a != t and b != t and lab[a] and lab[b] and b in S0 and a not in S0]
    if not lk: return None
    As = [a for a, _ in lk]
    best = max(As, key=lambda a: sum((np.abs(Xf[a2] - Xf[a]) < 1e-6).mean() >= .9 for a2 in As))
    D1 = [j for j in vi if (np.abs(Xf[j] - Xf[best]) < 1e-6).mean() >= .9 and j not in S0]
    L0 = [k for k in S0 if lab[k]]; LD = [j for j in D1 if lab[j]]
    taken = set()
    for k in L0:
        if not LD: break
        cj = [cost(j, k) for j in LD]; j = LD[int(np.argmin(cj))]
        if min(cj) <= 2 and min(L0, key=lambda q: cost(j, q)) == k: taken.add(j)
    left = [j for j in D1 if j not in taken]
    return dict(nD1=len(D1), nS0=len(S0) + 1, nL0=len(L0), ntaken=len(taken), nleft=len(left), pick=left[0] if len(left) == 1 else None,
                D1=" ".join(rid(j) + ("*" if lab[j] else "") + ("x" if j in taken else "") for j in D1))
rows = []
for b, a in truth.items():
    if not TRAIN[b]: continue
    lab = TRAIN.copy(); lab[b] = False
    vis = TRAIN | ((FARM == FARM[b]) & (NUM < NUM[b])); vis[b] = False
    r = run(b, vis, lab)
    if r is None: continue
    r.update(rec=rid(b), truth=rid(a), truth_inD1=int(a in [idx[s.rstrip('*x')] for s in r["D1"].split()]),
             ok=(int(r["pick"] == a) if r["pick"] is not None else -1), seg2=int(NUM[b] >= 179))
    if r["pick"] is not None: r["pick"] = rid(r["pick"])
    rows.append(r)
D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "vf3c_loo_v1.csv"), index=False, encoding="utf-8-sig")
print("LOO records with D1 found:", len(D), "| truth in D1 %.3f" % D.truth_inD1.mean())
P = D[D.ok >= 0]
print("unique-leftover cases:", len(P), "accuracy %.3f" % P.ok.mean())
print(P.groupby(["nD1", "ntaken"]).ok.agg(["size", "mean"]).round(3).to_string())
print("full configuration (nD1=4, ntaken=3):", len(P[(P.nD1 == 4) & (P.ntaken == 3)]), "acc %.3f" % P[(P.nD1 == 4) & (P.ntaken == 3)].ok.mean())
print("wrong cases:"); print(P[P.ok == 0][["rec", "truth", "pick", "D1", "nS0", "nL0"]].to_string())
# evaluation
E = pd.read_csv(os.path.join(os.path.dirname(OUT), "..", "local", "linker_v7", "eval60_predecessor_v7.csv"))
print("\nEVAL:")
for r_ in E.itertuples():
    t = idx[r_.row]; lab = TRAIN.copy(); vis = lab | ((FARM == FARM[t]) & (NUM < NUM[t])); vis[t] = False
    r = run(t, vis, lab)
    if r is None: print(f"  {r_.row}: no training-linked previous date"); continue
    pk = rid(r["pick"]) if r["pick"] is not None else None
    print(f"  {r_.row}: v7 cand {r_.prev_h0} | D1 {r['D1']} | taken {r['ntaken']} left {r['nleft']} -> pick {pk}" + (f" EC23 {EC[r['pick'],23]:.3f}" if r["pick"] is not None and TRAIN[r["pick"]] else ""))
