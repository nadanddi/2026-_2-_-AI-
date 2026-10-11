# -*- coding: utf-8 -*-
"""SAC-r (post-SAC fix, pre-registered before running: w = MEDIAN over the 8 leave-one-fold-out LS slopes (robust to one high-leverage record; SAC fold 5 slope 1.04 vs others .25-.33), computed on pass-1 targets of the arm, then clipped [0,1]; also prints the highest-leverage targets).  Base: SAC (2026-10-12 집 클로드; user: "1번 진행" = use the DM1 competitor judgement (unlabelled same-ID winner .93 correct) as the LINK
CHECK of the soft anchor SA (6.632).  Fixed before running.  Base: EV EA2 final chain (BASE season, seeds 7/101/2024).
Soft anchor: pred_h = EA2_t,h + c_h * w * (EA2_p,23 - EA2_t,0) for EV val t without EA2 anchor, p = same-ID smaller-number val record
of the same fold (inputs allowed, 6.631); w = LS over other folds' pass-1 targets of the same arm, clipped [0,1]; c_h = DIAG10 pass-1
0 h residual carry.
Arms: SA0 = p from linker v7m (unlabelled, same ID, p >= .8)  [reference, = 6.632 SA];
      SAC1 = p = DM1 winner (date map + competitors at 0 h) when the winner is an unlabelled same-ID record with gap >= .5;
      SAC2 = SA0 AND SAC1 agree on the same p.
PASS (SAC1 / SAC2 each): pass-1 all 3 seeds better AND chunk-cluster P(worse) < .0125 AND top-record share <= 50%; pass-2 direction.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
import feat6 as F6
from collections import Counter
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
Xf = X.reshape(N, -1); par = list(range(N))
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for i in range(N):
    eq = (np.abs(Xf[i + 1:] - Xf[i]) < 1e-6).mean(1)
    for j in np.where(eq >= .9)[0] + i + 1: par[fd(j)] = fd(i)
comp = np.array([fd(i) for i in range(N)])
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLa[TLa.cert]; links = [(int(a), int(b)) for a, b in zip(TLc.a1, TLc.b) if TRAIN[int(a)] and TRAIN[int(b)]]
truth = {rid(int(b)): rid(int(a)) for a, b in zip(TLa.a1, TLa.b)}
Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); V = Fv[Fv.role == "val"].set_index("rec")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
anch = R.groupby("rec").anchored.first()
ECY = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); ECY = ECY[ECY.row_id.str[:3].isin(["F13", "F47"])]
ECY["rec"] = ECY.row_id.str[:7]; ECY["h"] = ECY.row_id.str[8:].astype(int); EC23 = ECY[ECY.h == 23].set_index("rec").sub_ec
rows = []
for k in sorted(Fv.fold.unique()):
    hidden = {idx[r] for r in Fv[Fv.fold == k].rec if r in idx}
    lk = [(a, b) for a, b in links if a not in hidden and b not in hidden]
    for r_ in V[V.fold == k].index:
        if r_ not in idx or bool(anch.get(r_, True)): continue
        t = idx[r_]
        okv = lambda j: j != t and ((TRAIN[j] and j not in hidden) or (FARM[j] == FARM[t] and NUM[j] < NUM[t] and rid(j) in V.index and V.at[rid(j), "fold"] == k))
        m = [j for j in range(N) if okv(j) and np.all(np.abs(X[j, 0] - X[t, 0]) < 1e-6)]
        if not m: continue
        E = Counter(comp[a] for a, b in lk if comp[b] == comp[m[0]] and comp[a] != comp[m[0]])
        if not E: continue
        D1 = E.most_common(1)[0][0]
        cand = [j for j in np.where(comp == D1)[0] if okv(j)]
        if len(cand) < 1: continue
        sc = np.array([np.nansum(np.minimum(np.abs(VARS[j, 23] - VARS[t, 0]) / sdV, 6)) + np.nan_to_num(F6.fp_dist(t, j, 0), nan=6) for j in cand])
        o = np.argsort(sc); win = cand[o[0]]; gap = (sc[o[1]] - sc[o[0]]) if len(cand) > 1 else 9
        lab = bool(TRAIN[win] and win not in hidden)
        rows.append(dict(t=r_, fold=k, win=rid(win), lab=lab, gap=gap, win_val=(not lab) and rid(win) in V.index, ncand=len(cand), n_unlab=sum(not (TRAIN[j] and j not in hidden) for j in cand),
                         truth=truth.get(r_), accept=lab and gap >= .5, A=EC23.get(rid(win), np.nan)))
D = pd.DataFrame(rows)
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
def sa0(t):
    if t not in L.index or bool(anch.get(t, True)): return None
    pr = L.loc[t]; p = pr.prev
    if pr.prev_is_labelled or pr.xf or pr.p < .8 or p not in V.index or V.at[p, "fold"] != V.at[t, "fold"] or p[:3] != t[:3] or int(p[4:7]) >= int(t[4:7]): return None
    return p
A0 = {t: sa0(t) for t in V.index}; A0 = {t: p for t, p in A0.items() if p}
A1 = {r.t: r.win for r in D.itertuples() if r.win_val and r.gap >= .5 and r.win[:3] == r.t[:3]}
A2 = {t: p for t, p in A0.items() if A1.get(t) == p}
print("targets: SA0 %d | SAC1 %d | SAC2 %d" % (len(A0), len(A1), len(A2)))
for nm, A in (("SA0", A0), ("SAC1", A1), ("SAC2", A2)):
    print("  %s link accuracy vs truth: %.2f (n %d)" % (nm, np.mean([truth.get(t) == p for t, p in A.items() if truth.get(t)]), sum(1 for t in A if truth.get(t))))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; QD = O.groupby(["rec", "hour"]).sg.mean().unstack(); YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
p1 = [r for r in QD.index if int(r[4:7]) < 179]; e = (YD.loc[p1] - QD.loc[p1]).values
c = np.array([(e[:, 0] * e[:, h]).sum() / (e[:, 0] ** 2).sum() for h in range(24)])
Y = R.groupby(["rec", "hour"]).y.first().unstack(); rng = np.random.default_rng(20261226)
for nm, A in (("SA0", A0), ("SAC1", A1), ("SAC2", A2)):
    T = pd.DataFrame([(t, p, V.at[t, "fold"], V.at[t, "chunk"]) for t, p in A.items()], columns=["t", "p", "fold", "chunk"])
    out = {}; base = {}
    for sd, G in R.groupby("seed"):
        P = G.pivot_table(index="rec", columns="hour", values="EA2"); base[sd] = P; new = P.copy()
        slopes = []
        for k in sorted(T.fold.unique()):
            oth = T[(T.fold != k) & (T.t.str[4:7].astype(int) < 179)]
            if len(oth) < 3: continue
            x = P.loc[oth.p, 23].values - P.loc[oth.t, 0].values; yv = Y.loc[oth.t, 0].values - P.loc[oth.t, 0].values
            slopes.append((x * yv).sum() / (x * x).sum())
        w = float(np.clip(np.median(slopes), 0, 1)) if slopes else .5
        for r in T.itertuples(): new.loc[r.t] = P.loc[r.t].values + c * w * (P.at[r.p, 23] - P.at[r.t, 0])
        if sd == list(R.seed.unique())[0]:
            lev = sorted(((abs(P.at[r.p, 23] - P.at[r.t, 0]), r.t, r.p) for r in T.itertuples()), reverse=True)[:3]
            print("  %s w %.3f | highest leverage: %s" % (nm, w, ", ".join("%s<-%s |x| %.2f" % (b, c_, a) for a, b, c_ in lev)))
        out[sd] = new
    for part, cond in (("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
        recs = [r for r in T.t if cond(r)]
        if not recs: print(f"  {nm} {part}: none"); continue
        h0 = np.mean([rm(out[s].loc[recs, 0] - Y.loc[recs, 0]) / rm(base[s].loc[recs, 0] - Y.loc[recs, 0]) - 1 for s in out])
        cells = [rm((out[s].loc[recs] - Y.loc[recs]).values) / rm((base[s].loc[recs] - Y.loc[recs]).values) - 1 for s in out]
        per = pd.Series(np.mean([(((out[s].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((base[s].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for s in out], 0), index=recs)
        cl = per.groupby(T.set_index("t").chunk.reindex(recs).values).sum(); ii = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
        share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
        verdict = ("PASS" if all(x < 0 for x in cells) and pw < .0125 and share <= .5 else "FAIL") if (part == "pass-1" and nm != "SA0") else ""
        print(f"  {nm} {part}: records {len(recs)} | 0h {100*h0:+.1f}% | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | cluster P {pw:.3f} | top share {share:.2f} {verdict}")
