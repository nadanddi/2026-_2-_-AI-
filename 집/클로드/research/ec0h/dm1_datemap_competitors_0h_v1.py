# -*- coding: utf-8 -*-
"""DM1 (2026-10-12 집 클로드; EXPLORATORY, designed after HA1 diagnosis).  EC 0 h anchor with DATE fixed by the training-label date
map and HOUSE chosen at 0 h among ALL visible records of the predecessor date, including UNLABELLED same-ID earlier records as
competitors (if one of them wins, no anchor -> avoids forcing a wrong labelled house).
Per EV fold k (hidden = val + del of fold k):
  own date    : comp of a visible record (training not hidden, or same-ID smaller-number val of fold k) with exactly equal 0 h outdoor.
  pred. date  : most frequent source comp of certain links between non-hidden TRAINING records into comp(t).
  candidates  : members of the predecessor date that are visible: training not hidden (labelled) + same-ID smaller-number val of fold k
                (unlabelled competitors).  'del' and other-ID val are excluded (rule).
  score       : z-midnight indoor/actuator continuity (sum over 8 vars of |V_c(23) - V_t(0)| / sdV, linker v4 definition) + fp_dist(t,c,0).
  accept      : winner labelled AND gap to runner-up >= .5 (fixed).  Correction (0 h decision, all hours):
                pred_h = EA2_t,h + c_h * w * (A_c23 - EA2_t,0), c_h = 0 h residual carry (DIAG10 pass-1), w from other folds' pass-1.
Reports: pick accuracy vs truth (accepted / all), 0 h RMSE and all-hour RMSE change on accepted pass-1 / pass-2 records, cluster P.
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
        rows.append(dict(t=r_, fold=k, win=rid(win), lab=lab, gap=gap, ncand=len(cand), n_unlab=sum(not (TRAIN[j] and j not in hidden) for j in cand),
                         truth=truth.get(r_), accept=lab and gap >= .5, A=EC23.get(rid(win), np.nan)))
D = pd.DataFrame(rows); D["ok"] = D.win == D.truth
K = D.dropna(subset=["truth"])
print("targets with date+candidates:", len(D), "| winner labelled %.2f | accepted %d" % (D.lab.mean(), int(D.accept.sum())))
print("pick accuracy (truth known): all %.2f (n %d) | accepted %.2f (n %d) | winner-unlabelled correct %.2f (n %d)" % (
    K.ok.mean(), len(K), K[K.accept].ok.mean() if K.accept.any() else float('nan'), int(K.accept.sum()), K[~K.lab].ok.mean() if (~K.lab).any() else float('nan'), int((~K.lab).sum())))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; QD = O.groupby(["rec", "hour"]).sg.mean().unstack(); YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
p1 = [r for r in QD.index if int(r[4:7]) < 179]; e = (YD.loc[p1] - QD.loc[p1]).values
c = np.array([(e[:, 0] * e[:, h]).sum() / (e[:, 0] ** 2).sum() for h in range(24)])
Acc = D[D.accept].copy(); Y = R.groupby(["rec", "hour"]).y.first().unstack(); out = {}; base = {}
for sd, G in R.groupby("seed"):
    P = G.pivot_table(index="rec", columns="hour", values="EA2"); base[sd] = P; new = P.copy()
    for k in Acc.fold.unique():
        oth = Acc[(Acc.fold != k) & (Acc.t.str[4:7].astype(int) < 179)]
        x = (oth.A - P.loc[oth.t, 0].values).values; yv = (Y.loc[oth.t, 0] - P.loc[oth.t, 0]).values
        w = float(np.clip((x * yv).sum() / (x * x).sum(), 0, 1)) if len(oth) >= 3 else .5
        for r in Acc[Acc.fold == k].itertuples(): new.loc[r.t] = P.loc[r.t].values + c * w * (r.A - P.at[r.t, 0])
    out[sd] = new
rng = np.random.default_rng(20261225)
for part, cond in (("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
    recs = [r for r in Acc.t if cond(r)]
    if not recs: print(part, "none"); continue
    h0 = np.mean([rm(out[s].loc[recs, 0] - Y.loc[recs, 0]) / rm(base[s].loc[recs, 0] - Y.loc[recs, 0]) - 1 for s in out])
    cells = [rm((out[s].loc[recs] - Y.loc[recs]).values) / rm((base[s].loc[recs] - Y.loc[recs]).values) - 1 for s in out]
    per = pd.Series(np.mean([(((out[s].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((base[s].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for s in out], 0), index=recs)
    cl = per.groupby(V.chunk.reindex(recs).values).sum(); ii = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
    share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
    print(f"{part}: accepted records {len(recs)} | 0h {100*h0:+.1f}% | all-hour cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | cluster P {pw:.3f} | top share {share:.2f}")
