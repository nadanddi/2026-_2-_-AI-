# -*- coding: utf-8 -*-
"""HA1 (2026-10-12 집 클로드; hourly anchor update for EC rows h >= 6 of records WITHOUT a 0 h anchor).  Fixed before running.
Rule check (all allowed): target's own inputs 0..h; training records (any date/farm, inputs+labels, 6.630); same-ID earlier records'
inputs (date matching only); no other-ID evaluation inputs; independent per-record decisions; constants from training folds only.
Per EV fold k (hidden = 'val' + 'del' of fold k):
  date of t      : components (full-day outdoor >= 90% equal) built from all records; t's component = that of a VISIBLE record
                   (training not hidden, or same-ID smaller-number val) whose hour-0 outdoor equals t's hour-0 outdoor exactly.
  date graph     : edges comp(a) -> comp(b) from certain truth links between TRAINING records not hidden in fold k; predecessor
                   date D1 = most frequent source into comp(t).
  candidates     : labelled training members of D1 not hidden in fold k.
  pick at hour h : argmin fp_dist(t, a, h) (linker v6 fingerprint, t's hours 0..h); accept if 2nd-best gap >= .3 (fixed).
  correction     : pred_h = EA2_t,h + w * (A_a23 - cm_t,h), cm = mean of EA2_t over hours 0..h, applied for h >= 6 when accepted.
  w              : LS over accepted targets of OTHER folds with number < 179 (rows h >= 6) of residual on (A - cm), clipped [0, 1].
PASS iff pass-1 targets with any accepted hour: all 3 seeds better (all-hour RMSE of those records) AND chunk-cluster P(worse) < .025
AND top-record share <= 50%; pass-2 direction reported.  0 h is unchanged by design.
"""
import os, sys, io
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
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TLa = TLa[TLa.cert]; links = [(int(a), int(b)) for a, b in zip(TLa.a1, TLa.b) if TRAIN[int(a)] and TRAIN[int(b)]]
Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); V = Fv[Fv.role == "val"].set_index("rec")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
anch = R.groupby("rec").anchored.first()
ECY = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); ECY = ECY[ECY.row_id.str[:3].isin(["F13", "F47"])]
ECY["rec"] = ECY.row_id.str[:7]; ECY["h"] = ECY.row_id.str[8:].astype(int); EC23 = ECY[ECY.h == 23].set_index("rec").sub_ec
picks = []   # (t, fold, h, a, gap)
for k in sorted(Fv.fold.unique()):
    hidden = {idx[r] for r in Fv[Fv.fold == k].rec if r in idx}
    lk = [(a, b) for a, b in links if a not in hidden and b not in hidden]
    labc = {}
    for j in np.where(TRAIN)[0]:
        if j not in hidden: labc.setdefault(comp[j], []).append(j)
    for r_ in V[V.fold == k].index:
        if r_ not in idx or bool(anch.get(r_, True)): continue
        t = idx[r_]
        vis = [j for j in range(N) if j != t and ((TRAIN[j] and j not in hidden) or (FARM[j] == FARM[t] and NUM[j] < NUM[t] and rid(j) in V.index and V.at[rid(j), "fold"] == k))]
        m = [j for j in vis if np.all(np.abs(X[j, 0] - X[t, 0]) < 1e-6)]
        if not m: continue
        ct = comp[m[0]]
        E = Counter(comp[a] for a, b in lk if comp[b] == ct and comp[a] != ct)
        if not E: continue
        mem = labc.get(E.most_common(1)[0][0], [])
        if not mem: continue
        for h in range(6, 24):
            sc = np.nan_to_num(np.array([F6.fp_dist(t, j, h) for j in mem]), nan=99); o = np.argsort(sc)
            gap = (sc[o[1]] - sc[o[0]]) if len(mem) > 1 else 9
            if gap >= .3: picks.append(dict(t=r_, fold=k, h=h, a=rid(mem[o[0]]), gap=gap))
    print("fold", k, "picks so far", len(picks), flush=True)
Pk = pd.DataFrame(picks); Pk["A"] = Pk.a.map(EC23)
print("targets with >=1 accepted hour:", Pk.t.nunique(), "| pass-2:", sorted(Pk[Pk.t.str[4:7].astype(int) >= 179].t.unique()))
Y = R.groupby(["rec", "hour"]).y.first().unstack(); out = {}; base = {}
for sd, G in R.groupby("seed"):
    P = G.pivot_table(index="rec", columns="hour", values="EA2"); base[sd] = P; new = P.copy()
    cm = P.cumsum(1) / np.arange(1, 25)
    for k in Pk.fold.unique():
        oth = Pk[(Pk.fold != k) & (Pk.t.str[4:7].astype(int) < 179)]
        x = np.array([r.A - cm.at[r.t, r.h] for r in oth.itertuples()]); yv = np.array([Y.at[r.t, r.h] - P.at[r.t, r.h] for r in oth.itertuples()])
        w = float(np.clip((x * yv).sum() / (x * x).sum(), 0, 1))
        for r in Pk[Pk.fold == k].itertuples(): new.at[r.t, r.h] = P.at[r.t, r.h] + w * (r.A - cm.at[r.t, r.h])
    out[sd] = new
print("w example %.3f" % w)
rng = np.random.default_rng(20261224); recs_all = sorted(Pk.t.unique())
for part, cond in (("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
    recs = [r for r in recs_all if cond(r)]
    if not recs: print(part, "none"); continue
    cells = [rm((out[s].loc[recs] - Y.loc[recs]).values) / rm((base[s].loc[recs] - Y.loc[recs]).values) - 1 for s in out]
    per = pd.Series(np.mean([(((out[s].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((base[s].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for s in out], 0), index=recs)
    ch = V.chunk.reindex(recs).values; cl = per.groupby(ch).sum(); ii = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
    share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
    print(f"{part}: records {len(recs)} | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | cluster P {pw:.3f} | top share {share:.2f}")
    if part == "pass-1": print("VERDICT HA1:", "PASS" if all(x < 0 for x in cells) and pw < .025 and share <= .5 else "FAIL")
