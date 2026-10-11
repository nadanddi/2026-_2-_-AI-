# -*- coding: utf-8 -*-
"""VERIFY3 v2 (2026-10-12 집 클로드). Plan addendum: ec0h/verify3/plan_v3_addendum_v2.md (fixed before running; post-hoc design).
Two-stage previous-day model: (1) previous DATE among visible date groups (+ 'date hidden' pseudo), (2) HOUSE within date
(+ 'hidden house' pseudo).  Calibrated on training mirror-CV world (190 label-certain links), block-out OOF.
Rule: target uses 0..h inputs only; visible = training (minus CV-hidden) + same-ID earlier records; no other-ID evaluation inputs,
no later same-ID evaluation inputs; groups formed among visible records only; 'taken' uses TRAINING labels only.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
import feat6 as F6, feat4 as F4
OUT = os.path.dirname(os.path.abspath(__file__)); H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLa[TLa.cert]; truth = {int(b): int(a) for a, b in zip(TLc.a1, TLc.b)}
Xf = X.reshape(N, -1)
EQ = np.zeros((N, N))
for i in range(N): EQ[i] = (np.abs(Xf - Xf[i]) < 1e-6).mean(1)
HS = [0, 6, 12, 23]
pe = lambda A, i: A[i, 23] + 0.5 * (A[i, 23] - A[i, 22]); pb = lambda A, j: A[j, 0] - 0.5 * (A[j, 1] - A[j, 0])

def world_cv(r):
    f = FARM[r]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo = NUM[FARM == f].min(); s = lo + ((NUM[r] - lo) // 10) * 10; m = s + sh
    rem = (FARM == f) & ((NUM == s - 1) | (NUM == s + 10)); oth = (FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)
    blk = (FARM == f) & (NUM >= s) & (NUM < s + 10); hidden = rem | oth | blk
    lab = TRAIN & ~hidden; vis = lab | ((FARM == f) & (NUM < NUM[r]) & ~rem & ~oth); vis[r] = False
    return vis, lab, f"{f}_{s}"
def world_eval(t):
    lab = TRAIN.copy(); vis = lab | ((FARM == FARM[t]) & (NUM < NUM[t])); vis[t] = False
    return vis, lab, "eval"

def groups(vi):
    par = {i: i for i in vi}
    def fd(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    sub = EQ[np.ix_(vi, vi)] >= .9
    for a, b in zip(*np.where(np.triu(sub, 1))): par[fd(vi[b])] = fd(vi[a])
    G = {}
    for i in vi: G.setdefault(fd(i), []).append(i)
    return list(G.values())

def build(t, vis, lab):
    """candidate dates and members with features for target t (all hours)."""
    vi = np.where(vis)[0]; Gs = groups(vi)
    own = set(vi[np.all(np.abs(X[vi, 0] - X[t, 0]) < 1e-6, axis=1)].tolist())
    L0 = [k for k in own if lab[k]]
    pa = X[vi, 23] + 0.5 * (X[vi, 23] - X[vi, 22]); ojv = dict(zip(vi, np.sum(np.array([1, 1, .3, .5]) * np.abs(pa - X[t, 0]) / sdO, 1)))
    dates = []
    for G in Gs:
        if any(j in own for j in G): continue
        oj = min(ojv[j] for j in G)
        if not np.isfinite(oj) or oj >= 6: continue
        mem = []
        for j in G:
            md = float(np.sum(np.minimum(np.nan_to_num(np.abs(VARS[j, 23] - VARS[t, 0]) / sdV, nan=2.0), 6)))
            dT = abs(IN_[t, 0, 0] - IN_[j, 23, 0]); dT = 3.0 if not np.isfinite(dT) else min(dT, 10)
            fp = {h: F6.fp_dist(t, j, h) for h in HS}
            taken = 0.0
            if lab[j] and L0:
                cst = [abs(pe(EC, j) - pb(EC, k)) / .01 + abs(pe(ST, j) - pb(ST, k)) / .3 for k in L0]
                kk = int(np.argmin(cst))
                if cst[kk] <= 2.0:
                    k = L0[kk]; labG = [q for q in G if lab[q]]
                    cq = [abs(pe(EC, q) - pb(EC, k)) / .01 + abs(pe(ST, q) - pb(ST, k)) / .3 for q in labG]
                    taken = float(labG[int(np.argmin(cq))] == j)
            mem.append(dict(j=j, mid=md, dT=dT, taken=taken, same=float(FARM[j] == FARM[t]), lab=float(lab[j]),
                            **{f"fp{h}": (min(fp[h], 6) if np.isfinite(fp[h]) else 2.0) for h in HS}))
        dates.append(dict(G=G, oj=oj, mem=mem))
    return dates, len(L0)

def date_X(d, h):
    if d is None: return np.array([0, 0, 0, 0, 0, 0, 1.0])
    M = d["mem"]
    return np.array([min(d["oj"], 6), min(m["mid"] for m in M) / 10, min(m["dT"] for m in M) / 5, min(m[f"fp{h}"] for m in M),
                     sum(m["taken"] for m in M), max(0, 4 - len(M)) / 4, 0.0])
DF = ["oj", "mid_min", "dT_min", "fp_min", "taken_n", "missing", "hidden_date"]
def house_X(d, h):
    M = d["mem"]; miss = max(0, 4 - len(M))
    R = [[m["mid"] / 10, m["dT"] / 5, m[f"fp{h}"], m["taken"], m["same"], 0.0, 0.0] for m in M]
    R.append([0, 0, 0, 0, 0, 1.0, np.log1p(miss)])
    return np.array(R)
HF = ["mid", "dT", "fp_h", "taken", "sameID", "hidden", "hidden_x_logmiss"]
def softmax(s): s = s - s.max(); p = np.exp(s); return p / p.sum()

# ---------- training mirror-CV groups ----------
C = pd.read_csv(os.path.join(H_, "local", "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[(O.tag == "CUR") & (O.validator == "DIAG10") & (O.hour == 0)]; P0cv = O.groupby(O.row_id.str[:7]).sg.mean()
REC = []
for b, a in truth.items():
    if not TRAIN[b]: continue
    vis, lab, blk = world_cv(b)
    dates, nL0 = build(b, vis, lab)
    ti = next((k for k, d in enumerate(dates) if a in d["G"]), None)
    REC.append(dict(b=b, a=a, blk=blk, dates=dates, ti=ti, vis_a=bool(vis[a]), nL0=nL0))
print("CV records:", len(REC), "| truth date visible & candidate %.2f | truth house visible %.2f | mean #cand dates %.1f" % (
    np.mean([r["ti"] is not None for r in REC]), np.mean([r["vis_a"] for r in REC]), np.mean([len(r["dates"]) for r in REC])))

def fit_models(recs, h):
    Gd, Gh = [], []
    for r in recs:
        Xd = np.array([date_X(d, h) for d in r["dates"]] + [date_X(None, h)])
        y = r["ti"] if r["ti"] is not None else len(r["dates"])
        Gd.append((Xd, y))
        if r["ti"] is not None:
            d = r["dates"][r["ti"]]; Xh = house_X(d, h)
            js = [m["j"] for m in d["mem"]]; y = js.index(r["a"]) if r["a"] in js else len(js)
            Gh.append((Xh, y))
    return fit_clogit(Gd, l2=0.3, iters=400), fit_clogit(Gh, l2=0.3, iters=400)
def posterior(dates, wd, wh, h):
    """returns dict j -> P(j is previous day), plus P(hidden date), and per-date P."""
    Xd = np.array([date_X(d, h) for d in dates] + [date_X(None, h)]); pdv = softmax(Xd @ wd)
    out = {}; ph = {}
    for k, d in enumerate(dates):
        p = softmax(house_X(d, h) @ wh)
        for m, q in zip(d["mem"], p[:-1]): out[m["j"]] = pdv[k] * q
        ph[k] = pdv[k] * p[-1]
    return out, pdv, ph

rows = []
for h in HS:
    for blk in sorted({r["blk"] for r in REC}):
        tr = [r for r in REC if r["blk"] != blk]; te = [r for r in REC if r["blk"] == blk]
        wd, wh = fit_models(tr, h)
        for r in te:
            post, pdv, _ = posterior(r["dates"], wd, wh, h)
            top = max(post, key=post.get) if post else None
            rc = rid(r["b"]); cand = C.prev.get(rc) if rc in C.index else None
            ci = idx[cand] if isinstance(cand, str) else None
            ok_lab = bool(rc in C.index and C.at[rc, "prev_is_labelled"] and not C.at[rc, "xf"])
            rows.append(dict(h=h, rec=rc, truth=rid(r["a"]), blk=blk, P_truth=post.get(r["a"], 0.0), top=rid(top) if top is not None else "",
                             P_top=post.get(top, 0.0) if top is not None else 0.0, top_ok=int(top == r["a"]),
                             v7m=cand, v7m_lab_same=ok_lab, P_v7m=post.get(ci, 0.0) if ci is not None else np.nan, v7m_ok=int(ci == r["a"]) if ci is not None else -1,
                             v7m_p=C.p.get(rc, np.nan), P_hidden_date=pdv[-1], A=EC[ci, 23] if ci is not None and TRAIN[ci] else np.nan,
                             y0=EC[r["b"], 0], pred0=P0cv.get(rc, np.nan)))
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "vf3b_cv_oof_v2.csv"), index=False, encoding="utf-8-sig")
def calib(D, pc, okc, name):
    s = f"  {name} n {len(D)} acc {D[okc].mean():.2f} Brier {np.mean((D[pc]-D[okc])**2):.3f} (const {np.mean((D[okc].mean()-D[okc])**2):.3f}) | "
    for a, b in [(0, .3), (.3, .5), (.5, .8), (.8, .9), (.9, 1.01)]:
        m = (D[pc] >= a) & (D[pc] < b); s += f"[{a:.1f},{b:.1f}) n{int(m.sum())} acc{D[m][okc].mean() if m.any() else float('nan'):.2f}  "
    print(s)
for h in HS:
    D = R[R.h == h]
    print(f"h={h}: top-choice acc {D.top_ok.mean():.3f}; mean P_truth {D.P_truth.mean():.3f}")
    calib(D, "P_top", "top_ok", "top choice")
    S = D[D.v7m_lab_same & (D.v7m_ok >= 0)]
    calib(S, "P_v7m", "v7m_ok", "v7m labelled same-ID choice")
    S2 = D[(D.v7m_ok >= 0)]; calib(S2, "P_v7m", "v7m_ok", "v7m any choice")
wd, wh = fit_models(REC, 0)
print("date weights h0:", " ".join(f"{n} {w:+.2f}" for n, w in zip(DF, wd)))
print("house weights h0:", " ".join(f"{n} {w:+.2f}" for n, w in zip(HF, wh)))
# anchor gain in v7m labelled same-ID subset (0h)
D = R[(R.h == 0) & R.v7m_lab_same & R.pred0.notna() & R.A.notna()].copy()
if len(D):
    G = D[D.v7m_ok == 1]; xg = G.A - G.pred0; yg = G.y0 - G.pred0; beta = float((xg * yg).sum() / (xg * xg).sum())
    rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
    for name, Q in (("all", D), ("|A-pred|>.3", D[(D.A - D.pred0).abs() > .3]), ("P>=.8", D[D.P_v7m >= .8]), ("P<.8", D[D.P_v7m < .8])):
        if len(Q) == 0: print("  ", name, "none"); continue
        E0 = Q.pred0 + Q.P_v7m * beta * (Q.A - Q.pred0)
        print(f"  0h [{name}] n {len(Q)} acc {Q.v7m_ok.mean():.2f}: model {rm(Q.y0-Q.pred0):.3f} anchor {rm(Q.y0-Q.A):.3f} E0 {rm(Q.y0-E0):.3f} anchor-if-P>=.8 {rm(Q.y0-np.where(Q.P_v7m>=.8,Q.A,Q.pred0)):.3f}  (beta {beta:.2f})")

# ---------- evaluation ----------
E = pd.read_csv(os.path.join(H_, "local", "linker_v7", "eval60_predecessor_v7.csv"))
S19 = pd.read_csv(os.path.join(env.ROOT, u"제출", u"12회차_2026-10-11(나)", "submission_19.csv"))
S19["rec"] = S19.row_id.str[:7]; S19["hh"] = S19.row_id.str[8:].astype(int); P19 = S19.pivot(index="rec", columns="hh", values="sub_ec")
W = {h: fit_models(REC, h) for h in HS}
ev = []; detail = []
for r_ in E.itertuples():
    t = idx[r_.row]; vis, lab, _ = world_eval(t); dates, nL0 = build(t, vis, lab)
    c = idx[r_.prev_h0]; A = EC[c, 23] if TRAIN[c] else np.nan; p0 = P19.at[r_.row, 0]
    d = dict(rec=r_.row, cand=r_.prev_h0, p_v7=r_.p_h0, cand_lab=bool(TRAIN[c]), A=A, pred0=p0, dis=abs(A - p0) if np.isfinite(A) else np.nan, nL0=nL0)
    for h in HS:
        post, pdv, ph = posterior(dates, *W[h], h)
        d[f"P{h}"] = post.get(c, 0.0); top = max(post, key=post.get)
        d[f"top{h}"] = rid(top); d[f"Ptop{h}"] = post[top]; d[f"Phid{h}"] = pdv[-1] + sum(ph.values())
        if h == 0:
            o = sorted(post, key=post.get, reverse=True)[:6]
            detail.append((r_.row, [(rid(j), round(post[j], 3), "L" if lab[j] else "U", round(EC[j, 23], 3) if lab[j] else round(P19.at[rid(j), 23], 3) if rid(j) in P19.index else None) for j in o],
                           round(pdv[-1], 3), round(sum(ph.values()), 3)))
    ev.append(d)
EV = pd.DataFrame(ev); EV.to_csv(os.path.join(OUT, "vf3b_eval_v2.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 260)
print("\nEVAL (candidate = linker v7 h0 choice):")
print(EV[["rec", "cand", "p_v7", "cand_lab", "A", "pred0", "dis", "nL0", "P0", "P6", "P12", "P23", "top0", "Ptop0", "top23", "Ptop23", "Phid0"]].round(3).to_string())
print("\nEVAL top-6 previous-day records at h0 (rec, P, L=labelled/U=unlabelled, EC23 label or submission_19 23h pred) | P(date hidden) | P(hidden house):")
for rr, lst, a1, a2 in detail: print(" ", rr, lst, a1, a2)
