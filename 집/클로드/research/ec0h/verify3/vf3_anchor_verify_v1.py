# -*- coding: utf-8 -*-
"""VERIFY3 (2026-10-12 집 클로드). Plan: ec0h/verify3/plan_v3_anchor_verify.md (fixed before running).
Is the linker-v7 labelled same-ID top candidate c really the previous date of evaluation record t, in records where the anchor
(EC[c,23]) and the current prediction (submission_19, 0 h) disagree a lot?  Evidence: date map, midnight continuity, operation
fingerprint (hour-progressive), disagreement size; calibrated on the training MIRROR CV world (cv400_v7m_critic.py blocks).
Rule: t uses only its 0..h inputs; visible = training records (any farm / date, minus CV-hidden) + same-ID earlier records
(unlabelled).  Other-ID evaluation inputs and later same-ID evaluation inputs are never used.  Independent per-record decisions.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
import feat6 as F6, feat4 as F4
from sklearn.linear_model import LogisticRegression
OUT = os.path.dirname(os.path.abspath(__file__))
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLa[TLa.cert]
truth = {int(b): int(a) for a, b in zip(TLc.a1, TLc.b)}
links = [(int(a), int(b)) for a, b in zip(TLc.a1, TLc.b) if TRAIN[int(a)] and TRAIN[int(b)]]
Xf = X.reshape(N, -1)
def same_day(i, js):   # >=90% equal outdoor cells
    return (np.abs(Xf[js] - Xf[i]) < 1e-6).mean(1) >= .9
HS = [0, 6, 12, 23]

def world_cv(r):
    f = FARM[r]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo = NUM[FARM == f].min(); s = lo + ((NUM[r] - lo) // 10) * 10; m = s + sh
    rem = (FARM == f) & ((NUM == s - 1) | (NUM == s + 10))
    oth = (FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)
    blk = (FARM == f) & (NUM >= s) & (NUM < s + 10)
    hidden = rem | oth | blk
    lab = TRAIN & ~hidden
    vis = lab | ((FARM == f) & (NUM < NUM[r]) & ~rem & ~oth)
    vis[r] = False
    return vis, lab, f"{f}_{s}"
def world_eval(t):
    lab = TRAIN.copy()
    vis = lab | ((FARM == FARM[t]) & (NUM < NUM[t])); vis[t] = False
    return vis, lab, "eval"

def mid(c, t):
    return float(np.sum(np.minimum(np.nan_to_num(np.abs(VARS[c, 23] - VARS[t, 0]) / sdV, nan=2.0), 6)))

_succ = {}
def pred_of(c, lab):
    k = lab.tobytes()
    if k not in _succ: _succ[k] = {b: a for a, b in F4.label_successor(lab).items()}
    return _succ[k].get(c)

def evidence(t, c, vis, lab):
    vi = np.where(vis)[0]
    S0 = set(vi[np.all(np.abs(X[vi, 0] - X[t, 0]) < 1e-6, axis=1)].tolist())
    lk = [(a, b) for a, b in links if lab[a] and lab[b] and b in S0 and a not in S0]
    D1 = []; date_na = 1
    if lk:
        A_ = sorted({a for a, _ in lk}); grp = []
        for a in A_:
            for G in grp:
                if same_day(a, np.array([G[0]]))[0]: G.append(a); break
            else: grp.append([a])
        cnt = [sum(1 for a, _ in lk if a in G) for G in grp]; G = grp[int(np.argmax(cnt))]
        D1 = [j for j in vi if same_day(j, np.array(G)).any() and j not in S0]
        date_na = 0
    date_ok = float(c in D1)
    comp = [j for j in D1 if j != c]
    mc = mid(c, t); mcomp = [mid(j, t) for j in comp]
    row = dict(date_na=date_na, date_ok=date_ok, nD1=len(D1), nD1_lab=int(sum(lab[j] for j in D1)), ncomp=len(comp),
               mid_c=mc, mid_margin=(min(mcomp) - mc) if comp else 0.0, mid_best=float(not comp or mc <= min(mcomp)),
               comp_na=float(not comp))
    for h in HS:
        fc = F6.fp_dist(t, c, h); fcomp = [F6.fp_dist(t, j, h) for j in comp]
        fcomp = [x for x in fcomp if np.isfinite(x)]
        row[f"fp{h}"] = fc
        row[f"fpm{h}"] = (min(fcomp) - fc) if (fcomp and np.isfinite(fc)) else 0.0
        row[f"fpbest{h}"] = float((not fcomp) or (np.isfinite(fc) and fc <= min(fcomp)))
    # (e) two days back: predecessor c' of c from TRAINING-label EC continuity; rank of fp(t,c') among visible same-date records
    cp = pred_of(c, lab)
    for h in HS:
        if cp is None: row[f"fp2best{h}"] = np.nan; continue
        dd = [j for j in vi if same_day(cp, np.array([j]))[0] and j not in S0 and j not in D1]
        fx = {j: F6.fp_dist(t, j, h) for j in dd + [cp]}
        row[f"fp2best{h}"] = float(all(not np.isfinite(fx[j]) or fx[cp] <= fx[j] for j in dd))
    row["c2"] = rid(cp) if cp is not None else ""
    return row, D1

# ---------------- training mirror-CV rows ----------------
C = pd.read_csv(os.path.join(H_, "local", "linker_v7", "cv_all400_choice_v7m.csv"))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[(O.tag == "CUR") & (O.validator == "DIAG10") & (O.hour == 0)]; P0cv = O.groupby(O.row_id.str[:7]).sg.mean()
cv = []
for r_ in C.itertuples():
    t = idx[r_.rec]; c = idx[r_.prev]
    if not (r_.prev_is_labelled and not r_.xf) or t not in truth: continue
    vis, lab, blk = world_cv(t)
    if not lab[c]: continue
    ev, D1 = evidence(t, c, vis, lab)
    tr = truth[t]
    cv.append(dict(rec=r_.rec, cand=r_.prev, blk=blk, p=r_.p, truth=rid(tr), ok=int(tr == c), truth_vis=int(vis[tr]), truth_inD1=int(tr in D1),
                   truth_lab=int(lab[tr]), A=EC[c, 23], y0=EC[t, 0], pred0=P0cv.get(r_.rec, np.nan),
                   dTair=IN_[t, 0, 0] - IN_[c, 23, 0], **ev))
CV = pd.DataFrame(cv); CV["dis"] = (CV.A - CV.pred0).abs()
CV.to_csv(os.path.join(OUT, "vf3_cv_rows_v1.csv"), index=False, encoding="utf-8-sig")
print("CV rows (labelled same-ID top choice, truth certain):", len(CV), "| acc %.3f | truth visible when wrong %.2f | truth in D1 when wrong %.2f" % (
    CV.ok.mean(), CV[CV.ok == 0].truth_vis.mean(), CV[CV.ok == 0].truth_inD1.mean()))
print("pred0 available:", int(CV.pred0.notna().sum()))

def Xmat(D, h):
    lp = np.log(np.clip(D.p, 1e-3, 1 - 1e-3) / (1 - np.clip(D.p, 1e-3, 1 - 1e-3)))
    return np.c_[lp, D.date_ok, D.date_na, D.mid_margin.clip(-15, 15), D[f"fpm{h}"].clip(-3, 3), D.dis.clip(0, 1.5)]
FN = ["logit_p", "date_ok", "date_na", "mid_margin", "fp_margin_h", "dis"]
CVd = CV.dropna(subset=["pred0"]).reset_index(drop=True)
mods = {}
for h in HS:
    Xc = Xmat(CVd, h); y = CVd.ok.values; mu, sd = Xc.mean(0), Xc.std(0) + 1e-9; Z = (Xc - mu) / sd
    oof = np.zeros(len(y))
    for b in CVd.blk.unique():
        tr_ = CVd.blk.values != b
        if len(set(y[tr_])) < 2: oof[~tr_] = y[tr_].mean(); continue
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Z[tr_], y[tr_]); oof[~tr_] = m.predict_proba(Z[~tr_])[:, 1]
    CVd[f"P{h}"] = oof
    base = y.mean(); bri = np.mean((oof - y) ** 2); bri0 = np.mean((base - y) ** 2)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y); mods[h] = (m, mu, sd)
    hi = oof >= .8; lo = oof <= .3
    print(f"h={h:2d}: OOF Brier {bri:.3f} vs const {bri0:.3f} | P>=.8 n {hi.sum()} acc {y[hi].mean() if hi.any() else np.nan:.2f} | "
          f"P<=.3 n {lo.sum()} acc {y[lo].mean() if lo.any() else np.nan:.2f} | coef " + " ".join(f"{n} {w:+.2f}" for n, w in zip(FN, m.coef_[0])))
    print("   calibration bins:", " ".join(f"[{a:.1f},{b:.1f}) n{int(((oof>=a)&(oof<b)).sum())} acc{(y[(oof>=a)&(oof<b)].mean() if ((oof>=a)&(oof<b)).any() else np.nan):.2f}"
                                       for a, b in [(0, .3), (.3, .5), (.5, .7), (.7, .8), (.8, .9), (.9, 1.01)]))
CVd.to_csv(os.path.join(OUT, "vf3_cv_rows_oof_v1.csv"), index=False, encoding="utf-8-sig")
# simple evidence table
CVd["dis3"] = (CVd.dis > .3).astype(int)
print("\nevidence-combination table (h=0): date_ok, mid_best, fpbest0, dis>.3 -> n, acc")
print(CVd.groupby(["date_ok", "mid_best", "fpbest0", "dis3"]).ok.agg(["size", "mean"]).round(2).to_string())
print("\nby dis>.3 alone:", CVd.groupby("dis3").ok.agg(["size", "mean"]).round(2).to_dict())
for h in HS: print(f"fpbest{h}: ", CVd.groupby(f"fpbest{h}").ok.agg(["size", "mean"]).round(2).to_dict())
print("fp2best0:", CVd.groupby("fp2best0").ok.agg(["size", "mean"]).round(2).to_dict(), "| fp2best23:", CVd.groupby("fp2best23").ok.agg(["size", "mean"]).round(2).to_dict())
print("dT_air<-3:", CVd.assign(dt=(CVd.dTair < -3).astype(int)).groupby("dt").ok.agg(["size", "mean"]).round(2).to_dict())
# beta and 0h comparison
G = CVd[CVd.ok == 1]; xg = G.A - G.pred0; yg = G.y0 - G.pred0; beta = float((xg * yg).sum() / (xg * xg).sum())
print(f"\nbeta (correct links, y0-pred0 ~ A-pred0): {beta:.3f} (n {len(G)})")
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
for name, D in (("all", CVd), ("dis>.3", CVd[CVd.dis > .3]), ("p<.8", CVd[CVd.p < .8]), ("p<.8&dis>.3", CVd[(CVd.p < .8) & (CVd.dis > .3)])):
    if len(D) == 0: continue
    E0 = D.pred0 + D.P0 * beta * (D.A - D.pred0)
    print(f"0h RMSE [{name}] n {len(D)} acc {D.ok.mean():.2f}: model {rm(D.y0-D.pred0):.3f} | anchor {rm(D.y0-D.A):.3f} | E0(OOF P) {rm(D.y0-E0):.3f} | "
          f"anchor-if-P0>=.8 {rm(D.y0-np.where(D.P0>=.8, D.A, D.pred0)):.3f}")
print("\nCV wrong rows with dis>.3:"); print(CVd[(CVd.ok == 0) & (CVd.dis > .3)][["rec", "cand", "truth", "p", "truth_vis", "date_ok", "mid_best", "fpbest0", "fpbest23", "A", "y0", "pred0", "P0", "P23"]].round(3).to_string())
print("\nCV correct rows with dis>.3:"); print(CVd[(CVd.ok == 1) & (CVd.dis > .3)][["rec", "cand", "p", "date_ok", "mid_best", "fpbest0", "fpbest23", "A", "y0", "pred0", "P0", "P23"]].round(3).to_string())

# ---------------- evaluation records ----------------
E = pd.read_csv(os.path.join(H_, "local", "linker_v7", "eval60_predecessor_v7.csv"))
S19 = pd.read_csv(os.path.join(env.ROOT, u"제출", u"12회차_2026-10-11(나)", "submission_19.csv"))
S19["rec"] = S19.row_id.str[:7]; S19["h"] = S19.row_id.str[8:].astype(int)
P19 = S19.pivot(index="rec", columns="h", values="sub_ec")
ev_rows = []
for r_ in E.itertuples():
    if not (r_.prev_is_labelled and not r_.xf): continue
    t = idx[r_.row]; c = idx[r_.prev_h0]
    vis, lab, _ = world_eval(t)
    ev, D1 = evidence(t, c, vis, lab)
    d = dict(rec=r_.row, cand=r_.prev_h0, p=r_.p_h0, A=EC[c, 23], pred0=P19.at[r_.row, 0], dTair=IN_[t, 0, 0] - IN_[c, 23, 0],
             D1=" ".join(rid(j) + ("*" if lab[j] else "") for j in D1), **ev)
    d["dis"] = abs(d["A"] - d["pred0"])
    D_ = pd.DataFrame([d])
    for h in HS:
        m, mu, sd = mods[h]; d[f"P{h}"] = float(m.predict_proba((Xmat(D_, h) - mu) / sd)[:, 1][0])
    d["E0"] = d["pred0"] + d["P0"] * beta * (d["A"] - d["pred0"])
    d["focus"] = int(d["dis"] > .3)
    ev_rows.append(d)
EV = pd.DataFrame(ev_rows); EV.to_csv(os.path.join(OUT, "vf3_eval_rows_v1.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
cols = ["rec", "cand", "p", "A", "pred0", "dis", "dTair", "date_na", "date_ok", "nD1", "nD1_lab", "mid_c", "mid_margin", "mid_best", "fp0", "fpm0", "fpbest0", "fpbest6", "fpbest12", "fpbest23", "fp2best0", "fp2best23", "P0", "P6", "P12", "P23", "E0"]
print("\nEVAL focus (|A - pred0| > .3):"); print(EV[EV.focus == 1][cols].round(3).to_string())
print("\nEVAL focus D1 lists:"); [print(" ", r.rec, "->", r.D1, "| c2", r.c2) for r in EV[EV.focus == 1].itertuples()]
print("\nEVAL all labelled same-ID candidates:"); print(EV[cols].round(3).to_string())
