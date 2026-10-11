# -*- coding: utf-8 -*-
"""agentY_econset y4: (a) 하루 EC 변화(dEC=당일-전날, 같은 동)를 전날/당일0~6시 입력이 알려 주는가
(b) onset/ramp/high 분류 AUC (사슬 묶음 GroupKFold, 로지스틱·얕은 나무, 폴드 내 특징 선택)
(c) 고EC 날 모델 잔차 vs 당일 0~6시 입력 (예측 통제 편상관, 최대통계 순열)
(d) 오라클 EC RMSE 이득: 표지된 날에 학습 폴드의 고EC 하루 평균 잔차를 더함.
연결 = truth a1, c1<=1.0, a1 중복 제거 (EC+배지온도 연속; 진단용 오라클)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from y_common import *
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(1)
OUTD = os.path.dirname(os.path.abspath(__file__))
prev, xf = links(1.0)
D = build(prev, xf)
P = load_oof()
dP = P.groupby("rid").agg(pred=("sg", "mean"), truth=("y", "mean")).reset_index()
dP["pred06"] = P[P.h <= 6].groupby("rid").sg.mean().reindex(dP.rid).values
D = D.reset_index().merge(dP, on="rid", how="left").set_index("i")
D["res"] = D.truth - D.pred; D["dEC"] = D.ec - D.ec_p1
FEAT = [c for c in D.columns if c.startswith(("o_", "p1_", "dp_", "do_"))]
FEAT = [c for c in FEAT if D[c].notna().mean() > 0.3 and D[c].nunique() > 2]
OWN = [c for c in FEAT if c.startswith("o_")]; PRV = [c for c in FEAT if not c.startswith("o_")]
print("records", len(D), "linked", D.p1.notna().sum(), "chains", D.grp.nunique(), "features own/prev", len(OWN), len(PRV))

def rk(v):
    v = np.where(np.isnan(v), np.nanmedian(v), v); return rankdata(v)
def partial_rank(y, x, ctrl):
    """rank partial correlation of y,x controlling ctrl (list of arrays)"""
    C = np.column_stack([np.ones(len(y))] + [rk(c) for c in ctrl])
    ry = rk(y) - C @ np.linalg.lstsq(C, rk(y), rcond=None)[0]
    rx = rk(x) - C @ np.linalg.lstsq(C, rk(x), rcond=None)[0]
    return np.corrcoef(ry, rx)[0, 1], ry, rx

def corr_scan(S, ycol, feats, ctrl, nperm=1000, title=""):
    y = S[ycol].values; ctr = [S[c].values for c in ctrl]
    out = []; RX = []
    for c in feats:
        x = S[c].values
        if np.isnan(x).mean() > 0.5 or np.nanstd(x) == 0: continue
        rho, ry, rx = partial_rank(y, x, ctr); out.append((c, rho)); RX.append(rx)
    RX = np.column_stack(RX); RX = (RX - RX.mean(0)) / (RX.std(0) + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    mx = np.array([np.max(np.abs(rng.permutation(ry) @ RX / len(ry))) for _ in range(nperm)])
    t = pd.DataFrame(out, columns=["feat", "prho"]); t["abs"] = t.prho.abs()
    t["p_fw"] = [(mx >= a).mean() for a in t["abs"]]
    t = t.sort_values("abs", ascending=False)
    print(f"\n--- {title}: n={len(S)}, feats {len(t)}, control={ctrl}; perm 95% max|rho| = {np.quantile(mx, .95):.3f}")
    print(t.head(12).round(3).to_string(index=False))
    return t

# (a) dEC along linked days
S = D[D.p1.notna() & D.dEC.notna()].copy()
print("\n(a) dEC distribution q05/50/95:", np.quantile(S.dEC, [.05, .5, .95]).round(3), " sd", S.dEC.std().round(3))
ta = corr_scan(S, "dEC", FEAT, ["ec_p1"], title="(a) dEC vs prev/own-early features (control ec_p1)")
ta.to_csv(os.path.join(OUTD, "y4a_dEC_corr.csv"), index=False)

def gkf_splits(groups, k=5, seed=0):
    ug = np.unique(groups); rs = np.random.default_rng(seed); rs.shuffle(ug)
    fold = {g: n % k for n, g in enumerate(ug)}; f = np.array([fold[g] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]

def prep(Xtr, Xte):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, (Xte - mu) / sd

# R2 of dEC by ridge (grouped CV)
Xa = S[FEAT].values; ya = S.dEC.values; g = S.grp.values
for nm_, cols in [("own", OWN), ("prev", PRV), ("all", FEAT)]:
    idx = [FEAT.index(c) for c in cols]; r2 = []
    for rep in range(5):
        pr = np.zeros(len(ya))
        for a_, b_ in gkf_splits(g, 5, rep):
            Xt, Xv = prep(Xa[a_][:, idx], Xa[b_][:, idx])
            pr[b_] = Ridge(alpha=50).fit(Xt, ya[a_]).predict(Xv)
        r2.append(1 - np.sum((ya - pr) ** 2) / np.sum((ya - ya.mean()) ** 2))
    print(f"(a) grouped-CV Ridge R2 for dEC [{nm_}]: mean {np.mean(r2):.3f} (reps {np.round(r2, 3)})")

# (b) classification
def cv_auc(Sx, lab, cols, model, reps=5, k=5, topk=10):
    Xm = Sx[cols].values; y = lab.astype(int); g = Sx.grp.values; aucs = []; P0 = None
    for rep in range(reps):
        pr = np.full(len(y), np.nan)
        for a_, b_ in gkf_splits(g, k, rep):
            if y[a_].sum() < 2: pr[b_] = y[a_].mean(); continue
            Xt, Xv = prep(Xm[a_], Xm[b_])
            if Xt.shape[1] > topk:   # in-fold univariate selection
                sc = [abs(roc_auc_score(y[a_], Xt[:, j]) - .5) if np.std(Xt[:, j]) > 0 else 0 for j in range(Xt.shape[1])]
                sel = np.argsort(sc)[::-1][:topk]; Xt, Xv = Xt[:, sel], Xv[:, sel]
            m = (LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000) if model == "logit"
                 else DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, class_weight="balanced", random_state=rep))
            pr[b_] = m.fit(Xt, y[a_]).predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(y, pr))
        if rep == 0: P0 = pr
    return np.mean(aucs), np.min(aucs), np.max(aucs), P0

L = D[D.p1.notna()].copy()
TARGETS = {
    "ONS10 (ec>=1 & prev<1  vs  ec<1 & prev<1)": (L[L.ec_p1 < 1.0], lambda s: s.ec >= 1.0),
    "ONS08 (ec>=.8 & prev<.8 vs ec<.8 & prev<.8)": (L[L.ec_p1 < 0.8], lambda s: s.ec >= 0.8),
    "RAMP (dEC>=+.08 & ec>=.7  vs other linked)": (L, lambda s: (s.dEC >= 0.08) & (s.ec >= 0.7)),
}
CLS = {}
for tname, (Sx, fn) in TARGETS.items():
    lab = fn(Sx).values
    print(f"\n(b) {tname}: positives {lab.sum()} / {len(lab)} ; chains with positives {Sx[lab].grp.nunique()}")
    if lab.sum() < 3: print("   too few positives"); continue
    print("   positives:", " ".join(f"{a}({b:.2f}<-{c:.2f})" for a, b, c in zip(Sx[lab].rid, Sx[lab].ec, Sx[lab].ec_p1)))
    for fs, cols in [("own0-6", OWN), ("prevdays", PRV), ("all", FEAT), ("ec_p1 only (oracle anchor)", ["ec_p1"])]:
        for model in ["logit", "tree"]:
            m, lo, hi, p0 = cv_auc(Sx, lab, cols, model)
            print(f"   {fs:28s} {model:5s} AUC {m:.3f} [{lo:.3f},{hi:.3f}]")
            CLS[(tname, fs, model)] = (Sx.index, p0)
    sp = Sx.pred06.notna().values
    if lab[sp].sum() >= 2 and (~lab[sp]).sum() > 2:
        print(f"   (model OOF, no CV needed) pred06 AUC {roc_auc_score(lab[sp], Sx.pred06[sp]):.3f}  pred(day) {roc_auc_score(lab[sp], Sx.pred[sp]):.3f}  n_pos {lab[sp].sum()}")

# HIGH detection without anchor: all days with OOF; does own 0-6 add to pred06?
H = D[D.pred06.notna()].copy(); labH = (H.ec >= 1.0).values
print(f"\n(b') HIGH (ec>=1) among all OOF days: {labH.sum()}/{len(H)}")
for fs, cols in [("pred06", ["pred06"]), ("own0-6", OWN), ("pred06+own0-6", ["pred06"] + OWN)]:
    m, lo, hi, p0 = cv_auc(H, labH, cols, "logit", topk=11)
    print(f"   {fs:16s} logit AUC {m:.3f} [{lo:.3f},{hi:.3f}]")
    CLS[("HIGH", fs, "logit")] = (H.index, p0)

# (c) residual on high days vs own-early features controlling pred
for title, Sx in [("high days ec>=1.0 (OOF)", D[(D.ec >= 1.0) & D.pred.notna()]),
                  ("deployable: pred06>=0.8 days", D[(D.pred06 >= 0.8)])]:
    tc = corr_scan(Sx.copy(), "res", OWN, ["pred"], title=f"(c) day residual vs own 0-6h, {title}")
    tc.to_csv(os.path.join(OUTD, f"y4c_res_corr_{'high' if 'high' in title else 'p06'}.csv"), index=False)
    # grouped-CV ridge on own 0-6 + pred -> residual
    Xm = Sx[OWN + ["pred"]].values; yr = Sx.res.values; gg = Sx.grp.values; r2 = []
    for rep in range(5):
        pr = np.zeros(len(yr))
        for a_, b_ in gkf_splits(gg, 5, rep):
            Xt, Xv = prep(Xm[a_], Xm[b_]); pr[b_] = Ridge(alpha=50).fit(Xt, yr[a_]).predict(Xv)
        r2.append(1 - np.sum((yr - pr) ** 2) / np.sum((yr - yr.mean()) ** 2))
    print(f"   grouped-CV Ridge(own0-6+pred) R2 for residual: {np.mean(r2):.3f} {np.round(r2, 3)}  (mean res {yr.mean():.3f})")

# (d) oracle RMSE gain
P2 = P.merge(D[["rid", "grp", "ec"]].reset_index(), on="rid", how="left")
base = np.sqrt(np.mean((P2.y - P2.sg) ** 2))
dd = D[D.pred.notna()]
hiset = dd[dd.ec >= 1.0]
def delta_for(grp):  # leave-chain-out mean high-day residual
    s = hiset[hiset.grp != grp].res; return s.mean()
def rmse_with(flag_rids, soft=None):
    adj = P2.rid.map(lambda x: 0.0).values.astype(float)
    m = P2.rid.isin(flag_rids).values
    dl = P2.grp.map(lambda gq: delta_for(gq)).values
    if soft is None: adj[m] = dl[m]
    else: w = P2.rid.map(soft).fillna(0).values; adj = w * dl
    return np.sqrt(np.mean((P2.y - (P2.sg + adj)) ** 2))
print(f"\n(d) baseline DIAG10 CUR hourly RMSE {base:.5f} (360 days); mean high-day residual {hiset.res.mean():.3f} (n {len(hiset)})")
ons = L[(L.ec >= 1.0) & (L.ec_p1 < 1.0)].rid
for nm_, fl in [("perfect onset ONS10 days", ons), ("perfect all high days (ref upper bound)", hiset.rid),
                ("perfect high days without linked prev", dd[(dd.ec >= 1.0) & dd.p1.isna()].rid)]:
    v = rmse_with(set(fl)); print(f"   {nm_:42s} n={len(set(fl) & set(dd.rid)):3d}  RMSE {v:.5f} ({(v / base - 1) * 100:+.2f}%)")
for key, (idx, p0) in CLS.items():
    if key[2] != "logit" or key[1] not in ("all", "own0-6", "pred06+own0-6", "pred06"): continue
    rids = D.loc[idx, "rid"].values
    for thr in [0.5, 0.8]:
        fl = set(rids[p0 >= thr]); v = rmse_with(fl)
        print(f"   flagged by {key[0][:6]}|{key[1]:14s} p>={thr}: n={len(fl & set(dd.rid)):3d} RMSE {v:.5f} ({(v / base - 1) * 100:+.2f}%)")
D.to_pickle(os.path.join(OUTD, "y4_D.pkl"))
