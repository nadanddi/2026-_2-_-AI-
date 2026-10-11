# -*- coding: utf-8 -*-
"""agentY_econset y1: 고EC 에피소드 시작(onset)의 전조 신호 탐색 (학습 자료만, 진단).
- 같은 동 연결 = linker_v3 truth.pkl 의 cert 연결 (b 의 이전 날짜 = a1). 정답 EC 연속성으로 만든 연결이므로 '진단용 오라클'.
- 정의 3종: H10 일평균>=1.0, H12 일평균>=1.2, HREL 일평균>=max(0.7, 1.5*직전 최대5 사슬날 중앙값).
- onset = 고EC 날 & 이전 날 존재 & 이전 날 고EC 아님.  control = 보통 날 & 이전 날 보통.
- 특징: 당일 0~6시(o_), 전날 하루 요약(p1_), 그 전날(p2_), 전날-그전날 변화(dp_), 당일 0~6시-전날 0~6시(do_).
"""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env, os, pickle, numpy as np, pandas as pd, warnings
from scipy.stats import mannwhitneyu, spearmanr, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
OUTD = os.path.dirname(os.path.abspath(__file__))
R = pickle.load(open(r"C:\work\farmai\집\클로드\research\local\linker_v3\base.pkl", "rb"))
r = R["recs"].reset_index(drop=True)
r["rid"] = r.farm + "_" + r.day.map(lambda d: f"{d:03d}")
T = pd.read_pickle(r"C:\work\farmai\집\클로드\research\local\linker_v3\truth.pkl")
T = T[T.cert].copy()
prev = dict(zip(T.b, T.a1))          # record index -> previous-date record index (same house)
xfarm = dict(zip(T.b, T.fa != T.fb))
nxt = {a: b for b, a in prev.items()}

X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
X["rid"] = X.row_id.str[:7]; X["h"] = X.row_id.str[-2:].astype(int)
Y["rid"] = Y.row_id.str[:7]
CH = ["in_temp", "in_hum", "in_co2", "out_temp", "out_hum", "out_rad", "out_wspd",
      "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
ACT = [c for c in CH if c.startswith("act_")]
rid2i = dict(zip(r.rid, r.index))
tr = r.index[r.split == "train"].values
A = np.full((len(r), 24, len(CH)), np.nan)
Xs = X.sort_values(["rid", "h"])
for rid, g in Xs.groupby("rid"):
    if rid in rid2i and len(g) == 24: A[rid2i[rid]] = g[CH].values
ecm = Y.groupby("rid").sub_ec.mean()
r["ec"] = ecm.reindex(r.rid).values
nm = lambda a, axis=None: np.nanmean(a, axis=axis)

def dayfeat(i):
    f = {}
    if i is None or np.all(np.isnan(A[i])): return f
    for k, c in enumerate(CH):
        v = A[i, :, k]
        if np.all(np.isnan(v)): continue
        night = np.r_[v[0:6], v[20:24]]; day = v[9:17]
        f[c + "_mean"] = nm(v); f[c + "_rng"] = np.nanmax(v) - np.nanmin(v)
        f[c + "_night"] = nm(night); f[c + "_day"] = nm(day); f[c + "_dn"] = nm(day) - nm(night)
        f[c + "_eve"] = nm(v[18:24]); f[c + "_e06"] = nm(v[0:7])
        if c in ACT:
            on = v > 0
            f[c + "_duty"] = np.nanmean(on); f[c + "_nchg"] = np.nansum(np.abs(np.diff(v)) > 0)
            f[c + "_first"] = int(np.argmax(on)) if on.any() else 24
            f[c + "_last"] = int(23 - np.argmax(on[::-1])) if on.any() else -1
            f[c + "_full"] = np.nanmean(v >= 99)
    return f

def early(i):
    f = {}
    for k, c in enumerate(CH):
        v = A[i, 0:7, k]
        if np.all(np.isnan(v)): continue
        f[c + "_m06"] = nm(v); f[c + "_d06"] = v[6] - v[0] if not np.isnan(v[6] - v[0]) else np.nan
        f[c + "_x06"] = np.nanmax(v) - np.nanmin(v)
        if c in ACT: f[c + "_on06"] = np.nanmean(v > 0)
    return f

rows = []
for i in tr:
    d = {"i": i, "rid": r.rid[i], "farm": r.farm[i], "num": r.day[i], "ec": r.ec[i]}
    p1 = prev.get(i); p2 = prev.get(p1) if p1 is not None else None
    d["p1"] = p1; d["p2"] = p2; d["xf"] = xfarm.get(i, False)
    d["ec_p1"] = r.ec[p1] if p1 is not None else np.nan
    for k, v in early(i).items(): d["o_" + k] = v
    F1 = dayfeat(p1); F2 = dayfeat(p2)
    for k, v in F1.items(): d["p1_" + k] = v
    for k, v in F2.items(): d["p2_" + k] = v
    for k in F1:
        if k in F2: d["dp_" + k] = F1[k] - F2[k]
    if p1 is not None:
        e1 = early(p1); e0 = early(i)
        for k in e0:
            if k.endswith("_m06") and k in e1: d["do_" + k] = e0[k] - e1[k]
    rows.append(d)
D = pd.DataFrame(rows).set_index("i")
# chain groups
par = {i: i for i in tr}
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for b, a in prev.items(): par[fd(b)] = fd(a)
D["grp"] = [fd(i) for i in D.index]
# chain-relative baseline for HREL
def prevmed(i, k=5):
    v = []; p = prev.get(i)
    while p is not None and len(v) < k: v.append(r.ec[p]); p = prev.get(p)
    return np.median(v) if v else np.nan
D["pmed"] = [prevmed(i) for i in D.index]
DEFS = {"H10": lambda s: s.ec >= 1.0, "H12": lambda s: s.ec >= 1.2,
        "HREL": lambda s: (s.ec >= 0.7) & (s.ec >= 1.5 * s.pmed)}
print("train recs", len(D), "cert prev links", D.p1.notna().sum(), "chains", D.grp.nunique(),
      "chain len max", D.grp.value_counts().max())
for nmz, fn in DEFS.items():
    hi = fn(D).fillna(False)
    D["hi_" + nmz] = hi
for nmz in DEFS:
    D["hp_" + nmz] = [D.at[int(p), "hi_" + nmz] if pd.notna(p) and int(p) in D.index else np.nan for p in D.p1]

OOF = pd.read_csv(r"C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv")
OOF = OOF[(OOF.tag == "CUR") & (OOF.validator == "DIAG10")]
P = OOF.groupby("row_id").agg(sg=("sg", "mean"), y=("sub_ec", "first")).reset_index()
P["rid"] = P.row_id.str[:7]; P["h"] = P.row_id.str[-2:].astype(int)
dP = P.groupby("rid").agg(pred=("sg", "mean"), truth=("y", "mean")).reset_index()
dP["pred06"] = P[P.h <= 6].groupby("rid").sg.mean().reindex(dP.rid).values
D = D.reset_index().merge(dP, on="rid", how="left").set_index("i")
D["res"] = D.truth - D.pred
print("OOF days", D.pred.notna().sum())

FEAT = [c for c in D.columns if c[:3] in ("o_c", "o_i", "o_o", "o_a") or c.startswith(("o_", "p1_", "p2_", "dp_", "do_"))]
FEAT = [c for c in FEAT if D[c].notna().mean() > 0.3 and D[c].nunique() > 2]
print("n features", len(FEAT))

def auc_feats(df, lab, feats):
    out = []
    y = lab.values.astype(int)
    for c in feats:
        v = df[c].values; ok = ~np.isnan(v)
        if ok.sum() < 10 or len(np.unique(y[ok])) < 2: continue
        a = roc_auc_score(y[ok], v[ok]); n1 = y[ok].sum()
        p = mannwhitneyu(v[ok][y[ok] == 1], v[ok][y[ok] == 0]).pvalue
        sd = np.nanstd(v[ok]); dd = (np.nanmean(v[ok][y[ok] == 1]) - np.nanmean(v[ok][y[ok] == 0])) / sd if sd > 0 else 0
        out.append((c, a, abs(a - .5), dd, p, n1, ok.sum() - n1,
                    np.nanmean(v[ok][y[ok] == 1]), np.nanmean(v[ok][y[ok] == 0])))
    return pd.DataFrame(out, columns=["feat", "auc", "dev", "d", "p", "n1", "n0", "m1", "m0"]).sort_values("dev", ascending=False)

def maxstat_perm(df, lab, feats, nperm=500, groups=None):
    """family-wise: permute onset/control labels (record level) -> max|AUC-.5|"""
    Z = df[feats].values; y = lab.values.astype(int); mx = []
    Zr = np.column_stack([rankdata(np.where(np.isnan(z), np.nanmedian(z), z)) for z in Z.T])
    n1 = y.sum(); n0 = len(y) - n1
    for _ in range(nperm):
        yp = rng.permutation(y)
        s = Zr[yp == 1].sum(0); a = (s - n1 * (n1 + 1) / 2) / (n1 * n0)
        mx.append(np.max(np.abs(a - .5)))
    return np.array(mx)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
RES = {}
for nmz in DEFS:
    hi = D["hi_" + nmz]; hp = D["hp_" + nmz]
    on = hi & (hp == False); ctl = (~hi) & (hp == False); cont = hi & (hp == True)
    nohi = hi & hp.isna()
    print(f"\n===== {nmz}: high {hi.sum()} | onset {on.sum()} | cont-high {cont.sum()} | high w/o cert prev {nohi.sum()} | control normal->normal {ctl.sum()}")
    print("   (참고) 보통 날 중 cert 이전 없음:", ((~hi) & hp.isna()).sum(), "/", (~hi).sum(),
          " 고EC 중:", nohi.sum(), "/", hi.sum())
    S = D[on | ctl].copy(); lab = on[on | ctl]
    print("   onset same-farm prev:", (~S[lab].xf.astype(bool)).sum(), "/", lab.sum(),
          " onset ec_p1 mean %.3f vs control %.3f" % (S[lab].ec_p1.mean(), S[~lab].ec_p1.mean()),
          " onset ec %.3f" % S[lab].ec.mean())
    tab = auc_feats(S, lab, FEAT)
    mx = maxstat_perm(S, lab, list(tab.feat), 500)
    tab["p_fw"] = [(mx >= dv).mean() for dv in tab.dev]
    print(tab.head(20).round(3).to_string(index=False))
    for pre in ["o_", "p1_", "p2_", "dp_", "do_"]:
        t2 = tab[tab.feat.str.startswith(pre)]
        if len(t2): print(f"   best {pre:4s}: {t2.feat.iloc[0]} AUC {t2.auc.iloc[0]:.3f} p_fw {t2.p_fw.iloc[0]:.3f} (n feats {len(t2)})")
    # model prediction as detector baseline (OOF days only)
    Sp = S[S.pred.notna()]
    if Sp[on.reindex(Sp.index)].shape[0] >= 2:
        lp = on.reindex(Sp.index).astype(int)
        print("   OOF subset onset %d control %d | AUC pred06 %.3f, pred(day) %.3f, ec_p1(oracle anchor) %.3f" % (
            lp.sum(), len(lp) - lp.sum(), roc_auc_score(lp, Sp.pred06), roc_auc_score(lp, Sp.pred),
            roc_auc_score(lp, Sp.ec_p1)))
        print("   onset days: res mean %.3f (truth %.3f pred %.3f); control res mean %.3f" % (
            Sp[lp == 1].res.mean(), Sp[lp == 1].truth.mean(), Sp[lp == 1].pred.mean(), Sp[lp == 0].res.mean()))
    RES[nmz] = (on, ctl, cont, tab)
    tab.to_csv(os.path.join(OUTD, f"y1_auc_{nmz}.csv"), index=False)
D.to_pickle(os.path.join(OUTD, "y1_D.pkl"))
import json; json.dump(FEAT, open(os.path.join(OUTD, "y1_feat.json"), "w"))
print("saved")
