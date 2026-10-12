# -*- coding: utf-8 -*-
"""SP2 (2026-10-12 집 클로드; 사용자: '2차처럼 진짜 전날이 보이는 검증을 만들어 다시 판정').  사전 고정 — 실행 전 커밋.
검증 세계 W2: 대상 t 와 같은 ID 번호 t±1 만 숨김(평가의 지워진 날 모사), 나머지 train 은 모두 보임(진짜 전날·같은 날 다른 동 포함).
  같은 ID 앞번호 기록은 입력도 보임.  덩어리(블록) = F13/F47 번호 //10 (가중치 학습·부트스트랩 단위).
2단계 모델(vf3b 와 같은 특징·조건부 로짓)을 W2 세계에서 진실 확실 연결(190)로 블록 밖 학습.
구조 예측 SP: q_h = sg_h + w_b · P_lab · (L − cm_h), L = 정답 후보 23시 EC 의 사후확률 가중평균, w_b 시각 블록별 다른 블록 최소제곱 [0,1.2].
HARD(같은 세계): 사후 1등 후보가 정답 있는 기록이고 P_top ≥ .8 이면 EA2 앵커(A = 그 23시 EC, w 1), 아니면 sg.
대상: DM11 달력 전날에 보이는 정답 기록 ≥1.  판정: SP 가 HARD 보다 3시드 모두 좋음 AND 블록 부트스트랩 P(worse) < .025 AND 최대 기록 몫 ≤ .5.
1·2차 따로, BASE(sg) 대비도 보고.
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


def world_w2(r):
    f = FARM[r]; hidden = (FARM == f) & (np.abs(NUM - NUM[r]) <= 1)
    lab = TRAIN & ~hidden; vis = lab | ((FARM == f) & (NUM < NUM[r]) & ~hidden); vis[r] = False
    return vis, lab, f"{f}_{NUM[r] // 10}"
REC = []
for b, a in truth.items():
    if not TRAIN[b]: continue
    vis, lab, blk = world_w2(b); dates, nL0 = build(b, vis, lab)
    ti = next((k for k, d in enumerate(dates) if a in d["G"]), None)
    REC.append(dict(b=b, a=a, blk=blk, dates=dates, ti=ti, vis_a=bool(vis[a]), nL0=nL0))
print("W2 CV records:", len(REC), "| truth date candidate %.2f | truth house visible %.2f" % (np.mean([r["ti"] is not None for r in REC]), np.mean([r["vis_a"] for r in REC])), flush=True)
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

OUT = os.path.dirname(os.path.abspath(__file__))
CAL = pd.read_csv(os.path.join(H_, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
Qs = {sd: G.groupby(["rec", "hour"]).sg.mean().unstack() for sd, G in O.groupby("seed")}; Y = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
blocks = sorted({world_w2(i)[2] for i in np.where(TRAIN)[0]})
W = {}
for blk in blocks:
    tr = [r for r in REC if r["blk"] != blk]
    for h in HS: W[(blk, h)] = fit_models(tr, h)
print("weights fitted for", len(blocks), "blocks", flush=True)
rows = []
for t in np.where(TRAIN)[0]:
    r = rid(t)
    if r not in Y.index or not np.isfinite(DAY.get(r, np.nan)): continue
    vis, lab, blk = world_w2(t)
    d0 = DAY[r]; prevlab = [j for j in np.where(vis & lab)[0] if DAY.get(rid(j)) == d0 - 1]
    if not prevlab: continue
    dates, _ = build(t, vis, lab); rec = dict(rec=r, blk=blk)
    for h in HS:
        post, _, _ = posterior(dates, *W[(blk, h)], h)
        pl = sum(p for j, p in post.items() if lab[j]); L = sum(p * EC[j, 23] for j, p in post.items() if lab[j]) / pl if pl > 0 else np.nan
        rec[f"Pl{h}"], rec[f"L{h}"] = pl, L
        if h == 0:
            top = max(post, key=post.get) if post else None
            rec["hard"] = float(EC[top, 23]) if (top is not None and lab[top] and post[top] >= .8) else np.nan
    rows.append(rec)
D = pd.DataFrame(rows).set_index("rec"); D.to_csv(os.path.join(OUT, "sp2_posteriors_v1.csv"))
hmap = lambda h: max(x for x in HS if x <= h)
print("eligible records (front info):", len(D), "| pass-2:", int((D.index.str[4:7].astype(int) >= 179).sum()), "| HARD anchors:", int(D.hard.notna().sum()), flush=True)
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261247)
recs = list(D.index); out = {}
for sd, Q in Qs.items():
    Q = Q.loc[recs]; cm = Q.cumsum(1) / np.arange(1, 25); Yr = Y.loc[recs]
    Xs = np.zeros((len(recs), 24))
    for h in range(24):
        hh = hmap(h); Xs[:, h] = D[f"Pl{hh}"].fillna(0).values * (D[f"L{hh}"].fillna(0).values - cm[h].values) * (D[f"Pl{hh}"].fillna(0).values > 0)
    SPv = Q.values.copy(); Qv = Q.values; bl = lambda h: 0 if h < 6 else (1 if h < 12 else (2 if h < 23 else 3))
    for blk in D.blk.unique():
        tr = (D.blk != blk).values; te = ~tr
        for b in range(4):
            hs = [h for h in range(24) if bl(h) == b]
            x = Xs[tr][:, hs].ravel(); y = (Yr.values[tr][:, hs] - Q.values[tr][:, hs]).ravel(); ok = np.isfinite(y)
            w = float(np.clip((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum(), 0, 1.2)) if (x[ok] ** 2).sum() > 0 else 0.0
            SPv[np.ix_(np.where(te)[0], hs)] = Qv[np.ix_(np.where(te)[0], hs)] + w * Xs[np.ix_(np.where(te)[0], hs)]
    HDv = Q.values.copy(); hm = D.hard.notna().values
    HDv[hm] = Q.values[hm] + (D.hard.values[hm][:, None] - cm.values[hm])
    out[sd] = dict(Q=Q, SP=pd.DataFrame(SPv, index=Q.index, columns=Q.columns), HD=pd.DataFrame(HDv, index=Q.index, columns=Q.columns), Y=Yr)
for comp in ("HD", "Q"):
    for part, cond in (("all", lambda s: True), ("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
        rr = [r for r in recs if cond(r)]
        if not rr: continue
        cells = [rm((o["SP"].loc[rr] - o["Y"].loc[rr]).values) / rm((o[comp].loc[rr] - o["Y"].loc[rr]).values) - 1 for o in out.values()]
        per = pd.Series(np.mean([(((o["SP"].loc[rr] - o["Y"].loc[rr]) ** 2).sum(1) - ((o[comp].loc[rr] - o["Y"].loc[rr]) ** 2).sum(1)).values for o in out.values()], 0), index=rr)
        cl = per.groupby(D.blk.reindex(rr).values).sum(); ii = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
        share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
        base_rm = np.mean([rm((o[comp].loc[rr] - o["Y"].loc[rr]).values) for o in out.values()])
        print(f"SP vs {'HARD' if comp == 'HD' else 'BASE'} {part:6s}: n {len(rr)} | {comp} RMSE {base_rm:.4f} | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | block P {pw:.4f} | top share {share:.2f}")
        if comp == "HD" and part == "all": print("   VERDICT SP2:", "PASS" if all(x < 0 for x in cells) and pw < .025 and share <= .5 else "FAIL")
