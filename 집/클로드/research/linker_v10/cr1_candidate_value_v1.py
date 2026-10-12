# -*- coding: utf-8 -*-
# CR1 (2026-10-12 집 클로드; 사용자: '전날이 아닌데도 도움 되는 후보는 보통 몇 번째, 그런 후보의 비중') 서술: v11 상위 5 정답 후보 각각을 온도 앵커(13회차 식, λ1)로 썼을 때 EV 7A 오차 변화.
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
exec(open("id1_input_only_linker_v1.py", encoding="utf-8").read().split("TRG = []")[0])
SUCC = {}
for a_, b_, c_ in zip(TLa.a1, TLa.b, TLa.cert):
    if c_: SUCC.setdefault(int(a_), []).append(int(b_))
F11 = ["oj"] + [f"v{j}" for j in range(8)] + ["fp", "excl", "cd1", "cdev", "cna"]
def date_of(i, lab):
    if lab[i] and np.isfinite(DAY[i]): return DAY[i]
    vi = np.where(lab)[0]; eq = np.all(np.abs(X[vi, 0] - X[i, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
    return pd.Series(ds).mode().iloc[0] if len(ds) else np.nan
def feats11(b, lab, removed):
    D = feats(b, lab, removed)
    if len(D) == 0: return D
    db = date_of(b, lab & (np.arange(N) != b)); a = D.a.values.astype(int); da = np.array([date_of(x, lab) for x in a])
    na = ~np.isfinite(da) | (not np.isfinite(db))
    D["cd1"] = np.where(na, 0.0, (da == db - 1).astype(float)); D["cdev"] = np.where(na, 0.0, np.minimum(np.abs(da - (db - 1)), 6) / 6); D["cna"] = na.astype(float)
    return D
TRG = []
for b, a in zip(TL7.b, TL7.a):
    lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
    df = feats11(b, lab, TEST_IDX)
    if len(df) == 0: continue
    y = np.where(df.a.values == a)[0]
    if len(y): TRG.append((b, a, df, y[0]))

H_ = os.path.join(env.ROOT, "집", "클로드", "research")

DD = pd.read_pickle(os.path.join(H_, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]; DD = DD.rename(columns={"sub_temp": "y"}).reset_index(drop=True)
FO = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy")); idx = {rid(i): i for i in range(N)}
G7 = {rec: G for rec, G in DD.groupby("rec")}; q18 = DD["v18_a_7A"].values; base = DD["v22_a_7A"].values; y = DD.y.values
rows = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        w11 = fit_clogit([(d[F11].values.astype(float), y_) for b_, a_, d, y_ in TRG if not touch[b_] and not touch[a_]], l2=0.5, iters=300)
        lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            r = rid(b)
            if not TRAIN[b] or r not in G7: continue
            D = feats11(b, lab, rem2)
            if len(D) == 0: continue
            a = D.a.values.astype(int); sc = D[F11].values.astype(float) @ w11; p = np.exp(sc - sc.max()); p /= p.sum()
            order = [k for k in np.argsort(-p) if lab[a[k]]][:5]
            G = G7[r]; k_ = int(G.fold.iloc[0]); ix = G.index.values; hh = G.hour.values; q0 = q18[ix[hh == 0]][0]
            sse0 = float(np.sum((q18[ix] - y[ix]) ** 2))
            for rank, k in enumerate(order, 1):
                pv = rid(a[k])
                if pv in hid[k_]: continue
                new = q18[ix] + BETA[hh] * np.clip(ST[a[k], 23] - .36 - q0, -2, 2)
                rows.append(dict(rec=r, p2=NUM[b] >= 179, rank=rank, cand=pv, p=float(p[k]), true=TP.get(b) == a[k], same_date=DAY[a[k]] == DAY[TP[b]] if b in TP else False,
                                 gain=sse0 - float(np.sum((new - y[ix]) ** 2)), sse0=sse0))
T = pd.DataFrame(rows); T["helps"] = T.gain > 0; T["big"] = T.gain > .25 * T.sse0
pd.set_option("display.width", 220)
print(f"targets {T.rec.nunique()} | candidate evaluations {len(T)} (top-5 labelled, EV-visible)")
print("by rank (all candidates): share helps / share big (>25% of that record's SSE) / mean gain / share true prev")
print(T.groupby("rank").agg(n=("gain", "size"), helps=("helps", "mean"), big=("big", "mean"), mean_gain=("gain", "mean"), true=("true", "mean")).round(3).to_string())
NT = T[~T.true]
print("NON-true candidates only, by rank:")
print(NT.groupby("rank").agg(n=("gain", "size"), helps=("helps", "mean"), big=("big", "mean"), mean_gain=("gain", "mean"), same_date_as_truth=("same_date", "mean")).round(3).to_string())
per = NT.groupby("rec").agg(any_help=("helps", "any"), any_big=("big", "any"))
print(f"targets with >=1 helpful NON-true candidate in top-5: {per.any_help.mean():.2f} | with >=1 big-help NON-true: {per.any_big.mean():.2f} (of {len(per)})")
print("non-true helpful candidates: same date as the true prev", round(NT[NT.helps].same_date.mean(), 2), "vs non-helpful", round(NT[~NT.helps].same_date.mean(), 2))
print("big helps happen when model was far off: median sse0 big-help", round(T[T.big].sse0.median(), 2), "vs all", round(T.sse0.median(), 2))
