# -*- coding: utf-8 -*-
"""LR2 (2026-10-12 집 클로드; 사용자: '정확도·포괄률을 더 확실하게 — 다른 방법·더 좋은 모델·계산법').  서술(거울 CV, v7/v10 기하).
(1) 계산법: 진단용 진짜 전날을 전체(a1) / 확실(cert)만 두 기준으로 정확도·포괄률.
(2) 모델: v10(저장), v10fp(FEATS10 + 0시 지문 거리 fp, 조건부 로짓), RK(LightGBM LambdaRank, FEATS10+fp, 덩어리마다 재학습).
(3) 메타: 후보 = v10fp 상위 3 ∪ RK 상위 3 ∪ DF 선택(보이는 정답 기록만), 특징 = 각 모델 확률·순위, fp, DF 일치, 표 수, |dT|, xf, lat, 2차, 후보 수;
    LightGBM 이진 분류, 덩어리 그룹 5겹 교차예측; 기록마다 최고 후보.
앵커 = 확률(메타는 메타 확률) ≥ 문턱 AND 전날이 보이는 정답 기록 AND |dT_air| ≤ 1.5.  이을 수 있는 = 진짜 전날이 보이는 정답 기록.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9")
from feat10 import *
import env, env_extra  # noqa
import lightgbm as lgb
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
_X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]; _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA, (0, 1))) / (np.nanstd(FPA, (0, 1)) + 1e-9)
fpd = lambda b, a: np.nan_to_num(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0][None]), 4), axis=1), nan=1.0)
FF = FEATS10 + ["fp"]
def feats(b, lab, removed):
    D, _ = features10(b, 0, lab, removed=removed)
    if len(D): D["fp"] = fpd(b, D.a.values.astype(int))
    return D
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TP = {int(b): int(a) for a, b in zip(TLa.a1, TLa.b)}; CERT = {int(b): bool(c) for b, c in zip(TLa.b, TLa.cert)}
TRG = []
for b, a in zip(TL7.b, TL7.a):
    lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
    df = feats(b, lab, TEST_IDX)
    if len(df) == 0: continue
    y = np.where(df.a.values == a)[0]
    if len(y): TRG.append((b, a, df[FF].values.astype(float), y[0]))
print("train groups", len(TRG))
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAYr = dict(zip(CAL.rec, CAL.day)); DAY = np.array([DAYr.get(rid(i), np.nan) for i in range(N)])
SUCC = {}
for a_, b_, c_ in zip(TLa.a1, TLa.b, TLa.cert):
    if c_: SUCC.setdefault(int(a_), []).append(int(b_))
def df_pick(b, vislab):
    vi = np.where(vislab)[0]; eq = np.all(np.abs(X[vi, 0] - X[b, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
    if len(ds) == 0: return None
    d = pd.Series(ds).mode().iloc[0]; cand = [a for a in vi if DAY[a] == d - 1 and not any(vislab[s] and DAY[s] == d for s in SUCC.get(a, []))]
    if not cand: return None
    return int(min(cand, key=lambda a: float(fpd(b, np.array([a]))[0])))
rows = []; base = {}
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        tr = [(X_, y_) for b_, a_, X_, y_ in TRG if not touch[b_] and not touch[a_]]
        w = fit_clogit(tr, l2=0.5, iters=300)
        Xr = np.vstack([x for x, _ in tr]); yr = np.concatenate([(np.arange(len(x)) == y).astype(int) for x, y in tr])
        rk = lgb.LGBMRanker(n_estimators=200, learning_rate=.05, num_leaves=15, min_child_samples=10, verbose=-1, random_state=5).fit(Xr, yr, group=[len(x) for x, _ in tr])
        lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if not TRAIN[b]: continue
            D = feats(b, lab, rem2)
            if len(D) == 0: continue
            Xc = D[FF].values.astype(float); a = D.a.values.astype(int)
            s1 = Xc @ w; p1 = np.exp(s1 - s1.max()); p1 /= p1.sum()
            s2 = rk.predict(Xc); p2 = np.exp(s2 - s2.max()); p2 /= p2.sum()
            r1 = np.argsort(np.argsort(-p1)); r2 = np.argsort(np.argsort(-p2))
            dp = df_pick(b, lab)
            keep = set(np.where(r1 < 3)[0]) | set(np.where(r2 < 3)[0])
            if dp is not None and dp in set(a): keep.add(int(np.where(a == dp)[0][0]))
            base[rid(b)] = dict(fp_top=rid(a[np.argmax(p1)]), fp_p=float(p1.max()), rk_top=rid(a[np.argmax(p2)]), rk_p=float(p2.max()), lab_fp=bool(lab[a[np.argmax(p1)]]), lab_rk=bool(lab[a[np.argmax(p2)]]))
            for k in keep:
                rows.append(dict(rec=rid(b), grp=f"{f}_{s}", cand=rid(a[k]), lab=bool(lab[a[k]]), p_fp=p1[k], r_fp=r1[k], p_rk=p2[k], r_rk=r2[k], fp=Xc[k, -1],
                                 df_agree=float(dp == a[k]), votes=float(r1[k] == 0) + float(r2[k] == 0) + float(dp == a[k]), dT=abs(IN_[b, 0, 0] - IN_[a[k], 23, 0]),
                                 xf=Xc[k, FF.index("xf")], lat=Xc[k, FF.index("lat")], p2=float(NUM[b] >= 179), ncand=len(a), y=float(TP.get(b) == a[k])))
M = pd.DataFrame(rows).fillna({"dT": 9.0}); MF = ["p_fp", "r_fp", "p_rk", "r_rk", "fp", "df_agree", "votes", "dT", "xf", "lat", "p2", "ncand"]
rng = np.random.default_rng(2); gr = M.grp.unique(); fo = {gg: k % 5 for k, gg in enumerate(rng.permutation(gr))}; M["fold"] = M.grp.map(fo); M["pm"] = np.nan
for k in range(5):
    trm, tem = M[(M.fold != k)], M[M.fold == k]
    mdl = lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=7, min_child_samples=15, verbose=-1, random_state=3).fit(trm[MF], trm.y)
    M.loc[tem.index, "pm"] = mdl.predict_proba(tem[MF])[:, 1]
M.to_csv(os.path.join(env.LOCAL, "linker_v10", "lr2_candidates.csv"), index=False)
idx = {rid(i): i for i in range(N)}
vis = {}
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0]; m = s + sh
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
        for b in blk:
            if TRAIN[b]: vis[rid(b)] = TRAIN & ~touch
def report(name, picks):   # picks: dict rec -> (prev, prob)
    for tname, tset in (("all-truth", None), ("cert-truth", True)):
        recs = [r for r in vis if idx[r] in TP and vis[r][TP[idx[r]]] and (tset is None or CERT[idx[r]])]
        can = set(recs); line = []
        for th in (.5, .7, .8, .9, .95):
            A = {r: p for r, (p, q) in picks.items() if q >= th and vis[r][idx[p]] and abs(IN_[idx[r], 0, 0] - IN_[idx[p], 23, 0]) <= 1.5 and (tset is None or CERT.get(idx[r], False) or idx[r] not in TP)}
            ok = sum(1 for r, p in A.items() if idx[r] in TP and rid(TP[idx[r]]) == p and r in can)
            nA = sum(1 for r in A if idx[r] in TP and (tset is None or CERT[idx[r]]))
            line.append(f"{th:.2f}: {nA:3d} anc prec {ok/max(1,nA):.3f} rec {ok/len(can):.2f}")
        print(f"  {name:6s} {tname:10s} (linkable {len(can)}) | " + " | ".join(line))
V10 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "cv_all400_choice_v10m.csv"), encoding="utf-8-sig")
report("v10", {r: (p, q) for r, p, q in zip(V10.rec, V10.prev, V10.p)})
report("v10fp", {r: (v["fp_top"], v["fp_p"]) for r, v in base.items()})
report("RK", {r: (v["rk_top"], v["rk_p"]) for r, v in base.items()})
B = M[M.lab].loc[lambda d: d.groupby("rec").pm.idxmax()]
report("META", {r: (c, q) for r, c, q in zip(B.rec, B.cand, B.pm)})
print("\ncandidate-pool ceiling (truth among kept labelled candidates) / linkable:", round(M[M.lab & (M.y == 1)].rec.nunique() / sum(1 for r in vis if idx[r] in TP and vis[r][TP[idx[r]]]), 3))
