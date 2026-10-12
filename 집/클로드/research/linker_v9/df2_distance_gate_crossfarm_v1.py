# -*- coding: utf-8 -*-
"""DF2 (2026-10-12 집 클로드; DF1·DF1b 후속).  사전 고정(커밋 후 실행).
DF1b: 진짜 전날이 후보 안에 있으면 지문 선택 95% 정확, 없으면 0%; 고른 후보의 0시 지문 거리 d_best 가 작으면 '후보 안에 있음'을 가려냄(≤ .08 → 89%).
그 문턱은 같은 자료에서 본 것이라 농장 교차로 판정: 문턱 T 를 한 농장(대상 기록의 ID) 거울 CV 선택들에서 고르고 다른 농장에서 판정.
  선택 규칙(한 농장 안): T ∈ {.04, .06, .08, .10, .12, .15} 중 '정확도 ≥ .85' 를 만족하는 가장 큰 T (없으면 .04).
  규칙 DF2 = DF1 선택(날짜 → d−1 정답 후보 → 지문 최근접) + d_best ≤ T (margin 조건 없음).  대상 = v7 앵커 없는 기록.
판정(EV, 13회차 기준 v22 / EA2 위에 얹기 — DF1 과 같은 공식·차단): 각 방향(F13 에서 고른 T 로 F47 대상, 반대)마다
  온도 4칸·EC 3칸 모두 그 농장 val 전체 RMSE 개선 AND 기록 부트스트랩 P(악화) < .025(칸마다) AND 상위 1기록 몫 ≤ .5.  두 방향 모두 통과해야 ADOPT(열마다 따로).
서술: 방향별 T, 선택 정확도, 적용 기록 수; 평가 60 에 붙는 기록(두 T 중 큰 값 기준 아님 — 각 농장 평가 기록에는 '다른 농장에서 고른 T' 적용).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9")
from feat9 import *
H = r"C:\work\farmai\집\클로드\research"
idx = {rid(i): i for i in range(N)}
fpd = lambda b, a: float(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0]), 4)))
P = pd.read_csv(os.path.join(env.LOCAL, "linker_v9", "df1_picks_mirror.csv")); P["truth"] = P.truth.fillna("")
P["d_best"] = [fpd(idx[r], idx[p]) for r, p in zip(P.rec, P.prev)]; P["farm"] = P.rec.str[:3]
U = P[~P.v7_anchored].copy()
GRID = (.04, .06, .08, .10, .12, .15)
def pickT(S):
    best = .04
    for T in GRID:
        g = S[(S.d_best <= T) & (S.truth != "")]
        if len(g) and (g.prev == g.truth).mean() >= .85: best = T
    return best
FO = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
D = pd.read_pickle(os.path.join(H, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); D["rec"] = D.row_id.str[:7]; D = D.rename(columns={"sub_temp": "y"})
BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")].copy()
lo, hi = R.y.min(), R.y.max()
rng = np.random.default_rng(20261013); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def score(G, base, new):
    y = G.y.values; recs = G.rec.values
    db = pd.Series((base - y) ** 2).groupby(recs).sum(); dn = pd.Series((new - y) ** 2).groupby(recs).sum(); dl = (dn - db).values
    pw = float((dl[rng.integers(0, len(dl), (4000, len(dl)))].sum(1) >= 0).mean()); gain = -dl
    return dict(base=rm(base - y), new=rm(new - y), P_worse=pw, top_share=float(gain.max() / gain.sum()) if gain.sum() > 0 else np.nan, changed=int((np.abs(dl) > 1e-12).sum()))
res = {"TEMP": [], "EC": []}
for sel, judge_f in (("F13", "F47"), ("F47", "F13")):
    T = pickT(U[U.farm == sel]); A = U[(U.farm == judge_f) & (U.d_best <= T)].set_index("rec")
    k = A[A.truth != ""]; print(f"T chosen on {sel} = {T:.2f} -> {judge_f}: picks {len(A)}, accuracy {(k.prev == k.truth).mean() if len(k) else float('nan'):.3f} (known {len(k)})")
    Dj = D[D.farm == judge_f]
    for c in ["7A", "7B", "101A", "101B"]:
        q18 = Dj[f"v18_a_{c}"].values; base = Dj[f"v22_a_{c}"].values.copy(); new = base.copy(); Dj_ = Dj.reset_index(drop=True)
        for rec, G in Dj_.groupby("rec"):
            if rec not in A.index: continue
            a = A.loc[rec]; ix = G.index.values
            if a.prev in hid[int(G.fold.iloc[0])] or abs(a.dT_air) > 1.5 or np.abs(base[ix] - q18[ix]).max() > 1e-9: continue
            q0 = q18[ix[G.hour.values == 0]][0]; new[ix] = q18[ix] + BETA[G.hour.values] * np.clip(a.A_st - .36 - q0, -2, 2)
        res["TEMP"].append(dict(dir=f"{sel}->{judge_f}", cell=c, **score(Dj_, base, new)))
    Rj = R[R.farm == judge_f]
    for sd, G0 in Rj.groupby("seed"):
        G0 = G0.sort_values(["rec", "hour"]).reset_index(drop=True); base = G0.EA2.values.copy(); new = base.copy()
        cm = G0.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values
        for rec, G in G0.groupby("rec"):
            if rec not in A.index or G.anchored.iloc[0]: continue
            a = A.loc[rec]
            if a.prev in hid[int(G.fold.iloc[0])]: continue
            ix = G.index.values; new[ix] = np.clip(G0.SG2.values[ix] + (a.A_ec - cm[ix]), lo, hi)
        res["EC"].append(dict(dir=f"{sel}->{judge_f}", cell=str(sd), **score(G0, base, new)))
pd.set_option("display.width", 220)
for col, rows in res.items():
    S = pd.DataFrame(rows); S["%"] = 100 * (S.new / S.base - 1); print(f"\n[{col}]"); print(S.round(4).to_string(index=False))
    ok = all(((g.new < g.base).all() and (g.P_worse < .025).all() and g.top_share.mean() <= .5) for _, g in S.groupby("dir"))
    print(f"{col}: {'ADOPT' if ok else 'FAIL'}")
# eval preview: each farm's eval records use T chosen on the OTHER farm
sys.argv = ["x"]
Tby = {"F47": pickT(U[U.farm == "F13"]), "F13": pickT(U[U.farm == "F47"])}
CAL = pd.read_csv(os.path.join(H, "ec0h", "rise", "dm11_calendar_v1.csv")); DAYr = dict(zip(CAL.rec, CAL.day)); DAY = np.array([DAYr.get(rid(i), np.nan) for i in range(N)])
TLc = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLc[TLc.cert]; SUCC = {}
for a_, b_ in zip(TLc.a1, TLc.b): SUCC.setdefault(int(a_), []).append(int(b_))
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig").set_index("row")
e7anc = set(E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)].index)
print("\neval preview (v7-unanchored, DF pick, d_best, T of own farm):")
for b in [i for i in range(N) if not TRAIN[i] and FARM[i] in ("F13", "F47")]:
    if rid(b) in e7anc: continue
    vi = np.where(TRAIN)[0]; eq = np.all(np.abs(X[vi, 0] - X[b, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
    if len(ds) == 0: continue
    d = pd.Series(ds).mode().iloc[0]; cand = [a for a in vi if DAY[a] == d - 1 and not any(TRAIN[s] and DAY[s] == d for s in SUCC.get(a, []))]
    if not cand: continue
    dd = sorted((fpd(b, a), a) for a in cand); T = Tby[FARM[b]]
    print(f"   {rid(b)} -> {rid(dd[0][1])} d_best {dd[0][0]:.3f} T {T:.2f} {'ANCHOR' if dd[0][0] <= T else ''}")
