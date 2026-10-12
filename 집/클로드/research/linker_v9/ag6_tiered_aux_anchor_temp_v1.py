# -*- coding: utf-8 -*-
"""AG6 (2026-10-12 집 클로드; 사용자: '확실한 전날은 투표로, 나머지는 버리지 말고 보조로').  사전 고정(이 파일 커밋 후 실행).  온도만.
AG5: 진단상 '틀린' 앵커도 같은 날짜 이웃 동 수준이라 모델이 크게 빗나간 날엔 도움 → 애매한 기록도 버리지 말고 약하게 쓰자.
층:
  T1 확실: v7 앵커(p ≥ .8, 정답 기록, |dT_air| ≤ 1.5) ∪ AG3 ADD(v7 비앵커이고 v7·DF 가 같은 정답 기록을 고름, |dT_air| ≤ 1.5).  식 = 13회차(q_h + BETA_h clip(A − .36 − q0, ±2)).
  T2 보조: T1 에 없는 기록 중 DF 날짜·후보가 있는 기록 — 후보(d−1 정답 기록, 다음날 이미 확정된 동 제외)마다 0시 지문 거리 fp,
          가중 w_a = softmax(−fp/τ), τ = .10;  A_mix = Σ w_a · ST[a,23];  같은 식에 반영 비율 λ = .5 를 곱해 얹음(q_h + λ BETA_h clip(A_mix − .36 − q0, ±2)).
          |dT_air| 차단은 가중 평균 실내 23시(Σ w_a IN[a,23]) 기준 ≤ 1.5.
안: ARM1 = T1 만, ARM2 = T1 + T2.  (본페로니 2안 α .0125)
점수·판정: EV(기준 13회차 v22), 전날 후보가 그 EV 폴드 val/del 이면 그 후보는 빼고(남은 후보로 가중; 없으면 적용 안 함).
  4칸 모두 전체 val RMSE 개선 AND 2차 val RMSE 개선(또는 같음) AND 기록 부트스트랩 P(악화) < .0125 (모든 칸) AND 상위 1기록 몫 ≤ .5 (칸 평균).
서술: 층별 기록 수, 평가 60 에서 층별 대상.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9")
from feat9 import *
H = r"C:\work\farmai\집\클로드\research"; TAU, LAM = .10, .5
idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(H, "ec0h", "rise", "dm11_calendar_v1.csv")); DAYr = dict(zip(CAL.rec, CAL.day)); DAY = np.array([DAYr.get(rid(i), np.nan) for i in range(N)])
TLc = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLc[TLc.cert]; SUCC = {}
for a_, b_ in zip(TLc.a1, TLc.b): SUCC.setdefault(int(a_), []).append(int(b_))
fpd = lambda b, a: float(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0]), 4)))
def df_cands(b, vislab):
    vi = np.where(vislab)[0]; eq = np.all(np.abs(X[vi, 0] - X[b, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
    if len(ds) == 0: return []
    d = pd.Series(ds).mode().iloc[0]
    return [a for a in vi if DAY[a] == d - 1 and not any(vislab[s] and DAY[s] == d for s in SUCC.get(a, []))]
# mirror-CV candidate sets (same geometry as DF1)
CANDS = {}
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist())
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        for b in blk:
            if TRAIN[b]: CANDS[rid(b)] = df_cands(b, TRAIN & ~touch)
P = pd.read_csv(os.path.join(env.LOCAL, "linker_v9", "df1_picks_mirror.csv")).set_index("rec")
V = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").set_index("rec")
V["agree"] = V.prev == P.prev.reindex(V.index); V["lab"] = V.prev_is_labelled.astype(bool)
v7a = (V.p >= .8) & V.lab & (V.dT_air.abs() <= 1.5); add = (~v7a) & V.lab & V.agree & (V.dT_air.abs() <= 1.5)
T1 = {r: V.at[r, "prev"] for r in V.index[v7a | add]}
T2 = {r: c for r, c in CANDS.items() if r not in T1 and len(c) > 0}
print(f"tier1 {len(T1)} | tier2 {len(T2)}")
FO = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
D = pd.read_pickle(os.path.join(H, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); D["rec"] = D.row_id.str[:7]; D = D.rename(columns={"sub_temp": "y"}).reset_index(drop=True)
BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
rng = np.random.default_rng(20261015); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def pred(c, use_t2):
    q18 = D[f"v18_a_{c}"].values; out = q18.copy(); n1 = n2 = 0
    for rec, G in D.groupby("rec"):
        k = int(G.fold.iloc[0]); ix = G.index.values; hh = G.hour.values; q0 = q18[ix[hh == 0]][0]; b = idx[rec]
        if rec in T1:
            if T1[rec] in hid[k]: continue
            a = idx[T1[rec]]; out[ix] = q18[ix] + BETA[hh] * np.clip(ST[a, 23] - .36 - q0, -2, 2); n1 += 1
        elif use_t2 and rec in T2:
            cs = [a for a in T2[rec] if rid(a) not in hid[k]]
            if not cs: continue
            fp = np.array([fpd(b, a) for a in cs]); w = np.exp(-(fp - fp.min()) / TAU); w /= w.sum()
            if abs(IN_[b, 0, 0] - float(np.sum(w * IN_[cs, 23, 0]))) > 1.5: continue
            A = float(np.sum(w * ST[cs, 23])); out[ix] = q18[ix] + LAM * BETA[hh] * np.clip(A - .36 - q0, -2, 2); n2 += 1
    return out, n1, n2
res = []
for c in ["7A", "7B", "101A", "101B"]:
    base = D[f"v22_a_{c}"].values; y = D.y.values; p2 = D.day.values >= 179
    for arm, u in (("ARM1_T1", False), ("ARM2_T1+T2", True)):
        new, n1, n2 = pred(c, u)
        db = pd.Series((base - y) ** 2).groupby(D.rec.values).sum(); dn = pd.Series((new - y) ** 2).groupby(D.rec.values).sum(); dl = (dn - db).values
        pw = float((dl[rng.integers(0, len(dl), (4000, len(dl)))].sum(1) >= 0).mean()); gain = -dl
        res.append(dict(arm=arm, cell=c, n_t1=n1, n_t2=n2, all_base=rm(base - y), all_new=rm(new - y), p2_base=rm((base - y)[p2]), p2_new=rm((new - y)[p2]),
                        P_worse=pw, top_share=float(gain.max() / gain.sum()) if gain.sum() > 0 else np.nan))
S = pd.DataFrame(res); S["all_%"] = 100 * (S.all_new / S.all_base - 1); S["p2_%"] = 100 * (S.p2_new / S.p2_base - 1)
pd.set_option("display.width", 220); print(S.round(4).to_string(index=False))
for k, g in S.groupby("arm"):
    ok = (g.all_new < g.all_base).all() and (g.p2_new <= g.p2_base).all() and (g.P_worse < .0125).all() and g.top_share.mean() <= .5
    print(f"{k}: {'ADOPT' if ok else 'FAIL'}")
# eval preview: tiers
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig").set_index("row")
t1e, t2e = [], []
for r, cc in E7.iterrows():
    b = idx[r]; cs = df_cands(b, TRAIN.copy())
    a7 = cc.p_h0 >= .8 and bool(cc.prev_is_labelled) and abs(cc.dT_air) <= 1.5
    dfp = rid(min(cs, key=lambda a: fpd(b, a))) if cs else None
    if a7 or (bool(cc.prev_is_labelled) and dfp == cc.prev_h0 and abs(cc.dT_air) <= 1.5): t1e.append(r)
    elif cs: t2e.append(r)
print(f"\neval tier1 {len(t1e)}: {t1e}\neval tier2 {len(t2e)}: {t2e}")
