# -*- coding: utf-8 -*-
"""AG7 (2026-10-12 집 클로드; 사용자: 'EC 도 해보면').  AG6 의 EC 판.  사전 고정(이 파일 커밋 후 실행).
안: E1 = v7 앵커(EA2 전체 이동) + T2 보조, E2 = (v7 ∪ ADD) + T2 보조.  T2 = AG6 와 같은 후보·가중(τ .10), A_mix = Σ w EC[a,23], 식 SG2_h + λ(A_mix − SG2 0..h 누적평균), λ .5; 실내 차단 없음(EC 앵커 EA2 와 같게).
기준 = EA2(시드 7/101/2024), 판정 = AG6 와 같은 조건(본페로니 2안 α .0125, 3칸).
(아래 AG6 설명은 온도판 원문)
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
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")].copy(); lo, hi = R.y.min(), R.y.max()
V7S = {r: V.at[r, "prev"] for r in V.index[v7a]}
rng = np.random.default_rng(20261016); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
res = []
for sd, G0 in R.groupby("seed"):
    G0 = G0.sort_values(["rec", "hour"]).reset_index(drop=True); base = G0.EA2.values; y = G0.y.values; p2 = G0.day.values >= 179
    cm = G0.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; sg = G0.SG2.values
    for arm, t1 in (("E1_v7+T2", V7S), ("E2_v7ADD+T2", T1)):
        new = sg.copy(); n1 = n2 = 0
        for rec, G in G0.groupby("rec"):
            k = int(G.fold.iloc[0]); ix = G.index.values; b = idx[rec]
            if rec in t1:
                if t1[rec] in hid[k]: continue
                new[ix] = np.clip(sg[ix] + (EC[idx[t1[rec]], 23] - cm[ix]), lo, hi); n1 += 1
            elif rec in T2 and not (rec in V7S):
                cs = [a for a in T2[rec] if rid(a) not in hid[k]]
                if not cs: continue
                fp = np.array([fpd(b, a) for a in cs]); w = np.exp(-(fp - fp.min()) / TAU); w /= w.sum()
                A = float(np.sum(w * EC[cs, 23])); new[ix] = np.clip(sg[ix] + LAM * (A - cm[ix]), lo, hi); n2 += 1
        db = pd.Series((base - y) ** 2).groupby(G0.rec.values).sum(); dn = pd.Series((new - y) ** 2).groupby(G0.rec.values).sum(); dl = (dn - db).values
        pw = float((dl[rng.integers(0, len(dl), (4000, len(dl)))].sum(1) >= 0).mean()); gain = -dl
        res.append(dict(arm=arm, seed=sd, n_t1=n1, n_t2=n2, all_base=rm(base - y), all_new=rm(new - y), p2_base=rm((base - y)[p2]), p2_new=rm((new - y)[p2]),
                        P_worse=pw, top_share=float(gain.max() / gain.sum()) if gain.sum() > 0 else np.nan))
S = pd.DataFrame(res); S["all_%"] = 100 * (S.all_new / S.all_base - 1); S["p2_%"] = 100 * (S.p2_new / S.p2_base - 1)
pd.set_option("display.width", 220); print(S.round(4).to_string(index=False))
for k, g in S.groupby("arm"):
    ok = (g.all_new < g.all_base).all() and (g.p2_new <= g.p2_base).all() and (g.P_worse < .0125).all() and g.top_share.mean() <= .5
    print(f"{k}: {'ADOPT' if ok else 'FAIL'}")
