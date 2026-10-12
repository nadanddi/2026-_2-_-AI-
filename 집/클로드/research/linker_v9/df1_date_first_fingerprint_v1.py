# -*- coding: utf-8 -*-
"""DF1 (2026-10-12 집 클로드; 사용자: '날짜 먼저 정하고 지문으로 동 고르는 실험').  사전 고정(이 파일 커밋 후 실행).
근거: NA4(6.703) — 앵커 없는 평가 기록 중 약 11개는 '날짜는 맞는데 그날 3~4동 중 어느 동인가'가 문제; LF1(6.695) — 날짜가 주어지면 0시 지문 최근접이 진짜 전날 동 82%.
규칙 DF (각 대상 기록 b 독립 판단, 0시 판단 — b 의 0시 입력 + 학습 기록만):
  1) 날짜: b 의 0시 외기 4칸이 정확히 같은 '보이는 학습 기록'들의 DM11 날짜(최빈) = d.  없으면 판단 안 함.
  2) 후보: 보이는 학습 기록 중 DM11 날짜 = d−1.  그중 같은 동 다음날이 이미 '보이는 학습 기록'으로 확실 연결(cert)된 기록은 제외(그 동은 b 가 아님).
  3) 동 선택: 지문 거리 fp(b, a) = 0시 구동기 7종 + 실내 CO2 (전체 표준화)의 |차| 평균(상한 4).  최소 후보 a*, 2등과 차 margin.
  4) 채택: margin ≥ M = 0.10 (사전 고정; 0.05·0.20 은 서술).
  적용 대상 = 짝찾기 v7 거울 CV(cv_all400_choice_v7m)에서 앵커가 없는 기록(p < .8 또는 전날 정답 없음 또는 |dT_air| > 1.5).
검증 세계: 거울 CV 기하(linker_v7/cv400_v7m_critic.py 와 같은 덩어리: 10기록 + 양옆 지워진 기록 + 다른 ID 거울 덩어리 숨김).
  주의: DM11 달력은 학습 확실 연결 전체로 만든 것이라 숨긴 기록의 연결도 달력 순서에 들어감(약한 낙관, 평가에서도 학습 연결은 모두 쓸 수 있음).
점수: EV(ev_folds_v1)에서 v7 앵커와 같은 방식으로 얹음(전날이 그 EV 폴드의 val/del 이면 버림).
  온도: 13회차 식 q_h + BETA_h · clip(A − .36 − q0, −2, 2), q = v18(4칸), BETA = local/linker_v7/ta8_beta.npy, |dT_air| ≤ 1.5 차단 유지. 기준 = v22(13회차 구성).
  EC: EA2 식 SG2_h + (A − SG2 0..h 누적평균), 기준 = EA2(시드 7/101/2024).
채택(열마다 따로, α .025): 모든 칸(온도 4, EC 3)에서 EV 전체 val RMSE 개선 AND 2차 val RMSE 개선(또는 2차 변화 기록 없음) AND
  기록 단위 부트스트랩(4000, 바뀐 기록 포함 전체 val 기록) P(악화) < .025 (모든 칸) AND 이득 상위 1기록 몫 ≤ .5 (칸 평균).
서술: DF 선택 수·정확도(진단용 truth), margin 별 정확도, 평가 60 에 적용하면 앵커 붙는 기록.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9")
from feat9 import *   # v7 features + FPA (8 vars x 24 h standardized) ; N, FARM, NUM, TRAIN, X (outdoor), IN_, EC, ST, rid
H = r"C:\work\farmai\집\클로드\research"
M_FIX = 0.10
CAL = pd.read_csv(os.path.join(H, "ec0h", "rise", "dm11_calendar_v1.csv")); DAYr = dict(zip(CAL.rec, CAL.day))
DAY = np.array([DAYr.get(rid(i), np.nan) for i in range(N)])
TLc = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLc[TLc.cert]; SUCC = {}
for a, b in zip(TLc.a1, TLc.b): SUCC.setdefault(int(a), []).append(int(b))
TRUTHP = {int(b): int(a) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
fpd = lambda b, a: float(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0]), 4)))
def df_pick(b, vislab):
    vi = np.where(vislab)[0]
    eq = np.all(np.abs(X[vi, 0] - X[b, 0]) < 1e-6, axis=1)
    ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
    if len(ds) == 0: return None
    d = pd.Series(ds).mode().iloc[0]
    cand = [a for a in vi if DAY[a] == d - 1 and not any(vislab[s] and DAY[s] == d for s in SUCC.get(a, []))]
    if len(cand) < 1: return None
    dd = sorted((fpd(b, a), a) for a in cand)
    margin = (dd[1][0] - dd[0][0]) if len(dd) > 1 else 9.0
    return dict(a=int(dd[0][1]), margin=margin, ncand=len(cand), d=d)
# ---- mirror-CV geometry (as cv400_v7m_critic.py) ----
out = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist())
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        vislab = TRAIN & ~touch
        for b in blk:
            if not TRAIN[b]: continue
            r = df_pick(b, vislab)
            if r is None: continue
            out.append(dict(rec=rid(b), prev=rid(r["a"]), margin=r["margin"], ncand=r["ncand"], truth=rid(TRUTHP[b]) if b in TRUTHP else "",
                            A_st=float(ST[r["a"], 23]), A_ec=float(EC[r["a"], 23]), dT_air=float(IN_[b, 0, 0] - IN_[r["a"], 23, 0])))
P = pd.DataFrame(out)
V7 = pd.read_csv(os.path.join(H, "local", "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").set_index("rec")
v7anc = set(V7[(V7.p >= .8) & V7.prev_is_labelled.astype(bool) & (V7.dT_air.abs() <= 1.5)].index)
P["v7_anchored"] = P.rec.isin(v7anc); P["correct"] = (P.prev == P.truth); P["known"] = P.truth != ""
Q = P[~P.v7_anchored & P.known]
print(f"DF picks: {len(P)} | on v7-unanchored with known truth: {len(Q)}")
for Mx in (0.0, 0.05, 0.10, 0.20):
    S = Q[Q.margin >= Mx]; print(f"   margin >= {Mx:.2f}: n {len(S)}, accuracy {S.correct.mean() if len(S) else float('nan'):.3f}, pass-2 n {int((S.rec.str[4:7].astype(int) >= 179).sum())}")
A = P[~P.v7_anchored & (P.margin >= M_FIX)].set_index("rec")
os.makedirs(os.path.join(env.LOCAL, "linker_v9"), exist_ok=True); P.to_csv(os.path.join(env.LOCAL, "linker_v9", "df1_picks_mirror.csv"), index=False)
# ---- EV scoring ----
FO = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv"))
hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
rng = np.random.default_rng(20261012); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def judge(name, frames):
    rows = []
    for cell, (G, base, new) in frames.items():
        y = G.y.values; recs = G.rec.values
        db = pd.Series((base - y) ** 2).groupby(recs).sum(); dn = pd.Series((new - y) ** 2).groupby(recs).sum()
        dl = (dn - db).values; idx = rng.integers(0, len(dl), (4000, len(dl))); pw = float((dl[idx].sum(1) >= 0).mean())
        gain = -dl; share = float(gain.max() / gain.sum()) if gain.sum() > 0 else np.nan
        p2 = G.day.values >= 179
        rows.append(dict(cell=cell, all_base=rm(base - y), all_new=rm(new - y), p2_base=rm((base - y)[p2]), p2_new=rm((new - y)[p2]), P_worse=pw, top_share=share, changed=int((np.abs(dl) > 1e-12).sum())))
    R = pd.DataFrame(rows); R["all_%"] = 100 * (R.all_new / R.all_base - 1); R["p2_%"] = 100 * (R.p2_new / R.p2_base - 1)
    print(f"\n[{name}]"); print(R.round(4).to_string(index=False))
    ok = (R.all_new < R.all_base).all() and ((R.p2_new < R.p2_base) | (R.p2_new == R.p2_base)).all() and (R.P_worse < .025).all() and R.top_share.mean() <= .5
    print(f"{name}: {'ADOPT' if ok else 'FAIL'}")
# temperature
D = pd.read_pickle(os.path.join(H, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); D["rec"] = D.row_id.str[:7]; D = D.rename(columns={"sub_temp": "y"})
BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
fr = {}; app_t = set()
for c in ["7A", "7B", "101A", "101B"]:
    q18 = D[f"v18_a_{c}"].values; base = D[f"v22_a_{c}"].values.copy(); new = base.copy()
    for rec, G in D.groupby("rec"):
        if rec not in A.index: continue
        a = A.loc[rec]; k = int(G.fold.iloc[0])
        if a.prev in hid[k] or abs(a.dT_air) > 1.5: continue
        ix = G.index.values; q0 = q18[ix[G.hour.values == 0]][0]
        if np.abs(base[ix] - q18[ix]).max() > 1e-9: continue   # already anchored in v22
        new[ix] = q18[ix] + BETA[G.hour.values] * np.clip(a.A_st - .36 - q0, -2, 2); app_t.add(rec)
    fr[c] = (D, base, new)
print(f"\ntemperature: DF anchors applied in EV to {len(app_t)} records")
judge("TEMP", fr)
# EC
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")].copy()
lo, hi = R.y.min(), R.y.max(); fr = {}; app_e = set()
for sd, G0 in R.groupby("seed"):
    G0 = G0.sort_values(["rec", "hour"]).reset_index(drop=True); base = G0.EA2.values.copy(); new = base.copy()
    cm = G0.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values
    for rec, G in G0.groupby("rec"):
        if rec not in A.index or G.anchored.iloc[0]: continue
        a = A.loc[rec]; k = int(G.fold.iloc[0])
        if a.prev in hid[k]: continue
        ix = G.index.values; new[ix] = np.clip(G0.SG2.values[ix] + (a.A_ec - cm[ix]), lo, hi); app_e.add(rec)
    fr[str(sd)] = (G0, base, new)
print(f"\nEC: DF anchors applied in EV to {len(app_e)} records")
judge("EC", fr)
# eval preview
TR = TRAIN.copy(); ev = [i for i in range(N) if not TRAIN[i] and FARM[i] in ("F13", "F47")]
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig").set_index("row")
e7anc = set(E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)].index)
pv = []
for b in ev:
    if rid(b) in e7anc: continue
    r = df_pick(b, TR)
    if r is not None: pv.append((rid(b), rid(r["a"]), round(r["margin"], 3), r["ncand"], "ACCEPT" if r["margin"] >= M_FIX else ""))
print("\neval preview (v7-unanchored):"); [print("  ", x) for x in pv]
