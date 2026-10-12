# -*- coding: utf-8 -*-
"""AG11 (2026-10-12 집 클로드; 사용자: '온도부터 — 안전하게 포괄률을 올리려면').  사전 고정(이 파일 커밋 후 실행).  온도만.
층(모두 거울 CV 선택, EV 에서 전날이 val/del 이면 버림):
  T1 확실(λ 1): v7 앵커(p ≥ .8·정답·|dT| ≤ 1.5, 13회차 그대로 유지) ∪ HP3 행 중 (HP2 선택 == META2 선택 & 메타 확률 ≥ .5 & 차단 아님).
  T2 애매(λ .5): T1 에 없는 HP3 행 중 (HP2 선택 == META2 선택 & 메타 확률 < .5) 또는 (HP2 선택 == META2 선택 & 차단됨) — 차단된 것도 절반으로 사용.
  T3 보조(λ .5): T1·T2 에 없고 DF 날짜·후보가 있는 기록 — AG6 T2 와 같음(지문 softmax τ .1 가중 평균 A_mix, 실내 가중 23시 차 ≤ 1.5).
  공식: q18_h + λ·BETA_h·clip(A − .36 − q0, ±2).
안: ARM_A = T1+T2, ARM_B = T1+T2+T3.  기준 = 13회차(v22).  판정(본페로니 2안 α .0125): 4칸 모두 전체 개선 AND 2차 개선(또는 같음)
  AND 기록 부트스트랩 P(악화) < .0125 AND 상위 1기록 몫 ≤ .5.
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
V = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").set_index("rec")
v7a = (V.p >= .8) & V.prev_is_labelled.astype(bool) & (V.dT_air.abs() <= 1.5)
H3 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "hp3_rows.csv"))
L2 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "lr2_candidates.csv")); B2 = L2[L2.lab].loc[lambda d: d.groupby("rec").pm.idxmax()].set_index("rec")
H3["meta"] = H3.rec.map(B2.cand); H3["metap"] = H3.rec.map(B2.pm).fillna(0); H3 = H3[(~H3.nocand.astype(bool)) & (H3.pick == H3.meta)]
T1 = {r: V.at[r, "prev"] for r in V.index[v7a]}
for r, p, mp, gt in zip(H3.rec, H3.pick, H3.metap, H3.gate.astype(bool)):
    if r not in T1 and mp >= .5 and not gt: T1[r] = p
T2m = {r: p for r, p, mp, gt in zip(H3.rec, H3.pick, H3.metap, H3.gate.astype(bool)) if r not in T1 and (mp < .5 or gt)}
T2 = {r: c for r, c in CANDS.items() if r not in T1 and r not in T2m and len(c) > 0}
print(f"tier1 {len(T1)} | tier2 {len(T2m)} | tier3 (aux mixture) {len(T2)}")
FO = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
D = pd.read_pickle(os.path.join(H, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); D["rec"] = D.row_id.str[:7]; D = D.rename(columns={"sub_temp": "y"}).reset_index(drop=True)
BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
rng = np.random.default_rng(20261015); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def pred(c, use_t2, use_t2m=True):
    q18 = D[f"v18_a_{c}"].values; out = q18.copy(); n1 = n2 = 0
    for rec, G in D.groupby("rec"):
        k = int(G.fold.iloc[0]); ix = G.index.values; hh = G.hour.values; q0 = q18[ix[hh == 0]][0]; b = idx[rec]
        if rec in T1:
            if T1[rec] in hid[k]: continue
            a = idx[T1[rec]]; out[ix] = q18[ix] + BETA[hh] * np.clip(ST[a, 23] - .36 - q0, -2, 2); n1 += 1
        elif use_t2m and rec in T2m:
            if T2m[rec] in hid[k]: continue
            a = idx[T2m[rec]]; out[ix] = q18[ix] + LAM * BETA[hh] * np.clip(ST[a, 23] - .36 - q0, -2, 2); n2 += 1
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
    for arm, u in (("ARM_A_T1+T2", False), ("ARM_B_T1+T2+T3", True)):
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
