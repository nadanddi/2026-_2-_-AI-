# -*- coding: utf-8 -*-
"""PP3 v2: 연결 기준을 c1 <= 5 로 넓힘(v1 cert|c1<=1 은 쌍 18·에피소드 2라 판단 불가) + 증가 평균 출력. PP3 (2026-10-13 집 클로드, hx8/physproxy).  '농축 가설'의 직접 점검 — 수준 대신 '변화'. 서술 + 가짜 평가.
(c) 같은 동 연속 고EC 날(진단 연결 a→b, 둘 다 학습, a 하루 평균 ≥ 1) 에서
    within_b = EC_b(23시) - EC_b(0시),  Δ = 하루평균_b - 하루평균_a 를 b 날의 파생 물리량(PP1)과 순위상관 + 순열 기준선.
    현재 모델 OOF 가 하루 안 변화를 얼마나 따라가나(q_b 23시 - 0시)도 같이.
(d) 가짜 평가(앵커가 k 일 앞에만 있는 고EC 날; 평가 F13_221·223·225 = F13_217 23시로부터 2·3·4일 상황 흉내):
    같은 동 고EC 사슬에서 시작일 s 의 23시 정답을 앵커로, k = 2..4 일 뒤 날 t 의 하루 수준 추정:
      M  현재 모델 q_t(OOF 하루 평균)
      P0 앵커 그대로(지속)                     — 6.689 BI1 의 한쪽 판
      P1 앵커 + k·(학습 평균 하루 증가; 사례 밖 적합)
      P2 앵커 + Σ_{중간·당일} (a + b·물리량)  (하루 증가를 물리량으로; 사례 밖 적합)
      B  β·앵커쪽 + (1-β)·모델  (β 사례 밖 적합)
    사례 단위 = 에피소드(사슬 성분) 하나 빼기.  평가 실제 상황은 81일(F13_219) 기록이 지워져 그 날의 입력이 없음 → P2 는 지운 날을 평균 증가로 채운다고 보고
    여기서는 '중간 날 하나는 물리량 없이 평균'인 판(P2m)도 같이 계산.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pp1_physproxy_v1 import *   # noqa

F23 = pd.read_csv(os.path.join(here, "pp1_train_feats_full_v1.csv"), index_col=0)
Od = O[O.validator == "DIAG10"].copy()
QH = Od.groupby(["rec", "hour"]).sg.mean().unstack()          # 3시드 평균 시간별 OOF
qday = QH.mean(1)
E = ymat
cert = dict(zip([(nm[int(a)], nm[int(b)]) for a, b in zip(TL.a1, TL.b)], TL.cert))
c1 = dict(zip([(nm[int(a)], nm[int(b)]) for a, b in zip(TL.a1, TL.b)], TL.c1))
# successor map restricted to training-training links with small midnight cost
succ = {}
for (a, b) in links:
    if a in E.index and b in E.index and c1[(a, b)] <= 5.0:
        succ[a] = b
pred_ = {b: a for a, b in succ.items()}

# ---------- (c) ----------
rows = []
for a, b in succ.items():
    if yday[a] >= 1.0:
        rows.append(dict(a=a, b=b, ya=yday[a], yb=yday[b], d=yday[b] - yday[a], within=E.loc[b, 23] - E.loc[b, 0],
                         mid=E.loc[b, 0] - E.loc[a, 23], qwithin=(QH.loc[b, 23] - QH.loc[b, 0]) if b in QH.index else np.nan,
                         qd=(qday[b] - qday[a]) if (a in qday.index and b in qday.index) else np.nan, cert=cert[(a, b)]))
P = pd.DataFrame(rows)
print(f"(c) consecutive same-house pairs starting high (a>=1.0): {len(P)} (cert {int(P.cert.sum())}) | mean d {P.d.mean():+.3f}, within {P.within.mean():+.3f}, midnight jump {P.mid.mean():+.3f}")
print(f"    model OOF: corr(qwithin, within) {spear(P.qwithin.values, P.within.values):+.2f}, corr(qd, d) {spear(P.qd.values, P.d.values):+.2f}  (n with OOF {P.qd.notna().sum()})")
Fb = F23.loc[P.b].reset_index(drop=True)
res = []
for c in FEATS:
    f = Fb[c].values.astype(float)
    res.append(dict(feat=c, rho_within=spear(f, P.within.values), rho_d=spear(f, P.d.values),
                    rho_within_resid=spear(f, (P.within - P.qwithin).values)))
R_ = pd.DataFrame(res).set_index("feat")
null_w, null_d = [], []
Fm = Fb[FEATS].values.astype(float)
for it in range(1000):
    pi = rng.permutation(len(P))
    null_w.append(max(abs(spear(Fm[pi, j], P.within.values)) for j in range(len(FEATS))))
    null_d.append(max(abs(spear(Fm[pi, j], P.d.values)) for j in range(len(FEATS))))
print(f"    null 95% max|rho| (record perm, 21 feats): within {np.quantile(null_w,.95):.2f}, d {np.quantile(null_d,.95):.2f}")
print(R_.reindex(R_.rho_within.abs().sort_values(ascending=False).index).round(2).to_string())

# ---------- (d) ----------
# chains of consecutive high days (a>=1 for all members except last may be anything? keep: start day s high, all days in between high)
def chain_from(s, kmax=4):
    out = [s]; x = s
    for _ in range(kmax):
        x = succ.get(x)
        if x is None: break
        out.append(x)
    return out
ep_all = episodes([r for r in E.index if yday[r] >= 1.0])
cases = []
for s in E.index:
    if yday[s] < 1.0: continue
    ch = chain_from(s)
    for k in range(2, len(ch)):
        t = ch[k]; mids = ch[1:k + 1]                        # days whose increment accumulates (s+1 .. t)
        if t not in qday.index or qday[t] < 0.8: continue    # 평가 8 은 고EC '예측'으로 골라짐 → 표적 날도 예측으로만 선택(정답 선택 편향 회피)
        cases.append(dict(s=s, t=t, k=k, anchor=E.loc[s, 23], y=yday[t], q=qday[t], mids=mids, ep=ep_all.get(s, s)))
C = pd.DataFrame(cases)
print(f"\n(d) pseudo-eval cases (anchor k days before, chain of training high days): {len(C)} | k counts {C.k.value_counts().to_dict()} | episodes {C.ep.nunique()}")
# daily increments for fitting: each consecutive high pair (a->b): inc = mean_b - EC_a(23)  [anchor-type increment]
inc_rows = [dict(a=a, b=b, inc=yday[b] - E.loc[a, 23], ep=ep_all.get(a, a)) for a, b in succ.items() if yday[a] >= 1.0]
I = pd.DataFrame(inc_rows)
I1 = pd.DataFrame([dict(a=a, b=b, inc1=E.loc[b, 23] - E.loc[a, 23], ep=ep_all.get(a, a)) for a, b in succ.items() if yday[a] >= 1.0])
print(f"    increments (all high pairs): 23h->23h mean {I1.inc1.mean():+.3f} sd {I1.inc1.std():.3f} n {len(I1)} | mean_b - EC_a(23) mean {I.inc.mean():+.3f} sd {I.inc.std():.3f}")
print("    by episode:", I1.groupby("ep").inc1.agg(["count", "mean"]).round(3).to_dict("index"))
print(C[["s", "t", "k", "anchor", "y", "q", "ep"]].round(3).to_string(index=False))
feats_try = ["Ebal05", "dAH_closed_day", "vpd_x_rad", "radE_sum", "AHx", "co2_draw", "dT_per_rad", "Ebal2"]
out = []
for _, c in C.iterrows():
    tr_inc = I[I.ep != c.ep]; tr1 = I1[I1.ep != c.ep]
    # day-level target 'mean_t - anchor' = sum of 23h-to-23h increments for s+1..t-1 + (mean_t - EC_{t-1}(23))
    m23 = tr1.inc1.mean(); mlast = tr_inc.inc.mean()
    p1 = c.anchor + (c.k - 1) * m23 + mlast
    row = dict(s=c.s, t=c.t, k=c.k, ep=c.ep, y=c.y, M=c.q, P0=c.anchor, P1=p1)
    for fn in feats_try:
        # fit inc1 ~ a + b f(b) and inc ~ a + b f(b) on other episodes
        fa = F23.loc[tr1.b, fn].values.astype(float); ok = np.isfinite(fa)
        b1 = np.polyfit(fa[ok], tr1.inc1.values[ok], 1) if ok.sum() >= 5 else np.array([0, m23])
        fl = F23.loc[tr_inc.b, fn].values.astype(float); ok2 = np.isfinite(fl)
        bl = np.polyfit(fl[ok2], tr_inc.inc.values[ok2], 1) if ok2.sum() >= 5 else np.array([0, mlast])
        mids = c.mids; fv = [F23.loc[m, fn] for m in mids]
        fv = [v if np.isfinite(v) else np.nanmedian(fa) for v in fv]
        p2 = c.anchor + sum(np.polyval(b1, v) for v in fv[:-1]) + np.polyval(bl, fv[-1])
        # P2m: the first intermediate day has no inputs (deleted-day situation) -> mean increment for it
        p2m = c.anchor + m23 + sum(np.polyval(b1, v) for v in fv[1:-1]) + np.polyval(bl, fv[-1]) if c.k >= 2 else p2
        row[f"P2_{fn}"] = p2; row[f"P2m_{fn}"] = p2m
    out.append(row)
Rd = pd.DataFrame(out)
# blend beta (anchor-type P1 vs model) fitted out-of-episode
for nmx in ["P0", "P1"]:
    bl = np.zeros(len(Rd))
    for i, r_ in Rd.iterrows():
        tr = Rd[Rd.ep != r_.ep]; dx = (tr[nmx] - tr.M).values; dy = (tr.y - tr.M).values
        beta = float(np.clip((dx * dy).sum() / ((dx * dx).sum() + 1e-9), 0, 1)); bl[i] = r_.M + beta * (r_[nmx] - r_.M)
    Rd[f"B_{nmx}"] = bl
cols = ["M", "P0", "P1", "B_P0", "B_P1"] + [f"P2_{f}" for f in feats_try] + [f"P2m_{f}" for f in feats_try]
print("    day-level RMSE by k (lower better); 'better_vs_M' = share of cases closer than model")
S = []
for kk in [2, 3, 4, "all"]:
    sub = Rd if kk == "all" else Rd[Rd.k == kk]
    if len(sub) == 0: continue
    d = {"k": kk, "n": len(sub), "ep": sub.ep.nunique()}
    for c in cols: d[c] = rmse(sub[c].values, sub.y.values)
    S.append(d)
S = pd.DataFrame(S).set_index("k"); print(S.round(3).T.to_string())
print("\n    share closer than model (all k):", {c: round(float(np.mean(np.abs(Rd[c] - Rd.y) < np.abs(Rd.M - Rd.y))), 2) for c in cols[1:]})
# episode-cluster bootstrap of RMSE difference P1 vs best P2 and vs M
def boot(c1_, c2_, B=2000):
    eps = Rd.ep.unique(); d = []
    for _ in range(B):
        pick = rng.choice(eps, len(eps)); sub = pd.concat([Rd[Rd.ep == e] for e in pick])
        d.append(rmse(sub[c1_].values, sub.y.values) - rmse(sub[c2_].values, sub.y.values))
    d = np.array(d); return float(np.mean(d)), float(np.quantile(d, .025)), float(np.quantile(d, .975)), float(np.mean(d >= 0))
for c in ["P1", "B_P1"] + [f"P2_{f}" for f in feats_try]:
    m, lo, hi, pw = boot(c, "M"); print(f"    RMSE({c}) - RMSE(M): {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  P(not better) {pw:.3f}")
for f in feats_try:
    m, lo, hi, pw = boot(f"P2_{f}", "P1"); print(f"    RMSE(P2_{f}) - RMSE(P1): {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  P(not better) {pw:.3f}")
Rd.to_csv(os.path.join(here, "pp3_cases_v2.csv"), index=False)
