# -*- coding: utf-8 -*-
"""ECH1 (2026-10-16 집 클로드; 사용자 'ec 도 이런 식으로 개선 부분 찾아봐').  사전 고정(커밋 후 실행).  EC.
온도 PCH 계열(6.979~6.985)과 같은 '예측 사슬'(주최 측 허용 6.978-②)을 EC 에: 27번 EC = NH + FL K8 EC 앵커(pred_h = x_h + (A23 − cm_h(x)), CB2 의 REF27).
사슬: K8 없는 기록 r 의 전날 p 가 (거울 연결표 FL6 확신 >= .8, 같은 농장 앞 날짜, 같은 폴드 가림 = 정답 없음, |r 0시 실내 − p 23시 실내| <= 1.5) 이면
  pred_h += w(h) * (A − cm_h(x)),  A = p 의 최종 예측(앵커·사슬 반영) 23시 값, cm_h(x) = r 의 0..h 기본 예측 누적평균 (K8 EC 식과 같은 꼴).
w(h) = s * exp(−h / tau), (s, tau) 격자(s .1~1.5, tau 1·2·3·4·6·8·12·16·24·무한).
P2LOO 는 폴드당 1기록이라 사슬이 생기지 않음 → 판정 배치 = EC DIAG10(6폴드, 219기록; 매개변수는 '다른 5폴드 사슬 행' 으로 맞춘 교차적합) + EL1(DIAG10 전체로 맞춘 값).
기준·사슬 모두 FL6 거울 표(K8 정답 앵커도).  3시드(1357·2468·3579) NH.
판정(단일 안, alpha .025): DIAG10 3시드 모두 전체 dSSE < 0 & 2차 행 dSSE < 0 & DIAG10 5일 블록 P(악화) < .025 & EL1 안전(유의한 악화 아님).
서술: 폴드별 (s, tau), 사슬 기록 수, 고EC 행·사슬 행 변화, 사슬 기록 도움/해.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ech1_ec_prediction_chain_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]; sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "cb2_ec_anchor_shaping_v1.py"), encoding="utf-8").read()
exec(src[:src.index("WS = tuple(")])
__file__ = os.path.join(HERE, "ech1_ec_prediction_chain_v1.py")
MIR = pd.read_csv(os.path.join(env.LOCAL, "fl6_ec_mirror_links.csv")); LKM = {(v, r): (p, q) for v, r, p, q in zip(MIR.layout, MIR.rec, MIR.prev, MIR.p)}
def dT(r, p):
    try: return abs(TIN.at[r, 0] - TIN.at[p, 23])
    except KeyError: return 9.0
def build(v):
    g = np.where((R.validator == v).values)[0]; S = R.iloc[g]; plan = []
    for k, Sk in S.groupby("fold"):
        recs = sorted(set(Sk.rec), key=lambda r: (r[:3], int(r[4:7]))); held = set(recs)
        for r in recs:
            ii = g[(S.fold.values == k) & (S.rec.values == r)]; kind, p = None, None
            if (v, r) in LKM:
                p, q = LKM[(v, r)]
                if isinstance(p, str) and q >= .8 and dT(r, p) <= 1.5:
                    if p not in held and p in idx_ and TR_[idx_[p]] and np.isfinite(EC_[idx_[p], 23]): kind = "K8"
                    elif p in held and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]): kind = "CH"
            plan.append((k, r, ii, kind, p))
    return plan
PLAN = {v: build(v) for v in ("DIAG10", "EL1")}
def run(v, x, wfun):
    out = x.copy(); U = np.zeros(len(x)); fin = {}
    for k, r, ii, kind, p in PLAN[v]:
        xx = x[ii]; cm = np.cumsum(xx) / np.arange(1, len(xx) + 1); hh = R.hour.values[ii]
        if kind == "K8": out[ii] = xx + (EC_[idx_[p], 23] - cm)
        elif kind == "CH" and (k, p) in fin:
            u = fin[(k, p)] - cm; U[ii] = u
            if wfun is not None: out[ii] = xx + wfun(hh) * u
        h23 = out[ii][hh == 23]
        if len(h23): fin[(k, r)] = float(h23[0])
    return out, U
NH = {s: (R[f"A1_{s}"].values - .1 * mt) / .9 for s in SEEDS}
y = R.y.values; hr = R.hour.values
GRID = [(s_, t_) for s_ in np.round(np.arange(.1, 1.51, .1), 2) for t_ in (1, 2, 3, 4, 6, 8, 12, 16, 24, 1e9)]
def fit(mask, base, U):
    m = mask & (U != 0); r_ = (y - base)[m]; u_ = U[m]; h_ = hr[m]
    return min(GRID, key=lambda st: float(((r_ - st[0] * np.exp(-h_ / st[1]) * u_) ** 2).sum())) if m.any() else (0.0, 1e9)
P = {"B": {}, "CH": {}}; PAR = []
for s in SEEDS:
    for v in ("DIAG10", "EL1"):
        g = (R.validator == v).values; base, U = run(v, NH[s], None); P["B"][(v, s)] = base
        if v == "DIAG10":
            ch = base.copy()
            for k in sorted(R.fold[g].unique()):
                tr = g & (R.fold.values != k); te = g & (R.fold.values == k); st = fit(tr, base, U); PAR.append(dict(seed=s, fold=k, s=st[0], tau=st[1]))
                out, _ = run(v, NH[s], lambda hh, st=st: st[0] * np.exp(-hh / st[1])); ch[te] = out[te]
            P["CH"][(v, s)] = ch
        else:
            st = fit((R.validator == "DIAG10").values, P["B"][("DIAG10", s)], run("DIAG10", NH[s], None)[1]); PAR.append(dict(seed=s, fold="EL1(all DIAG10)", s=st[0], tau=st[1]))
            P["CH"][(v, s)] = run(v, NH[s], lambda hh, st=st: st[0] * np.exp(-hh / st[1]))[0]
pd.set_option("display.width", 200); print(pd.DataFrame(PAR).to_string(index=False))
for v in ("DIAG10", "EL1"):
    kinds = pd.Series([kd for k, r, ii, kd, p in PLAN[v]]).value_counts(dropna=False).to_dict(); print(f"{v}: plan {kinds}")
rows, recd = [], []
for v in ("DIAG10", "EL1"):
    g = (R.validator == v).values; yy = y[g]; p2 = R.p2.values[g]; hi = R.hi.values[g].astype(bool); chm = np.zeros(g.sum(), bool)
    idx_g = np.where(g)[0]; pos = {i: j for j, i in enumerate(idx_g)}
    for k, r, ii, kd, p in PLAN[v]:
        if kd == "CH": chm[[pos[i] for i in ii]] = True
    for s in SEEDS:
        a, b = P["CH"][(v, s)][g], P["B"][(v, s)][g]; dd = (a - yy) ** 2 - (b - yy) ** 2
        rows.append(dict(val=v, seed=s, dSSE_all=dd.sum(), dSSE_p2=dd[p2].sum(), rel_all=100 * (rm(a - yy) / rm(b - yy) - 1), rel_p2=100 * (rm((a - yy)[p2]) / rm((b - yy)[p2]) - 1),
                         rel_hi=100 * (rm((a - yy)[hi]) / rm((b - yy)[hi]) - 1) if hi.any() else np.nan, rel_chained=100 * (rm((a - yy)[chm]) / rm((b - yy)[chm]) - 1) if chm.any() else np.nan))
        for r_, xx in pd.Series(dd[chm]).groupby(R.rec.values[g][chm]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=xx))
S = pd.DataFrame(rows); print(S.round(4).to_string(index=False))
RC = pd.DataFrame(recd)
if len(RC): T_ = RC.groupby(["val", "rec"]).d.mean(); print(f"chained records helped {int((T_ < -1e-6).sum())} harmed {int((T_ > 1e-6).sum())}"); print(T_.round(4).sort_values().to_string())
rng = np.random.default_rng(20270157)
def bP(v, worse=True):
    g = (R.validator == v).values; yy = y[g]; key = (R.farm[g] + "_" + (R.day[g] // 5).astype(str)).values
    dd = np.mean([(P["CH"][(v, s)][g] - yy) ** 2 - (P["B"][(v, s)][g] - yy) ** 2 for s in SEEDS], 0); xb = pd.Series(dd).groupby(key).mean().values
    bs = xb[rng.integers(0, len(xb), (20000, len(xb)))].mean(1); return float((bs >= 0).mean() if worse else (bs < 0).mean())
D = S[S.val == "DIAG10"]; pD = bP("DIAG10"); E = S[S.val == "EL1"]; pE = bP("EL1", worse=False); safe = not (E.dSSE_all.mean() > 0 and pE < .025)
ok = bool((D.dSSE_all < 0).all() and (D.dSSE_p2 < 0).all() and pD < .025 and safe)
print(f"ECH1: DIAG10 3 seeds all/p2 better {bool((D.dSSE_all < 0).all())}/{bool((D.dSSE_p2 < 0).all())} | DIAG10 block P(worse) {pD:.4f} | EL1 mean dSSE {E.dSSE_all.mean():+.4f} safe {safe} -> {'PASS' if ok else 'FAIL'}")
EV = pd.read_csv(os.path.join(env.LOCAL, "fl3_eval_links.csv")); same = [isinstance(p, str) and p[:3] == r[:3] and int(p[4:7]) < int(r[4:7]) for r, p in zip(EV.rec, EV.prev)]
print(f"  eval (links only): chain-eligible records {int((np.array(same) & ~EV.prev_is_labelled & (EV.p >= .8) & (EV.dT_air.abs() <= 1.5)).sum())}")
