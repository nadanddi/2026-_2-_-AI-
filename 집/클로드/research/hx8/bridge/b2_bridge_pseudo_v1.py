# -*- coding: utf-8 -*-
"""B2 (2026-10-13 집 클로드, hx8/bridge) — 상태공간 '다리' 추정: 가짜 평가 사례 검증 + 평가 8(+1)기록 추정치.

[무엇이 BI1(6.689)과 다른가]
 BI1 = 양 끝 정답 선형 보간 I 와 모델 q 를 간격별 고정 β 로 섞음.  여기(KB)는
  (1) 숨은 수준 x(τ) 를 반나절 격자 위 확률과정(무작위 걸음 RW / 로그 RW / 적분 무작위 걸음 IRW=추세)으로 두고,
  (2) 양 끝 정답(앞 기록 0시·평균·23시, 뒤 기록 0시·평균·23시, 있으면 그 바깥 하루 평균)을 '거의 정확한 관측',
  (3) 사이 날들의 모델 예측을 '압축된 관측' p_s = a + b·x̄_s + e_s (e 는 날 사이 AR(1) 상관 ρ) 로 넣어
  가우스 조건부 평균(칼만 평활과 같음)으로 목표 날 하루 평균을 추정.  → 예측의 압축(b<1)을 되돌리되, 그 무게는 위치(앵커와의 거리)·
  예측 잡음·예측끼리 상관에 따라 자동으로 정해짐.  같은 ID 앞 번호(=목표보다 앞 날) 예측만 관측으로 씀(뒤 날 예측 금지).
 모든 계수(q, qI, μ, a, b, r, ρ)는 학습 사슬에서만, 검증에서는 '사슬 단위 하나 빼기(LOCO)' 로 숨긴 사례의 사슬 밖에서만 추정.

[가짜 평가 사례] 학습 사슬(b1_chains, 진단용 연결 = 같은 동 식별 완벽 가정; F47_229→F47_231→F47_132 는 순수 창 '번호−2' 규칙(6.549)+
 자정 EC 연속(.078, .046)으로 이어 붙임)에서 목표 위치 t(OOF 있음), 앞 앵커 t−k(k ≥ 2), 뒤 앵커 t+m(m ≥ 1), 3 ≤ k+m ≤ 10.
 사이 날은 정답을 숨기고, 예측은 (t−k, t] 의 날만(평가의 '같은 ID 앞 번호'와 같은 방향).
 1차 선택(관측만으로, 평가와 같은 조건): 앞 앵커 23시 ≥ 1.0, 뒤 앵커 0시 ≥ 1.0, 목표 모델 예측 q_t ≥ 0.9.  참고 선택: 목표 정답 하루 평균 ≥ 1.0.
 판정 단위 = 사례(t,k,m) 하루 평균 제곱오차; 대조 = 현재 모델 q_t(DIAG10 OOF sg, 시드별).

[사전 고정 판정 규칙 — 실행 전 고정]
 1차 방법 = KB-RW(선형 무작위 걸음), 1차 묶음 = 관측 선택 & 간격 6~10.
 PASS ⇔ DIAG10 3시드 모두 (i) RMSE(KB-RW) < RMSE(모델), (ii) RMSE(KB-RW) < RMSE(BI1형 β 섞기, LOCO), (iii) 사슬 단위 부호 뒤집기 검정 한쪽 p < .025.
 나머지(RW-log, IRW, 자기 예측만 j=1, 간격 3~5, 정답 선택)는 서술용.
"""
import sys, os, itertools
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
HERE = os.path.dirname(os.path.abspath(__file__)); LOC = os.path.join(HERE, "local")
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
QS = O.groupby(["validator", "seed", "rec"]).sg.mean()
SEEDS = sorted(O.seed.unique())
C0 = pd.read_csv(os.path.join(LOC, "chains_v1.csv"))
# ---- chain 47 front extension (F47_229 -> F47_231 -> F47_132) ----
c47 = int(C0[C0.rec == "F47_132"].chain.iloc[0])
ext = []
for r in ("F47_229", "F47_231"):
    i = idx[r]; ext.append(dict(rec=r, chain=c47, pos=-1, y=float(np.nanmean(EC[i])), e0=float(EC[i, 0]), e23=float(EC[i, 23])))
C0 = C0[~C0.rec.isin(["F47_229", "F47_231"])]   # in case they sat in their own short chain
C0 = pd.concat([pd.DataFrame(ext), C0], ignore_index=True)
C0.loc[C0.rec == "F47_229", "pos"] = -2; C0.loc[C0.rec == "F47_231", "pos"] = -1
C0 = C0.sort_values(["chain", "pos"]).reset_index(drop=True)
C0["pos"] = C0.groupby("chain").cumcount()
CH = {c: G.reset_index(drop=True) for c, G in C0.groupby("chain")}
HICH = [c for c, G in CH.items() if (G.y >= 1).any()]
print("chains", len(CH), "high chains", HICH)

# ================= Gaussian machinery =================
def cov_grid(tau, q, qI, P0=4.0, Pv=0.0):
    s = tau - tau.min()
    S1, S2 = np.meshgrid(s, s, indexing="ij"); mn = np.minimum(S1, S2); mx = np.maximum(S1, S2)
    K = P0 + Pv * S1 * S2 + q * mn + qI * (mn ** 2 * mx / 2 - mn ** 3 / 6)
    return K
def avg_row(tau, center):
    w = np.zeros(len(tau))
    for off, wt in ((-0.5, .25), (0.0, .5), (0.5, .25)):
        w[np.argmin(np.abs(tau - (center + off)))] += wt
    return w
def point_row(tau, t):
    w = np.zeros(len(tau)); w[np.argmin(np.abs(tau - t))] = 1.0; return w
def posterior(tau, mean, K, H, obs, R, wt):
    """target = wt·z; obs = H z + noise(R)."""
    S = H @ K @ H.T + R
    g = np.linalg.solve(S, obs - H @ mean)
    m = wt @ mean + wt @ K @ H.T @ g
    v = wt @ K @ wt - wt @ K @ H.T @ np.linalg.solve(S, H @ K @ wt)
    return float(m), float(v)
SIGA = 0.03   # anchor point/daily-mean observation noise (hourly value vs path)

def build_obs(anchors, preds, tau, tf, par):
    """anchors: list of (kind, day, value) kind in {e0, mean, e23}; preds: list of (day, value). tf = value transform."""
    Hs, ob, rd = [], [], []
    for kind, d, v in anchors:
        if not np.isfinite(v): continue
        if kind == "mean": Hs.append(avg_row(tau, d))
        elif kind == "e0": Hs.append(point_row(tau, d - 0.5))
        else: Hs.append(point_row(tau, d + 0.5))
        ob.append(tf(v)); rd.append(SIGA ** 2 if tf is IDT else (SIGA / max(v, .2)) ** 2)
    na = len(ob)
    pd_ = [(d, v) for d, v in preds if np.isfinite(v)]
    for d, v in pd_:
        Hs.append(par["b"] * avg_row(tau, d)); ob.append(tf(v) - par["a"])
    H = np.array(Hs); ob = np.array(ob)
    R = np.zeros((len(ob), len(ob))); R[np.arange(na), np.arange(na)] = rd
    if pd_:
        dd = np.array([d for d, _ in pd_]); R[na:, na:] = par["r"] * par["rho"] ** np.abs(dd[:, None] - dd[None, :])
    return H, ob, R
IDT = lambda v: v
LOGT = lambda v: np.log(max(v, 0.05))

def estimate(anchors, preds, target_day, par, model):
    days = [d for _, d, _ in anchors] + [d for d, _ in preds] + [target_day]
    lo, hi = min(days) - 1, max(days) + 1
    tau = np.arange(lo - 0.5, hi + 0.5001, 0.5)
    tf = LOGT if model == "rwlog" else IDT
    q, qI, Pv = (par["q"], 0.0, 0.0) if model in ("rw", "rwlog") else (par["q"], par["qI"], 1.0)
    K = cov_grid(tau, q, qI, P0=4.0, Pv=Pv)
    av = [tf(v) for k_, d, v in anchors if np.isfinite(v)]
    m0 = np.mean(av) if av else tf(1.0)
    mean = np.full(len(tau), m0)
    H, ob, R = build_obs(anchors, preds, tau, tf, par)
    m, v = posterior(tau, mean, K, H, ob, R, avg_row(tau, target_day))
    return float(np.exp(m)) if model == "rwlog" else m

# ================= parameter fitting (on a set of chains) =================
_DYN = {}
def fit_dyn(chains, model):
    key = (tuple(sorted(chains)), model)
    if key not in _DYN: _DYN[key] = _fit_dyn(chains, model)
    return _DYN[key]
def _fit_dyn(chains, model):
    """ML grid for q (RW / RWlog) or (q,qI) (IRW) on runs with y >= .8 (length >= 2), obs = e0, mean, e23 per day."""
    tf = LOGT if model == "rwlog" else IDT
    runs = []
    for c in chains:
        G = CH[c]; ok = G.y.values >= 0.8; s = 0
        while s < len(G):
            if not ok[s]: s += 1; continue
            e = s
            while e + 1 < len(G) and ok[e + 1]: e += 1
            if e - s + 1 >= 2: runs.append(G.iloc[s:e + 1])
            s = e + 1
    qg = np.geomspace(0.002, 1.0, 22) if model != "irw" else np.geomspace(0.002, 0.5, 12)
    qIg = [0.0] if model != "irw" else list(np.geomspace(0.001, 0.3, 10))
    best = None
    for q in qg:
        for qI in qIg:
            ll = 0.0
            for G in runs:
                n = len(G); tau = np.arange(-0.5, n - 0.5 + 0.001, 0.5)
                anchors = [(kd, d, (G.iloc[d].y if kd == "mean" else G.iloc[d][kd])) for d in range(n) for kd in ("e0", "mean", "e23")]
                H, ob, R = build_obs(anchors, [], tau, tf, dict(a=0, b=1, r=1, rho=0))
                K = cov_grid(tau, q, qI, P0=4.0, Pv=1.0 if model == "irw" else 0.0)
                S = H @ K @ H.T + R; mu = np.full(len(ob), np.mean(ob))
                sgn, ld = np.linalg.slogdet(S); dlt = ob - mu
                ll += -0.5 * (ld + dlt @ np.linalg.solve(S, dlt))
            if best is None or ll > best[0]: best = (ll, q, qI)
    return dict(q=best[1], qI=best[2])
def fit_meas(chains, seed, model, val="DIAG10"):
    tf = LOGT if model == "rwlog" else IDT
    xs, ps, pairs = [], [], []
    for c in chains:
        G = CH[c]; prev = None
        for r in G.itertuples():
            p = QS.get((val, seed, r.rec), np.nan)
            if np.isfinite(p) and r.y >= 0.8:
                xs.append(tf(r.y)); ps.append(tf(p)); cur = (r.pos, len(xs) - 1)
                if prev is not None and prev[0] == r.pos - 1: pairs.append((prev[1], cur[1]))
                prev = cur
            else:
                prev = None
    xs, ps = np.array(xs), np.array(ps)
    b, a = np.polyfit(xs, ps, 1); res = ps - (a + b * xs); r = float(res.var())
    rho = float(np.corrcoef([res[i] for i, j in pairs], [res[j] for i, j in pairs])[0, 1]) if len(pairs) > 3 else 0.0
    return dict(a=float(a), b=float(b), r=r, rho=max(0.0, min(rho, 0.95)), n=len(xs), npair=len(pairs))

# ================= pseudo cases =================
def make_cases():
    cases = []
    for c, G in CH.items():
        n = len(G)
        for t in range(n):
            for k in range(2, 9):
                for m in range(1, 9):
                    if t - k < 0 or t + m >= n or not (3 <= k + m <= 10): continue
                    if c in HICH: cases.append((c, t, k, m))
    return cases
CASES = make_cases()
def anchors_for(G, A, B, extra=True):
    an = []
    for d, kinds in ((A, ("e0", "mean", "e23")), (B, ("e0", "mean", "e23"))):
        if d is None: continue
        r = G.iloc[d]
        for kd in kinds: an.append((kd, d, r.y if kd == "mean" else r[kd]))
    if extra:
        if A - 1 >= 0: an.append(("mean", A - 1, G.iloc[A - 1].y))
        if B is not None and B + 1 < len(G): an.append(("mean", B + 1, G.iloc[B + 1].y))
    return an

def run_seed(seed, val_target="DIAG10"):
    rows = []
    chains_all = list(CH.keys())
    cache = {}
    for c in chains_all:
        G = CH[c]
        if not any(c == cc for cc, *_ in CASES): continue
        others = [x for x in chains_all if x != c]
        key = c
        par = {}
        for model in ("rw", "rwlog", "irw"):
            d = fit_dyn([x for x in others if x in HICH], model); mm = fit_meas(others, seed, model)
            par[model] = {**d, **mm}
        # calibration-only (6.177-like) and BI1 beta are fitted below from other chains' cases
        cache[c] = par
    # calibration (y on q, q>=.9) LOCO
    def calib(others):
        xs, ys = [], []
        for x in others:
            for r in CH[x].itertuples():
                p = QS.get(("DIAG10", seed, r.rec), np.nan)
                if np.isfinite(p) and p >= 0.9: xs.append(p); ys.append(r.y)
        b, a = np.polyfit(xs, ys, 1); return a, b
    for (c, t, k, m) in CASES:
        G = CH[c]; tr = G.iloc[t]
        qt = QS.get((val_target, seed, tr.rec), np.nan)
        if not np.isfinite(qt): continue
        A, B = t - k, t + m
        eA, eB = G.iloc[A].e23, G.iloc[B].e0
        par = cache[c]
        preds = [(d, QS.get(("DIAG10", seed, G.iloc[d].rec), np.nan)) for d in range(A + 1, t)] + [(t, qt)]
        an = anchors_for(G, A, B)
        est = {}
        for model in ("rw", "rwlog", "irw"):
            est["KB_" + model] = estimate(an, preds, t, par[model], model)
        est["KB_rw_j1"] = estimate(an, [(t, qt)], t, par["rw"], "rw")
        est["KB_rw_left"] = estimate(anchors_for(G, A, None), preds, t, par["rw"], "rw")
        est["INT"] = estimate(an, [], t, par["rw"], "rw")      # bridge mean without preds (≈ interpolation)
        I = eA + (eB - eA) * (t - (A + 0.5)) / ((B - 0.5) - (A + 0.5))
        rows.append(dict(seed=seed, chain=c, t=t, k=k, m=m, G=k + m, rec=tr.rec, y=tr.y, q=qt, I=I, eA=eA, eB=eB, **est))
    D = pd.DataFrame(rows)
    # LOCO calibration & BI1-type beta (by gap group, obs-selected cases of other chains)
    D["grp"] = pd.cut(D.G, [2, 5, 7, 10], labels=["3-5", "6-7", "8-10"])
    D["sel"] = (D.eA >= 1) & (D.eB >= 1) & (D.q >= 0.9)
    D["CAL"] = np.nan; D["BI1"] = np.nan
    for c in D.chain.unique():
        oth = [x for x in CH if x != c]
        a, b = calib(oth); D.loc[D.chain == c, "CAL"] = a + b * D.loc[D.chain == c, "q"]
        Oo = D[(D.chain != c) & D.sel]
        for g in ["3-5", "6-7", "8-10"]:
            S = Oo[Oo.grp == g]; x = (S.I - S.q).values; r_ = (S.y - S.q).values
            beta = float((x * r_).sum() / (x ** 2).sum()) if len(S) >= 3 else 0.0
            msk = (D.chain == c) & (D.grp == g)
            D.loc[msk, "BI1"] = D.loc[msk, "q"] + beta * (D.loc[msk, "I"] - D.loc[msk, "q"])
    return D, cache

METH = ["q", "CAL", "INT", "BI1", "KB_rw", "KB_rwlog", "KB_irw", "KB_rw_j1", "KB_rw_left"]
def signflip(D, col):
    g = D.groupby("chain").apply(lambda S: ((S[col] - S.y) ** 2 - (S.q - S.y) ** 2).mean())
    v = g.values; obs = v.sum(); cnt = 0; tot = 0
    for sg in itertools.product([1, -1], repeat=len(v)):
        tot += 1; cnt += (np.dot(sg, np.abs(v)) <= obs + 1e-12)
    return cnt / tot, len(v), g
pd.set_option("display.width", 250)
ALL = []
verdict = []
for seed in SEEDS:
    D, cache = run_seed(seed)
    ALL.append(D)
    print(f"\n===== seed {seed}  (cases {len(D)}, obs-selected {D.sel.sum()}) =====")
    if seed == SEEDS[0]:
        for c in HICH:
            if c in cache:
                p = cache[c]; print(f"  LOCO params w/o chain {c}: rw q {p['rw']['q']:.3f} | rwlog q {p['rwlog']['q']:.4f} | irw q {p['irw']['q']:.3f} qI {p['irw']['qI']:.4f} | meas a {p['rw']['a']:.3f} b {p['rw']['b']:.3f} r {p['rw']['r']:.4f} rho {p['rw']['rho']:.2f} (n {p['rw']['n']}, pairs {p['rw']['npair']})")
    for nm, S0 in (("OBS-selected", D[D.sel]), ("TRUTH-selected y>=1", D[D.y >= 1])):
        for g in ["3-5", "6-7", "8-10", "6-10"]:
            S = S0[S0.G.between(6, 10)] if g == "6-10" else S0[S0.grp == g]
            if len(S) == 0: continue
            line = " ".join(f"{mm} {np.sqrt(((S[mm] - S.y) ** 2).mean()):.3f}" for mm in METH)
            imp = (np.abs(S.KB_rw - S.y) < np.abs(S.q - S.y)).mean()
            print(f"  {nm:20s} G {g:5s} n {len(S):3d} tgt {S.rec.nunique():2d} ch {S.chain.nunique()} | {line} | KB_rw better {imp:.2f}")
    S = D[D.sel & D.G.between(6, 10)]
    if len(S):
        p, nch, g = signflip(S, "KB_rw")
        r_kb = np.sqrt(((S.KB_rw - S.y) ** 2).mean()); r_q = np.sqrt(((S.q - S.y) ** 2).mean()); r_bi = np.sqrt(((S.BI1 - S.y) ** 2).mean())
        verdict.append((r_kb < r_q) and (r_kb < r_bi) and (p < 0.025))
        print(f"  PRIMARY seed {seed}: RMSE KB_rw {r_kb:.4f} vs model {r_q:.4f} vs BI1 {r_bi:.4f} | sign-flip p {p:.3f} (chains {nch}) | per-chain dMSE {g.round(4).to_dict()}")
        for mm in ("KB_rwlog", "KB_irw", "KB_rw_j1"):
            pp, _, _ = signflip(S, mm); print(f"     {mm}: sign-flip p {pp:.3f}")
    else:
        verdict.append(False)
print("\nPRIMARY VERDICT:", "PASS" if all(verdict) else "FAIL", verdict)
A = pd.concat(ALL); A.to_csv(os.path.join(LOC, "b2_cases_v1.csv"), index=False)
# per-target summary (obs-selected, G 6-10), averaged over seeds and geometries
S = A[A.sel & A.G.between(6, 10)]
if len(S):
    T = S.groupby("rec")[["y", "q", "I", "BI1", "KB_rw", "KB_rwlog", "KB_irw"]].mean().round(3); T["n"] = S.groupby("rec").size()
    print("\nper target (obs-selected, G 6-10, mean over seeds & geometries):\n", T.to_string())
