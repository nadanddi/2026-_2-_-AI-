# -*- coding: utf-8 -*-
"""B3 (2026-10-13 집 클로드, hx8/bridge) — b2(사전 고정 판정 FAIL) 이어서: (A) FAIL 원인 진단(b2_cases_v1.csv 재사용, 재적합 없음),
(B) 학습 전용 사슬 구조(라벨 교대·'번호 −2' 날짜 매기기) 확인, (C) 평가 8기록 다리 추정치(서술 — 채택 아님, b2 FAIL).

[평가 날짜 매기기 — 학습 + 같은 ID 번호만, DM11 안 씀]
 F13: 학습 F13_217(학습 달력 day 10) → F13_219 결번(지워진 날) → F13_221, 223, 225 = +2, +3, +4 일.
 F47: 학습 F47_211(F13_217 과 같은 seg·day 인지 확인) → 213, 215(학습) → 217 결번 → 219, 221, 223, 225, 227 = +4..+8 일 → F47_229(학습) = +9, F47_231 = +10.
 (상대 날짜 '80일' = F13_217.)
[두 가설 — 둘 다 계산, 고르지 않음]
 H-SW(ID 무관 앵커): 고EC 예측 기록(자기 예측 ≥ .9)의 앞/뒤 앵커 = 날짜상 가장 가까운 학습 고EC(하루 평균 ≥ 1) 기록. → 앞 F13_217(80), 뒤 F47_229(89)+F47_231(90).
        관측 예측 = 자기 + 같은 ID 앞 번호 중 −2 사슬 위이고 예측 ≥ .9 인 것(F47_219 .66 은 제외).
 H-ST(같은 ID −2 사슬 고수): F13 = 앞 F13_217, 뒤 F13_231(87)·F13_233(88);  F47 = 앞 F47_215(82)·F47_213(81), 뒤 F47_229·F47_231, F47_219 예측 포함.
 F13_237(90): 앞 F13_233(88) −2 사슬, 관측 F13_235(89) 예측 + 자기; 뒤 앵커 없음(왼쪽만).
 평가 '모델 예측' = 13회차 submission_22 하루 평균(이 8기록은 앵커 없음 = 9회차 기본 모델). 측정 모델(a,b,r,ρ)은 학습 OOF(DIAG10 CUR sg)에서 → 불일치 가능성 있음(주의).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
HERE = os.path.dirname(os.path.abspath(__file__)); LOC = os.path.join(HERE, "local")
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
pd.set_option("display.width", 250)
E0 = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); D0 = dict(zip(E0.rec, E0.day)); G0 = dict(zip(E0.rec, E0.seg))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
QS = O.groupby(["validator", "seed", "rec"]).sg.mean(); SEEDS = sorted(O.seed.unique())
S13 = pd.read_csv(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)", "submission_22.csv")); S13["rec"] = S13.row_id.str[:7]
P13 = S13.groupby("rec").sub_ec.mean()

# ===================== (A) b2 FAIL 진단 =====================
A = pd.read_csv(os.path.join(LOC, "b2_cases_v1.csv"))
print("=== (A) b2 사례 재사용: 행", len(A), "시드", A.seed.nunique())
rm = lambda S, c: np.sqrt(((S[c] - S.y) ** 2).mean())
M = ["q", "INT", "BI1", "KB_rw", "CAL"]
for nm, S0 in (("OBS-sel", A[A.sel]), ("TRUTH y>=1", A[A.y >= 1])):
    S9 = S0[S0.G == 9]
    print(f"\n[{nm}] 평가와 같은 간격 G=9, k(앞 앵커 거리)별 (시드 3개 합친 RMSE, n=사례/시드)")
    for k, S in S9.groupby("k"):
        print(f"  k{k} m{9-k}: n {len(S)//3:2d} ch {S.chain.nunique()} tgt {S.rec.nunique():2d} | " + " ".join(f"{c} {rm(S, c):.3f}" for c in M))
    S = S0[S0.G.between(6, 10)]
    dk, dy = (S.KB_rw - S.q).values, (S.y - S.q).values
    sl = (dk * dy).sum() / (dk ** 2).sum()
    print(f"  G6-10 KB 보정 방향: corr(KB−q, y−q) {np.corrcoef(dk, dy)[0,1]:+.3f}, 기울기 {sl:+.3f} (1 이면 보정 크기 맞음, 0 이면 잡음) | 평균 KB−q {dk.mean():+.3f}, 평균 y−q {dy.mean():+.3f}")
    di, _ = (S.INT - S.q).values, None
    print(f"  G6-10 보간 방향: corr(INT−q, y−q) {np.corrcoef(di, dy)[0,1]:+.3f}, 기울기 {(di*dy).sum()/(di**2).sum():+.3f}")
    for c, Sc in S.groupby("chain"):
        print(f"    chain {c:2d} n {len(Sc)//3:3d}: 평균 y {Sc.y.mean():.2f} q {Sc.q.mean():.2f} INT {Sc.INT.mean():.2f} KB {Sc.KB_rw.mean():.2f} | dMSE(KB−q) {((Sc.KB_rw-Sc.y)**2-(Sc.q-Sc.y)**2).mean():+.4f}")

# ---- 사슬(b2 와 같은 구성) ----
C0 = pd.read_csv(os.path.join(LOC, "chains_v1.csv"))
c47 = int(C0[C0.rec == "F47_132"].chain.iloc[0]); ext = []
for r in ("F47_229", "F47_231"):
    i = idx[r]; ext.append(dict(rec=r, chain=c47, pos=-1, y=float(np.nanmean(EC[i])), e0=float(EC[i, 0]), e23=float(EC[i, 23])))
C0 = C0[~C0.rec.isin(["F47_229", "F47_231"])]; C0 = pd.concat([pd.DataFrame(ext), C0], ignore_index=True)
C0.loc[C0.rec == "F47_229", "pos"] = -2; C0.loc[C0.rec == "F47_231", "pos"] = -1
C0 = C0.sort_values(["chain", "pos"]).reset_index(drop=True); C0["pos"] = C0.groupby("chain").cumcount()
CH = {c: G.reset_index(drop=True) for c, G in C0.groupby("chain")}
HICH = [c for c, G in CH.items() if (G.y >= 1).any()]

# 측정 모델 잔차의 사슬별 평균(공유 측정 모델 가정 점검)
print("\n[측정 모델 점검] 모든 사슬로 p = a + b·y 적합(DIAG10, y ≥ .8) 후 고EC 사슬별 평균 잔차")
for seed in SEEDS[:1]:
    xs, ps, cs = [], [], []
    for c in CH:
        for r in CH[c].itertuples():
            p = QS.get(("DIAG10", seed, r.rec), np.nan)
            if np.isfinite(p) and r.y >= 0.8: xs.append(r.y); ps.append(p); cs.append(c)
    xs, ps, cs = np.array(xs), np.array(ps), np.array(cs); b, a = np.polyfit(xs, ps, 1); res = ps - a - b * xs
    print(f"  seed {seed}: a {a:.3f} b {b:.3f} resid sd {res.std():.3f}, corr(p,y) {np.corrcoef(xs, ps)[0,1]:.3f} (n {len(xs)})")
    for c in sorted(set(cs)):
        mk = cs == c; print(f"    chain {c:2d} n {mk.sum():2d}: 평균 잔차 {res[mk].mean():+.3f}, y 범위 {xs[mk].min():.2f}~{xs[mk].max():.2f}")
    # 사슬 평균을 빼면 남는 상관 = 사슬 안에서 p 가 y 를 따라가나
    xw = xs - np.array([xs[cs == c].mean() for c in cs]); pw = ps - np.array([ps[cs == c].mean() for c in cs])
    print(f"  사슬 안(within) corr(p,y) {np.corrcoef(xw, pw)[0,1]:+.3f}, 기울기 {np.polyfit(xw, pw, 1)[0]:.3f}")

# ===================== (B) 학습 전용 구조 =====================
print("\n=== (B) 학습 사슬(진단용 연결) 연속 날 라벨(F13/F47) 교대 비율")
sw = []
for c, G in CH.items():
    for i in range(1, len(G)):
        a_, b_ = G.rec.iloc[i - 1], G.rec.iloc[i]
        sw.append(dict(a=a_, b=b_, sw=a_[:3] != b_[:3], seg=G0.get(b_, -1), hi=G.y.iloc[i] >= 1))
SW = pd.DataFrame(sw)
print(f"  전체 연결 {len(SW)}: 라벨 바뀜 {SW.sw.mean():.2f} | 고EC 날(b ≥ 1) {SW[SW.hi].sw.mean():.2f} (n {SW.hi.sum()})")
print("  seg 별 라벨 바뀜:", SW.groupby("seg").sw.agg(['mean', 'size']).round(2).T.to_dict())
for r in ("F13_127", "F13_217", "F47_211", "F47_213", "F47_215", "F13_231", "F13_233", "F47_229", "F47_231"):
    print(f"  {r}: seg {G0.get(r)} day {D0.get(r)} y {np.nanmean(EC[idx[r]]):.3f} e0 {EC[idx[r],0]:.3f} e23 {EC[idx[r],23]:.3f}")
print("  F13_127→F13_217 같은 seg·연속 날?", G0.get("F13_127") == G0.get("F13_217") and D0.get("F13_217") == D0.get("F13_127", -9) + 1)
print("  F13_217·F47_211 같은 seg·같은 날?", G0.get("F13_217") == G0.get("F47_211") and D0.get("F13_217") == D0.get("F47_211"))

# ===================== (C) 평가 추정 =====================
SIGA = 0.03
def cov_grid(tau, q, qI, P0=4.0, Pv=0.0):
    s = tau - tau.min(); S1, S2 = np.meshgrid(s, s, indexing="ij"); mn = np.minimum(S1, S2); mx = np.maximum(S1, S2)
    return P0 + Pv * S1 * S2 + q * mn + qI * (mn ** 2 * mx / 2 - mn ** 3 / 6)
def avg_row(tau, center):
    w = np.zeros(len(tau))
    for off, wt in ((-0.5, .25), (0.0, .5), (0.5, .25)): w[np.argmin(np.abs(tau - (center + off)))] += wt
    return w
def point_row(tau, t):
    w = np.zeros(len(tau)); w[np.argmin(np.abs(tau - t))] = 1.0; return w
IDT = lambda v: v
LOGT = lambda v: np.log(max(v, 0.05))
def build_obs(anchors, preds, tau, tf, par):
    Hs, ob, rd = [], [], []
    for kind, d, v in anchors:
        if not np.isfinite(v): continue
        Hs.append(avg_row(tau, d) if kind == "mean" else point_row(tau, d - 0.5) if kind == "e0" else point_row(tau, d + 0.5))
        ob.append(tf(v)); rd.append(SIGA ** 2 if tf is IDT else (SIGA / max(v, .2)) ** 2)
    na = len(ob); pd_ = [(d, v) for d, v in preds if np.isfinite(v)]
    for d, v in pd_: Hs.append(par["b"] * avg_row(tau, d)); ob.append(tf(v) - par["a"])
    H = np.array(Hs); ob = np.array(ob); R = np.zeros((len(ob), len(ob))); R[np.arange(na), np.arange(na)] = rd
    if pd_:
        dd = np.array([d for d, _ in pd_]); R[na:, na:] = par["r"] * par["rho"] ** np.abs(dd[:, None] - dd[None, :])
    return H, ob, R
def estimate(anchors, preds, target_day, par, model):
    days = [d for _, d, _ in anchors] + [d for d, _ in preds] + [target_day]
    tau = np.arange(min(days) - 1.5, max(days) + 1.5001, 0.5)
    tf = LOGT if model == "rwlog" else IDT
    q, qI, Pv = (par["q"], 0.0, 0.0) if model in ("rw", "rwlog") else (par["q"], par["qI"], 1.0)
    K = cov_grid(tau, q, qI, P0=4.0, Pv=Pv)
    av = [tf(v) for _, _, v in anchors if np.isfinite(v)]; mean = np.full(len(tau), np.mean(av) if av else tf(1.0))
    H, ob, R = build_obs(anchors, preds, tau, tf, par); wt = avg_row(tau, target_day)
    S = H @ K @ H.T + R; g = np.linalg.solve(S, ob - H @ mean)
    m = wt @ mean + wt @ K @ H.T @ g; v = wt @ K @ wt - wt @ K @ H.T @ np.linalg.solve(S, H @ K @ wt)
    return (float(np.exp(m)) if model == "rwlog" else float(m)), float(np.sqrt(max(v, 0)))
def fit_dyn(chains, model):
    tf = LOGT if model == "rwlog" else IDT; runs = []
    for c in chains:
        G = CH[c]; ok = G.y.values >= 0.8; s = 0
        while s < len(G):
            if not ok[s]: s += 1; continue
            e = s
            while e + 1 < len(G) and ok[e + 1]: e += 1
            if e - s + 1 >= 2: runs.append(G.iloc[s:e + 1])
            s = e + 1
    qg = np.geomspace(0.002, 1.0, 22) if model != "irw" else np.geomspace(0.002, 0.5, 12)
    qIg = [0.0] if model != "irw" else list(np.geomspace(0.001, 0.3, 10)); best = None
    for q in qg:
        for qI in qIg:
            ll = 0.0
            for G in runs:
                n = len(G); tau = np.arange(-0.5, n - 0.5 + 0.001, 0.5)
                an = [(kd, d, (G.iloc[d].y if kd == "mean" else G.iloc[d][kd])) for d in range(n) for kd in ("e0", "mean", "e23")]
                H, ob, R = build_obs(an, [], tau, tf, dict(a=0, b=1, r=1, rho=0))
                K = cov_grid(tau, q, qI, P0=4.0, Pv=1.0 if model == "irw" else 0.0); S = H @ K @ H.T + R
                dl = ob - np.mean(ob); ll += -0.5 * (np.linalg.slogdet(S)[1] + dl @ np.linalg.solve(S, dl))
            if best is None or ll > best[0]: best = (ll, q, qI)
    return dict(q=best[1], qI=best[2])
def fit_meas(chains, seed, model):
    tf = LOGT if model == "rwlog" else IDT; xs, ps, pairs = [], [], []
    for c in chains:
        prev = None
        for r in CH[c].itertuples():
            p = QS.get(("DIAG10", seed, r.rec), np.nan)
            if np.isfinite(p) and r.y >= 0.8:
                xs.append(tf(r.y)); ps.append(tf(p)); cur = (r.pos, len(xs) - 1)
                if prev is not None and prev[0] == r.pos - 1: pairs.append((prev[1], cur[1]))
                prev = cur
            else: prev = None
    xs, ps = np.array(xs), np.array(ps); b, a = np.polyfit(xs, ps, 1); res = ps - (a + b * xs)
    rho = float(np.corrcoef([res[i] for i, j in pairs], [res[j] for i, j in pairs])[0, 1]) if len(pairs) > 3 else 0.0
    return dict(a=float(a), b=float(b), r=float(res.var()), rho=max(0.0, min(rho, 0.95)))

yv = lambda r: float(np.nanmean(EC[idx[r]]))
def anc(r, d, kinds=("e0", "mean", "e23")):
    i = idx[r]; return [(k, d, yv(r) if k == "mean" else float(EC[i, 0] if k == "e0" else EC[i, 23])) for k in kinds]
DAY = {"F13_221": 82, "F13_223": 83, "F13_225": 84, "F47_219": 84, "F47_221": 85, "F47_223": 86, "F47_225": 87, "F47_227": 88,
       "F13_235": 89, "F13_237": 90}
F13H, F47H = ["F13_221", "F13_223", "F13_225"], ["F47_221", "F47_223", "F47_225", "F47_227"]
SW_F = [("mean", 79, yv("F13_127"))] + anc("F13_217", 80); SW_B = anc("F47_229", 89) + [("mean", 90, yv("F47_231"))]
def geo(target, hyp):
    """returns anchors, preds(day,value), eA(front 23시), eB(back 0시), A_day, B_day"""
    if target == "F13_237":
        an = [("mean", 87, yv("F13_231"))] + anc("F13_233", 88); pr = [(89, P13["F13_235"]), (90, P13["F13_237"])]
        return an, pr, EC[idx["F13_233"], 23], np.nan, 88, np.nan
    fam = F13H if target in F13H else F47H; t = DAY[target]
    prv = [(DAY[r], P13[r]) for r in fam if DAY[r] <= t]
    if hyp == "SW":
        return SW_F + SW_B, prv, EC[idx["F13_217"], 23], EC[idx["F47_229"], 0], 80, 89
    if target in F13H:
        an = SW_F + anc("F13_231", 87) + [("mean", 88, yv("F13_233"))]
        return an, prv, EC[idx["F13_217"], 23], EC[idx["F13_231"], 0], 80, 87
    an = [("mean", 81, yv("F47_213"))] + anc("F47_215", 82) + SW_B
    return an, [(84, P13["F47_219"])] + prv, EC[idx["F47_215"], 23], EC[idx["F47_229"], 0], 82, 89

print("\n=== (C) 평가 추정 (모든 학습 사슬로 계수 적합) ===")
DYN = {m: fit_dyn(HICH, m) for m in ("rw", "rwlog", "irw")}
print("  동역학:", {m: {k: round(v, 4) for k, v in d.items()} for m, d in DYN.items()})
# BI1 β (간격 묶음 8-10, OBS-sel, 모든 사슬) / CAL — b2 사례에서
beta = {}; cal = {}
for seed in SEEDS:
    S = A[(A.seed == seed) & A.sel & (A.grp == "8-10")]; x = (S.I - S.q).values; r_ = (S.y - S.q).values
    beta[seed] = float((x * r_).sum() / (x ** 2).sum())
    xs, ys = [], []
    for c in CH:
        for r in CH[c].itertuples():
            p = QS.get(("DIAG10", seed, r.rec), np.nan)
            if np.isfinite(p) and p >= 0.9: xs.append(p); ys.append(r.y)
    bb, aa = np.polyfit(xs, ys, 1); cal[seed] = (aa, bb)
print("  BI1 β(8-10, OBS-sel, 사례 수 작음):", {s: round(b, 3) for s, b in beta.items()}, "| CAL(a,b):", {s: tuple(round(v, 3) for v in c) for s, c in cal.items()})
rows = []
for target in F13H + F47H + ["F13_237"]:
    for hyp in (("SW", "ST") if target != "F13_237" else ("LEFT",)):
        an, pr, eA, eB, Ad, Bd = geo(target, hyp); t = DAY[target]; q13 = float(P13[target])
        I = eA + (eB - eA) * (t - (Ad + 0.5)) / ((Bd - 0.5) - (Ad + 0.5)) if np.isfinite(eB) else np.nan
        out = dict(rec=target, hyp=hyp, day=t, k=t - Ad, m=(Bd - t) if np.isfinite(Bd) else np.nan, p13=q13, I=I)
        for model in ("rw", "rwlog", "irw"):
            ms, ss = [], []
            for seed in SEEDS:
                par = {**DYN[model], **fit_meas(list(CH), seed, model)}
                m_, s_ = estimate(an, pr, t, par, model); ms.append(m_); ss.append(s_)
            out["KB_" + model] = np.mean(ms)
            if model == "rw": out["KB_rw_sd"] = np.mean(ss)
        par = {**DYN["rw"], **fit_meas(list(CH), SEEDS[0], "rw")}
        out["INT"] = estimate(an, [], t, par, "rw")[0]
        out["INT_sd"] = estimate(an, [], t, par, "rw")[1]
        out["BI1"] = np.mean([q13 + beta[s] * (I - q13) for s in SEEDS]) if np.isfinite(I) else np.nan
        out["CAL"] = np.mean([cal[s][0] + cal[s][1] * q13 for s in SEEDS])
        rows.append(out)
R = pd.DataFrame(rows)
print(R.round(3).to_string())
R.round(4).to_csv(os.path.join(HERE, "b3_eval_estimates_v1.csv"), index=False)
for hyp in ("SW", "ST"):
    S = R[R.hyp == hyp]
    print(f"  {hyp}: 7기록 평균 변화(KB_rw − 13회차) {np.mean(S.KB_rw - S.p13):+.3f}, RMS {np.sqrt(np.mean((S.KB_rw - S.p13) ** 2)):.3f} | INT − 13회차 평균 {np.mean(S.INT - S.p13):+.3f}")
print("저장:", os.path.join(HERE, "b3_eval_estimates_v1.csv"))
