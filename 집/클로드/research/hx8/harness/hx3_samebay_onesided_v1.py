# -*- coding: utf-8 -*-
"""HX3 harness (2026-10-13 집 클로드; HX1·HX2 이어서 — 이전 에이전트가 사용량 한도로 멈춘 뒤 완성 단계).  진단·서술(채택 판정 아님, 제출 아님).
HX1: 같은 동 확실 연결(cert, 학습끼리) 사슬은 짧아 '양쪽 간격 9일' 사례 0개.  HX2: 날짜 앵커(그 날짜 고EC 학습 기록, 동 무관)는
     다른 동 수준이 섞여 k=1 에서도 모델보다 나쁨(+26%) → 평가 조건(같은 동 앵커)과 다름.
HX3 의 평가 매핑(사전 고정, 규정 안전): 평가 입력·DM11 을 쓰지 않고 ① 번호 구조 6.549(순수 2차 창: '번호 −2 = 같은 동 전날')와
  ② 학습 전용 달력(F13_216/217 = seg1 day10, F13_218 = day11, F47_215/216 = day12)만으로 정한 '한쪽 앵커' 조건:
    F13_221 앞 2일(F13_217 23시), F13_223 앞 3일, F13_225 앞 4일(84일 F13 3기록이라 구조 불확실), 뒤 앵커 미상(먼 거리);
    F47_227 뒤 1일(F47_229 0시), F47_225 뒤 2일, F47_223 뒤 3일, F47_221 뒤 4일, 앞 앵커 미상;  F13_237 같은 동 앵커 없음 → E5(같은 날짜 학습 평균).
  DM11 양쪽 구도(간격 9일)는 서술 대조만(같은 동 사례 0개라 검증 불가 — HX1·HX2 숫자 재사용).
사례(가짜 평가): 대상 r(DIAG10 OOF 있음) 의 같은 동 사슬(확실 연결 = 주 판정; 보조 = 확실 + '학습 전용 달력상 같은 seg·하루 차' 비확실 연결)에서
  k 단계 앞 기록 a(23시 A), m 단계 뒤 기록 b(0시 B).  사이 날 정답은 안 씀(숨김).  한쪽만(prev-only, next-only)·양쪽 사례.
대상 집합(고정): Sq = 시드 평균 OOF q ≥ 1.0 (주, 평가처럼 '예측이 높아' 고름), Sy = 정답 하루 평균 ≥ 1.0 (보조, 선택 편향).
추정기(하루 평균):
  E0 q | E1 가까운 앵커 그대로 | E2 앞뒤 선형 보간(양쪽만) | E3 a + b·q (학습 q ≥ .9 기록, 성분 밖) |
  E4 q + w(쪽, 거리 구간)·(앵커 − q) (성분 밖 최소제곱, [0,1]) | E4p q + w1·exp(−(d−1)/τ)·(앵커 − q) (쪽 합침, 격자 적합, 성분 밖) |
  E8 q + ρ(쪽, 거리 구간)·(앵커 기록의 OOF 잔차 = 앵커 하루 평균 정답 − 앵커 OOF q) (6.212·6.597 계열을 고EC·거리별로 다시 잼) |
  E5 같은 날짜(외기 70% 이상 동일) 학습 기록 하루 평균의 평균 (기록 단위).
거리 구간: 1, 2, 3-4, 5+.  교차적합: 주 = 확실 연결 성분 하나 빼기, 감도 = 학습 전용 달력 seg 하나 빼기.
판독 규칙(실행 전 고정): 칸마다 E0 보다 RMSE 가 낮고 (i) 세 시드 모두 E0 보다 낮고 (ii) 성분 부트스트랩 P(ΔRMSE ≥ 0) < .025/5 = .005 인
  추정기 중 RMSE 최소를 '그 칸 승자'로, 없으면 'E0 유지'.  평가 기록 추정치 = 그 칸 승자(계수는 전체 사례로 적합), 불확실성 = 그 칸 사례 RMSE(n 명시).
감도: 대상의 사슬 이웃(±1)이 모두 같은 DIAG10 폴드(= 모델도 못 봄)인 사례만 / 모델 크기 q < 1.5 vs ≥ 1.5.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import numpy as np, pandas as pd
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_v1"); os.makedirs(OUT, exist_ok=True)
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rng = np.random.default_rng(20261013)
KMAX = 9; NBOOT = 2000; ALPHA = 0.025 / 5
BINS = ("1", "2", "3-4", "5+")
dbin = lambda d: "1" if d <= 1 else ("2" if d <= 2 else ("3-4" if d <= 4 else "5+"))
# ---------------- links (training only) ----------------
TL = pd.read_pickle(os.path.join(S3, "truth.pkl"))
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); SD = {r: (int(s), int(d)) for r, s, d in zip(CAL.rec, CAL.seg, CAL.day)}
def make_links(ext):
    pre, cnt = {}, {}
    for a, b, c in zip(TL.a1.astype(int), TL.b.astype(int), TL.cert.astype(bool)):
        if not (TRAIN[a] and TRAIN[b]): continue
        ok = bool(c)
        if ext and not ok:
            ra, rb = rid(a), rid(b)
            ok = ra in SD and rb in SD and SD[ra][0] == SD[rb][0] and SD[rb][1] - SD[ra][1] == 1
        if ok: pre[b] = a; cnt[a] = cnt.get(a, 0) + 1
    pre = {b: a for b, a in pre.items() if cnt[a] == 1}          # no branching
    suc = {a: b for b, a in pre.items()}
    par = {i: i for i in range(N)}
    def fd(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for b, a in pre.items(): par[fd(a)] = fd(b)
    return pre, suc, np.array([fd(i) for i in range(N)])
def walk(r, k, M):
    x = r
    for _ in range(k):
        x = M.get(x)
        if x is None: return None
    return x
# ---------------- OOF ----------------
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
QS = O.groupby(["rec", "seed"]).sg.mean().unstack(); SEEDS = list(QS.columns); QS["mean"] = QS[SEEDS].mean(1)
FOLD = O.groupby("rec").fold.first()
YM = {i: float(np.nanmean(EC[i])) for i in np.where(TRAIN)[0]}
Xf = X.reshape(N, -1); TRIDX = np.where(TRAIN)[0]
def same_date(i):
    eq = (np.abs(Xf[TRIDX] - Xf[i]) < 1e-6).mean(1)
    return [int(j) for j in TRIDX[eq > 0.7] if j != i]
# ---------------- cases ----------------
def build(sel, pre, suc, COMP):
    rows = []
    for r in sel:
        i = idx[r]
        p1, n1 = pre.get(i), suc.get(i)
        nbh = all((FOLD.get(rid(j)) == FOLD[r]) for j in (p1, n1) if j is not None and rid(j) in FOLD.index)
        P = [(None, None)] + [(k, walk(i, k, pre)) for k in range(1, KMAX + 1)]
        Nn = [(None, None)] + [(m, walk(i, m, suc)) for m in range(1, KMAX + 1)]
        for k, a in P:
            if k is not None and a is None: continue
            for m, b in Nn:
                if m is not None and b is None: continue
                if k is None and m is None: continue
                rows.append(dict(rec=r, comp=int(COMP[i]), seg=SD.get(r, (-1, 0))[0], y=YM[i], k=k, m=m,
                                 A=float(EC[a, 23]) if a is not None else np.nan, B=float(EC[b, 0]) if b is not None else np.nan,
                                 ra=rid(a) if a is not None else None, rb=rid(b) if b is not None else None, nbh=nbh))
    C = pd.DataFrame(rows)
    for c in ("k", "m"): C[c] = C[c].astype(float)
    C["side"] = np.where(C.k.notna() & C.m.notna(), "two", np.where(C.k.notna(), "prev", "next"))
    C["d"] = np.fmin(C.k.fillna(99), C.m.fillna(99))
    C["db"] = [dbin(d) for d in C.d]
    C["g"] = C.k + C.m
    C["anc"] = np.where(C.side == "prev", C.A, np.where(C.side == "next", C.B,
                        np.where(C.k < C.m, C.A, np.where(C.m < C.k, C.B, (C.A + C.B) / 2))))
    C["ancr"] = np.where(C.side == "prev", C.ra, np.where(C.side == "next", C.rb, np.where(C.k <= C.m, C.ra, C.rb)))
    C["aside"] = np.where(C.side == "two", np.where(C.k <= C.m, "prev", "next"), C.side)
    return C
def ls_w(x, r):
    x = np.asarray(x, float); r = np.asarray(r, float); ok = np.isfinite(x) & np.isfinite(r); x, r = x[ok], r[ok]
    d = float((x * x).sum())
    return float(np.clip((x * r).sum() / d, 0, 1)) if d > 0 else 0.0
TAUS = (0.5, 1.0, 2.0, 4.0, 8.0); W1S = np.round(np.arange(0, 1.0001, 0.05), 2)
def fit(Pc, Rr):
    """coefficients from fitting cases Pc (with q, y, anc, er columns) and calibration records Rr."""
    f = {}
    G = Rr[Rr.q >= 0.9]; f["b3"], f["a3"] = np.polyfit(G.q, G.y, 1)
    one = Pc[Pc.side != "two"]
    for s in ("prev", "next"):
        for b in BINS:
            H = one[(one.side == s) & (one.db == b)]
            f[f"w_{s}_{b}"] = ls_w(H.anc - H.q, H.y - H.q) if len(H) else 0.0
            H8 = H[H.er.notna()]
            f[f"r_{s}_{b}"] = ls_w(H8.er, H8.y - H8.q) if len(H8) else 0.0
    best = (np.inf, 0.0, 1.0)
    if len(one):
        x = (one.anc - one.q).values; r = (one.y - one.q).values; dd = one.d.values
        for t in TAUS:
            dec = np.exp(-(dd - 1) / t)
            for w1 in W1S:
                sse = float(((r - w1 * dec * x) ** 2).sum())
                if sse < best[0]: best = (sse, w1, t)
    f["w1"], f["tau"] = best[1], best[2]
    two = Pc[Pc.side == "two"]
    f["wg"] = ls_w(two.I - two.q, two.y - two.q) if len(two) else 0.0
    return f
def apply(D, f):
    D = D.copy()
    D["E0"] = D.q
    D["E1"] = D.anc
    D["I"] = D.A + (D.B - D.A) * D.k / (D.k + D.m)
    D["E2"] = np.where(D.side == "two", D.I, np.nan)
    D["E3"] = f["a3"] + f["b3"] * D.q
    w = np.array([f[f"w_{s}_{b}"] for s, b in zip(D.aside, D.db)])
    D["E4"] = D.q + w * (D.anc - D.q)
    D["E4p"] = D.q + f["w1"] * np.exp(-(D.d - 1) / f["tau"]) * (D.anc - D.q)
    rr = np.array([f[f"r_{s}_{b}"] for s, b in zip(D.aside, D.db)])
    D["E8"] = np.where(D.er.notna(), D.q + rr * D.er.fillna(0), D.q)
    return D
ESTS = ["E0", "E1", "E2", "E3", "E4", "E4p", "E8"]
def prep(C, qcol):
    C = C.copy(); C["q"] = [float(QS.loc[r, qcol]) for r in C.rec]
    C["qa"] = [float(QS.loc[r, qcol]) if (r is not None and r in QS.index) else np.nan for r in C.ancr]
    C["ya"] = [YM[idx[r]] for r in C.ancr]
    C["er"] = C.ya - C.qa
    C["I"] = C.A + (C.B - C.A) * C.k / (C.k + C.m)
    return C
def crossfit(C, qcol, unit, COMP):
    C = prep(C, qcol)
    R = pd.DataFrame([dict(rec=r, comp=int(COMP[idx[r]]), seg=SD.get(r, (-1, 0))[0], y=YM[idx[r]], q=float(QS.loc[r, qcol])) for r in QS.index])
    outs = []
    for u in C[unit].unique():
        f = fit(C[C[unit] != u], R[R[unit] != u])
        outs.append(apply(C[C[unit] == u], f))
    return pd.concat(outs).sort_index()
# ---------------- summaries ----------------
def boot_p(G, c):
    H = pd.DataFrame(dict(comp=G.comp.values, s0=((G.E0 - G.y) ** 2).values, sc=((G[c] - G.y) ** 2).values, one=1.0))
    A = H.groupby("comp")[["s0", "sc", "one"]].sum()
    nc = len(A)
    W = rng.multinomial(nc, np.full(nc, 1.0 / nc), size=NBOOT).astype(float)
    n = W @ A.one.values
    d = np.sqrt((W @ A.sc.values) / n) - np.sqrt((W @ A.s0.values) / n)
    return float((d >= 0).mean())
def cell_stats(G, c, per_seed=None):
    H = G[G[c].notna()]
    if len(H) == 0: return None
    rm = float(np.sqrt(((H[c] - H.y) ** 2).mean())); r0 = float(np.sqrt(((H.E0 - H.y) ** 2).mean()))
    out = dict(est=c, n=len(H), nrec=H.rec.nunique(), ncomp=H.comp.nunique(), rmse=rm, rmse_q=r0, d_pct=100 * (rm - r0) / r0 if r0 > 0 else 0,
               better=float((np.abs(H[c] - H.y) < np.abs(H.E0 - H.y) - 1e-12).mean()), bias=float((H[c] - H.y).mean()),
               p_notbetter=boot_p(H, c) if c != "E0" else 1.0)
    if per_seed is not None:
        dirs = []
        for sd, Gs in per_seed.items():
            Hs = Gs.loc[H.index]; Hs = Hs[Hs[c].notna()]
            dirs.append(np.sqrt(((Hs[c] - Hs.y) ** 2).mean()) < np.sqrt(((Hs.E0 - Hs.y) ** 2).mean()))
        out["seeds_better"] = int(np.sum(dirs))
    return out
def cond_masks():
    M = []
    for s in ("prev", "next"):
        for b in BINS: M.append((f"{s} only d={b}", lambda C, s=s, b=b: (C.side == s) & (C.db == b)))
    M.append(("two-sided gap<=4", lambda C: (C.side == "two") & (C.g <= 4)))
    M.append(("two-sided gap 5-7", lambda C: (C.side == "two") & C.g.between(5, 7)))
    M.append(("two-sided gap>=8", lambda C: (C.side == "two") & (C.g >= 8)))
    M.append(("EVAL-LIKE two k>=2 & gap>=7", lambda C: (C.side == "two") & (C.k >= 2) & (C.g >= 7)))
    return M
def table(Cm, per_seed, label, extra=None):
    rows = []
    print(f"\n================ {label} ================")
    for lab, f in cond_masks():
        G = Cm[f(Cm)]
        if extra is not None: G = G[extra(G)]
        if len(G) == 0: print(f"  [{lab}] n 0"); continue
        print(f"  [{lab}]")
        stats = []
        for c in ESTS:
            s = cell_stats(G, c, per_seed)
            if s is None: continue
            stats.append(s)
            print(f"    {c:4s} n {s['n']:4d} rec {s['nrec']:2d} comp {s['ncomp']:2d} | RMSE {s['rmse']:.3f} (q {s['rmse_q']:.3f}, {s['d_pct']:+5.0f}%) | better {s['better']:.2f} | bias {s['bias']:+.3f} | seeds {s.get('seeds_better', '-')}/3 | P(not better) {s['p_notbetter']:.3f}")
        win = [s for s in stats if s["est"] != "E0" and s["rmse"] < s["rmse_q"] and s.get("seeds_better", 0) == 3 and s["p_notbetter"] < ALPHA]
        w = min(win, key=lambda s: s["rmse"])["est"] if win else "E0"
        print(f"    -> 칸 승자(사전 규칙): {w}")
        for s in stats: s.update(cond=lab, winner=w, set=label); rows.append(s)
    return rows
# ================= main =================
ALL = []
for ext in (False, True):
    pre, suc, COMP = make_links(ext)
    lname = "cert+cal" if ext else "cert"
    lens = []
    for i in set(COMP[TRAIN]):
        lens.append(int((COMP[TRAIN] == i).sum()))
    print(f"\n######## link set {lname}: links {len(pre)} | components(train) {len(lens)} | size dist {pd.Series(lens).value_counts().sort_index().to_dict()}")
    Sq = [r for r in QS.index if QS.loc[r, 'mean'] >= 1.0]; Sy = [r for r in QS.index if YM[idx[r]] >= 1.0]
    for sname, sel in (("Sq", Sq), ("Sy", Sy)):
        if ext and sname == "Sy": continue
        C = build(sel, pre, suc, COMP)
        print(f"  {sname}: targets {len(sel)} | cases {len(C)} | one-sided {int((C.side != 'two').sum())} two-sided {int((C.side == 'two').sum())} | comps {C.comp.nunique()}")
        print("  availability (target records) side x d-bin:", C.groupby(['side', 'db']).rec.nunique().to_dict())
        units = ("comp", "seg") if (sname == "Sq" and not ext) else ("comp",)
        for unit in units:
            per_seed = {sd: crossfit(C, sd, unit, COMP) for sd in SEEDS}
            Cm = crossfit(C, "mean", unit, COMP)
            lab = f"{lname} | {sname} | seed-mean q | cross-fit leave-one-{unit}-out"
            ALL += table(Cm, per_seed, lab)
            if sname == "Sq" and not ext and unit == "comp":
                Cm.to_csv(os.path.join(OUT, "hx3_cases_cert_Sq_v1.csv"), index=False)
                table(Cm, per_seed, lab + " | SENS nbh (chain neighbours hidden in same fold)", extra=lambda G: G.nbh)
                table(Cm, per_seed, lab + " | SENS q<1.5", extra=lambda G: G.q < 1.5)
                table(Cm, per_seed, lab + " | SENS q>=1.5", extra=lambda G: G.q >= 1.5)
                CM_MAIN, PS_MAIN, C_MAIN, COMP_MAIN = Cm, per_seed, C, COMP
pd.DataFrame(ALL).to_csv(os.path.join(OUT, "hx3_condtable_v1.csv"), index=False)
# ---------------- E5 (record level) ----------------
print("\n================ E5 same-date training mean (record level, cert Sq/Sy targets) ================")
for sname, sel in (("Sq", [r for r in QS.index if QS.loc[r, 'mean'] >= 1.0]), ("Sy", [r for r in QS.index if YM[idx[r]] >= 1.0])):
    rr = []
    for r in sel:
        i = idx[r]; sd = same_date(i)
        if sd: rr.append(dict(rec=r, y=YM[i], q=float(QS.loc[r, 'mean']), E5=float(np.mean([YM[j] for j in sd])), nsd=len(sd)))
    R5 = pd.DataFrame(rr)
    for lab, G in (("all", R5), ("nsd==2", R5[R5.nsd == 2])):
        print(f"  {sname} {lab}: n {len(G)} | RMSE q {np.sqrt(((G.q-G.y)**2).mean()):.3f} vs E5 {np.sqrt(((G.E5-G.y)**2).mean()):.3f} | E5 better {(np.abs(G.E5-G.y) < np.abs(G.q-G.y)).mean():.2f} | E5 bias {(G.E5-G.y).mean():+.3f}")
# ---------------- distance decay (fit on all cert Sq one-sided cases) ----------------
Pall = prep(C_MAIN, "mean")
Rall = pd.DataFrame([dict(rec=r, comp=0, seg=0, y=YM[idx[r]], q=float(QS.loc[r, "mean"])) for r in QS.index])
F = fit(Pall, Rall)
print("\n================ coefficients fit on ALL cert-Sq cases (for eval application) ================")
print("  E3: y = %.3f + %.3f q" % (F["a3"], F["b3"]))
for s in ("prev", "next"):
    print(f"  E4 w[{s}] " + " ".join(f"{b}:{F[f'w_{s}_{b}']:.2f}" for b in BINS) + f" | E8 rho[{s}] " + " ".join(f"{b}:{F[f'r_{s}_{b}']:.2f}" for b in BINS))
print(f"  E4p: w1 {F['w1']:.2f} tau {F['tau']:.1f}  -> w(d) = " + " ".join(f"d{d}:{F['w1']*np.exp(-(d-1)/F['tau']):.2f}" for d in (1, 2, 3, 4, 5, 7, 9)))
# ---------------- eval 8 mapping ----------------
S13 = pd.read_csv(r"C:\work\farmai\제출\13회차_2026-10-12(나)\submission_22.csv"); S13["rec"] = S13.row_id.str[:7]; P13 = S13.groupby("rec").sub_ec.mean()
aF13, aF47 = idx["F13_217"], idx["F47_229"]
erF13 = YM[aF13] - float(QS.loc["F13_217", "mean"]); erF47 = YM[aF47] - float(QS.loc["F47_229", "mean"])
print(f"\n  anchors: F13_217 23h {EC[aF13,23]:.3f} day-mean {YM[aF13]:.3f} OOF q {QS.loc['F13_217','mean']:.3f} resid {erF13:+.3f} | F47_229 0h {EC[aF47,0]:.3f} day-mean {YM[aF47]:.3f} OOF q {QS.loc['F47_229','mean']:.3f} resid {erF47:+.3f}")
EV = [("F13_221", "prev", 2, EC[aF13, 23], erF13), ("F13_223", "prev", 3, EC[aF13, 23], erF13), ("F13_225", "prev", 4, EC[aF13, 23], erF13),
      ("F47_227", "next", 1, EC[aF47, 0], erF47), ("F47_225", "next", 2, EC[aF47, 0], erF47), ("F47_223", "next", 3, EC[aF47, 0], erF47), ("F47_221", "next", 4, EC[aF47, 0], erF47)]
E = pd.DataFrame([dict(rec=r, side=s, aside=s, d=float(d), db=dbin(d), q=float(P13[r]), anc=float(a), er=float(e), A=np.nan, B=np.nan, k=np.nan, m=np.nan) for r, s, d, a, e in EV])
E = apply(E, F)
CT = pd.DataFrame(ALL); CT = CT[CT.set == "cert | Sq | seed-mean q | cross-fit leave-one-comp-out"]
rows = []
print("\n================ EVAL 8 (rule-safe one-sided mapping: number structure 6.549 + training calendar) ================")
print("    rec    side d   q13   E1    E3    E4    E4p   E8  | cell: winner, n/rec/comp, RMSE q, RMSE winner, bias q")
for _, e in E.iterrows():
    lab = f"{e.side} only d={e.db}"; G = CT[CT.cond == lab]
    w = G.winner.iloc[0] if len(G) else "E0"
    g0 = G[G.est == "E0"].iloc[0] if len(G) else None; gw = G[G.est == w].iloc[0] if len(G) else None
    est = float(e[w]) if w in e.index else float(e.q)
    print(f"  {e.rec} {e.side:4s} {int(e.d)} {e.q:.3f} {e.E1:.3f} {e.E3:.3f} {e.E4:.3f} {e.E4p:.3f} {e.E8:.3f} | {w}, "
          + (f"{int(g0.n)}/{int(g0.nrec)}/{int(g0.ncomp)}, {g0.rmse_q:.3f}, {gw.rmse:.3f}, {g0.bias:+.3f}" if g0 is not None else "n 0"))
    rows.append(dict(rec=e.rec, side=e.side, d=int(e.d), q13=e.q, E1=e.E1, E3=e.E3, E4=e.E4, E4p=e.E4p, E8=e.E8, cell=lab, winner=w, estimate=est,
                     n_cases=int(g0.n) if g0 is not None else 0, n_rec=int(g0.nrec) if g0 is not None else 0, n_comp=int(g0.ncomp) if g0 is not None else 0,
                     rmse_q=float(g0.rmse_q) if g0 is not None else np.nan, rmse_winner=float(gw.rmse) if gw is not None else np.nan,
                     bias_q=float(g0.bias) if g0 is not None else np.nan))
sd237 = same_date(idx["F13_237"]); E5_237 = float(np.mean([YM[j] for j in sd237]))
print(f"  F13_237: no same-bay anchor | same-date training {[(rid(j), round(YM[j], 3)) for j in sd237]} E5 {E5_237:.3f} | q13 {P13['F13_237']:.3f} | E3 {F['a3'] + F['b3'] * P13['F13_237']:.3f}")
rows.append(dict(rec="F13_237", side="none", d=-1, q13=float(P13["F13_237"]), E3=F["a3"] + F["b3"] * P13["F13_237"], cell="E5 record-level", winner="E0", estimate=float(P13["F13_237"]), E5=E5_237))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "hx3_eval8_v1.csv"), index=False)
print("\nsaved:", os.path.join(OUT, "hx3_condtable_v1.csv"), os.path.join(OUT, "hx3_eval8_v1.csv"), os.path.join(OUT, "hx3_cases_cert_Sq_v1.csv"))
