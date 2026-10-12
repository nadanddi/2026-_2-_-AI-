# -*- coding: utf-8 -*-
"""HX2 harness (2026-10-13 집 클로드; HX1 후속).  진단·서술(채택 판정 아님).
HX1 결과: 확실 연결(cert) 사슬은 짧아(앞 ≤ 4일·뒤 ≤ 5일) 평가와 같은 '양쪽 정답, 간격 9일' 사례가 0개.
HX2 는 평가 매핑과 같은 규칙으로 사례를 만든다 — '앵커 = 그 날짜의 고EC(정답 하루 평균 ≥ 1) 학습 기록'(평가: 80일 F13_217, 89일 F47_229 도
고EC 라서 앵커로 고름; ID 무관).  날짜 = 학습 전용 달력 local/dmt3e_E0_days.csv (seg, day; 학습 입력·학습 확실 연결만으로 만든 것).
사례: 대상 r(OOF 있음, Sq: q ≥ 1.0 / Sy: y ≥ 1.0), 같은 seg 에서 r 날짜 d 의 k 일 앞 날짜에 고EC 학습 기록(r 제외)이 있으면 A_k = 그 기록(들) 23시 EC 평균,
  m 일 뒤 날짜에 있으면 B_m = 0시 EC 평균.  (k, m) 조합마다 한 사례 = 그 사이 날짜의 고EC 기록은 모두 숨긴 셈.  k, m ≤ 10.
  앵커 날짜에 고EC 가 둘이면 평균(nA, nB 로 표시) — 감도: 앵커 고EC 하나뿐인 사례만.
추정기: HX1 과 같음(E0 q, E1 A, E1n B, E1c 가까운 쪽, E2 보간, E2w, E3 기울기(학습 q ≥ .9 기록, 에피소드 밖), E4·E4n 한쪽 혼합, E4t 양쪽 혼합,
  E7 E3+앞 앵커 혼합, E6 A + δk) + E9 = 보간 + 간격 구간별 '가운데 부풂'(y − I 평균, 에피소드 밖).
교차적합 단위 = seg(고EC 에피소드; 학습 고EC 는 seg 1·2·3·4 의 네 덩어리 + seg 8·9 한 날씩) → 독립 단위가 4개뿐임을 명시.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import numpy as np, pandas as pd
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_v1"); os.makedirs(OUT, exist_ok=True)
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rng = np.random.default_rng(20261013)
KMAX = 10
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv"))
SEGDAY = {r: (int(s), int(d)) for r, s, d in zip(CAL.rec, CAL.seg, CAL.day)}
YM = {i: float(np.nanmean(EC[i])) for i in np.where(TRAIN)[0]}
HI = {}  # (seg, day) -> list of high training record indices
for r, (s, d) in SEGDAY.items():
    i = idx[r]
    if YM[i] >= 1.0: HI.setdefault((s, d), []).append(i)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
QS = O.groupby(["rec", "seed"]).sg.mean().unstack(); SEEDS = list(QS.columns)
FOLD = O.groupby("rec").fold.first()
def qval(r, qcol): return float(QS.loc[r].mean()) if qcol == "mean" else float(QS.loc[r, qcol])
def build_cases(qcol, sel):
    rows = []
    for r in sel:
        if r not in SEGDAY: continue
        i = idx[r]; s, d = SEGDAY[r]; q = qval(r, qcol); y = YM[i]
        prevs = [(None, np.nan, 0)]; nexts = [(None, np.nan, 0)]
        for k in range(1, KMAX + 1):
            L = [j for j in HI.get((s, d - k), []) if j != i]
            if L: prevs.append((k, float(np.mean([EC[j, 23] for j in L])), len(L)))
            L = [j for j in HI.get((s, d + k), []) if j != i]
            if L: nexts.append((k, float(np.mean([EC[j, 0] for j in L])), len(L)))
        for k, A, nA in prevs:
            for m, B, nB in nexts:
                if k is None and m is None: continue
                rows.append(dict(rec=r, comp=s, y=y, q=q, k=k, m=m, A=A, B=B, nA=nA, nB=nB, day=d))
    C = pd.DataFrame(rows)
    for c in ("k", "m"): C[c] = C[c].astype(float)
    return C
def ls_w(x, r):
    d = float((x * x).sum())
    return float(np.clip((x * r).sum() / d, 0, 1)) if d > 0 else 0.0
KB = lambda d: np.select([d <= 1, d <= 2, d <= 4], ["1", "2", "3-4"], "5-10")
GB = lambda g: np.select([g <= 4, g <= 7], ["<=4", "5-7"], "8+")
BINS = ("1", "2", "3-4", "5-10"); GBINS = ("<=4", "5-7", "8+")
def calib_records(qcol):
    return pd.DataFrame([dict(rec=r, comp=SEGDAY.get(r, (1000 + idx[r], 0))[0], y=YM[idx[r]], q=qval(r, qcol)) for r in QS.index])
def design_t(D):
    return np.column_stack([((D.kb == b) * (D.A - D.q)).values for b in BINS] + [((D.mb == b) * (D.B - D.q)).values for b in BINS])
def fit_all(P, Rr):
    """all coefficients from fitting rows P (cases) and Rr (calibration records)."""
    f = {}
    G = Rr[Rr.q >= 0.9]; f["b1"], f["a1"] = np.polyfit(G.q, G.y, 1)
    Gy = Rr[Rr.y >= 1.0]; f["b2"], f["a2"] = np.polyfit(Gy.q, Gy.y, 1)
    for gb in GBINS:
        Pg = P[(P.gb == gb)]
        f["wg" + gb] = ls_w((Pg.I - Pg.q).values, (Pg.y - Pg.q).values) if len(Pg) else 0.0
        f["bulge" + gb] = float((Pg.y - Pg.I).mean()) if len(Pg) else 0.0
    for kb in BINS:
        Pk = P[(P.kb == kb) & P.m.isna()]
        f["wk" + kb] = ls_w((Pk.A - Pk.q).values, (Pk.y - Pk.q).values) if len(Pk) else 0.0
        q3 = f["a1"] + f["b1"] * Pk.q
        f["w7" + kb] = ls_w((Pk.A - q3).values, (Pk.y - q3).values) if len(Pk) else 0.0
        Pm = P[(P.mb == kb) & P.k.isna()]
        f["wm" + kb] = ls_w((Pm.B - Pm.q).values, (Pm.y - Pm.q).values) if len(Pm) else 0.0
    P2 = P[P.k.notna() & P.m.notna()]
    f["wt"] = np.clip(np.linalg.lstsq(design_t(P2), (P2.y - P2.q).values, rcond=None)[0], 0, 1) if len(P2) else np.zeros(8)
    Pd = P[P.k.notna() & P.m.isna()]; f["delta"] = float(((Pd.y - Pd.A) / Pd.k).mean()) if len(Pd) else 0.0
    return f
def apply(C, f):
    C = C.copy()
    C["E3"] = f["a1"] + f["b1"] * C.q; C["E3y"] = f["a2"] + f["b2"] * C.q
    C["E2w"] = C.q + C.gb.map(lambda g: f.get("wg" + g, np.nan)) * (C.I - C.q)
    C["E9"] = C.I + C.gb.map(lambda g: f.get("bulge" + g, np.nan))
    C["E4"] = C.q + C.kb.map(lambda b: f.get("wk" + b, np.nan)) * (C.A - C.q)
    C["E7"] = C.E3 + C.kb.map(lambda b: f.get("w7" + b, np.nan)) * (C.A - C.E3)
    C["E4n"] = C.q + C.mb.map(lambda b: f.get("wm" + b, np.nan)) * (C.B - C.q)
    two = C.k.notna() & C.m.notna()
    C["E4t"] = np.nan
    if two.any(): C.loc[two, "E4t"] = C.loc[two, "q"].values + design_t(C[two]) @ f["wt"]
    C["E6"] = C.A + f["delta"] * C.k
    return C
def prep(C):
    C = C.copy(); C["kb"] = np.where(C.k.notna(), KB(C.k.fillna(99)), ""); C["mb"] = np.where(C.m.notna(), KB(C.m.fillna(99)), "")
    C["g"] = C.k + C.m; C["gb"] = np.where(C.g.notna(), GB(C.g.fillna(99)), "")
    C["I"] = C.A + (C.B - C.A) * C.k / (C.k + C.m)
    C["E0"] = C.q; C["E1"] = C.A; C["E1n"] = C.B
    C["E1c"] = np.where(C.k.notna() & (C.m.isna() | (C.k <= C.m)), C.A, C.B); C["E2"] = C.I
    return C
def crossfit(C, R):
    parts = []
    for comp in C.comp.unique():
        f = fit_all(C[C.comp != comp], R[R.comp != comp]); parts.append(apply(C[C.comp == comp], f))
    return pd.concat(parts)
EST = ["E0", "E1", "E1n", "E1c", "E2", "E2w", "E9", "E3", "E3y", "E4", "E4n", "E4t", "E7", "E6"]
def summ(G, cols, boot=1000):
    out = {}
    for c in cols:
        H = G[G[c].notna()]
        if len(H) == 0: continue
        e = (H[c] - H.y).values; e0 = (H.E0 - H.y).values
        rm, rm0 = float(np.sqrt((e ** 2).mean())), float(np.sqrt((e0 ** 2).mean()))
        # record-level (average squared error per target record first) to de-weight records with many (k,m) combos
        rr = H.assign(se=e ** 2, se0=e0 ** 2).groupby("rec")[["se", "se0"]].mean()
        cs = H.comp.unique(); grp = {cc: np.where(H.comp.values == cc)[0] for cc in cs}
        dl = []
        for _ in range(boot):
            pick = np.concatenate([grp[cc] for cc in rng.choice(cs, len(cs))])
            dl.append(float((e[pick] ** 2).mean() - (e0[pick] ** 2).mean()))
        out[c] = dict(n=len(H), nrec=H.rec.nunique(), ncomp=len(cs), rmse=rm, rmse0=rm0, d=(rm - rm0) / rm0,
                      rrec=float(np.sqrt(rr.se.mean())), rrec0=float(np.sqrt(rr.se0.mean())),
                      imp=float((np.abs(e) < np.abs(e0)).mean()), bias=float(e.mean()), pw=float(np.mean(np.array(dl) >= 0)))
    return out
def fmt(o, label):
    s = [f"  [{label}]"]
    for c, v in o.items():
        s.append(f"    {c:4s} n {v['n']:4d} rec {v['nrec']:2d} seg {v['ncomp']} | RMSE {v['rmse']:.3f} (q {v['rmse0']:.3f}, {100*v['d']:+4.0f}%) | rec-avg {v['rrec']:.3f} (q {v['rrec0']:.3f}) | better {v['imp']:.2f} | bias {v['bias']:+.3f} | P(not better) {v['pw']:.2f}")
    return "\n".join(s)
two = lambda C: C.k.notna() & C.m.notna()
CONDS = [
    ("prev only k=1", lambda C: (C.k == 1) & C.m.isna()),
    ("prev only k=2", lambda C: (C.k == 2) & C.m.isna()),
    ("prev only k=3-4", lambda C: C.k.between(3, 4) & C.m.isna()),
    ("prev only k=5-10", lambda C: C.k.between(5, 10) & C.m.isna()),
    ("next only m=1", lambda C: (C.m == 1) & C.k.isna()),
    ("next only m=2-4", lambda C: C.m.between(2, 4) & C.k.isna()),
    ("next only m=5-10", lambda C: C.m.between(5, 10) & C.k.isna()),
    ("two-sided gap<=4", lambda C: two(C) & (C.g <= 4)),
    ("two-sided gap 5-7", lambda C: two(C) & C.g.between(5, 7)),
    ("two-sided gap 8+", lambda C: two(C) & (C.g >= 8)),
    ("EVAL-LIKE two-sided k>=2 & gap>=7", lambda C: two(C) & (C.k >= 2) & (C.g >= 7)),
    ("EVAL-LIKE ... nearest=prev (k<=m) [F13_221/223/225]", lambda C: two(C) & (C.k >= 2) & (C.g >= 7) & (C.k <= C.m)),
    ("EVAL-LIKE ... nearest=next m>=2 (m<k) [F47_221/223/225]", lambda C: two(C) & (C.k >= 2) & (C.g >= 7) & (C.m < C.k) & (C.m >= 2)),
    ("EVAL-LIKE ... m=1 (k>=6) [F47_227]", lambda C: two(C) & (C.k >= 6) & (C.m == 1)),
]
pd.set_option("display.width", 250)
res = {}
for qcol in ["mean"] + SEEDS:
    R = calib_records(qcol)
    for sname, sel in (("Sq", R[R.q >= 1.0].rec), ("Sy", R[R.y >= 1.0].rec)):
        C = prep(build_cases(qcol, list(sel)))
        res[(qcol, sname)] = crossfit(C, R)
    if qcol == "mean":
        Cq = res[("mean", "Sq")]
        print(f"Sq targets with calendar: {Cq.rec.nunique()} | cases {len(Cq)} | segs {sorted(Cq.comp.unique())}")
        print("Sq availability: target records per (k, m) [0 = none]")
        print(Cq.assign(kk=Cq.k.fillna(0).astype(int), mm=Cq.m.fillna(0).astype(int)).groupby(["kk", "mm"]).rec.nunique().unstack(fill_value=0).to_string())
        print("Sq records:", Cq.drop_duplicates("rec").sort_values(["comp", "day"])[["rec", "comp", "day", "y", "q"]].round(3).to_string(index=False))
for sname in ("Sq", "Sy"):
    print(f"\n================ {sname} (seed-mean q, cross-fit leave-one-seg-out) ================")
    C = res[("mean", sname)]
    for lab, f in CONDS:
        G = C[f(C)]
        print(fmt(summ(G, EST), lab) if len(G) else f"  [{lab}] n 0")
# robustness: anchors with single high record only, eval-like
print("\n================ Sq EVAL-LIKE, single high anchor record on both anchor dates ================")
C = res[("mean", "Sq")]; G = C[two(C) & (C.k >= 2) & (C.g >= 7) & (C.nA == 1) & (C.nB == 1)]
print(fmt(summ(G, EST), "single-anchor eval-like") if len(G) else "  n 0")
print("\n================ Sq EVAL-LIKE per seg (episode) — record-level RMSE ================")
G = C[two(C) & (C.k >= 2) & (C.g >= 7)]
for s, H in G.groupby("comp"):
    print(f"  seg {s}: n {len(H)} rec {H.rec.nunique()} | " + " | ".join(f"{c} {np.sqrt(((H[c]-H.y)**2).mean()):.3f}" for c in ["E0", "E1c", "E2", "E2w", "E9", "E3", "E4t"]) + f" | mean(y-I) {(H.y-H.I).mean():+.3f} | mean(y-q) {(H.y-H.q).mean():+.3f}")
print("\n================ per-seed (Sq): RMSE change vs q (better share) ================")
for lab, f in [CONDS[10], CONDS[11], CONDS[12], CONDS[13], CONDS[1], CONDS[2], CONDS[3]]:
    print(f"  [{lab}]")
    for c in ["E1c", "E2", "E2w", "E9", "E3", "E4", "E4t", "E7"]:
        s = []
        for qcol in SEEDS:
            G = res[(qcol, "Sq")]; G = G[f(G) & G[c].notna()]
            if len(G) == 0: s.append("n/a"); continue
            rm = np.sqrt(((G[c] - G.y) ** 2).mean()); rm0 = np.sqrt(((G.E0 - G.y) ** 2).mean())
            s.append(f"{100*(rm-rm0)/rm0:+5.0f}% ({(np.abs(G[c]-G.y) < np.abs(G.E0-G.y)).mean():.2f})")
        print(f"    {c:4s} " + " | ".join(f"s{sd}: {x}" for sd, x in zip(SEEDS, s)))
print("\n================ Sq by model size, EVAL-LIKE ================")
for lab, f in (("q<1.2", lambda C: C.q < 1.2), ("1.2<=q<1.5", lambda C: C.q.between(1.2, 1.5, inclusive="left")), ("q>=1.5", lambda C: C.q >= 1.5)):
    G = C[f(C) & two(C) & (C.k >= 2) & (C.g >= 7)]
    print(fmt(summ(G, ["E0", "E1c", "E2", "E2w", "E9", "E3", "E4t"]), "EVAL-LIKE & " + lab) if len(G) else f"  [{lab}] n 0")
# ---------- eval mapping ----------
print("\n================ EVAL 8 (DM11-based narrative geometry; coefficients fit on ALL Sq cases) ================")
R = calib_records("mean"); P = res[("mean", "Sq")]
f = fit_all(P, R)
print("  E3 y = %.3f + %.3f q | E2w w_gap %s | E9 bulge %s | E4t wA %s wB %s | E4 w_k %s | E4n w_m %s" % (
    f["a1"], f["b1"], {g: round(f["wg" + g], 2) for g in GBINS}, {g: round(f["bulge" + g], 3) for g in GBINS},
    dict(zip(BINS, np.round(f["wt"][:4], 2))), dict(zip(BINS, np.round(f["wt"][4:], 2))), {b: round(f["wk" + b], 2) for b in BINS}, {b: round(f["wm" + b], 2) for b in BINS}))
A0, B0 = float(EC[idx["F13_217"], 23]), float(EC[idx["F47_229"], 0])
S13 = pd.read_csv(r"C:\work\farmai\제출\13회차_2026-10-12(나)\submission_22.csv"); S13["rec"] = S13.row_id.str[:7]; P13 = S13.groupby("rec").sub_ec.mean()
EV = [("F13_221", 2, 7), ("F13_223", 3, 6), ("F13_225", 4, 5), ("F47_221", 5, 4), ("F47_223", 6, 3), ("F47_225", 7, 2), ("F47_227", 8, 1)]
E = pd.DataFrame([dict(rec=r, k=float(k), m=float(m), q=float(P13[r]), A=A0, B=B0, y=np.nan) for r, k, m in EV])
E = apply(prep(E), f)
cols = ["rec", "k", "m", "q", "E1c", "E2", "E2w", "E9", "E3", "E4", "E4n", "E4t", "E7"]
print(E[cols].round(3).to_string(index=False))
E[cols].to_csv(os.path.join(OUT, "hx2_eval8_estimates_v1.csv"), index=False)
for c in ("mean",): res[(c, "Sq")].to_csv(os.path.join(OUT, "hx2_cases_Sq_seedmean_v1.csv"), index=False)
