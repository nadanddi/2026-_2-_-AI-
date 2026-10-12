# -*- coding: utf-8 -*-
"""HX1 harness (2026-10-13 집 클로드; 사용자: '고EC 8기록의 수준을 맞히는 방법 더 찾아봐').  진단·서술(채택 판정 아님).
목적: 평가 순수 2차 창 고EC 8기록(F13_221·223·225·237, F47_221·223·225·227)과 같은 조건 — 같은 동 앞뒤 며칠 정답 없음,
가장 가까운 같은 동 정답이 앞 k 일·뒤 m 일 — 을 학습 고EC 기록에서 흉내 낸 '가짜 평가 사례'로 만들고, 하루 수준 추정기 비교.
사례: 학습 기록 r(OOF 있음) 에서 진단용 같은 동 연결(truth.pkl, cert, 학습끼리 = 같은 동 다음날)을 따라 k 단계 앞 기록 a(정답 23시 A),
      m 단계 뒤 기록 b(정답 0시 B) 를 잡음(사이 날 정답은 안 씀 = 숨김).  k, m ∈ 1..9, 한쪽만 있는 사례도 포함.
대상 집합: Sq = OOF 예측 하루 평균 q ≥ 1.0 (평가처럼 '예측이 높아서' 고른 집합, 주 판정), Sy = 정답 하루 평균 ≥ 1.0 (지시된 집합, 선택 편향 있음).
추정기(하루 평균):
  E0 q (DIAG10 CUR sg, 시드 평균; 13회차 = 9회차 EC14 와 같은 구성) | E1 A(앞 정답 그대로) | E1n B(뒤 정답 그대로) | E1c 더 가까운 쪽 정답
  E2 앞뒤 선형 보간 I = A + (B−A)k/(k+m) | E2w q + w_gap(I − q)
  E3 기울기 보정 a + b·q (학습 q ≥ 0.9 기록에서 적합, 사례 성분 제외) | E3y 같은 식을 정답 ≥ 1 기록에서 적합(편향 참고용)
  E4 q + w_k(A − q) (w 는 거리 구간별, 사례 성분 밖 최소제곱, [0,1] 자름) | E4n q + w_m(B − q) | E4t q + wA_k(A−q) + wB_m(B−q) (동시 적합)
  E7 E3 + w_k(A − E3) | E6 A + δ·k (δ = 고EC 하루 변화, 성분 밖)
  E5 같은 날짜 학습 기록(외기 70% 이상 같음, 자신 제외) 하루 평균의 평균
교차적합: 모든 계수는 사례의 '같은 동 연결 성분'(cert 사슬) 전체를 뺀 나머지에서만 적합.  시드 3개 각각 반복.
평가 매핑(서술): DM11 날짜(평가 입력으로 만든 달력 — 서술 대조만, 계수·선택에 안 씀)와 '81일 지움, 85일 ID 전환' 가정으로
  F13_221/223/225 = 앞 F13_217(80일 23시 1.663) 2/3/4 일, F47_221/223/225/227 = 5/6/7/8 일; 뒤 F47_229(89일 0시 1.716) 7/6/5/4/3/2/1 일.
  F13_237 = 같은 날짜 학습 F47_230·F47_231.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import numpy as np, pandas as pd
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_v1"); os.makedirs(OUT, exist_ok=True)
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rng = np.random.default_rng(20261013)
KMAX = 9
# ---------- same-bay chain (diagnostic truth links, training only) ----------
TL = pd.read_pickle(os.path.join(S3, "truth.pkl"))
pre, suc = {}, {}
for a, b, c in zip(TL.a1.astype(int), TL.b.astype(int), TL.cert.astype(bool)):
    if c and TRAIN[a] and TRAIN[b]:
        pre[b] = a; suc[a] = b
par = {i: i for i in range(N)}
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for b, a in pre.items(): par[fd(a)] = fd(b)
COMP = np.array([fd(i) for i in range(N)])
def walk(r, k, M):
    x = r
    for _ in range(k):
        x = M.get(x)
        if x is None: return None
    return x
# ---------- OOF (DIAG10 CUR) ----------
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
QS = O.groupby(["rec", "seed"]).sg.mean().unstack()          # day-mean prediction per seed
FOLD = O.groupby("rec").fold.first()
SEEDS = list(QS.columns)
YM = {i: float(np.nanmean(EC[i])) for i in np.where(TRAIN)[0]}
# same-date training records (outdoor equality > 70%)
Xf = X.reshape(N, -1)
def same_date(i, pool):
    eq = (np.abs(Xf[pool] - Xf[i]) < 1e-6).mean(1)
    return [int(j) for j in pool[eq > 0.7] if j != i]
TRIDX = np.where(TRAIN)[0]
# ---------- cases ----------
def build_cases(qcol):
    rows = []
    for r_name in QS.index:
        i = idx[r_name]; q = float(QS.loc[r_name, qcol]) if qcol != "mean" else float(QS.loc[r_name].mean()); y = YM[i]
        sd = same_date(i, TRIDX); E5 = float(np.mean([YM[j] for j in sd])) if sd else np.nan
        prevs = [(None, None)] + [(k, walk(i, k, pre)) for k in range(1, KMAX + 1)]
        nexts = [(None, None)] + [(m, walk(i, m, suc)) for m in range(1, KMAX + 1)]
        # neighbours +-1 hidden in the same DIAG10 fold? (eval-likeness of q)
        p1, n1 = pre.get(i), suc.get(i)
        nb_hidden = all((FOLD.get(rid(j)) == FOLD[r_name]) if (j is not None and rid(j) in FOLD.index) else True for j in (p1, n1))
        for k, a in prevs:
            if k is not None and a is None: continue
            for m, b in nexts:
                if m is not None and b is None: continue
                if k is None and m is None: continue
                rows.append(dict(rec=r_name, i=i, comp=COMP[i], y=y, q=q, k=k, m=m, A=float(EC[a, 23]) if a is not None else np.nan,
                                 B=float(EC[b, 0]) if b is not None else np.nan, E5=E5, nsd=len(sd), nbh=nb_hidden, p2=NUM[i] >= 179))
    C = pd.DataFrame(rows)
    return C
def ls_w(x, r):
    d = float((x * x).sum())
    return float(np.clip((x * r).sum() / d, 0, 1)) if d > 0 else 0.0
KB = lambda d: np.select([d <= 1, d <= 2, d <= 4], ["1", "2", "3-4"], "5-9")
GB = lambda g: np.select([g <= 4, g <= 7], ["<=4", "5-7"], "8-10")
def calib_records(qcol):
    R = pd.DataFrame([dict(rec=r, i=idx[r], comp=COMP[idx[r]], y=YM[idx[r]], q=float(QS.loc[r, qcol]) if qcol != "mean" else float(QS.loc[r].mean())) for r in QS.index])
    return R
def estimators(C, R):
    """cross-fitted estimators; coefficients from rows whose comp != case comp."""
    C = C.copy(); C["kb"] = np.where(C.k.notna(), KB(C.k.fillna(99)), ""); C["mb"] = np.where(C.m.notna(), KB(C.m.fillna(99)), "")
    C["g"] = C.k + C.m; C["gb"] = np.where(C.g.notna(), GB(C.g.fillna(99)), "")
    C["I"] = C.A + (C.B - C.A) * C.k / (C.k + C.m)
    C["E0"] = C.q; C["E1"] = C.A; C["E1n"] = C.B
    C["E1c"] = np.where(C.k.notna() & (C.m.isna() | (C.k <= C.m)), C.A, C.B); C["E2"] = C.I
    for c in ("E2w", "E3", "E3y", "E4", "E4n", "E4t", "E7", "E6"): C[c] = np.nan
    pool = C  # cases used for fitting weights: all case rows of Sq U Sy (passed in)
    for comp in C.comp.unique():
        m_ = C.comp == comp; P = pool[pool.comp != comp]; Rr = R[R.comp != comp]
        # E3 slope on q>=0.9 records ; E3y on y>=1 records
        G = Rr[Rr.q >= 0.9]; b1, a1 = np.polyfit(G.q, G.y, 1)
        Gy = Rr[Rr.y >= 1.0]; b2, a2 = np.polyfit(Gy.q, Gy.y, 1)
        C.loc[m_, "E3"] = a1 + b1 * C.loc[m_, "q"]; C.loc[m_, "E3y"] = a2 + b2 * C.loc[m_, "q"]
        # E2w by gap bin
        for gb in ("<=4", "5-7", "8-10"):
            Pg = P[(P.gb == gb)]; w = ls_w((Pg.I - Pg.q).values, (Pg.y - Pg.q).values) if len(Pg) else 0.0
            mm = m_ & (C.gb == gb); C.loc[mm, "E2w"] = C.loc[mm, "q"] + w * (C.loc[mm, "I"] - C.loc[mm, "q"])
        # E4 prev-only weights by k bin (fit on cases with prev, any m) ; E4n by m bin
        for kb in ("1", "2", "3-4", "5-9"):
            Pk = P[(P.kb == kb) & P.m.isna()]; w = ls_w((Pk.A - Pk.q).values, (Pk.y - Pk.q).values) if len(Pk) else 0.0
            mm = m_ & (C.kb == kb); C.loc[mm, "E4"] = C.loc[mm, "q"] + w * (C.loc[mm, "A"] - C.loc[mm, "q"])
            Pk3 = P[(P.kb == kb) & P.m.isna()]; q3 = a1 + b1 * Pk3.q
            w7 = ls_w((Pk3.A - q3).values, (Pk3.y - q3).values) if len(Pk3) else 0.0
            C.loc[mm, "E7"] = C.loc[mm, "E3"] + w7 * (C.loc[mm, "A"] - C.loc[mm, "E3"])
            Pm = P[(P.mb == kb) & P.k.isna()]; w = ls_w((Pm.B - Pm.q).values, (Pm.y - Pm.q).values) if len(Pm) else 0.0
            mm = m_ & (C.mb == kb); C.loc[mm, "E4n"] = C.loc[mm, "q"] + w * (C.loc[mm, "B"] - C.loc[mm, "q"])
        # E4t: joint two-sided, design = (A-q)*1[kb], (B-q)*1[mb]
        P2 = P[P.k.notna() & P.m.notna()]
        bins = ("1", "2", "3-4", "5-9")
        Xd = np.column_stack([((P2.kb == b) * (P2.A - P2.q)).values for b in bins] + [((P2.mb == b) * (P2.B - P2.q)).values for b in bins])
        w = np.linalg.lstsq(Xd, (P2.y - P2.q).values, rcond=None)[0] if len(P2) else np.zeros(8)
        w = np.clip(w, 0, 1)
        mm = m_ & C.k.notna() & C.m.notna(); Cm = C[mm]
        Xm = np.column_stack([((Cm.kb == b) * (Cm.A - Cm.q)).values for b in bins] + [((Cm.mb == b) * (Cm.B - Cm.q)).values for b in bins])
        C.loc[mm, "E4t"] = Cm.q.values + Xm @ w
        # E6 drift: delta = mean over prev-only cases of (y - A)/k
        Pd = P[P.k.notna() & P.m.isna()]; dl = float(((Pd.y - Pd.A) / Pd.k).mean()) if len(Pd) else 0.0
        mm = m_ & C.k.notna(); C.loc[mm, "E6"] = C.loc[mm, "A"] + dl * C.loc[mm, "k"]
    return C
EST = ["E0", "E1", "E1n", "E1c", "E2", "E2w", "E3", "E3y", "E4", "E4n", "E4t", "E7", "E6", "E5"]
def summ(G, cols, boot=1000):
    out = {}
    comps = G.comp.unique()
    for c in cols:
        H = G[G[c].notna()]
        if len(H) == 0: continue
        e = H[c] - H.y; e0 = H.E0 - H.y
        rm, rm0 = float(np.sqrt((e ** 2).mean())), float(np.sqrt((e0 ** 2).mean()))
        # component bootstrap of delta MSE
        cs = H.comp.unique(); grp = {cc: np.where(H.comp.values == cc)[0] for cc in cs}
        dl = []
        for _ in range(boot):
            pick = np.concatenate([grp[cc] for cc in rng.choice(cs, len(cs))])
            dl.append(float((e.values[pick] ** 2).mean() - (e0.values[pick] ** 2).mean()))
        out[c] = dict(n=len(H), nrec=H.rec.nunique(), ncomp=len(cs), rmse=rm, rmse0=rm0, d=(rm - rm0) / rm0, imp=float((e.abs() < e0.abs()).mean()),
                      bias=float(e.mean()), pw=float(np.mean(np.array(dl) >= 0)))
    return out
def fmt(o, label):
    s = [f"  [{label}]"]
    for c, v in o.items():
        s.append(f"    {c:4s} n {v['n']:4d} rec {v['nrec']:2d} comp {v['ncomp']:2d} | RMSE {v['rmse']:.3f} (q {v['rmse0']:.3f}, {100*v['d']:+.0f}%) | better-than-q {v['imp']:.2f} | bias {v['bias']:+.3f} | P(not better) {v['pw']:.2f}")
    return "\n".join(s)
# ---------- run per seed + seed mean ----------
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
allres = {}
for qcol in ["mean"] + SEEDS:
    R = calib_records(qcol)
    C = build_cases(qcol)
    sel_q = set(R[R.q >= 1.0].rec); sel_y = set(R[R.y >= 1.0].rec)
    C = C[C.rec.isin(sel_q | sel_y)].copy()
    Cq = estimators(C[C.rec.isin(sel_q)], R); Cq["Sq"] = True; Cq["Sy"] = False   # weights fitted within the same selection set
    Cy = estimators(C[C.rec.isin(sel_y)], R); Cy["Sq"] = False; Cy["Sy"] = True
    allres[(qcol, "Sq")] = Cq; allres[(qcol, "Sy")] = Cy
    C = pd.concat([Cq, Cy])
    allres[qcol] = C
    if qcol == "mean":
        C.to_csv(os.path.join(OUT, "hx1_cases_seedmean_v1.csv"), index=False)
        print(f"high-EC targets: Sq (q>=1.0) {len(sel_q)} rec, Sy (y>=1.0) {len(sel_y)} rec, both {len(sel_q & sel_y)}; cases {len(C)}; comps {C.comp.nunique()}")
        print("Sq records:", sorted(sel_q)); print("Sy records:", sorted(sel_y))
        print("Sq record-level: q mean %.3f y mean %.3f | Sy: q %.3f y %.3f" % (R[R.rec.isin(sel_q)].q.mean(), R[R.rec.isin(sel_q)].y.mean(), R[R.rec.isin(sel_y)].q.mean(), R[R.rec.isin(sel_y)].y.mean()))
        G = R[R.q >= 0.9]; print("E3 full-data fit on q>=0.9 (n %d): y = %.3f + %.3f q" % (len(G), *np.polyfit(G.q, G.y, 1)[::-1]))
        Gy = R[R.y >= 1]; print("E3y full-data fit on y>=1 (n %d): y = %.3f + %.3f q" % (len(Gy), *np.polyfit(Gy.q, Gy.y, 1)[::-1]))
        # availability table
        print("\ncase availability (Sq U Sy) — rows k (prev distance), cols m (next distance), count of target records:")
        av = C.assign(kk=C.k.fillna(0).astype(int), mm=C.m.fillna(0).astype(int)).groupby(["kk", "mm"]).rec.nunique().unstack(fill_value=0)
        print(av.to_string())
def table(qcol, sname, conds):
    C = allres[(qcol, sname)]
    lines = []
    for lab, f in conds:
        G = C[f(C)]
        if len(G) == 0: lines.append(f"  [{lab}] n 0"); continue
        lines.append(fmt(summ(G, EST), lab))
    return "\n".join(lines)
CONDS = [
    ("one-sided prev only, k=1", lambda C: (C.k == 1) & C.m.isna()),
    ("one-sided prev only, k=2", lambda C: (C.k == 2) & C.m.isna()),
    ("one-sided prev only, k=3-4", lambda C: C.k.between(3, 4) & C.m.isna()),
    ("one-sided prev only, k=5-9", lambda C: C.k.between(5, 9) & C.m.isna()),
    ("one-sided next only, m=1", lambda C: (C.m == 1) & C.k.isna()),
    ("one-sided next only, m=2-4", lambda C: C.m.between(2, 4) & C.k.isna()),
    ("two-sided, gap<=4", lambda C: C.k.notna() & C.m.notna() & ((C.k + C.m) <= 4)),
    ("two-sided, gap 5-7", lambda C: C.k.notna() & C.m.notna() & (C.k + C.m).between(5, 7)),
    ("two-sided, gap 8-10", lambda C: C.k.notna() & C.m.notna() & (C.k + C.m).between(8, 10)),
    ("EVAL-LIKE two-sided k>=2, gap>=7 (eval gap 9)", lambda C: C.k.notna() & C.m.notna() & (C.k >= 2) & ((C.k + C.m) >= 7)),
    ("EVAL-LIKE k>=2,gap>=7, nearest=prev (k<=m)", lambda C: C.k.notna() & C.m.notna() & (C.k >= 2) & ((C.k + C.m) >= 7) & (C.k <= C.m)),
    ("EVAL-LIKE k>=2,gap>=7, nearest=next (m<k)", lambda C: C.k.notna() & C.m.notna() & (C.k >= 2) & ((C.k + C.m) >= 7) & (C.m < C.k)),
]
for sname in ("Sq", "Sy"):
    print(f"\n================ {sname} (seed-mean q) ================")
    print(table("mean", sname, CONDS))
# q-size split on eval-like set and on one-sided prev k>=2
print("\n================ Sq by model-prediction size (seed-mean) ================")
C = allres[("mean", "Sq")]
for lab, f in (("q<1.2", lambda C: C.q < 1.2), ("1.2<=q<1.5", lambda C: C.q.between(1.2, 1.5, inclusive="left")), ("q>=1.5", lambda C: C.q >= 1.5)):
    G = C[f(C) & C.k.notna() & C.m.notna() & (C.k >= 2) & ((C.k + C.m) >= 7)]
    if len(G): print(fmt(summ(G, ["E0", "E1", "E2", "E2w", "E3", "E4t", "E7"]), "EVAL-LIKE & " + lab))
    G = C[f(C) & C.k.notna() & C.m.isna() & (C.k >= 2)]
    if len(G): print(fmt(summ(G, ["E0", "E1", "E3", "E4", "E7", "E6"]), "prev-only k>=2 & " + lab))
# eval-likeness of q: neighbours hidden in same fold
print("\n================ Sq record level: q error by whether +-1 same-bay neighbours were also hidden in DIAG10 fold ================")
Rr = allres[("mean", "Sq")].drop_duplicates("rec")
for v, G in Rr.groupby("nbh"): print(f"  neighbours hidden={v}: n {len(G)} | mean(q-y) {(G.q-G.y).mean():+.3f} | RMSE {np.sqrt(((G.q-G.y)**2).mean()):.3f}")
# per-seed robustness for main conditions
print("\n================ per-seed (Sq): RMSE change vs q, better-than-q share ================")
MAIN = [CONDS[9], CONDS[1], CONDS[2], CONDS[3], CONDS[4]]
for lab, f in MAIN:
    print(f"  [{lab}]")
    for c in ["E1", "E2", "E2w", "E3", "E4", "E4t", "E7"]:
        s = []
        for qcol in SEEDS:
            C = allres[(qcol, "Sq")]; G = C[f(C)]
            G = G[G[c].notna()]
            if len(G) == 0: s.append("  n/a"); continue
            rm = np.sqrt(((G[c] - G.y) ** 2).mean()); rm0 = np.sqrt(((G.E0 - G.y) ** 2).mean())
            s.append(f"{100*(rm-rm0)/rm0:+5.0f}% ({(np.abs(G[c]-G.y) < np.abs(G.E0-G.y)).mean():.2f})")
        print(f"    {c:4s} " + " | ".join(f"s{sd}: {x}" for sd, x in zip(SEEDS, s)))
# ---------- E5 same-date ----------
print("\n================ E5 same-date training mean (record level, Sq U Sy) ================")
for sname in ("Sq", "Sy"):
    Rr = allres[("mean", sname)].drop_duplicates("rec"); H = Rr[Rr.nsd > 0]
    print(f"  {sname}: n {len(H)} | RMSE q {np.sqrt(((H.q-H.y)**2).mean()):.3f} vs E5 {np.sqrt(((H.E5-H.y)**2).mean()):.3f} | E5 better {(np.abs(H.E5-H.y)<np.abs(H.q-H.y)).mean():.2f} | E5 bias {(H.E5-H.y).mean():+.3f} | nsd dist {H.nsd.value_counts().to_dict()}")
    H2 = H[H.nsd == 2]
    if len(H2): print(f"     nsd==2 (like F13_237): n {len(H2)} | RMSE q {np.sqrt(((H2.q-H2.y)**2).mean()):.3f} vs E5 {np.sqrt(((H2.E5-H2.y)**2).mean()):.3f} | mix .5 {np.sqrt((((H2.q+H2.E5)/2-H2.y)**2).mean()):.3f}")
# ---------- eval mapping ----------
print("\n================ EVAL 8 mapping (DM11-based narrative geometry; coefficients fit on all cases, no left-out) ================")
R = calib_records("mean")
G = R[R.q >= 0.9]; b1, a1 = np.polyfit(G.q, G.y, 1)
P = allres[("mean", "Sq")]   # eval records are selected by prediction -> Sq-fitted weights
def wfit(F, xcol): return ls_w((F[xcol] - F.q).values, (F.y - F.q).values)
bins = ("1", "2", "3-4", "5-9")
P2 = P[P.k.notna() & P.m.notna()]
Xd = np.column_stack([((P2.kb == b) * (P2.A - P2.q)).values for b in bins] + [((P2.mb == b) * (P2.B - P2.q)).values for b in bins])
wt = np.clip(np.linalg.lstsq(Xd, (P2.y - P2.q).values, rcond=None)[0], 0, 1)
print("  E3: y = %.3f + %.3f q | E4t weights A by k-bin %s, B by m-bin %s" % (a1, b1, dict(zip(bins, np.round(wt[:4], 2))), dict(zip(bins, np.round(wt[4:], 2)))))
wk = {kb: wfit(P[(P.kb == kb) & P.m.isna()], "A") for kb in bins}; wm = {kb: wfit(P[(P.mb == kb) & P.k.isna()], "B") for kb in bins}
wg = {}
for gb in ("<=4", "5-7", "8-10"):
    F = P[P.gb == gb]; wg[gb] = ls_w((F.I - F.q).values, (F.y - F.q).values)
print("  E4 w_k", {k: round(v, 2) for k, v in wk.items()}, "| E4n w_m", {k: round(v, 2) for k, v in wm.items()}, "| E2w w_gap", {k: round(v, 2) for k, v in wg.items()})
A0, B0 = float(EC[idx["F13_217"], 23]), float(EC[idx["F47_229"], 0])
EV = [("F13_221", 1.615, 2, 7), ("F13_223", 1.668, 3, 6), ("F13_225", 1.796, 4, 5), ("F47_221", 1.401, 5, 4), ("F47_223", 1.431, 6, 3), ("F47_225", 1.688, 7, 2), ("F47_227", 1.657, 8, 1)]
S13 = pd.read_csv(r"C:\work\farmai\제출\13회차_2026-10-12(나)\submission_22.csv"); S13["rec"] = S13.row_id.str[:7]; P13 = S13.groupby("rec").sub_ec.mean()
rows = []
for r, _, k, m in EV:
    q = float(P13[r]); I = A0 + (B0 - A0) * k / (k + m); kb = str(KB(np.array([k]))[0]); mb = str(KB(np.array([m]))[0]); gb = str(GB(np.array([k + m]))[0])
    e3 = a1 + b1 * q
    e4t = q + wt[bins.index(kb)] * (A0 - q) + wt[4 + bins.index(mb)] * (B0 - q)
    rows.append(dict(rec=r, k=k, m=m, q13=q, E1_A=A0, E1n_B=B0, E2_I=I, E2w=q + wg[gb] * (I - q), E3=e3, E4=q + wk[kb] * (A0 - q), E4n=q + wm[mb] * (B0 - q), E4t=e4t))
EVT = pd.DataFrame(rows); print(EVT.round(3).to_string(index=False))
# F13_237: same-date training
i237 = idx["F13_237"]; sd = same_date(i237, TRIDX)
print("  F13_237: same-date training", [(rid(j), round(YM[j], 3)) for j in sd], "E5 = %.3f | q13 %.3f | E3 %.3f" % (np.mean([YM[j] for j in sd]), P13["F13_237"], a1 + b1 * P13["F13_237"]))
EVT.to_csv(os.path.join(OUT, "hx1_eval8_estimates_v1.csv"), index=False)
