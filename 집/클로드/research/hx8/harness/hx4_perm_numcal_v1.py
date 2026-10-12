# -*- coding: utf-8 -*-
"""HX4 harness (2026-10-13 집 클로드; HX3 후속 점검).  진단·서술.
점검 이유: 확실 연결(truth.pkl cert)은 자정 EC·배지온도 연속성 비용(linker_v3/t1: |ΔEC|/.01 + |ΔST|/.3 ≤ 1)으로 고른 것이라
  d=1 앵커(앞 23시·뒤 0시)가 대상의 경계 시각 값과 거의 같도록 '선택'됨 → HX3 의 d=1 이득(−52~−58%)은 상한일 수 있음.
  평가 매핑 연결은 EC 로 고르지 않은 번호 구조(6.549)이고 결번(F13_219, F47_228)을 건넘(6.549(d) 경고 배치).
(a) 순열 기준선: HX3 주 사례(cert, Sq, 한쪽 앵커)에서 (쪽, 거리 구간) 칸 안의 앵커 값을 '다른 성분' 사례의 앵커로 무작위 교체 1000회 →
    E1(앵커 그대로)·E4f(q + w·(앵커 − q), w = HX3 전체 적합 값) RMSE 분포 vs 실제.  p = 순열 RMSE ≤ 실제 비율.
    해석: p 작음 = 같은 동 앵커가 '고EC 일반 수준' 이상의 정보를 줌.
(b) 번호+달력 검증기(평가 매핑과 같은 연결 방식, EC 비선택): 대상 r(Sq: 시드 평균 q ≥ 1, 또는 전체 학습 OOF 기록)의 같은 ID 번호 −2k(앞)/+2m(뒤) 학습 기록이
    학습 전용 달력에서 같은 seg·정확히 k(m)일 차이면 앵커.  k, m = 1..4.  cert 연결과 일치율(같은 동 신뢰도) 보고.
    추정기 E0, E1, E4(쪽·거리별 w, seg 하나 빼기 교차적합), E4h(HX3 cert 전체 적합 w 를 그대로 적용 = 평가 적용과 같은 계수).
    교차적합 단위 = 학습 전용 달력 seg (번호 연결은 성분을 만들지 않으므로), 부트스트랩 단위 = seg.
판독: 서술(채택 판정 아님).  표본 작음을 명시.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out_v1")
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rng = np.random.default_rng(20261013)
BINS = ("1", "2", "3-4")
dbin = lambda d: "1" if d <= 1 else ("2" if d <= 2 else "3-4")
W_HX3 = {("prev", "1"): 0.89, ("prev", "2"): 0.00, ("prev", "3-4"): 0.00, ("next", "1"): 0.82, ("next", "2"): 0.31, ("next", "3-4"): 0.00}
# ---------------- (a) permutation baseline ----------------
C = pd.read_csv(os.path.join(OUT, "hx3_cases_cert_Sq_v1.csv"))
C = C[C.side != "two"].reset_index(drop=True)
print("================ (a) permutation baseline on HX3 cert-Sq one-sided cases (anchor swapped with other-component case in same cell) ================")
prow = []
for (s, b), G in C.groupby(["side", "db"]):
    if G.comp.nunique() < 2 or len(G) < 3: print(f"  [{s} d={b}] n {len(G)} comps {G.comp.nunique()} - skip"); continue
    w = W_HX3.get((s, b), 0.0)
    y, q, anc, comp = G.y.values, G.q.values, G.anc.values, G.comp.values
    r_e1 = np.sqrt(((anc - y) ** 2).mean()); r_e4 = np.sqrt(((q + w * (anc - q) - y) ** 2).mean()); r_q = np.sqrt(((q - y) ** 2).mean())
    P1, P4 = [], []
    for _ in range(1000):
        sw = np.empty(len(G))
        for j in range(len(G)):
            cand = np.where(comp != comp[j])[0]
            sw[j] = anc[rng.choice(cand)]
        P1.append(np.sqrt(((sw - y) ** 2).mean())); P4.append(np.sqrt(((q + w * (sw - q) - y) ** 2).mean()))
    P1, P4 = np.array(P1), np.array(P4)
    print(f"  [{s} d={b}] n {len(G)} rec {G.rec.nunique()} comp {G.comp.nunique()} | RMSE q {r_q:.3f} | E1 actual {r_e1:.3f} vs perm median {np.median(P1):.3f} (p {np.mean(P1 <= r_e1):.3f})"
          f" | E4f(w {w:.2f}) actual {r_e4:.3f} vs perm median {np.median(P4):.3f} (p {np.mean(P4 <= r_e4):.3f})")
    prow.append(dict(side=s, db=b, n=len(G), ncomp=G.comp.nunique(), rmse_q=r_q, e1=r_e1, e1_perm_med=np.median(P1), p_e1=np.mean(P1 <= r_e1), w=w, e4=r_e4, e4_perm_med=np.median(P4), p_e4=np.mean(P4 <= r_e4)))
pd.DataFrame(prow).to_csv(os.path.join(OUT, "hx4_perm_v1.csv"), index=False)
# ---------------- (b) number + calendar validator ----------------
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); SD = {r: (int(s), int(d)) for r, s, d in zip(CAL.rec, CAL.seg, CAL.day)}
TL = pd.read_pickle(os.path.join(S3, "truth.pkl"))
certpre = {int(b): int(a) for a, b, c in zip(TL.a1, TL.b, TL.cert.astype(bool)) if c and TRAIN[int(a)] and TRAIN[int(b)]}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
QS = O.groupby(["rec", "seed"]).sg.mean().unstack(); SEEDS = list(QS.columns); QS["mean"] = QS[SEEDS].mean(1)
YM = {i: float(np.nanmean(EC[i])) for i in np.where(TRAIN)[0]}
def numcal(r, k, sign):
    """same ID number +-2k training record on same seg & exactly k days apart (training-only calendar)."""
    i = idx[r]; r2 = f"{FARM[i]}_{NUM[i] + sign * 2 * k:03d}"
    if r2 not in SD or r not in SD: return None
    s, d = SD[r]; s2, d2 = SD[r2]
    if s2 != s or (d2 - d) != sign * k: return None
    return idx[r2]
def cert_walk(i, k):
    x = i
    for _ in range(k):
        x = certpre.get(x)
        if x is None: return None
    return x
certsuc = {a: b for b, a in certpre.items()}
def cert_walk_f(i, m):
    x = i
    for _ in range(m):
        x = certsuc.get(x)
        if x is None: return None
    return x
def build(sel):
    rows = []
    for r in sel:
        i = idx[r]
        for k in range(1, 5):
            a = numcal(r, k, -1)
            if a is not None:
                cw = cert_walk(i, k)
                rows.append(dict(rec=r, seg=SD[r][0], side="prev", d=k, anc=float(EC[a, 23]), ra=rid(a), y=YM[i], agree=(cw == a) if cw is not None else np.nan))
            b = numcal(r, k, +1)
            if b is not None:
                cw = cert_walk_f(i, k)
                rows.append(dict(rec=r, seg=SD[r][0], side="next", d=k, anc=float(EC[b, 0]), ra=rid(b), y=YM[i], agree=(cw == b) if cw is not None else np.nan))
    D = pd.DataFrame(rows); D["db"] = [dbin(d) for d in D.d]
    return D
def ls_w(x, r):
    d = float((x * x).sum()); return float(np.clip((x * r).sum() / d, 0, 1)) if d > 0 else 0.0
def run(D, qcol):
    D = D.copy(); D["q"] = [float(QS.loc[r, qcol]) for r in D.rec]
    D["E0"] = D.q; D["E1"] = D.anc
    D["E4h"] = D.q + np.array([W_HX3[(s, b)] for s, b in zip(D.side, D.db)]) * (D.anc - D.q)
    e4 = pd.Series(np.nan, index=D.index)
    for u in D.seg.unique():
        F = D[D.seg != u]
        for (s, b), G in D[D.seg == u].groupby(["side", "db"]):
            H = F[(F.side == s) & (F.db == b)]
            w = ls_w((H.anc - H.q).values, (H.y - H.q).values) if len(H) else 0.0
            e4.loc[G.index] = G.q + w * (G.anc - G.q)
    D["E4"] = e4
    return D
def boot_p(G, c, nb=2000):
    H = pd.DataFrame(dict(u=G.seg.values, s0=((G.E0 - G.y) ** 2).values, sc=((G[c] - G.y) ** 2).values, one=1.0)).groupby("u").sum()
    nc = len(H)
    if nc < 2: return np.nan
    W = rng.multinomial(nc, np.full(nc, 1.0 / nc), size=nb).astype(float); n = W @ H.one.values
    return float(((np.sqrt((W @ H.sc.values) / n) - np.sqrt((W @ H.s0.values) / n)) >= 0).mean())
for sname, sel in (("Sq (q>=1)", [r for r in QS.index if QS.loc[r, "mean"] >= 1.0]), ("all OOF records", list(QS.index))):
    D = build(sel)
    print(f"\n================ (b) number+calendar validator | {sname}: targets {len(sel)} | cases {len(D)} | segs {D.seg.nunique()} ================")
    ag = D.groupby(["side", "db"]).agree.agg(["count", "mean"]); print("  agreement with cert chain where cert chain exists (count, share):", {k: (int(v['count']), round(v['mean'], 2)) for k, v in ag.iterrows()})
    Dm = run(D, "mean"); Ds = {sd: run(D, sd) for sd in SEEDS}
    for (s, b), G in Dm.groupby(["side", "db"]):
        line = []
        for c in ("E0", "E1", "E4", "E4h"):
            rm = np.sqrt(((G[c] - G.y) ** 2).mean()); r0 = np.sqrt(((G.E0 - G.y) ** 2).mean())
            sb = sum(np.sqrt(((Ds[sd].loc[G.index, c] - G.y) ** 2).mean()) < np.sqrt(((Ds[sd].loc[G.index, "E0"] - G.y) ** 2).mean()) for sd in SEEDS)
            pn = boot_p(G, c) if c != "E0" else np.nan
            line.append(f"{c} {rm:.3f}({100*(rm-r0)/r0:+.0f}%, s{sb}/3, P {pn:.2f}, bias {(G[c]-G.y).mean():+.3f})" if c != "E0" else f"E0 {rm:.3f} (bias {(G.E0-G.y).mean():+.3f})")
        print(f"  [{s} d={b}] n {len(G)} rec {G.rec.nunique()} seg {G.seg.nunique()} | " + " | ".join(line))
    if sname.startswith("Sq"): Dm.to_csv(os.path.join(OUT, "hx4_numcal_Sq_v1.csv"), index=False)
