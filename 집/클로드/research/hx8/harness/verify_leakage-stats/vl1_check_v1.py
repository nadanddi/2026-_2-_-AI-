# -*- coding: utf-8 -*-
"""verify_leakage-stats VL1 (2026-10-13 집 클로드, 하네스 HX3/HX4 반박 점검; 진단 전용).
(1) 확실 연결 선택 누수: d=1 앵커 vs 대상 자신의 경계 시각 값(prev: r 0시, next: r 23시) — 앵커가 숨긴 정답의 경계값을 사실상 복사하는지.
(2) 기준 공정성: d>=2 사례에서 '사이 날' 기록이 DIAG10 에서 r 과 다른 폴드(= 모델이 정답을 학습함)인 비율, 그 여부별 q RMSE.
(3) 독점: d=1 이득(E1/E4p)의 ΔSSE 가 상위 1~2 성분에 몰렸는지.
(4) 번호+달력 next d=1 (F47_227 근거) 사례 내역.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import numpy as np, pandas as pd
H = r"C:\work\farmai\집\클로드\research\hx8\harness\out_v1"
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
C = pd.read_csv(os.path.join(H, "hx3_cases_cert_Sq_v1.csv"))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
FOLD = O.groupby("rec").fold.first()
rm = lambda e: float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))
# chain reconstruction (cert, same as HX3 make_links(False))
TL = pd.read_pickle(os.path.join(S3, "truth.pkl"))
pre, cnt = {}, {}
for a, b, c in zip(TL.a1.astype(int), TL.b.astype(int), TL.cert.astype(bool)):
    if TRAIN[a] and TRAIN[b] and c: pre[b] = a; cnt[a] = cnt.get(a, 0) + 1
pre = {b: a for b, a in pre.items() if cnt[a] == 1}; suc = {a: b for b, a in pre.items()}
def between(r, side, d):
    i = idx[r]; M = pre if side == "prev" else suc; out = []
    x = i
    for _ in range(int(d) - 1):
        x = M.get(x); out.append(rid(x))
    return out
one = C[C.side != "two"].copy()
print("=== (1) cert selection leak at d=1: anchor vs target's own boundary value ===")
for s, hcol in (("prev", 0), ("next", 23)):
    G = one[(one.side == s) & (one.d == 1)].copy()
    own = np.array([EC[idx[r], hcol] for r in G.rec])
    print(f"  {s} d=1 n {len(G)} | |anc - own{hcol}h| median {np.median(np.abs(G.anc - own)):.4f} max {np.max(np.abs(G.anc - own)):.4f}"
          f" | RMSE q {rm(G.q - G.y):.3f} E1 {rm(G.E1 - G.y):.3f} own{hcol}h-as-predictor {rm(own - G.y):.3f} E4p {rm(G.E4p - G.y):.3f}")
# all OOF records: own boundary value as predictor (how much a perfect boundary value is worth)
recs = [r for r in FOLD.index if r in idx]
QS = O.groupby("rec").sg.mean()
y_all = np.array([np.nanmean(EC[idx[r]]) for r in recs]); q_all = QS.loc[recs].values
for h in (0, 23):
    v = np.array([EC[idx[r], h] for r in recs]); hi = q_all >= 1.0
    print(f"  all OOF recs q>=1 (n {hi.sum()}): own {h}h-as-predictor RMSE {rm(v[hi] - y_all[hi]):.3f} vs q {rm(q_all[hi] - y_all[hi]):.3f}")
print("\n=== (2) baseline fairness: in-between days seen by model? (DIAG10 fold differs from target => label in training) ===")
for s in ("prev", "next"):
    for d in (2, 3, 4):
        G = one[(one.side == s) & (one.d == d)]
        if not len(G): continue
        seen = []
        for _, c in G.iterrows():
            bt = between(c.rec, s, d)
            seen.append(any(FOLD.get(b) != FOLD[c.rec] for b in bt if b in FOLD.index))
        seen = np.array(seen)
        msg = f"  {s} d={d} n {len(G)} | between-day label seen by model: {seen.sum()}/{len(G)}"
        if seen.any(): msg += f" | q RMSE seen {rm((G.q - G.y)[seen]):.3f}"
        if (~seen).any(): msg += f" | q RMSE hidden {rm((G.q - G.y)[~seen]):.3f} (n {(~seen).sum()})"
        print(msg)
# anchor itself seen by model at d=1 (eval: anchor F47_229 is training => seen)
for s in ("prev", "next"):
    G = one[(one.side == s) & (one.d == 1)]
    seen = np.array([FOLD.get(a) != FOLD[r] for r, a in zip(G.rec, G.ancr)])
    print(f"  {s} d=1 anchor record seen by model {seen.sum()}/{len(G)} | q RMSE seen {rm((G.q-G.y)[seen]):.3f} hidden {rm((G.q-G.y)[~seen]) if (~seen).any() else float('nan'):.3f}")
print("\n=== (3) concentration of d=1 gains (ΔSSE vs q) ===")
for s, est in (("prev", "E1"), ("next", "E4p"), ("next", "E1")):
    G = one[(one.side == s) & (one.d == 1)].copy()
    G["dsse"] = (G[est] - G.y) ** 2 - (G.q - G.y) ** 2
    byc = G.groupby("comp").dsse.sum().sort_values()
    tot = byc.sum()
    print(f"  {s} d=1 {est}: total ΔSSE {tot:+.3f} | top1 comp share {byc.iloc[0]/tot:.2f} top2 {byc.iloc[:2].sum()/tot:.2f} | comps improved {(byc<0).sum()}/{len(byc)}"
          f" | without top2 comps: RMSE q {rm((G.q-G.y)[~G.comp.isin(byc.index[:2])]):.3f} vs {est} {rm((G[est]-G.y)[~G.comp.isin(byc.index[:2])]):.3f}")
    print("     |anc-q| median %.3f ; F47_227 |anc-q13| = %.3f" % (np.median(np.abs(G.anc - G.q)), abs(1.716 - 1.657)))
    sm = G[np.abs(G.anc - G.q) < 0.15]
    if len(sm): print(f"     small-gap cases |anc-q|<.15: n {len(sm)} RMSE q {rm(sm.q-sm.y):.3f} {est} {rm(sm[est]-sm.y):.3f}")
print("\n=== (4) number+calendar next d=1 (Sq) — the non-EC-selected support for F47_227 ===")
D = pd.read_csv(os.path.join(H, "hx4_numcal_Sq_v1.csv"))
G = D[(D.side == "next") & (D.d == 1)].copy()
G["own23"] = [EC[idx[r], 23] for r in G.rec]
print(G[["rec", "seg", "ra", "agree", "y", "q", "anc", "own23", "E4h"]].round(3).to_string(index=False))
G["dsse"] = (G.E4h - G.y) ** 2 - (G.q - G.y) ** 2
print("  ΔSSE by seg:", G.groupby("seg").dsse.sum().round(3).to_dict(), "| by rec top3:", G.sort_values("dsse").head(3)[["rec", "dsse"]].round(3).values.tolist())
nc = G[G.agree != True]
print(f"  non-cert-agreeing cases n {len(nc)} | RMSE q {rm(nc.q-nc.y) if len(nc) else float('nan'):.3f} E4h {rm(nc.E4h-nc.y) if len(nc) else float('nan'):.3f}")
G2 = D[(D.side == "prev") & (D.d == 1)]
print(f"  prev d=1 numcal agree==True n {(G2.agree==True).sum()} E1 RMSE {rm((G2.E1-G2.y)[G2.agree==True]):.3f} vs q {rm((G2.q-G2.y)[G2.agree==True]):.3f} | not agree n {(G2.agree!=True).sum()} E1 {rm((G2.E1-G2.y)[G2.agree!=True]):.3f} q {rm((G2.q-G2.y)[G2.agree!=True]):.3f}")
G3 = D[(D.side == "next") & (D.d == 1)]
print(f"  next d=1 numcal agree==True n {(G3.agree==True).sum()} E1 {rm((G3.E1-G3.y)[G3.agree==True]):.3f} q {rm((G3.q-G3.y)[G3.agree==True]):.3f} | not agree n {(G3.agree!=True).sum()} E1 {rm((G3.E1-G3.y)[G3.agree!=True]):.3f} q {rm((G3.q-G3.y)[G3.agree!=True]):.3f}")
