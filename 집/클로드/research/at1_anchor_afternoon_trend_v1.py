# -*- coding: utf-8 -*-
"""AT1 (2026-10-14 집 클로드; 사용자: '앵커에 전날 오후 추세 넣어서 해봐').  사전 고정(커밋 후 실행).  EC.  과거 정보만.
DG2(6.858): 전날 24시간 곡선에 23시 값 외 정보가 조금 있음(CV R2 .77 -> .82, 오후 추세 상관 .47).
기준 V7 = v7 확신 앵커(EA2: P_h = sg_h + (A − cm_h), A = 짝 23시 EC, cm_h = sg 0..h 누적평균).
후보 AT = 같은 기록(v7 앵커 기록만 — 새 짝 없음)에서 P_h = sg_h + c0 (A − cm_h) + c1 (A − A12),  A12 = 짝 12시 EC.
  c0·c1 = DG2 의 학습 확실한 1일 쌍(cp2 1일 연결, 잠금 제외)에서 '오늘 하루 평균 잔차(y − 기본 OOF) ~ (A − 오늘 기본 평균) + (A − A12)' 선형 회귀(절편 없음) — 검증기 결과로 고르지 않음.
  [한계 명시] 계수 2개는 학습 쌍 전체로 맞춰 검증 기록의 정답도 포함(폴드 밖 적합 아님) — 매개변수 2개라 누수는 작지만 낙관 쪽.
판정(EC 분야 배정 규칙, 링커 의존 — 짝은 기준과 같으나 v7 짝 사용, k = 1): (1) P2LOO·EL1 3시드 모두 2차 dSSE(vs V7) < 0 AND (P2LOO 블록 P(악화) < .025 OR 나빠진 2차 기록 없음);
  (2) EV 3시드 모두 전체 행 dSSE <= 0;  (3) DIAG10 손해 감시.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261210); lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
D2 = pd.read_csv(os.path.join(HERE, "dg2_pairs_v1.csv"))
Xc = np.c_[D2.F0.values, D2.F1.values]; c0, c1 = np.linalg.lstsq(Xc, D2.r.values, rcond=None)[0]
print(f"fitted on {len(D2)} training pairs: c0 {c0:.3f} (anchor), c1 {c1:.3f} (afternoon trend A23 - A12)", flush=True)
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
res, recd, blocks = [], [], {}
def run(Gv, y, sg, cm, recrows, prevvis, tag, sd):
    base = sg.copy(); P = sg.copy()
    for r, ix in recrows.items():
        if r not in prevvis: continue
        a = idx[prevvis[r]]; A, A12 = EC[a, 23], EC[a, 12]
        base[ix] = np.clip(sg[ix] + (A - cm[ix]), lo_, hi_); P[ix] = np.clip(sg[ix] + c0 * (A - cm[ix]) + c1 * (A - A12), lo_, hi_)
    p2 = Gv.day.values >= 179; d = (P - y) ** 2 - (base - y) ** 2
    res.append(dict(val=tag, seed=str(sd), dSSE_all=float(d.sum()), dSSE_p2=float(d[p2].sum()), all_pct=100 * (rm(P - y) / rm(base - y) - 1), p2_pct=100 * (rm((P - y)[p2]) / rm((base - y)[p2]) - 1), anchored=len(prevvis)))
    bk = (Gv.farm + "_" + (Gv.day // 5).astype(str)).values
    blocks.setdefault((tag, "all"), []).append(pd.Series(d).groupby(bk).mean()); blocks.setdefault((tag, "p2"), []).append(pd.Series(d[p2]).groupby(bk[p2]).mean())
    for r_, v in pd.Series(d[p2]).groupby(Gv.rec.values[p2]).sum().items(): recd.append(dict(val=tag, seed=str(sd), rec=r_, dsse=v))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values
    cm = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict(); recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    prevvis = {r: p for r, p in V7.items() if r in recrows and p not in hidden.get(fold_of[r], set())}
    run(Gv, y, sg, cm, recrows, prevvis, v, sd)
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.y.values; sg = Gv.SG2.values
    cm = Gv.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    anchored = set(Gv.rec[Gv.anchored.astype(bool)]); prevvis = {r: V7[r] for r in anchored if r in V7}
    run(Gv, y, sg, cm, recrows, prevvis, "EV", sd)
S = pd.DataFrame(res); RD = pd.DataFrame(recd); pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
def boot(key):
    s = pd.concat(blocks[key], axis=1).mean(axis=1).values; ix = rng.integers(0, len(s), (20000, len(s))); m = s[ix].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1_ = bool((main.dSSE_p2 < 0).all()); pP2, _ = boot(("P2LOO", "p2")); harmed = RD[RD.val.isin(["P2LOO", "EL1"]) & (RD.dsse > 1e-6)]
c1b = pP2 < .025 or len(harmed) == 0
print(f"\n(1) main: cells better {int((main.dSSE_p2 < 0).sum())}/{len(main)} | P2LOO P(worse) {pP2:.4f} | harmed record-cells {len(harmed)} {harmed.groupby('rec').dsse.max().round(3).to_dict()} -> {'PASS' if c1_ and c1b else 'FAIL'}")
ev = S[S.val == "EV"]; c2 = bool((ev.dSSE_all <= 0).all()); print(f"(2) EV: all-rows dSSE {ev.dSSE_all.round(3).tolist()} -> {'PASS' if c2 else 'FAIL'}")
dg = S[S.val == "DIAG10"]; _, pb = boot(("DIAG10", "all")); c3 = not (dg.dSSE_all.mean() > 0 and pb < .025); print(f"(3) DIAG10 guard: mean dSSE {dg.dSSE_all.mean():+.3f} share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
g = RD.groupby(["val", "rec"]).dsse.mean(); print("pass-2 record changes:", g[g.abs() > 1e-6].round(3).to_dict())
print("AT1:", "ADOPT" if (c1_ and c1b and c2 and c3) else "FAIL")
