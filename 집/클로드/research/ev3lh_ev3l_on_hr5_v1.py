# -*- coding: utf-8 -*-
"""EV3LH (2026-10-15 집 클로드; 사용자 취침 중).  사전 고정(커밋 후 실행).  EC.
EV3L(ev3l_v1.log ADOPT): EC 2층 GRU + 링커 같은 동 전날 창, 0.9·sg + 0.1·EV3L 이 오늘만 GRU(E0) 섞기보다 2차 6/6칸 개선, P2LOO P .008, DIAG10 손해 감시 PASS.
지금 EC 구성은 sg 위에 HR5(λ .5) 가 붙음(15회차).  EV3L 이득이 HR5 위에서도 남는가:
  기준 = HR5 행(hr5_rows_v1.csv.gz 'new' = sg + .5·HR5, 같은 ec_rv1 배치·시드),  후보 = 0.9·HR5 + 0.1·EV3L.
  EV3L 은 ev3l 스크립트와 같은 코드·링커·숨김으로 다시 학습(예측 저장 — 첫 실행은 저장 안 했음; GRU 내부 시드 같음).
판정(EC 분야 배정 규칙, 후보 vs HR5): (1) P2LOO·EL1 3시드 모두 2차 dSSE < 0 & (P2LOO 블록 P(악화) < .025 OR 나빠진 2차 기록-칸 0); (3) DIAG10 손해 감시.
  [링커 사용 → EV 견고성 미검(ec_rv1 에 EV 없음), 한계로 보고]
서술: 장부 제외 dSSE, 보통/고EC, sg 대비 %.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ev3lh_ev3l_on_hr5_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py")
exec(src[:src.index("res, B = [], []")].replace('PRED = {"EV3L": {}, "E0": {}}', 'PRED = {"EV3L": {}}').replace('for kind, CC in (("EV3L", C48), ("E0", C00)):', 'for kind, CC in (("EV3L", C48),):'))
__file__ = os.path.join(HERE, "ev3lh_ev3l_on_hr5_v1.py")
pd.to_pickle(PRED, os.path.join(env.LOCAL, "ev3l_preds_v1.pkl"))
H = pd.read_csv(os.path.join(HERE, "hr5_rows_v1.csv.gz"))
H["ev"] = [PRED["EV3L"][(v, r)][h] for v, r, h in zip(H.validator, H.rec, H.hour)]
H["cand"] = .9 * H.new + .1 * H.ev; H["d"] = (H.cand - H.y) ** 2 - (H.new - H.y) ** 2; H["p2"] = H.day >= 179; H["hi"] = H.groupby(["validator", "seed", "rec"]).y.transform("mean") >= 1.0
LEDGER = {"F13_231", "F13_233", "F47_216", "F47_229", "F47_241", "F13_217", "F13_243", "F47_231"}
res = []
for (v, sd), g in H.groupby(["validator", "seed"]):
    p2 = g.p2.values; f = lambda m: 100 * (rm((g.cand - g.y)[m]) / rm((g.new - g.y)[m]) - 1) if m.any() else np.nan
    res.append(dict(val=v, seed=sd, dSSE_p2=float(g.d[p2].sum()), p2=f(p2), all=f(np.ones(len(g), bool)), p2_normal=f(p2 & ~g.hi.values), p2_high=f(p2 & g.hi.values),
                    cand_vs_sg=100 * (rm(g.cand - g.y) / rm(g.base - g.y) - 1), hr5_vs_sg=100 * (rm(g.new - g.y) / rm(g.base - g.y) - 1), X0_p2=float(g.d[p2 & ~g.rec.isin(LEDGER).values].sum())))
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
def boot(v, p2only):
    g = H[(H.validator == v) & (H.p2 if p2only else True)]; s = g.groupby(["seed", g.farm + "_" + (g.day // 5).astype(str)]).d.mean().groupby(level=1).mean().values
    m = s[np.random.default_rng(20270102).integers(0, len(s), (20000, len(s)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP, _ = boot("P2LOO", True)
RD = H[H.validator.isin(["P2LOO", "EL1"]) & H.p2].groupby(["validator", "seed", "rec"]).d.sum(); nh = int((RD > 1e-6).sum())
_, pb = boot("DIAG10", False); c3 = not (H[H.validator == "DIAG10"].groupby("seed").d.sum().mean() > 0 and pb < .025)
print(f"\n(1) main cells better {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P(worse) {pP:.4f} | harmed record-cells {nh} | (3) DIAG10 share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
print("record dSSE (seed mean) best/worst:", H.groupby("rec").d.sum().div(3).sort_values().round(3).head(5).to_dict(), H.groupby("rec").d.sum().div(3).sort_values().round(3).tail(5).to_dict())
print("EV3LH:", "ADOPT" if (c1 and (pP < .025 or nh == 0) and c3) else "FAIL")
