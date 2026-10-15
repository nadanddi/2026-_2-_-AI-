# -*- coding: utf-8 -*-
"""NH2 (2026-10-15 집 클로드; 사용자: '2(EV3L 을 구성에 넣는 방식) 먼저 진행').  사전 고정(커밋 후 실행).  EC.
5점(현장 정보) EC 구성 후보 F5E = 0.9·NH + 0.1·EV3L,  NH = pre_SEA + .5·HR5  (= 날짜 계절 + R3/TabPFN + HR5 + 같은 동 전날 GRU; SG2·B5·W2 없음).
비교 = S = fin_SEA (SG2 포함 구조판, 지금 EC 의 기본 단계).  NH1 A1(= 0.9·NH + 0.1·MT1) 은 6/6칸 개선이나 P .49 FAIL(6.901).
독립 자료: SZ2 체크포인트(새 시드 1357·2468·3579, 새 PFN 문맥, SZ2 폴드 62).  NH 는 NH1 저장 행(nh1_rows_v1.csv.gz) 에서 되살림:
  NH_s = (A1_s − 0.1·MT1) / 0.9  (MT1 = local/nh1_mt1_v1.pkl, NH1 과 같은 값 → 수치 동일).
EV3L: ev3l 스크립트와 같은 코드·링커(v15R 거울 p0 ≥ .5)·학습, SZ2 폴드마다 다시 학습(숨김 = NH1 과 같음: 검증 날 + 같은 농장 ±1 + 다른 농장 ±3) → local/nh2_ev3l_v1.pkl.
판정(EC 분야 배정 규칙, F5E vs S, 단일 α .025): (1) P2LOO·EL1 3시드 모두 2차 dSSE < 0 & (P2LOO 블록 P(악화) < .025 OR 나빠진 2차 기록-칸 0); (3) DIAG10 손해 감시.
  [링커 사용 → EV 견고성 미검, 한계]
서술: F5E vs A1(NHM), vs SH(S + HR5 = 15회차형), 장부 제외, 보통/고EC.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u nh2_field5_ec_ev3l_formal_v1.py
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py")
exec(src[:src.index("\nO = pd.read_csv")])
__file__ = os.path.join(HERE, "nh2_field5_ec_ev3l_formal_v1.py")
OTH = {"F13": "F47", "F47": "F13"}; SEEDS = (1357, 2468, 3579)
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb"))
EF = os.path.join(env.LOCAL, "nh2_ev3l_v1.pkl"); EP = pickle.load(open(EF, "rb")) if os.path.exists(EF) else {}
for (v, k), G in R.groupby(["validator", "fold"]):
    if (v, k) in EP: continue
    te = sorted(set(G.rec)); vd = {(r[:3], int(r[4:])) for r in te}
    hide = {r for r in LABREC if any((r[:3] == f and abs(int(r[4:]) - d) <= 1) or (r[:3] == OTH[f] and abs(int(r[4:]) - d) <= 3) for f, d in vd)} | set(te)
    tr = [r for r in LABREC if r not in hide]; p = fit48(tr, te, C48); EP[(v, k)] = {r: p[j] for j, r in enumerate(te)}
    pickle.dump(EP, open(EF, "wb")); print("EV3L", v, k, "done", flush=True)
mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
ev = np.array([EP[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
for s in SEEDS:
    R[f"NH_{s}"] = (R[f"A1_{s}"] - .1 * mt) / .9; R[f"F5E_{s}"] = .9 * R[f"NH_{s}"] + .1 * ev
LEDGER = {"F13_231", "F13_233", "F47_216", "F47_229", "F47_241", "F13_217", "F13_243", "F47_231"}
rows, blocks, recd = [], {}, []
for v in ("DIAG10", "EL1", "P2LOO"):
    g = R[R.validator == v]; y = g.y.values; p2 = g.p2.values; hi = g.hi.values; key = (g.farm + "_" + (g.day // 5).astype(str)).values; led = ~g.rec.isin(LEDGER).values
    for s in SEEDS:
        a, n = g[f"S_{s}"].values, g[f"F5E_{s}"].values; d = (n - y) ** 2 - (a - y) ** 2
        fx = lambda ref, m: 100 * (rm((n - y)[m]) / rm((g[ref].values - y)[m]) - 1) if m.any() else np.nan
        rows.append(dict(val=v, seed=s, dSSE_p2=float(d[p2].sum()), dSSE_all=float(d.sum()), vsS_all=fx(f"S_{s}", np.ones(len(y), bool)), vsS_p2=fx(f"S_{s}", p2),
                         vsS_p2_normal=fx(f"S_{s}", p2 & ~hi), vsS_p2_high=fx(f"S_{s}", p2 & hi), vsNHM_p2=fx(f"A1_{s}", p2), vsSH_p2=fx(f"SH_{s}", p2), vsSH_all=fx(f"SH_{s}", np.ones(len(y), bool)),
                         X0_p2=float(d[p2 & led].sum())))
        blocks.setdefault((v, "p2"), []).append(pd.Series(d[p2]).groupby(key[p2]).mean()); blocks.setdefault((v, "all"), []).append(pd.Series(d).groupby(key).mean())
        for r_, dd in pd.Series(d[p2]).groupby(g.rec.values[p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, dsse=dd))
S = pd.DataFrame(rows); pd.set_option("display.width", 240); print(S.round(3).to_string(index=False))
def boot(key_):
    x = pd.concat(blocks[key_], axis=1).mean(axis=1).values; m = x[np.random.default_rng(20261226).integers(0, len(x), (20000, len(x)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP, _ = boot(("P2LOO", "p2"))
RD = pd.DataFrame(recd); harmed = RD[RD.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).dsse.sum(); nh = int((harmed > 1e-6).sum())
dg = S[S.val == "DIAG10"]; _, pb = boot(("DIAG10", "all")); c3 = not (dg.dSSE_all.mean() > 0 and pb < .025)
print(f"\n(1) main cells better {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P(worse) {pP:.4f} | harmed record-cells {nh} | (3) DIAG10 mean dSSE {dg.dSSE_all.mean():+.3f} share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
print("P2LOO worst records:", RD[RD.val == "P2LOO"].groupby("rec").dsse.mean().sort_values().round(3).tail(6).to_dict())
print("NH2:", "ADOPT (F5E replaces S)" if (c1 and (pP < .025 or nh == 0) and c3) else "FAIL")
