# -*- coding: utf-8 -*-
"""EA3 (2026-10-15 집 클로드; 사용자: 26번 설계 → '2,3번 진행' 의 3번 — W2 를 링커 어제 EC 앵커로 대체).  사전 고정(커밋 후 실행).  EC.
16회차(SG2·W2·B5 제거) EC +25%: 손해의 큰 몫이 W2(F13_235 +.37). W2 가 하는 일 = '같은 동 어제 EC 에 맞추기'(현장 정보 = 어제 측정값, R),
  문제는 어제를 '번호 −2' 로 찾는 것뿐 → 링커로 찾은 어제(정답 있는 학습 기록만)로 같은 일을 하는 앵커를 F5E 위에 붙여 판정.
F5E = 0.9·NH + 0.1·EV3L (NH2 와 같은 SZ2 독립 기본 예측; nh1_rows + nh1_mt1 + nh2_ev3l).
앵커(EA 식, 6.574): pred_h = p_h + (A − pm_h), A = 짝 23시 EC 정답, pm_h = 그날 0..h 예측 누적 평균.
짝 = 패키지 25 의 MASK 세계 거울 링커(v15r_mirror_masked_pkg25.csv, 26번에서 학습 행 연결로 쓸 것과 같은 규칙) prev0, 짝이 정답 있는 학습 기록이고
  그 폴드의 검증 기록이 아닐 때만.  안 2개(본페로니 k 2 → .0125):  AN8 = p0 ≥ .8,  AN95 = p0 ≥ .95.
판정(EC 분야 배정 규칙, 안 vs F5E): (1) P2LOO·EL1 3시드 모두 2차 dSSE < 0 & (P2LOO 블록 P(악화) < .0125 OR 나빠진 2차 기록-칸 0); (3) DIAG10 손해 감시.
서술: 앵커 받은 기록 수, 장부 제외, 바뀐 기록 상위/하위, 15회차형(SH) 대비.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ea3_ec_linker_yesterday_anchor_f5e_v1.py
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC as ECA, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); SEEDS = (1357, 2468, 3579)
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); EP = pickle.load(open(os.path.join(env.LOCAL, "nh2_ev3l_v1.pkl"), "rb"))
mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)]); ev = np.array([EP[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
for s in SEEDS: R[f"F5E_{s}"] = .9 * ((R[f"A1_{s}"] - .1 * mt) / .9) + .1 * ev
L = pd.read_csv(os.path.join(env.LOCAL, "v15r_mirror_masked_pkg25.csv"))
ARMS = {"AN8": .8, "AN95": .95}; PB = .025 / 2; LEDGER = {"F13_231", "F13_233", "F47_216", "F47_229", "F47_241", "F13_217", "F13_243", "F47_231"}
for arm, th in ARMS.items():
    link = {r: p for r, p, q in zip(L.rec, L.prev0, L.p0) if isinstance(p, str) and q >= th and p in idx and TRAIN[idx[p]] and np.isfinite(ECA[idx[p], 23])}
    n_used = 0
    for s in SEEDS: R[f"{arm}_{s}"] = R[f"F5E_{s}"].values
    for (v, k, r), g in R.groupby(["validator", "fold", "rec"]):
        p = link.get(r)
        if p is None or p in set(R.rec[(R.validator == v) & (R.fold == k)]): continue
        ii = g.sort_values("hour").index.values; n_used += 1
        for s in SEEDS:
            q = R.loc[ii, f"F5E_{s}"].values; pm = np.cumsum(q) / np.arange(1, len(q) + 1); R.loc[ii, f"{arm}_{s}"] = q + (ECA[idx[p], 23] - pm)
    print(f"{arm}: links {len(link)}, anchored record-folds {n_used}", flush=True)
pd.set_option("display.width", 220); VER = {}
for arm in ARMS:
    rows, blocks, recd = [], {}, []
    for v in ("DIAG10", "EL1", "P2LOO"):
        g = R[R.validator == v]; y = g.y.values; p2 = g.p2.values; hi = g.hi.values; key = (g.farm + "_" + (g.day // 5).astype(str)).values; led = ~g.rec.isin(LEDGER).values
        for s in SEEDS:
            a, n = g[f"F5E_{s}"].values, g[f"{arm}_{s}"].values; d = (n - y) ** 2 - (a - y) ** 2
            fx = lambda ref, m: 100 * (rm((n - y)[m]) / rm((g[ref].values - y)[m]) - 1) if m.any() else np.nan
            rows.append(dict(val=v, seed=s, dSSE_p2=float(d[p2].sum()), dSSE_all=float(d.sum()), vsF5E_all=fx(f"F5E_{s}", np.ones(len(y), bool)), vsF5E_p2=fx(f"F5E_{s}", p2),
                             p2_high=fx(f"F5E_{s}", p2 & hi), vsSH_p2=fx(f"SH_{s}", p2), vsSH_all=fx(f"SH_{s}", np.ones(len(y), bool)), X0_p2=float(d[p2 & led].sum())))
            blocks.setdefault((v, "p2"), []).append(pd.Series(d[p2]).groupby(key[p2]).mean()); blocks.setdefault((v, "all"), []).append(pd.Series(d).groupby(key).mean())
            for r_, dd in pd.Series(d).groupby(g.rec.values).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, dsse=dd, p2=int(r_[4:]) >= 179))
    S = pd.DataFrame(rows); print(f"\n=== {arm} vs F5E ==="); print(S.round(3).to_string(index=False))
    def boot(key_):
        x = pd.concat(blocks[key_], axis=1).mean(axis=1).values; m = x[np.random.default_rng(20270103).integers(0, len(x), (20000, len(x)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
    main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP, _ = boot(("P2LOO", "p2"))
    RD = pd.DataFrame(recd); harmed = RD[RD.val.isin(["P2LOO", "EL1"]) & RD.p2].groupby(["val", "seed", "rec"]).dsse.sum(); nh = int((harmed > 1e-6).sum())
    dg = S[S.val == "DIAG10"]; _, pb = boot(("DIAG10", "all")); c3 = not (dg.dSSE_all.mean() > 0 and pb < .025)
    print(f"(1) main cells better {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P(worse) {pP:.4f} (< {PB}) | harmed record-cells {nh} | (3) DIAG10 mean dSSE {dg.dSSE_all.mean():+.3f} share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
    t = RD.groupby("rec").dsse.mean().loc[lambda x: x.abs() > 1e-6].sort_values(); print("changed records best:", t.head(5).round(3).to_dict(), "| worst:", t.tail(5).round(3).to_dict())
    VER[arm] = c1 and (pP < PB or nh == 0) and c3; print(f"EA3 {arm}:", "ADOPT" if VER[arm] else "FAIL")
print("\nEA3:", VER)
