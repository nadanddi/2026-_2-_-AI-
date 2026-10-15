# -*- coding: utf-8 -*-
"""FL4 (2026-10-15 집 클로드; 5점 모델 27번).  사전 고정(커밋 후 실행).  EC + 온도.
(A) EV3L-FL: EV3L(6.921) 의 전날 창 링크를 현장형 링커 FL(fl3_train_links.csv, p ≥ .5)로 바꿔 SZ2 폴드(62)에 다시 학습.
    판정(EC 분야 규칙, α .025): F5E-FL = .9·NH + .1·EV3L-FL  vs  NH(EV3L 없음) — EV3L-FL 이 제 몫을 하는가.
    서술: F5E(v15R 링크) 대비, 그리고 F5E-FL + FL 어제 EC 앵커(FL2 표 fl2_cv_FL 대신 FL3 학습 표 p ≥ .8·정답 전날·|dT| ≤ 1.5) vs S·SH.
(B) 사슬-FL: 사슬 새벽 보정의 전날 후보(지금: 같은 농장 앞 1~3 번호 중 이어짐 최선)를 FL 학습 표의 짝(p ≥ .5)으로 바꾼 사슬표 CTF.
    v23 틀에서 lit 없는 사슬(A3 Y2 꼴) + 앵커 LV: CTF vs CT(번호 후보), 그리고 사슬 없음(NOCH) — 판 DIAG10·EL1·F·G·H·I.
    판정(α .025): CTF 사슬 vs NOCH 가 모든 셀 개선 & 합친 P < .025 이면 '사슬-FL 유지', 아니면 27번에서 사슬 제거.  서술: CTF vs CT.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u fl4_ev3l_chain_with_fl_v1.py {ec|temp}
"""
import os, sys, pickle, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MODE = sys.argv[1] if len(sys.argv) > 1 else "ec"
import env  # noqa: F401
import numpy as np, pandas as pd
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv"))
if MODE == "ec":
    src = open(os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ev3l_ec_linked_yesterday_gru_v1.py")
    exec(src[:src.index("\nO = pd.read_csv")])
    __file__ = os.path.join(HERE, "fl4_ev3l_chain_with_fl_v1.py")
    LINK = {r: p for r, p, q in zip(FLT.rec, FLT.prev, FLT.p) if q >= .5 and p != r and p in idx}
    C48 = np.full((N, 48, CUBE.shape[2]), np.nan, np.float32); C48[:, 24:] = CUBE
    for r, p in LINK.items(): C48[idx[r], :24] = CUBE[idx[p]]
    OTH = {"F13": "F47", "F47": "F13"}; SEEDS = (1357, 2468, 3579)
    R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); EF = os.path.join(env.LOCAL, "fl4_ev3lf_v1.pkl"); EPF = pickle.load(open(EF, "rb")) if os.path.exists(EF) else {}
    for (v, k), G in R.groupby(["validator", "fold"]):
        if (v, k) in EPF: continue
        te = sorted(set(G.rec)); vd = {(r[:3], int(r[4:])) for r in te}
        hide = {r for r in LABREC if any((r[:3] == f and abs(int(r[4:]) - d) <= 1) or (r[:3] == OTH[f] and abs(int(r[4:]) - d) <= 3) for f, d in vd)} | set(te)
        p = fit48([r for r in LABREC if r not in hide], te, C48); EPF[(v, k)] = {r: p[j] for j, r in enumerate(te)}; pickle.dump(EPF, open(EF, "wb"))
    print("EV3L-FL fits done", flush=True)
    MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); EP = pickle.load(open(os.path.join(env.LOCAL, "nh2_ev3l_v1.pkl"), "rb"))
    mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)]); ev = np.array([EP[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
    evf = np.array([EPF[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
    AN = {r: p for r, p, q in zip(FLT.rec, FLT.prev, FLT.p) if q >= .8 and p in idx and TRAIN[idx[p]]}
    _X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
    _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int); TIN = _X.pivot(index="rec", columns="h", values="in_temp")
    AN = {r: p for r, p in AN.items() if abs(TIN.at[r, 0] - TIN.at[p, 23]) <= 1.5}
    for s in SEEDS:
        nh = (R[f"A1_{s}"] - .1 * mt) / .9; R[f"NH_{s}"] = nh; R[f"F5E_{s}"] = .9 * nh + .1 * ev; R[f"F5EF_{s}"] = .9 * nh + .1 * evf
        out = R[f"F5EF_{s}"].values.copy()
        for (v, k, r), g in R.groupby(["validator", "fold", "rec"]):
            p = AN.get(r)
            if p is None or not np.isfinite(EC[idx[p], 23]) or p in set(R.rec[(R.validator == v) & (R.fold == k)]): continue
            ii = g.sort_values("hour").index.values; q = R.loc[ii, f"F5EF_{s}"].values; pm = np.cumsum(q) / np.arange(1, len(q) + 1); out[ii] = q + (EC[idx[p], 23] - pm)
        R[f"FULLF_{s}"] = out
    rows, blocks, recd = [], [], []
    for v in ("DIAG10", "EL1", "P2LOO"):
        g = R[R.validator == v]; y = g.y.values; p2 = g.p2.values; key = (g.farm + "_" + (g.day // 5).astype(str)).values
        for s in SEEDS:
            a, n = g[f"NH_{s}"].values, g[f"F5EF_{s}"].values; d = (n - y) ** 2 - (a - y) ** 2
            f = lambda c, ref, m: 100 * (rm((g[c].values - y)[m]) / rm((g[ref].values - y)[m]) - 1)
            rows.append(dict(val=v, seed=s, dSSE_p2=float(d[p2].sum()), dSSE_all=float(d.sum()), F5EF_vs_NH_p2=f(f"F5EF_{s}", f"NH_{s}", p2), F5EF_vs_F5E_p2=f(f"F5EF_{s}", f"F5E_{s}", p2),
                             FULLF_vs_S_p2=f(f"FULLF_{s}", f"S_{s}", p2), FULLF_vs_SH_p2=f(f"FULLF_{s}", f"SH_{s}", p2), FULLF_vs_SH_all=f(f"FULLF_{s}", f"SH_{s}", np.ones(len(y), bool))))
            if v == "P2LOO": blocks.append(pd.Series(d[p2]).groupby(key[p2]).mean())
            for r_, x in pd.Series(d[p2]).groupby(g.rec.values[p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=x))
    S = pd.DataFrame(rows); pd.set_option("display.width", 240); print(S.round(3).to_string(index=False))
    main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); x = pd.concat(blocks, axis=1).mean(axis=1).values
    pP = float((x[np.random.default_rng(20270110).integers(0, len(x), (20000, len(x)))].mean(1) >= 0).mean()); RD = pd.DataFrame(recd)
    nh_ = int((RD[RD.val.isin(["P2LOO", "EL1"])].groupby(["val", "seed", "rec"]).d.sum() > 1e-6).sum()); c3 = not (S[S.val == "DIAG10"].dSSE_all.mean() > 0)
    print(f"(A) EV3L-FL: cells {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P {pP:.4f} | harmed {nh_} | DIAG10 {S[S.val == 'DIAG10'].dSSE_all.mean():+.3f} -> {'PASS' if c1 and (pP < .025 or nh_ == 0) and c3 else 'FAIL'}")
    sys.exit(0)
# ---------------- temp chain ----------------
src = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3fl4"}
exec(src[:src.index("JUDGE = [")], g)
T, B, C2, anch, LV, CT, rmf, IN23, EVE = g["T"], g["B"], g["C2"], g["anch"], g["LV"], g["CT"], g["rm"], g["IN23"], g["EVE"]
LF = {r: p for r, p, q in zip(FLT.rec, FLT.prev, FLT.p) if q >= .5}
CTF = CT.copy(); CTF["ch_in23"] = np.nan; CTF["ch_eve"] = np.nan; CTF["ch_cost"] = np.nan
for i, r in CTF.iterrows():
    k = f"{r.farm}_{int(r.day):03d}"
    if k in LF and LF[k] in IN23.index: CTF.at[i, "ch_in23"] = IN23[LF[k]]; CTF.at[i, "ch_eve"] = EVE[LF[k]]; CTF.at[i, "ch_cost"] = 0.0
FEAT0 = list(C2.T.FEAT); FEATNL = [f for f in FEAT0 if f not in ("lit1_in23", "lit2_in23")]
D1 = T.layout_frame("DIAG10", CT); D1F = T.layout_frame("DIAG10", CTF)
rng = np.random.default_rng(20270111); JUDGE = ["DIAG10", "EL1", "F", "G", "H", "I"]; ok = True; diffs = []
for L in JUDGE:
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); MF = T.layout_frame(L, CTF); y = M.sub_temp.values; key = (M.farm + "_" + (M.day // 5).astype(str)).values; P = {"NOCH": [], "CT": [], "CTF": []}
    for bi, b in enumerate(bs):
        for fam in "AB":
            a = B.w40(M, b, cx, fam); P["NOCH"].append(anch(M, a, LV)); C2.T.FEAT[:] = FEATNL
            P["CT"].append(anch(M, C2.corrected(M, a, D1, bi, fam, 7), LV)); P["CTF"].append(anch(MF, C2.corrected(MF, B.w40(MF, b, cx, fam), D1F, bi, fam, 7), LV)); C2.T.FEAT[:] = FEAT0
    ch = [rmf(n - y) / rmf(b_ - y) - 1 for n, b_ in zip(P["CTF"], P["NOCH"])]; chc = [rmf(n - y) / rmf(b_ - y) - 1 for n, b_ in zip(P["CTF"], P["CT"])]
    ok &= all(x < 0 for x in ch)
    if L != "EL1": diffs.append(pd.Series((P["CTF"][0] - y) ** 2 - (P["NOCH"][0] - y) ** 2).groupby(key).mean())
    print(f"(B) {L:6s} chain-FL vs no-chain {' '.join('%+.2f' % (100 * x) for x in ch)} | vs chain-number-candidates {np.mean(chc) * 100:+.2f}", flush=True)
S = pd.concat(diffs, axis=1).mean(axis=1).values; P_ = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
print(f"(B) pooled P {P_:.4f} -> {'KEEP chain-FL' if ok and P_ < .025 else 'DROP chain'}")
