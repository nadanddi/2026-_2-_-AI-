# -*- coding: utf-8 -*-
"""ECT2 (2026-10-16 집 클로드; 사용자 '좀 더 조사·실험' — EC 쪽: AC1(6.925)의 넓힌 EC 앵커가 틀린 연결 F47_132 로 손해 -> TAG2(7.030) 현장형 두 갈래 일치로 거르기).
사전 고정(커밋 후 실행).  EC.  학습 없음(저장 NH 예측).  AC1 틀 그대로(nh1_rows: DIAG10·EL1·P2LOO, 시드 1357·2468·3579, 앵커 전날이 그 폴드 검증 기록이면 제외).
링크 = fl3_train_links(FL, 학습 표) + 같은 후보·가중(fl3_w)에서 갈래 선택 OUT(바깥 날씨 이어짐)·IN(실내 센서·구동기 이어짐) (학습 표 규칙: 같은 농장 번호 ±1 정답 숨김, 평가 기록 후보 제외).
  B   (= 29·31회차 EC) : FL p >= .8 & 정답 전날 & |dT| <= 1.5 -> pred_h = q_h + .8·p·(A − cm_h(q))
  ECT2 (후보)          : B 앵커 ∪ {.5 <= p < .8 & 정답 전날 & |dT| <= 1.5 & OUT 또는 IN 이 같은 전날} , 같은 식(.8·p)
판정(EC 분야 규칙, 단일 안 α .025): P2LOO·EL1 3시드 모두 2차 dSSE < 0 & (P2LOO 블록 P(악화) < .025 OR 나빠진 2차 기록-시드 0) & DIAG10 평균 dSSE 손해 아님.
서술: 새 앵커 기록 수·진짜 비율, F47_132 포함 여부, 새 기록만의 변화.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ect2_ec_field_agreement_anchor_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "ac1_anchor_coverage_ec_v1.py"), encoding="utf-8").read()
exec(compile(src[src.index("import os, sys, pickle"):src.index("ARMS = {")], "ac1pre", "exec"))
__file__ = os.path.join(HERE, "ect2_ec_field_agreement_anchor_v1.py")
import importlib.util
sys.path.insert(0, os.path.join(HERE, "linker_v7")); import feat7 as F7
FLc = ["oj"] + [f"v{j}" for j in range(8)] + ["excl", "xf"]; wv = np.load(os.path.join(env.LOCAL, "fl3_w.npy"))
MO = np.array([1] + [0] * 8 + [1, 1], float); MI = np.array([0] + [1] * 8 + [1, 1], float)
VIEW = {}
for b in range(F7.N):
    if not F7.TRAIN[b] or F7.FARM[b] not in ("F13", "F47"): continue
    lab = F7.TRAIN.copy(); lab[(F7.FARM == F7.FARM[b]) & (np.abs(F7.NUM - F7.NUM[b]) <= 1)] = False
    df, _ = F7.features7(b, 0, lab, removed=set(F7.TEST_IDX))
    if len(df) == 0: continue
    X_ = df[FLc].values.astype(float); a_ = df.a.values
    VIEW[F7.rid(b)] = (F7.rid(int(a_[np.argmax(X_ @ (wv * MO))])), F7.rid(int(a_[np.argmax(X_ @ (wv * MI))])))
TT = pd.read_pickle(os.path.join(env.LOCAL, "linker_v3", "truth.pkl")); TT = TT[TT.cert]
TRUEP = {f"{fb}_{int(nb):03d}": f"{fa}_{int(na):03d}" for fb, nb, fa, na in zip(TT.fb, TT.nb, TT.fa, TT.na)}
A8 = anchors(.8); A5 = anchors(.5)
NEW = {r: v for r, v in A5.items() if r not in A8 and r in VIEW and (VIEW[r][0] == v[0] or VIEW[r][1] == v[0])}
kn = [TRUEP.get(r) == v[0] for r, v in NEW.items() if r in TRUEP]
print("B anchors", len(A8), "| ECT2 new", len(NEW), "| true", sum(kn), "/", len(kn), "known | F47_132 in new:", "F47_132" in NEW, "| rejected by agreement:", len([r for r in A5 if r not in A8]) - len(NEW), flush=True)
ARMS = {"B": A8, "ECT2": {**A8, **NEW}}
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz"))
MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
FOLDREC = {vk: set(G.rec) for vk, G in R.groupby(["validator", "fold"])}
for s in SEEDS:
    R[f"NH_{s}"] = (R[f"A1_{s}"] - .1 * mt) / .9
    for name, AN in ARMS.items():
        out = R[f"NH_{s}"].values.copy()
        for (v, k, r), g in R.groupby(["validator", "fold", "rec"]):
            if r not in AN: continue
            p, q = AN[r]
            if p in FOLDREC[(v, k)]: continue
            ii = g.sort_values("hour").index.values; x = R.loc[ii, f"NH_{s}"].values; pm = np.cumsum(x) / np.arange(1, len(x) + 1)
            out[ii] = x + .8 * q * (EC[idx[p], 23] - pm)
        R[f"{name}_{s}"] = out
rows, blocks, harmed = [], [], 0
for v in ("DIAG10", "EL1", "P2LOO"):
    g = R[R.validator == v]; y = g.y.values; p2 = g.p2.values; key = (g.farm + "_" + (g.day // 5).astype(str)).values; nw = g.rec.isin(set(NEW)).values
    for s in SEEDS:
        d = (g[f"ECT2_{s}"].values - y) ** 2 - (g[f"B_{s}"].values - y) ** 2
        rr = pd.Series(d[p2]).groupby(g.rec.values[p2]).sum(); harmed += int((rr > 1e-9).sum()) if v == "P2LOO" else 0
        rows.append(dict(val=v, seed=s, dSSE_p2=float(d[p2].sum()), dSSE_all=float(d.sum()), p2_pct=100 * (rm((g[f"ECT2_{s}"].values - y)[p2]) / rm((g[f"B_{s}"].values - y)[p2]) - 1),
                         new_pct=100 * (rm((g[f"ECT2_{s}"].values - y)[nw]) / rm((g[f"B_{s}"].values - y)[nw]) - 1) if nw.any() else np.nan))
        if v == "P2LOO": blocks.append(pd.Series(d[p2]).groupby(key[p2]).sum())
    top = pd.Series(np.mean([(g[f"ECT2_{s}"].values - y) ** 2 - (g[f"B_{s}"].values - y) ** 2 for s in SEEDS], 0)).groupby(g.rec.values).sum().sort_values()
    print(v, "most helped", top.head(3).round(3).to_dict(), "| most harmed", top.tail(3).round(3).to_dict())
S = pd.DataFrame(rows); pd.set_option("display.width", 200); print(S.round(4).to_string(index=False))
Bk = pd.concat(blocks, axis=1).mean(axis=1).values; rng = np.random.default_rng(20270201); pP = float((Bk[rng.integers(0, len(Bk), (20000, len(Bk)))].mean(1) >= 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); c3 = S[S.val == "DIAG10"].dSSE_all.mean() <= 0
print(f"ECT2 vs B: P2LOO/EL1 cells {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P {pP:.4f} | harmed P2LOO record-seeds {harmed} | DIAG10 {S[S.val == 'DIAG10'].dSSE_all.mean():+.3f} -> {'PASS' if c1 and (pP < .025 or harmed == 0) and c3 else 'FAIL'}")
