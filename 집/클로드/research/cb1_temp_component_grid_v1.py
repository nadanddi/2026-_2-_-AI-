# -*- coding: utf-8 -*-
"""CB1-T (2026-10-15 집 클로드; 사용자 '부품들끼리 결합하거나 나누거나 다른 부품도 붙여보기').  사전 고정(커밋 후 실행).  온도.  학습 없음(저장 셀·V3LF).
부품 격자 192조합 = GRU 4 x 앵커 집합 3 x 앵커 꼬리 2 x 앵커 강도 2 x TX3 2 x KF3 약한 사슬 2:
  GRU : none | S10 = V3LF 10% 엄격 게이트(실내 10·그날 최저 12) | O10 = 10% 원 게이트(8·10) | S20 = 20% 엄격
  앵커 집합: FL(fl3 p ≥ .8) | FLN(fln1 p ≥ .5) | UNI(FL 있으면 FL, 없으면 FLN ≥ .5)  — 모두 전날이 그 폴드 학습 쪽 정답·|dT| ≤ 1.5
  꼬리: ALL(BETA 24시간) | DAWN(h ≤ 7 만, 이후 0)        강도 w ∈ {.75, 1}:  + w·BETA[h]·clip(A23 − .36 − q0, ±2)
  TX3 on/off(6.931)        KF3: 앵커 없는 기록에 KF3 v2 식(FLN 사슬 ≤ 3단계, 누적 P ≥ .5) on/off
배치: N(고르기) | CK2·EL1(판정) — OS1 v2 저장 셀(N·CK2), YL1 기준 셀(EL1), V3LF = os1_gru_ckpt(EL1 은 gru_el1_v1.py).
고르기: N 4셀 평균 RMSE 최소(같으면 격자 순서상 앞).  기준 R27 = none·FL·ALL·1·TX3 on·KF3 off.
판정(단일 안 α .025): 선택 조합 vs R27 — CK2·EL1 8셀 모두 개선 & 두 배치 합친 블록 P(악화) < .025.   서술: 부품별 주효과(N·CK2·EL1), 상위 10조합.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cb1_temp_component_grid_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])   # O1, B, BETA, SHAPE, B23, S2, L2, H2, BIAS, IDX, ST, IN_, RD, CM, lab, LF, W, LINK(FLN)
__file__ = os.path.join(HERE, "cb1_temp_component_grid_v1.py")
import common
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); FLK = {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p)}
ELF = dict(W.worlds()[7])["EL1"]; CKR = os.path.join(env.LOCAL, "yl1_ref_ckpt")
GRUS = ("none", "S10", "O10", "S20"); ANS = ("FL", "FLN", "UNI"); TAILS = ("ALL", "DAWN"); WS = (.75, 1.0); TXS = (True, False); KFS = (False, True)
COMBOS = list(itertools.product(GRUS, ANS, TAILS, WS, TXS, KFS)); REF = ("none", "FL", "ALL", 1.0, True, False)
def gate(M, lo_g, lo_h):
    g_ = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - lo_g) / 2, 0, 1)); order = M.sort_values(["farm", "day", "hour"]).index
    mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index); h_ = np.where(mm.isna(), 1, np.clip((mm - lo_h) / 2, 0, 1)); return g_ * h_
RES = {}
for L, ckdir, folds, mk in (("N", O1.CKB, LF["N"], O1.masks), ("CK2", O1.CKB, LF["CK2"], O1.masks), ("EL1", CKR, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckdir, f)) for f in sorted(os.listdir(ckdir)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(O1.CKG, f)) for f in sorted(os.listdir(O1.CKG)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    cg = np.nan_to_num(np.clip((10 - M.in_temp.values) / 2, 0, 1), nan=0.0); cm = np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key, h)])
    tx3 = .25 * np.clip((np.nan_to_num(cm, nan=0) - .8) / .4, 0, 1) * cg; v = M.V3LF.values; miss = np.isnan(v)
    GW = {"none": np.zeros(len(M)), "S10": .10 * gate(M, 10, 12), "O10": .10 * gate(M, 8, 10), "S20": .20 * gate(M, 10, 12)}
    TRS, VAL = {}, {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format)); vs = set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format))
        for r in vs: TRS[r], VAL[r] = trs, vs
    def anc(LK, pmin):
        return {r: float(ST[IDX[LK[r][0]], 23]) for r in recs if r in LK and r in TRS and LK[r][1] >= pmin and LK[r][0] in TRS[r] and np.isfinite(ST[IDX[LK[r][0]], 23])
                and abs(IN_[IDX[r], 0, 0] - IN_[IDX[LK[r][0]], 23, 0]) <= 1.5}
    AF, AN = anc(FLK, .8), anc(LINK, .5); ASET = {"FL": AF, "FLN": AN, "UNI": {**AN, **AF}}
    cells = [B.w40(M, "base_R_%d" % s, "codex_R", fam) for s in O1.SEEDS for fam in "AB"]; qbar = np.mean(cells, 0)
    Q = {r: np.array([qbar[pos[r]][h[pos[r]] == t][0] if (h[pos[r]] == t).any() else np.nan for t in range(24)]) for r in recs}
    def kf(anchored):
        out = {}
        for r in recs:
            if r in anchored or r not in TRS: continue
            trs, vs = TRS[r], VAL[r]; chain, P, cur = [], 1.0, r
            for step in range(3):
                if cur not in LINK: chain = None; break
                pv, p = LINK[cur]; P *= p
                if P < .5: chain = None; break
                chain.append(pv)
                if pv in trs and np.isfinite(ST[IDX[pv], 23]): break
                if pv not in vs or pv not in Q: chain = None; break
                cur = pv
            if not chain or not (chain[-1] in trs and np.isfinite(ST[IDX[chain[-1]], 23])): continue
            est, var = float(ST[IDX[chain[-1]], 23]), L2
            for m in reversed(chain[:-1]): est = Q[m][23] + B23 * ((est - BIAS) - Q[m][0]); var = B23 ** 2 * var + H2
            if np.isfinite(est) and np.isfinite(Q[r][0]): out[r] = float(np.clip(S2 / (S2 + var) * P * ((est - BIAS) - Q[r][0]), -2, 2))
        return out
    KFD = {a: kf(ASET[a]) for a in ANS}; bk = (L + "_" + M.farm + "_" + (M.day // 5).astype(str)).values
    TAILW = {"ALL": BETA[h], "DAWN": np.where(h <= 7, BETA[h], 0.0)}
    for c in COMBOS:
        gn, an, tl, w, tx, kfon = c; preds = []
        for q in cells:
            x = q + GW[gn] * (~miss) * (np.nan_to_num(v) - q); out = x + (tx3 if tx else 0.0)
            for r, A in ASET[an].items():
                ix = pos[r]; x0 = x[ix][h[ix] == 0]
                if len(x0): out[ix] += w * TAILW[tl][ix] * np.clip(A - BIAS - x0[0], -2, 2)
            if kfon:
                for r, gk in KFD[an].items(): ix = pos[r]; out[ix] += SHAPE[h[ix]] * gk
            preds.append(out)
        RES[(L, c)] = preds
    RES[(L, "y")] = y; RES[(L, "bk")] = bk; print(L, "done | anchors FL", len(AF), "FLN", len(AN), "| KF fired", {a: len(KFD[a]) for a in ANS}, flush=True)
score = {c: np.mean([rm(p - RES[("N", "y")]) for p in RES[("N", c)]]) for c in COMBOS}; best = min(COMBOS, key=lambda c: (round(score[c], 6), COMBOS.index(c)))
def rel(L, c): y = RES[(L, "y")]; return 100 * np.mean([rm(p - y) / rm(b - y) - 1 for p, b in zip(RES[(L, c)], RES[(L, REF)])])
T = pd.DataFrame([dict(gru=c[0], anchor=c[1], tail=c[2], w=c[3], tx3=c[4], kf3=c[5], N=rel("N", c), CK2=rel("CK2", c), EL1=rel("EL1", c)) for c in COMBOS])
pd.set_option("display.width", 200); print(T.sort_values("N").head(12).round(3).to_string(index=False))
print("\nmain effects (mean % vs R27):")
for col in ("gru", "anchor", "tail", "w", "tx3", "kf3"): print(" ", col, T.groupby(col)[["N", "CK2", "EL1"]].mean().round(3).to_dict("index"))
print("\nselected on N:", best)
ok = True; diffs = []
for L in ("CK2", "EL1"):
    y = RES[(L, "y")]; ch = [rm(p - y) / rm(b - y) - 1 for p, b in zip(RES[(L, best)], RES[(L, REF)])]; ok &= all(x < 0 for x in ch)
    diffs.append(pd.Series(np.mean([(p - y) ** 2 - (b - y) ** 2 for p, b in zip(RES[(L, best)], RES[(L, REF)])], 0)).groupby(RES[(L, "bk")]).mean())
    print(f"  {L} selected vs R27 {' '.join('%+.3f' % (100 * x) for x in ch)}")
S = pd.concat(diffs).values; P_ = float((S[np.random.default_rng(20270129).integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
print(f"CB1-T {best}: CK2+EL1 8 cells better {ok} | pooled P {P_:.4f} -> {'PASS' if best != REF and ok and P_ < .025 else 'FAIL'}")
