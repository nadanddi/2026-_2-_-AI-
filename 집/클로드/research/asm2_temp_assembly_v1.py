# -*- coding: utf-8 -*-
"""ASM2 (2026-10-15 집 클로드; CB1-T·CB3-T·CB4-T 통과 부품 조립 확인).  사전 고정(커밋 후 실행).  온도.  학습 없음.
조립 NEW = W40G-S 셀 → V3LF 30%·원 게이트(8/10)(CB3-T) → UNI 앵커(FL p≥.8 우선, 없으면 FLN p≥.5; ALL·강도 1)(CB1-T) → TX3(c .40·T0 12·e0 .7)(CB4-T).
비교: R27 = 셀 → FL K8 → TX3(.25·10·.8) (27번 온도);  서술: NEW_oldTX3 = NEW 에서 TX3 만 옛 값(조립 상태에서 TX3 다듬기 몫).
판정(α .025): NEW vs R27 — CK2·EL1 8셀 모두 개선 & 합친 블록 P < .025 & EXT10·EXT12(셀 + GRU + TX3, 앵커 없음) 안전(셀 평균 악화 & P(나음) < .025 이면 FAIL).
서술: N, 추운 행·2차, NEW vs NEW_oldTX3.  한계: 모든 판이 오늘 여러 번 쓰임(고르기는 각 단계에서 다른 판).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u asm2_temp_assembly_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])
__file__ = os.path.join(HERE, "asm2_temp_assembly_v1.py")
import common
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); FLK = {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p)}
ELF = dict(W.worlds()[7])["EL1"]; CKR = os.path.join(env.LOCAL, "yl1_ref_ckpt")
def gate(M, lo_g, lo_h):
    g_ = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - lo_g) / 2, 0, 1)); order = M.sort_values(["farm", "day", "hour"]).index
    mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index); h_ = np.where(mm.isna(), 1, np.clip((mm - lo_h) / 2, 0, 1)); return g_ * h_
def tx3(M, key, c, T0, e0):
    cm = np.nan_to_num(np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key, M.hour.values)]), nan=0)
    return c * np.clip((cm - e0) / .4, 0, 1) * np.nan_to_num(np.clip((T0 - M.in_temp.values) / 2, 0, 1), nan=0.0)
RES = {}
for L, ckdir, folds, mk in (("N", O1.CKB, LF["N"], O1.masks), ("CK2", O1.CKB, LF["CK2"], O1.masks), ("EL1", CKR, ELF, common.split_mask)):
    M = pd.concat([pd.read_csv(os.path.join(ckdir, f)) for f in sorted(os.listdir(ckdir)) if f.startswith(L + "_")], ignore_index=True)
    Gd = pd.concat([pd.read_csv(os.path.join(O1.CKG, f)) for f in sorted(os.listdir(O1.CKG)) if f.startswith(L + "_")], ignore_index=True)
    M = M.merge(Gd, on="row_id", how="left"); M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    y = M.sub_temp.values; h = M.hour.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values; recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    v = M.V3LF.values; miss = np.isnan(v); TX_old, TX_new = tx3(M, key, .25, 10, .8), tx3(M, key, .40, 12, .7); G8 = gate(M, 8, 10)
    TRS = {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs
    def anc(LK, pmin):
        return {r: float(ST[IDX[LK[r][0]], 23]) for r in recs if r in LK and r in TRS and LK[r][1] >= pmin and LK[r][0] in TRS[r] and np.isfinite(ST[IDX[LK[r][0]], 23])
                and abs(IN_[IDX[r], 0, 0] - IN_[IDX[LK[r][0]], 23, 0]) <= 1.5}
    AF = anc(FLK, .8); UNI = {**anc(LINK, .5), **AF}
    def apply(x, A):
        out = x.copy()
        for r, a in A.items():
            ix = pos[r]; x0 = x[ix][h[ix] == 0]
            if len(x0): out[ix] += BETA[h[ix]] * np.clip(a - BIAS - x0[0], -2, 2)
        return out
    P = {"R27": [], "NEW": [], "NEW_oldTX3": []}
    for s in O1.SEEDS:
        for fam in "AB":
            q = B.w40(M, "base_R_%d" % s, "codex_R", fam); x = q + .30 * G8 * (~miss) * (np.nan_to_num(v) - q)
            P["R27"].append(apply(q, AF) + TX_old); P["NEW"].append(apply(x, UNI) + TX_new); P["NEW_oldTX3"].append(apply(x, UNI) + TX_old)
    RES[L] = (P, y, (L + "_" + M.farm + "_" + (M.day // 5).astype(str)).values, (M.in_temp <= 8).values, M.day.values >= 179); print(L, "done", flush=True)
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3asm2"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
for L in ("EXT10", "EXT12"):
    d, pre_, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); y = M.sub_temp.values; key = (M.farm + "_" + M.day.map("{:03d}".format)).values
    VF = pd.concat([pd.read_csv(os.path.join(env.LOCAL, "v3lf_ckpt", f)) for f in os.listdir(os.path.join(env.LOCAL, "v3lf_ckpt")) if f.startswith(L + "_")]).drop_duplicates("row_id")
    v = M[["row_id"]].merge(VF[["row_id", "V3LF"]], on="row_id", how="left").V3LF.values; miss = np.isnan(v); G8 = gate(M, 8, 10); TX_old, TX_new = tx3(M, key, .25, 10, .8), tx3(M, key, .40, 12, .7)
    P = {"R27": [], "NEW": [], "NEW_oldTX3": []}
    for b in bs:
        for fam in "AB":
            q = B2.w40(M, b, cx, fam); x = q + .30 * G8 * (~miss) * (np.nan_to_num(v) - q); P["R27"].append(q + TX_old); P["NEW"].append(x + TX_new); P["NEW_oldTX3"].append(x + TX_old)
    RES[L] = (P, y, (M.farm + "_" + (M.day // 5).astype(str)).values, (M.in_temp <= 8).values, M.day.values >= 179); print(L, "done", flush=True)
def cmp(L, a, b):
    P, y, bk, cold, p2 = RES[L]; ch = [rm(p - y) / rm(q - y) - 1 for p, q in zip(P[a], P[b])]
    dd = pd.Series(np.mean([(p - y) ** 2 - (q - y) ** 2 for p, q in zip(P[a], P[b])], 0)).groupby(bk).mean()
    sub = lambda m: 100 * np.mean([rm((p - y)[m]) / rm((q - y)[m]) - 1 for p, q in zip(P[a], P[b])]) if m.any() else np.nan
    return ch, dd, sub(cold), sub(p2)
ok = True; diffs = []; rng = np.random.default_rng(20270137); safe = {}
for L in ("N", "CK2", "EL1", "EXT10", "EXT12"):
    ch, dd, c_, p_ = cmp(L, "NEW", "R27"); ch2, _, c2, _ = cmp(L, "NEW", "NEW_oldTX3")
    print(f"{L:5s} NEW vs R27 {' '.join('%+.3f' % (100 * x) for x in ch)} | cold {c_:+.2f}% p2 {p_:+.2f}% || TX3-new share in assembly: NEW vs NEW_oldTX3 {np.mean(ch2) * 100:+.3f}% (cold {c2:+.2f}%)", flush=True)
    if L in ("CK2", "EL1"): ok &= all(x < 0 for x in ch); diffs.append(dd)
    if L in ("EXT10", "EXT12"):
        x_ = dd.values; pb = float((x_[rng.integers(0, len(x_), (20000, len(x_)))].mean(1) < 0).mean()); safe[L] = not (np.mean(ch) > 0 and pb < .025)
S = pd.concat(diffs).values; P_ = float((S[rng.integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
print(f"ASM2: CK2+EL1 8 cells better {ok} | pooled P {P_:.4f} | EXT safety {safe} -> {'PASS' if ok and P_ < .025 and all(safe.values()) else 'FAIL'}")
