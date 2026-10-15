# -*- coding: utf-8 -*-
"""CB3-T (2026-10-15 집 클로드; 사용자 '괜찮아 보이는 부품을 이리 깎고 저리 깎기' — CB1-T 후속).  사전 고정(커밋 후 실행).  온도.  학습 없음.
CB1-T(6.953) PASS 조합 = S20·UNI·ALL·1·TX3; 주효과에서 GRU 원 게이트(8·10) 가 판정 배치에서 더 좋았으나 예전 추운 외삽(EXT10) 손해 → GRU 손잡이를 다듬음.
격자 15 = GRU 비율 {.10, .15, .20, .25, .30} x 게이트 {8/10, 9/11, 10/12}(실내 g 시작·그날 최저 h 시작, 램프 2℃).  나머지 = UNI 앵커·ALL·강도 1·TX3 on(CB1-T 선택).
고르기: N 4셀 평균 RMSE 최소.  기준 = CB1-T 선택(S20 = .20·10/12).
판정(단일 안 α .025): 선택 vs 기준 — CK2·EL1 8셀 모두 개선 & 합친 블록 P(악화) < .025 & 추운 외삽 안전: EXT10·EXT12(v3lf_ckpt V3LF + TV3 틀 W40G-S 셀, 앵커 없음)에서
  선택 게이트·비율의 GRU 섞기 vs 기준 섞기 — 셀 평균 악화 & 블록 P(나음) < .025 이면 FAIL.
선택이 기준과 같으면 FAIL(바꿀 것 없음).  서술: 15조합 표(N·CK2·EL1·EXT10·EXT12).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cb3_temp_gru_shaping_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])
__file__ = os.path.join(HERE, "cb3_temp_gru_shaping_v1.py")
import common
FLT = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")); FLK = {r: (p, q) for r, p, q in zip(FLT.rec, FLT.prev, FLT.p)}
ELF = dict(W.worlds()[7])["EL1"]; CKR = os.path.join(env.LOCAL, "yl1_ref_ckpt")
WTS = (.10, .15, .20, .25, .30); GATES = ((8, 10), (9, 11), (10, 12)); COMBOS = list(itertools.product(WTS, GATES)); REF = (.20, (10, 12))
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
    TRS = {}
    for k, fd in enumerate(folds):
        tm, vm = mk(lab, fd); trs = set(lab.farm[tm] + "_" + lab.day[tm].map("{:03d}".format))
        for r in set(lab.farm[vm] + "_" + lab.day[vm].map("{:03d}".format)): TRS[r] = trs
    def anc(LK, pmin):
        return {r: float(ST[IDX[LK[r][0]], 23]) for r in recs if r in LK and r in TRS and LK[r][1] >= pmin and LK[r][0] in TRS[r] and np.isfinite(ST[IDX[LK[r][0]], 23])
                and abs(IN_[IDX[r], 0, 0] - IN_[IDX[LK[r][0]], 23, 0]) <= 1.5}
    UNI = {**anc(LINK, .5), **anc(FLK, .8)}; cells = [B.w40(M, "base_R_%d" % s, "codex_R", fam) for s in O1.SEEDS for fam in "AB"]
    GT = {g: gate(M, *g) for g in GATES}
    for c in COMBOS:
        wt, gg = c; preds = []
        for q in cells:
            x = q + wt * GT[gg] * (~miss) * (np.nan_to_num(v) - q); out = x + tx3
            for r, A in UNI.items():
                ix = pos[r]; x0 = x[ix][h[ix] == 0]
                if len(x0): out[ix] += BETA[h[ix]] * np.clip(A - BIAS - x0[0], -2, 2)
            preds.append(out)
        RES[(L, c)] = preds
    RES[(L, "y")] = y; RES[(L, "bk")] = (L + "_" + M.farm + "_" + (M.day // 5).astype(str)).values; print(L, "done", flush=True)
# EXT safety frames (TV3 harness cells, V3LF from v3lf_ckpt; no anchors)
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3cb3"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
for L in ("EXT10", "EXT12"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); y = M.sub_temp.values
    VF = pd.concat([pd.read_csv(os.path.join(env.LOCAL, "v3lf_ckpt", f)) for f in os.listdir(os.path.join(env.LOCAL, "v3lf_ckpt")) if f.startswith(L + "_")]).drop_duplicates("row_id")
    v = M[["row_id"]].merge(VF[["row_id", "V3LF"]], on="row_id", how="left").V3LF.values; miss = np.isnan(v); GT = {g: gate(M, *g) for g in GATES}
    for c in COMBOS:
        wt, gg = c; RES[(L, c)] = [B2.w40(M, b, cx, fam) + wt * GT[gg] * (~miss) * (np.nan_to_num(v) - B2.w40(M, b, cx, fam)) for b in bs for fam in "AB"]
    RES[(L, "y")] = y; RES[(L, "bk")] = (M.farm + "_" + (M.day // 5).astype(str)).values; print(L, "done", flush=True)
rel = lambda L, c: 100 * np.mean([rm(p - RES[(L, "y")]) / rm(b - RES[(L, "y")]) - 1 for p, b in zip(RES[(L, c)], RES[(L, REF)])])
score = {c: np.mean([rm(p - RES[("N", "y")]) for p in RES[("N", c)]]) for c in COMBOS}; best = min(COMBOS, key=lambda c: (round(score[c], 6), COMBOS.index(c)))
pd.set_option("display.width", 200)
print(pd.DataFrame([dict(w=c[0], gate=f"{c[1][0]}/{c[1][1]}", N=rel("N", c), CK2=rel("CK2", c), EL1=rel("EL1", c), EXT10=rel("EXT10", c), EXT12=rel("EXT12", c)) for c in COMBOS]).round(3).to_string(index=False))
print("selected on N:", best)
ok = best != REF; diffs = []
for L in ("CK2", "EL1"):
    y = RES[(L, "y")]; ch = [rm(p - y) / rm(b - y) - 1 for p, b in zip(RES[(L, best)], RES[(L, REF)])]; ok &= all(x < 0 for x in ch)
    diffs.append(pd.Series(np.mean([(p - y) ** 2 - (b - y) ** 2 for p, b in zip(RES[(L, best)], RES[(L, REF)])], 0)).groupby(RES[(L, "bk")]).mean())
    print(f"  {L} {' '.join('%+.3f' % (100 * x) for x in ch)}")
S = pd.concat(diffs).values; P_ = float((S[np.random.default_rng(20270132).integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean())
safe = {}
for L in ("EXT10", "EXT12"):
    y = RES[(L, "y")]; ch = [rm(p - y) / rm(b - y) - 1 for p, b in zip(RES[(L, best)], RES[(L, REF)])]
    dd = pd.Series(np.mean([(p - y) ** 2 - (b - y) ** 2 for p, b in zip(RES[(L, best)], RES[(L, REF)])], 0)).groupby(RES[(L, "bk")]).mean().values
    pb = float((dd[np.random.default_rng(20270133).integers(0, len(dd), (20000, len(dd)))].mean(1) < 0).mean()); safe[L] = not (np.mean(ch) > 0 and pb < .025)
    print(f"  {L} safety: {' '.join('%+.3f' % (100 * x) for x in ch)} P(better) {pb:.3f} -> {'ok' if safe[L] else 'UNSAFE'}")
print(f"CB3-T {best}: CK2+EL1 8 cells better {ok} | pooled P {P_:.4f} | EXT safety {safe} -> {'PASS' if ok and P_ < .025 and all(safe.values()) else 'FAIL'}")
