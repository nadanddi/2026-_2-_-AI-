# -*- coding: utf-8 -*-
"""CB4-T (2026-10-15 집 클로드; 사용자 '둘 다 진행' — TX3 손잡이 다듬기).  사전 고정(커밋 후 실행).  온도.  학습 없음.
TX3(6.931): + c · clip((cmEC_h − e0)/.4, 0, 1) · clip((T0 − in_temp_h)/2, 0, 1),  지금 c .25 · e0 .8 · T0 10 (TX2 에서 미리 정한 값).
격자 72 = c {.15, .20, .25, .30, .35, .40} x T0 {9, 10, 11, 12} x e0 {.7, .8, .9}.   cmEC = 27번 EC 기본 NH(DIAG10 OOF) 0..h 누적평균.
TX3 는 더하기 항이라 기본 예측과 독립 → 기본 = 각 배치의 W40G-S 셀(앵커·GRU 없음)에 TX3 만 바꿔 비교.
고르기: CK2(추운 날 배치, OS1 v2) 4셀 평균 RMSE 최소.  기준 REF = (.25, 10, .8).
판정(단일 안 α .025): 선택 vs REF — EXT10·EXT12(TV3 틀 셀) 8셀 모두 개선 & 두 배치 합친 블록 P(악화) < .025 & N·EL1 안전(셀 평균 악화 & P(나음) < .025 이면 FAIL).
선택이 REF 와 같으면 FAIL(바꿀 것 없음).  서술: 손잡이별 주효과(CK2·EXT10·EXT12·N·EL1).
한계: EXT10 은 TX2/TX3 판정에 쓰인 판(상수는 그때 미리 정한 값이었음).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cb4_temp_tx3_shaping_v1.py
"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "kf3_field_state_kalman_temp_v2.py"), encoding="utf-8").read()
exec(src[:src.index("rng = np.random.default_rng(20270118)")])
__file__ = os.path.join(HERE, "cb4_temp_tx3_shaping_v1.py")
CKR = os.path.join(env.LOCAL, "yl1_ref_ckpt")
CS = (.15, .20, .25, .30, .35, .40); TS = (9, 10, 11, 12); ES = (.7, .8, .9); COMBOS = list(itertools.product(CS, TS, ES)); REF = (.25, 10, .8)
FR = {}
for L, ckdir in (("N", O1.CKB), ("CK2", O1.CKB), ("EL1", CKR)):
    M = pd.concat([pd.read_csv(os.path.join(ckdir, f)) for f in sorted(os.listdir(ckdir)) if f.startswith(L + "_")], ignore_index=True)
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    FR[L] = (M, [B.w40(M, "base_R_%d" % s, "codex_R", fam) for s in O1.SEEDS for fam in "AB"])
src2 = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); g2 = {"__file__": os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), "__name__": "tv3cb4"}
exec(src2[:src2.index("JUDGE = [")], g2); T, B2, CT = g2["T"], g2["B"], g2["CT"]
for L in ("EXT10", "EXT12"):
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); FR[L] = (M, [B2.w40(M, b, cx, fam) for b in bs for fam in "AB"])
RES = {}
for L, (M, cells) in FR.items():
    key = (M.farm + "_" + M.day.map("{:03d}".format)).values; cm = np.array([CM.get((k, int(hh)), np.nan) for k, hh in zip(key, M.hour.values)]); cmv = np.nan_to_num(cm, nan=0)
    t = M.in_temp.values
    for c, T0, e0 in COMBOS:
        add = c * np.clip((cmv - e0) / .4, 0, 1) * np.nan_to_num(np.clip((T0 - t) / 2, 0, 1), nan=0.0); RES[(L, (c, T0, e0))] = [q + add for q in cells]
    RES[(L, "y")] = M.sub_temp.values; RES[(L, "bk")] = (L + "_" + M.farm + "_" + (M.day // 5).astype(str)).values
rel = lambda L, c: 100 * np.mean([rm(p - RES[(L, "y")]) / rm(b - RES[(L, "y")]) - 1 for p, b in zip(RES[(L, c)], RES[(L, REF)])])
score = {c: np.mean([rm(p - RES[("CK2", "y")]) for p in RES[("CK2", c)]]) for c in COMBOS}; best = min(COMBOS, key=lambda c: (round(score[c], 7), COMBOS.index(c)))
T_ = pd.DataFrame([dict(c=c[0], T0=c[1], e0=c[2], **{L: rel(L, c) for L in ("CK2", "EXT10", "EXT12", "N", "EL1")}) for c in COMBOS])
pd.set_option("display.width", 200)
for col in ("c", "T0", "e0"): print(col, T_.groupby(col)[["CK2", "EXT10", "EXT12", "N", "EL1"]].mean().round(3).to_dict("index"))
print(T_.sort_values("CK2").head(10).round(3).to_string(index=False)); print("selected on CK2:", best)
ok = best != REF; diffs = []
for L in ("EXT10", "EXT12"):
    y = RES[(L, "y")]; ch = [rm(p - y) / rm(b - y) - 1 for p, b in zip(RES[(L, best)], RES[(L, REF)])]; ok &= all(x < 0 for x in ch)
    diffs.append(pd.Series(np.mean([(p - y) ** 2 - (b - y) ** 2 for p, b in zip(RES[(L, best)], RES[(L, REF)])], 0)).groupby(RES[(L, "bk")]).mean())
    print(f"  {L} {' '.join('%+.3f' % (100 * x) for x in ch)}")
S = pd.concat(diffs).values; P_ = float((S[np.random.default_rng(20270134).integers(0, len(S), (20000, len(S)))].mean(1) >= 0).mean()); safe = {}
for L in ("N", "EL1"):
    y = RES[(L, "y")]; ch = [rm(p - y) / rm(b - y) - 1 for p, b in zip(RES[(L, best)], RES[(L, REF)])]
    dd = pd.Series(np.mean([(p - y) ** 2 - (b - y) ** 2 for p, b in zip(RES[(L, best)], RES[(L, REF)])], 0)).groupby(RES[(L, "bk")]).mean().values
    pb = float((dd[np.random.default_rng(20270135).integers(0, len(dd), (20000, len(dd)))].mean(1) < 0).mean()); safe[L] = not (np.mean(ch) > 0 and pb < .025)
    print(f"  {L} safety {np.mean(ch) * 100:+.3f}% P(better) {pb:.3f}")
print(f"CB4-T {best}: EXT10+EXT12 8 cells better {ok} | pooled P {P_:.4f} | safety {safe} -> {'PASS' if ok and P_ < .025 and all(safe.values()) else 'FAIL'}")
