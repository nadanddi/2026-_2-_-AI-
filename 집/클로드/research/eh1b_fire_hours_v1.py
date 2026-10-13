# -*- coding: utf-8 -*-
"""EH1B (2026-10-14 집 클로드; 서술).  EH1 발동이 몇 시에 걸렸나 — 가짜(F47_212·214·216)와 진짜(F13_243), 평가 발동 기록."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "eh1_continuing_high_lift_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "eh1_continuing_high_lift_v1.py")
exec(src[:src.index("res, recd, blocks, fl = [], [], {}, []")])
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.seed == 47)].copy(); O["rec"] = O.row_id.str[:7]
for v in ("P2LOO", "EL1", "DIAG10"):
    G = O[O.validator == v]; hidden = G.groupby("fold").rec.apply(set).to_dict(); fo = G.groupby("rec").fold.first().to_dict()
    for r in ("F47_212", "F47_214", "F47_216", "F13_243"):
        if r not in fo: continue
        hs = hidden[fo[r]]; vis = TRAIN & np.isfinite(DM) & ~np.array([rid(i) in hs for i in range(N)]); f, pick = fires(idx[r], vis)
        print(f"{v:6s} {r} ({'high' if DM[idx[r]] >= 1 else 'normal'}): fired hours {[h for h in range(24) if f[h]]}")
visE = TRAIN & np.isfinite(DM)
for r in ("F13_221", "F47_233", "F47_235", "F47_236"):
    f, pick = fires(idx[r], visE); print(f"EVAL   {r}: fired hours {[h for h in range(24) if f[h]]}")
