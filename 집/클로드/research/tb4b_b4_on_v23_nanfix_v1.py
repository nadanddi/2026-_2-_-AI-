# -*- coding: utf-8 -*-
"""TB4 (2026-10-14 집 클로드; 2차 비평 1순위).  사전 고정(커밋 후 실행).  온도.  과거 정보만.
B4(사슬 새벽 보정 전날 = v15R p0 >= .5 짝, 0~7시; 6.790 ADOPT P .0008, 6.792 COMBO PASS P .0082, EXT10 +.26~+.35%)는 14회차(v23 = 13회차 + TNA8)에 안 들어감
  (6.796: RS2 기본 위 EV 에선 B4 가 +.03~+.21% — 하네스 따라 엇갈림).  TNA8 위에서는 시험된 적 없음.
1단계(이 파일) — DIAG10 계열 하네스(ext1 와 같은 T.LAY 배치·셀·앵커 식):
  기준 V23 = TC1c 사슬 셀 + v7 앵커 + TNA8 거울 앵커(tb2 의 ARMS['TNA8'][1], 13회차 앵커 식).
  B4a = v15R 사슬 셀(CT2, H 7) + 같은 앵커.
  B4w = 앵커 전 셀 단계에서 B4 변화량 x (1 − g), g = clip((10 − in_temp)/2, 0, 1)(TX2 COLD 와 같은 물리 문턱 8·10℃, 새로 안 고름) + 같은 앵커.
판정(안 2개, α .0125):  B4a(전체 범위): DIAG10·EL1·F~L·EXT10 모든 셀 개선 & 합친(DIAG10+F~K 5일 블록) P(악화) < .0125.
  B4w(따뜻한 쪽 범위): DIAG10·EL1·F~L 모든 셀 개선 & 합친 P < .0125 & EXT10 어느 셀도 +.10% 넘게 악화 안 함.
  통과 안 → 2단계(EV 확인) 대상.  1단계만으로 후보 아님.  [한계: B4 가 이미 판정받은 같은 배치 — 새 정보는 TNA8 과의 겹침뿐]
서술: 2차 행, EXT12, 추운 행(in_temp <= 8) RMSE.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tb2_temp_agree_any_p_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tb2_temp_agree_any_p_v1.py")
exec(src[:src.index("DD = pd.read_pickle")])
TNA8M = dict(ARMS["TNA8"][1]); print("TNA8 mirror anchors", len(TNA8M), flush=True)
src = open(os.path.join(HERE, "ext1_combo_b4_tn8_window_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ext1_combo_b4_tn8_window_v1.py")
exec(src[:src.index("res = {k: []")])
__file__ = os.path.join(HERE, "tb4b_b4_on_v23_nanfix_v1.py"); rng = np.random.default_rng(20261218)
add = pd.DataFrame(dict(rec=list(TNA8M), prev=list(TNA8M.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in TNA8M.values()], dT_air=0.0, steps_to_anchor=1))
LV = pd.concat([L7f, add], ignore_index=True)
res = {k: [] for k in ("B4a", "B4w")}; ok = {k: True for k in res}; diffs = {k: [] for k in res}; cold = {k: [] for k in res}
for L, (d, pre, bs, cx) in T.LAY.items():
    M1 = T.layout_frame(L, CT); M2 = T.layout_frame(L, CT2); y = M1.sub_temp.values
    assert (M1.row_id.values == M2.row_id.values).all()
    q1 = cells(M1, D1, 7, bs, cx); q2 = cells(M2, D2, 7, bs, cx); g = np.nan_to_num(np.clip((10 - M1.in_temp.values) / 2, 0, 1), nan=0.0)   # [실행 오류 수정: in_temp 결측 행 g NaN -> 0(B4 그대로); 기준 불변, B4w 일부 결과 본 뒤 수정]
    base = [anch(M1, q, LV) for q in q1]
    arm = {"B4a": [anch(M2, q, LV) for q in q2], "B4w": [anch(M1, a + (b - a) * (1 - g), LV) for a, b in zip(q1, q2)]}
    p2 = M1.day.values >= 179; cm = M1.in_temp.values <= 8
    for k, new in arm.items():
        ch = [rm(n - y) / rm(r_ - y) - 1 for n, r_ in zip(new, base)]; ch2 = [rm((n - y)[p2]) / rm((r_ - y)[p2]) - 1 for n, r_ in zip(new, base)]
        chc = [rm((n - y)[cm]) / rm((r_ - y)[cm]) - 1 for n, r_ in zip(new, base)] if cm.any() else []
        res[k].append(f"{L:6s} all {' '.join('%+.2f' % (100 * x) for x in ch)} | p2 {' '.join('%+.2f' % (100 * x) for x in ch2)} | cold(n {int(cm.sum())}) {' '.join('%+.2f' % (100 * x) for x in chc)}")
        if k == "B4a":
            if L != "EXT12": ok[k] &= all(x < 0 for x in ch)
        else:
            if L not in ("EXT10", "EXT12"): ok[k] &= all(x < 0 for x in ch)
            if L == "EXT10": ok[k] &= all(x <= .001 for x in ch)
        if L not in ("EL1", "EXT10", "EXT12"):
            key = M1.farm + "_" + (M1.day // 5).astype(str); diffs[k].append(pd.Series((new[0] - y) ** 2 - (base[0] - y) ** 2).groupby(key.values).mean())
    print("done", L, flush=True)
verdict = {}
for k in res:
    S = pd.concat(diffs[k], axis=1).mean(axis=1).values; ix = rng.integers(0, len(S), (20000, len(S))); P = float((S[ix].mean(1) >= 0).mean())
    verdict[k] = ok[k] and P < .0125
    print(f"\n== {k} vs V23 (TC1c + v7 + TNA8)"); print("\n".join(res[k]))
    print(f"{k}: cell conditions {ok[k]} | pooled P {P:.4f} -> {'PASS stage 1 (EV check next)' if verdict[k] else 'FAIL'}")
print("TB4 stage 1:", verdict)
