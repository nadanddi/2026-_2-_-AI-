# -*- coding: utf-8 -*-
"""TD4X-(a) (2026-10-14 집 클로드; 2차 비평 3순위).  서술(정답 없음).  온도.
TD4(v7 ∪ v11 >= .6) 는 DM11 달력(평가 입력 사용, 6.752) 의존 → 규정 안 판 = TD8(v12 학습 전용 달력 DMT2, 차단 짝은 HP2 일치 시).
v23(14회차) 온도 앵커 = v7 + TNA8.  TD8 평가 새 앵커 중 v23 에 없는 기록 수·짝을 셈.  결정 규칙(미리 정함): 새 기록 < 2 이면 TD4X 중단, >= 2 이면 (b) 판정.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tb2_temp_agree_any_p_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tb2_temp_agree_any_p_v1.py")
exec(src[:src.index("DD = pd.read_pickle")])
__file__ = os.path.join(HERE, "td4x_eval_exposure_v1.py")
E15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_eval.csv")); E12 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_eval.csv"))
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
v7e = set(E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)].row)
allv = lambda r: TRAIN.copy()
tna8 = {**v15_new(E15, allv, v7e, 0.0, True), **v15_new(E15, allv, v7e, .8, False)}
td8, _, _ = new_anchors(E12, allv, v7e)
v23 = set(v7e) | set(tna8)
new = {r: p for r, p in td8.items() if r not in v23}
print(f"v7 eval anchors {len(v7e)} | TNA8 eval {len(tna8)} | v23 total {len(v23)} | TD8 eval new anchors {len(td8)}")
print("TD8 anchors beyond v23:", dict(sorted(new.items())))
print("TD8 vs TNA8 same pick on overlap:", {r: (td8[r], tna8[r]) for r in td8 if r in tna8})
print("TD4X decision:", "go to (b)" if len(new) >= 2 else "stop (fewer than 2 new eval records)")
