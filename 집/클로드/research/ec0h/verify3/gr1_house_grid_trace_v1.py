# -*- coding: utf-8 -*-
"""GR1 (2026-10-12 집 클로드; 사용자: '데이터 대조해서 어제/오늘 동 격자를 확인, 83일까지 거슬러 갈 수도').  서술.
(1) 날짜 격자: DM11 달력 78~92일 각 날짜의 기록(train / 평가*).
(2) 평가 기록마다 같은 동을 거꾸로 추적: 2단계 모델(vf3b, 대상 0시 기준) 1등 전날 → 그 전날이 평가면 그 기록의 1등 전날 … train 을 만날 때까지(최대 6단계).
    각 단계 확률 Ptop0 의 곱 = 그 사슬이 맞을 대략적 확률(독립 가정, 낙관적).  train 에 닿으면 그 기록 23시 EC, 걸린 일수.
규정 메모: 이 추적은 각 평가 기록의 0시 입력 + 같은 ID 앞 기록 + train 으로 만든 2단계 모델 선택을 잇는 것(서술용).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__)); H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
CAL = pd.read_csv(os.path.join(H_, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day)); SPL = dict(zip(CAL.rec, CAL.split))
B = pd.read_csv(os.path.join(here, "vf3b_eval_v2.csv")).set_index("rec")
Yl = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Yl["rec"] = Yl.row_id.str[:7]; Yl["h"] = Yl.row_id.str[8:].astype(int); EC23 = Yl[Yl.h == 23].set_index("rec").sub_ec
S = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); S["rec"] = S.row_id.str[:7]; S["h"] = S.row_id.str[8:].astype(int)
P0 = S[S.h == 0].set_index("rec").sub_ec
print("(1) date grid, days 76-93  [* = evaluation]")
for d in range(76, 94):
    m = CAL[CAL.day == d].sort_values(["farm", "num"])
    print(f"  day {d:3d}: " + ", ".join(r + ("*" if s == "eval" else "") for r, s in zip(m.rec, m.split)))
print("\n(2) trace back along most probable previous-day choices until a training record")
rows = []
for r in sorted(B.index):
    cur, prob, path, k = r, 1.0, [], 0
    while k < 6:
        if cur not in B.index: break
        nxt, p = B.at[cur, "top0"], float(B.at[cur, "Ptop0"])
        if not isinstance(nxt, str): break
        prob *= p; k += 1; path.append(f"{nxt}({p:.2f})")
        if SPL.get(nxt) == "train": break
        cur = nxt
    end = path[-1].split("(")[0] if path else None
    lab = end is not None and SPL.get(end) == "train"
    dd = DAY.get(r, np.nan) - DAY.get(end, np.nan) if end else np.nan
    rows.append(dict(rec=r, day=DAY.get(r), steps=k, reaches_train=lab, days_back=dd, chain_prob=prob, A=EC23.get(end, np.nan) if lab else np.nan, pred0=P0.get(r, np.nan), path=" <- ".join(path)))
T = pd.DataFrame(rows); pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 120)
print(T.round(3).to_string(index=False))
print("\nsummary: reaches a training record in 1 step: %d | 2 steps: %d | 3+ steps: %d | never: %d" % (
    int(((T.steps == 1) & T.reaches_train).sum()), int(((T.steps == 2) & T.reaches_train).sum()), int(((T.steps >= 3) & T.reaches_train).sum()), int((~T.reaches_train).sum())))
print("chain probability >= .8 and reaches train in 1 step: %d | in 2 steps: %d" % (int(((T.steps == 1) & T.reaches_train & (T.chain_prob >= .8)).sum()), int(((T.steps == 2) & T.reaches_train & (T.chain_prob >= .8)).sum())))
T.to_csv(os.path.join(here, "gr1_trace_v1.csv"), index=False, encoding="utf-8-sig")
