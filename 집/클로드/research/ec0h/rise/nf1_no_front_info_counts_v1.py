# -*- coding: utf-8 -*-
"""NF1 (2026-10-12 집 클로드; 사용자: '앞 정보가 없는 날들이 훈련·평가에 몇 개나 있나').  서술.
(1) 날짜 수준(동 무관): 기록 날짜 d(DM11)의 d−1·d+1 날짜에 있는 학습(정답) 기록 수(자기 제외).  0 이면 어느 동이든 정답에 닿지 않음.
(2) 동 수준(학습 기록만, 진단용 연결 truth a1): 진짜 전날 / 진짜 다음날 기록이 학습(정답) 기록인가.  평가 기록은 동을 모르므로 (1)만.
학습·평가, 1·2차별로 집계."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv("dm11_calendar_v1.csv"); DAY = dict(zip(CAL.rec, CAL.day)); SPL = dict(zip(CAL.rec, CAL.split))
lab_by_day = CAL[CAL.split == "train"].dropna(subset=["day"]).groupby("day").rec.apply(list).to_dict()
TL = pd.read_pickle(S3 + r"\truth.pkl"); prev = {rid(int(b)): (rid(int(a)), bool(c)) for a, b, c in zip(TL.a1, TL.b, TL.cert)}
nxt = {}
for b, (a, c) in prev.items(): nxt.setdefault(a, []).append((b, c))
rows = []
for r in CAL.rec:
    d = DAY.get(r); sp = SPL[r]
    if not np.isfinite(d): rows.append(dict(rec=r, split=sp, p2=int(r[4:7]) >= 179, lab_prev=np.nan, lab_next=np.nan)); continue
    lp = [x for x in lab_by_day.get(d - 1, []) if x != r]; ln = [x for x in lab_by_day.get(d + 1, []) if x != r]
    row = dict(rec=r, split=sp, p2=int(r[4:7]) >= 179, day=d, lab_prev=len(lp), lab_next=len(ln))
    if sp == "train":
        a, c = prev.get(r, (None, False)); row["true_prev_lab"] = (a is not None and SPL.get(a) == "train" and DAY.get(a) == d - 1)
        sc = [b for b, cc in nxt.get(r, []) if DAY.get(b) == d + 1]; row["true_next_lab"] = any(SPL.get(b) == "train" for b in sc)
    rows.append(row)
T = pd.DataFrame(rows)
def cat(r):
    if not np.isfinite(r.lab_prev): return "날짜 미확정"
    if r.lab_prev == 0 and r.lab_next == 0: return "앞뒤 모두 정답 0 (정보 없음)"
    if r.lab_prev == 0: return "앞 0 · 뒤만 있음"
    if r.lab_next == 0: return "앞만 있음 · 뒤 0"
    return "앞뒤 모두 있음"
T["cat"] = T.apply(cat, axis=1)
pd.set_option("display.width", 200)
print("(1) date-level: labelled records on the previous / next calendar day")
print(pd.crosstab([T.split, T.p2], T.cat, margins=True).to_string())
Tr = T[T.split == "train"]
print("\n(2) house-level for TRAINING records (truth links, diagnostic): true previous day labelled / true next day labelled")
Tr = Tr.assign(hp=Tr.true_prev_lab.fillna(False).astype(bool), hn=Tr.true_next_lab.fillna(False).astype(bool))
Tr["hcat"] = np.select([Tr.hp & Tr.hn, Tr.hp, Tr.hn], ["same-house prev & next labelled", "prev only", "next only"], "neither (no same-house label adjacent)")
print(pd.crosstab(Tr.p2, Tr.hcat, margins=True).to_string())
print("\neval records with NO labelled record on either adjacent day:", sorted(T[(T.split == "eval") & (T.cat == "앞뒤 모두 정답 0 (정보 없음)")].rec))
T.to_csv("nf1_v1.csv", index=False)
