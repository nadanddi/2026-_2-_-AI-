# -*- coding: utf-8 -*-
"""GR2 (2026-10-12 집 클로드; '지워진 날 건너뛰기 소거').  서술 먼저.
지워진 기록(번호 결번, 6.536) 때문에 동이 하나 빠진 날짜 d(묶음 3개)가 있으면: d−1 날짜 train 기록 중 d 의 train 기록으로 이어지지 않는(정답 자정 연속 비용 > 2 또는 상호 최선 아님) 기록이
'지워진 동'의 d−1 모습 → 그 동은 d+1 에 다시 나타남.  d+1 의 평가 기록 중 하나가 그 동이면 '이틀 전 정답'을 앵커로 쓸 수 있음(k=2 정보).
(1) 81일 사례 직접 확인, (2) 전 달력에서 '묶음 3개 날 + 앞날 4개 모두 train + 다음날에 평가 기록' 구조를 모두 찾아 같은 판정.
학습 쪽 검증: 같은 구조가 학습 기록끼리 있는 곳(다음날도 train)에서 이 추론이 맞는지 진실 연결로 확인.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
here = os.path.dirname(os.path.abspath(__file__))
CAL = pd.read_csv(os.path.join(here, "..", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
pe = lambda A, i: A[i, 23] + 0.5 * (A[i, 23] - A[i, 22]); pb = lambda A, j: A[j, 0] - 0.5 * (A[j, 1] - A[j, 0])
cost = lambda j, k: abs(pe(EC, j) - pb(EC, k)) / .01 + abs(pe(ST, j) - pb(ST, k)) / .3
TL = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TL[TL.cert]; succ = {}
for a, b in zip(TLc.a1, TLc.b): succ.setdefault(rid(int(a)), []).append(rid(int(b)))
S = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); S["rec"] = S.row_id.str[:7]; S["h"] = S.row_id.str[8:].astype(int)
P0 = S[S.h == 0].set_index("rec").sub_ec; Pm = S.groupby("rec").sub_ec.mean()
by = CAL.dropna(subset=["day"]).groupby("day").rec.apply(list).to_dict()
def unmatched(dprev, d):
    A = [idx[r] for r in by.get(dprev, []) if TRAIN[idx[r]]]; Bt = [idx[r] for r in by.get(d, []) if TRAIN[idx[r]]]
    taken = set()
    for k in Bt:
        cj = [cost(j, k) for j in A]
        if not cj: continue
        j = A[int(np.argmin(cj))]
        if min(cj) <= 2 and min(Bt, key=lambda q: cost(j, q)) == k: taken.add(j)
    return [rid(j) for j in A if j not in taken], [rid(j) for j in taken]
for d in sorted(by):
    if len(by[d]) != 3 or len(by.get(d - 1, [])) != 4 or not all(TRAIN[idx[r]] for r in by[d - 1]): continue
    left, taken = unmatched(d - 1, d)
    nxt = by.get(d + 1, []); ev_next = [r for r in nxt if not TRAIN[idx[r]]]
    print(f"day {d - 1}->{d}->{d + 1}: day{d - 1} {by[d - 1]} | day{d} {by[d]} | continued {taken} | NOT continued (skipped house?) {left}")
    for r in left:
        print(f"     {r}: EC 23h {EC[idx[r], 23]:.3f}, ST 23h {ST[idx[r], 23]:.1f} | true certain successors: {succ.get(r, [])}")
    if ev_next:
        print(f"     day {d + 1} eval records: " + ", ".join(f"{r} (pred0 {P0.get(r, np.nan):.2f}, day mean {Pm.get(r, np.nan):.2f})" for r in ev_next))
    tr_next = [r for r in nxt if TRAIN[idx[r]]]
    if tr_next and left:
        for r in left:
            cs = sorted(((cost(idx[r], idx[q]), q) for q in tr_next))
            print(f"     train on day {d + 1}: label-continuity from {r}: " + ", ".join(f"{q} {c:.1f} (EC0 {EC[idx[q], 0]:.2f})" for c, q in cs))
