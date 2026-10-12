# -*- coding: utf-8 -*-
"""DMT3 (2026-10-13 집 클로드; 사용자: '달력 보강 해봐').  학습 기록만으로 만든 달력 DMT(6.753)·DMT2(끊김) 보강.
문제(진단): DMT 는 '1차 날짜(R)와 직접 확실 연결된 끼어든 날짜'만 넣음 → 끼어든 날짜가 여러 날 줄지은 곳(순수 2차 창)은 첫날만 들어가고
  나머지 학습 기록 15개(F13_218, F47_229~231 등)는 날짜 없음, 그 뒤 날짜도 실제보다 당겨짐(예: F47_132 의 진짜 전날 F47_231 날짜 없음).
보강(학습 입력·학습 확실 연결만):
  (1) 사슬 끼워 넣기 반복: 날짜 없는 묶음 u 의 확실 연결 전날 묶음이 달력에 있으면(끼어든 날짜여도) 그 바로 뒤에, 전날 없이 다음날 묶음만 있으면 그 바로 앞에.
  (2) 외기 연속 붙이기: 남은 묶음은 '끊김 지점'(다음 날과의 외기 자정 비용 > T, 또는 끝)에만, 비용(앞 묶음 → u) ≤ T 인 가장 싼 곳 뒤에. 반복.
  T = 학습 확실 연결의 외기 자정 비용 99% 분위(DMT2 와 같음).
  (3) 구간 seg: 이웃 날 비용 > T 이면 끊김.
출력 local/dmt3_days.csv(rec, day, seg). 검산: 확실 연결 날짜 차 = +1 개수, 날짜 없는 학습 기록 수, 끊김 수.
대조(서술만, 쓰지 않음): DM11(평가 입력 쓴 달력)과 같은 구간 안 기록 쌍의 날짜 차 일치율 — DMT2 vs DMT3.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
from collections import defaultdict
here = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
D = pd.read_csv(os.path.join(here, "dmt_calendar_v1.csv")); D = D[D.split == "train"]
W = np.array([1, 1, .3, .5])
oj = lambda a, b: float(np.sum(W * np.abs(X[a, 23] + .5 * (X[a, 23] - X[a, 22]) - X[b, 0]) / sdO))
TL = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TL[TL.cert]
CL = [(int(a), int(b)) for a, b in zip(TLc.a1, TLc.b) if TRAIN[int(a)] and TRAIN[int(b)]]
Tq = float(np.quantile([oj(a, b) for a, b in CL], .99))
# groups: placed DMT days in order, plus unplaced records grouped by >=90% same outdoor
order = [list(map(idx.get, g.rec)) for _, g in D.dropna(subset=["day"]).sort_values("day").groupby("day", sort=True)]
un = [idx[r] for r in D[D.day.isna()].rec]; Xf = X.reshape(N, -1); ug = []
for i in un:
    for g in ug:
        if (np.abs(Xf[g[0]] - Xf[i]) < 1e-6).mean() >= .9: g.append(i); break
    else: ug.append([i])
print(f"T {Tq:.2f} | DMT days {len(order)} | unplaced train records {len(un)} in {len(ug)} groups:", [[rid(i) for i in g] for g in ug])
gcost = lambda g1, g2: min(oj(a, b) for a in g1 for b in g2)
def where(i):
    for k, g in enumerate(order):
        if i in g: return k
    return None
log = []
changed = True
while changed:  # (1) chain insertion
    changed = False
    for g in list(ug):
        pk = [where(a) for a, b in CL if b in g and a not in g]; pk = [k for k in pk if k is not None]
        nk = [where(b) for a, b in CL if a in g and b not in g]; nk = [k for k in nk if k is not None]
        if pk:
            k = max(set(pk), key=pk.count); order.insert(k + 1, g); ug.remove(g); changed = True; log.append(("chain-after", [rid(i) for i in g], [rid(i) for i in order[k]][:2]))
        elif nk:
            k = max(set(nk), key=nk.count); order.insert(k, g); ug.remove(g); changed = True; log.append(("chain-before", [rid(i) for i in g], [rid(i) for i in order[k + 1]][:2]))
while ug:  # (2) continuity at break points
    best = None
    for g in ug:
        for k in range(len(order)):
            if k + 1 < len(order) and gcost(order[k], order[k + 1]) <= Tq: continue
            c = gcost(order[k], g)
            if c <= Tq and (best is None or c < best[0]): best = (c, g, k)
    if best is None: break
    c, g, k = best; order.insert(k + 1, g); ug.remove(g); log.append((f"continuity({c:.2f})", [rid(i) for i in g], [rid(i) for i in order[k]][:2]))
for x in log: print("  ", x)
day = {i: k + 1 for k, g in enumerate(order) for i in g}; seg = {}; s = 0; nb = 0
for k, g in enumerate(order):
    if k > 0 and gcost(order[k - 1], g) > Tq: s += 1; nb += 1
    for i in g: seg[i] = s
chk = pd.Series([day[b] - day[a] for a, b in CL if a in day and b in day]).value_counts().sort_index().to_dict()
print(f"DMT3 days {len(order)} | still unplaced {sum(len(g) for g in ug)} | breaks {nb} | certain-link day diff counts {chk}")
out = pd.DataFrame([dict(rec=rid(i), day=day.get(i, np.nan), seg=seg.get(i, np.nan)) for i in np.where(TRAIN)[0]])
out.to_csv(os.path.join(env.LOCAL, "dmt3_days.csv"), index=False)
# descriptive cross-check vs DM11 (not used anywhere)
D11 = pd.read_csv(os.path.join(here, "dm11_calendar_v1.csv")).set_index("rec").day
def agree(fn):
    C = pd.read_csv(os.path.join(env.LOCAL, fn)).dropna(); C["d11"] = C.rec.map(D11); C = C.dropna()
    ok = tot = 0
    for _, g in C.groupby("seg"):
        a, b = g.day.values, g.d11.values; m = np.subtract.outer(a, a) == np.subtract.outer(b, b); ok += m.sum(); tot += m.size
    nd = int(TRAIN.sum()) - len(C)
    return ok / tot, nd
for fn in ("dmt2_days.csv", "dmt3_days.csv"):
    a, nd = agree(fn); print(f"{fn}: within-segment pairwise day-difference agreement with DM11 {a:.4f} | train records without day {nd}")
C3 = out.set_index("rec")
for r in ["F47_132", "F47_231", "F13_217", "F13_127", "F13_218", "F47_229"]: print("  ", r, "DMT3", C3.day.get(r), C3.seg.get(r), "| DM11", D11.get(r))
