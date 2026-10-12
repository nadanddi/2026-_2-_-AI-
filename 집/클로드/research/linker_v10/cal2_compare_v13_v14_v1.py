# -*- coding: utf-8 -*-
"""CAL2 (2026-10-13 집 클로드; v14 날짜 투표 결과 정확도·포괄률) 서술: v13 E0 vs v14 L/M 1순위(CAL1 과 같은 표).
(원) CAL1: 달력 DMT2(v12) vs DMT3e E0/E50(v13) 짝찾기 비교, 거울 CV.
깨끗한 정답(진단 연결·달력 차 1, 숨김 반영) 기준 확신 문턱 .6/.8/.9 의 정확도·포괄률·확인 불가 수(전체/2차), 날짜 상태 분포, 핵심 기록의 짝.
EC 변형(v7 ∪ 새 짝 HP2 일치) 은 HP2 가 달력을 쓰므로 여기선 링커만 비교."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAYr = dict(zip(CAL.rec, CAL.day))
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TP = {rid(int(b)): rid(int(a)) for a, b in zip(TLa.a1, TLa.b)}
idx = {rid(i): i for i in range(N)}
def mirror_lab(r):
    b = idx[r]; f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
clean = lambda r: r in TP and mirror_lab(r)[idx[TP[r]]] and DAYr.get(r, np.nan) - DAYr.get(TP[r], np.nan) == 1
OUT = os.path.join(env.LOCAL, "linker_v10"); pd.set_option("display.width", 220)
F = {"v13 E0": "v13E0_picks_mirror.csv", "v14 L": "v14_mirror.csv", "v14 M": "v14_mirror.csv"}
rows = []; P = {}
for nm, fn in F.items():
    D = pd.read_csv(os.path.join(OUT, fn))
    if nm.startswith("v14"):
        t = nm[-1]; D = D.assign(prev=D[f"{t}_prev0"], p=D[f"{t}_p0"], labelled=D[f"{t}_lab0"], date_status=D.date_src)
    P[nm] = D.set_index("rec")
    for part, sel in (("all", lambda r: True), ("pass-2", lambda r: int(r[4:7]) >= 179)):
        S = D[[sel(r) for r in D.rec]]; can = [r for r in S.rec if clean(r)]
        for th in (.6, .8, .9):
            A = S[(S.p >= th) & S.labelled]; ok = sum(clean(r) and TP[r] == p for r, p in zip(A.rec, A.prev)); jd = sum(clean(r) for r in A.rec)
            rows.append(dict(cal=nm, part=part, thr=th, anchors=len(A), precision=round(ok / max(jd, 1), 3), recall=round(ok / max(len(can), 1), 3), unverifiable=len(A) - jd))
    print(nm, "date_status:", D.date_status.value_counts().to_dict(), "| top-1 correct among clean:", round(np.mean([TP[r] == p for r, p in zip(D.rec, D.prev) if clean(r)]), 3))
print(pd.DataFrame(rows).to_string(index=False))
key = ["F47_132", "F13_217", "F13_241", "F47_146", "F13_177", "F13_149", "F13_214", "F47_079", "F47_178", "F13_116"]
print("\nkey records: truth prev / (pick, p, date_status) per calendar")
for r in key:
    print(f"  {r} truth {TP.get(r, '-')} clean {clean(r)} |", " | ".join(f"{nm}: {P[nm].prev.get(r)} {P[nm].p.get(r, np.nan):.2f} {P[nm].date_status.get(r)}" for nm in F))
