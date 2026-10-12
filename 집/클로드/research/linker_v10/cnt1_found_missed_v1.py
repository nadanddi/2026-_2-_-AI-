# -*- coding: utf-8 -*-
"""CNT1 (2026-10-13 집 클로드; 사용자: '잘 찾은 / 있는데 못 찾은 / 없어서 못 찾은 개수 정리') 서술, 거울 CV 학습 기록 400(전체·2차).
전날 '있음' = 깨끗한 정답(진단 연결 a1 이 거울에서 보이는 정답 기록 & DM11 달력 차 1).  '없음/확인 불가' = 그 외를 세분:
  hidden(진짜 전날 기록은 있으나 거울에서 숨김), uncertain(진단 연결 있으나 달력 차 ≠ 1 — 연결 불확실), none(진단 연결 없음).
앵커 = 1순위 p ≥ .8 & 정답 기록(v7 은 추가로 |dT| ≤ 1.5).  링커: v7(13회차), v14 L(날짜 투표), v15pi15(없음 선택지)."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAYr = dict(zip(CAL.rec, CAL.day))
TLa = pd.read_pickle(os.path.join(S3, "truth.pkl")); TP = {rid(int(b)): rid(int(a)) for a, b in zip(TLa.a1, TLa.b)}
idx = {rid(i): i for i in range(N)}
def mirror_lab(r):
    b = idx[r]; f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
def status(r):
    if r not in TP: return "none"
    if DAYr.get(r, np.nan) - DAYr.get(TP[r], np.nan) != 1: return "uncertain"
    return "has" if mirror_lab(r)[idx[TP[r]]] else "hidden"
OUT = os.path.join(env.LOCAL, "linker_v10")
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
A = {"v7 (13th)": {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}}
D = pd.read_csv(os.path.join(OUT, "v14_mirror.csv")); A["v14 L (date vote)"] = {r: p for r, p, q, l in zip(D.rec, D.L_prev0, D.L_p0, D.L_lab0) if q >= .8 and l}
D = pd.read_csv(os.path.join(OUT, "v15pi15_mirror.csv")); A["v15 PI.15 (null option)"] = {r: p for r, p, q, l in zip(D.rec, D.prev0, D.p0, D.lab0) if q >= .8 and l}
recs = [rid(i) for i in np.where(TRAIN & np.isin(FARM, ["F13", "F47"]))[0]]
ST = {r: status(r) for r in recs}
for part, sel in (("ALL 400", lambda r: True), ("PASS-2", lambda r: int(r[4:7]) >= 179)):
    R = [r for r in recs if sel(r)]
    print(f"\n=== {part}: records {len(R)} | prev status:", pd.Series([ST[r] for r in R]).value_counts().to_dict())
    rows = []
    for nm, AA in A.items():
        has = [r for r in R if ST[r] == "has"]; no = [r for r in R if ST[r] != "has"]
        found = sum(r in AA and AA[r] == TP[r] for r in has); wrong_has = sum(r in AA and AA[r] != TP[r] for r in has)
        rows.append({"linker": nm, "HAS prev: found": found, "HAS: missed (no anchor)": len(has) - found - wrong_has, "HAS: wrong anchor": wrong_has,
                     "NO prev: abstained": sum(r not in AA for r in no), "NO prev: anchored anyway": sum(r in AA for r in no),
                     "  of which hidden/uncertain/none": "/".join(str(sum(r in AA and ST[r] == k for r in no)) for k in ("hidden", "uncertain", "none"))})
    pd.set_option("display.width", 250); print(pd.DataFrame(rows).to_string(index=False))
# 2차 '있는데 못 찾은' 기록 상세(v15 PI.15)
D = pd.read_csv(os.path.join(OUT, "v15pi15_mirror.csv")).set_index("rec"); AA = A["v15 PI.15 (null option)"]; V7 = A["v7 (13th)"]
print("\nPASS-2 HAS-prev records missed by v15 PI.15:")
for r in [r for r in recs if int(r[4:7]) >= 179 and ST[r] == "has" and not (r in AA and AA[r] == TP[r])]:
    row = D.loc[r]; rank = next((k for k in range(6) if row.get(f"prev{k}") == TP[r]), None)
    print(f"  {r} truth {TP[r]} | pick {row.prev0} p0 {row.p0:.2f} p_null {row.p_null:.2f} date {row.date_src} | truth rank {rank} p {row.get(f'p{rank}', np.nan) if rank is not None else np.nan:.2f} | v7 anchored {r in V7}")
