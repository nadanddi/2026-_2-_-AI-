# ECL2 (2026-10-13 집 클로드; 사용자: '온도 구성을 EC 특징에 맞게 조금만 손대면? 온도 구성 정확도·포괄률은?') 서술, 거울 CV.
# TEMP(TD8) = v7 앵커 ∪ v12(p≥.6, 정답, 날짜 불일치 아님): |dT|≤1.5 는 그대로, |dT|>1.5 는 HP2 일치 시만.
# EC 변형 = v7 앵커 ∪ v12(같은 조건) 중 '모든' 새 짝에 HP2 일치 요구(dT 무관).
# 깨끗한 정답(달력 차 1) 기준 정확도·포괄률 + '맞았는지 확인 불가' 앵커 수.
import os, sys
import env
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "td8_rule_clean_strict_v1.py"), encoding="utf-8").read()
exec(src[:src.index("rm = lambda e:")])
def mirror_lab(r):
    b = idx[r]; f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
v7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_mirror.csv"))
TPd = {rid(int(b)): rid(int(a)) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); DAYr = dict(zip(CAL.rec, CAL.day))
def clean(r):
    if r not in TPd: return False
    return mirror_lab(r)[idx[TPd[r]]] and DAYr.get(r, np.nan) - DAYr.get(TPd[r], np.nan) == 1
temp, ec = dict(v7), dict(v7)
for r, pv, p, dT, ds, lb in zip(PM.rec, PM.prev, PM.p, PM.dT, PM.date_status, PM.labelled):
    if r in v7 or not lb or p < .6 or ds == "mismatch": continue
    agree = hp2(idx[r], mirror_lab(r)) == pv
    if abs(dT) <= 1.5 or agree: temp[r] = pv
    if agree: ec[r] = pv
recs = [r for r in PM.rec]
for part, sel in (("all", lambda r: True), ("pass-2", lambda r: int(r[4:7]) >= 179)):
    can = [r for r in recs if clean(r) and sel(r)]
    for nm, S in (("v7 only (13th)", v7), ("TEMP config (TD8)", temp), ("EC variant (HP2 on all new)", ec)):
        A = {r: p for r, p in S.items() if sel(r)}
        ok = sum(1 for r, p in A.items() if clean(r) and TPd[r] == p); judged = sum(1 for r in A if clean(r))
        print(f"{part:6s} {nm:28s} anchors {len(A):3d} | precision {ok / max(1, judged):.3f} | recall {ok / max(1, len(can)):.2f} | correct {ok} | unverifiable {len(A) - judged}")
