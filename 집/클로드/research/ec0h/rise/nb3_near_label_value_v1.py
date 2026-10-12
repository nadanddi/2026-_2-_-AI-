# -*- coding: utf-8 -*-
"""NB3 (2026-10-12 집 클로드; 사용자: '실마리 있는 3개를 근처 정답에 맞춰서 검증').
(1) 학습 기록으로 '이 거리의 같은 동 정답이 얼마나 믿을 만한가' (진단용 연결 truth a1 로 같은 동을 안다고 가정 — 식별 오류 제외한 상한):
   F (다음날, k=1 뒤): 대상 23시 정답 vs 모델 23시 예측(DIAG10 sg) vs 같은 동 다음날 0시 정답.
   B2 (이틀 전): 대상 0시 정답 vs 모델 0시 예측 vs 같은 동 이틀 전 23시 정답(사슬 a1 두 번).
   F2 (이틀 뒤): 대상 23시 정답 vs 같은 동 이틀 뒤 0시 정답.
   보고: |오차| 중앙·RMSE, 정답 쪽이 더 가까운 비율, 그리고 최적 반영 비율 β(모델→정답 쪽 이동, 최소제곱) — 전체 / 고EC(대상 하루평균 ≥ 1).
(2) 평가 3기록(F13_221←F13_217 이틀 전, F47_227→F47_229 다음날, F47_225→F47_229 이틀 뒤)에 β 를 적용하면 12회차 예측이 얼마나 바뀌나.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv("dm11_calendar_v1.csv"); DAY = dict(zip(CAL.rec, CAL.day))
TL = pd.read_pickle(S3 + r"\truth.pkl"); pre = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}
suc = {}
for b, a in pre.items(): suc.setdefault(a, []).append(b)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
Q = O.groupby(["rec", "hour"]).sg.mean().unstack()
lab = lambda r: r in idx and TRAIN[idx[r]]
rows = []
for r in Q.index:
    i = idx[r]; d = DAY.get(r); hi = float(np.nanmean(EC[i])) >= 1
    s = [x for x in suc.get(r, []) if lab(x) and DAY.get(x) == d + 1]
    if s: rows.append(dict(kind="F next day", rec=r, hi=hi, y=EC[i, 23], q=Q.at[r, 23], A=EC[idx[s[0]], 0]))
    p1 = pre.get(r); p2 = pre.get(p1) if p1 else None
    if p2 and lab(p2) and DAY.get(p2) == d - 2: rows.append(dict(kind="B2 two days before", rec=r, hi=hi, y=EC[i, 0], q=Q.at[r, 0], A=EC[idx[p2], 23]))
    s2 = [y for x in suc.get(r, []) if DAY.get(x) == d + 1 for y in suc.get(x, []) if lab(y) and DAY.get(y) == d + 2]
    if s2: rows.append(dict(kind="F2 two days after", rec=r, hi=hi, y=EC[i, 23], q=Q.at[r, 23], A=EC[idx[s2[0]], 0]))
D = pd.DataFrame(rows); D["eM"] = D.y - D.q; D["eA"] = D.y - D.A; D["x"] = D.A - D.q
pd.set_option("display.width", 200); BETA = {}
for k, G in D.groupby("kind"):
    for nm, S in (("all", G), ("high EC", G[G.hi])):
        if len(S) < 3: continue
        b = float((S.x * S.eM).sum() / (S.x ** 2).sum()); BETA[(k, nm)] = b
        print(f"{k:20s} {nm:8s} n {len(S):3d} | |err| median model {S.eM.abs().median():.3f} vs label {S.eA.abs().median():.3f} | RMSE model {np.sqrt((S.eM**2).mean()):.3f} vs label {np.sqrt((S.eA**2).mean()):.3f} | label closer {(S.eA.abs() < S.eM.abs()).mean():.2f} | best beta {b:.2f} -> RMSE {np.sqrt(((S.eM - b * S.x)**2).mean()):.3f}")
S19 = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); S19["rec"] = S19.row_id.str[:7]; S19["h"] = S19.row_id.str[8:].astype(int)
P = S19.pivot(index="rec", columns="h", values="sub_ec")
print("\n(2) eval application (12th-round prediction -> moved toward the near label with the high-EC beta):")
for r, ref, kind, hcol, refh in (("F13_221", "F13_217", "B2 two days before", 0, 23), ("F47_227", "F47_229", "F next day", 23, 0), ("F47_225", "F47_229", "F2 two days after", 23, 0)):
    A = float(EC[idx[ref], refh]); q = float(P.at[r, hcol]); b = BETA.get((kind, "high EC"), BETA.get((kind, "all")))
    new = q + b * (A - q)
    print(f"   {r}: pred at {hcol}h {q:.3f} | near label {ref} {refh}h {A:.3f} | beta {b:.2f} -> {new:.3f} (change {new - q:+.3f}) | day mean pred {P.loc[r].mean():.3f}")
