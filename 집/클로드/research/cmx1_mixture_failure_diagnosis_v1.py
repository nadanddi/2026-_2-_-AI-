# -*- coding: utf-8 -*-
"""CMX1 (2026-10-15 집 클로드; 사용자 'AG6 를 뜯어봐서 어디가 이상한지, 왜 확률을 섞어도 도움이 안 되는지').  서술 진단(판정 없음), 실행 전 커밋.
AG6(6.709) 와 CMA1(6.943) 의 차이: AG6 T2 후보 = DF(날짜 먼저) 링커가 정한 '어제 날짜'의 기록들(같은 날 다른 동) 지문 softmax 가중;
  CMA1 후보 = FL 0시 후보 전체(날짜 모름 → 여러 날짜 섞임).  가설: 섞기의 가치는 '같은 날짜 이웃 동 수준'에서 나오고, 날짜가 섞이면 잡음.
대상: cert 정답 짝(TL7) 이 있는 학습 기록(숨김 = 학습표와 같음: 같은 농장 ±1 정답 숨김·평가 후보 제외; 후보표 local/cma1_candidates.pkl, FL 가중치).
같은 날짜 판정(현장형, 공유 외기 관측소): 두 기록의 out_temp·out_hum 0..23 시가 모두 1e-6 안에서 같으면 같은 날.
측정:
  (1) 후보 확률 질량: 진짜 전날 / 진짜 전날과 같은 날짜의 다른 기록 / 다른 날짜 / 정답 없는 후보.
  (2) 오늘 0시 배지 y0 추정 오차(RMSE): ① 진짜 전날 A23 − .36  ② FL 1위 A23 − .36  ③ CMA 혼합 Ā − .36(정답 후보만, 확률 가중)
      ④ 1위 후보의 날짜 묶음(같은 날짜 정답 기록) A23 평균 − .36  ⑤ ④ 를 0시 실내 차 |dT| ≤ 1.5 인 기록만  ⑥ 학습 평균 y0(상수)
  (3) EC: 오늘 하루 평균 EC 추정 오차, 같은 ①~⑥ (A = 23시 EC).
  (4) 1위 후보가 틀렸을 때 그 1위 후보가 '진짜 전날과 같은 날짜'인 비율.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u cmx1_mixture_failure_diagnosis_v1.py
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
sys.path.insert(0, os.path.join(HERE, "linker_v7")); sys.path.insert(0, HERE)
import env  # noqa: F401
from feat7 import *  # noqa: F401,F403
import numpy as np, pandas as pd
IDX = {rid(i): i for i in range(N)}; CAND = pickle.load(open(os.path.join(env.LOCAL, "cma1_candidates.pkl"), "rb"))
_X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]; _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
OT = _X.pivot(index="rec", columns="h", values="out_temp"); OH = _X.pivot(index="rec", columns="h", values="out_hum")
sig = {r: tuple(np.round(np.r_[OT.loc[r].values, OH.loc[r].values], 5)) for r in OT.index}
DATEGRP = {}
for r, s in sig.items(): DATEGRP.setdefault(s, []).append(r)
same_date = lambda a, b: a in sig and b in sig and sig[a] == sig[b]
lab_ok = lambda a: a in IDX and bool(TRAIN[IDX[a]]) and np.isfinite(ST[IDX[a], 23])
rows = []
for b, a in zip(TL7.b, TL7.a):
    r, t = rid(int(b)), rid(int(a))
    if r not in CAND or not np.isfinite(ST[IDX[r], 0]): continue
    C = CAND[r]; tot = sum(q for _, q in C)
    m_true = sum(q for c, q in C if c == t); m_sd = sum(q for c, q in C if c != t and same_date(c, t)); m_un = sum(q for c, q in C if not lab_ok(c))
    m_oth = tot - m_true - m_sd - m_un
    top = max(C, key=lambda z: z[1])[0]; L = [(c, q) for c, q in C if lab_ok(c)]
    if not L or not lab_ok(top) or not lab_ok(t): continue
    m = sum(q for _, q in L); est = {}
    for nm, arr in (("ST", ST), ("EC", EC)):
        A = lambda c: float(arr[IDX[c], 23]); grp = [c for c in DATEGRP.get(sig.get(top), []) if lab_ok(c) and c != r]
        grp15 = [c for c in grp if abs(IN_[IDX[r], 0, 0] - IN_[IDX[c], 23, 0]) <= 1.5]
        est[nm] = dict(true=A(t), top1=A(top), mix=sum(q * A(c) for c, q in L) / m, dgrp=np.mean([A(c) for c in grp]) if grp else np.nan, dgrp15=np.mean([A(c) for c in grp15]) if grp15 else np.nan)
    rows.append(dict(rec=r, p2=NUM[IDX[r]] >= 179, m_true=m_true / tot, m_samedate=m_sd / tot, m_otherdate=m_oth / tot, m_unlab=m_un / tot, top_ok=top == t, top_samedate=(top != t) and same_date(top, t),
                     y0=float(ST[IDX[r], 0]), ydm=float(np.nanmean(EC[IDX[r]])), n_grp=len(DATEGRP.get(sig.get(top), [])),
                     **{f"T_{k}": v for k, v in est["ST"].items()}, **{f"E_{k}": v for k, v in est["EC"].items()}))
D = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(f"targets {len(D)} (pass-2 {int(D.p2.sum())})")
print("(1) probability mass:", D[["m_true", "m_samedate", "m_otherdate", "m_unlab"]].mean().round(3).to_dict(), "| pass-2:", D[D.p2][["m_true", "m_samedate", "m_otherdate", "m_unlab"]].mean().round(3).to_dict())
print("(4) top-1 correct %.3f | top-1 wrong but SAME DATE as true prev %.3f | wrong & other date %.3f" % (D.top_ok.mean(), D.top_samedate.mean(), 1 - D.top_ok.mean() - D.top_samedate.mean()))
cy0 = D.y0.mean(); cyd = D.ydm.mean()
def tab(sub, tag):
    out = {}
    for k in ("true", "top1", "mix", "dgrp", "dgrp15"):
        mT = sub[f"T_{k}"].notna(); mE = sub[f"E_{k}"].notna()
        out[k] = (np.sqrt(((sub[f"T_{k}"] - .36 - sub.y0)[mT] ** 2).mean()), np.sqrt(((sub[f"E_{k}"] - sub.ydm)[mE] ** 2).mean()), int(mT.sum()))
    out["const"] = (np.sqrt(((cy0 - sub.y0) ** 2).mean()), np.sqrt(((cyd - sub.ydm) ** 2).mean()), len(sub))
    print(f"(2)(3) {tag}: " + " | ".join(f"{k} T {v[0]:.3f} EC {v[1]:.3f} (n {v[2]})" for k, v in out.items()))
tab(D, "all"); tab(D[D.p2], "pass-2"); tab(D[~D.top_ok], "top-1 WRONG"); tab(D[D.top_ok], "top-1 right")
print("date-group size of top-1 (records sharing its outdoor day): median", int(D.n_grp.median()), "| share with >=2:", round(float((D.n_grp >= 2).mean()), 3))
D.to_csv(os.path.join(HERE, "cmx1_rows_v1.csv"), index=False)
