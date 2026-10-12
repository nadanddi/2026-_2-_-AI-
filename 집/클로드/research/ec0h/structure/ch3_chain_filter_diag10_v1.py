# -*- coding: utf-8 -*-
"""CH3 (2026-10-12 집 클로드; 사용자: '거르기 규칙 검증').  사전 고정(이 파일 커밋 후 실행).
CH2(6.693)에서 사후로 떠올린 '자기 사슬만 보는' 거르기 규칙을 CH2 와 다른 검증 세계에서 판정.
  규칙 F1: R→P1 짝 확신 p_r ≥ .95 AND P1→P2 p_p1 ≥ .95 AND P1 이 R 과 같은 ID·앞 번호 AND P2 도 R 과 같은 ID.
검증 세계(DIAG10 흉내): 저장된 DIAG10 OOF 예측(연구실/클로드/results/ec_rv1_preds_v1.csv, CUR, sg, 시드 47/1414/6464)을 모든 학습 기록의 '앵커 없는 예측'으로 보고,
  학습 기록 R 마다 짝찾기 v7 거울 CV 선택(cv_all400_choice_v7m) R→P1, P1→P2 를 따라 'R·P1 은 정답 숨김, P2 는 정답 있음'으로 가정.
  대상 = P1 ≠ R, P2 ∉ {R, P1}, P2 학습 기록.  CH2 의 28 대상 기록(ch2_v1.log)은 판정에서 제외(규칙 고른 자료와 분리).
  주안 C2L: A = P2 의 23시 정답, R 예측 q_h + (A − q 의 0..h 누적평균) (EA2 공식), 학습 정답 범위로 자름.
  서술 C2P: A = P1 에 같은 공식(P2 정답)을 건 예측의 23시 값.
판정(F1 적용 대상, 주안 C2L만, α .025): 3시드 모두 대상 기록 RMSE 개선 AND 기록 단위 부트스트랩(4000) P(악화) < .025 (3시드 모두)
  AND 이득 상위 1기록 몫 ≤ .5 (시드 평균).  비교 서술: 거르기 없음(p ≥ .8, 같은 ID P1) 대상, 진짜 사슬 비율(진단용 truth a1).
"""
import sys, os, re
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd
H = r"C:\work\farmai\집\클로드\research"; here = os.path.dirname(os.path.abspath(__file__))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
CH = pd.read_csv(os.path.join(H, "local", "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").set_index("rec")
TL = pd.read_pickle(os.path.join(H, "local", "linker_v3", "truth.pkl"))
sys.path.insert(0, os.path.join(H, "linker_v4"))
from feat4 import FARM, NUM  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; TP = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}
log = open(os.path.join(here, "ch2_v1.log"), encoding="utf-8", errors="ignore").read()
CH2 = set(re.findall(r"^(F\d\d_\d\d\d)\s+\d+\s+F", log, re.M))
Y = O.groupby(["rec", "hour"]).sub_ec.first().unstack(); LAB = set(Y.index); lo, hi = float(Y.values.min()), float(Y.values.max())
rows = []
for r in Y.index:
    if r not in CH.index: continue
    c = CH.loc[r]; p1 = c.prev
    if not (isinstance(p1, str) and p1 != r and p1 in CH.index and p1 in LAB and p1[:3] == r[:3] and int(p1[4:7]) < int(r[4:7])): continue
    c1 = CH.loc[p1]; p2 = c1.prev
    if not (isinstance(p2, str) and p2 not in (r, p1) and p2 in LAB): continue
    base = c.p >= .8 and c1.p >= .8
    f1 = c.p >= .95 and c1.p >= .95 and p2[:3] == r[:3]
    if base or f1:
        rows.append(dict(rec=r, p1=p1, p2=p2, p_r=c.p, p_p1=c1.p, base=base, F1=f1, true=(TP.get(r) == p1) and (TP.get(p1) == p2), inCH2=r in CH2, A=float(Y.at[p2, 23])))
T = pd.DataFrame(rows).set_index("rec")
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def ea2(q, A):
    return np.clip(q + (A - np.cumsum(q) / np.arange(1, 25)), lo, hi)
res = []; per = []; rng = np.random.default_rng(20261012)
for sd, G in O.groupby("seed"):
    Q = G.pivot_table(index="rec", columns="hour", values="sg")
    for grp, sel in (("F1 (judged, CH2 excluded)", T.F1 & ~T.inCH2), ("no filter p>=.8 (CH2 excluded)", T.base & ~T.inCH2), ("F1 incl. CH2", T.F1)):
        S = T[sel]
        if len(S) == 0: continue
        b = np.array([Q.loc[r].values for r in S.index]); y = np.array([Y.loc[r].values for r in S.index])
        cl = np.array([ea2(Q.loc[r].values, S.at[r, "A"]) for r in S.index])
        cp = np.array([ea2(Q.loc[r].values, ea2(Q.loc[S.at[r, "p1"]].values, S.at[r, "A"])[23]) for r in S.index])
        db = ((b - y) ** 2).sum(1); dl = ((cl - y) ** 2).sum(1)
        idx = rng.integers(0, len(S), (4000, len(S))); pw = float((dl[idx].sum(1) >= db[idx].sum(1)).mean())
        g = db - dl; share = float(g.max() / g.sum()) if g.sum() > 0 else np.nan
        tr = S.true.values
        res.append(dict(seed=sd, group=grp, n=len(S), true_rate=tr.mean(), base=rm(b - y), C2L=rm(cl - y), C2P=rm(cp - y), P_worse=pw, top_share=share,
                        improved=int((dl < db).sum()), C2L_true=rm((cl - y)[tr]) if tr.any() else np.nan, base_true=rm((b - y)[tr]) if tr.any() else np.nan,
                        C2L_false=rm((cl - y)[~tr]) if (~tr).any() else np.nan, base_false=rm((b - y)[~tr]) if (~tr).any() else np.nan))
        if grp.startswith("F1 (judged") and sd == 47:
            for i, r in enumerate(S.index): per.append(dict(rec=r, p1=S.at[r, "p1"], p2=S.at[r, "p2"], true=S.at[r, "true"], y=y[i].mean(), base=b[i].mean(), C2L=cl[i].mean(), d_sse=dl[i] - db[i]))
R = pd.DataFrame(res); R["C2L_%"] = 100 * (R.C2L / R.base - 1); R["C2P_%"] = 100 * (R.C2P / R.base - 1)
pd.set_option("display.width", 260); print(R.round(4).to_string(index=False))
J = R[R.group.str.startswith("F1 (judged")]
ok = (J.C2L < J.base).all() and (J.P_worse < .025).all() and J.top_share.mean() <= .5
print(f"\nF1 judged (C2L): all seeds better {bool((J.C2L < J.base).all())} | P_worse {J.P_worse.round(4).tolist()} | top share {J.top_share.mean():.2f} -> {'ADOPT' if ok else 'FAIL'}")
print("\nper record (seed 47, F1 judged):"); print(pd.DataFrame(per).sort_values("d_sse").round(3).to_string(index=False))
