# -*- coding: utf-8 -*-
"""SP1c (2026-10-12 집 클로드; 사용자: 'SP1 1차 손해가 동이 일자마다 섞여 있어서인가').  서술 — 저장된 SP1 사후확률 + 학습 정답.
대상 222 기록마다: 그 세계에서 보이는 '전날(달력 d−1)' 정답 기록 수 n_prev, 그 기록들 23시 EC 의 퍼짐(max−min) spread,
0시 사후 정답확률합 Pl0, 혼합 수준 L0, 진짜 전날(확실 연결) 이 보이는지·그 23시 EC.
보고: 1차/2차별로 n_prev·spread·Pl0 분포, 0시 오차 |y0−L0| vs |y0−q0| (q0 = DIAG10 sg 시드평균) 를 spread 구간·Pl0 구간별로."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
here = os.path.dirname(os.path.abspath(__file__)); H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
D = pd.read_csv(os.path.join(here, "sp1_posteriors_v1.csv"), index_col=0)
CAL = pd.read_csv(os.path.join(H_, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; q0 = O[O.hour == 0].groupby("rec").sg.mean()
TL = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TL[TL.cert]; truth = {rid(int(b)): rid(int(a)) for a, b in zip(TLc.a1, TLc.b)}
def world_cv(r):
    f = FARM[r]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo = NUM[FARM == f].min(); s = lo + ((NUM[r] - lo) // 10) * 10; m = s + sh
    rem = (FARM == f) & ((NUM == s - 1) | (NUM == s + 10)); oth = (FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)
    blk = (FARM == f) & (NUM >= s) & (NUM < s + 10); hidden = rem | oth | blk
    lab = TRAIN & ~hidden; vis = lab | ((FARM == f) & (NUM < NUM[r]) & ~rem & ~oth); vis[r] = False
    return vis, lab
rows = []
for r in D.index:
    t = idx[r]; vis, lab = world_cv(t); d0 = DAY[r]
    pv = [j for j in np.where(vis & lab)[0] if DAY.get(rid(j)) == d0 - 1]
    e = np.array([EC[j, 23] for j in pv]); tp = truth.get(r); tvis = tp is not None and vis[idx[tp]] and lab[idx[tp]]
    rows.append(dict(rec=r, p2=int(r[4:7]) >= 179, n_prev=len(pv), spread=float(e.max() - e.min()) if len(e) > 1 else 0.0,
                     Pl0=D.at[r, "Pl0"], L0=D.at[r, "L0"], y0=float(EC[t, 0]), q0=q0.get(r, np.nan), truth_visible=tvis,
                     truthA=float(EC[idx[tp], 23]) if tp else np.nan, hard=D.at[r, "hard"]))
T = pd.DataFrame(rows); T["errL"] = (T.y0 - T.L0).abs(); T["errQ"] = (T.y0 - T.q0).abs()
pd.set_option("display.width", 220)
for p in (False, True):
    S = T[T.p2 == p]
    print(f"\n=== pass-{2 if p else 1}: n {len(S)} | prev-day labelled houses n_prev: {S.n_prev.value_counts().sort_index().to_dict()} | median spread {S.spread.median():.2f} | median Pl0 {S.Pl0.median():.2f} | true prev visible {S.truth_visible.mean():.2f} (of known)")
    S = S.assign(sb=pd.cut(S.spread, [-.01, .1, .3, .6, 9]), pb=pd.cut(S.Pl0, [-.01, .3, .6, .8, 1.01]))
    print("  by spread of prev-day EC: n | median |y0-L0| | median |y0-q0| | share where L0 closer")
    print(S.groupby("sb", observed=True).apply(lambda G: pd.Series(dict(n=len(G), eL=G.errL.median(), eq=G.errQ.median(), Lbetter=(G.errL < G.errQ).mean()))).round(3).to_string())
    print("  by Pl0:")
    print(S.groupby("pb", observed=True).apply(lambda G: pd.Series(dict(n=len(G), eL=G.errL.median(), eq=G.errQ.median(), Lbetter=(G.errL < G.errQ).mean()))).round(3).to_string())
T.to_csv(os.path.join(here, "sp1c_v1.csv"), index=False)
