# -*- coding: utf-8 -*-
"""HD1 (2026-10-12 집 클로드; 고EC 오차 상위 기록 기록 단위 해부, 서술).
EV EA2 SSE 상위 25 기록마다: 진짜 전날(truth a1, 진단용 — EC 연속으로 만든 것) 과 그 23시 EC, 확실 여부, 전날이 그 EV 폴드에서 보이는가(숨김 = val+del),
짝찾기 v7m 선택·p·같은지, EA2 앵커 여부, 0시 예측 vs 정답, 같은 날짜 형제(DM11 같은 날) 정답 EC 평균, 진짜 다음날(succ) 0시 EC.
분류: A '정답 있는 전날이 보였는데 못 씀(식별 실패)', B '전날이 숨겨짐(EV 구조)', C '전날 자체가 정답 없음/다른 날짜', D '전날 정답이 있어도 그날 크게 변함(|y0 − A| > .2)'."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
D = pd.read_csv(os.path.join(H_, "ec0h", "rise", "er2_ec_records_v1.csv"), index_col=0).sort_values("SSE", ascending=False)
CAL = pd.read_csv(os.path.join(H_, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); V = Fv[Fv.role == "val"].set_index("rec")
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
TL = pd.read_pickle(S3 + r"\truth.pkl"); pre = {rid(int(b)): (rid(int(a)), bool(c)) for a, b, c in zip(TL.a1, TL.b, TL.cert)}
suc = {}
for b, (a, c) in pre.items():
    if c: suc.setdefault(a, []).append(b)
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
Q0 = R[R.hour == 0].groupby("rec").EA2.mean()
rows = []
for r in D.index[:25]:
    i = idx[r]; k = V.at[r, "fold"]; hid = set(Fv[Fv.fold == k].rec)
    a, cert = pre.get(r, (None, False)); ja = idx.get(a) if a else None
    A = float(EC[ja, 23]) if (ja is not None and TRAIN[ja]) else np.nan
    vis = (a is not None) and TRAIN[ja] and (a not in hid)
    pick = L.prev.get(r); p = L.p.get(r)
    d = DAY.get(r); sib = [j for j in range(N) if j != i and DAY.get(rid(j)) == d and TRAIN[j]]
    sib_ec = float(np.mean([EC[j].mean() for j in sib])) if sib else np.nan
    sc = suc.get(r, []); nxt0 = float(EC[idx[sc[0]], 0]) if sc and TRAIN[idx[sc[0]]] else np.nan
    y0 = float(EC[i, 0])
    if a is None or not TRAIN[ja]: cls = "C 전날 정답 없음"
    elif not vis: cls = "B 전날 숨겨짐(EV)"
    elif abs(y0 - A) > .2: cls = "D 전날 있어도 그날 변함"
    else: cls = "A 식별 실패" if pick != a or (p is not None and p < .8) else "A' 앵커 있어야 했음"
    rows.append(dict(rec=r, share=D.at[r, "share"], mean_err=D.at[r, "mean_err"], y_mean=D.at[r, "y_mean"], y0=y0, pred0=Q0.get(r, np.nan),
                     true_prev=a, cert=cert, prev_A23=A, prev_visible=vis, v7m=pick, p=p, pick_ok=(pick == a), sib_ec=sib_ec, next0=nxt0, cls=cls))
T = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(T.round(3).to_string(index=False))
print("\nSSE share by class:", T.groupby("cls").share.sum().round(3).to_dict(), "| count:", T.cls.value_counts().to_dict())
T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hd1_v1.csv"), index=False)
