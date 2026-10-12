# -*- coding: utf-8 -*-
"""HI1 (2026-10-12 집 클로드; 사용자: '고EC 날만 따로 모아서 다시 분석').  서술.
고EC = 학습 기록 하루 평균 EC ≥ 1.0.  (1) 개수·농장·차수·달력 위치·같은 동 연속(진단용 연결로 이어지는 고EC 사슬 길이),
(2) 모델(DIAG10 sg, EV EA2/SG2) 예측 vs 정답: 하루 평균 비율·차이, 수준별 압축(정답 하루평균을 예측 하루평균에 회귀), 0시 오차,
(3) 정보: 같은 동 앞뒤 정답(NF1), EV 앵커 여부와 앵커 효과, 전날 같은 동이 고EC 였나(고EC 지속),
(4) 평가: 12회차 예측 하루 평균 ≥ 1.0 인 기록과 그 날짜 정보 상태(NF1 날짜 수준).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
here = os.path.dirname(os.path.abspath(__file__)); H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
CAL = pd.read_csv(os.path.join(here, "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
NF = pd.read_csv(os.path.join(here, "nf1_v1.csv")).set_index("rec")
TL = pd.read_pickle(S3 + r"\truth.pkl"); prev = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
QD = O.groupby(["rec", "hour"]).sg.mean().unstack()
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
QE = R.groupby(["rec", "hour"]).EA2.mean().unstack(); QS = R.groupby(["rec", "hour"]).SG2.mean().unstack(); anc = R.groupby("rec").anchored.first()
rows = []
for i in np.where(TRAIN)[0]:
    r = rid(i); ym = float(np.nanmean(EC[i])); p = prev.get(r)
    rows.append(dict(rec=r, farm=r[:3], p2=int(r[4:7]) >= 179, day=DAY.get(r), y=ym, y0=float(EC[i, 0]), qD=QD.loc[r].mean() if r in QD.index else np.nan,
                     qS=QS.loc[r].mean() if r in QS.index else np.nan, qE=QE.loc[r].mean() if r in QE.index else np.nan, anchored=bool(anc.get(r, False)),
                     prev_high=(float(np.nanmean(EC[idx[p]])) >= 1.0) if (p in idx and TRAIN[idx[p]]) else np.nan,
                     hp=bool(NF.at[r, "true_prev_lab"]) if r in NF.index and NF.at[r, "true_prev_lab"] == NF.at[r, "true_prev_lab"] else False,
                     hn=bool(NF.at[r, "true_next_lab"]) if r in NF.index and NF.at[r, "true_next_lab"] == NF.at[r, "true_next_lab"] else False))
T = pd.DataFrame(rows); H = T[T.y >= 1.0].copy()
print(f"(1) high-EC training records: {len(H)} of {len(T)} | F13 {int((H.farm=='F13').sum())} F47 {int((H.farm=='F47').sum())} | pass-2 {int(H.p2.sum())} | calendar days {sorted(H.day.dropna().astype(int).unique().tolist())}")
print("    previous same-house day also high (truth link): %d / %d known" % (int(H.prev_high.sum()), int(H.prev_high.notna().sum())))
print("    same-house labelled neighbours: prev&next %d, prev only %d, next only %d, neither %d" % (int((H.hp & H.hn).sum()), int((H.hp & ~H.hn).sum()), int((~H.hp & H.hn).sum()), int((~H.hp & ~H.hn).sum())))
print("\n(2) model vs truth on high days (day means):")
for c, nm in (("qD", "DIAG10 sg"), ("qS", "EV SG2 (no anchor)"), ("qE", "EV EA2 (anchor)")):
    G = H.dropna(subset=[c]); slope = np.polyfit(G[c], G.y, 1)[0]
    print(f"   {nm:18s} n {len(G)} | mean truth {G.y.mean():.2f} vs pred {G[c].mean():.2f} | mean ratio pred/truth {(G[c]/G.y).mean():.2f} | RMSE day-mean {np.sqrt(((G[c]-G.y)**2).mean()):.3f} | slope truth~pred {slope:.2f}")
A = T.dropna(subset=["qD"]); print("   all days: ratio pred/truth by truth level band:", A.groupby(pd.cut(A.y, [0, .4, .6, .8, 1.0, 1.5, 2.0, 9]), observed=True).apply(lambda G: round(float((G.qD / G.y).mean()), 2)).to_dict())
print("\n(3) EV anchors on high days: anchored %d / %d | day-mean |err| anchored: SG2 %.3f -> EA2 %.3f | not anchored: SG2 %.3f" % (
    int(H.anchored.sum()), len(H), (H[H.anchored].qS - H[H.anchored].y).abs().mean(), (H[H.anchored].qE - H[H.anchored].y).abs().mean(), (H[~H.anchored].qS - H[~H.anchored].y).abs().mean()))
pd.set_option("display.width", 220)
print(H.sort_values("day")[["rec", "day", "y", "y0", "qD", "qS", "qE", "anchored", "prev_high", "hp", "hn"]].round(2).to_string(index=False))
S = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); S["rec"] = S.row_id.str[:7]; Pm = S.groupby("rec").sub_ec.mean()
EH = Pm[Pm >= 1.0]
print("\n(4) eval records predicted high (12th-round day mean >= 1.0):", len(EH))
for r, v in EH.sort_values(ascending=False).items():
    print(f"   {r} pred {v:.2f} | day {DAY.get(r)} | labelled on prev day {NF.at[r,'lab_prev'] if r in NF.index else 'na'}, next day {NF.at[r,'lab_next'] if r in NF.index else 'na'}")
