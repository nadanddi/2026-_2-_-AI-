# -*- coding: utf-8 -*-
"""EA4 (2026-10-12 집 클로드; SH1 비평 3번: 앵커 공식이 모양을 납작하게 만드는지 — 공식 자체 고치기).  사전 고정 — 실행 전 커밋.
현재 EA2: q_h = p1_h + (A − cm_h), cm_h = p1 의 0..h 누적평균, A = 전날 23시 정답(앵커).  (rows 파일: SG2 = p1, A = EA2 의 0시 값)
대안(모두 0..h 예측만 사용):
  V1 평행 이동: q_h = p1_h + (A − p1_0)
  V2 감쇠 이동: q_h = p1_h + c_h (A − p1_0), c_h = 다른 폴드 1차 앵커 기록에서 (y_h − p1_h) 를 (A − p1_0) 에 최소제곱(시각별, [0,1.2] 자르기)
판정(EV 앵커 기록, 대안마다, 본페로니 2): 3시드 모두 EA2 보다 RMSE 좋음 AND 덩어리 부트스트랩 P(worse) < .0125 AND 최대 기록 몫 ≤ .5. 2차 따로 보고.
참고: SH1(EA2 + .5·M2 모양 보정)도 같은 표로 다시 계산해 나란히 보고.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); ch = Fv[Fv.role == "val"].set_index("rec").chunk
anc = R.groupby("rec").anchored.first(); A_rec = R[R.hour == 0].groupby("rec").EA2.first()
fold = R.groupby("rec").fold.first()
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261238)
out = {}
for sd, G in R.groupby("seed"):
    P1 = G.pivot_table(index="rec", columns="hour", values="SG2"); E2 = G.pivot_table(index="rec", columns="hour", values="EA2"); Y = G.groupby(["rec", "hour"]).y.first().unstack().loc[P1.index]
    a = anc.reindex(P1.index).fillna(False).values.astype(bool); recs = P1.index[a]
    A = A_rec.reindex(recs).values; p0 = P1.loc[recs, 0].values
    V1 = P1.loc[recs].values + (A - p0)[:, None]
    V2 = np.zeros_like(V1); fo = fold.reindex(recs).values; p1mask = np.array([int(r[4:7]) < 179 for r in recs])
    for k in np.unique(fo):
        tr = (fo != k) & p1mask; x = (A - p0)[tr]; yv = (Y.loc[recs].values - P1.loc[recs].values)[tr]
        c = np.clip((x[:, None] * yv).sum(0) / (x ** 2).sum(), 0, 1.2); te = fo == k
        V2[te] = P1.loc[recs].values[te] + c[None, :] * (A - p0)[te][:, None]
    out[sd] = dict(recs=recs, Y=Y.loc[recs].values, EA2=E2.loc[recs].values, V1=V1, V2=V2, c=c)
    print("seed", sd, "anchored", len(recs), "| example c_h (last fold) 0,6,12,18,23:", np.round(c[[0, 6, 12, 18, 23]], 2).tolist())
for arm in ("V1", "V2"):
    cells, c2, per = [], [], []
    for sd, D in out.items():
        cells.append(rm(D[arm] - D["Y"]) / rm(D["EA2"] - D["Y"]) - 1)
        m2 = np.array([int(r[4:7]) >= 179 for r in D["recs"]]); c2.append(rm((D[arm] - D["Y"])[m2]) / rm((D["EA2"] - D["Y"])[m2]) - 1)
        per.append(pd.Series(np.nansum((D[arm] - D["Y"]) ** 2, 1) - np.nansum((D["EA2"] - D["Y"]) ** 2, 1), index=D["recs"]))
    pr = pd.concat(per, axis=1).mean(1); cl = pr.groupby(ch.reindex(pr.index).values).sum(); ii = rng.integers(0, len(cl), (20000, len(cl)))
    pw = float((cl.values[ii].sum(1) >= 0).mean()); share = float(pr.min() / pr.sum()) if pr.sum() < 0 else float("nan")
    print(f"{arm}: cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | pass-2 {' '.join('%+.2f%%' % (100*x) for x in c2)} | improved {int((pr<0).sum())}/{len(pr)} | chunk P {pw:.4f} | top share {share:.2f}"
          f" -> {'PASS' if all(x < 0 for x in cells) and pw < .0125 and share <= .5 else 'FAIL'}")
D = next(iter(out.values()))
print("mean |q_h - q_0| on anchored records: EA2 %.3f | V1 %.3f | truth %.3f" % (np.nanmean(np.abs(D["EA2"] - D["EA2"][:, :1])), np.nanmean(np.abs(D["V1"] - D["V1"][:, :1])), np.nanmean(np.abs(D["Y"] - D["Y"][:, :1]))))
