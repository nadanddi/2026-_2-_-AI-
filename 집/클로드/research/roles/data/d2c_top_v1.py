# -*- coding: utf-8 -*-
"""D2b (진단): 같은 처방(SA: pred_t,h += c_h * w * (q_p,23 - q_t,0), w = 다른 fold 대상으로 적합한 원점 기울기 [0,1], c_h = 1차 DIAG10 0시 잔차 지속)을
DIAG10 OOF(sg 3시드 평균, 앵커 전)와 EV(R3 앵커 전 / EA2 앵커 후)에 똑같이 적용. 대상: v7m 1등 전날 p>=.8, 같은 ID, 앞 번호, 같은 fold.
링크 출처 2가지: v7m(현실) / 진실(linker_v3 truth, 상한). 판정 아님 — EV vs DIAG10 모순이 자료 차이인지 처방 차이인지 가르는 용도."""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env, pickle
import numpy as np, pandas as pd
H = r"C:\work\farmai\집\클로드\research"
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]; O["rec"] = O.row_id.str[:7]
QD = O.groupby(["rec", "hour"]).sg.mean().unstack(); YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack(); FD = O.groupby("rec").fold.first()
p1 = [x for x in QD.index if int(x[4:7]) < 179]; e = (YD.loc[p1] - QD.loc[p1]).values
c = np.array([(e[:, 0] * e[:, h]).sum() / (e[:, 0] ** 2).sum() for h in range(24)])
print("c_h(0,6,12,18,23):", np.round(c[[0, 6, 12, 18, 23]], 2))
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
T = pd.read_pickle(os.path.join(H, "local/linker_v3/truth.pkl")); rr = pickle.load(open(os.path.join(H, "local/linker_v3/base.pkl"), "rb"))["recs"].reset_index(drop=True)
ids = np.array([f"{f}_{d:03d}" for f, d in zip(rr.farm, rr.day)]); TP = {ids[int(z.b)]: ids[int(z.a1)] for _, z in T.iterrows()}
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
FE = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); FE = FE[FE.role == "val"].drop_duplicates("rec").set_index("rec").fold
YE = R.groupby(["rec", "hour"]).y.first().unstack(); anc = R.groupby("rec").anchored.first()
def run(P, Y, FO, link, name, skip_anchored):
    rows = []
    for t in P.index:
        if skip_anchored and bool(anc.get(t, False)): continue
        if link == "v7m":
            if t not in L.index: continue
            z = L.loc[t]
            if z.p < .8 or z.xf: continue
            p = z.prev
        else:
            p = TP.get(t)
        if p is None or p not in P.index or p[:3] != t[:3] or int(p[4:7]) >= int(t[4:7]) or FO[p] != FO[t]: continue
        if link == "v7m" and bool(z.prev_is_labelled) and skip_anchored: continue
        rows.append((t, p, FO[t], TP.get(t) == p))
    A = pd.DataFrame(rows, columns=["t", "p", "fold", "true"])
    new = P.copy()
    for k in A.fold.unique():
        o = A[A.fold != k]; x = P.loc[o.p, 23].values - P.loc[o.t, 0].values; yv = Y.loc[o.t, 0].values - P.loc[o.t, 0].values
        w = float(np.clip((x * yv).sum() / (x * x).sum(), 0, 1))
        for _, r in A[A.fold == k].iterrows(): new.loc[r.t] = P.loc[r.t].values + c * w * (P.at[r.p, 23] - P.at[r.t, 0])
    for part, cond in (("1차", lambda s: int(s[4:7]) < 179), ("2차", lambda s: int(s[4:7]) >= 179)):
        rc = [x for x in A.t if cond(x)]
        if not rc: continue
        d0 = rm(new.loc[rc, 0] - Y.loc[rc, 0]) / rm(P.loc[rc, 0] - Y.loc[rc, 0]) - 1
        dd = rm((new.loc[rc] - Y.loc[rc]).values) / rm((P.loc[rc] - Y.loc[rc]).values) - 1
        per = ((new.loc[rc] - Y.loc[rc]) ** 2).sum(axis=1) - ((P.loc[rc] - Y.loc[rc]) ** 2).sum(axis=1)
        top = per.min() / per.sum() if per.sum() < 0 else np.nan
        At = A.set_index("t"); tr = At.loc[rc, "true"].values; per_ = per.values; xx = (P.loc[A.p, 23].values - P.loc[A.t, 0].values); yy = (Y.loc[A.t, 0].values - P.loc[A.t, 0].values)
        print("    상위 손해:", per.sort_values().tail(4).round(2).to_dict(), "상위 이득:", per.sort_values().head(3).round(2).to_dict()); print(f"    [분해] 진짜링크 SSE변화 {per_[tr].sum():+.3f} (n={tr.sum()}) / 가짜링크 {per_[~tr].sum():+.3f} (n={(~tr).sum()}) / 기준 SSE {((P.loc[rc]-Y.loc[rc])**2).values.sum():.3f} / 전체 w {np.clip((xx*yy).sum()/(xx*xx).sum(),0,1):.2f}")
        print(f"  {name:22s} {link:5s} {part}: n={len(rc):3d} 진짜링크 {A.set_index('t').loc[rc,'true'].mean():.2f} | 0시 {100*d0:+.1f}% 하루 {100*dd:+.1f}% | 개선 {(per<0).sum()}/{len(rc)} | 최대기록몫 {100*top:.0f}%")
for link in ("v7m",):
    run(QD, YD, FD, link, "DIAG10 sg(앵커 전)", False)
    for col in ("R3", "EA2"):
        P = R.groupby(["rec", "hour"])[col].mean().unstack()
        run(P, YE, FE, link, f"EV {col}", col == "EA2")
    print()
