# -*- coding: utf-8 -*-
"""FU1 (2026-10-12 집 클로드; '우리만의 답' — 규정 안 0시 정보원을 기록별 신뢰도로 가중 결합).  사전 고정 — 실행 전 커밋.
검증기: EV(ev_folds_v1: 숨김 = val + del), 기준 예측 SG2(앵커 전 모델), 현재 최종 EA2.  시드 3개.
기록 t 의 정보원(모두 SG2_t,0 = b 에 대한 차이, 없으면 0 + 존재 표시):
  x1 정답 앵커 A − b (EA2 앵커 붙은 기록; A = EA2_t,0)
  x2 짝찾기 v7 1등이 정답 있는 같은 ID 기록(그 폴드에서 숨기지 않음)이고 앵커 아님: (EC23_prev − b)·p, (EC23_prev − b)·(1 − p)
  x2x 1등이 정답 있는 다른 ID 기록: (EC23_prev − b)·p
  x3 1등이 같은 폴드 같은 ID 앞번호 검증 기록(정답 없음, 평가와 같은 조건): (EA2_prev,23 − b)·p   [계산된 예측 — 허용 6.631]
  x4 같은 조건의 저녁 평균: (mean EA2_prev,18..23 − b)·p
  x5 다음날 train 앵커(NA1 0시 선택): (EC0_next − b)/ncand
보조 특징 없음(선형).  모델: Ridge(alpha 1, 절편) 를 다른 폴드 기록 전체로 학습해 r0 = y_t,0 − b 예측.
팔(본페로니 2):
  F1  q_h = SG2_h + c_h·r0_hat  (c_h = 다른 폴드에서 (y_h − SG2_h) 를 (y_0 − SG2_0) 에 회귀)
  F2  EA2 앵커 기록은 EA2 그대로, 나머지는 F1
판정(EV 전체 기록, 팔마다): 3시드 모두 EA2 보다 RMSE 좋음 AND 덩어리 부트스트랩 P(worse) < .0125 AND 최대 기록 몫 ≤ .5.  1·2차 따로 보고.
통과해도 DIAG10 재현 전에는 '후보' 아님(SH1 교훈).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); V = Fv[Fv.role == "val"].set_index("rec")
hid = {k: set(Fv[Fv.fold == k].rec) for k in Fv.fold.unique()}
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
Yl = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Yl["rec"] = Yl.row_id.str[:7]; Yl["h"] = Yl.row_id.str[8:].astype(int)
EC23 = Yl[Yl.h == 23].set_index("rec").sub_ec; EC0 = Yl[Yl.h == 0].set_index("rec").sub_ec
anc = R.groupby("rec").anchored.first()
NA = pd.read_csv(os.path.join(H_, "ec0h", "verify3", "na1_picks_v1.csv")); NA = NA[NA.h == 0].set_index("t")
FE = ["x1", "i1", "x2p", "x2q", "i2", "x2x", "x3", "i3", "x4", "x5", "i5"]
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261243)
outs = {}
for sd, G in R.groupby("seed"):
    S = G.pivot_table(index="rec", columns="hour", values="SG2"); E = G.pivot_table(index="rec", columns="hour", values="EA2")
    Y = G.groupby(["rec", "hour"]).y.first().unstack().loc[S.index]; recs = S.index
    rows = []
    for t in recs:
        k = V.at[t, "fold"]; b = S.at[t, 0]; f = dict.fromkeys(FE, 0.0)
        if bool(anc.get(t, False)): f["x1"] = E.at[t, 0] - b; f["i1"] = 1
        if t in L.index:
            pr = L.loc[t]; p, pv = float(pr.p), pr.prev
            lab_vis = (pv in EC23.index) and (pv not in hid[k])
            if lab_vis and not pr.xf and not f["i1"]:
                d = EC23[pv] - b; f["x2p"], f["x2q"], f["i2"] = d * p, d * (1 - p), 1
            elif lab_vis and pr.xf and not f["i1"]:
                f["x2x"] = (EC23[pv] - b) * p
            elif (pv in V.index) and V.at[pv, "fold"] == k and pv[:3] == t[:3] and int(pv[4:7]) < int(t[4:7]) and pv in E.index:
                f["x3"] = (E.at[pv, 23] - b) * p; f["x4"] = (E.loc[pv, 18:23].mean() - b) * p; f["i3"] = 1
        if t in NA.index:
            r_ = NA.loc[t]; r_ = r_.iloc[0] if isinstance(r_, pd.DataFrame) else r_
            if r_.pick in EC0.index: f["x5"] = (EC0[r_.pick] - b) / r_.ncand; f["i5"] = 1
        f.update(rec=t, fold=k, r0=Y.at[t, 0] - b); rows.append(f)
    D = pd.DataFrame(rows).set_index("rec"); hat = pd.Series(0.0, index=D.index); C = {}
    for k in D.fold.unique():
        tr, te = D.fold != k, D.fold == k
        m = Ridge(1.0).fit(D.loc[tr, FE], D.loc[tr, "r0"]); hat[te] = m.predict(D.loc[te, FE])
        e = (Y.loc[tr[tr].index] - S.loc[tr[tr].index]).values; C[k] = np.array([np.nansum(e[:, 0] * e[:, h]) / np.nansum(e[:, 0] ** 2) for h in range(24)])
    F1 = S.copy(); F2 = E.copy()
    for t in recs:
        F1.loc[t] = S.loc[t].values + C[D.at[t, "fold"]] * hat[t]
        if not bool(anc.get(t, False)): F2.loc[t] = F1.loc[t]
    outs[sd] = dict(S=S, E=E, Y=Y, F1=F1, F2=F2)
    print("seed", sd, "| coefs (last fold):", dict(zip(FE, np.round(m.coef_, 2))), "| sources present: anchor %d, lab %d, xf %d, prev-eval %d, next %d" % (
        int(D.i1.sum()), int(D.i2.sum()), int((D.x2x != 0).sum()), int(D.i3.sum()), int(D.i5.sum())), flush=True)
ch = V.chunk
for arm in ("F1", "F2"):
    for part, cond in (("all", lambda r: True), ("pass-1", lambda r: int(r[4:7]) < 179), ("pass-2", lambda r: int(r[4:7]) >= 179)):
        cells, per = [], []
        for sd, O in outs.items():
            idx = [r for r in O["S"].index if cond(r)]
            cells.append(rm((O[arm].loc[idx] - O["Y"].loc[idx]).values) / rm((O["E"].loc[idx] - O["Y"].loc[idx]).values) - 1)
            per.append(((O[arm].loc[idx] - O["Y"].loc[idx]) ** 2).sum(1) - ((O["E"].loc[idx] - O["Y"].loc[idx]) ** 2).sum(1))
        pr = pd.concat(per, axis=1).mean(1); cl = pr.groupby(ch.reindex(pr.index).values).sum(); ii = rng.integers(0, len(cl), (20000, len(cl)))
        pw = float((cl.values[ii].sum(1) >= 0).mean()); share = float(pr.min() / pr.sum()) if pr.sum() < 0 else float("nan")
        print(f"{arm} {part:6s}: vs EA2 cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((pr<-1e-9).sum())}/{int((pr.abs()>1e-9).sum())} changed | chunk P {pw:.4f} | top share {share:.2f}")
        if part == "all": print(f"   VERDICT {arm}:", "PASS" if all(x < 0 for x in cells) and pw < .0125 and share <= .5 else "FAIL")
