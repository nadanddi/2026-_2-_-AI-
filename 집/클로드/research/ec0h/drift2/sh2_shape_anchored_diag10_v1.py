# -*- coding: utf-8 -*-
"""SH2 (2026-10-12 집 클로드; SH1 비평 2번: 두 번째 검증기 DIAG10 에서 같은 방향인지).  사전 고정 — 실행 전 커밋.
DIAG10 CUR sg(시드 3개, fold 열)에 EA2 방식 앵커를 붙임: 짝찾기 v7 거울 CV 선택(cv_all400_choice_v7m.csv)에서
  p >= .8, prev_is_labelled, 같은 ID(xf 아님), prev 가 그 DIAG10 폴드의 검증 기록·±1 이웃(숨김)이 아닌 경우 → q_h = sg_h + (A − sg 의 0..h 누적평균), A = prev 23시 정답.
그 위에 DS1 v2 M2 모양 보정(다른 폴드 학습, 대상은 앵커 붙은 q 기준)을 앵커 기록에만 s = .5 로 적용.
판정: 앵커 기록 RMSE 3시드 모두 개선 AND 기록 부트스트랩 P(worse) < .025 AND 최대 기록 몫 ≤ .5.  2차 따로 보고.  (DIAG10 은 기록 단위 부트스트랩)
"""
import sys, os, io
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(here, "ds1_within_day_shape_corrector_v2.py"), encoding="utf-8").read(); src = src[:src.index("rmse = lambda")]
g = {"__name__": "ds1"}; exec(compile(src, "ds1", "exec"), g)
model_g, H_ = g["model_g"], g["H_"]
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
Ylab = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Ylab["rec"] = Ylab.row_id.str[:7]; Ylab["h"] = Ylab.row_id.str[8:].astype(int); EC23 = Ylab[Ylab.h == 23].set_index("rec").sub_ec
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261242)
res = {}
for sd, G in O.groupby("seed"):
    Q = G.pivot_table(index="rec", columns="hour", values="sg"); Y = G.groupby(["rec", "hour"]).sub_ec.first().unstack().loc[Q.index]
    fo = G.groupby("rec").fold.first().reindex(Q.index)
    hidden = {k: {(r[:3], int(r[4:7]) + j) for r in fo[fo == k].index for j in (-1, 0, 1)} for k in fo.unique()}
    QE = Q.copy(); a = np.zeros(len(Q), bool); cm = Q.cumsum(1) / np.arange(1, 25)
    for n, r in enumerate(Q.index):
        if r not in L.index: continue
        pr = L.loc[r]
        if pr.xf or pr.p < .8 or not pr.prev_is_labelled or pr.prev not in EC23.index: continue
        if (pr.prev[:3], int(pr.prev[4:7])) in hidden[fo[r]]: continue
        QE.loc[r] = Q.loc[r].values + (EC23[pr.prev] - cm.loc[r].values); a[n] = True
    recs = Q.index.values
    Gm = model_g("M2", recs, QE.values, Y.values, fo.values)
    res[sd] = (recs, a, Q.values, QE.values, QE.values + .5 * Gm * a[:, None], Y.values)
    print("seed", sd, "anchored", int(a.sum()), "| anchor alone on anchored: %+.2f%%" % (100 * (rm((QE.values - Y.values)[a]) / rm((Q.values - Y.values)[a]) - 1)), flush=True)
cells, c2, per = [], [], []
for sd, (recs, a, Q, QE, P, Y) in res.items():
    m2 = np.array([int(r[4:7]) >= 179 for r in recs])
    cells.append(rm((P - Y)[a]) / rm((QE - Y)[a]) - 1); c2.append(rm((P - Y)[a & m2]) / rm((QE - Y)[a & m2]) - 1 if (a & m2).any() else np.nan)
    per.append(pd.Series(np.nansum((P - Y) ** 2, 1) - np.nansum((QE - Y) ** 2, 1), index=recs)[a])
pr = pd.concat(per, axis=1).mean(1); ii = rng.integers(0, len(pr), (20000, len(pr))); pw = float((pr.values[ii].sum(1) >= 0).mean())
share = float(pr.min() / pr.sum()) if pr.sum() < 0 else float("nan")
print(f"SH2 (shape .5 on anchored, vs anchored EA2-style): records {len(pr)} | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | pass-2 {' '.join('%+.2f%%' % (100*x) for x in c2)} | improved {int((pr<0).sum())}/{len(pr)} | P {pw:.4f} | top share {share:.2f}")
print("VERDICT SH2:", "PASS" if all(x < 0 for x in cells) and pw < .025 and share <= .5 else "FAIL")
