# -*- coding: utf-8 -*-
"""TSH1 (2026-10-12 집 클로드; SH1 의 온도판).  사전 고정 — 실행 전 커밋.
기준 예측: 온도 EV(ev_validator/temp/local_ckpt/ev_all.pkl) v22_a, v22_b (두 구성 = 두 '시드'로 취급), anchored = 온도 앵커 붙은 기록.
보정기: DS1 v2(인과 수정판)의 M2(LGBM, 대상 r_h = (y_h − q_h) − (y_0 − q_0), 특징 = 같은 기록 0..h 입력 + 예측 모양·수준, 다른 폴드 기록으로 학습)를
  온도 정답·예측으로 그대로 학습.  적용: 앵커 기록에만 P = q + s·G, s ∈ {.5, 1.0} (본페로니 2).
판정(앵커 기록, 구성마다): 두 구성 모두 RMSE 개선 AND 덩어리 부트스트랩 P(worse) < .0125 AND 최대 기록 몫 ≤ .5.  2차 따로 보고.
"""
import sys, os, io
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(here, "ds1_within_day_shape_corrector_v2.py"), encoding="utf-8").read(); src = src[:src.index("rmse = lambda")]
g = {"__name__": "ds1"}; exec(compile(src, "ds1", "exec"), g)
model_g, H_ = g["model_g"], g["H_"]
T = pd.read_pickle(os.path.join(H_, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); T["rec"] = T.row_id.str[:7]
Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); ch = Fv[Fv.role == "val"].set_index("rec").chunk
anc = T.groupby("rec").anchored.first()
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261240)
res = {}
for col in ("v22_a", "v22_b"):
    Q = T.pivot(index="rec", columns="hour", values=col); Y = T.pivot(index="rec", columns="hour", values="sub_temp").loc[Q.index]
    fo = T.groupby("rec").fold.first().reindex(Q.index).values; recs = Q.index.values
    G = model_g("M2", recs, Q.values, Y.values, fo); a = np.array([bool(anc.get(r, False)) for r in recs])
    res[col] = (recs, a, Q.values, G, Y.values); print(col, "anchored records", int(a.sum()), flush=True)
for sv in (.5, 1.0):
    cells, c2, per = [], [], []
    for col, (recs, a, Q, G, Y) in res.items():
        P = Q + sv * G * a[:, None]; m2 = np.array([int(r[4:7]) >= 179 for r in recs])
        cells.append(rm((P - Y)[a]) / rm((Q - Y)[a]) - 1); c2.append(rm((P - Y)[a & m2]) / rm((Q - Y)[a & m2]) - 1)
        per.append(pd.Series(np.nansum((P - Y) ** 2, 1) - np.nansum((Q - Y) ** 2, 1), index=recs)[a])
    pr = pd.concat(per, axis=1).mean(1); cl = pr.groupby(ch.reindex(pr.index).values).sum(); ii = rng.integers(0, len(cl), (20000, len(cl)))
    pw = float((cl.values[ii].sum(1) >= 0).mean()); share = float(pr.min() / pr.sum()) if pr.sum() < 0 else float("nan")
    full = [rm(np.where(res[c][1][:, None], res[c][2] + sv * res[c][3], res[c][2]) - res[c][4]) / rm(res[c][2] - res[c][4]) - 1 for c in res]
    print(f"s {sv}: anchored {len(pr)} | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | pass-2 {' '.join('%+.2f%%' % (100*x) for x in c2)} | improved {int((pr<0).sum())}/{len(pr)} | chunk P {pw:.4f} | top share {share:.2f} | whole EV {' '.join('%+.2f%%' % (100*x) for x in full)}")
    print("   VERDICT", "PASS" if all(x < 0 for x in cells) and pw < .0125 and share <= .5 else "FAIL")
# control: shuffle-target corrector on anchored
for col, (recs, a, Q, G, Y) in res.items():
    GS = model_g("M2S", recs, Q, Y, T.groupby("rec").fold.first().reindex(recs).values)
    print(f"control M2S {col} s .5: anchored {100*(rm((Q + .5*GS - Y)[a]) / rm((Q - Y)[a]) - 1):+.2f}%")
