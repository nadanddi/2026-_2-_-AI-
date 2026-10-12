# -*- coding: utf-8 -*-
"""SH1 v2 (DS1 v2 인과 수정판 사용, 판정 기준 동일) — SH1 (2026-10-12 집 클로드; '하루 안 모양 보정을 0시가 정확한 앵커 기록에만').  사전 고정 — 실행 전 커밋.
DS1(6.661)의 M2(LGBM, 같은 기록 0..h 입력 + 예측 모양, 다른 폴드 학습) 보정 G 를 그대로 계산하고, EV 에서 EA2 앵커가 붙은 기록에만 적용:
  P = EA2 + s·G (앵커 기록), 나머지는 그대로.  s ∈ {.5, 1.0} (본페로니 2).
판정(EV 앵커 기록만, 1·2차 합): 3시드 모두 RMSE 개선 AND 덩어리 부트스트랩 P(worse) < .0125 AND 최대 기록 몫 ≤ .5.  2차 따로 보고.
"""
import sys, os, io
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds1_within_day_shape_corrector_v2.py"), encoding="utf-8").read()
src = src[:src.index("rmse = lambda")]
g = {"__name__": "ds1"}; exec(compile(src, "ds1", "exec"), g)
load, model_g = g["load"], g["model_g"]; H_ = g["H_"]
D = load("EV")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
anc = R.groupby("rec").anchored.first(); Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); ch = Fv[Fv.role == "val"].set_index("rec").chunk
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261237)
res = {}
for s, (recs, Q, Y, fo) in D.items():
    G = model_g("M2", recs, Q, Y, fo); a = np.array([bool(anc.get(r, False)) for r in recs])
    for sv in (.5, 1.0):
        P = Q + sv * G * a[:, None]
        res[(s, sv)] = (recs, a, Q, P, Y)
    print("seed", s, "anchored records", int(a.sum()), flush=True)
for sv in (.5, 1.0):
    cells, per = [], []
    for s in D:
        recs, a, Q, P, Y = res[(s, sv)]
        cells.append(rm((P - Y)[a]) / rm((Q - Y)[a]) - 1)
        per.append(pd.Series(np.nansum((P - Y) ** 2, 1) - np.nansum((Q - Y) ** 2, 1), index=recs)[a])
    pr = pd.concat(per, axis=1).mean(1); cl = pr.groupby(ch.reindex(pr.index).values).sum()
    ii = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
    share = float(pr.min() / pr.sum()) if pr.sum() < 0 else float("nan")
    p2 = np.array([int(r[4:7]) >= 179 for r in pr.index])
    c2 = [rm((res[(s, sv)][3] - res[(s, sv)][4])[res[(s, sv)][1] & np.array([int(r[4:7]) >= 179 for r in res[(s, sv)][0]])]) /
          rm((res[(s, sv)][2] - res[(s, sv)][4])[res[(s, sv)][1] & np.array([int(r[4:7]) >= 179 for r in res[(s, sv)][0]])]) - 1 for s in D]
    print(f"s {sv}: anchored records {len(pr)} | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | pass-2 {' '.join('%+.2f%%' % (100*x) for x in c2)} | improved {int((pr<0).sum())}/{len(pr)} | chunk P {pw:.4f} | top share {share:.2f}")
    print("   VERDICT", "PASS" if all(x < 0 for x in cells) and pw < .0125 and share <= .5 else "FAIL")
