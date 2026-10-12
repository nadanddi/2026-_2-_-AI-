# -*- coding: utf-8 -*-
"""CB2 (2026-10-12 집 클로드; 사용자: '검증 표본 늘려봐').  사전 고정 — 실행 전 커밋.
CB1 에서 EV 로 정한 늦은 시각 감쇠 w = [.90, .82, .73, .59](0-5/6-11/12-17/18-23시, 다른 동 제거 없음)를 다시 맞추지 않고
2차형 세계(SP2: 같은 ID ±1 만 숨김)의 딱딱한 앵커 118 기록(sp2_posteriors_v1.csv 'hard', 기본 DIAG10 sg)에 적용.
판정: 감쇠가 하루 전체 앵커보다 3시드 모두 좋음 AND 블록(농장+번호//10) 부트스트랩 P(worse) < .025 AND 기본보다 3시드 모두 좋음.
보고: 전체 / CB1 DIAG10 판정 34기록과 겹치지 않는 기록만 / 1·2차.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
W = np.array([.90, .82, .73, .59]); blk = np.array([0] * 6 + [1] * 6 + [2] * 6 + [3] * 6)
D = pd.read_csv(os.path.join(here, "sp2_posteriors_v1.csv"), index_col=0); D = D[D.hard.notna()]
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
Yl = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Yl["rec"] = Yl.row_id.str[:7]; Yl["h"] = Yl.row_id.str[8:].astype(int); EC23 = Yl[Yl.h == 23].set_index("rec").sub_ec
# CB1 DIAG10 judged set (to report the non-overlapping part)
c34 = set()
for sd, G in O.groupby("seed"):
    fo = G.groupby("rec").fold.first(); hidden = {k: {(r[:3], int(r[4:7]) + j) for r in fo[fo == k].index for j in (-1, 0, 1)} for k in fo.unique()}
    for r in fo.index:
        if r in L.index and not L.at[r, "xf"] and L.at[r, "p"] >= .8 and L.at[r, "prev_is_labelled"] and L.at[r, "prev"] in EC23.index and (L.at[r, "prev"][:3], int(L.at[r, "prev"][4:7])) not in hidden[fo[r]]: c34.add(r)
    break
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261253)
recs = [r for r in D.index if r in YD.index]; A = D.hard.reindex(recs).values; out = {}
for sd, G in O.groupby("seed"):
    Q = G.pivot_table(index="rec", columns="hour", values="sg").loc[recs].values; cm = np.cumsum(Q, 1) / np.arange(1, 25)
    full = Q + (A[:, None] - cm); fade = Q + W[blk][None, :] * (A[:, None] - cm); out[sd] = (Q, full, fade)
Yv = YD.loc[recs].values
for part, sel in (("all", lambda r: True), ("not in CB1 34", lambda r: r not in c34), ("pass-1", lambda r: int(r[4:7]) < 179), ("pass-2", lambda r: int(r[4:7]) >= 179)):
    m = np.array([sel(r) for r in recs])
    if not m.any(): continue
    res = [(rm(Q[m] - Yv[m]), rm(full[m] - Yv[m]), rm(fade[m] - Yv[m])) for Q, full, fade in out.values()]
    per = pd.Series(np.mean([((fade[m] - Yv[m]) ** 2).sum(1) - ((full[m] - Yv[m]) ** 2).sum(1) for Q, full, fade in out.values()], 0), index=np.array(recs)[m])
    cl = per.groupby([r[:3] + str(int(r[4:7]) // 10) for r in per.index]).sum(); ii = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
    share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
    print(f"{part:14s} n {int(m.sum()):3d} | base/full/fade per seed: " + " ; ".join(f"{b:.4f}/{f:.4f}/{d:.4f}" for b, f, d in res) +
          f" | fade vs full {np.mean([d / f - 1 for b, f, d in res])*100:+.1f}% | improved {int((per<0).sum())}/{len(per)} | block P {pw:.4f} | top share {share:.2f}")
    if part == "all":
        ok = all(d < f and d < b for b, f, d in res) and pw < .025
        print("   VERDICT CB2:", "PASS" if ok else "FAIL")
