# -*- coding: utf-8 -*-
"""HA2 (2026-10-12 집 클로드; 사용자: 'EC 에도 딱딱한 앵커 적용해서 검증').  사전 고정 — 실행 전 커밋.  세 검증 세계에서 같은 기준:
  (A) EV: EA2(SG2 + v7m p≥.8 정답 같은 ID 앵커) vs SG2.
  (B) SP2 2차형 세계(같은 ID ±1 만 숨김): HARD(2단계 사후 1등이 정답·P≥.8, EA2 공식) vs DIAG10 sg  — sp2_posteriors_v1.csv 의 hard 사용.
  (C) DIAG10: sg + EA2 식 앵커(v7m p≥.8, 정답 같은 ID, DIAG10 폴드 숨김 ±1 아님) vs sg.
판정(세계마다): 바뀐 기록 RMSE 3시드 모두 개선 AND 부트스트랩 P(worse) < .025(EV 덩어리, B·C 기록/블록) AND 최대 기록 몫 ≤ .5. 2차 따로.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research"); here = os.path.dirname(os.path.abspath(__file__))
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261248)
def judge(name, base, new, Y, grp):
    for part, cond in (("all", lambda s: True), ("pass-2", lambda s: int(s[4:7]) >= 179)):
        recs = sorted(r for r in set.union(*[set(r for r in new[s].index if not np.allclose(new[s].loc[r].values, base[s].loc[r].values, equal_nan=True)) for s in new]) if cond(r))
        if not recs: print(name, part, "none"); continue
        cells = [rm((new[s].loc[recs] - Y.loc[recs]).values) / rm((base[s].loc[recs] - Y.loc[recs]).values) - 1 for s in new]
        per = pd.Series(np.mean([(((new[s].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((base[s].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for s in new], 0), index=recs)
        cl = per.groupby(grp(recs)).sum(); ii = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
        share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
        v = "PASS" if all(x < 0 for x in cells) and pw < .025 and share <= .5 else "FAIL"
        print(f"{name} {part:6s}: changed {len(recs)} | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | P {pw:.4f} | top share {share:.2f} -> {v}")
        if part == "all": print("   largest:", ", ".join("%s %+.3f" % (a, b) for a, b in list(per.sort_values().items())[:3] + list(per.sort_values().items())[-3:]))
# (A) EV
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); ch = Fv[Fv.role == "val"].set_index("rec").chunk
YE = R.groupby(["rec", "hour"]).y.first().unstack(); bA, nA = {}, {}
for sd, G in R.groupby("seed"):
    bA[sd] = G.pivot_table(index="rec", columns="hour", values="SG2"); nA[sd] = G.pivot_table(index="rec", columns="hour", values="EA2")
judge("(A) EV EA2 vs SG2", bA, nA, YE, lambda recs: ch.reindex(recs).values)
# (B) SP2 world
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
D2 = pd.read_csv(os.path.join(here, "sp2_posteriors_v1.csv"), index_col=0)
bB, nB = {}, {}
for sd, G in O.groupby("seed"):
    Q = G.pivot_table(index="rec", columns="hour", values="sg"); cm = Q.cumsum(1) / np.arange(1, 25); Nq = Q.values.copy()
    hard = D2.hard.reindex(Q.index); m = hard.notna().values
    Nq[m] = Q.values[m] + (hard.values[m][:, None] - cm.values[m]); bB[sd] = Q; nB[sd] = pd.DataFrame(Nq, index=Q.index, columns=Q.columns)
judge("(B) W2 HARD vs sg", bB, nB, YD, lambda recs: np.array([r[:3] + str(int(r[4:7]) // 10) for r in recs]))
# (C) DIAG10 with v7m anchors
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
Yl = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Yl["rec"] = Yl.row_id.str[:7]; Yl["h"] = Yl.row_id.str[8:].astype(int); EC23 = Yl[Yl.h == 23].set_index("rec").sub_ec
bC, nC = {}, {}
for sd, G in O.groupby("seed"):
    Q = G.pivot_table(index="rec", columns="hour", values="sg"); fo = G.groupby("rec").fold.first().reindex(Q.index)
    hidden = {k: {(r[:3], int(r[4:7]) + j) for r in fo[fo == k].index for j in (-1, 0, 1)} for k in fo.unique()}
    cm = Q.cumsum(1) / np.arange(1, 25); Nq = Q.values.copy()
    for n, r in enumerate(Q.index):
        if r not in L.index: continue
        pr = L.loc[r]
        if pr.xf or pr.p < .8 or not pr.prev_is_labelled or pr.prev not in EC23.index or (pr.prev[:3], int(pr.prev[4:7])) in hidden[fo[r]]: continue
        Nq[n] = Q.values[n] + (EC23[pr.prev] - cm.values[n])
    bC[sd] = Q; nC[sd] = pd.DataFrame(Nq, index=Q.index, columns=Q.columns)
judge("(C) DIAG10 v7m anchor vs sg", bC, nC, YD, lambda recs: np.array(recs))
