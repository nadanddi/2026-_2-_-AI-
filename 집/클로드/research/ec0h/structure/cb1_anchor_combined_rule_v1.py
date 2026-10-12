# -*- coding: utf-8 -*-
"""CB1 (2026-10-12 집 클로드; 사용자: '두 가지 합친 규칙으로 검증').  사전 고정 — 실행 전 커밋.
규칙: (1) |자정 실내 온도 차 dT_air|(v7m 표) > t 이면 앵커 제거(다른 동 걸러내기), (2) 남은 앵커는 시각 블록 b(0-5/6-11/12-17/18-23)별 반영 비율 w_b:
  q_h = 기본_h + w_b · (A − 기본 0..h 누적평균).
선택(EV 앵커 54, 기본 SG2): t ∈ {1.0, 1.3, 1.6, 2.0, 무한} 마다 w_b 를 최소제곱([0, 1.2])으로 맞추고 EV RMSE 최소 (t, w) 선택.
판정(DIAG10 앵커 34, 기본 sg, v7m 앵커): 합친 규칙이 하루 전체 앵커(w=1, 제거 없음)보다 3시드 모두 좋음 AND 기록 부트스트랩 P(worse) < .05 AND 기본보다 3시드 모두 좋음.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
dTa = lambda r: abs(float(L.at[r, "dT_air"])) if (r in L.index and np.isfinite(L.at[r, "dT_air"])) else 0.0
blk = np.array([0] * 6 + [1] * 6 + [2] * 6 + [3] * 6)
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261252)
def pred(Q, A, keep, w):
    cm = np.cumsum(Q, 1) / np.arange(1, 25); out = Q.copy(); k = keep
    out[k] = Q[k] + w[blk][None, :] * (A[k][:, None] - cm[k]); return out
# ---- EV selection ----
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
anc = R.groupby("rec").anchored.first(); recsE = [r for r in anc.index[anc.values.astype(bool)]]; Y = R.groupby(["rec", "hour"]).y.first().unstack()
EVs = []
for sd, G in R.groupby("seed"):
    S = G.pivot_table(index="rec", columns="hour", values="SG2").loc[recsE].values; A = G.pivot_table(index="rec", columns="hour", values="EA2").loc[recsE, 0].values
    EVs.append((S, A, Y.loc[recsE].values))
dE = np.array([dTa(r) for r in recsE]); best = None
for t in (1.0, 1.3, 1.6, 2.0, np.inf):
    keep = dE <= t; w = np.ones(4)
    for b in range(4):
        hs = np.where(blk == b)[0]; xs, ys = [], []
        for S, A, Yv in EVs:
            cm = np.cumsum(S, 1) / np.arange(1, 25); xs.append((A[keep][:, None] - cm[keep][:, hs]).ravel()); ys.append((Yv[keep][:, hs] - S[keep][:, hs]).ravel())
        x, y = np.concatenate(xs), np.concatenate(ys); ok = np.isfinite(y); w[b] = float(np.clip((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum(), 0, 1.2))
    score = np.mean([rm(pred(S, A, keep, w) - Yv) for S, A, Yv in EVs])
    print(f"EV t={t}: dropped {int((~keep).sum())} | w {np.round(w, 2).tolist()} | RMSE {score:.4f}")
    if best is None or score < best[0]: best = (score, t, w)
_, tB, wB = best; print("chosen t", tB, "w", np.round(wB, 2).tolist(), "| EV full-anchor RMSE %.4f" % np.mean([rm(pred(S, A, np.ones(len(A), bool), np.ones(4)) - Yv) for S, A, Yv in EVs]))
# ---- DIAG10 judgement ----
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
Yl = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Yl["rec"] = Yl.row_id.str[:7]; Yl["h"] = Yl.row_id.str[8:].astype(int); EC23 = Yl[Yl.h == 23].set_index("rec").sub_ec
res, per = [], []
for sd, G in O.groupby("seed"):
    Q = G.pivot_table(index="rec", columns="hour", values="sg"); fo = G.groupby("rec").fold.first().reindex(Q.index)
    hidden = {k: {(r[:3], int(r[4:7]) + j) for r in fo[fo == k].index for j in (-1, 0, 1)} for k in fo.unique()}
    recs = [r for r in Q.index if r in L.index and not L.at[r, "xf"] and L.at[r, "p"] >= .8 and L.at[r, "prev_is_labelled"] and L.at[r, "prev"] in EC23.index
            and (L.at[r, "prev"][:3], int(L.at[r, "prev"][4:7])) not in hidden[fo[r]]]
    Qv = Q.loc[recs].values; A = np.array([EC23[L.at[r, "prev"]] for r in recs]); Yv = YD.loc[recs].values; keep = np.array([dTa(r) <= tB for r in recs])
    full = pred(Qv, A, np.ones(len(recs), bool), np.ones(4)); comb = pred(Qv, A, keep, wB)
    res.append(dict(seed=sd, n=len(recs), dropped=int((~keep).sum()), base=rm(Qv - Yv), full=rm(full - Yv), combined=rm(comb - Yv)))
    per.append(pd.Series(((comb - Yv) ** 2).sum(1) - ((full - Yv) ** 2).sum(1), index=recs))
Rz = pd.DataFrame(res); print(Rz.round(4).to_string(index=False))
pr = pd.concat(per, axis=1).mean(1); ii = rng.integers(0, len(pr), (20000, len(pr))); pw = float((pr.values[ii].sum(1) >= 0).mean())
ok = (Rz.combined < Rz.full).all() and pw < .05 and (Rz.combined < Rz.base).all()
print(f"combined vs full: dSSE {pr.sum():+.3f}, improved {int((pr<0).sum())}/{int((pr!=0).sum())}, P {pw:.4f} -> VERDICT CB1: {'PASS' if ok else 'FAIL'}")
print("largest:", ", ".join("%s %+.3f" % (a, b) for a, b in list(pr.sort_values().items())[:4] + list(pr.sort_values().items())[-3:]))
