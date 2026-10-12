# -*- coding: utf-8 -*-
"""AK1 (2026-10-12 집 클로드; 사용자: '23시→0시 앵커만 하면 안 돼?').  사전 고정 — 실행 전 커밋.
앵커(EA2 공식: q_h = 기본_h + (A − 기본 0..h 누적평균))를 0..K 시에만 적용하고 그 뒤는 기본 예측 그대로.
K ∈ {0, 5, 11, 14, 23(=현재 하루 전체)}.  K 는 EV 앵커 54(기본 SG2)에서 RMSE 최소로 선택 → DIAG10 앵커 34(기본 sg, v7m 앵커)에서 판정.
판정: 선택 K 가 K=23(하루 전체)보다 3시드 모두 좋음 AND 기록 부트스트랩 P(worse) < .05 AND 기본보다 3시드 모두 좋음.  모든 K 의 두 세계 결과도 보고.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
KS = [0, 5, 11, 14, 23]
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261251)
def apply(Q, A, K):
    cm = np.cumsum(Q, 1) / np.arange(1, 25); out = Q.copy(); out[:, :K + 1] = Q[:, :K + 1] + (A[:, None] - cm[:, :K + 1]); return out
# EV
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
anc = R.groupby("rec").anchored.first(); recsE = list(anc.index[anc.values.astype(bool)]); Y = R.groupby(["rec", "hour"]).y.first().unstack()
ev = {K: [] for K in KS}; evb = []
for sd, G in R.groupby("seed"):
    S = G.pivot_table(index="rec", columns="hour", values="SG2").loc[recsE].values; A = G.pivot_table(index="rec", columns="hour", values="EA2").loc[recsE, 0].values; Yv = Y.loc[recsE].values
    evb.append(rm(S - Yv))
    for K in KS: ev[K].append(rm(apply(S, A, K) - Yv))
print("EV anchored 54 | base %.4f |" % np.mean(evb), " | ".join(f"K={K}: {np.mean(v):.4f}" for K, v in ev.items()))
Kb = min(KS, key=lambda K: np.mean(ev[K])); print("chosen K:", Kb)
# DIAG10
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
Yl = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Yl["rec"] = Yl.row_id.str[:7]; Yl["h"] = Yl.row_id.str[8:].astype(int); EC23 = Yl[Yl.h == 23].set_index("rec").sub_ec
res = []; per = []
for sd, G in O.groupby("seed"):
    Q = G.pivot_table(index="rec", columns="hour", values="sg"); fo = G.groupby("rec").fold.first().reindex(Q.index)
    hidden = {k: {(r[:3], int(r[4:7]) + j) for r in fo[fo == k].index for j in (-1, 0, 1)} for k in fo.unique()}
    recs = [r for r in Q.index if r in L.index and not L.at[r, "xf"] and L.at[r, "p"] >= .8 and L.at[r, "prev_is_labelled"] and L.at[r, "prev"] in EC23.index
            and (L.at[r, "prev"][:3], int(L.at[r, "prev"][4:7])) not in hidden[fo[r]]]
    Qv = Q.loc[recs].values; A = np.array([EC23[L.at[r, "prev"]] for r in recs]); Yv = YD.loc[recs].values
    row = dict(seed=sd, n=len(recs), base=rm(Qv - Yv)); row.update({f"K{K}": rm(apply(Qv, A, K) - Yv) for K in KS}); res.append(row)
    per.append(pd.Series(((apply(Qv, A, Kb) - Yv) ** 2).sum(1) - ((apply(Qv, A, 23) - Yv) ** 2).sum(1), index=recs))
Rz = pd.DataFrame(res); print(Rz.round(4).to_string(index=False))
pr = pd.concat(per, axis=1).mean(1); ii = rng.integers(0, len(pr), (20000, len(pr))); pw = float((pr.values[ii].sum(1) >= 0).mean())
ok = (Rz[f"K{Kb}"] < Rz["K23"]).all() and pw < .05 and (Rz[f"K{Kb}"] < Rz.base).all()
print(f"K={Kb} vs K=23: dSSE {pr.sum():+.3f}, improved {int((pr<0).sum())}/{int((pr!=0).sum())}, P {pw:.4f} -> VERDICT AK1: {'PASS' if ok else 'FAIL'}" if Kb != 23 else "chosen K = 23 (no change)")
