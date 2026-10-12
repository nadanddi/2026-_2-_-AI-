# -*- coding: utf-8 -*-
"""AD1 (2026-10-12 집 클로드; 사용자: '그날 바뀌면 앵커 영향 줄이는 방식').  사전 고정 — 실행 전 커밋.
예측_h = 기본_h + w(blk(h), bin(fp_h)) · (A − cm_h),  cm_h = 기본 0..h 누적평균, A = 앵커(전날 23시 정답).
fp_h = 운영 지문 거리(linker v6 fp_dist: 대상 0..h 시 vs 전날 기록 같은 시각) — 대상 0..h 입력 + train 만(규정 안).
칸: 시각 블록 0-5/6-11/12-17/18-23 × fp 구간 (≤.4, .4~.7, >.7) = 12.  w = EV 앵커 54(기본 SG2) 행으로 칸별 최소제곱, [0, 1.2].
판정(DIAG10 앵커 34, 기본 sg, v7m 앵커): AD1 이 '끝까지 앵커(w=1)'보다 3시드 모두 좋음 AND 기록 부트스트랩 P(worse) < .05 AND 기본보다 3시드 모두 좋음.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
import feat6 as F6
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
Yl = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Yl["rec"] = Yl.row_id.str[:7]; Yl["h"] = Yl.row_id.str[8:].astype(int); EC23 = Yl[Yl.h == 23].set_index("rec").sub_ec
blk = lambda h: 0 if h < 6 else 1 if h < 12 else 2 if h < 18 else 3
fbin = lambda f: 0 if f <= .4 else 1 if f <= .7 else 2
FP = {}
def fp(r, h):
    k = (r, h)
    if k not in FP: v = F6.fp_dist(idx[r], idx[L.at[r, "prev"]], h); FP[k] = float(v) if np.isfinite(v) else 1.0
    return FP[k]
# ---- fit on EV ----
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
anc = R.groupby("rec").anchored.first(); EVA = [r for r in anc.index[anc.values.astype(bool)] if r in L.index]
Y = R.groupby(["rec", "hour"]).y.first().unstack()
xs = {c: [] for c in range(12)}; ys = {c: [] for c in range(12)}
for sd, G in R.groupby("seed"):
    S = G.pivot_table(index="rec", columns="hour", values="SG2"); E = G.pivot_table(index="rec", columns="hour", values="EA2"); cm = S.cumsum(1) / np.arange(1, 25)
    for r in EVA:
        A = E.at[r, 0]
        for h in range(24):
            c = blk(h) * 3 + fbin(fp(r, h)); xs[c].append(A - cm.at[r, h]); ys[c].append(Y.at[r, h] - S.at[r, h])
W = {}
for c in range(12):
    x, y = np.array(xs[c]), np.array(ys[c]); W[c] = float(np.clip((x * y).sum() / (x * x).sum(), 0, 1.2)) if len(x) > 20 and (x * x).sum() > 0 else 1.0
print("weights (rows = hour block 0-5/6-11/12-17/18-23, cols = fp <=.4/.4-.7/>.7):"); print(np.round(np.array([W[c] for c in range(12)]).reshape(4, 3), 2))
print("rows per cell:", [len(xs[c]) for c in range(12)])
# ---- judge on DIAG10 ----
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261250); res = []; per = []
for sd, G in O.groupby("seed"):
    Q = G.pivot_table(index="rec", columns="hour", values="sg"); fo = G.groupby("rec").fold.first().reindex(Q.index)
    hidden = {k: {(r[:3], int(r[4:7]) + j) for r in fo[fo == k].index for j in (-1, 0, 1)} for k in fo.unique()}
    cm = Q.cumsum(1) / np.arange(1, 25); recs = []
    for r in Q.index:
        if r not in L.index: continue
        pr = L.loc[r]
        if pr.xf or pr.p < .8 or not pr.prev_is_labelled or pr.prev not in EC23.index or (pr.prev[:3], int(pr.prev[4:7])) in hidden[fo[r]]: continue
        recs.append(r)
    Yr = YD.loc[recs].values; Qv = Q.loc[recs].values; C = cm.loc[recs].values; A = np.array([EC23[L.at[r, "prev"]] for r in recs])
    full = Qv + (A[:, None] - C)
    wmat = np.array([[W[blk(h) * 3 + fbin(fp(r, h))] for h in range(24)] for r in recs]); fade = Qv + wmat * (A[:, None] - C)
    res.append(dict(seed=sd, n=len(recs), base=rm(Qv - Yr), full=rm(full - Yr), fade=rm(fade - Yr)))
    per.append(pd.Series(((fade - Yr) ** 2).sum(1) - ((full - Yr) ** 2).sum(1), index=recs))
Rz = pd.DataFrame(res); print(Rz.round(4).to_string(index=False))
pr = pd.concat(per, axis=1).mean(1); ii = rng.integers(0, len(pr), (20000, len(pr))); pw = float((pr.values[ii].sum(1) >= 0).mean())
ok = (Rz.fade < Rz.full).all() and pw < .05 and (Rz.fade < Rz.base).all()
print(f"fade vs full: dSSE sum {pr.sum():+.3f}, improved {int((pr<0).sum())}/{int((pr!=0).sum())}, P {pw:.4f} -> VERDICT AD1: {'PASS' if ok else 'FAIL'}")
print("largest:", ", ".join("%s %+.3f" % (a, b) for a, b in list(pr.sort_values().items())[:4] + list(pr.sort_values().items())[-3:]))
