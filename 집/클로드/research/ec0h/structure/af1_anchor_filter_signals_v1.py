# -*- coding: utf-8 -*-
"""AF1 (2026-10-12 집 클로드; 사용자: '잘못 붙은 앵커 걸러내는 방법').  서술 — 신호 탐색.
앵커 기록(EV: EA2 앵커 54, DIAG10: v7m p≥.8 앵커 34) 마다 결과 = 앵커의 ΔSSE(3시드 평균; > 0 이면 손해), 진실 연결과 일치 여부(확실 연결 있는 경우).
0시에 알 수 있는 신호(규정 안): p(v7m), gap = |A − q0|(앵커 vs 모델 0시), dT_air(자정 실내 차, v7m 표), date_ok(DM11 달력 전날 = 대상 −1),
  dST = |전날 23시 배지온 정답 − 온도 EV v22 대상 0시 예측|(같은 동이면 이어짐), fp0(운영 지문 0시), numgap(번호 차), sameID.
보고: 신호별 손해 판별 AUC(두 세계), 진실 일치와의 관계.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
import feat6 as F6
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
CAL = pd.read_csv(os.path.join(H_, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
TL = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TL[TL.cert]; truth = {rid(int(b)): rid(int(a)) for a, b in zip(TLc.a1, TLc.b)}
T = pd.read_pickle(os.path.join(H_, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); T["rec"] = T.row_id.str[:7]
TP0 = T[T.hour == 0].groupby("rec")[["v22_a", "v22_b"]].mean().mean(axis=1)
def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, int); ok = np.isfinite(x); x, y = x[ok], y[ok]
    a, b = x[y == 1], x[y == 0]
    return float((a[:, None] > b[None, :]).mean() + .5 * (a[:, None] == b[None, :]).mean()) if len(a) and len(b) else np.nan
def feats(r, A, q0):
    pr = L.loc[r]; p = pr.prev; t, j = idx[r], idx[p]
    return dict(p=float(pr.p), gap=abs(A - q0), dT=abs(float(pr.dT_air)) if np.isfinite(pr.dT_air) else np.nan,
                date_ok=int(DAY.get(p, np.nan) == DAY.get(r, np.nan) - 1), dST=abs(float(ST[j, 23]) - TP0.get(r, np.nan)),
                fp0=float(np.nan_to_num(F6.fp_dist(t, j, 0), nan=6)), numgap=int(NUM[t] - NUM[j]) if FARM[t] == FARM[j] else 99,
                sameID=int(not pr.xf), truth_ok=(int(truth[r] == p) if r in truth else -1), prev=p)
# ---- EV ----
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
Y = R.groupby(["rec", "hour"]).y.first().unstack(); anc = R.groupby("rec").anchored.first(); rows = []
dsse = []
for sd, G in R.groupby("seed"):
    S = G.pivot_table(index="rec", columns="hour", values="SG2"); E = G.pivot_table(index="rec", columns="hour", values="EA2")
    dsse.append(((E - Y.loc[E.index]) ** 2).sum(1) - ((S - Y.loc[S.index]) ** 2).sum(1)); q0 = S[0]; A0 = E[0]
ds = pd.concat(dsse, axis=1).mean(1)
for r in anc.index[anc.values.astype(bool)]:
    if r not in L.index: continue
    f = feats(r, A0[r], q0[r]); f.update(world="EV", rec=r, dsse=ds[r]); rows.append(f)
# ---- DIAG10 ----
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
Yl = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Yl["rec"] = Yl.row_id.str[:7]; Yl["h"] = Yl.row_id.str[8:].astype(int); EC23 = Yl[Yl.h == 23].set_index("rec").sub_ec
dl = []; q0d = None
for sd, G in O.groupby("seed"):
    Q = G.pivot_table(index="rec", columns="hour", values="sg"); fo = G.groupby("rec").fold.first().reindex(Q.index)
    hidden = {k: {(r[:3], int(r[4:7]) + j) for r in fo[fo == k].index for j in (-1, 0, 1)} for k in fo.unique()}
    cm = Q.cumsum(1) / np.arange(1, 25); d = {}
    for r in Q.index:
        if r not in L.index: continue
        pr = L.loc[r]
        if pr.xf or pr.p < .8 or not pr.prev_is_labelled or pr.prev not in EC23.index or (pr.prev[:3], int(pr.prev[4:7])) in hidden[fo[r]]: continue
        new = Q.loc[r].values + (EC23[pr.prev] - cm.loc[r].values)
        d[r] = float(((new - YD.loc[r].values) ** 2).sum() - ((Q.loc[r].values - YD.loc[r].values) ** 2).sum())
    dl.append(pd.Series(d)); q0d = Q[0]
dd = pd.concat(dl, axis=1).mean(1)
for r in dd.index:
    f = feats(r, float(EC23[L.at[r, "prev"]]), q0d[r]); f.update(world="DIAG10", rec=r, dsse=dd[r]); rows.append(f)
D = pd.DataFrame(rows); D["harm"] = (D.dsse > 0).astype(int)
pd.set_option("display.width", 250)
for w, G in D.groupby("world"):
    print(f"\n=== {w}: anchors {len(G)} | harmful {int(G.harm.sum())} (sum harm {G.dsse[G.dsse>0].sum():+.2f}, sum gain {G.dsse[G.dsse<0].sum():+.2f}) | truth known {int((G.truth_ok>=0).sum())}, wrong-house {int((G.truth_ok==0).sum())}")
    print("  AUC for 'harmful' (higher signal -> more harm):", " | ".join(f"{c} {auc(G[c], G.harm):.2f}" for c in ("gap", "dT", "dST", "fp0", "numgap")), "| (lower -> harm) p", f"{1-auc(G.p, G.harm):.2f}", "date_ok", f"{1-auc(G.date_ok, G.harm):.2f}", "sameID", f"{1-auc(G.sameID, G.harm):.2f}")
    K = G[G.truth_ok >= 0]
    if len(K) and K.truth_ok.nunique() > 1:
        print("  AUC for 'wrong house' (truth known):", " | ".join(f"{c} {auc(K[c], 1-K.truth_ok):.2f}" for c in ("gap", "dT", "dST", "fp0", "numgap")))
    print("  harmful anchors:"); print(G[G.harm == 1].sort_values("dsse", ascending=False)[["rec", "prev", "dsse", "p", "gap", "dT", "dST", "fp0", "date_ok", "numgap", "truth_ok"]].round(3).to_string(index=False))
D.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "af1_v1.csv"), index=False)
