# -*- coding: utf-8 -*-
"""OR3 (2026-10-12 집 클로드, OR2 다음 단): (EC) 0시를 정확히 알고 난 뒤 남는 오차는 어디서? — 시간대, 하루 변화(23−0시) 큰 날,
고EC 날; 앵커 없는 기록이 같은 동 정답일에서 며칠 떨어졌는지(정보 도달 가능성).  (온도) 하루 평균을 알면 .23 수준 → 온도 하루 수준
오차는 어디에 몰리나(농장, 추운 날, 시간대), 0시 오차와 낮 오차의 관계.  EV 기준 예측(EA2·v22), 서술."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
Q = R.groupby(["rec", "hour"]).EA2.mean().unstack(); Y = R.groupby(["rec", "hour"]).y.first().unstack().reindex(Q.index); ANC = R.groupby("rec").anchored.first().reindex(Q.index).fillna(False).astype(bool)
E = Y - Q; P2 = np.array([int(r[4:7]) >= 179 for r in Q.index])
e1 = E[~P2].values; c = np.array([np.nansum(e1[:, 0] * e1[:, h]) / np.nansum(e1[:, 0] ** 2) for h in range(24)])
R1 = E.values - c[None, :] * E[0].values[:, None]
dY = (Y[23] - Y[0]).values; dQ = (Q[23] - Q[0]).values
print("[EC] after exact 0h (c_h carry): remaining SSE share by hour block (pass-1 / pass-2)")
for part, s in (("p1", ~P2), ("p2", P2)):
    tot = np.nansum(R1[s] ** 2)
    blk = {f"{a}-{b}h": np.nansum(R1[s][:, a:b + 1] ** 2) / tot for a, b in ((0, 5), (6, 11), (12, 17), (18, 23))}
    big = np.abs(dY) >= .15
    print(f"  {part}: " + " ".join(f"{k} {v:.2f}" for k, v in blk.items()) + f" | days with |true 0->23 change|>=.15: {int((s&big).sum())}/{int(s.sum())} hold {np.nansum(R1[s & big]**2)/tot:.2f} of remaining SSE")
    print(f"       corr(remaining day-mean error, true change - predicted change) {np.corrcoef(np.nanmean(R1[s],1), (dY-dQ)[s])[0,1]:.2f} | RMSE of predicted 0->23 change {np.sqrt(np.nanmean((dY-dQ)[s]**2)):.3f} vs sd true change {np.nanstd(dY[s]):.3f}")
# which unanchored records carry the 0h value; how far is the nearest labelled same-house day (truth links, diagnostic)
print("\n[EC] unanchored records: 0h error by level and farm (pass-2)")
U = pd.DataFrame(dict(rec=Q.index, p2=P2, anc=ANC.values, e0=E[0].values, y0=Y[0].values, farm=[r[:3] for r in Q.index]))
for nm, G in U[U.p2].groupby(["anc"]):
    print(f"  anchored={nm[0]}: n {len(G)} | 0h RMSE {np.sqrt((G.e0**2).mean()):.3f} | share of p2 0h SSE {(G.e0**2).sum()/(U[U.p2].e0**2).sum():.2f}")
G = U[U.p2 & ~U.anc]
for lo, hi in ((0, .6), (.6, 1.0), (1.0, 9)):
    s = (G.y0 >= lo) & (G.y0 < hi); print(f"   unanchored true 0h {lo}-{hi}: n {int(s.sum())} | 0h RMSE {np.sqrt((G.e0[s]**2).mean()):.3f} | share of unanchored 0h SSE {(G.e0[s]**2).sum()/(G.e0**2).sum():.2f}")
print("   unanchored, largest 0h errors:", ", ".join(f"{r.rec} e0 {r.e0:+.2f} (y0 {r.y0:.2f})" for r in G.reindex(G.e0.abs().sort_values(ascending=False).index).head(10).itertuples()))
# ---------- temperature ----------
T = pd.read_pickle(os.path.join(H_, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); T["rec"] = T.row_id.str[:7]
vc = [cc for cc in T.columns if cc.startswith("v22_")]; T["q"] = T[vc].mean(axis=1)
QT = T.pivot(index="rec", columns="hour", values="q"); YT = T.pivot(index="rec", columns="hour", values="sub_temp"); IT = T.pivot(index="rec", columns="hour", values="in_temp")
ET = YT - QT; P2T = np.array([int(r[4:7]) >= 179 for r in QT.index]); dm = ET.mean(axis=1)
print("\n[TEMP] day-mean error (offset) structure")
for part, s in (("p1", ~P2T), ("p2", P2T)):
    d = dm[s]; tot = (ET[s] ** 2).values.sum()
    f47 = np.array([r[:3] == "F47" for r in QT.index])[s]; cold = (IT.mean(axis=1) < 12).values[s]
    print(f"  {part}: day-offset RMSE {np.sqrt((d**2).mean()):.3f} | day-level SSE share {(d**2).sum()*24/tot:.2f} | F47 share of offset SSE {(d[f47]**2).sum()/(d**2).sum():.2f} (records {f47.mean():.2f}) | cold days (mean in_temp<12) share {(d[cold]**2).sum()/(d**2).sum():.2f} (records {cold.mean():.2f}) | offset sign: mean {d.mean():+.2f}")
    print(f"       hourly RMSE 0,3,6,9,12,15,18,21: " + " ".join(f"{np.sqrt((ET[s][h]**2).mean()):.2f}" for h in (0, 3, 6, 9, 12, 15, 18, 21)))
    print(f"       corr(0h error, day offset) {np.corrcoef(ET[s][0], d)[0,1]:.2f} | corr(6h error, offset) {np.corrcoef(ET[s][6], d)[0,1]:.2f} | corr(14h error, offset) {np.corrcoef(ET[s][14], d)[0,1]:.2f}")
    top = d.abs().sort_values(ascending=False); k = max(1, int(.1 * len(top)))
    print(f"       top 10% days hold {(top.iloc[:k]**2).sum()/(d**2).sum():.2f} of offset SSE; examples: " + ", ".join(f"{r} {d[r]:+.1f}" for r in top.index[:8]))
