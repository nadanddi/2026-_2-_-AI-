# -*- coding: utf-8 -*-
"""HX9 verify_rootzone v1 (2026-10-13 집 클로드, 적대적 검증 하위 에이전트)
대상: rootzone 가족 보고의 'weak' 판정 4개(in_cold10, heat_hours, heat_mean, heat_night)와 Y3(연속일 Δ) 결과, 다변량 상한.
읽기만: hx9/rootzone/vars_v1.csv, hx9/data/records.csv. 산출: 이 폴더의 로그만.
사전 판정 규칙(실행 전 고정): 변수는 아래 모두 통과해야 'weak 이상' 유지.
  (a) 사슬 하나씩 빼기 8번 모두 부호 동일하고 |r|>=.15, (b) 사슬 안 평균 제거(within-chain) r 부호 동일 & |r|>=.15,
  (c) 사슬 단위 순열 p<.2(사슬 라벨 섞기가 아니라 사슬 평균 r n=8은 서술만), (d) pass1/pass2 부호 동일,
  (e) BH(39개) q<.2. 하나라도 실패하면 artifact/inconclusive.
Y3: 연결 안 된 hi10 쌍(같은 농장/다른 농장) placebo 에서 Δ상관이 비슷하면 '같은 동 시간 효과'가 아니라 동 간 차이(오연결/이질성).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import env  # noqa
import numpy as np, pandas as pd
from scipy import stats
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
HX9 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = pd.read_csv(os.path.join(HX9, "rootzone", "vars_v1.csv"), index_col="rec")
R = pd.read_csv(os.path.join(HX9, "data", "records.csv"), index_col="rec")
RNG = np.random.default_rng(7)
hi = V[(V.hi10 == 1) & V.Y2.notna()].copy()
hi["ep"] = hi.ep_id.astype(int)
hi["otd"] = R.loc[hi.index, "out_temp_d"].values; hi["otn"] = R.loc[hi.index, "out_temp_n"].values
hi["dm11"] = R.loc[hi.index, "dm11_day"].values
print(f"hi10 with OOF n={len(hi)}, chains {hi.ep.nunique()}: sizes {hi.ep.value_counts().sort_index().to_dict()}")
WEAK = ["in_cold10", "heat_hours", "heat_mean", "heat_night"]
def r(x, y): return np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan
def partial(x, y, Z):
    Z = np.c_[np.ones(len(x)), Z]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]; ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return r(rx, ry)
for v in WEAK:
    x = hi[v].values; y = hi.Y2.values
    print(f"\n== {v}: r {r(x,y):+.3f}, rho {stats.spearmanr(x,y)[0]:+.3f}; nonzero {int((x>0).sum())}/{len(x)}; "
          f"nonzero by chain {hi.assign(nz=x>0).groupby('ep').nz.sum().to_dict()}")
    drops = {e: r(x[hi.ep != e], y[hi.ep != e]) for e in sorted(hi.ep.unique())}
    print("   drop-one-chain r:", {k: round(v_, 3) for k, v_ in drops.items()})
    xc = x - hi.groupby("ep")[v].transform("mean").values; yc = y - hi.groupby("ep").Y2.transform("mean").values
    print(f"   within-chain r {r(xc, yc):+.3f}; chain-mean r (n {hi.ep.nunique()}) {r(hi.groupby('ep')[v].mean(), hi.groupby('ep').Y2.mean()):+.3f}")
    print(f"   partial | out_temp day&night {partial(x, y, hi[['otd','otn']].values):+.3f}; | farm {partial(x, y, (hi.farm=='F47').values.astype(float)):+.3f}; "
          f"| sg_mean {partial(x, y, hi.sg_mean.values):+.3f}")
    # restricted chain-level permutation: permute x blocks across chains is ill-defined; use record permutation within chain (tests within-chain association)
    obs = abs(r(xc, yc)); cnt = 0; P = 4000
    for _ in range(P):
        xp = x.copy()
        for e in hi.ep.unique():
            m = (hi.ep == e).values; xp[m] = RNG.permutation(xp[m])
        xpc = xp - pd.Series(xp).groupby(hi.ep.values).transform("mean").values
        cnt += abs(r(xpc, yc)) >= obs - 1e-12
    print(f"   within-chain permutation p {(1+cnt)/(P+1):.3f}")
    p1 = hi.pass2 == 0
    print(f"   pass1 r {r(x[p1], y[p1]):+.3f} (n {p1.sum()}), pass2 r {r(x[~p1], y[~p1]):+.3f} (n {(~p1).sum()})")
    # on the whole set: does the variable just mark 'hi10 vs not' ? residual sign among normal
    al = V[V.Y2.notna()]
    print(f"   all records r(var, Y2) {r(al[v], al.Y2):+.3f}; normal-only {r(al[al.hi10==0][v], al[al.hi10==0].Y2):+.3f}")
# heating zero mass: which records carry it
print("\nheat_mean values by chain (hi10):"); print(hi.groupby("ep").agg(n=("Y2", "size"), heat=("heat_mean", "mean"), heat_max=("heat_mean", "max"),
                                                                    Y2=("Y2", "mean"), farm=("farm", "first"), dm11=("dm11", "median")).round(3))

# ---------------- Y3 ----------------
print("\n==== Y3 (hi10 -> hi10 clean links): placebo with unlinked hi10 pairs ====")
L = V[(V.hi10 == 1) & V.prev_rec.notna()].copy(); L = L[V.loc[L.prev_rec, "hi10"].values == 1]
L["ep"] = R.loc[L.index, "ep_id"].astype(int).values
L["sf"] = R.loc[L.index, "prev_same_farm"].values
def dvals(b, a, v): return V.loc[b, v].values - V.loc[a, v].values
for v in ("d_st_mean", "d_pst_min", "d_in_min"):
    base = v[2:]
    dx = dvals(L.index, L.prev_rec, base); dy = np.log(V.loc[L.index, "ec_mean"].values) - np.log(V.loc[L.prev_rec, "ec_mean"].values)
    drops = {e: round(r(dx[L.ep != e], dy[L.ep != e]), 3) for e in sorted(L.ep.unique())}
    print(f"{v}: linked r {r(dx, dy):+.3f} n {len(dx)}; same-farm {r(dx[L.sf==1], dy[L.sf==1]):+.3f} (n {(L.sf==1).sum()}), "
          f"cross {r(dx[L.sf==0], dy[L.sf==0]):+.3f} (n {(L.sf==0).sum()}); drop-one-chain {drops}")
    # placebo: random pairs of hi10 records within +-2 DM11 days but NOT linked (different houses same weather window)
    H10 = V[V.hi10 == 1].copy(); H10["dm11"] = R.loc[H10.index, "dm11_day"].values
    pairs = [(b, a) for b in H10.index for a in H10.index if b != a and 0 < H10.dm11[b] - H10.dm11[a] <= 2]
    pb = np.array([p[0] for p in pairs]); pa = np.array([p[1] for p in pairs])
    linked = set(zip(L.index, L.prev_rec))
    keep = np.array([(b, a) not in linked for b, a in pairs])
    pb, pa = pb[keep], pa[keep]
    px = dvals(pb, pa, base); py = np.log(V.loc[pb, "ec_mean"].values) - np.log(V.loc[pa, "ec_mean"].values)
    sfp = (V.loc[pb, "farm"].values == V.loc[pa, "farm"].values)
    sl = stats.linregress(px, py)
    print(f"   placebo unlinked hi10 pairs (DM11 gap 1-2) n {len(px)}: r {r(px,py):+.3f}, slope {sl.slope:+.4f}; same-farm {r(px[sfp],py[sfp]):+.3f}, cross {r(px[~sfp],py[~sfp]):+.3f}")
    # same-day different houses (gap 0): pure between-house relation
    pairs0 = [(b, a) for b in H10.index for a in H10.index if b < a and H10.dm11[b] == H10.dm11[a]]
    b0 = [p[0] for p in pairs0]; a0 = [p[1] for p in pairs0]
    qx = dvals(b0, a0, base); qy = np.log(V.loc[b0, "ec_mean"].values) - np.log(V.loc[a0, "ec_mean"].values)
    print(f"   same-day between-house hi10 pairs n {len(qx)}: r {r(qx,qy):+.3f}, slope {stats.linregress(qx,qy).slope:+.4f}")
print("\nY3 cross-farm links by chain:", L[L.sf == 0].groupby("ep").size().to_dict(), "| same-farm:", L[L.sf == 1].groupby("ep").size().to_dict())
